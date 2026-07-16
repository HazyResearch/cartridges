from __future__ import annotations
import os
from typing import Dict, List, Literal, Optional, Any
from abc import abstractmethod
from collections import deque
import json
import pickle
from pathlib import Path
import random
from dataclasses import dataclass

from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizerFast
from pydrantic import ObjectConfig, BaseConfig
import numpy as np

from cartridges.structs import Conversation, MessageDict, read_conversations
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING
from cartridges.utils import get_logger
from cartridges.utils.hf import read_conversations_from_hf
from cartridges.utils.wandb import read_conversations_from_wandb

# SE(04/02): required to silence tokenizer warnings when using dataloders with
# multiple worker processes
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = get_logger(__name__)

BOS_TOKEN_ID = 128000
EOS_TOKEN_ID = 128009
START_HEADER_ID = 128006
END_HEADER_ID = 128007
USER_TOKEN_ID = 882
ASSISTANT_TOKEN_ID = 78191



@dataclass
class TokenCounts:
    num_system_and_user_tokens: int = 0
    num_assistant_tokens: int = 0

    @property
    def num_tokens(self):
        return self.num_system_and_user_tokens + self.num_assistant_tokens

    def __add__(self, other: "TokenCounts"):
        return TokenCounts(
            num_system_and_user_tokens=self.num_system_and_user_tokens
            + other.num_system_and_user_tokens,
            num_assistant_tokens=self.num_assistant_tokens + other.num_assistant_tokens,
        )

def _base_convert_messages_to_element_retokenize(
    messages: List[Conversation.Message],
    tokenizer: PreTrainedTokenizerFast,
    message_start_tokens: dict[str, list[int]],
    message_end_tokens: dict[str, list[int]],
    message_extra_end_tokens: dict[str, list[int]],
) -> DatasetElement:
    input_ids, topk_token_ids, topk_logprobs, topk_token_idxs = [], [], [], []
    token_counts = TokenCounts()

    for i, message in enumerate(messages):
        token_ids = tokenizer.encode(message.content, add_special_tokens=False)
        token_ids += message_end_tokens[message.role] + message_extra_end_tokens[message.role]
        msg_input_ids = message_start_tokens[message.role] + token_ids

        if message.role == "assistant":
            topk_token_ids.append(token_ids)
            topk_logprobs.append(np.zeros(len(token_ids), dtype=np.float16))
            topk_token_idxs.append(np.arange(len(token_ids), dtype=np.int32) + len(input_ids) + len(message_start_tokens[message.role]))

        input_ids.extend(msg_input_ids)
    
        # FIXME: this is broken in the case that we truncate the input ids
        token_counts += TokenCounts(
            num_system_and_user_tokens=len(input_ids) if message.role == "user" else 0,
            num_assistant_tokens=len(input_ids) if message.role == "assistant" else 0,
        )

    return DatasetElement(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        topk_token_ids=torch.from_numpy(np.concatenate(topk_token_ids)),
        topk_logprobs=torch.from_numpy(np.concatenate(topk_logprobs)),
        topk_token_idxs=torch.from_numpy(np.concatenate(topk_token_idxs)),
        metadata=[],
        token_counts=token_counts,
    )



def _base_convert_messages_to_element(
    messages: List[Conversation.Message],
    message_start_tokens: dict[str, list[int]],
    message_end_tokens: dict[str, list[int]],
    message_extra_end_tokens: dict[str, list[int]],
    tokenizer: PreTrainedTokenizerFast,
) -> DatasetElement:
    input_ids, topk_token_ids, topk_logprobs, topk_token_idxs = [], [], [], []
    token_counts = TokenCounts()

    for i, message in enumerate(messages):
        if message.token_ids is None:
            # Some synthesizers may not return token ids for some messages, in which case
            # we actually need to do the retokenization. 
            msg_token_ids = tokenizer.encode(message.content, add_special_tokens=False)
        else:
            msg_token_ids = list(message.token_ids)
        msg_input_ids = message_start_tokens[message.role] + msg_token_ids
        
        end_tokens = message_end_tokens[message.role]
        # usually, messages will end with some tokenizer-specific "end" token(s) (e.g. <|endoftext|>)
        ends_with_eot = len(msg_input_ids) >= len(end_tokens) and msg_input_ids[-len(end_tokens):] == end_tokens
        if (not ends_with_eot and i < len(messages) - 1):
            # (1) if it does not, this means that the generation hit the max token limit. In 
            # this case, if we're not the last message in the conversation, we add the end 
            # tokens in, so that the next message starts with the correct tokens.
            # Otherwise, we can just leave it as is.
            msg_input_ids += end_tokens + message_extra_end_tokens[message.role]
        elif ends_with_eot and i < len(messages) - 1:
            # (2) if it does end with the eot tokens, then we need to add the extra end 
            # then we just need to add any extra end tokens that come after the 
            # eot tokens and before the next message starts.
            msg_input_ids += message_extra_end_tokens[message.role]
        elif ends_with_eot and i == len(messages) - 1:
            # (3) if we're the last message in the 
            # conversation, then we don't need to add anything.
            pass

        if message.top_logprobs is not None:
            topk_token_ids.append(message.top_logprobs.token_id)
            topk_logprobs.append(message.top_logprobs.logprobs)
            topk_token_idxs.append(message.top_logprobs.token_idx + len(input_ids) + len(message_start_tokens[message.role]))

        input_ids.extend(msg_input_ids)
    
        # FIXME: this is broken in the case that we truncate the input ids
        token_counts += TokenCounts(
            num_system_and_user_tokens=len(input_ids) if message.role == "user" else 0,
            num_assistant_tokens=len(input_ids) if message.role == "assistant" else 0,
        )

    return DatasetElement(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        topk_token_ids=torch.from_numpy(np.concatenate(topk_token_ids)),
        topk_logprobs=torch.from_numpy(np.concatenate(topk_logprobs)),
        topk_token_idxs=torch.from_numpy(np.concatenate(topk_token_idxs)),
        metadata=[],
        token_counts=token_counts,
    )


def _find_subsequence(
    sequence: list[int],
    subsequence: list[int],
    start: int,
) -> int:
    if not subsequence:
        raise ValueError("Chat-template marker tokenized to an empty sequence")
    last_start = len(sequence) - len(subsequence)
    for idx in range(start, last_start + 1):
        if sequence[idx:idx + len(subsequence)] == subsequence:
            return idx
    raise ValueError("Could not locate a message marker in the rendered chat template")


def _chat_template_ids(
    tokenizer: PreTrainedTokenizerFast,
    messages: list[dict[str, str]],
) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        chat_template=MODEL_TO_CHAT_TEMPLATE.get(tokenizer.name_or_path),
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Expected one rendered chat-template sequence")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _max_prefix_overlap(left: list[int], right: list[int]) -> int:
    for size in range(min(len(left), len(right)), 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def tokenizer_messages_to_element(
    messages: List[Conversation.Message],
    retokenize: bool = False,
    tokenizer: PreTrainedTokenizerFast | None = None,
) -> DatasetElement:
    """Build an element from the tokenizer's chat template, without model token constants.

    The generated assistant IDs are substituted into a marker-rendered template so
    teacher IDs remain exact. A tokenizer-only round trip is checked first, making
    unsupported templates fail instead of silently shifting training targets.
    """
    if tokenizer is None:
        raise ValueError("A tokenizer is required for tokenizer-derived conversion")
    if not messages:
        raise ValueError("messages must not be empty")

    marker_texts = [
        f"<|cartridges_message_{idx}_7f39a2|>" for idx in range(len(messages))
    ]
    marker_messages = [
        {"role": message.role, "content": marker}
        for message, marker in zip(messages, marker_texts, strict=True)
    ]
    rendered_markers = _chat_template_ids(tokenizer, marker_messages)

    chunks: list[list[int]] = []
    cursor = 0
    for marker in marker_texts:
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        marker_start = _find_subsequence(rendered_markers, marker_ids, cursor)
        chunks.append(rendered_markers[cursor:marker_start])
        cursor = marker_start + len(marker_ids)
    chunks.append(rendered_markers[cursor:])

    retokenized_content = [
        list(tokenizer.encode(message.content, add_special_tokens=False))
        for message in messages
    ]

    def compose(
        replacements: list[list[int]],
        *,
        remove_suffix_overlap: bool,
    ) -> tuple[list[int], list[int]]:
        output = list(chunks[0])
        starts: list[int] = []
        for idx, replacement in enumerate(replacements):
            starts.append(len(output))
            output.extend(replacement)
            following = chunks[idx + 1]
            if remove_suffix_overlap:
                overlap = _max_prefix_overlap(replacement, following)
                following = following[overlap:]
            output.extend(following)
        return output, starts

    roundtrip_ids, _ = compose(
        retokenized_content,
        remove_suffix_overlap=False,
    )
    expected_ids = _chat_template_ids(
        tokenizer,
        [
            {"role": message.role, "content": message.content}
            for message in messages
        ],
    )
    if roundtrip_ids != expected_ids:
        raise ValueError(
            "Tokenizer chat-template round-trip failed; content token boundaries "
            "cannot be derived safely for this checkpoint"
        )

    replacements = []
    for message, fallback_ids in zip(messages, retokenized_content, strict=True):
        if retokenize or message.token_ids is None:
            replacements.append(fallback_ids)
        else:
            replacements.append([int(token_id) for token_id in message.token_ids])

    input_ids, content_starts = compose(
        replacements,
        remove_suffix_overlap=not retokenize,
    )
    topk_token_ids, topk_logprobs, topk_token_idxs = [], [], []
    for message, content_start in zip(messages, content_starts, strict=True):
        if message.top_logprobs is None:
            continue
        if message.token_ids is None:
            raise ValueError("Top-logprob messages must include exact completion token IDs")
        if len(message.token_ids) != message.top_logprobs.shape[0]:
            raise ValueError(
                "Top-logprob token rows do not align with the message token IDs"
            )
        topk_token_ids.append(np.asarray(message.top_logprobs.token_id))
        topk_logprobs.append(np.asarray(message.top_logprobs.logprobs))
        topk_token_idxs.append(
            np.asarray(message.top_logprobs.token_idx, dtype=np.int64) + content_start
        )

    if topk_token_ids:
        flat_token_ids = np.concatenate(topk_token_ids)
        flat_logprobs = np.concatenate(topk_logprobs)
        flat_token_idxs = np.concatenate(topk_token_idxs)
    else:
        flat_token_ids = np.empty(0, dtype=np.int64)
        flat_logprobs = np.empty(0, dtype=np.float32)
        flat_token_idxs = np.empty(0, dtype=np.int64)

    token_counts = TokenCounts(
        num_system_and_user_tokens=sum(
            len(ids)
            for message, ids in zip(messages, replacements, strict=True)
            if message.role != "assistant"
        ),
        num_assistant_tokens=sum(
            len(ids)
            for message, ids in zip(messages, replacements, strict=True)
            if message.role == "assistant"
        ),
    )
    return DatasetElement(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        topk_token_ids=torch.from_numpy(flat_token_ids),
        topk_logprobs=torch.from_numpy(flat_logprobs),
        topk_token_idxs=torch.from_numpy(flat_token_idxs),
        metadata=[],
        token_counts=token_counts,
    )


# Family-named entry points remain useful in configs/tests, while all boundaries
# are derived from each checkpoint's tokenizer rather than model-name constants.
glm_messages_to_element = tokenizer_messages_to_element
kimi_messages_to_element = tokenizer_messages_to_element


def qwen_messages_to_element(
    messages: List[Conversation.Message],
    retokenize: bool = False,
    tokenizer: PreTrainedTokenizerFast | None = None,
) -> DatasetElement:
    fn = _base_convert_messages_to_element_retokenize if retokenize else _base_convert_messages_to_element

    return fn(
        messages,
        tokenizer=tokenizer,
        message_start_tokens={
            "user": [151644, 872,198],
            "assistant": [151644, 77091,198],
            "system": [151644, 8948, 198],
        },
        message_end_tokens={
            "user": [151645],
            "assistant": [151645],
            "system": [151645],
        },
        message_extra_end_tokens={
            "user": [198],
            "assistant": [198],
            "system": [198],
        },
    )

def llama3_messages_to_element(
    messages: List[Conversation.Message],
    retokenize: bool = False,
    tokenizer: PreTrainedTokenizerFast | None = None,
) -> DatasetElement:
    fn = _base_convert_messages_to_element_retokenize if retokenize else _base_convert_messages_to_element

    return fn(
        messages,
        tokenizer=tokenizer,
        message_start_tokens={
            # "<|start_header_id|>", "user", "<|end_header_id|>", "\n\n"
            "user": [128006, 882, 128007, 271],
            # "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n\n"
            "assistant": [128006, 78191, 128007, 271],
            # "<|start_header_id|>", "system", "<|end_header_id|>", "\n\n"
            "system": [128006, 9125, 128007, 271],
        },
        message_end_tokens={
            # "<|eot_id|>"
            "user": [128009],
            "assistant": [128009],
            "system": [128009],
        },
        message_extra_end_tokens={
            "user": [],
            "assistant": [],
            "system": [],
        },
    )


MODEL_TO_MESSAGE_CONVERTER = {
    "Qwen/Qwen3-0.6b": qwen_messages_to_element,
    "Qwen/Qwen3-1.7b": qwen_messages_to_element,
    "Qwen/Qwen3-4b": qwen_messages_to_element,
    "Qwen/Qwen3-8b": qwen_messages_to_element,
    "Qwen/Qwen3-14b": qwen_messages_to_element,
    "Qwen/Qwen3-32b": qwen_messages_to_element,
    "meta-llama/Llama-3.1-8B-Instruct": llama3_messages_to_element,
    "meta-llama/Llama-3.2-3B-Instruct": llama3_messages_to_element,
    "meta-llama/Llama-3.2-1B-Instruct": llama3_messages_to_element,
    "zai-org/GLM-4.5": glm_messages_to_element,
    "zai-org/GLM-4.5-Air": glm_messages_to_element,
    "zai-org/GLM-4.5-FP8": glm_messages_to_element,
    "zai-org/GLM-4.6": glm_messages_to_element,
    "zai-org/GLM-4.6-FP8": glm_messages_to_element,
    "zai-org/GLM-4.7": glm_messages_to_element,
    "moonshotai/Kimi-K2-Instruct": kimi_messages_to_element,
    "moonshotai/Kimi-K2-Instruct-0905": kimi_messages_to_element,
    "moonshotai/Kimi-K2-Thinking": kimi_messages_to_element,
}
MODEL_TO_MESSAGE_CONVERTER = {k.lower(): v for k, v in MODEL_TO_MESSAGE_CONVERTER.items()}


@dataclass
class DatasetElement:
    input_ids: torch.Tensor

    metadata: list[dict[str, Any]]
    token_counts: TokenCounts

    topk_logprobs: Optional[torch.Tensor] = None
    topk_token_ids: Optional[torch.Tensor] = None
    topk_token_idxs: Optional[torch.Tensor] = None

@dataclass
class DatasetBatch:
    input_ids: torch.Tensor
    element_ids: torch.Tensor
    position_ids: torch.Tensor

    metadata: list[dict[str, Any]]
    token_counts: TokenCounts
    loss_weight: Optional[torch.Tensor] = None

    topk_logprobs: Optional[torch.Tensor] = None
    topk_token_ids: Optional[torch.Tensor] = None
    topk_token_idxs: Optional[torch.Tensor] = None


def msg(content, role: Literal["user"] | Literal["assistant"] | Literal["system"]):
    return {"content": content, "role": role}


class DataSource(BaseConfig):
    path: str
    type: Literal["local", "wandb", "hf"]
    limit: int | None = None

def _prepare_data_source(source: str | DataSource) -> list[Conversation]:
    if isinstance(source, str):
        is_local = ".pkl" in source or ".parquet" in source
        source = DataSource(path=source, type="local" if is_local else "wandb")
    
    if source.type == "local":
        data = read_conversations(source.path)
    elif source.type == "wandb":
        data = read_conversations_from_wandb(source.path)
    elif source.type == "hf":
        data = read_conversations_from_hf(source.path)
    else:
        raise ValueError(f"Unsupported data source type: {source.type}")

    return data[:source.limit] if source.limit is not None else data


class TrainDataset(Dataset):
    """ This dataset 

    - In "truncate" mode, if adding an element to the current batch would exceed the `seq_length`, the element 
    is added to a new batch instead. However, if an individual element's sequence length exceeds `seq_length`, 
    it is forced to be truncated and added to a new batch on its own.
    
    - In "pad" mode, elements are added to the current batch until adding another would exceed `seq_length`. 
    The current batch is then finalized, and a new batch is started. The final batch is padded to ensure 
    consistent batch sizes.

    Args:
        packing_mode (Literal["truncate", "pad"]): The mode of operation for batching, either "truncate" or "pad".
        packed_seq_length (int): The maximum allowed sequence length for each batch.
        shuffle (bool): Whether to shuffle the batches. Currently, only shuffling is supported.
    """

    class Config(ObjectConfig):
        _pass_as_config = True
        data_sources: list[str | DataSource]  
        is_wandb: bool = False
        top_k_logits: int = 20
        targets: Literal["logits", "tokens"] = "logits"

        packing_mode: Literal["truncate", "pad"]="pad"
        packed_seq_length: int = 2048

        user_prompt_prefix: list[str] | None = None


    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):

        self.config = config
        self.tokenizer = tokenizer
        
        self.elements: List[DatasetElement] = self._prepare_elements()
        # each batch is a list of element indices
        self.batches: List[List[int]] = self._prepare_batches(seed=seed)  

    def _prepare_elements(self) -> list[DatasetElement]:
        data = []
        for source in self.config.data_sources:
            data.extend(_prepare_data_source(source))

        elements = []
        for row in data:
            converter = MODEL_TO_MESSAGE_CONVERTER.get(
                self.tokenizer.name_or_path.lower(),
                tokenizer_messages_to_element,
            )
            elements.append(converter(
                row.messages,
                retokenize=self.config.targets == "tokens",
                tokenizer=self.tokenizer,
            ))

        return elements
    
    def _prepare_batches(self, seed: int) -> List[List[int]]:
        """
        This function attempts to create batches of dataset elements such that the total sequence length of each batch 
        does not exceed a specified limit (`seq_length`). The batching process can operate in two modes: "truncate" 
        and "pad". 

        Note that this function does not actually handle the truncation or padding, this is left to the collate function.
        Which is applied the fly in the dataloader worker. 
        """
        batches = [] 
        elem_idxs = random.Random(seed).sample(range(len(self.elements)), len(self.elements))
        queue = deque(elem_idxs)

        curr_batch, curr_seq_len = [], 0
        while queue:
            idx = queue[0]
            elem: DatasetElement = self._get_element(idx)
            
            if curr_seq_len == 0 and len(elem.input_ids) > self.config.packed_seq_length:
                # if the current element is by itself longer than the sequence length,
                # then the only option is to truncate it. So we just add it to the batch
                # and start a new batch.
                curr_batch.append(queue.popleft())
                batches.append(curr_batch)
                curr_batch, curr_seq_len = [], 0
            elif curr_seq_len + len(elem.input_ids) > self.config.packed_seq_length:
                # when the current batch would be too long if we add the current element,
                # if we are in truncate mode, then we just add the current element to the batch
                # and start a new batch. Otherwise, we just start a new batch.
                if self.config.packing_mode == "truncate":
                    curr_batch.append(queue.popleft())
                batches.append(curr_batch)
                curr_batch, curr_seq_len = [], 0
            else:
                # otherwise, we just add the current element to the batch and continue
                curr_batch.append(queue.popleft())
                curr_seq_len += len(elem.input_ids)
        
        if curr_batch:
            # need to add the last batch
            batches.append(curr_batch)
        
        return batches

    def __len__(self):
        return len(self.batches)
    
    def _get_element(self, elem_idx: int) -> DatasetElement:
        return self.elements[elem_idx]
    
    def _get_batch(self, batch_idx: int):
        elem_idxs = self.batches[batch_idx]
        elements = [self._get_element(elem_idx) for elem_idx in elem_idxs]
        return self.collate(elements)

    def __getitem__(self, index: int) -> DatasetBatch:
        return self._get_batch(index)
        
    def reload(self):
        # Check if dataset has data_source_indices attribute
        if hasattr(self, "data_source_indices") and self.data_source_indices:
            combined = list(zip(self.data, self.data_source_indices))
            random.shuffle(combined)
            self.data, self.data_source_indices = zip(*combined)
            self.data = list(self.data)
            self.data_source_indices = list(self.data_source_indices)
        else:
            # Just shuffle the data if no data_source_indices
            random.shuffle(self.data)

    def collate(
        self,
        batch: list[DatasetElement]
    ) -> DatasetBatch:
        """
        Collate a list of dataset elements into a single sequence of length `self.config.packed_seq_length`.
        The elements are packed into a single sequence 
        """
        if not batch:
            raise ValueError("Empty batch provided to collate function")

        input_ids, element_ids, position_ids = [], [], []
        topk_token_ids, topk_logprobs, topk_token_idxs = [], [], []
        metadatas = []
        token_counts = TokenCounts()
        curr_token_idx = 0
        for element_id, element in enumerate(batch):
            input_ids.append(element.input_ids)
            element_ids.append(torch.full_like(element.input_ids, element_id, dtype=torch.long))
            position_ids.append(torch.arange(len(element.input_ids), dtype=torch.long))
            topk_token_ids.append(element.topk_token_ids)
            topk_logprobs.append(element.topk_logprobs)
            topk_token_idxs.append(element.topk_token_idxs + curr_token_idx)
            metadatas.append(element.metadata)
            token_counts += element.token_counts
            curr_token_idx += len(element.input_ids)
        
        input_ids = torch.cat(input_ids, dim=0)
        element_ids = torch.cat(element_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        topk_token_ids = torch.cat(topk_token_ids, dim=0)
        topk_logprobs = torch.cat(topk_logprobs, dim=0)
        topk_token_idxs = torch.cat(topk_token_idxs, dim=0)

        if len(input_ids) > self.config.packed_seq_length:
            # if the input ids are longer than the sequence length, 
            # we need to truncate them
            input_ids = input_ids[:self.config.packed_seq_length]
            element_ids = element_ids[:self.config.packed_seq_length]
            position_ids = position_ids[:self.config.packed_seq_length]

            # we also need to filter out any targets that are from the truncated part of
            # the input ids
            mask = topk_token_idxs < self.config.packed_seq_length
            topk_token_ids = topk_token_ids[mask]
            topk_logprobs = topk_logprobs[mask]
            topk_token_idxs = topk_token_idxs[mask]

        elif len(input_ids) < self.config.packed_seq_length:
            # if the input ids are shorter than the sequence length, we need to pad them
            # it is critical that the sequence length stays constant to avoid 
            # flex attention recompiles.
            padding = torch.full((self.config.packed_seq_length - len(input_ids),), 0, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            element_ids = torch.cat([element_ids, padding])
            position_ids = torch.cat([position_ids, padding])

        return DatasetBatch(
            input_ids=input_ids,
            element_ids=element_ids,
            position_ids=position_ids,
            topk_token_ids=topk_token_ids,
            topk_logprobs=topk_logprobs,
            topk_token_idxs=topk_token_idxs,
            metadata=metadatas,
            token_counts=token_counts,
        )

class LossEvalDataset(TrainDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        
        # path to a file containing conversations 
        # accepted formats are: *.jsonl, *.parquet, *.pkl, a wandb artifact id, or a huggingface repo id
        data_source: str | DataSource

        is_wandb: bool = False
        targets: Literal["tokens"] = "tokens"

        packing_mode: Literal["truncate", "pad"]="pad"
        packed_seq_length: int = 2048

        system_prompt: str | None = None
        

        user_prompt_prefix: list[str] | None = None


    def _prepare_elements(self) -> list[Conversation]:
        data: list[Conversation] = []
        data = _prepare_data_source(self.config.data_source)

        elements = []
        for row in data:
            if self.config.system_prompt is not None:
                messages = [
                    Conversation.Message(role="system", content=self.config.system_prompt, token_ids=None),
                    *row.messages,
                ]
            else:
                messages = row.messages

            converter = MODEL_TO_MESSAGE_CONVERTER.get(
                self.tokenizer.name_or_path.lower(),
                tokenizer_messages_to_element,
            )
            elements.append(converter(
                messages,
                retokenize=self.config.targets == "tokens",
                tokenizer=self.tokenizer,
            ))
        
        return elements
    

@dataclass
class GenerateEvalDatasetElement:
    input_ids: torch.Tensor
    prompt: str

    answer: Optional[str]
    metadata: dict[str, Any]
    convo_id: Optional[str] = None
    
    # this is needed for some datasets, like MMLU, where the in context examples
    # are structured as prior messages
    prompt_messages: Optional[List[Dict[str,str]]] = None


@dataclass
class GenerateEvalDatasetElement:
    input_ids: torch.Tensor

    # messages to be used as the prompt
    prompt: List[MessageDict] | str

    answer: Optional[str]
    metadata: dict[str, Any]
    convo_id: Optional[str] = None

class GenerateEvalDataset(Dataset):
    class Config(ObjectConfig):
        _pass_as_config = True

        data_source: Optional[str | DataSource] = None
        cot: bool = False
    
    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer
        self.seed = seed

        assert self.config.data_source is not None, "data_source is required for GenerateEvalDataset"
        self.data: list[Conversation] = _prepare_data_source(self.config.data_source)

    @abstractmethod
    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        convo: Conversation = self.data[index]
        assert len(convo.messages) > 1, "Conversation must have at least 2 messages"
        assert convo.messages[-1].role == "assistant", "Last message must be assistant"

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = self.config.cot

        input_ids = self.tokenizer.apply_chat_template(
            [
                {"role": msg.role, "content": msg.content}
                for msg in convo.messages[:-1]
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,
        )

        return GenerateEvalDatasetElement(
            input_ids=input_ids,    
            prompt=[
                {"role": msg.role, "content": msg.content}
                for msg in convo.messages[:-1]
            ],
            answer=convo.messages[-1].content,
            convo_id=str(index),
            metadata={"idx": index}
        )
    
    def __len__(self):
        return len(self.data)

        