import numpy as np
import pytest

from cartridges.clients.base import FlatTopLogprobs
from cartridges.datasets import (
    glm_messages_to_element,
    kimi_messages_to_element,
    tokenizer_messages_to_element,
)
from cartridges.structs import Conversation


class TemplateTokenizer:
    name_or_path = "test/template-model"

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        chat_template=None,
    ):
        assert tokenize is True
        output = [1]
        for message in messages:
            output.extend([2, *self.encode(message["role"]), 3])
            output.extend(self.encode(message["content"]))
            output.append(900)
        return output


def sparse_targets(num_tokens):
    return FlatTopLogprobs(
        token_idx=np.repeat(np.arange(num_tokens), 2),
        token_id=np.arange(num_tokens * 2) + 1000,
        logprobs=np.tile(np.array([-0.1, -1.0], dtype=np.float32), num_tokens),
        shape=(num_tokens, 2),
    )


@pytest.mark.parametrize(
    "converter",
    [glm_messages_to_element, kimi_messages_to_element],
)
def test_tokenizer_derived_family_converters_preserve_generated_ids(converter):
    tokenizer = TemplateTokenizer()
    generated_ids = [500, 501, 900]
    messages = [
        Conversation.Message(role="user", content="question", token_ids=None),
        Conversation.Message(
            role="assistant",
            content="answer",
            token_ids=generated_ids,
            top_logprobs=sparse_targets(len(generated_ids)),
        ),
    ]

    element = converter(messages, tokenizer=tokenizer)
    input_ids = element.input_ids.tolist()

    generated_start = input_ids.index(500)
    assert input_ids[generated_start:generated_start + 3] == generated_ids
    assert input_ids[-2:] != [900, 900]
    np.testing.assert_array_equal(
        element.topk_token_idxs.numpy(),
        np.repeat(np.arange(3) + generated_start, 2),
    )


def test_tokenizer_derived_converter_round_trips_retokenized_messages():
    tokenizer = TemplateTokenizer()
    messages = [
        Conversation.Message(role="system", content="system", token_ids=None),
        Conversation.Message(role="user", content="question", token_ids=None),
        Conversation.Message(role="assistant", content="answer", token_ids=None),
    ]

    element = tokenizer_messages_to_element(
        messages,
        retokenize=True,
        tokenizer=tokenizer,
    )
    expected = tokenizer.apply_chat_template(
        [{"role": message.role, "content": message.content} for message in messages],
        tokenize=True,
        add_generation_prompt=False,
        chat_template=None,
    )

    assert element.input_ids.tolist() == expected


def test_tokenizer_derived_converter_rejects_unsafe_boundaries():
    class ContextSensitiveTokenizer(TemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            output = super().apply_chat_template(messages, **kwargs)
            if not any("cartridges_message" in message["content"] for message in messages):
                output.insert(-1, 777)
            return output

    with pytest.raises(ValueError, match="round-trip failed"):
        tokenizer_messages_to_element(
            [Conversation.Message(role="user", content="hello", token_ids=None)],
            tokenizer=ContextSensitiveTokenizer(),
        )
