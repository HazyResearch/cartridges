import re  # Added for CoT answer extraction
from cartridges.data.mtob import (
    load_test_ek,
    load_test_ke,
    load_train_examples,
)  # Added load_train_examples
from cartridges.datasets import (
    CartridgeTrainDataset,
    CartridgeGenerateDataset,
    TEMPLATE,
    CartridgeGenerateDatasetElement,
)
from cartridges.structs import ContextConvo, Message
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig
import evaluate

# --- Prompts ---


def prompt_direct(input_sentence, source_language, target_language):
    return f"""You are tasked with translating the following sentence from {source_language} to {target_language}: "{input_sentence}".
I understand that you may not be familiar enough with {source_language} or {target_language} to make a confident translation, but please give your best guess.
Respond with only the translation and no other text."""


def prompt_cot(input_sentence, source_language, target_language):
    return f"""You are tasked with translating the following sentence from {source_language} to {target_language}: "{input_sentence}".
I understand that you may not be familiar enough with {source_language} or {target_language} to make a confident translation, but please try your best.

First, think step-by-step about the translation process. Consider grammar, word choice, and potential ambiguities. Write down your thoughts.

Finally, provide the most likely translation enclosed within <answer> tags. For example: <answer>Your translation here.</answer>
Respond with only the thinking steps followed by the final tagged answer."""


chrf_metric = evaluate.load("chrf")


def make_eval_dataset(dataset, source_language, target_language):
    """Prepares data for evaluation datasets (using ContextConvo)."""
    return [
        ContextConvo(
            messages=[
                Message(
                    role="user",
                    # Eval datasets typically use direct prompting
                    content=prompt_direct(
                        item["original"], source_language, target_language
                    ),
                ),
                Message(role="assistant", content=item["ground_truth"]),
            ],
            type=f"Mtob_translation_{source_language}_to_{target_language}",
            metadata={},
        )
        for item in dataset  # Changed variable name for clarity
    ]


def extract_answer_from_cot(text: str) -> str:
    """Extracts text within <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # Fallback: return the original text if tags are not found
        # Consider adding logging here if available: log.warning("Could not find <answer> tags in output: %s", text)
        return text.strip()


class MtobEnglishToKalamangEvalDataset(CartridgeTrainDataset):
    class Config(CartridgeTrainDataset.Config):
        _pass_as_config = True
        label_type: str = "tokens"
        data_sources: list = []

    def __init__(self, config: Config, tokenizer):
        self.config = config
        # Prepare data structure for eval format
        eval_data = [
            {"original": item["original"], "ground_truth": item["ground_truth"]}
            for item in load_test_ek()
        ]
        self.data = make_eval_dataset(
            eval_data,
            source_language="English",
            target_language="Kalamang",
        )
        self.tokenizer = tokenizer


class MtobKalamangToEnglishEvalDataset(CartridgeTrainDataset):
    class Config(CartridgeTrainDataset.Config):
        _pass_as_config = True
        label_type: str = "tokens"
        data_sources: list = []

    def __init__(self, config: Config, tokenizer):
        self.config = config
        # Prepare data structure for eval format
        eval_data = [
            {"original": item["original"], "ground_truth": item["ground_truth"]}
            for item in load_test_ke()
        ]
        self.data = make_eval_dataset(
            eval_data,
            source_language="Kalamang",
            target_language="English",
        )
        self.tokenizer = tokenizer


class MtobKalamangToEnglishTrainDataset(CartridgeTrainDataset):
    class Config(CartridgeTrainDataset.Config):
        _pass_as_config = True
        label_type: str = "tokens"
        data_sources: list = []

    def __init__(self, config: Config, tokenizer):
        self.config = config
        # Prepare data structure for eval format
        eval_data = [
            {"original": item["translation"], "ground_truth": item["original"]}
            for item in load_train_examples()
        ]
        self.data = make_eval_dataset(
            eval_data,
            source_language="Kalamang",
            target_language="English",
        )
        self.tokenizer = tokenizer


class MtobEnglishToKalamangTrainDataset(CartridgeTrainDataset):
    class Config(CartridgeTrainDataset.Config):
        _pass_as_config = True
        label_type: str = "tokens"
        data_sources: list = []

    def __init__(self, config: Config, tokenizer):
        self.config = config
        # Prepare data structure for eval format
        eval_data = [
            {"original": item["original"], "ground_truth": item["translation"]}
            for item in load_train_examples()
        ]
        self.data = make_eval_dataset(
            eval_data,
            source_language="English",
            target_language="Kalamang",
        )
        self.tokenizer = tokenizer


class MtobGenerateDataset(CartridgeGenerateDataset):
    class Config(CartridgeGenerateDataset.Config):
        _pass_as_config = True
        use_cot: bool = False  # Added CoT config flag
        # ignored
        label_type: str = "tokens"
        data_sources: list = []

    # Added config argument to __init__
    def __init__(self, config: Config, tokenizer):
        self.config = config  # Store config
        self.tokenizer = tokenizer
        # Data loading and setting source/target languages will happen in subclasses

    def __getitem__(self, index: int) -> CartridgeGenerateDatasetElement:
        row = self.data[index]

        # Select prompt based on config
        if self.config.use_cot:
            prompt_func = prompt_cot
        else:
            prompt_func = prompt_direct

        user_content = prompt_func(row["original"], self.source, self.target)

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CartridgeGenerateDatasetElement(
            input_ids=input_ids,
            prompt=user_content,  # Keep original sentence as prompt context if needed
            answer=row["ground_truth"],
            convo_id=f"mtob_{self.source}_to_{self.target}_index_{index}",
            metadata={"idx": index, "use_cot": self.config.use_cot},  # Added metadata
        )

    def __len__(self):
        return len(self.data)

    def batch_score(self, decoded: list[str]) -> int:
        assert len(decoded) == len(self.data)

        assert not self.config.use_cot
        # predictions = [
        #     extract_answer_from_cot(d) if self.config.use_cot else d for d in decoded
        # ]

        predictions = []
        for d in decoded:
            if d.endswith("<|eot_id|>"):
                d = d[: -len("<|eot_id|>")]

            predictions.append(d)

        references = [d["ground_truth"] for d in self.data]

        return chrf_metric.compute(
            predictions=predictions,
            references=references,
        )["score"]

    def batch_score_with_answers(
        self,
        decoded: list[str],
        references: list[str],
    ) -> float:
        predictions = []
        for d in decoded:
            if d.endswith("<|eot_id|>"):
                d = d[: -len("<|eot_id|>")]

            predictions.append(d)

        return chrf_metric.compute(
            predictions=predictions,
            references=references,
        )["score"]


# --- Generation Datasets using TEST data ---


class MtobEnglishToKalamangGenerateDataset(MtobGenerateDataset):
    class Config(MtobGenerateDataset.Config):  # Inherit from base Config
        pass  # No changes needed here unless specific overrides

    def __init__(self, config: Config, tokenizer):
        super().__init__(config, tokenizer)  # Call parent __init__
        # load_test_ek() returns [{'original': ..., 'ground_truth': ...}]
        self.data = load_test_ek()
        self.source = "English"
        self.target = "Kalamang"


class MtobKalamangToEnglishGenerateDataset(MtobGenerateDataset):
    class Config(MtobGenerateDataset.Config):  # Inherit from base Config
        pass  # No changes needed here unless specific overrides

    def __init__(self, config: Config, tokenizer):
        super().__init__(config, tokenizer)  # Call parent __init__
        # load_test_ke() returns [{'original': ..., 'ground_truth': ...}]
        self.data = load_test_ke()
        self.source = "Kalamang"
        self.target = "English"


class MtobEnglishToKalamangTrainGenerateDataset(MtobGenerateDataset):
    class Config(MtobGenerateDataset.Config):  # Inherit from base Config
        pass

    def __init__(self, config: Config, tokenizer):
        super().__init__(config, tokenizer)
        # load_train_examples() returns [{'original': English, 'translation': Kalamang, ...}]
        # Map to {'original': ..., 'ground_truth': ...} format
        raw_data = load_train_examples()
        self.data = [
            {"original": item["original"], "ground_truth": item["translation"]}
            for item in raw_data
        ]
        self.source = "English"
        self.target = "Kalamang"


class MtobKalamangToEnglishTrainGenerateDataset(MtobGenerateDataset):
    class Config(MtobGenerateDataset.Config):  # Inherit from base Config
        pass

    def __init__(self, config: Config, tokenizer):
        super().__init__(config, tokenizer)
        # load_train_examples() returns [{'original': English, 'translation': Kalamang, ...}]
        # Map to {'original': ..., 'ground_truth': ...} format, swapping source/target
        raw_data = load_train_examples()
        self.data = [
            {"original": item["translation"], "ground_truth": item["original"]}
            for item in raw_data
        ]
        self.source = "Kalamang"
        self.target = "English"


def mtob_eval_datasets(
    local_batch_size,
):
    return [
        EvalDatasetConfig(
            name_for_wandb="mmtob-ke-eval",
            dataset=MtobKalamangToEnglishEvalDataset.Config(),
            local_batch_size=local_batch_size,
            only_eval_rank_0=True,
        ),
        EvalDatasetConfig(
            name_for_wandb="mmtob-ek-eval",
            dataset=MtobEnglishToKalamangEvalDataset.Config(),
            local_batch_size=local_batch_size,
            only_eval_rank_0=True,
        ),
        EvalDatasetConfig(
            name_for_wandb="mmtob-ke-train",
            dataset=MtobKalamangToEnglishTrainDataset.Config(),
            local_batch_size=local_batch_size,
            only_eval_rank_0=True,
        ),
        EvalDatasetConfig(
            name_for_wandb="mmtob-ek-train",
            dataset=MtobEnglishToKalamangTrainDataset.Config(),
            local_batch_size=local_batch_size,
            only_eval_rank_0=True,
        ),
    ]


def mtob_generate_datasets(
    batch_size: int = 16, use_cot: bool = False, num_samples: int = 1, temperature: float = 0.0
):  # Added use_cot flag
    """
    Configures generation datasets for both test and train splits.

    Args:
        use_cot: Whether to configure the datasets to use Chain-of-Thought prompting.
    """
    cot_suffix = "-cot" if use_cot else ""
    return [
        GenerateDatasetConfig(
            name_for_wandb=f"mmtob-ke-test{cot_suffix}",
            dataset=MtobKalamangToEnglishGenerateDataset.Config(use_cot=use_cot),
            batch_size=batch_size,
            num_samples=num_samples,
            temperature=temperature,
        ),
        GenerateDatasetConfig(
            name_for_wandb=f"mmtob-ek-test{cot_suffix}",
            dataset=MtobEnglishToKalamangGenerateDataset.Config(use_cot=use_cot),
            batch_size=batch_size,
            num_samples=num_samples,
            temperature=temperature,
        ),
        # GenerateDatasetConfig(
        #     name_for_wandb=f"mmtob-ke-train{cot_suffix}",
        #     dataset=MtobKalamangToEnglishTrainGenerateDataset.Config(use_cot=use_cot),
        #     batch_size=batch_size,
        # ),
        # GenerateDatasetConfig(
        #     name_for_wandb=f"mmtob-ek-train{cot_suffix}",
        #     dataset=MtobEnglishToKalamangTrainGenerateDataset.Config(use_cot=use_cot),
        #     batch_size=batch_size,
        # ),
    ]
