import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.ruler.evals import NIAHGenerateDataset
from cartridges.initialization import KVFromRandomText
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset
from cartridges.utils.wandb import WandBConfig

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))

NUM_KEYS = (2, 2)
num_keys_str = f"k{NUM_KEYS[0]}_{NUM_KEYS[1]}"

MODEL = os.environ.get("MODEL", "llama")
if MODEL == "llama":
    DATA_SOURCES = {
        (1, 2): [
            "/data/sabri/cartridges/2025-07-28-10-41-03-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl",

            "/data/sabri/cartridges/2025-07-28-11-40-25-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl",

            "/data/sabri/cartridges/2025-07-28-11-46-58-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl",

            "/data/sabri/cartridges/2025-07-28-12-08-34-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl",
        ],
        (1, 1): [
            "/data/sabri/cartridges/2025-07-28-12-18-54-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536_k1-0/artifact/dataset.pkl",
            "/data/sabri/cartridges/2025-07-28-12-28-46-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536_k1-0/artifact/dataset.pkl",
        ],
        (2, 2): [
            "/data/sabri/cartridges/2025-07-28-14-04-29-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536_k2_2-0/artifact/dataset.pkl",
            "/data/sabri/cartridges/2025-07-28-15-02-07-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536_k2_2-0/artifact/dataset.pkl",
            "/data/sabri/cartridges/2025-07-28-15-17-49-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536_k2_2-0/artifact/dataset.pkl",
            "/data/sabri/cartridges/2025-07-28-15-26-52-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536_k2_2-0/artifact/dataset.pkl",
        ]
    }
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")


NUM_KEYS_TO_PATH = {
    (1, 1): "/home/sabri/code/cartridges/cartridges/data/ruler/_data/qwen3_4b-l100000-n1-k128-v1_1-essay-key_words-val_numbers-e83970e8.json",
    (1, 2): "/home/sabri/code/cartridges/cartridges/data/ruler/_data/qwen3_4b-l100000-n1-k128-v1_2-essay-key_words-val_numbers--1660737731696865120.json",
    (2, 2): "/home/sabri/code/cartridges/cartridges/data/ruler/_data/qwen3_4b-l100000-n1-k128-v2_2-essay-key_words-val_numbers-a7104531.json",
}

data_sources = DATA_SOURCES[NUM_KEYS]
niah_path = NUM_KEYS_TO_PATH[NUM_KEYS]

config = TrainConfig(
    model=model,
    kv_cache_initializer=KVFromRandomText.Config(max_tokens=NUM_TOKENS),
    
    lr=2e-2,
    loss_type="logits",
    epochs=2,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=data_sources,
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    generate_eval_every_n_steps=128,
    generate_evals=[
        GenerationEvalConfig(
            dataset=NIAHGenerateDataset.Config(
                niah_path=niah_path,
                thinking=True,
            ),
            name_for_wandb=f"niah_mc",
            num_samples=8,
            override_max_tokens=256,
            temperature=0.3,
            batch_size=16,
        ),
        
    ],
    loss_eval_every_n_steps=512,
    loss_evals=[],
    distributed_backend="gloo",

    wandb=WandBConfig(tags=["train", "niah", num_keys_str]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable("niah_train_lr{lr}_toks{kv_cache_initializer.max_tokens}_{num_keys_str}"),
)


if __name__ == "__main__":
    pydrantic.main(config)