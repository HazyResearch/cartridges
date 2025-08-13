import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.mtob.evals import MTOBKalamangToEnglishGenerateDataset
from cartridges.initialization import KVFromRandomText
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset
from cartridges.utils.wandb import WandBConfig


NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "8192"))

MODEL = os.environ.get("MODEL", "llama")
if MODEL == "qwen":
    data_sources = [
        "/data/sabri/cartridges/2025-07-28-18-33-28-m07d28_mtob_synthesize/m07d28_mtob_synthesize_qwen3-4b_n65536-0/artifact/dataset.pkl",
        "/data/sabri/cartridges/2025-07-28-20-04-14-m07d28_mtob_synthesize/m07d28_mtob_synthesize_qwen3-4b_n65536-0/artifact/dataset.pkl"

    ]
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    )
elif MODEL == "llama":
    data_sources = [
        "/data/sabri/cartridges/2025-07-30-19-03-42-m07d28_mtob_synthesize/m07d28_mtob_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl",
        "/data/sabri/cartridges/2025-07-30-19-18-45-m07d28_mtob_synthesize/m07d28_mtob_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl"

    ]
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")

configs = []
for lr in [2e-2]:
    config = TrainConfig(
        model=model,
        kv_cache_initializer=KVFromRandomText.Config(
            max_tokens=NUM_TOKENS
        ),
        
        lr=lr,
        epochs=1,
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
                name_for_wandb=f"mtob-ke-test",
                dataset=MTOBKalamangToEnglishGenerateDataset.Config(use_cot=False),
                batch_size=16,
                generate_max_new_tokens=128,
                num_samples=1,
                temperature=0,
            ),
        ],
        loss_eval_every_n_steps=512,
        loss_evals=[],
        distributed_backend="gloo",

        wandb=WandBConfig(tags=["train", "mtob"]),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        name=FormatStringVariable("mtob_train_lr{lr}_toks{kv_cache_initializer.max_tokens}"),
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)