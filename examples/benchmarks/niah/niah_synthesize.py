import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.ruler.resources import NIAHResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig



client = TokasaurusClient.Config(
    url="http://0.0.0.0:10210",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

NUM_KEYS = (2, 2)
num_keys_str = f"k{NUM_KEYS[0]}_{NUM_KEYS[1]}"

BASE_PATH = os.path.join(os.environ.get("CARTRIDGES_DATA_DIR"), "ruler", "_data")

NUM_KEYS_TO_PATH = {
    (1, 1): f"{BASE_PATH}/qwen3_4b-l100000-n1-k128-v1_1-essay-key_words-val_numbers-e83970e8.json",
    (1, 2): f"{BASE_PATH}/qwen3_4b-l100000-n1-k128-v1_2-essay-key_words-val_numbers--1660737731696865120.json",
    (2, 2): f"{BASE_PATH}/qwen3_4b-l100000-n1-k128-v2_2-essay-key_words-val_numbers-a7104531.json",
}

niah_path = NUM_KEYS_TO_PATH[NUM_KEYS]

config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        use_tools_a=False, 
        use_tools_b=False,
        
        tools=[],
        resources=[
            NIAHResource.Config(
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                niah_path=niah_path,
                sentences_per_chunk=(1, 1),
                chunks_per_prompt=(64, 256),
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65536, 
    batch_size=32,    # Smaller batches 
    
    max_num_batches_in_parallel=256,

    name=FormatStringVariable("niah_synthesize_n{num_samples}_{num_keys_str}"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(tags=["niah", "synthesis"]),
    upload_to_wandb=False,
    save_wandb_preview=False,
)


if __name__ == "__main__": 
    pydrantic.main([config])