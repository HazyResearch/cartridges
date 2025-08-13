import os

import pydrantic

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.mtob.evals import MTOBKalamangToEnglishGenerateDataset
from cartridges.data.mtob.resources import MTOBResource
from cartridges.evaluate import ICLBaseline, EvaluateConfig
from cartridges.evaluate import GenerationEvalConfig

from cartridges.utils.wandb import WandBConfig


client = TokasaurusClient.Config(
    url="http://0.0.0.0:10210",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

SYSTEM_PROMPT_TEMPLATE = f"""Please reference the material below to help the user translate from Kalamang to English.

{{content}}"""


configs = [
    EvaluateConfig(
        name=f"mtob_baseline",
        generator=ICLBaseline.Config(
            client=client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.0,
            max_completion_tokens=128,
            enable_thinking=False,
            context=MTOBResource.Config(
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                setup="medium_and_sentences",
            )
        ),
        eval=GenerationEvalConfig(
            name_for_wandb=f"mmtob-ke-test",
            dataset=MTOBKalamangToEnglishGenerateDataset.Config(use_cot=False),
            batch_size=16,
            generate_max_new_tokens=128,
            num_samples=1,
            temperature=0.0,
        ),
        max_num_batches_in_parallel=32,
        batch_size=32,
        wandb=WandBConfig(tags=[f"mtob", "genbaseline", "icl"]),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
