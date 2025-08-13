from pathlib import Path

import pydrantic

from cartridges.clients.openai import CartridgeConfig, OpenAIClient
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.evaluate import ICLBaseline, EvaluateConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.evaluate import GenerationEvalConfig

from cartridges.utils.wandb import WandBConfig

client = OpenAIClient.Config(
    base_url="https://hazyresearch--vllm-qwen3-4b-1xh100-serve.modal.run/v1",
    model_name="Qwen/Qwen3-4b",
)

client = OpenAIClient.Config(
    base_url="http://localhost:10210/v1/cartridge",
    model_name="Qwen/Qwen3-8b",
    cartridges=[CartridgeConfig(
        id="1wqgicqv",
        source="wandb",
        force_redownload=False
    )]
)

SYSTEM_PROMPT_TEMPLATE = f"""Please reference the patient medical records included below to answer the user's questions.

<patient-records>
{{content}}
</patient-records>

Do not think for too long (only a few sentences, you only have 512 tokens to work with).
"""


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

configs = [
    EvaluateConfig(
        name=f"longhealth_mc_{patients_str}",
        generator=ICLBaseline.Config(
            client=client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.3,
            max_completion_tokens=2048,
            context=LongHealthResource.Config(
                patient_ids=patient_ids,
            ),
        ),
        eval=GenerationEvalConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids, 
                cot=True,
            ),
            name_for_wandb=f"longhealth_mc",
            num_samples=1,
            temperature=0.3,
        ),
        max_num_batches_in_parallel=32,
        batch_size=32,
        wandb=WandBConfig(
            tags=[f"longhealth", "genbaseline", f"patients_{patients_str}", "icl"],
        ),
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
