import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.mtob.resources import MTOBResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig


client = TokasaurusClient.Config(
    url="http://0.0.0.0:10210",
    model_name="Qwen/Qwen3-4b",
)


config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        use_tools_a=False, 
        use_tools_b=False,

        tools=[],
        resources=[
            MTOBResource.Config(
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                setup="latex_and_sentences",
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65536, 
    batch_size=32,    # Smaller batches 
    
    max_num_batches_in_parallel=256,

    name=FormatStringVariable("mtob_synthesize_n{num_samples}"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(tags=[f"mtob_synthesis"]),
    upload_to_wandb=False,
    save_wandb_preview=False,
)


if __name__ == "__main__": 
    pydrantic.main([config])