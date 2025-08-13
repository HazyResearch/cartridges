from __future__ import annotations
from abc import ABC
import asyncio
import json
import os
from pathlib import Path
import math

import time
from typing import Optional

import concurrent.futures
from pydrantic import RunConfig
from pydantic import Field
import pandas as pd
import tqdm
import wandb

from cartridges.synthesizers.base import AsyncConvoSynthesizer
from cartridges.structs import Conversation, write_conversations
from cartridges.utils import get_logger
from cartridges.utils.wandb import prepare_wandb, WandBConfig
from cartridges.utils.hf import upload_run_dir_to_hf


logger = get_logger(__name__)


class SynthesizeConfig(RunConfig):

    # the configuration for the synthesizer to use for the dataset
    # if you're not sure what to use, you should use `cartridges.synthesizers.SelfStudySynthesizer`.
    # however, you can also use your own synthesizer by subclassing `AsyncConvoSynthesizer`
    # and implementing the `sample_convos` method.
    # See `examples/arxiv/arxiv_synthesize.py` for an example.
    synthesizer: AsyncConvoSynthesizer.Config
    
    # total number of conversation to synthesize 
    num_samples: int

    # these two parameters (`batch_size` and `max_num_batches_in_parallel`) are critical 
    # for maximizing GPU utilization during synthesis. 
    # a batch is a set of conversations that are conditioned on the same chunk of the context
    # when `batch_size > 1`, we exploit prefix-sharing, which improves GPU utilization.
    # `max_num_batches_in_parallel` controls the number of workers, with each runing one
    # batch at a time.  As a very rough rule of thumb, you should aim to have 
    # `batch_size * num_batches_in_parallel` ~= 128 - 256. Larger batch sizes will hurt
    # data diversity, but lead to higher throughput. 
    # if you're using Modal or another configuration with horizontal autoscaling, be 
    # sure to coordinate `max_num_batches_in_parallel` with `allow_concurrent_inputs`
    # so that you can have about 128 - 256 conversations running per GPU at a time.
    batch_size: int
    max_num_batches_in_parallel: int
    worker_timeout: int = 6 * 60  # only allow six minutes between completed batches

    # --- BEGIN CONFIGURATIONS FOR LOGGING AND SAVING  ---

    # name of the run, will be used to name the run directory where the generated dataset
    # is saved
    name: Optional[str] = "synthesize"

    # wandb configuration, `upload_to_wandb` controls whether to upload the dataset to 
    # wandb as an artifact. `save_wandb_preview` controls whether to save a small 
    # preview of the dataset to the wandb for inspection. 
    wandb: Optional[WandBConfig] = Field(default_factory=WandBConfig)
    upload_to_wandb: bool = False
    save_wandb_preview: bool = True
    
    # whether to upload the generated dataset to huggingface (as a parquet file)
    # "hf_repo_id" is the name of the huggingface repo to upload to and it can 
    # include {wandb_run_id} and {name} in the repo id and they will be replaced
    # with the wandb run id and name of the run
    upload_to_hf: bool = False
    hf_repo_id: Optional[str] = None 

    # this is the root directory for outputs, a subdirectory will be created for the run
    # based on "name"
    output_dir: Optional[str] = Field(default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."))

    # --- END CONFIGURATIONS FOR LOGGING AND SAVING  ---


    def run(self):
        assert self.name is not None
        assert not self.upload_to_hf or self.hf_repo_id is not None

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        assert self.run_dir is not None
        self.run_dir = Path(self.run_dir)
        logger.info(f"Generating dataset with run dir: {self.run_dir}")

        if self.wandb is not None:
            self.wandb.name = self.name
            prepare_wandb(self.wandb, self.to_dict())

        total_batches = math.ceil(self.num_samples / self.batch_size)

        all_rows: list[Conversation] = asyncio.run(
            self._run_async_batches_with_queue(total_batches=total_batches),
        )
    
        t = time.time()
        logger.info(f"Generation done, starting to save {len(all_rows)} rows to artifact")

        if self.save_wandb_preview:
            _save_wandb_preview(all_rows)

        output_dir = self.run_dir / "artifact"
        output_dir.mkdir()
        final_output_path = output_dir / "dataset.parquet"
        write_conversations(all_rows, final_output_path)

        logger.info(f"Final output saved to {final_output_path}")

        if self.upload_to_wandb:
            artifact = wandb.Artifact(name=self.name, type="dataset")
            artifact.add_dir(local_path=str(output_dir.absolute()), name="dataset")
            wandb.log_artifact(artifact)

            # important to wait for the artifact to be saved so we get the version
            artifact.wait()
            logger.info(
                f"Saved dataset to wandb as artifact {artifact.name}, took {time.time() - t:.1f}s"
            )
        
        if self.upload_to_hf:
            hf_id = self.hf_repo_id.format(wandb_run_id=wandb.run.id, name=self.name)
            upload_run_dir_to_hf(self.run_dir, hf_id)

        wandb.finish()

    async def _run_async_batches_with_queue(
        self, 
        total_batches: int
    ) -> list[Conversation]:
        """Run batches using a queue for better control."""
        all_rows: list[Conversation] = []
        
        # Create queue of batch indices
        batch_queue = asyncio.Queue()
        for batch_idx in range(total_batches):
            batch_queue.put_nowait(batch_idx)
        
        # Results queue
        results_queue = asyncio.Queue()
        
        logger.info(f"Instantiating convo generator...")
        synthesizer = self.synthesizer.instantiate()
        await synthesizer.setup()  # this is needed to run async steps like starting MCP clients
        
        async def worker(worker_id: int):
            """Worker that processes batches from the queue.""" 
            
            while True:
                try:
                    # Get batch with timeout
                    batch_idx = await asyncio.wait_for(
                        batch_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    break
                
                print(f"Processing batch {batch_idx}")
                batch_rows = await _process_batch_async(
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    synthesizer=synthesizer,
                    config=self,
                )
                await results_queue.put((batch_idx, batch_rows))
                logger.info(f"Batch {batch_idx} completed")
                batch_queue.task_done()
        
        
        # Start workers
        workers = [
            asyncio.create_task(worker(i))
            for i in range(self.max_num_batches_in_parallel)
        ]
        
        # Collect results with progress tracking
        completed_batches = 0
        with tqdm.tqdm(total=total_batches, desc="Processing batches") as pbar:
            while completed_batches < total_batches:
                batch_idx, batch_rows = await asyncio.wait_for(
                    results_queue.get(),
                    timeout=self.worker_timeout
                )
                all_rows.extend(batch_rows)
                completed_batches += 1
                pbar.update(1)

        
        # Wait for all workers to finish
        await asyncio.gather(*workers, return_exceptions=True)
                    
        # Clean up synthesizer and its tools
        await synthesizer.cleanup()
        logger.info("Synthesizer cleanup completed successfully")

        return all_rows

async def _process_batch_async(
    batch_idx: int,
    total_batches: int,
    synthesizer: AsyncConvoSynthesizer,
    config: SynthesizeConfig,
) -> list[Conversation]:
    batch_size = min(
        config.batch_size,
        config.num_samples - batch_idx * config.batch_size,
    )

    try:
        convos = await synthesizer.sample_convos(batch_idx, batch_size, total_batches)
    except Exception as e:
        logger.error(
            f"\n{'='*60}\n"
            f"Error processing batch {batch_idx + 1}/{total_batches}\n"
            f"Exception Type: {type(e).__name__}\n"
            f"Exception Message: {e}\n"
            f"{'-'*60}\n"
            f"Full Traceback:\n",
            exc_info=True
        )
        return []

    return convos

def _save_wandb_preview(rows: list[Conversation]):
    import random
    sampled_rows = random.sample(rows, min(256, len(rows)))
    preview_df = pd.DataFrame(
        [
            {   
                "type": row.type,
                # SE (05/01): convert this to a string to avoid any wandb bugs
                "metadata": json.dumps(row.metadata, indent=2),
                **{f"message_{i}": row.messages[i].content for i in range(len(row.messages))},
            }
            for row in sampled_rows
        ]
    )
    wandb.log({"preview": preview_df})
