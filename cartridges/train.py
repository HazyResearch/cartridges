from __future__ import annotations
from abc import ABCMeta, abstractmethod
import contextlib
from dataclasses import dataclass, field
import itertools
import math
from math import cos, pi
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import Field
from pydrantic import BaseConfig, ObjectConfig, RunConfig
import torch
import torch.amp
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import wandb

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.datasets import (
    DatasetBatch,
    GenerateEvalDataset,
    LossEvalDataset,
    TrainDataset,
    DataSource,
)
from cartridges.models.config import ModelConfig
from cartridges.utils import get_logger, seed_everything
from cartridges.utils.wandb import download_artifacts, WandBConfig, prepare_wandb


logger = get_logger(__name__)


class LossEvalConfig(BaseConfig):
    dataset: LossEvalDataset.Config | TrainDataset.Config
    name_for_wandb: str

class GenerationEvalConfig(BaseConfig):
    dataset: GenerateEvalDataset.Config

    name_for_wandb: str
    generate_max_new_tokens: int = 128
    dataloader_num_workers: int = 0
    num_samples: int = 1
    num_samples_final: Optional[int] = None
    temperature: float = 0.0
    batch_size: int = 1
    override_max_tokens: int | None = None


class TrainConfig(RunConfig):
    name: str = "default"  # A name for the run for wandb
    output_dir: str = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")

    model: ModelConfig
    wandb: Optional[WandBConfig] = Field(default_factory=WandBConfig)
    dataset: TrainDataset.Config

    # datasets for evaluating perplexity on other generations
    # NOTE: steps here is the number of **optimizer steps**, which we keep track of
    # with the `optimizer_step` variable. This is different than the number of batches
    # processed, which is given by `iter_idx`.
    loss_eval_every_n_steps: Optional[int] = None
    loss_evals: list[LossEvalConfig] = field(default_factory=list)

    # datasets for actually producing generations
    # NOTE: steps here is the number of **optimizer steps**, which we keep track of
    # with the `optimizer_step` variable. This is different than the number of batches
    # processed, which is given by `iter_idx`.
    generate_eval_every_n_steps: Optional[int] = None
    generate_before_training: bool = True
    generate_evals: list[GenerationEvalConfig] = field(default_factory=list)

    # the `global_batch_size` is the total batch size across all devices and gradient
    # accumulation steps. We will infer the number of gradient accumulation steps from the
    # `global_batch_size`, and the number of devices.
    global_batch_size: int = 1

    epochs: int = 5
    device: str = "cuda"
    distributed_backend: Literal["nccl", "gloo"] = "nccl"

    optimizer: Literal["adam"] = "adam"
    lr: float = 1e-4
    lr_scheduler: Optional[Scheduler.Config] = None
    weight_decay: float = 0.0

    kv_cache_initializer: Optional[KVCacheFactory.Config] = None
    pretrained_cache_path: Optional[str] = None

    # NOTE: steps here is the number of **optimizer steps**, which we keep track of
    # with the `optimizer_step` variable. This is different than the number of batches
    # processed, which is given by `iter_idx`.
    save_every_n_steps: Optional[int] = 512
    save_after_training: bool = True
    keep_last_n_saved: int = 1
    save_to_wandb: bool = True
    log_time: bool = False

    max_optimizer_steps: int = -1

    seed: int = 42

    log_logprob_viz: bool = False

    def run(self):
        return train(self)

def train(config: TrainConfig):
    seed_everything(config.seed)
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        # SE (03/21): Sometimes, we want to run train multiple times within
        # a single launch of `torchrun`. So, we need to check if the process group
        # if the process group is already initialized, in which case we can reuse it.
        if not dist.is_initialized():
            dist.init_process_group(
                backend=config.distributed_backend, device_id=torch.device(local_rank)
            )
        logger.info(f"[Rank {dist.get_rank()}] initialized.")
        is_rank_zero = dist.get_rank() == 0
        num_devices = dist.get_world_size()
    else:
        local_rank = config.device
        is_rank_zero = True
        num_devices = 1

    # compute the correct number of gradient accumulation steps
    assert config.global_batch_size % num_devices == 0
    accumulate_grad_steps = (
        config.global_batch_size // num_devices
    )
    logger.info(f"Global batch size: {config.global_batch_size}")
    logger.info(f"Num devices: {num_devices}")

    logger.info(f"Train outputs will be saved to {config.run_dir}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)    

    t0 = time.time()
    dataset = config.dataset.instantiate(tokenizer=tokenizer, seed=config.seed)
    logger.info(
        f"Fininshed loading training dataset from disk, took {(time.time() - t0):.3}s"
    )

    ppl_evals: list[tuple[LossEvalConfig, TrainDataset]] = [    
        (ds_config, ds_config.dataset.instantiate(tokenizer=tokenizer, seed=config.seed))
        for ds_config in config.loss_evals
    ]
    generate_evals: list[tuple[GenerationEvalConfig, GenerateEvalDataset]] = [
        (eval_config, eval_config.dataset.instantiate(tokenizer=tokenizer, seed=config.seed))
        for eval_config in config.generate_evals
    ]
    logger.info(
        f"Finished loading eval and generate datasets from disk, took {(time.time() - t0):.2}s"
    )

    model = config.model.instantiate().to(local_rank).to(torch.bfloat16)
    attn_config=AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=(
            model.config.head_dim
            if hasattr(model.config, "head_dim")
            else model.config.hidden_size // model.config.num_attention_heads
        ),
    )
    
    assert config.model.tuning_method in ("custom_prefix", "peft")
    use_peft = config.model.tuning_method == "peft"
    
    cache = None
    if not use_peft:
        load_start_time = time.time()
        logger.info("Using custom prefix tuning (TrainableCache)")

        initializer = config.kv_cache_initializer.instantiate()
        cache: TrainableCache = initializer.initialize_kv_cache(
            tokenizer=tokenizer, model=model, attn_config=attn_config,
        )
        logger.info(
            f"Done loading trainable cache, time: {(time.time() - load_start_time):.2f}s"
        )
        cache_tuning = True
    else:
        cache_tuning = False


    # Different model wrapping logic based on tuning method
    if use_peft:
        # When using PEFT, we want to train only the PEFT parameters
        # PEFT models already have trainable parameters set up correctly
        logger.info("Using PEFT tuning method")
        wrapped_model = model
    else:
        # For custom prefix tuning, freeze the model params and train only the cache
        for param in model.parameters():
            param.requires_grad = False

        cache = cache.to(local_rank)
        wrapped_model = CacheAndModel(cache, model,)

    if is_ddp:

        train_sampler = DistributedSampler(
            dataset,
            shuffle=True,
            seed=config.seed,
        )

        logger.info("Wrapping model in DDP")
        logger.info(f"local_rank: {local_rank}")
        wrapped_model = DDP(wrapped_model, device_ids=[local_rank])
        dist.barrier()
    else:
        # SE (04/17): We need to set a seed for the random sampler so that the 
        # results are reproducible with or without DDP
        train_sampler = torch.utils.data.RandomSampler(
            dataset,
            replacement=False,
            num_samples=None,
            generator=torch.Generator().manual_seed(config.seed),
        )

    dataloader = DataLoader(
        dataset, 
        sampler=train_sampler,
        # our dataset already handles batching, so we force the batch size to 1
        # and extract it in the collate. We still use the dataloader to leverage
        # a single worker to avoid blocking the main process
        batch_size=1, 
        collate_fn=lambda x: x[0], 
        num_workers=1, 
    )

    optimizer = optim.Adam(
        wrapped_model.parameters() if use_peft else cache.parameters(), 
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Initialize counter variables
    # optimizer_step: number of optimizer steps taken, iter_idx: number of loop iterations, epoch_idx: number of epochs completed
    optimizer_step, iter_idx, epoch_idx = 0, 0, 0
    accum_num_input_tokens, accum_num_target_tokens = 0, 0
    total_num_input_tokens, total_num_target_tokens = 0, 0
    accum_loss = 0.0

    logger.info("Starting training")

    # Only set up W&B if rank 0 or running single-process
    if config.wandb is not None and is_rank_zero:
        config.wandb.name = config.name
        prepare_wandb(config.wandb, config.to_dict())

        wandb_log_dict = {
            "num_model_params": sum(p.numel() for p in wrapped_model.parameters()),
            "num_trainable_params": sum(
                p.numel() for p in wrapped_model.parameters() if p.requires_grad
            ),
        }

        if cache_tuning:
            wandb_log_dict.update(
                {
                    "num_trainable_tokens": cache._num_trainable_tokens,
                    "cache_trainable_params": sum(
                        p.numel() for p in cache.parameters() if p.requires_grad
                    ),
                }
            )

        wandb.log(
            wandb_log_dict,
            # SE (03/10): by setting commit=False, we avoid incrementing the step count
            # to 1. Without this, the first evaluation at step 0 will not be logged
            commit=False,
        )

        logger.info(f"Setup wandb with model stats: {wandb_log_dict}")


    def do_evaluation():
        for ds_config, dataset in ppl_evals:
            evaluate_perplexity(
                config=config,
                model=wrapped_model,
                cache=cache,
                eval_dataset=dataset,
                ds_config=ds_config,
                optimizer_step=optimizer_step,
                epoch=epoch_idx,
                local_rank=local_rank,
                cache_tuning=cache_tuning,
            )

    def do_evaluate_generations(step: int = None, final: bool = False):
        for eval_config, generate_dataset in generate_evals:
            evaluate_generations(
                config=eval_config,
                model=wrapped_model,
                tokenizer=tokenizer,
                dataset=generate_dataset,
                optimizer_step=optimizer_step,
                local_rank=local_rank,
                step=step,
                final=final,
                log_to_wandb=config.wandb is not None,
            )

    if config.lr_scheduler is not None:
        lr_scheduler: Scheduler = config.lr_scheduler.instantiate()
    else:
        lr_scheduler = None
    for epoch_idx in range(1, config.epochs + 1):

        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)

        train_pbar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch {epoch_idx}",
            leave=False,
            disable=not is_rank_zero,
        )
        logger.info(f"Dataloader length: {len(dataloader)}")
        for batch in train_pbar:
            batch: DatasetBatch
            do_step = (iter_idx + 1) % accumulate_grad_steps == 0

            if (
                config.loss_eval_every_n_steps is not None
                and optimizer_step % config.loss_eval_every_n_steps == 0
                and iter_idx % accumulate_grad_steps
                == 0  # only on the first batch of each optimizer step
            ):
                do_evaluation()

            
            if (
                config.generate_eval_every_n_steps is not None
                and optimizer_step % config.generate_eval_every_n_steps == 0
                and iter_idx % accumulate_grad_steps == 0
                and (config.generate_before_training or iter_idx > 0)
            ):
                do_evaluate_generations(step=iter_idx)

            # SE (05/02): We are careful to only reduce the loss across processes
            # when we are on the last batch of gradient accumulation before the optimizer
            # step. See here:
            ddp_ctx_manager = (
                contextlib.nullcontext()
                if do_step or not is_ddp
                else wrapped_model.no_sync()
            )
            with ddp_ctx_manager:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                    t0 = time.time()
                    outputs = wrapped_model(
                        input_ids=batch.input_ids.to(local_rank),
                        seq_ids=batch.element_ids.to(local_rank),
                        position_ids=batch.position_ids.to(local_rank),
                    )
                    if config.log_time:
                        torch.cuda.synchronize()
                        logger.info(f"Forward pass time: {time.time() - t0:.2f}s")

                    topk_pred_logprobs = F.log_softmax(outputs.logits, dim=-1)[
                        0, 
                        batch.topk_token_idxs.to(local_rank) - 1, 
                        batch.topk_token_ids.to(local_rank)
                    ] 

                    # ce is sum -p(x)logq(x), where p is the true distr and q is the model distr
                    ce_by_token = (
                        -batch.topk_logprobs.to(local_rank).exp()  # p(x), true distr
                        * topk_pred_logprobs  # q(x), model distr
                    )

                    loss = (ce_by_token.mean() / accumulate_grad_steps)

                # the backward pass should go outside of the automated-mixed precision context
                # see here for an example: https://pytorch.org/docs/stable/notes/amp_examples.html
                # but it should go inside the ddp context manager
                t0 = time.time()
                loss.backward()
                if config.log_time:
                    torch.cuda.synchronize()
                    logger.info(f"Backward pass time: {time.time() - t0:.2f}s")

            # Update the accumulated metrics
            accum_loss += loss.detach()
            accum_num_input_tokens += torch.tensor(
                batch.input_ids.size(0), device=local_rank
            )
            accum_num_target_tokens += torch.tensor(0, device=local_rank) # mask.sum().detach() # TODO: fix thisTI

            if do_step:
                optimizer.step()
                optimizer.zero_grad()              

                # SE (05/02): We are careful to only reduce the loss immediately
                # after the optimizer step. Doing this outside the `do_step` block
                # could introduce a race condition.
                if is_ddp:
                    dist.all_reduce(accum_loss, op=dist.ReduceOp.SUM)
                    accum_loss /= dist.get_world_size()  # Compute the mean loss
                    dist.all_reduce(accum_num_input_tokens, op=dist.ReduceOp.SUM)
                    dist.all_reduce(accum_num_target_tokens, op=dist.ReduceOp.SUM)

                if lr_scheduler is not None:
                    new_lr = lr_scheduler.get_lr(config.lr, optimizer_step)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr

            # Update chunk progress bar
            if is_rank_zero and do_step:
                train_pbar.set_postfix(
                    {
                        "loss": f"{accum_loss.item():.4f}",
                        "ppl": f"{torch.exp(accum_loss).item():.2f}",
                        "optimizer_step": f"{optimizer_step}",
                    }
                )

            if config.wandb is not None and is_rank_zero and do_step:
                total_num_input_tokens += accum_num_input_tokens.item()
                total_num_target_tokens += accum_num_target_tokens.item()
                wandb.log(
                    {
                        "train/loss": accum_loss,
                        "train/perplexity": torch.exp(accum_loss).item(),
                        "train/epoch_idx": epoch_idx,
                        "train/optimizer_step": optimizer_step,
                        "train/iter_idx": iter_idx,
                        "train/step_num_input_tokens": accum_num_input_tokens,
                        "train/step_num_target_tokens": accum_num_target_tokens,
                        "train/num_input_tokens": total_num_input_tokens,
                        "train/num_target_tokens": total_num_target_tokens,
                        **{
                            f"optimizer/lr_group{i}": param_group["lr"]
                            for i, param_group in enumerate(optimizer.param_groups)
                        },
                    },
                    step=optimizer_step,
                )

            if (
                config.save_every_n_steps is not None
                and optimizer_step > 0  # don't save on the first step
                and optimizer_step % config.save_every_n_steps == 0
                and is_rank_zero
            ) or (optimizer_step == config.max_optimizer_steps):
                if cache_tuning:
                    save_cache(config, cache, optimizer_step=optimizer_step)
                else:
                    assert use_peft

                    # Save intermediate PEFT model
                    logger.info(
                        f"Saving PEFT model at step {optimizer_step} to {config.run_dir}/peft_model_{optimizer_step}"
                    )
                    model.save_pretrained(
                        f"{config.run_dir}/peft_model_{optimizer_step}"
                    )

            if cache_tuning:
                cache.clear()

            iter_idx += 1
            if do_step:
                optimizer_step += 1
                accum_loss = 0.0
                accum_num_input_tokens, accum_num_target_tokens = 0, 0

    # We do a final evaluation and generation after training.
    # This is also useful for doing evaluation of a pretrained model without training
    do_evaluation()
    do_evaluate_generations(final=True)

    if config.save_after_training and is_rank_zero:
        if cache_tuning:
            save_cache(config, cache, optimizer_step=optimizer_step)
        else:
            # Save PEFT model
            logger.info(f"Saving PEFT model to {config.run_dir}/peft_model")
            model.save_pretrained(f"{config.run_dir}/peft_model")
    
    logger.info(f"Done training waiting for final barrier.")

    # SE (03/21): Careful to synchronize all processes before finishing in case
    # there's another call to `train` in the same process
    # after that will reuse the same process group.
    if is_ddp:
        dist.barrier()

    if config.wandb is not None and is_rank_zero:
        wandb.finish()


def evaluate_perplexity(
    config: TrainConfig,
    model,  # Can be either CacheAndModel or a PEFT model
    cache: Optional[TrainableCache],
    eval_dataset: LossEvalDataset,
    ds_config: LossEvalConfig,
    optimizer_step: int,
    epoch: int,
    local_rank: int,
    cache_tuning: bool,
):
    assert cache_tuning == (cache is not None)
    is_ddp = "LOCAL_RANK" in os.environ
    is_rank_zero = (not is_ddp) or (dist.get_rank() == 0)
    world_size = dist.get_world_size() if is_ddp else 1

    logger.info(
        f"Evaluating `{ds_config.name_for_wandb}` (n={len(eval_dataset)}, {len(eval_dataset) // world_size} per device, world size={world_size})"
    )

    sampler = DistributedSampler(eval_dataset)if is_ddp else None
    dataloader = DataLoader(
        eval_dataset,
        sampler=sampler,

        # our dataset already handles batching, so we force the batch size to 1
        # and extract it in the collate. We still use the dataloader to leverage
        # a single worker to avoid blocking the main process
        batch_size=1, 
        collate_fn=lambda x: x[0], 
        num_workers=1, 
    )

    dataloader_pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Evaluation [step={optimizer_step}] ({ds_config.name_for_wandb})",
        leave=False,
        disable=not is_rank_zero,
    )

    prefix = f"eval_{ds_config.name_for_wandb}"

    results = []
    with torch.no_grad():
        epoch_loss, epoch_denom = 0.0, torch.tensor(0, device="cuda")
        epoch_num_system_and_user_tokens = torch.tensor(0, device="cuda")
        epoch_num_assistant_tokens = torch.tensor(0, device="cuda")
        epoch_num_elements = torch.tensor(0, device="cuda")

        for batch in dataloader_pbar:
            batch: DatasetBatch
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch.input_ids.to(local_rank),
                    seq_ids=batch.element_ids.to(local_rank),
                    position_ids=batch.position_ids.to(local_rank),
                )

                topk_pred_logprobs = F.log_softmax(outputs.logits, dim=-1)[
                    0, 
                    batch.topk_token_idxs.to(local_rank) - 1, 
                    batch.topk_token_ids.to(local_rank)
                ] 

                # ce is sum -p(x)logq(x), where p is the true distr and q is the model distr
                ce_by_token = (
                    -batch.topk_logprobs.to(local_rank).exp()  # p(x), true distr
                    * topk_pred_logprobs  # q(x), model distr
                )

                epoch_loss += (ce_by_token.sum())
                epoch_denom += batch.input_ids.shape[0]

            if cache_tuning:
                assert cache is not None
                cache.clear()

            del outputs

            if is_ddp:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(epoch_denom, op=dist.ReduceOp.SUM)
                dist.all_reduce(epoch_num_system_and_user_tokens, op=dist.ReduceOp.SUM)
                dist.all_reduce(epoch_num_assistant_tokens, op=dist.ReduceOp.SUM)
                dist.all_reduce(epoch_num_elements, op=dist.ReduceOp.SUM)

        if is_ddp:
            dist.barrier()
        logger.info(f"Eval loss - {epoch_loss / epoch_denom} ")

    if is_ddp:
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_results, results)
        # flatten the list of lists
        results = [item for sublist in gathered_results for item in sublist]


    if config.wandb is not None and is_rank_zero:
        if results:
            elem_losses = [item["loss"] for item in results]
            macro_loss = sum(elem_losses) / len(elem_losses)
            macro_perplexity = math.exp(macro_loss)
        else:
            macro_loss = None
            macro_perplexity = None

        wandb.log(
            {
                f"{prefix}/loss": epoch_loss / epoch_denom,
                f"{prefix}/perplexity": math.exp(epoch_loss / epoch_denom),
                f"{prefix}/macro_loss": macro_loss,
                f"{prefix}/macro_perplexity": macro_perplexity,
                f"{prefix}/table": pd.DataFrame(results),
                f"{prefix}/num_system_and_user_tokens": epoch_num_system_and_user_tokens
                / epoch_num_elements,
                f"{prefix}/num_assistant_tokens": epoch_num_assistant_tokens
                / epoch_num_elements,
                f"{prefix}/num_elements": epoch_num_elements,
            },
            step=optimizer_step,
        )

    logger.info("done evaling - " + ds_config.name_for_wandb)

    if is_ddp:
        # SE (05/03): this barrier is just to be safe, can probably be removed
        dist.barrier()



def evaluate_generations(
    config: GenerationEvalConfig,
    
    model: CacheAndModel,
    tokenizer: AutoTokenizer,
    dataset: GenerateEvalDataset,
    optimizer_step: int,
    local_rank,
    step: int = None,
    final: bool = False,
    log_to_wandb: bool = True,
):
    from cartridges.generation import flex_generate

    is_ddp = "LOCAL_RANK" in os.environ
    is_rank_zero = (not is_ddp) or (dist.get_rank() == 0)
    world_size = dist.get_world_size() if is_ddp else 1

    
    if is_ddp:
        cache = model.module.cache
        model = model.module.model
    else:
        cache = model.cache
        model = model.model

    logger.info(
        f"Generating `{config.name_for_wandb}` (n={len(dataset)}, {len(dataset) // world_size} per device)"
    )

    has_score = hasattr(dataset, "score")
    has_batch_score = hasattr(dataset, "batch_score")
    prefix = f"generate_{config.name_for_wandb}"

    if config.num_samples_final is not None and final:
        num_samples = config.num_samples_final
    else:
        num_samples = config.num_samples

    batch_size = config.batch_size
    indexes = list(range(len(dataset)))
    if is_ddp:
        indexes = indexes[local_rank::world_size]

    results = []
    for batch_start in tqdm(
        range(0, len(indexes), batch_size),
        desc=f"Generating [step={optimizer_step}] ({config.name_for_wandb})",
        leave=False,
        disable=not is_rank_zero,
    ):
        for sample_idx in range(num_samples):
            logger.info(f"Generating sample {sample_idx} of {num_samples}")

            elements = [
                (i, dataset[indexes[i]])
                for i in range(batch_start, batch_start + batch_size)
                if i < len(indexes)
            ]
            if len(elements) == 0:
                continue
            input_ids = torch.cat([elem.input_ids[0] for _, elem in elements]).to(local_rank)
            seq_ids = torch.cat(
                [
                    torch.full((elem.input_ids.shape[1],), idx, dtype=torch.long, device=local_rank)
                    for idx, elem in elements
                ]
            )
            position_ids = torch.cat(
                [torch.arange(elem.input_ids.shape[1], device=local_rank) for _, elem in elements]
            )
            pred_ids: Dict[int, List[int]] = flex_generate(
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                cache=cache,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=(
                    config.generate_max_new_tokens
                    if config.override_max_tokens is None
                    else config.override_max_tokens
                ),
                temperature=config.temperature,
                show_progress=is_rank_zero
            )
            
            pred = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            elements = {seq_id: elem for seq_id, elem in elements}

            for  (seq_id, curr_pred_ids) in pred_ids.items():
                element = elements[seq_id]
                pred = tokenizer.decode(curr_pred_ids, skip_special_tokens=True)
                
                if has_score:
                    metrics, extras = dataset.score(
                        pred=pred, answer=element.answer, convo_id=element.convo_id
                    )
                else:
                    metrics, extras = None, {}

                if not isinstance(metrics, dict):
                    # Support for older datasets that return a single bool or float as metrics
                    metrics = {"score": metrics}
                else:
                    metrics = {f"{k}_score": v for k, v in metrics.items()}
                
                results.append(
                    {
                        "index": indexes[seq_id],
                        "optimizer_step": optimizer_step,
                        "prompt": element.prompt,
                        "answer": element.answer,
                        "pred": pred,
                        "convo_id": element.convo_id,
                        "sample_idx": sample_idx,
                        "num_system_and_user_tokens": element.input_ids.shape[1],
                        "num_assistant_tokens": len(pred_ids),
                        **metrics,
                        **element.metadata,
                        **extras,
                    }
                )
    logger.info(f"Generated {len(results)} samples")

    batch_score = None
    if has_batch_score:
        answers = [(result["index"], result["pred"], result["answer"]) for result in results]

        if is_ddp:
            gathered_answers = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_answers, answers)
            # flatten the list of lists
            answers = [item for sublist in gathered_answers for item in sublist]

        if is_rank_zero:
            pred = [i[1] for i in answers]
            gt = [i[2] for i in answers]
            batch_score = dataset.batch_score_with_answers(pred, gt)

    if is_ddp:
        dist.barrier()
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_results, results)
        # flatten the list of lists
        results = [item for sublist in gathered_results for item in sublist]

    if is_rank_zero:
        df = pd.DataFrame(results)

        if not df.duplicated(subset=["convo_id", "sample_idx"]).any():
            logger.warning(
                "There are duplicate convo_ids in the generation results, dropping duplicates rows."
            )
            df = df.drop_duplicates(subset=["convo_id", "sample_idx"])

        score_cols = [col for col in df.columns if col.endswith("score")]
        avg_scores = {f"{prefix}/{col}": df[col].mean() for col in score_cols}

        if log_to_wandb:
            log_dict = {
                **avg_scores,
                f"{prefix}/table": df,
                f"{prefix}/num_system_and_user_tokens": df[
                    "num_system_and_user_tokens"
                ].mean(),
                f"{prefix}/num_assistant_tokens": df["num_assistant_tokens"].mean(),
            }
            logger.info(avg_scores)

            if batch_score is not None:
                log_dict[f"{prefix}/batch_score"] = batch_score

            wandb.log(
                log_dict,
                step=optimizer_step,
            )

    if is_ddp:
        dist.barrier()
    
    return results



# ---- Learning rate scheduler code ----
# 
class Scheduler(metaclass=ABCMeta):
    class Config(ObjectConfig):
        _pass_as_config = True

        # This maximum number of steps is used to calculate the schedule, it is not
        # used to limit the number of steps that training runs for. Once the steps
        # exceed this, the schedule will simply use the final learning rate.
        max_steps: Optional[int]

        # The starting point for the warmup
        warmup_min_lr: Optional[float] = 5e-3

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        raise NotImplementedError

    def _linear_warmup(self, initial_lr: float, step: int, warmup_steps: int = 2000) -> float:
        warmup_min_lr = self.config.warmup_min_lr if self.config.warmup_min_lr is not None else initial_lr * 0.10
        assert 0 <= warmup_min_lr < initial_lr
        return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, warmup_steps) / warmup_steps


@dataclass
class CosWithWarmup(Scheduler):

    class Config(Scheduler.Config):
        warmup_steps: int
        alpha_f: float = 0.1
    
    def __init__(self, config: Config):
        self.config = config

    def get_lr(self, initial_lr: float, step: int) -> float:
        max_steps = self.config.max_steps
        eta_min = initial_lr * self.config.alpha_f
        if step < self.config.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.config.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.config.warmup_steps
            max_steps = max_steps - self.config.warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2


@dataclass
class LinearWithWarmup(Scheduler):
    class Config(Scheduler.Config):
        warmup_steps: int
        alpha_f: float = 0.1
    
    def __init__(self, config: Config):
        self.config = config

    def get_lr(self, initial_lr: float, step: int) -> float:
        max_steps = self.config.max_steps
        eta_min = initial_lr * self.config.alpha_f
        if step < self.config.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.config.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.config.warmup_steps
            max_steps = max_steps - self.config.warmup_steps
            return initial_lr - (initial_lr - eta_min) * (step / max_steps)


class CacheAndModel(nn.Module):
    def __init__(self, cache, model):
        super(CacheAndModel, self).__init__()
        self.cache = cache
        self.model = model


    def forward(
        self, 
        input_ids: torch.Tensor, 
        seq_ids: torch.Tensor, 
        position_ids: torch.Tensor,
        # labels: torch.Tensor
    ):

        out = self.model(
            input_ids=input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            # labels=labels,
            use_cache=True,
            past_key_values=self.cache
        )

        return out

def save_cache(config: TrainConfig, cache: TrainableCache, optimizer_step: int):
    """
    Saves the trainable cache to a file and manages saved checkpoints.

    Args:
        config (TrainConfig): The training configuration object
        cache (TrainableCache): The trainable cache to save
        optimizer_step (int): The current optimizer step number

    The cache is saved to {config.run_dir}/cache-step{step}.pt.
    Only keeps the most recent config.keep_last_n_saved checkpoints.
    If config.save_to_wandb is True, also saves the cache to wandb.
    """
    run_dir = Path(config.run_dir)
    run_dir.mkdir(exist_ok=True, parents=True)

    filename = f"cache-step{optimizer_step}.pt"
    save_path = Path(config.run_dir) / filename

    cache.save(save_path)

    # Create/update symlink to latest checkpoint
    symlink_path = os.path.join(config.run_dir, "cache_last.pt")
    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
        os.remove(symlink_path)
    os.symlink(save_path, symlink_path)

    # Save to wandb if configured

    if config.save_to_wandb and config.wandb is not None:
        logger.info(f"Saving cache to wandb: {filename}")
        # by passing base_path, we save the files to the root of the wandb run
        # instead of duplicating the full path including the run directory
        wandb.save(save_path, base_path=config.run_dir, policy="now")

    # Remove older saves if we exceed keep_last_n_saved
    pattern = r"^cache-epoch(\d+)\.pt$"

    # Find all saved checkpoints in the current directory
    all_checkpoints = [
        fname for fname in os.listdir(config.run_dir) if re.match(pattern, fname)
    ]

    # Sort the checkpoint filenames by their epoch number
    def parse_epoch(filename: str) -> int:
        match = re.search(pattern, filename)
        return int(match.group(1)) if match else 0

    all_checkpoints.sort(key=parse_epoch)

    # Remove oldest if we have more than keep_last_n_saved
    while len(all_checkpoints) > config.keep_last_n_saved:
        oldest = all_checkpoints.pop(0)
        os.remove(os.path.join(config.run_dir, oldest))


