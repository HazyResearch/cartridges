from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydrantic import BaseConfig, ObjectConfig
from pydantic import BaseModel
from cartridges.clients.usage import Usage
from dataclasses import dataclass, asdict

class CartridgeConfig(BaseConfig):
    id: str
    source: Literal["huggingface", "wandb"]
    force_redownload: bool = False

class ClientConfig(ObjectConfig):
    _pass_as_config: bool = True

    model_name: str

    show_progress_bar: bool = False

    def instantiate(self, *args, **kwargs) -> "Client":
        return super().instantiate(*args, **kwargs)


class Client(ABC):
    def __init__(self, config: ClientConfig):
        self.config = config

    @abstractmethod
    async def chat(
        self, 
        chats: List[List[Dict[str, Any]]], 
        temperature: float = 0.6, 
        stop: List[str] = [], 
        max_completion_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        top_logprobs: int = 1,
        logprobs_start_message: Optional[int] = None,
        modal_upstream_id: Optional[str] = None,
    ) -> ClientResponse:
        raise NotImplementedError


@dataclass(slots=True)
class ClientResponse:
    samples: List[ClientSample]
    usage: Usage
    
    timings: Optional[List[Dict[str, Any]]] = None
    def to_dict(self):
        return asdict(self)


@dataclass(slots=True)
class ClientSample:
    text: str
    token_ids: Optional[List[int]] = None

    top_logprobs: Optional[TopLogprobs] = None



@dataclass(slots=True)
class FlatTopLogprobs:
    """
    A flat / sparse view produced by `TopLogprobs.flatten`.

    token_idx  – 1-D  [N]   row number in the original matrix
    token_id   – 1-D  [N]   vocabulary id
    logprobs   – 1-D  [N]   natural-log probabilities
    shape      – tuple(int,int) – (num_tokens , num_top_logprobs)
    """
    token_idx: np.ndarray
    token_id:  np.ndarray
    logprobs:  np.ndarray
    shape:     tuple[int, int]

    # ──────────────────────────────────────────────────────────
    # Inflate the sparse view back to dense [T , K] tensors.
    # Missing entries: token_id = −1 , logprob = −∞
    # ──────────────────────────────────────────────────────────
    def reconstruct(self) -> "TopLogprobs":
        T, K = self.shape
        dense_logp = np.full((T, K), -np.inf, dtype=self.logprobs.dtype)
        dense_ids  = np.full((T, K), -1,      dtype=self.token_id.dtype)

        # The kept entries for each row always occupy the first *n*
        # columns, where n = (token_idx == row).sum().
        # We fill the matrix row-by-row (vectorised per row, tiny loop
        # over T only – negligible cost).
        for row in range(T):
            row_mask = self.token_idx == row          # boolean mask
            n_keep   = row_mask.sum()
            if n_keep:                                # skip empty rows
                dense_logp[row, :n_keep] = self.logprobs[row_mask]
                dense_ids [row, :n_keep] = self.token_id[row_mask]

        return TopLogprobs(logprobs=dense_logp, token_ids=dense_ids)


@dataclass(slots=True)
class TopLogprobs:
    """
    logprobs  – [num_tokens , num_top_logprobs]  (sorted, ln p)
    token_ids – [num_tokens , num_top_logprobs]
    """
    logprobs:  np.ndarray
    token_ids: np.ndarray

    # ──────────────────────────────────────────────────────────
    # Flatten while retaining only the leading columns whose
    # cumulative probability per row ≥ `threshold`
    # ──────────────────────────────────────────────────────────
    def flatten(self, threshold: float = 0.99) -> FlatTopLogprobs:
        if self.logprobs.ndim != 2 or self.token_ids.ndim != 2:
            raise ValueError("logprobs and token_ids must be 2-D arrays")
        if self.logprobs.shape != self.token_ids.shape:
            raise ValueError("Shapes of logprobs and token_ids differ")
        if not (0.0 < threshold <= 1.0):
            raise ValueError("threshold must be in (0, 1]")

        T, K = self.logprobs.shape

        # 1. cumulative probability mass (rows already sorted)
        probs      = np.exp(self.logprobs)                 # [T , K]
        cum_mass   = np.cumsum(probs, axis=1)              # [T , K]

        # 2. per-row cut-off index (inclusive). If the returned top-k mass
        # never reaches the threshold, retain all K entries.
        reaches_threshold = cum_mass >= threshold
        cut_idx = reaches_threshold.argmax(axis=1)
        cut_idx = np.where(reaches_threshold.any(axis=1), cut_idx, K - 1)

        # 3. build a boolean mask: keep columns 0 … cut_idx[row]
        mask       = np.arange(K) < (cut_idx[:, None] + 1)    # [T , K]

        # 4. flatten
        token_idx  = np.repeat(np.arange(T), K)[mask.ravel()]  # [N]
        token_id   = self.token_ids[mask]                      # [N]
        logprobs   = self.logprobs[mask]                       # [N]

        return FlatTopLogprobs(
            token_idx=token_idx,
            token_id=token_id,
            logprobs=logprobs,
            shape=(T, K),
        )