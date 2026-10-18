import abc
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Optional

from pydrantic import ObjectConfig
import torch
import torch.nn as nn

from cartridges.utils import get_logger

logger = get_logger(__name__)

@dataclass
class AttnConfig:
    n_layers: int
    n_heads: int
    head_dim: int

CARTRIDGE_SEQ_ID = -1

class TrainableCache(nn.Module):
    """A trainable packed cache for generation with FlexAttention.
    
    The cache must do two things, which a standard Hugging Face cache does not:

    - Keep track of sequence membership of the cache and expose it to the model via
    the seq_ids method. The model will use this once per forward pass to construct 
    the appropriate block mask. 
    - Keep track of keys and values and expose them to the model in a packed manner via 
    the update method.
    
    TODO (Sabri): Ensure that tokens from the same sequence are contiguous. Eventually,
    should just page the keys and values.

    Args:
        config: The attention configuration, which we use to construct the 
        init_keys (list[torch.Tensor], optional): A `config.n_layers` length list of 
            trainable keys for the cache, should be of shape (1, n_heads, num_trainable_tokens, head_dim).
        init_values (list[torch.Tensor]): A `config.n_layers` length list of 
            trainable values for the cache, should be of shape (1, n_heads, num_trainable_tokens, head_dim).
        num_frozen_tokens (int): The number of the trainable tokens to freeze at the 
            beginning of the cache.
    """
    def __init__(
        self,        
        config: AttnConfig,
        init_keys: list[torch.Tensor]=None,
        init_values: list[torch.Tensor]=None,
        num_frozen_tokens: int = 0,
    ):
        super().__init__()
        self.config = config
        self._keys = [None] * config.n_layers  # List of tensors per layer
        self._values = [None] * config.n_layers  # List of tensors per layer
        self._num_tokens = 0

        assert (init_keys is None) == (init_values is None)
        if init_keys is None:
            self._num_trainable_tokens, self._num_frozen_tokens = 0, 0
            self.frozen_keys, self.frozen_values = None, None
            self.trainable_keys, self.trainable_values = None, None
            self._seq_ids = None
            self._init_seq_ids = None
        else:
            self._num_init_tokens = init_keys[0].shape[2]
            self._num_frozen_tokens = num_frozen_tokens
            self._num_trainable_tokens = self._num_init_tokens - num_frozen_tokens
            assert len(init_keys) == config.n_layers == len(init_values)
            
            # we initialize the seq ids for the first 
            # `num_trainable_tokens + num_frozen_tokens` tokens to -1, which means that 
            # the tokens are part of the cartridge and should be attended to by 
            # all tokens.
            _seq_ids =torch.full(
                (self._num_init_tokens,),
                fill_value=CARTRIDGE_SEQ_ID, 
                dtype=torch.long,
            )
            self.register_buffer("_init_seq_ids", _seq_ids)
            self.register_buffer("_seq_ids", _seq_ids)  # .to moves the tensor to the correct device

            for vec in itertools.chain(init_keys, init_values):
                assert vec.shape == (1, config.n_heads, self._num_init_tokens, config.head_dim)

            self.frozen_keys = nn.ParameterList(
                [
                    nn.Parameter(keys_vec[:, :, :num_frozen_tokens].contiguous())
                    for keys_vec in init_keys
                ]
                if num_frozen_tokens
                else []
            )
            self.frozen_values = nn.ParameterList(
                [
                    nn.Parameter(values_vec[:, :, :num_frozen_tokens].contiguous())
                    for values_vec in init_values
                ]
                if num_frozen_tokens
                else []
            )

            for param in itertools.chain(self.frozen_keys, self.frozen_values):
                param.requires_grad = False

            self.trainable_keys = nn.ParameterList(
                [
                    nn.Parameter(keys_vec[:, :, num_frozen_tokens:].contiguous())
                    for keys_vec in init_keys
                ]
            )
            self.trainable_values = nn.ParameterList(
                [
                    nn.Parameter(values_vec[:, :, num_frozen_tokens:].contiguous())
                    for values_vec in init_values
                ]
            )
            logger.info(f"num_trainable_tokens: {self._num_trainable_tokens}")
            logger.info(f"num_frozen_tokens: {self._num_frozen_tokens}")
                
    def update(
        self, 
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        new_seq_ids: torch.Tensor,
        layer_idx: int,
        skip_append: bool = False,
    ):
        """Update the cache with new keys and values while maintaining sequence contiguity.
        
        Args:
            new_keys: (1, num_heads, seq_len, head_dim) tensor of new keys
            new_values: (1, num_heads, seq_len, head_dim) tensor of new values  
            new_seq_ids: (seq_len,) tensor of sequence ids for the new tokens
            layer_idx: index of the layer in the model.
            skip_append: if True, do not append the new keys and values to the cache, 
                just return the concatenation of the new_keys and values. 
        """
        assert new_seq_ids.shape[0] == new_keys.shape[2]
        assert new_seq_ids.shape[0] == new_values.shape[2]

        if layer_idx == 0 and not skip_append:
            # we assume the same seq ids at every layer. This allows us to create
            # a single block mask for the entire model. 
            if self._seq_ids is None:
                self._seq_ids = new_seq_ids
            else:
                self._seq_ids = torch.cat([self._seq_ids, new_seq_ids], dim=0)
            self._num_tokens += new_keys.shape[2]
        
        keys = [new_keys]
        values = [new_values]

        if self._keys[layer_idx] is not None:
            # Concatenate along sequence dimension while maintaining contiguous sequences
            keys = [self._keys[layer_idx]] + keys
            values = [self._values[layer_idx]] + values

        if not skip_append:
            self._keys[layer_idx] = torch.cat(keys, dim=2)
            self._values[layer_idx] = torch.cat(values, dim=2)
        
        if self._num_trainable_tokens > 0:
            keys = [self.trainable_keys[layer_idx]] + keys
            values = [self.trainable_values[layer_idx]] + values
        
        if self._num_frozen_tokens > 0:
            keys = [self.frozen_keys[layer_idx]] + keys
            values = [self.frozen_values[layer_idx]] + values
        
        if self._num_trainable_tokens == 0 and self._num_frozen_tokens == 0:
            return self._keys[layer_idx], self._values[layer_idx]

        return torch.cat(keys, dim=2), torch.cat(values, dim=2)
    
    def num_tokens(self) -> int:
        """Get the sequence length of the cache."""
        return self._num_frozen_tokens + self._num_trainable_tokens + self._num_tokens
    
    def num_cartridge_tokens(self) -> int:
        """Get the number of tokens in the cartridge."""
        return self._num_frozen_tokens + self._num_trainable_tokens
    
    def seq_ids(self) -> torch.Tensor:
        """Returns the sequence ids of the cache."""
        return self._seq_ids
       
    def clear(self):
        self._keys = [None] * self.config.n_layers
        self._values = [None] * self.config.n_layers
        self._num_tokens = 0
        self._seq_ids = self._init_seq_ids

    def save(self, path: str):
        """Saves the trainable keys and values to the specified path."""
        torch.save(
            {
                "trainable_keys": self.trainable_keys,
                "trainable_values": self.trainable_values,
                "frozen_keys": self.frozen_keys,
                "frozen_values": self.frozen_values,
            },
            path,
        )

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None):
        if not isinstance(path, str):
            raise TypeError(f"path must be a string, got {type(path)}")
        print(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Ensure necessary keys are in the checkpoint
        for key in ["trainable_keys", "trainable_values", "frozen_keys", "frozen_values"]:
            if key not in checkpoint:
                raise KeyError(f"Key '{key}' not found in checkpoint")

        n_layers = len(checkpoint["trainable_keys"])
        n_heads = checkpoint["trainable_keys"][0].size(1)
        num_tokens = checkpoint["trainable_keys"][0].size(2)
        head_dim = checkpoint["trainable_keys"][0].size(3)

        if len(checkpoint["frozen_keys"]) != n_layers:
            raise AssertionError(
                "Mismatch in number of layers between trainable and fixed keys"
            )
        if checkpoint["frozen_keys"]:
            if (
                checkpoint["frozen_keys"][0].size(1) != n_heads
                or checkpoint["frozen_keys"][0].size(3) != head_dim
            ):
                raise AssertionError(
                    "Mismatch in head configuration between trainable and fixed keys"
                )

        config = AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
        # Here, num_tokens is inferred from trainable keys, but note that the total tokens may be different if fixed tokens exist.
        # The number of fixed tokens can be inferred from frozen_keys if available.
        # K/V tensors are (batch, n_heads, n_tokens, head_dim): tokens are dim 2.
        num_frozen_tokens = (
            checkpoint["frozen_keys"][0].size(2) if checkpoint["frozen_keys"] else 0
        )

        return cls(
            config=config,
            init_keys=[
                (
                    torch.cat([fixed, trainable], dim=2).contiguous()
                    if num_frozen_tokens > 0
                    else trainable
                )
                for fixed, trainable in zip(
                    checkpoint["frozen_keys"], checkpoint["trainable_keys"]
                )
            ],
            init_values=[
                (
                    torch.cat([fixed, trainable], dim=2).contiguous()
                    if num_frozen_tokens > 0
                    else trainable
                )
                for fixed, trainable in zip(
                    checkpoint["frozen_values"], checkpoint["trainable_values"]
                )
            ],
            num_frozen_tokens=num_frozen_tokens,
        )


class KVCacheFactory(abc.ABC):
    class Config(ObjectConfig):
        _pass_as_config = True

        # SE (03/26): we freeze the first token to prevent forgetting
        num_frozen_tokens: int = 1

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def initialize_kv_cache(
        self, tokenizer, model, attn_config: AttnConfig 
    ) -> TrainableCache:
        raise NotImplementedError()


class KVCacheFactoryWithStateSaving(abc.ABC):
    class Config(KVCacheFactory.Config):
        directory: str
        is_wandb: bool
        force_recreate: bool = False

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def initalize_kv_cache_impl(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> tuple[TrainableCache, dict]:
        raise NotImplementedError()

    @property
    def local_kv_cache_path(self) -> Path:
        # TODO: better file extension
        return Path(self.config.directory) / "kv_cache.torch"

    @property
    def local_metadata_path(self) -> Path:
        # TODO: better file extension
        return Path(self.config.directory) / "metadata.json"

    def maybe_load_cached(self) -> Optional[TrainableCache]:
        if self.config.force_recreate:
            return

        if not self.config.is_wandb:
            if self.local_kv_cache_path.exists():
                logger.info(
                    f"State Saving KV initializer: loading KV cache from: {self.local_kv_cache_path}"
                )
                return TrainableCache.from_pretrained(
                    str(self.local_kv_cache_path.absolute()),
                )

            return

        raise NotImplementedError("Need to add saving to wanb")

    def initalize_kv_cache(
        self, tokenizer, model, attn_config: AttnConfig
    ) -> TrainableCache:
        maybe_cache = self.maybe_load_cached()
        if maybe_cache is not None:
            assert (
                maybe_cache._num_trainable_tokens + maybe_cache._num_frozen_tokens
                == self.config.num_tokens
            )
            assert maybe_cache.config == attn_config
            return maybe_cache

        cache, metadata = self.initalize_kv_cache_impl(
            tokenizer, model, attn_config
        )

        Path(self.config.directory).mkdir(parents=True, exist_ok=True)

        with open(self.local_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        cache.save(str(self.local_kv_cache_path.absolute()))
        logger.info(
            f"State Saving KV initializer: saving KV cache to: {self.local_kv_cache_path}"
        )

        return cache
