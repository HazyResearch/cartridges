from __future__ import annotations

import asyncio
from numbers import Integral, Real
from typing import Any, Dict, List, Literal, Optional, Sequence

import aiohttp
import numpy as np
import requests
from pydantic import Field
from transformers import AutoTokenizer

from cartridges.clients.base import Client, ClientConfig, ClientResponse, ClientSample, TopLogprobs
from cartridges.clients.usage import Usage
from cartridges.utils import get_logger


logger = get_logger(__name__)


class SGLangClient(Client):
    """Batched client for SGLang's token-aligned native ``/generate`` API."""

    class Config(ClientConfig):
        model_name: str
        url: str

        tokenizer_name: Optional[str] = None
        tokenizer_revision: Optional[str] = None
        trust_remote_code: bool = False
        chat_template: Optional[str] = None
        custom_chat_template: Optional[str] = None

        chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict)
        thinking_template_key: Optional[str] = "enable_thinking"
        thinking_mode: Literal["toggleable", "always", "unsupported"] = "unsupported"

        max_retries: int = 10
        base_timeout: float = 90.0
        timeout_multiplier: float = 1.5
        on_failure: Literal["raise", "continue"] = "raise"

        validate_server_identity: bool = True
        verify_top_logprobs_at_startup: bool = True
        startup_top_logprobs: int = 20
        server_model_name: Optional[str] = None
        server_tokenizer_name: Optional[str] = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self.logger = get_logger("SGLangClient")
        self.url = config.url.rstrip("/")
        self._warned_always_thinking = False
        self._warned_unsupported_thinking = False

        if config.chat_template is not None and config.custom_chat_template is not None:
            raise ValueError("Set only one of chat_template and custom_chat_template")
        if config.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if config.startup_top_logprobs < 1:
            raise ValueError("startup_top_logprobs must be positive")

        model_info = self._get_model_info()
        tokenizer_name = config.tokenizer_name or model_info.get("tokenizer_path") or config.model_name
        self._validate_model_info(model_info, tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=config.tokenizer_revision,
            trust_remote_code=config.trust_remote_code,
        )

        if config.verify_top_logprobs_at_startup:
            self._verify_top_logprobs_support(config.startup_top_logprobs)

    def _get_model_info(self) -> Dict[str, Any]:
        try:
            response = requests.get(
                f"{self.url}/get_model_info",
                timeout=self.config.base_timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to read SGLang model information: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("SGLang /get_model_info did not return an object")
        if data.get("is_generation") is False:
            raise ValueError("SGLang server is not running a generation model")
        return data

    @staticmethod
    def _normalized_identity(value: Any) -> str:
        return str(value).rstrip("/").lower()

    def _validate_model_info(self, model_info: Dict[str, Any], tokenizer_name: str) -> None:
        if not self.config.validate_server_identity:
            return

        expected_model = self.config.server_model_name or self.config.model_name
        expected_tokenizer = self.config.server_tokenizer_name or tokenizer_name
        actual_model = model_info.get("model_path")
        actual_tokenizer = model_info.get("tokenizer_path")

        if actual_model is None or actual_tokenizer is None:
            raise ValueError(
                "SGLang /get_model_info must include model_path and tokenizer_path"
            )
        if self._normalized_identity(actual_model) != self._normalized_identity(expected_model):
            raise ValueError(
                f"SGLang model mismatch: expected {expected_model!r}, got {actual_model!r}"
            )
        if self._normalized_identity(actual_tokenizer) != self._normalized_identity(expected_tokenizer):
            raise ValueError(
                "SGLang tokenizer mismatch: "
                f"expected {expected_tokenizer!r}, got {actual_tokenizer!r}"
            )

    def _template_kwargs(self, enable_thinking: bool) -> Dict[str, Any]:
        kwargs = dict(self.config.chat_template_kwargs)
        mode = self.config.thinking_mode

        if mode == "toggleable":
            if not self.config.thinking_template_key:
                raise ValueError(
                    "thinking_template_key is required when thinking_mode='toggleable'"
                )
            kwargs[self.config.thinking_template_key] = enable_thinking
        elif mode == "always" and not enable_thinking and not self._warned_always_thinking:
            self.logger.warning(
                "Thinking was requested off, but this checkpoint is configured as "
                "always-thinking; preserving its token-aligned raw output."
            )
            self._warned_always_thinking = True
        elif mode == "unsupported" and enable_thinking and not self._warned_unsupported_thinking:
            self.logger.warning(
                "Thinking was requested, but this checkpoint has no configured thinking toggle."
            )
            self._warned_unsupported_thinking = True
        return kwargs

    def _render_chat(
        self,
        chat: List[Dict[str, Any]],
        *,
        enable_thinking: bool,
    ) -> str:
        template = self.config.custom_chat_template or self.config.chat_template
        return self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=template,
            **self._template_kwargs(enable_thinking),
        )

    @staticmethod
    def _sampling_params(
        *,
        temperature: float,
        stop: Optional[List[str]],
        max_completion_tokens: Optional[int],
        frequency_penalty: float,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
        }
        if max_completion_tokens is not None:
            params["max_new_tokens"] = max_completion_tokens
        if stop:
            params["stop"] = stop
        return params

    def _make_payload(
        self,
        prompts: str | List[str],
        *,
        temperature: float,
        stop: Optional[List[str]],
        max_completion_tokens: Optional[int],
        frequency_penalty: float,
        top_logprobs: Optional[int],
    ) -> Dict[str, Any]:
        if top_logprobs is not None and top_logprobs < 1:
            raise ValueError("top_logprobs must be positive or None")
        payload = {
            "text": prompts,
            "sampling_params": self._sampling_params(
                temperature=temperature,
                stop=stop,
                max_completion_tokens=max_completion_tokens,
                frequency_penalty=frequency_penalty,
            ),
            "return_logprob": top_logprobs is not None,
            "return_text_in_logprobs": False,
        }
        if top_logprobs is not None:
            payload["top_logprobs_num"] = top_logprobs
        return payload

    def _verify_top_logprobs_support(self, top_k: int) -> None:
        prompt = self._render_chat(
            [{"role": "user", "content": "Reply with one short word."}],
            enable_thinking=False,
        )
        payload = self._make_payload(
            prompt,
            temperature=0.0,
            stop=None,
            max_completion_tokens=1,
            frequency_penalty=0.0,
            top_logprobs=top_k,
        )
        try:
            response = requests.post(
                f"{self.url}/generate",
                json=payload,
                timeout=self.config.base_timeout,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                if len(data) != 1:
                    raise ValueError(f"expected one verification response, got {len(data)}")
                data = data[0]
            self._parse_sample(data, top_k)
        except Exception as exc:
            raise RuntimeError(
                "SGLang startup top-logprob verification failed. The server must honor "
                f"top_logprobs_num={top_k}: {exc}"
            ) from exc

    async def _send_generate(self, payload: Dict[str, Any]) -> Any:
        error: Optional[Exception] = None
        for retry_idx in range(self.config.max_retries):
            timeout = self.config.base_timeout * (
                self.config.timeout_multiplier ** retry_idx
            )
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as session:
                    async with session.post(
                        f"{self.url}/generate",
                        json=payload,
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            raise RuntimeError(f"HTTP {response.status}: {body}")
                        return await response.json(content_type=None)
            except Exception as exc:
                error = exc
                self.logger.warning(
                    "SGLang request failed (attempt "
                    f"{retry_idx + 1}/{self.config.max_retries}): {type(exc).__name__}: {exc}"
                )
                if retry_idx + 1 < self.config.max_retries:
                    await asyncio.sleep(2**retry_idx)

        if self.config.on_failure == "continue":
            self.logger.error(
                f"SGLang request failed after {self.config.max_retries} attempts: {error}"
            )
            return None
        raise RuntimeError(
            f"SGLang request failed after {self.config.max_retries} attempts"
        ) from error

    @staticmethod
    def _candidate(candidate: Any) -> tuple[float, int]:
        if isinstance(candidate, dict):
            if "logprob" in candidate and (
                "token_id" in candidate or "id" in candidate
            ):
                token_id = candidate.get("token_id", candidate.get("id"))
                return float(candidate["logprob"]), int(token_id)
            if len(candidate) == 1:
                token_id, logprob = next(iter(candidate.items()))
                return float(logprob), int(token_id)
        elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            if (
                len(candidate) >= 2
                and isinstance(candidate[0], Real)
                and isinstance(candidate[1], Integral)
            ):
                return float(candidate[0]), int(candidate[1])
        raise ValueError(f"Malformed SGLang top-logprob candidate: {candidate!r}")

    @classmethod
    def _parse_top_logprobs(cls, rows: Any, top_k: int) -> TopLogprobs:
        if not isinstance(rows, list):
            raise ValueError("meta_info.output_top_logprobs must be a list")

        parsed_rows: List[List[tuple[float, int]]] = []
        for row_idx, row in enumerate(rows):
            if not isinstance(row, (list, tuple)):
                raise ValueError(f"top-logprob row {row_idx} is not a list")
            parsed = sorted(
                (cls._candidate(candidate) for candidate in row),
                key=lambda item: item[0],
                reverse=True,
            )
            if len(parsed) != top_k:
                raise ValueError(
                    f"top-logprob row {row_idx} has {len(parsed)} entries; expected {top_k}"
                )
            if any(token_id < 0 for _, token_id in parsed):
                raise ValueError(f"top-logprob row {row_idx} contains a negative token ID")
            parsed_rows.append(parsed)

        return TopLogprobs(
            logprobs=np.asarray(
                [[logprob for logprob, _ in row] for row in parsed_rows],
                dtype=np.float32,
            ).reshape(len(parsed_rows), top_k),
            token_ids=np.asarray(
                [[token_id for _, token_id in row] for row in parsed_rows],
                dtype=np.int64,
            ).reshape(len(parsed_rows), top_k),
        )

    @staticmethod
    def _chosen_token_id(entry: Any) -> int:
        if isinstance(entry, dict):
            token_id = entry.get("token_id", entry.get("id"))
            if token_id is not None:
                return int(token_id)
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            if len(entry) >= 2 and isinstance(entry[1], Integral):
                return int(entry[1])
        raise ValueError(f"Malformed output_token_logprobs entry: {entry!r}")

    @classmethod
    def _parse_sample(
        cls,
        response: Any,
        top_k: Optional[int],
    ) -> tuple[ClientSample, Usage]:
        if not isinstance(response, dict):
            raise ValueError("SGLang generation response must be an object")
        meta = response.get("meta_info")
        if not isinstance(meta, dict):
            raise ValueError("SGLang response is missing meta_info")

        output_ids = response.get("output_ids")
        if not isinstance(output_ids, list) or any(
            not isinstance(token_id, Integral) for token_id in output_ids
        ):
            raise ValueError("SGLang response output_ids must be a list of integers")
        output_ids = [int(token_id) for token_id in output_ids]
        if any(token_id < 0 for token_id in output_ids):
            raise ValueError("SGLang response contains a negative completion token ID")

        completion_tokens = meta.get("completion_tokens")
        prompt_tokens = meta.get("prompt_tokens")
        if not isinstance(completion_tokens, Integral) or completion_tokens < 0:
            raise ValueError("meta_info.completion_tokens must be a nonnegative integer")
        if not isinstance(prompt_tokens, Integral) or prompt_tokens < 0:
            raise ValueError("meta_info.prompt_tokens must be a nonnegative integer")

        if len(output_ids) < completion_tokens:
            raise ValueError(
                f"completion token count is {completion_tokens}, but received "
                f"only {len(output_ids)} output IDs"
            )
        # Some SGLang releases expose incremental-detokenization prefix IDs in
        # output_ids. completion_tokens reliably identifies the generated suffix.
        completion_output_ids = (
            output_ids[-completion_tokens:] if completion_tokens else []
        )
        token_ids = completion_output_ids
        parsed_top_logprobs = None
        if top_k is not None:
            chosen = meta.get("output_token_logprobs")
            if not isinstance(chosen, list):
                raise ValueError("meta_info.output_token_logprobs must be a list")
            chosen_ids = [cls._chosen_token_id(entry) for entry in chosen]
            if chosen_ids != completion_output_ids:
                raise ValueError(
                    "generated suffix of output_ids does not match token IDs in "
                    "output_token_logprobs"
                )
            token_ids = chosen_ids
            rows = meta.get("output_top_logprobs")
            if not isinstance(rows, list) or len(rows) != completion_tokens:
                row_count = len(rows) if isinstance(rows, list) else "missing"
                raise ValueError(
                    f"output_top_logprobs has {row_count} rows; "
                    f"expected {completion_tokens}"
                )
            parsed_top_logprobs = cls._parse_top_logprobs(rows, top_k)

        if len(token_ids) != completion_tokens:
            raise ValueError(
                f"completion token count is {completion_tokens}, but received "
                f"{len(token_ids)} token IDs"
            )

        text = response.get("text")
        if not isinstance(text, str):
            raise ValueError("SGLang response text must be a string")
        return (
            ClientSample(
                text=text,
                token_ids=token_ids,
                top_logprobs=parsed_top_logprobs,
            ),
            Usage(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
            ),
        )

    async def chat(
        self,
        chats: List[List[Dict[str, Any]]],
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        max_completion_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        top_logprobs: Optional[int] = None,
        logprobs_start_message: Optional[int] = None,
        modal_upstream_id: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> ClientResponse:
        del logprobs_start_message, modal_upstream_id
        if kwargs:
            raise TypeError(f"Unsupported SGLang client arguments: {sorted(kwargs)}")
        if not chats:
            raise ValueError("chats must not be empty")

        prompts = [
            self._render_chat(chat, enable_thinking=enable_thinking) for chat in chats
        ]
        payload = self._make_payload(
            prompts,
            temperature=temperature,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            frequency_penalty=frequency_penalty,
            top_logprobs=top_logprobs,
        )
        raw_response = await self._send_generate(payload)
        if raw_response is None:
            return ClientResponse(
                samples=[
                    ClientSample(text="", token_ids=None, top_logprobs=None)
                    for _ in chats
                ],
                usage=Usage(),
            )

        if isinstance(raw_response, dict) and len(chats) == 1:
            raw_response = [raw_response]
        if not isinstance(raw_response, list):
            raise ValueError("Batched SGLang response must be a list")
        if len(raw_response) != len(chats):
            raise ValueError(
                f"Expected {len(chats)} SGLang responses, got {len(raw_response)}"
            )

        samples: List[ClientSample] = []
        usage = Usage()
        for response in raw_response:
            sample, sample_usage = self._parse_sample(response, top_logprobs)
            samples.append(sample)
            usage += sample_usage
        return ClientResponse(samples=samples, usage=usage)
