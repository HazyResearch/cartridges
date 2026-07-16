import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import requests

from cartridges.clients.base import ClientResponse
from cartridges.clients.sglang import SGLangClient


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, chat, **kwargs):
        self.calls.append((chat, kwargs))
        return f"rendered:{chat[-1]['content']}"


def make_client(**config_overrides):
    config = SGLangClient.Config(
        model_name="test/model",
        url="http://sglang.test",
        verify_top_logprobs_at_startup=False,
        **config_overrides,
    )
    client = object.__new__(SGLangClient)
    client.config = config
    client.url = config.url
    client.tokenizer = FakeTokenizer()
    client.logger = MagicMock()
    client._warned_always_thinking = False
    client._warned_unsupported_thinking = False
    return client


def generation_response(*, output_ids=(7, 8), top_k=3, text="answer"):
    rows = []
    for token_id in output_ids:
        rows.append([
            [-2.0, token_id + 20, None],
            [-0.1, token_id, None],
            [-1.0, token_id + 10, None],
        ][:top_k])
    return {
        "text": text,
        "output_ids": list(output_ids),
        "meta_info": {
            "prompt_tokens": 5,
            "completion_tokens": len(output_ids),
            "output_token_logprobs": [
                [-0.1, token_id, None] for token_id in output_ids
            ],
            "output_top_logprobs": rows,
        },
    }


@pytest.mark.parametrize(
    ("family", "thinking_mode", "enable_thinking", "expected_kwargs"),
    [
        ("qwen", "toggleable", True, {"enable_thinking": True}),
        ("glm", "toggleable", False, {"enable_thinking": False}),
        ("kimi", "always", False, {}),
    ],
)
def test_model_capability_profiles(
    family, thinking_mode, enable_thinking, expected_kwargs
):
    client = make_client(
        thinking_mode=thinking_mode,
        chat_template_kwargs={},
    )

    assert client._template_kwargs(enable_thinking) == expected_kwargs
    if family == "kimi":
        client.logger.warning.assert_called_once()


def test_named_and_custom_chat_templates_are_forwarded():
    named = make_client(chat_template="thinking")
    named._render_chat([{"role": "user", "content": "hi"}], enable_thinking=False)
    assert named.tokenizer.calls[0][1]["chat_template"] == "thinking"

    custom = make_client(custom_chat_template="{{ messages }}")
    custom._render_chat([{"role": "user", "content": "hi"}], enable_thinking=False)
    assert custom.tokenizer.calls[0][1]["chat_template"] == "{{ messages }}"


@pytest.mark.asyncio
async def test_chat_batches_prompts_and_aggregates_usage():
    client = make_client(thinking_mode="toggleable")
    client._send_generate = AsyncMock(
        return_value=[
            generation_response(output_ids=(7, 8), text="first"),
            generation_response(output_ids=(9,), text="second"),
        ]
    )

    response = await client.chat(
        [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ],
        temperature=0.5,
        stop=["STOP"],
        max_completion_tokens=10,
        frequency_penalty=0.2,
        top_logprobs=3,
        enable_thinking=True,
        modal_upstream_id="ignored",
    )

    assert isinstance(response, ClientResponse)
    assert [sample.text for sample in response.samples] == ["first", "second"]
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 3
    payload = client._send_generate.await_args.args[0]
    assert payload["text"] == ["rendered:one", "rendered:two"]
    assert payload["sampling_params"] == {
        "temperature": 0.5,
        "frequency_penalty": 0.2,
        "max_new_tokens": 10,
        "stop": ["STOP"],
    }
    assert payload["return_logprob"] is True
    assert payload["top_logprobs_num"] == 3

    first = response.samples[0]
    assert first.token_ids == [7, 8]
    assert first.top_logprobs.logprobs.shape == (2, 3)
    assert first.top_logprobs.token_ids.shape == (2, 3)
    np.testing.assert_array_equal(first.top_logprobs.token_ids[:, 0], [7, 8])
    assert np.all(np.diff(first.top_logprobs.logprobs, axis=1) <= 0)


@pytest.mark.asyncio
async def test_chat_without_logprobs_still_returns_exact_output_ids():
    client = make_client()
    raw = generation_response(output_ids=(31, 32))
    raw["meta_info"].pop("output_token_logprobs")
    raw["meta_info"].pop("output_top_logprobs")
    client._send_generate = AsyncMock(return_value=[raw])

    response = await client.chat(
        [[{"role": "user", "content": "hi"}]],
        max_completion_tokens=2,
    )

    assert response.samples[0].token_ids == [31, 32]
    assert response.samples[0].top_logprobs is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda response: response["meta_info"].update(completion_tokens=3),
            "completion token count",
        ),
        (
            lambda response: response["meta_info"]["output_top_logprobs"].pop(),
            "output_top_logprobs",
        ),
        (
            lambda response: response["meta_info"]["output_top_logprobs"][0].pop(),
            "entries",
        ),
        (
            lambda response: response.update(output_ids=[99, 8]),
            "does not match",
        ),
    ],
)
def test_malformed_or_misaligned_responses_fail(mutate, message):
    response = generation_response()
    mutate(response)
    with pytest.raises(ValueError, match=message):
        SGLangClient._parse_sample(response, 3)


def test_output_ids_incremental_decode_prefix_is_ignored():
    response = generation_response(output_ids=(7, 8))
    response["output_ids"] = [101, 102, 7, 8]

    sample, _ = SGLangClient._parse_sample(response, 3)

    assert sample.token_ids == [7, 8]


@pytest.mark.asyncio
async def test_continue_mode_returns_empty_samples_after_batch_failure():
    client = make_client(on_failure="continue")
    client._send_generate = AsyncMock(return_value=None)

    response = await client.chat(
        [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ]
    )

    assert [sample.text for sample in response.samples] == ["", ""]
    assert all(sample.token_ids is None for sample in response.samples)


@pytest.mark.asyncio
async def test_native_request_retries_with_growing_timeouts(monkeypatch):
    client = make_client(max_retries=2, base_timeout=1, timeout_multiplier=2)
    timeouts = []

    class FailingSession:
        def __init__(self, timeout):
            timeouts.append(timeout.total)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            raise RuntimeError("temporary failure")

    monkeypatch.setattr(
        "cartridges.clients.sglang.aiohttp.ClientSession", FailingSession
    )
    monkeypatch.setattr(
        "cartridges.clients.sglang.asyncio.sleep", AsyncMock()
    )

    assert await client._send_generate({"text": ["prompt"]}) is None
    assert timeouts == [1, 2]


def test_startup_validates_identity_and_top20_support():
    model_info_response = MagicMock()
    model_info_response.json.return_value = {
        "model_path": "test/model",
        "tokenizer_path": "test/tokenizer",
        "is_generation": True,
    }
    generate_response_mock = MagicMock()
    generate_response_mock.json.return_value = generation_response(
        output_ids=(7,), top_k=20
    )
    generate_response_mock.json.return_value["meta_info"]["output_top_logprobs"] = [
        [[-float(i + 1), i, None] for i in range(20)]
    ]

    tokenizer = FakeTokenizer()
    with (
        patch(
            "cartridges.clients.sglang.requests.get",
            return_value=model_info_response,
        ),
        patch(
            "cartridges.clients.sglang.requests.post",
            return_value=generate_response_mock,
        ) as post,
        patch(
            "cartridges.clients.sglang.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as from_pretrained,
    ):
        SGLangClient(
            SGLangClient.Config(
                model_name="test/model",
                tokenizer_name="test/tokenizer",
                url="http://sglang.test/",
            )
        )

    from_pretrained.assert_called_once_with(
        "test/tokenizer",
        revision=None,
        trust_remote_code=False,
    )
    assert post.call_args.kwargs["json"]["top_logprobs_num"] == 20


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("SGLANG_URL") or not os.getenv("SGLANG_MODEL_FAMILY"),
    reason="Set SGLANG_URL and SGLANG_MODEL_FAMILY for a live SGLang test",
)
async def test_live_server_top20_alignment():
    family = os.environ["SGLANG_MODEL_FAMILY"].lower()
    if family not in {"glm", "kimi", "qwen"}:
        pytest.fail("SGLANG_MODEL_FAMILY must be glm, kimi, or qwen")

    model_info = requests.get(
        f"{os.environ['SGLANG_URL'].rstrip('/')}/get_model_info",
        timeout=30,
    ).json()
    profiles = {
        "glm": {"thinking_mode": "toggleable"},
        "kimi": {"thinking_mode": "always", "trust_remote_code": True},
        "qwen": {"thinking_mode": "toggleable"},
    }
    client = SGLangClient(
        SGLangClient.Config(
            model_name=model_info["model_path"],
            tokenizer_name=model_info["tokenizer_path"],
            url=os.environ["SGLANG_URL"],
            **profiles[family],
        )
    )

    response = await client.chat(
        [[{"role": "user", "content": "Reply with one short sentence."}]],
        temperature=0.0,
        max_completion_tokens=16,
        top_logprobs=20,
        enable_thinking=False,
    )
    sample = response.samples[0]
    assert sample.token_ids
    assert sample.top_logprobs.logprobs.shape == (len(sample.token_ids), 20)
    assert np.all(sample.top_logprobs.token_ids >= 0)
