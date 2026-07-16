Add an SGLang teacher-rollout client

Design





Add [cartridges-project/cartridges/clients/sglang.py](cartridges-project/cartridges/clients/sglang.py) implementing the existing Client.chat() contract.



Render each chat with the server-reported Hugging Face tokenizer and add_generation_prompt=True, then send the rendered prompts as one native SGLang /generate batch. Support tokenizer revisions, named Hugging Face templates, and an explicit custom-template override so local native-API rendering can exactly match the deployed server.



Add a small configuration-driven model capability profile instead of model-name branches: chat_template_kwargs, an optional thinking_template_key, and a thinking mode such as toggleable, always, or unsupported. Map the existing enable_thinking argument only for toggleable templates, preserve raw token-aligned output for always-thinking models, and emit a clear warning when a caller requests thinking off but the checkpoint cannot disable it.



Cover the requested families explicitly: Qwen3/Qwen3.5 uses enable_thinking when toggleable; GLM behavior is selected per checkpoint because GLM-4.x and newer hybrid checkpoints differ; Kimi K2 Thinking is treated as always-thinking, while instruct checkpoints can use a non-thinking profile. Keep SGLang reasoning/tool parser flags in deployment documentation because they are server concerns, not client-side architecture assumptions.



Request return_logprob=true and top_logprobs_num=K; map each response’s text, output_ids, meta_info.output_top_logprobs, and usage counts into ClientSample, TopLogprobs, and ClientResponse. Prefer meta_info.output_token_logprobs token IDs as the source of completion IDs (they are unambiguously per-generated-token), cross-checking against output_ids and completion_tokens since output_ids semantics have shifted across SGLang releases.



Load the tokenizer with trust_remote_code configurable (required for some GLM and Kimi checkpoints), validate server model/tokenizer identity via /get_model_info at startup, and verify once at startup that the server actually honors top_logprobs_num=20 (issue reports exist of /generate silently ignoring it) plus per-request that top-k rows equal completion_tokens and each row has K entries. Fail loudly on any misalignment rather than degrading silently, and never retokenize decoded output or infer IDs from token strings.



Defensively sort each top-k row by descending logprob before building TopLogprobs, since flatten()'s cumulative-mass cutoff silently produces wrong sparse targets on unsorted rows.



Set generation options so token IDs stay aligned with text handling: keep raw completion IDs (including EOS/stop tokens) for token_ids while returning detokenized text with special tokens skipped for conversation continuation, matching what datasets.py end-token logic expects.



Document that cross-architecture distillation is valid only when teacher and student token IDs have identical semantics (normally the exact same tokenizer/vocabulary); architecture similarity alone is insufficient.



Mirror Tokasaurus retry/timeout behavior (growing timeouts, exponential backoff, on_failure="continue" returning empty samples so a synthesis run survives transient failures), support stop sequences and frequency penalty through native sampling parameters, and treat modal_upstream_id as an ignored compatibility argument because SGLang manages radix-cache reuse itself.

Downstream pipeline gaps for GLM and Kimi

The client alone is not enough for GLM/Kimi end-to-end; these per-model registries currently only know Qwen and Llama and must gain entries (or tokenizer-derived fallbacks) for each new family:





MODEL_TO_MESSAGE_CONVERTER in [cartridges-project/cartridges/datasets.py](cartridges-project/cartridges/datasets.py) — hard-coded per-role start/end token IDs; a wrong entry silently corrupts training targets, so derive them from the tokenizer's chat template and assert round-trip correctness in tests.



MODELS_WITH_THINKING and MODEL_TO_CHAT_TEMPLATE in [cartridges-project/cartridges/initialization/tokenization_utils.py](cartridges-project/cartridges/initialization/tokenization_utils.py).



MODEL_TO_TOOL_TEMPLATE / MODEL_TO_TOOL_CALL_PARSER in [cartridges-project/cartridges/data/__init__.py](cartridges-project/cartridges/data/__init__.py) already default to Qwen/Hermes-style handling, which is acceptable for tool-free synthesis but should be verified before enabling use_tools_a/b with GLM or Kimi.



MODEL_TO_THINKING_OVERRIDES in [cartridges-project/cartridges/utils/thinking.py](cartridges-project/cartridges/utils/thinking.py) — superseded by the client capability profile for the SGLang path, but keep entries consistent.

Correctness and tests





Add [cartridges-project/cartridges/tests/clients/test_sglang.py](cartridges-project/cartridges/tests/clients/test_sglang.py) with mocked GLM, Kimi, and Qwen profiles covering template rendering, toggleable/always-thinking behavior, batching, usage aggregation, exact [T,K] token-ID/logprob matrices, malformed or misaligned responses, retries, and no-logprob bot-A calls.



Add parameterized, opt-in live-server tests using SGLANG_URL and SGLANG_MODEL_FAMILY={glm,kimi,qwen}. Each performs the same top_logprobs=20 teacher request used by [cartridges-project/cartridges/synthesizers/self_study.py](cartridges-project/cartridges/synthesizers/self_study.py) and checks output-ID/top-k alignment without requiring every model in normal CI.



Harden [cartridges-project/cartridges/clients/base.py](cartridges-project/cartridges/clients/base.py) and [cartridges-project/cartridges/tests/clients/test_base.py](cartridges-project/cartridges/tests/clients/test_base.py): when the returned top-k mass never reaches min_prob_mass, retain all K entries instead of the current argmax(False...) behavior, which incorrectly retains only one. This matters more at temperature_b > 0 and for FP8 MoE models whose top-20 mass can stay below 0.99.



Reconcile the reconstruct() fill value (-1000.0 in code, -np.inf asserted in test_base.py) while touching this file — the existing tests fail as written.

Documentation and verification





Replace the currently broken README import examples with the implemented client configuration in [cartridges-project/README.md](cartridges-project/README.md).



Extend [cartridges-project/runbook/sglang setup.md](cartridges-project/runbook/sglang%20setup.md) with GLM, Kimi, and Qwen launch/profile examples, their appropriate reasoning parser settings, and a shared /generate top-k smoke test showing the expected output_ids / output_top_logprobs schema. Note that the native /generate path bypasses the server reasoning parser, so raw think tags remain in the text — which is what token-aligned distillation needs.



Advise disabling speculative decoding (the runbook's optional EAGLE flags) for teacher rollouts until top-k logprob parity with the non-speculative path is verified on the deployed version.



Run focused client/base tests, then run one small SelfStudySynthesizer batch per available GLM, Kimi, and Qwen deployment and verify the serialized conversations contain real completion token IDs and nonnegative top-k vocabulary IDs.

