# Run Qwen3.6 with SGLang on Slurm GPUs

This runbook starts from a Slurm GPU allocation and launches
`Qwen/Qwen3.6-35B-A3B-FP8` with SGLang.

## Architecture

```text
Login node
  ├─ source code, pyproject.toml, uv.lock
  ├─ persistent uv wheel cache: /home/$USER/.cache/uv
  └─ persistent Hugging Face cache: /home/$USER/.cache/huggingface

Slurm allocation
  └─ compute node with GPUs
       └─ srun --pty bash
            └─ SGLang HTTP server
                 ├─ frontend on port 30000
                 ├─ TP worker 0
                 └─ TP worker 1
                      └─ NCCL communication across GPUs
```

The model weights are downloaded from Hugging Face into the shared cache
once. Every server start still loads the checkpoint into GPU memory.

## 1. Allocate GPUs

Use a non-scavenge partition for a long model startup. Scavenge jobs can be
preempted while loading model shards.

```bash
salloc -p gpu_devel \
  --account=pi_jss233 \
  --nodes=1 \
  --gpus=2 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=4:00:00 \
  --no-shell
```

Record the job ID returned by Slurm. It will be different every time; use
`<JOBID>` below instead of reusing an old ID.

Enter the allocation:

```bash
srun --jobid=<JOBID> --pty bash
```

The login node cannot access the GPUs. `nvidia-smi` should only be expected to
work after entering the compute-node shell.

## 2. Load the runtime modules

```bash
cd /home/$USER/alexm/cartridges-project

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.9.1

nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
echo "CUDA_HOME=${CUDA_HOME-}"
```

`CUDA/12.9.1` supplies `CUDA_HOME`, which SGLang's FP8/DeepGEMM startup
requires. If the cluster's CUDA module names change, discover them with:

```bash
module spider CUDA
```

## 3. Create the node-local Python environment

The project source and uv cache stay on `/home`; the virtual environment is
placed on node-local `/tmp`. This avoids installing hundreds of packages over
the shared filesystem.

```bash
export GPU_VENV=/tmp/${USER}-cartridges-venv
export UV_PROJECT_ENVIRONMENT=$GPU_VENV
export UV_CACHE_DIR=/home/$USER/.cache/uv
export UV_LINK_MODE=symlink

/home/aam244/.local/bin/uv sync \
  --no-install-project \
  --prerelease=allow

source "$GPU_VENV/bin/activate"
```

The scratch environment is temporary. Repeat `uv sync` for a new allocation,
but the persistent uv cache means the large wheels should not need to be
downloaded again. `--no-install-project` is sufficient for running SGLang; use
`uv sync` without that flag if the local `cartridges` package must also be
installed into the environment.

Verify the runtime before launching:

```bash
python -c '
import torch, sglang
print("SGLang:", sglang.__version__)
print("Torch:", torch.__version__)
print("GPUs:", torch.cuda.device_count())
print([torch.cuda.get_device_name(i)
       for i in range(torch.cuda.device_count())])
'
```

The expected B200 setup reports two `NVIDIA B200` devices.

## 4. Launch Qwen3.6 with tensor parallelism

For two homogeneous B200s:

```bash
export CUDA_VISIBLE_DEVICES=0,1

sglang serve \
  --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
  --tp-size 2 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

Keep this command running. The server does not open its HTTP port until it
has initialized NCCL and loaded all model shards.

The first launch downloads approximately 35 GB of FP8 weights. Later launches
reuse `/home/$USER/.cache/huggingface`, but loading the weights into GPU memory
still happens every time.

Optional speculative decoding can be added after the baseline server works:

```bash
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

Do not enable speculative decoding for teacher rollouts until the deployed
SGLang version has been checked for top-k logprob parity with the
non-speculative path. Exact per-token distributions matter more than decode
throughput for these runs.

### GLM and Kimi launch profiles

The reasoning and tool parsers are server concerns. They are useful for
OpenAI-compatible requests, but do not change the client capability profile:

```bash
# GLM-4.5 / GLM-4.6 (GLM-4.7 uses --tool-call-parser glm47)
sglang serve \
  --model-path zai-org/GLM-4.6-FP8 \
  --tp-size 8 \
  --reasoning-parser glm45 \
  --tool-call-parser glm45 \
  --host 0.0.0.0 \
  --port 30000

# Kimi K2 Thinking
sglang serve \
  --model-path moonshotai/Kimi-K2-Thinking \
  --tp-size 8 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --host 0.0.0.0 \
  --port 30000
```

Qwen3/Qwen3.5/Qwen3.6 uses `--reasoning-parser qwen3`; choose the
checkpoint-specific tool parser supported by the installed SGLang release.
Tool-free synthesis does not require a tool parser.

## 5. Test the server

Run these commands from the compute-node shell:

```bash
curl http://127.0.0.1:30000/health
```

Example chat request:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "Hello from Qwen3.6."}],
    "max_tokens": 128
  }'
```

### Verify native token-aligned top-k output

Teacher rollouts use `/generate`, which bypasses the OpenAI reasoning parser.
Raw think delimiters therefore remain in `text`; this is intentional because
the output must stay aligned to the generated token IDs.

```bash
python - <<'PY'
import requests

url = "http://127.0.0.1:30000/generate"
payload = {
    "text": "Reply with one short word:",
    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
    "return_logprob": True,
    "top_logprobs_num": 20,
    "return_text_in_logprobs": False,
}
response = requests.post(url, json=payload, timeout=120)
response.raise_for_status()
data = response.json()
meta = data["meta_info"]

chosen_ids = [row[1] for row in meta["output_token_logprobs"]]
assert data["output_ids"][-meta["completion_tokens"]:] == chosen_ids
assert len(chosen_ids) == meta["completion_tokens"]
assert len(meta["output_top_logprobs"]) == len(chosen_ids)
assert all(len(row) == 20 for row in meta["output_top_logprobs"])
assert all(candidate[1] >= 0 for row in meta["output_top_logprobs"]
           for candidate in row)
print({
    "text": data["text"],
    "output_ids": data["output_ids"],
    "topk_shape": [len(chosen_ids), 20],
})
PY
```

If this fails because `output_top_logprobs` is absent or has fewer than 20
entries per token, do not run synthesis. Some SGLang builds have silently
ignored `top_logprobs_num`.

### Configure the Cartridges client

```python
from cartridges.clients.sglang import SGLangClient

qwen = SGLangClient.Config(
    model_name="Qwen/Qwen3.6-35B-A3B-FP8",
    url="http://127.0.0.1:30000",
    thinking_mode="toggleable",
    thinking_template_key="enable_thinking",
)

glm = SGLangClient.Config(
    model_name="zai-org/GLM-4.6-FP8",
    url="http://127.0.0.1:30000",
    trust_remote_code=True,
    thinking_mode="toggleable",
    thinking_template_key="enable_thinking",
)

kimi_thinking = SGLangClient.Config(
    model_name="moonshotai/Kimi-K2-Thinking",
    url="http://127.0.0.1:30000",
    trust_remote_code=True,
    thinking_mode="always",
)
```

For a Kimi instruct checkpoint, use `thinking_mode="unsupported"`. Older GLM
templates may expose the inverse `add_nothink_token` switch. For those
checkpoints, supply a `custom_chat_template` that exposes a positive
`enable_thinking` variable; the client intentionally does not infer or invert
behavior from the model name.

The server-reported tokenizer is loaded locally to render chats. Teacher and
student can have different architectures, but token-level distillation is valid
only when their token IDs have identical semantics—normally the exact same
tokenizer and vocabulary.

## GPU selection rules

### Two identical GPUs

Use tensor parallelism:

```bash
--tp-size 2
```

One request uses both GPUs. This is the normal setup for two B200s.

### One GPU

Expose one GPU and use:

```bash
--tp-size 1
```

Qwen3.6-35B-A3B-FP8 fits on a single H200 or B200.

### Different GPU models

Do not assume tensor parallelism is safe or efficient across different GPU
architectures, such as one B200 and one H200. Kernel support, memory size,
and performance may differ, and the slower GPU becomes the bottleneck.

Safer choices are:

1. Use only one GPU with `--tp-size 1`; or
2. Run one full model replica per GPU and route requests with data
   parallelism/Model Gateway.

If Slurm sets `CUDA_VISIBLE_DEVICES`, inspect it instead of blindly replacing
it with `0,1`. The `--tp-size` value must match the number of visible GPUs
assigned to the server.

## Common problems

- `nvidia-smi` fails on the login node: enter the allocation with `srun`.
- `CUDA_HOME` assertion: load `CUDA/12.9.1` before importing SGLang.
- `ModuleNotFoundError` during `uv sync`: wait for uv to finish with exit code
  zero before testing imports.
- Hugging Face unauthenticated warning: the Qwen checkpoint is public; an
  `HF_TOKEN` can improve rate limits.
- Long model startup: weights are being read from the shared Hugging Face
  cache. This is normal but slower than a node-local model copy.
- TorchCodec/FFmpeg warnings: they are normally ignored for text-only serving.
  Install/load FFmpeg if image or video inputs are required.
- Server killed while loading: the Slurm partition was likely preemptible.
  Use a stable GPU partition for long startup and serving sessions.

When finished, press `Ctrl-C` to stop SGLang, then exit the compute shell.
The Slurm allocation ends when its allocation shell is exited or canceled.
