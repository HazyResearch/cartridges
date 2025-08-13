<div align="center">
    <img src="assets/banner.png" height=100 alt="Cartridges logo"/>

**Storing long contexts in tiny KV caches with self-study.**



<!-- ![GitHub Workflow Status](https://github.com/HazyResearch/meerkat/actions/workflows/.github/workflows/ci.yml/badge.svg) -->
[![GitHub](https://img.shields.io/github/license/HazyResearch/cartridges)](https://img.shields.io/github/license/HazyResearch/cartridges)
[![arXiv](https://img.shields.io/badge/arXiv-2402.18668-b31b1b.svg)](https://arxiv.org/abs/2506.06266)

</div>


**What is this?** This repository provides code for training a cartridge, a small KV cache that represents a large corpus of textual information. It uses a test-time training recipe called self-study.
The code is based on our paper *[Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266)*.

**tl;dr** When we put lots of text (*e.g.* a whole code repo) into a language model's context, generation cost soars because of the KV cache's size. *What if we trained a smaller KV cache for our documents offline?* Using a test-time training recipe called self-study, we show that this simple idea can improve throughput by 
26× while maintaining quality. (See our [blogpost](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges) for more.)


**Table of contents**
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Running Self-Study](#running-self-study)
  - [Step 1: Synthesize training data](#step-1-synthesize-training-data)
    - [Step 1.1: Configure Resources](#step-11-configure-resources)
    - [Step 1.2: Prepare an Inference Server](#step-12-prepare-an-inference-server)
    - [Step 1.3: Run the Synthesis](#step-13-run-the-synthesis)
  - [Step 2: Run context-distillation (i.e. training) on the synthesized data](#step-2-run-context-distillation-ie-training-on-the-synthesized-data)
    - [Step 2.1: Evaluation](#step-21-evaluation)
- [Serving Cartridges](#serving-cartridges)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments and Citation](#acknowledgments-and-citation)


## Setup

**Step 1:** Clone the repository and install the Python package.

```bash
git clone https://github.com/HazyResearch/cartridges && cd cartridges
pip install uv
uv pip install -e . 
```

**Step 2:** Set some environment variables

The codebase relies on your setting the following variables. We recommend adding them to your `~/.bashrc`, `~/.zshrc`, `DockerFile`, etc.

```bash
# path to your the directory where you cloned this repo
export CARTRIDGES_DIR=/path/to/cartridges

# path to a directory where you want to store outputs like models checkpoints and such
export CARTRIDGES_OUTPUT_DIR=/path/to/cartridges/outputs

# the code in this repository is tightly integrated with wandb
# set your wandb project and entity here
export CARTRIDGES_WANDB_PROJECT=cartridges
export CARTRIDGES_WANDB_ENTITY=
```


## Running Self-Study

**What is self-study?** Self-study is an approach for training a model to understand a corpus of text. It works by generating synthetic conversations about a corpus of text and then training the model on those conversations with a context-distillation objective. The process consists of two AI agents in conversation with one another: one asks questions or makes requests about the content, and another responds using the provided context. 

**Quickstart**: Take a look at the scripts at `examples/arxiv/arxiv_synthesize.py` and `examples/arxiv/arxiv_train.py` for a basic example of how to synthesize training data and run context-distillation on the synthesized data. To run the synthesis script, you will need to spin up an inference server (either [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) or [SGLang](https://github.com/sgl-project/sglang)) and set the `client` variable to point to it. [See below for more details on how to do this.](#step-3-prepare-an-inference-server)

Below we walk through the process of generating synthetic training data for a corpus of text. As a running example, we'll be training a cartridge on our [paper on Cartridges](https://arxiv.org/abs/2506.06266). How meta!
<!-- Here are the steps:
1. Synth
1. Configure resources that contain the data you want to store in the cartridge
2. Ensure you have an inference server running (either [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) or [SGLang](https://github.com/ScalingIntelligence/tokasaurus)) and configure your client to point to it
3. Instantiate a `SynthesizeConfig` object that contains the parameters for the self-study process
4. Put it all together in one script and run it!
5. Run context-distillation (i.e. training) on the synthesized data -->

> **Note:** For configuration, we use [Pydantic](https://docs.pydantic.dev/latest/) models. Pydantic models are useful for defining the schema of the config and quickly ensuring that the config is valid at the beginning of a run. We also rely on [`pydrantic`](https://github.com/seyuboglu/pydrantic), which provides a few utilities for working with configs.


### Step 1: Synthesize training data
*Note: See `examples/arxiv/arxiv_synthesize.py` for the full example developed in this section.*

Below is the outline of a script for running the synthesis. It simply instantiates a [`SynthesizeConfig`](./cartridges/synthesize.py#L10) object and runs it with `pydrantic.main([config])`. *Note: Using `pydrantic.main` allow us to override the config on the command line like `python your_synthesis_script.py num_samples=1024`.*

The config has a couple of key fields missing: the resource, which controls what data is used , and a client of an inference server (*e.g.* SGLang or Tokasaurus). We'll cover those two below. 
There are many other configuration options we're not covering here, so refer to the [`SynthesizeConfig`](./cartridges/synthesize.py#L10) and [`SelfStudySynthesizer`](./cartridges/synthesizers/self_study.py#L10) for the full list.

```python
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer

resource_config = ...  # see 'Step 1.1: Configure Resources'
client_config = ...  # see 'Step 1.2: Prepare an Inference Server'

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client_config,
        resources=[resource_config],
    ),
    num_samples=512,
    name="cartridges-tutorial",
)

if __name__ == "__main__": 
    # library that allows us to override the Pydantic configs from the command line
    import pydrantic  
    pydrantic.main([config])
```


#### Step 1.1: Configure Resources
A "resource" is an object that feeds chunks of the context and a "seed prompt" to a synthesizer.  See Section 4 of [our paper](https://arxiv.org/pdf/2506.06266) for more details.

Since we want to train Cartridge for a research paper, we'll use the [`TextFileResource`](./cartridges/data/resources.py) type.

```python 
from cartridges.data.resources import TextFileResource

resource_config = TextFileResource.Config(
    path= "examples/arxiv/cartridges.tex",
    seed_prompts=["structuring", "summarization", "question"],
    chunker=TokenChunker.Config(
        tokenizer=client.model_name,
        min_tokens_per_chunk=512,
        max_tokens_per_chunk=1024,
    ),
)
```

We provide several other basic resource types for common data formats like `JSONResource`. 

We're also gradually adding more specialized resource types like that do a better job chunking specific data formats and feeding relevant seed prompts:
- [`LaTeXResource`](./cartridges/data/tex/resources.py) for training a Cartridge on a LaTeX project. In fact, we could have used this instead of the `TextFileResource` above: `LaTeXResource.Config(arxiv_id="2506.06266", ...)`
- [`SlackResource`](./cartridges/data/slack/resources.py) for training a Cartridge on Slack messages through the Slack API. This uses the Slack API to fetch recent messages from your channels. 
- [`GMailResource`](./cartridges/data/gmail/resources.py) for Gmail messages. This uses an MCP server to fetch recent messages from your inbox.


#### Step 1.2: Prepare an Inference Server

Self-study requires an inference server to generate the synthetic conversations. We need to configure a [`Client`](./cartridges/clients/base.py#L10) object that points to the inference server. We support two options:
- [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) (recommended) - We ran all of our experiments with Tokasaurus, which provides higher throughput generation and is easier to modify. 
- [SGLang](https://github.com/sgl-project/sglang) - We're also providing support for SGLang, but we have not tested it extensively.

<details>
<summary>
Option A: Modal Deployment (Tokasaurus)
</summary>

We found it easy to run data generation with Modal's serverless horizontal scaling.

For cloud deployment, you can deploy on Modal:
```bash
modal deploy infra/modal_deploy_tksrs.py
```

Then configure with the modal URL:
```python
from cartridges.clients.tokasaurus import TokasaurusClient

client_config = TokasaurusClient.Config(
    url="https://your-modal-deployment-url.modal.run",
    model_name="Qwen/Qwen3-4b"
)
```

> **Note:** Make sure to tune the `ALLOW_CONCURRENT_INPUTS` (which controls the number of concurrent requests each container can handle) parameter in the `deploy` modal script in conjunction with the `batch_size` configuration field. Each batch corresponds to a single request so if `batch_size=32` and `ALLOW_CONCURRENT_INPUTS=8`, then each container will be processing 256 conversations in parallel. As a (very rough) rule of thumb, you want to aim to have about 128 - 256 conversations running concurrently per container to maximize utilization. The exact number will depend on the amount of prefix-sharing etc. 
> Also, make sure to tune the `MIN_CONTAINERS` and `MAX_CONTAINER` parameters in the `deploy` modal script in conjunction with the `max_num_batches_in_parallel` configuration field. This will control how much the service will scale out horizontally. The more batches in parallel, the more containers will be spun up -- up to the `MAX_CONTAINER` limit.
---
</details>


<details>
<summary>
Option B: Local deployment (Tokasaurus)
</summary>

If you have access to GPUs, you can run also run a local Tokasaurus server:

1. Clone and install Tokasaurus:
```bash
git clone https://github.com/ScalingIntelligence/tokasaurus
cd tokasaurus
uv pip install -e .
```

2. Start the server:
```bash
tksrs model=Qwen3/Qwen3-4b kv_cache_num_tokens='(512 * 1024)' max_top_logprobs=20 dp_size=1
```


3. Configure your client:
```python
from cartridges.clients.tokasaurus import TokasaurusClient

client_config = TokasaurusClient.Config(
    url="http://localhost:8001",  # Make sure to use the port from the output of tksrs
    model_name="Qwen/Qwen3-4b" 
)
```

> **Note:** If you want to use data parallel, update the `dp_size` parameter and make sure you set the `max_num_batches_in_parallel` configuration field to aim for about 128 - 256 conversations running concurrently per data-parallel worker. 
</details>

<details>
<summary>
Option C: Modal deployment (SGLang)
</summary>

We found it easiest to run data generation with Modal because it provides serverless horizontal scaling.

For cloud deployment, you can deploy on Modal:
```bash
modal deploy infra/modal_deploy_sglang.py
```

Then configure with the modal URL:
```python
from cartridges.clients.sglang import SGLangClient

client_config = SGLangClient.Config(
    url="https://your-modal-deployment-url.modal.run",
    model_name="Qwen/Qwen3-4b"
)
```
</details>

<details>
<summary>
Option D: Local deployment (SGLang)
</summary>

1. Install and launch a SGLang server following the instructions [here](https://docs.sglang.ai/start/install.html).
2. Configure your client:
```python
from cartridges.clients.sglang import SGLangClient

client_config = SGLangClient.Config(
    model_name="Qwen/Qwen3-4b",
    url="http://localhost:8000",
)
```
</details>


#### Step 1.3: Run the Synthesis

Once you've created the script, run it with: 
```bash
python examples/arxiv/arxiv_synthesize.py
```

You can update the config on the command line like `python examples/arxiv/arxiv_synthesize.py num_samples=1024`.

Once the run is complete, it will save the results to a pickle file and print the path:
```bash
>>> Final output saved to /path/to/output/dir/artifact/dataset.parquet
```
Copy this path to your clipboard. See [`TrainingExample`](./cartridges/structs.py#L10) for the schema of the output.

<details>
<summary>
💻 <i>Explore synthesized dataset in the visualization UI!</i>
</summary>

We (read: Claude Code) implemented a nice visualization tool for exploring the synthesized dataset.
To run it, follow the instruction in [`viz/README.md`](./viz/README.md).

<img src="assets/viz.png" alt="Visualization" width="400"/>

</details>

### Step 2: Run context-distillation (i.e. training) on the synthesized data

*Note: See `examples/arxiv/arxiv_train.py` for the full script developed in this section.*


See [`TrainConfig`](./cartridges/train.py#L66) for documentation on the configuration. Below we provide a simple example of a config file.




**Make sure to set the `DATA_SOURCE` variable to the path to the synthesized dataset from above.**

```python
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.train import TrainConfig, CartridgeTrainDataset, DataSource
from cartridges.initialization.random import KVFromRandomText

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromRandomText.Config(max_tokens=2048),
    
    lr=2e-2,
    epochs=1,
    global_batch_size=32,

    dataset=CartridgeTrainDataset.Config(
        data_sources=[DataSource(path="path/to/your/dataset.parquet", type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    name="cartridges-tutorial-train",
)

if __name__ == "__main__":
    import pydrantic
    pydrantic.main(config)
```


**Data parallel training.** To launch a data parallel training run, use `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 path/to/file.py
```

> **Note:** We're occassionally seeing a NCCL collective operation timeout when running 
> data parallel training. If you encounter this error, you can set `distributed_backend="gloo"`
> while we debug the issue.

**KV Cache Initialization.** The `kv_cache_initializer` controls how the cartridge's KV cache is initialized before training. In [`cartridges/initialization`](./cartridges/initialization), we provide several different options including `KVFromRandomText`, `KVFromRandomVectors`, and `KVCacheFromPretrained`. `KVFromRandomText` is usually the best choice. The `max_tokens` parameter determines the size of the cartridge ($p$ in the paper). Larger values improve performance, but increase memory usage and training time.

> **Note:** We've also experimented with more sophisticated initialization strategies based off of summarization and other KV cache compression techniques. We will eventually add support for these to this codebase.

#### Step 2.1: Evaluation

During training, we can periodically evaluate the Cartridge on a held-out dataset. We support two types of evaluation: loss and generation-based evaluations. The metrics will be logged to wandb using the name in `name_for_wandb` in the config.

**Loss Evaluations.** These measure the perplexity of the model with cartridge on held-out ground truth responses. Lower perplexity indicates the cartridge is helping the model better predict the expected outputs.

For example, in `examples/arxiv/arxiv_synthesize_eval.py`, we synthesize high-quality conversations about the paper with GPT-5-mini to create ground truth responses. During training, we measure how well the model with cartridge can predict these ground truth tokens.

```python 
config = TrainConfig(
    ...
    loss_eval_every_n_steps=100,
    loss_evals=[
        LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(path="path/to/eval/dataset.parquet", type="local"),
                packed_seq_length=4096,
            ),
            name_for_wandb="loss_eval",
        )
    ],
    ...
)
```

**Generation Evaluations.** These evaluate the quality of text generated by the model with cartridge by sampling responses to prompts. The generated responses will be logged to WandB as a [table](https://docs.wandb.ai/guides/models/tables/) for manual inspection. 

```python 
from cartridges.train import GenerationEvalConfig
from cartridges.datasets import GenerateEvalDataset

config = TrainConfig(
    ...
    generate_eval_every_n_steps=100,
    generate_evals=[
        GenerationEvalConfig(
            dataset=GenerateEvalDataset.Config(
                data_source=DataSource(path="path/to/eval/dataset.parquet", type="local"),
                max_samples=50,  # Limit for faster evaluation
            ),
            num_samples=2,  # Generate multiple responses per prompt
            temperature=0.7,  # Add randomness for diverse outputs
            name_for_wandb="generation_eval",
        )
    ]
    ...
)
```

The generated responses are logged as WandB tables where you can manually review output quality with the cartridge.

You can also subclass `GenerateEvalDataset` to create more complex evaluation datasets with custom `score` functions. For example, we have a `LongHealthMultipleChoiceGenerateDataset` that evaluates the model's ability to answer multiple-choice questions about the patient's health history.


## Serving Cartridges

We describe two ways to serve and chat with a trained Cartridge: a simple, but slow way that just uses a pure PyTorch generation loop, and a faster one that uses a Tokasaurus server.

### Chatting with a Cartridge locally

Use the interactive CLI to chat with your trained cartridge:

```bash
python -m cartridges.utils.chat <wandb_run_id>
```

> **Note**: We currently only support downloading cartridges from wandb, but should eventually get this working for local cartridges as well.

**Finding your run ID**: You can find the run ID in your output. Look for lines like:
```
wandb: Run data is saved locally in /tmp/wandb/run-20241201_123456-abc1def2
wandb: Synced 5 files to https://wandb.ai/your-entity/cartridges/runs/abc1def2
```
The run ID is in the full path format: `your-entity/your-project/abc1def2`
You can also find the run ID in the "Overview" tab of the WandB UI under "Run path". 


### Serving with Tokasuaurus [Fastest and recommended]
We've implemented (h/t @geoffreyangus) an integration with [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus), a simple LLM inference server optimized for high throughput. 

To run the Tokasaurus server, you will need to (install Tokasaurus from source)[], switch to the branch `geoff/cartridges`, and then start up the server:

```bash
tksrs model=Qwen/Qwen3-4b kv_cache_num_tokens='(512 * 1024)'
```

Once you have a Tokasaurus server running, you can make requests using an OpenAI-like API with an extra field for Cartridges. Don't forget to make sure that the port matches the outputs of `tksrs`.

```python 
import requests

# Make a request with a cartridge from HuggingFace
response = requests.post("http://localhost:10210/v1/cartridge/chat/completions", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Help me understand this."}],
    "max_tokens": 50,
    "cartridges": [{
        "id": "hazyresearch/cartridge-wauoq23f",
        "source": "wandb",
        "force_redownload": False
    }]
})
```

Tokasaurus can also pull Cartridges from HuggingFace and local files. You can also compose multiple cartridges in a single request. See the [Tokasaurus documentation](https://github.com/ScalingIntelligence/tokasaurus/tree/geoff/cartridges?tab=readme-ov-file#cartridges) for the full instructions.


## Acknowledgments and Citation
There are tons of people and organizations who have supported this project. Below we shout out a few, but check out the the paper for a full list.

The compute for this project was provided by [Modal](https://modal.com/) — who made it super easy to scale out horizontally when running the synthetic data generation for self-study — and [Together](https://www.together.ai/) — who provided the compute for training the Cartridges on the synthetic data. [Prime Intellect](https://www.google.com/search?q=prime+intellect&oq=prime+intell&sourceid=chrome&ie=UTF-8&sei=dkNPaKfxNeq50PEPrdqiwA4), [Voltage Park](https://dashboard.voltagepark.com/), and [Azure](https://azure.microsoft.com/en-us/) through the HAI Grants program also contributed compute towards this project.


```bibtex
@article{eyuboglu2025cartridges,
  title={Cartridges: Lightweight and general-purpose long context representations via self-study},
  author={Eyuboglu, Sabri and Ehrlich, Ryan and Arora, Simran and Guha, Neel and Zinsley, Dylan and Liu, Emily and Tennien, Will and Rudra, Atri and Zou, James and Mirhoseini, Azalia and others},
  journal={arXiv preprint arXiv:2506.06266},
  year={2025}
}
```

