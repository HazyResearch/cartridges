import json
import os
from pathlib import Path
import tempfile
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import concurrent.futures
from matplotlib.figure import Figure
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb
from pydrantic import BaseConfig
from pydantic import Field

import torch.distributed as dist


if TYPE_CHECKING:
    from cartridges.structs import Conversation
    from cartridges.train import CacheAndModel


class WandBConfig(BaseConfig):
    project: str = os.environ.get("CARTRIDGES_WANDB_PROJECT", "cartridges")
    entity: Optional[str] = os.environ.get("CARTRIDGES_WANDB_ENTITY", None)
    name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    group: Optional[str] = None


def prepare_wandb(
    config: WandBConfig,
    config_dict: Dict[str, Any],
):
    wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.name,
        tags=config.tags,
        notes=config.notes,
        group=config.group,
        config=config_dict,
    )


def flatten(data: dict, parent_key: str = None, sep: str = "."):
    """
    Flatten a multi-level nested collection of dictionaries and lists into a flat dictionary.

    The function traverses nested dictionaries and lists and constructs keys in the resulting
    flat dictionary by concatenating nested keys and/or indices separated by a specified separator.

    Parameters:
    - data (dict or list): The multi-level nested collection to be flattened.
    - parent_key (str, optional): Used in the recursive call to keep track of the current key
                                  hierarchy. Defaults to an empty string.
    - sep (str, optional): The separator used between concatenated keys. Defaults to '.'.

    Returns:
    - dict: A flat dictionary representation of the input collection.

    Example:

    >>> nested_data = {
    ...    "a": 1,
    ...    "b": {
    ...        "c": 2,
    ...        "d": {
    ...            "e": 3
    ...        }
    ...    },
    ...    "f": [4, 5]
    ... }
    >>> flatten(nested_data)
    {'a': 1, 'b.c': 2, 'b.d.e': 3, 'f.0': 4, 'f.1': 5}
    """
    items = {}
    if isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key is not None else str(i)
            items.update(flatten(v, new_key, sep=sep))
    elif isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key is not None else k
            items.update(flatten(v, new_key, sep=sep))
    else:
        items[parent_key] = data
    return items


def unflatten(d: dict) -> dict:
    """
    Takes a flat dictionary with '/' separated keys, and returns it as a nested dictionary.

    Parameters:
    d (dict): The flat dictionary to be unflattened.

    Returns:
    dict: The unflattened, nested dictionary.
    """
    import numpy as np

    result = {}

    for key, value in d.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]

        if isinstance(value, (np.float64, np.float32, float)) and np.isnan(value):
            # need to check if value is nan, because wandb will create a column for every
            # possible value of a categorical variable, even if it's not present in the data
            continue

        d[parts[-1]] = value

    # check if any dicts have contiguous numeric keys, which should be converted to list
    def convert_to_list(d):
        if isinstance(d, dict):
            try:
                keys = [int(k) for k in d.keys()]
                keys.sort()
                if keys == list(range(min(keys), max(keys) + 1)):
                    return [d[str(k)] for k in keys]
            except ValueError:
                pass
            return {k: convert_to_list(v) for k, v in d.items()}
        return d

    return convert_to_list(result)


def fetch_wandb_runs(
    project_name: str = "hazy-research/cartridges",
    filters: dict = None,
    wandb_run_ids: List[str] = None,
    step: Optional[int] = None,
    return_steps: bool = False,
    step_keys: List[str] = None,
    **kwargs,
) -> Tuple[pd.DataFrame]:
    """
    Fetches run data from a W&B project into a pandas DataFrame.

    Parameters:
    - project_name (str): The name of the W&B project.
    - run_paths (List[str]): A list of runs 
    
    Returns:
    - DataFrame: A pandas DataFrame containing the run data.
    """
    # Initialize an API client
    api = wandb.Api()

    filters = {} if filters is None else filters
    for k, v in kwargs.items():
        if isinstance(v, List):
            filters[f"config.{k}"] = {"$in": v}
        else:
            filters[f"config.{k}"] = v
    
    if wandb_run_ids is not None:
        # allow people to pass the full run path (e.g. hazy-research/Cartridges/rfdhxjn6)
        # or just the run id (e.g. rfdhxjn6)
        wandb_run_ids = [
            os.path.split(wandb_run_id)[-1]
            for wandb_run_id in wandb_run_ids
        ]
        filters["name"] = {"$in": wandb_run_ids}

    # Get all runs from the specified project (and entity, if provided)
    runs = api.runs(project_name, filters=filters)

    # Create a list to store run data
    run_data = []
    steps_data = []

    # Iterate through each run and extract relevant data
    for run in runs:
    
        if step is None:
            summary = {**run.summary}
        else:
            # todo: probably should be using scan_history
            hist_df = run.history(samples=100_000)
            hist_df = hist_df[hist_df["train/optimizer_step"] == step].dropna(axis=1)
            try:
                step_data = hist_df.iloc[-1]
            except IndexError:
                breakpoint()
            summary = {**run.summary, **step_data}
        data = {
            "wandb_run_id": run.id,
            "name": run.name,
            "project": run.project,
            "user": run.user.name,
            "state": run.state,
            **flatten(run.config),
            **flatten(summary),
        }
        run_data.append(data)

        if return_steps:
            steps_df = run.history(keys=step_keys)
            steps_df["run_id"] = run.id
            steps_data.append(steps_df)

    # Convert list of run data into a DataFrame
    df = pd.DataFrame(run_data)

    df = df.dropna(axis="columns", how="all")

    # can't be serialized
    if "_wandb" in df.columns:
        df = df.drop(columns=["_wandb"])
    if "val_preds" in df.columns:
        df = df.drop(columns=["val_preds"])

    if return_steps:
        steps_df = pd.concat(steps_data)
        return df, steps_df

    return df, None

def read_conversations_from_wandb(
    artifact_id: str,
    versions: Literal["latest", "all"] = "latest",
    project_name: str = "cartridges",
    entity: str = "hazy-research",
) -> list["Conversation"]:
    """
    Read conversations from a Weights & Biases artifact containing conversation data.

    Parameters:
    - artifact_id (str): The ID of the artifact containing conversations. Can include version (e.g., "artifact:v1")
    - versions (Literal["latest", "all"]): Whether to fetch only the latest version or all versions. Default is "latest".
    - project_name (str): The name of the W&B project. Default is "cartridges".
    - entity (str): The W&B entity (username or team name). Default is "hazy-research".

    Returns:
    - list[Conversation]: A list of Conversation objects loaded from the artifact.
    """
    from cartridges.structs import read_conversations
    # Initialize wandb API
    api = wandb.Api()
    
    # Build full artifact path
    full_artifact_id = f"{entity}/{project_name}/{artifact_id}"
    
    if ":" not in full_artifact_id:
        # If no version is specified, get the latest version
        artifact = api.artifact(f"{full_artifact_id}:latest")
    else:
        artifact = api.artifact(full_artifact_id)
    
    # Download the artifact to a temporary directory
    artifact_dir = artifact.download(root=tempfile.mkdtemp())
    
    # Look for conversation files in the artifact directory
    conversations = []
    
    # Search for parquet or pkl files containing conversations
    for root, dirs, files in os.walk(artifact_dir):
        for file in files:
            file_path = Path(root) / file
            
            if file.endswith(".parquet") or file.endswith(".pkl"):
                try:
                    file_conversations = read_conversations(str(file_path))
                    conversations.extend(file_conversations)
                except Exception as e:
                    # Skip files that can't be read as conversations
                    continue
    
    if not conversations:
        raise ValueError(f"No conversation files found in artifact {artifact_id}")
    
    return conversations

def fetch_wandb_table(
    artifact_id: str,
    versions: Literal["latest", "all"] = "latest",
    project_name: str = "cartridges",
    entity: str = "hazy-research",
) -> pd.DataFrame:
    """
    Fetches a table from a Weights & Biases artifact.

    Parameters:
    - artifact_id (str): The ID of the artifact containing the table. This can be found in the
                         W&B UI by navigating to the run page, clicking on the "Artifacts" tab,
                         and looking at the artifact name (e.g., "run-g9cflk5x-generate_multiple_choice_generationstable").
                         The id can include the version number, e.g. "run-g9cflk5x-generate_multiple_choice_generationstable:v1"
                         in which case the version parameter is ignored.
    - version (Literal["latest", "all"]): Whether to fetch only the latest version or all versions of the artifact.
                                          Default is "latest".
    - project_name (str): The name of the W&B project. Default is "Cartridges".
    - entity (str): The W&B entity (username or team name). Default is "hazy-research".

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the table data.
    """

    # Define the artifact ID
    artifact_id = f"{entity}/{project_name}/{artifact_id}"

    if ":" not in artifact_id:
        # If no version is specified, list available versions
        api = wandb.Api()

        # List all versions of this artifact
        version_ids = list(api.artifact_versions("run_table", artifact_id))

        if versions == "latest":
            artifact_ids = [f"{artifact_id}:{version_ids[0].version}"]
        elif versions == "all":
            artifact_ids = [
                f"{artifact_id}:{version.version}" for version in version_ids
            ]
        else:
            raise ValueError(f"Invalid versions: {versions}")
    else:
        artifact_ids = [artifact_id]

    # Initialize wandb
    api = wandb.Api()

    def process_artifact(artifact_id, api):
        # Download the artifact
        artifact = api.artifact(artifact_id)
        artifact_dir = artifact.download(root=tempfile.mkdtemp())

        # Search recursively for the table.table.json file in the artifact directory
        table_path = None
        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                if file == "table.table.json":
                    table_path = os.path.join(root, file)
                    break
            if table_path:
                break

        if table_path is None:
            raise ValueError(
                f"Table not found in the artifact directory for {artifact_id}"
            )

        with open(table_path) as f:
            data = json.load(f)

        curr_df = pd.DataFrame(
            data["data"],
            columns=data["columns"],
        )
        curr_df["artifact_id"] = artifact_id
        return curr_df

    # Use ThreadPoolExecutor to process artifacts in parallel
    all_dfs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a partial function with the api already set
        process_func = partial(process_artifact, api=api)
        # Map the function to all artifact IDs and collect results
        results = list(executor.map(process_func, artifact_ids))
        all_dfs.extend(results)

    df = pd.concat(all_dfs)
    return df


def compare_runs(
    run_id_a: str,
    run_id_b: str,
):
    import tempfile
    import os
    import subprocess
    import shlex

    with tempfile.TemporaryDirectory() as tmp_dir:
        api = wandb.Api()
        code_dirs = []
        for run_id in [run_id_a, run_id_b]:
            run = api.run(run_id)

            code_artifacts = [x for x in run.logged_artifacts() if x.type == "code"]

            if len(code_artifacts) == 0:
                raise ValueError(f"No code artifacts found for run {run_id}")

            elif len(code_artifacts) > 1:
                print(f"Multiple code artifacts found for run {run_id}, taking first.")
            code = code_artifacts[0]

            # download to temporary directory
            code_dir = code.download(root=os.path.join(tmp_dir, run_id))
            code_dirs.append(code_dir)

        import subprocess
        import shlex

        command = f"diff -r -u --color=always {code_dirs[0]} {code_dirs[1]} | less -R"
        print(shlex.split(command))
        subprocess.call(command, shell=True)


def get_artifact_dir(artifact_name: str):
    return Path(os.environ.get("WANDB_CACHE_DIR", "./artifacts")) / artifact_name


def download_artifact(artifact_name: str):
    if artifact_name.endswith(".pkl"):
        return

    download_dir = get_artifact_dir(artifact_name)
    dataset_dir = download_dir / "dataset"

    # we only download the artifact on rank 0 so as to avoid race conditions
    run = wandb.run

    if run is not None:
        # Inside a run - use run.use_artifact
        artifact = run.use_artifact(artifact_name)
    else:
        # Outside a run - use the API
        api = wandb.Api()
        artifact = api.artifact(artifact_name)

    # download the artifact to the temporary directory
    artifact_dir = artifact.download(download_dir)
    assert Path(artifact_dir) / "dataset" == dataset_dir


def download_artifacts(artifacts: list[str]):
    is_ddp = "LOCAL_RANK" in os.environ
    is_rank_zero = not is_ddp or dist.get_rank() == 0

    # RE(3/20): this could be made faster, but who cares.
    if is_rank_zero:
        for artifact_name in artifacts:
            download_artifact(artifact_name)

    elif is_ddp:
        ...
        # logger.info("Waiting for download to finish on rank 0")

    if is_ddp:
        # wait for download to finish successfully on rank 0 before proceeding
        # to load the dataset
        dist.barrier()

def figure_to_wandb(fig: Figure) -> wandb.Image:
    import io
    import wandb
    from PIL import Image
    import matplotlib.pyplot as plt

    # Save figure to an in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Convert buffer to PIL Image
    pil_img = Image.open(buf)

    # Create a wandb Image from the PIL image
    wandb_img = wandb.Image(pil_img)
    plt.close(fig)
    return wandb_img

def _list_cache_files(run_id: str) -> list[str]:
    import wandb
    import re

    api = wandb.Api()

    # Get all files from the run
    files = [file.name for file in api.run(run_id).files()]

    # Filter for cache-*.pt files using regex
    cache_files = [file for file in files if re.match(r"^cache-.*\.pt$", file)]

    # Extract the epoch or step number from each cache file and create a mapping
    file_to_step = {}
    for file in cache_files:
        # Try to match both epoch and step patterns
        match = re.search(r"cache-(epoch|step)(\d+)\.pt", file)
        if match:
            step_num = int(match.group(2))
            file_to_step[file] = step_num

    # Sort the files by their step/epoch number
    sorted_cache_files = sorted(cache_files, key=lambda x: file_to_step.get(x, 0), reverse=True)
    return sorted_cache_files

def load_model_and_cache_from_wandb(
    wandb_run_id: str,
    filename: Optional[str] = None,
    device: str = "cuda",
) -> tuple["CacheAndModel", AutoTokenizer]:
    from cartridges.train import TrainConfig, CacheAndModel
    from cartridges.cache import TrainableCache

    is_ddp = "LOCAL_RANK" in os.environ
    is_rank_zero = (not is_ddp) or (dist.get_rank() == 0)

    train_config = TrainConfig.from_wandb(wandb_run_id, strict=False)

    model = train_config.model.instantiate().to(device)
    # tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer)


    cache_files = _list_cache_files(wandb_run_id)
    if len(cache_files) == 0:
        raise ValueError(f"No cache checkpoints found for wandb run {wandb_run_id}")
    
    if filename is not None:
        assert filename in cache_files, f"Cache file {filename} not found in wandb run {wandb_run_id}"
    else:
        filename = cache_files[0]
    print(f"Loading cache from {filename}")

    if is_rank_zero:
        out = wandb.restore(
            filename, run_path=wandb_run_id, root=train_config.run_dir
        )
    
    if is_ddp:
        dist.barrier()

    cache = TrainableCache.from_pretrained(
        os.path.join(train_config.run_dir, filename), 
        device=device
    )

    return CacheAndModel(cache=cache, model=model)

