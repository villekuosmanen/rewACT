#!/usr/bin/env python
"""
Modal-based RL Training Pipeline Factory for RewACT.

This script orchestrates the complete offline RL training loop:
1. Train RewACT value function
2. Precompute advantages using the trained value function
3. Train ACTvantage policy using the computed advantages

Usage:
    modal run scripts/modal_pipeline.py

Setup secrets first:
    modal secret create huggingface-secret HF_TOKEN=hf_...
    modal secret create wandb-secret WANDB_API_KEY=...
    modal secret create discord-webhook DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import modal

TRAIN_VERSION = "1.5.0"

# Define the Modal app
app = modal.App("rewact-training-pipeline")
volume = modal.Volume.from_name("rewact-datasets-cache", create_if_missing=True)

# Define the Docker image with all dependencies
image = (
    modal.Image.from_registry(
        # "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        "nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04",
        add_python="3.10",
    )
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y git libglib2.0-0",
        "apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip python3-dev git wget curl ffmpeg libsm6 libxext6 linux-headers-generic build-essential clang && rm -rf /var/lib/apt/lists/*",
    )
    # .apt_install(
    #     "python3-dev",
    #     "git",
    #     "wget",
    #     "curl",
    #     "ffmpeg",
    #     "libsm6",
    #     "libxext6",
    #     "linux-headers-generic",
    #     "build-essential",
    #     "libgl1-mesa-glx",
    #     "libglib2.0-0",
    # )
    .uv_pip_install(
        # Core dependencies
        "lerobot==0.4.2",
        "einops>=0.6.0",
        "scipy>=1.7.0",
        # Additional dependencies
        "wandb",
        "hydra-core>=1.2.0",
        "pandas",
        "pyarrow",
        "huggingface-hub",
        "termcolor",
        "tqdm",
        "requests",
        "ffmpeg",
    )
    .add_local_python_source("robocandywrapper", copy=True)
    .add_local_python_source("rewact")
)


def send_discord_notification(message: str, embed: Optional[Dict] = None):
    """Send a notification to Discord via webhook."""
    import requests
    
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logging.warning("Discord webhook URL not configured, skipping notification")
        return
    
    payload = {"username": "RewACT Pipeline"}
    
    if embed:
        payload["embeds"] = [embed]
    else:
        payload["content"] = message
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logging.warning(f"Failed to send Discord notification: {e}")


@app.function(
    image=image,
    gpu="H100",
    cpu=12.0,
    memory=65536, # 64GB memory
    timeout=86400,  # 24 hours
    volumes={"/root/.cache/huggingface": volume},  # Cache HF datasets
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("discord-webhook"),
    ],
    env={
        "MKL_SERVICE_FORCE_INTEL": "1",
        "MKL_THREADING_LAYER": "GNU",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    },
)
def train_value_function(
    dataset_repos: List[str],
    output_repo_id: str,
    job_name: str,
    steps: int = 50000,
    batch_size: int = 24,
    save_freq: int = 2000,
    sampler_config: Optional[Dict] = None,
    checkpoint_push_freq: Optional[int] = None,
    modal_run_id: Optional[str] = None,
    step_number: int = 1,
) -> Dict:
    """Train RewACT value function on Modal."""
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/rewact")
    
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.utils import init_logging
    from rewact.trainers import RewACTTrainer
    
    init_logging()
    
    # Send start notification
    modal_url = f"https://modal.com/apps/rewact-training-pipeline/{modal_run_id}" if modal_run_id else None
    embed = {
        "title": f"🚀 Step {step_number}: Training Value Function",
        "description": f"Starting RewACT value function training",
        "color": 0x3498db,  # Blue
        "fields": [
            {"name": "Datasets", "value": f"{len(dataset_repos)} datasets", "inline": True},
            {"name": "Output Repo", "value": f"`{output_repo_id}`", "inline": False},
            {"name": "Steps", "value": str(steps), "inline": True},
            {"name": "Batch Size", "value": str(batch_size), "inline": True},
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }
    if modal_url:
        embed["fields"].append({"name": "Modal Run", "value": f"[View]({modal_url})", "inline": False})
    
    send_discord_notification(None, embed)
    
    # Build config
    dataset_repo_str = "[" + ", ".join(dataset_repos) + "]"
    
    args = [
        f"--dataset.repo_id={dataset_repo_str}",
        "--dataset.image_transforms.enable=true",
        "--policy.type=rewact",
        f"--policy.repo_id={output_repo_id}",
        f"--output_dir=/tmp/train/{job_name}",
        f"--job_name={job_name}",
        f"--batch_size={batch_size}",
        "--eval_freq=-1",
        "--log_freq=20",
        f"--save_freq={save_freq}",
        f"--steps={steps}",
        "--policy.n_action_steps=50",
        "--policy.n_decoder_layers=4",
        "--wandb.enable=true",
        "--policy.push_to_hub=true",
    ]
    
    # Parse config
    cfg = draccus.parse(TrainPipelineConfig, args=args)
    
    # Create trainer with loaded sampler config
    trainer = RewACTTrainer(
        cfg,
        sampler_config=sampler_config,
        checkpoint_push_freq=checkpoint_push_freq,
    )
    
    # Train
    result = trainer.train()
    
    # Send completion notification
    embed = {
        "title": f"✅ Step {step_number} Complete: Value Function Trained",
        "description": f"RewACT value function training finished successfully",
        "color": 0x00ff00,  # Green
        "fields": [
            {"name": "Model Repo", "value": f"[{result['model_repo_id']}](https://huggingface.co/{result['model_repo_id']})", "inline": False},
            {"name": "Final Loss", "value": f"{result['final_loss']:.4f}", "inline": True},
            {"name": "Total Steps", "value": str(result['total_steps']), "inline": True},
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }
    if result.get('wandb_url'):
        embed["fields"].append({"name": "WandB Run", "value": f"[View]({result['wandb_url']})", "inline": False})
    
    send_discord_notification(None, embed)
    
    return result


@app.function(
    image=image,
    gpu="A10",  # Cheaper GPU for inference
    cpu=8.0, # 8 cores
    memory=8192, # 8GB memory - requires intense processing
    timeout=7200,  # 2 hours per dataset
    volumes={"/root/.cache/huggingface": volume},  # Cache HF datasets
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("discord-webhook"),
    ],
)
def compute_advantages(
    dataset_repo: str,
    value_model_repo: str,
    output_repo_id: str,
    n_step: int = 50,
    modal_run_id: Optional[str] = None,
    step_number: int = 2,
    dataset_index: int = 0,
    total_datasets: int = 1,
) -> Dict:
    """Compute advantages for one dataset on Modal."""
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/rewact")
    
    from lerobot.utils.utils import init_logging
    from rewact.trainers import AdvantageCalculator
    
    init_logging()
    
    # Send start notification
    embed = {
        "title": f"🧮 Step {step_number}.{dataset_index + 1}: Computing Advantages",
        "description": f"Processing dataset {dataset_index + 1}/{total_datasets}",
        "color": 0x3498db,  # Blue
        "fields": [
            {"name": "Dataset", "value": f"`{dataset_repo}`", "inline": False},
            {"name": "Value Model", "value": f"[{value_model_repo}](https://huggingface.co/{value_model_repo})", "inline": False},
            {"name": "N-step", "value": str(n_step), "inline": True},
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    send_discord_notification(None, embed)
    
    # Create calculator
    output_dir = f"/tmp/advantages/{dataset_repo.split('/')[-1]}"
    calculator = AdvantageCalculator(
        dataset_path=dataset_repo,
        value_model_path=value_model_repo,
        output_dir=output_dir,
        n_step_lookahead=n_step,
        device="cuda",
    )
    
    # Compute advantages
    result = calculator.compute()
    
    # Push to Hub
    hub_url = calculator.push_to_hub(
        repo_id=output_repo_id,
        commit_message=f"Add advantages computed from {value_model_repo}"
    )
    
    result["hub_url"] = hub_url
    result["advantage_repo_id"] = output_repo_id
    
    # Send completion notification
    embed = {
        "title": f"✅ Step {step_number}.{dataset_index + 1} Complete: Advantages Computed",
        "description": f"Dataset {dataset_index + 1}/{total_datasets} processed",
        "color": 0x00ff00,  # Green
        "fields": [
            {"name": "Dataset", "value": f"`{dataset_repo}`", "inline": False},
            {"name": "Advantage Repo", "value": f"[{output_repo_id}](https://huggingface.co/datasets/{output_repo_id})", "inline": False},
            {"name": "Frames Processed", "value": str(result['statistics']['num_frames']), "inline": True},
            {"name": "Mean Advantage", "value": f"{result['statistics']['mean']:.4f}", "inline": True},
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    send_discord_notification(None, embed)
    
    return result


@app.function(
    image=image,
    gpu="H100",
    cpu=12.0,
    memory=65536, # 64GB memory
    timeout=86400,  # 24 hours
    volumes={"/root/.cache/huggingface": volume},  # Cache HF datasets
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("discord-webhook"),
    ],
)
def train_actvantage_policy(
    dataset_repos: List[str],
    advantage_repos: Dict[str, str],
    output_repo_id: str,
    job_name: str,
    steps: int = 100000,
    batch_size: int = 24,
    save_freq: int = 2000,
    advantage_percentile: float = 30.0,
    sampler_config: Optional[Dict] = None,
    checkpoint_push_freq: Optional[int] = None,
    modal_run_id: Optional[str] = None,
    step_number: int = 3,
) -> Dict:
    """Train ACTvantage policy on Modal."""
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/rewact")
    
    from pathlib import Path
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.utils import init_logging
    from rewact.trainers import ACTvantageTrainer
    from huggingface_hub import snapshot_download
    
    init_logging()
    
    # Send start notification
    embed = {
        "title": f"🎯 Step {step_number}: Training ACTvantage Policy",
        "description": f"Starting advantage-conditioned policy training",
        "color": 0x3498db,  # Blue
        "fields": [
            {"name": "Datasets", "value": f"{len(dataset_repos)} datasets", "inline": True},
            {"name": "Output Repo", "value": f"`{output_repo_id}`", "inline": False},
            {"name": "Steps", "value": str(steps), "inline": True},
            {"name": "Batch Size", "value": str(batch_size), "inline": True},
            {"name": "Advantage Percentile", "value": f"{advantage_percentile}%", "inline": True},
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    send_discord_notification(None, embed)
    
    # Download advantage datasets from Hub
    advantage_dirs = {}
    for dataset_repo, advantage_repo in advantage_repos.items():
        local_dir = f"/tmp/advantages/{advantage_repo.split('/')[-1]}"
        logging.info(f"Downloading advantages from {advantage_repo}...")
        snapshot_download(
            repo_id=advantage_repo,
            repo_type="dataset",
            local_dir=local_dir,
        )
        advantage_dirs[dataset_repo] = Path(local_dir)
    
    # Build config
    dataset_repo_str = "[" + ", ".join(dataset_repos) + "]"
    
    args = [
        f"--dataset.repo_id={dataset_repo_str}",
        "--policy.type=actvantage",
        f"--policy.repo_id={output_repo_id}",
        f"--output_dir=/tmp/train/{job_name}",
        f"--job_name={job_name}",
        f"--batch_size={batch_size}",
        "--eval_freq=-1",
        "--log_freq=20",
        f"--save_freq={save_freq}",
        f"--steps={steps}",
        "--wandb.enable=true",
        "--policy.push_to_hub=true",
    ]
    
    # Parse config
    cfg = draccus.parse(TrainPipelineConfig, args=args)
    
    # Create trainer with loaded sampler config
    trainer = ACTvantageTrainer(
        cfg,
        advantage_dirs=advantage_dirs,
        sampler_config=sampler_config,
        advantage_percentile=advantage_percentile,
        checkpoint_push_freq=checkpoint_push_freq,
    )
    
    # Train
    result = trainer.train()
    
    # Send completion notification
    embed = {
        "title": f"🎉 Step {step_number} Complete: ACTvantage Policy Trained",
        "description": f"Policy training finished successfully",
        "color": 0x00ff00,  # Green
        "fields": [
            {"name": "Policy Repo", "value": f"[{result['model_repo_id']}](https://huggingface.co/{result['model_repo_id']})", "inline": False},
            {"name": "Final Loss", "value": f"{result['final_loss']:.4f}", "inline": True},
            {"name": "Total Steps", "value": str(result['total_steps']), "inline": True},
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }
    if result.get('wandb_url'):
        embed["fields"].append({"name": "WandB Run", "value": f"[View]({result['wandb_url']})", "inline": False})
    
    send_discord_notification(None, embed)
    
    return result


@app.function(
    image=image,
    timeout=86400,  # 24 hours
    nonpreemptible=True,
    secrets=[modal.Secret.from_name("discord-webhook")],
)
async def orchestrate_pipeline(
    dataset_repos: List[str],
    value_function_repo: str,
    actvantage_repo: str,
    value_job_name: str,
    policy_job_name: str,
    value_steps: int = 50000,
    policy_steps: int = 100000,
    batch_size: int = 24,
    n_step_advantage: int = 50,
    advantage_percentile: float = 30.0,
    skip_value_training: bool = False,
    skip_advantage_computation: bool = False,
    skip_policy_training: bool = False,
    checkpoint_push_freq: Optional[int] = None,
    rewact_sampler_config: Optional[Dict] = None,
    actvantage_sampler_config: Optional[Dict] = None,
) -> Dict:
    """
    Orchestrate the full RL training pipeline.
    
    This runs remotely on Modal (without GPU) to coordinate the pipeline steps.
    """
    import asyncio
    import sys
    sys.path.insert(0, "/root")
    
    results = {}
    
    # Send pipeline start notification
    embed = {
        "title": "🏭 RewACT Training Pipeline Started",
        "description": "Full offline RL training loop initiated",
        "color": 0x9b59b6,  # Purple
        "fields": [
            {"name": "Datasets", "value": "\n".join([f"• `{d}`" for d in dataset_repos]), "inline": False},
            {"name": "Pipeline Steps", "value": 
                f"{'~~' if skip_value_training else ''}1. Train Value Function{'~~' if skip_value_training else ''}\n"
                f"{'~~' if skip_advantage_computation else ''}2. Compute Advantages{'~~' if skip_advantage_computation else ''}\n"
                f"{'~~' if skip_policy_training else ''}3. Train Policy{'~~' if skip_policy_training else ''}",
                "inline": False
            },
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    send_discord_notification(None, embed)
    
    # Step 1: Train value function
    if not skip_value_training:
        logging.info("Step 1: Training value function...")
        value_result = await train_value_function.remote.aio(
            dataset_repos=dataset_repos,
            output_repo_id=value_function_repo,
            job_name=value_job_name,
            steps=value_steps,
            batch_size=batch_size,
            checkpoint_push_freq=checkpoint_push_freq,
            step_number=1,
            sampler_config=rewact_sampler_config,
        )
        results["value_function"] = value_result
        logging.info(f"Value function trained: {value_result['model_repo_id']}")
    else:
        logging.info("Skipping value function training (using existing model)")
        results["value_function"] = {"model_repo_id": value_function_repo, "skipped": True}
    
    # Step 2: Compute advantages (in parallel for all datasets)
    if not skip_advantage_computation:
        logging.info("Step 2: Computing advantages...")
        logging.info(f"Spawning {len(dataset_repos)} advantage computation jobs in parallel...")
        
        # Submit all jobs in parallel using async API
        async_tasks = []
        dataset_info = []
        for idx, dataset_repo in enumerate(dataset_repos):
            advantage_repo_id = f"{dataset_repo}-advantages"
            # Use .remote.aio() for async, non-blocking call
            task = compute_advantages.remote.aio(
                dataset_repo=dataset_repo,
                value_model_repo=results["value_function"]["model_repo_id"],
                output_repo_id=advantage_repo_id,
                n_step=n_step_advantage,
                step_number=2,
                dataset_index=idx,
                total_datasets=len(dataset_repos),
            )
            async_tasks.append(task)
            dataset_info.append((dataset_repo, advantage_repo_id))
        
        logging.info(f"All {len(dataset_repos)} jobs submitted. Waiting for completion in parallel...")
        
        # Wait for all jobs to complete in parallel using asyncio.gather
        all_results = await asyncio.gather(*async_tasks)
        
        # Build advantage repos mapping from results
        advantage_repos = {}
        for (dataset_repo, advantage_repo_id), result in zip(dataset_info, all_results):
            advantage_repos[dataset_repo] = result["advantage_repo_id"]
            logging.info(f"✓ Completed: {dataset_repo} -> {advantage_repo_id}")
        
        results["advantages"] = advantage_repos
        logging.info(f"✅ All {len(advantage_repos)} advantage computations completed")
    else:
        logging.info("Skipping advantage computation (using existing advantages)")
        # Assume advantages already exist with standard naming
        advantage_repos = {repo: f"{repo}-advantages" for repo in dataset_repos}
        results["advantages"] = advantage_repos
    
    # Step 3: Train ACTvantage policy
    if not skip_policy_training:
        logging.info("Step 3: Training ACTvantage policy...")
        policy_result = await train_actvantage_policy.remote.aio(
            dataset_repos=dataset_repos,
            advantage_repos=results["advantages"],
            output_repo_id=actvantage_repo,
            job_name=policy_job_name,
            steps=policy_steps,
            batch_size=batch_size,
            advantage_percentile=advantage_percentile,
            checkpoint_push_freq=checkpoint_push_freq,
            step_number=3,
            sampler_config=actvantage_sampler_config,
        )
        results["policy"] = policy_result
        logging.info(f"Policy trained: {policy_result['model_repo_id']}")
    else:
        logging.info("Skipping policy training")
        results["policy"] = {"skipped": True}
    
    # Send final summary notification
    embed = {
        "title": "🏆 Pipeline Complete!",
        "description": "All steps of the RewACT training pipeline finished successfully",
        "color": 0xf39c12,  # Gold
        "fields": [],
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if not skip_value_training:
        embed["fields"].append({
            "name": "✅ Value Function",
            "value": f"[{results['value_function']['model_repo_id']}](https://huggingface.co/{results['value_function']['model_repo_id']})",
            "inline": False
        })
    
    if not skip_advantage_computation:
        advantage_list = "\n".join([f"• [{repo}](https://huggingface.co/datasets/{repo})" for repo in results["advantages"].values()][:3])
        if len(results["advantages"]) > 3:
            advantage_list += f"\n• ... and {len(results['advantages']) - 3} more"
        embed["fields"].append({
            "name": "✅ Advantages Computed",
            "value": advantage_list,
            "inline": False
        })
    
    if not skip_policy_training:
        embed["fields"].append({
            "name": "✅ ACTvantage Policy",
            "value": f"[{results['policy']['model_repo_id']}](https://huggingface.co/{results['policy']['model_repo_id']})",
            "inline": False
        })
    
    send_discord_notification(None, embed)
    
    return results


@app.local_entrypoint()
def main(
    
    # Dataset configuration
    datasets: str = "villekuosmanen/build_block_tower, villekuosmanen/dAgger_build_block_tower_1.0.0, villekuosmanen/dAgger_build_block_tower_1.1.0, villekuosmanen/dAgger_build_block_tower_1.2.0, villekuosmanen/dAgger_build_block_tower_1.3.0, villekuosmanen/dAgger_build_block_tower_1.4.0",
    
    # Output configuration
    value_function_repo: str = f"villekuosmanen/rewact_build_block_tower_{TRAIN_VERSION}",
    actvantage_repo: str = f"villekuosmanen/actvantage_build_block_tower_{TRAIN_VERSION}",
    
    # Job names
    value_job_name: str = f"rewact_build_block_tower_{TRAIN_VERSION}",
    policy_job_name: str = f"actvantage_build_block_tower_{TRAIN_VERSION}",
    
    # Training hyperparameters
    value_steps: int = 35000,
    policy_steps: int = 35000,
    batch_size: int = 64,
    n_step_advantage: int = 50,
    advantage_percentile: float = 55.0,
    
    # Pipeline control
    skip_value_training: bool = True,
    skip_advantage_computation: bool = False,
    skip_policy_training: bool = True,
    
    # Checkpoint configuration
    checkpoint_push_freq: int = 2000,
):
    """
    Run the full RewACT training pipeline on Modal.
    
    This entrypoint runs locally but launches the orchestration on Modal,
    which then coordinates all the remote training steps.
    
    Example usage:
        modal run scripts/modal_pipeline.py
        
        # Skip value training (use existing model):
        modal run scripts/modal_pipeline.py --skip-value-training --value-function-repo villekuosmanen/existing-model
        
        # Push checkpoints every 10k steps:
        modal run scripts/modal_pipeline.py --checkpoint-push-freq 10000
    """
    import sys
    from robocandywrapper.samplers import load_sampler_config
    
    # Parse datasets
    dataset_repos = [d.strip() for d in datasets.split(",")]
    
    print("=" * 80)
    print("RewACT Training Pipeline Factory")
    print("=" * 80)
    print(f"\nDatasets ({len(dataset_repos)}):")
    for repo in dataset_repos:
        print(f"  • {repo}")
    print(f"\nPipeline Configuration:")
    print(f"  Value Function Repo: {value_function_repo}")
    print(f"  ACTvantage Repo: {actvantage_repo}")
    print(f"  Value Training Steps: {value_steps}")
    print(f"  Policy Training Steps: {policy_steps}")
    print(f"  Batch Size: {batch_size}")
    print(f"\nPipeline Steps:")
    print(f"  {'[SKIP]' if skip_value_training else '[RUN] '} 1. Train Value Function")
    print(f"  {'[SKIP]' if skip_advantage_computation else '[RUN] '} 2. Compute Advantages")
    print(f"  {'[SKIP]' if skip_policy_training else '[RUN] '} 3. Train ACTvantage Policy")
    print("=" * 80)
    print("\n🚀 Launching pipeline orchestration on Modal...\n")

    rewact_sampler_config = load_sampler_config("scripts/configs/sampler_rewact.json")
    actvantage_sampler_config = load_sampler_config("scripts/configs/sampler_actvantage.json")
    
    # Launch orchestration (runs remotely on Modal)
    result = orchestrate_pipeline.remote(
        dataset_repos=dataset_repos,
        value_function_repo=value_function_repo,
        actvantage_repo=actvantage_repo,
        value_job_name=value_job_name,
        policy_job_name=policy_job_name,
        value_steps=value_steps,
        policy_steps=policy_steps,
        batch_size=batch_size,
        n_step_advantage=n_step_advantage,
        advantage_percentile=advantage_percentile,
        skip_value_training=skip_value_training,
        skip_advantage_computation=skip_advantage_computation,
        skip_policy_training=skip_policy_training,
        checkpoint_push_freq=checkpoint_push_freq,
        rewact_sampler_config=rewact_sampler_config,
        actvantage_sampler_config=actvantage_sampler_config,
    )
    
    print("\n" + "=" * 80)
    print("✅ Pipeline Complete!")
    print("=" * 80)
    
    if not skip_value_training:
        print(f"\n📦 Value Function: https://huggingface.co/{result['value_function']['model_repo_id']}")
        if result['value_function'].get('wandb_url'):
            print(f"   WandB: {result['value_function']['wandb_url']}")
    
    if not skip_advantage_computation:
        print(f"\n📊 Advantages:")
        for dataset_repo, advantage_repo in result['advantages'].items():
            print(f"   • {dataset_repo}")
            print(f"     → https://huggingface.co/datasets/{advantage_repo}")
    
    if not skip_policy_training:
        print(f"\n🤖 ACTvantage Policy: https://huggingface.co/{result['policy']['model_repo_id']}")
        if result['policy'].get('wandb_url'):
            print(f"   WandB: {result['policy']['wandb_url']}")
    
    print("\n" + "=" * 80)
    
    return result

