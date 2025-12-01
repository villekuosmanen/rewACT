"""
Example configuration for Modal training pipeline.

This shows all available parameters with their defaults.
You can copy and modify this for your own training runs.
"""

# Default configuration used in the pipeline
PIPELINE_CONFIG = {
    # Dataset configuration
    "datasets": [
        "villekuosmanen/pack_toothbrush_Nov19",
        "villekuosmanen/pack_toothbrush_Nov26",
        "villekuosmanen/dAgger_pack_toothbrush_Nov22",
        "villekuosmanen/dAgger_pack_toothbrush_Nov26",
    ],
    
    # Output repositories
    "value_function_repo": "villekuosmanen/rewact_value_function",
    "actvantage_repo": "villekuosmanen/actvantage_policy",
    
    # Job names (used for WandB and local output directories)
    "value_job_name": "rewact_value",
    "policy_job_name": "actvantage_policy",
    
    # Training hyperparameters
    "value_steps": 50000,           # Training steps for value function
    "policy_steps": 100000,         # Training steps for policy
    "batch_size": 24,               # Batch size for both trainings
    "n_step_advantage": 50,         # N-step lookahead for advantage computation
    "advantage_percentile": 30.0,   # Percentile threshold for advantage filtering
    
    # Pipeline control
    "skip_value_training": False,
    "skip_advantage_computation": False,
    "skip_policy_training": False,
    
    # Checkpoint configuration
    "checkpoint_push_freq": None,   # Set to int (e.g., 10000) to push intermediate checkpoints
}


# Example: Training on different datasets
CUSTOM_DATASET_CONFIG = {
    **PIPELINE_CONFIG,
    "datasets": [
        "username/my_dataset_1",
        "username/my_dataset_2",
    ],
    "value_function_repo": "username/my_value_function",
    "actvantage_repo": "username/my_policy",
}


# Example: Resume from existing value function
RESUME_CONFIG = {
    **PIPELINE_CONFIG,
    "skip_value_training": True,
    "value_function_repo": "username/existing_value_function",  # Use existing model
}


# Example: Only compute advantages (no training)
ADVANTAGE_ONLY_CONFIG = {
    **PIPELINE_CONFIG,
    "skip_value_training": True,
    "skip_policy_training": True,
    "value_function_repo": "username/existing_value_function",
}


# Example: Push checkpoints every 10k steps
CHECKPOINT_CONFIG = {
    **PIPELINE_CONFIG,
    "checkpoint_push_freq": 10000,  # Push to Hub every 10k steps
}


# Example: Quick test run with fewer steps
TEST_CONFIG = {
    **PIPELINE_CONFIG,
    "value_steps": 1000,
    "policy_steps": 1000,
    "batch_size": 8,
}


def config_to_args(config: dict) -> list:
    """Convert config dict to command-line arguments."""
    args = []
    
    # Handle datasets specially (needs to be comma-separated string)
    if "datasets" in config:
        datasets_str = ",".join(config["datasets"])
        args.extend(["--datasets", datasets_str])
    
    # Handle other parameters
    param_map = {
        "value_function_repo": "--value-function-repo",
        "actvantage_repo": "--actvantage-repo",
        "value_job_name": "--value-job-name",
        "policy_job_name": "--policy-job-name",
        "value_steps": "--value-steps",
        "policy_steps": "--policy-steps",
        "batch_size": "--batch-size",
        "n_step_advantage": "--n-step-advantage",
        "advantage_percentile": "--advantage-percentile",
        "checkpoint_push_freq": "--checkpoint-push-freq",
    }
    
    for key, flag in param_map.items():
        if key in config and config[key] is not None:
            args.extend([flag, str(config[key])])
    
    # Handle boolean flags
    bool_flags = {
        "skip_value_training": "--skip-value-training",
        "skip_advantage_computation": "--skip-advantage-computation",
        "skip_policy_training": "--skip-policy-training",
    }
    
    for key, flag in bool_flags.items():
        if config.get(key, False):
            args.append(flag)
    
    return args


if __name__ == "__main__":
    print("Example configurations for Modal training pipeline")
    print("=" * 60)
    
    configs = {
        "Default": PIPELINE_CONFIG,
        "Custom Datasets": CUSTOM_DATASET_CONFIG,
        "Resume from Existing": RESUME_CONFIG,
        "Advantage Only": ADVANTAGE_ONLY_CONFIG,
        "With Checkpoints": CHECKPOINT_CONFIG,
        "Quick Test": TEST_CONFIG,
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        args = config_to_args(config)
        print(f"  modal run scripts/modal_pipeline.py {' '.join(args)}")





