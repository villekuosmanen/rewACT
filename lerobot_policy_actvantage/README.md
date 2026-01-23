# lerobot_policy_actvantage

ACTvantage: Advantage-conditioned ACT policy in the style of pi*0.6 - A LeRobot Policy Plugin

## Installation

```bash
pip install lerobot_policy_actvantage
```

Or install from source:

```bash
cd lerobot_policy_actvantage
pip install -e .
```

## Usage

Once installed, the ACTvantage policy automatically integrates with LeRobot's training and evaluation tools:

```bash
lerobot-train \
    --policy.type actvantage \
    --env.type pusht \
    --steps 200000
```

## What is ACTvantage?

ACTvantage is an advantage-conditioned policy that extends ACT (Action Chunking with Transformers) in the style of pi*0.6. It conditions the policy on advantage values, allowing the model to learn from both high and low advantage trajectories.

## Features

- Advantage-conditioned action prediction
- Compatible with all LeRobot datasets and environments
- Minimal dependencies (only lerobot >= 0.4 required)
- Full integration with LeRobot's training pipeline



