# lerobot_policy_rewact

RewACT: Reward-Augmented Action Chunking with Transformers - A LeRobot Policy Plugin

## Installation

```bash
pip install lerobot_policy_rewact
```

Or install from source:

```bash
cd lerobot_policy_rewact
pip install -e .
```

## Usage

Once installed, the RewACT policy automatically integrates with LeRobot's training and evaluation tools:

```bash
lerobot-train \
    --policy.type rewact \
    --env.type pusht \
    --steps 200000
```

## What is RewACT?

RewACT extends the ACT (Action Chunking with Transformers) model with reward-based learning. It adds a reward prediction head to the standard ACT transformer model and trains it via supervised learning, integrating reward predictions into the loss function.

The reward model predicts dense rewards for robotic actions, providing feedback on task progress. This is particularly useful for:
- Detecting when tasks are complete
- Providing intermediate feedback during task execution
- Improving policy learning through reward signals

## Features

- Distributional value prediction using a discrete value function
- Compatible with all LeRobot datasets and environments
- Minimal dependencies (only lerobot >= 0.4 required)
- Full integration with LeRobot's training pipeline


