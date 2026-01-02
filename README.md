# RewACT: Reward-Augmented Action Chunking with Transformers

This repository contains LeRobot policy plugins and tools for reward-based learning in robotics.

## Package Structure

This repository is organized into several installable packages:

### Policy Plugins

- **`lerobot_policy_rewact`**: RewACT policy plugin - extends ACT with reward prediction using a distributional value function
- **`lerobot_policy_actvantage`**: ACTvantage policy plugin - advantage-conditioned ACT policy in the style of pi*0.6

Both policy plugins are standalone, installable packages that integrate with LeRobot's training and evaluation tools. They have minimal dependencies (only `lerobot >= 0.4` required).

### Tools Package

- **`rewact_tools`**: Contains plugins for robocandywrapper, factory functions for creating processors, and dataset utilities for managing reward labels.

### Scripts

The `scripts/` directory contains training and visualization scripts that are not installable but can be run directly when you clone this repository.

### Internal Infrastructure

The `rewact_modal_infra/` directory contains trainers and Modal scripts for internal infrastructure. This code is not part of the public packages and will be moved to a private monorepo.

## Installation

### Install Policy Plugins

```bash
# Install RewACT policy
cd lerobot_policy_rewact
pip install -e .

# Install ACTvantage policy
cd ../lerobot_policy_actvantage
pip install -e .
```

### Install Tools

```bash
cd rewact_tools
pip install -e .
```

## Usage

Once installed, the policies automatically integrate with LeRobot's training tools:

```bash
# Train RewACT policy
lerobot-train \
    --policy.type rewact \
    --env.type pusht \
    --steps 200000

# Train ACTvantage policy
lerobot-train \
    --policy.type actvantage \
    --env.type pusht \
    --steps 200000
```

## What is RewACT?

RewACT extends the ACT (Action Chunking with Transformers) model with reward-based learning. It adds a reward prediction head to the standard ACT transformer model and trains it via supervised learning, integrating reward predictions into the loss function.

The reward model predicts dense rewards for robotic actions, providing feedback on task progress. This is particularly useful for:
- Detecting when tasks are complete
- Providing intermediate feedback during task execution
- Improving policy learning through reward signals

## What is ACTvantage?

ACTvantage is an advantage-conditioned policy that extends ACT in the style of pi*0.6. It conditions the policy on advantage values, allowing the model to learn from both high and low advantage trajectories.

## Quick Start

### Train a RewACT policy

You can train a RewACT policy for any standard LeRobot dataset using the `scripts/train.py` script:

```bash
python scripts/train.py \
    --dataset.repo_id=danaaubakirova/so100_task_2 \
    --dataset.episodes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] \
    --policy.type=rewact \
    --policy.repo_id=villekuosmanen/so100_test \
    --output_dir=outputs/train/so100_test_2 \
    --job_name=so100_test \
    --batch_size=32 \
    --eval_freq=-1 \
    --log_freq=50 \
    --save_freq=1000 \
    --steps=10000
```

### Evaluating the trained RewACT policy

To avoid overfitting, evaluate the trained policy using a validation or test dataset:

```bash
python scripts/visualise_reward_predictions.py \
    --dataset-repo-id "danaaubakirova/so100_task_2" \
    --episode-id 24 \
    --policy-path "outputs/train/so100_test/checkpoints/last/pretrained_model"
```

## How does this work?

RewACT is incredibly simple - we add a new _reward head_ to the standard ACT transformer model and train it via supervised learning, integrating reward predictions into the loss function.

### Where do the labels come?

In imitation learning, all of our training episodes are assumed to be successful. Thus, we can assume the reward of the final frame of each episode is 1, while the start frame is 0. To get dense rewards we can linearly interpolate over all frames to calculate what the reward value _should_ be at that point. Because most datasets contain a few frames with limited motion in the beginning and end, we can start and end the linear interpolation at 5% and 95% to make values of 0 and 1 more likely.

The linear interpolation method is not the most accurate - we can improve the method by labelling partial and full rewards in the dataset. In a pick and place task, we can label the pick with a reward of 0.5, and the placement with a reward of 1.

## License

MIT License - see LICENSE file for details.
