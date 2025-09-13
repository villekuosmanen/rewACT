# RewACT: Reward-Augmented Action Chunking with Transformers

A PyTorch implementation of RewACT, extending the ACT (Action Chunking with Transformers) model with reward-based learning for improved robotic control. Compatible with LeRobot.

## Installation

### Install from source

```bash
# Clone the repository
git clone https://github.com/your-username/rewACT.git
cd rewACT

# Install in development mode (recommended for development)
pip install -e .

# Or install with development tools
pip install -e ".[dev]"

# Or install with training script dependencies
pip install -e ".[scripts]"

# Or install everything
pip install -e ".[all]"

# Production mode
pip install .
```

### Using the installation helper

```bash
# Development installation
python install.py dev

# Production installation  
python install.py prod

# Test installation
python install.py test
```

## Quick Start

### Using the Dataset Wrapper

```python
import rewact
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Create a base dataset
base_dataset = LeRobotDataset("your_dataset_id")

# Wrap it with reward calculation
dataset_with_reward = rewact.LeRobotDatasetWithReward(
    dataset=base_dataset,
    reward_start_pct=0.05,  # Start giving rewards at 5% episode progress
    reward_end_pct=0.95     # Max reward at 95% episode progress
)

# Use the dataset - now includes calculated rewards
item = dataset_with_reward[0]
print(f"Reward: {item['reward']}")
```

### Creating a RewACT Policy

```python
import rewact
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rewact.policy import RewACTConfig

# Load dataset and config
dataset = LeRobotDataset("your_dataset_id")
config = RewACTConfig()

# Create policy using the utility function
policy = rewact.make_rewact_policy(config, dataset.meta)

# Use for inference
with torch.no_grad():
    actions = policy(observation_batch)
```

## What is a reward model?

When tracking success and failure in robotics, the concept of a reward is often invoked. In a typical scenario, a reward of 1 describes success while a reward of 0 describes non-success. Such binary setup describes a _sparse rewards_ environment, which is usually hard to for machine learning methods to optimise. In many cases we would prefer _dense rewards_ where we can grade robot actions on a sliding scale between 0 and 1. A **reward model** is an AI model that can predict what reward an action should be assigned, aiming to provide dense reward feedback which minimal human intervention or labelling. Reward models are most often used in reinforcement learning thought they can be used in other contexts as well.

## How does this work?

RewACT is incredibly simple - we add a new _reward head_ to the standard ACT transformer model and train it via supervised learning, integrating reward predictions into the loss function.

```
self.reward_head = nn.Sequential(
    nn.Linear(config.dim_model, config.dim_model // 2),
    nn.ReLU(),
    nn.Linear(config.dim_model // 2, config.dim_model // 4),
    nn.ReLU(),
    nn.Linear(config.dim_model // 4, 1),
    nn.Sigmoid(),
)
```

### Where do the labels come?

In imitation learning, all of our training episodes are assumed to be successful. Thus, we can assume the reward of the final frame of each episode is 1, while the start frame is 0. To get dense rewards we can linearly interpolate over all frames to calculate what the reward value _should_ be at that point. Because most datasets contain a few frames with limited motion in the beginning and end, we can start and end the linear interpolation at 5% and 95% to make values of 0 and 1 more likely.

The linear interpolation method is not the most accurate - we can improve the method by labelling partial and full rewards in the dataset. In a pick and place task, we can label the pick with a reward of 0.5, and the placement with a reward of 1.

## How do I use it?

To use rewACT, replace the following files inside `lerobot` with the versions shown here (valid since 5th of August). Alternatively, [merge this pull request to your fork](https://github.com/huggingface/lerobot/pull/1696) or apply the changes manually to your fork. The main changes are:

1. A method for inferring a dense reward based on episode progress.
2. A reward prediction head bolt-on to the ACT transformer model.

You can use rewACT with any pre-existing LeRobot dataset. Just train an ACT model using the default training script - the reward prediction is integrated into the loss function and will be optimised as part of the training process. You can follow this in Wandb as usual.

You can test the reward prediction on an existing dataset using `visualise.py` - the reward value will be rendered in the output video. You should use unseen data for this test - either a different dataset for the same task, or episodes not used for training if you only have 1 dataset (you can restruct these using the `dataset.episodes` param, for example `--dataset.episodes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]`).

You can use the rewACT model like any ACT model, you only need to add a second return value to the model call (i.e. replace `action = policy.select_action(...) with action, reward = policy.select_action(...)`). **You can use the predicted reward value for many purposes, such as detecting when the task is complete.**

## But does it _really_ work? It feels like this shouldn't work...

Kind of. Here's a demo from one of my datasets - the rewACT model was trained on 25 episodes, with the episode in the demo being unseen during training.

![Pepsi stacking demo](https://github.com/villekuosmanen/rewACT/blob/main/videos/pepsi_cans_rewards.gif?raw=true)

Because this method is supervised, it doesn't generalise well outside distribution (like ACT in general). A reward model trained with successful episodes only isn't able to handle failed scenarios well either - to do this we need data with failures (e.g. eval data) labelled with moments where the reward falls to zero. This process requires more human engineering effort but can deliver cool results like this.

![Dice tower demo](https://github.com/villekuosmanen/rewACT/blob/main/videos/dice_tower_rewards.gif?raw=true)

Reward models also struggle to tell between a success and failure if the beginning and end of the task look the same. An example of this is a self-resetting task like the one in the above demo.

