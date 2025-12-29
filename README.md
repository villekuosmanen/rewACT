# RewACT: Reward-Augmented Action Chunking with Transformers

A PyTorch implementation of RewACT, extending the ACT (Action Chunking with Transformers) model with reward-based learning for improved robotic control. Compatible with the newest version of LeRobot.

## Installation

`pip install rewact`

You can also install the package in editable mode by cloning the repository and using `pip install -e .`.

## DINOv3 vision backbones (optional)

RewACT can swap the vision frontend between the default torchvision ResNet feature-map pipeline and **DINOv3** backbones (ViT and ConvNeXt).

### Install DINOv3 (local repo)

DINOv3 requires **Python >= 3.11**. After activating your env:

```
cd /path/to/dinov3
pip install -e .
```

### Enable DINOv3 in training

Use these policy flags (example: **ViT-B/16**):

```
--policy.vision_encoder_type=dinov3 \
--policy.dinov3.variant=vitb16 \
--policy.dinov3.weights=/ABS/PATH/TO/dinov3_vitb16_pretrain_*.pth
```

Supported `dinov3.variant` values:
- `vitb16`
- `vitl16`
- `convnext_base`
- `convnext_large`

Note: `freeze_vision_encoder` defaults to `True`. To finetune the vision backbone, set `--policy.freeze_vision_encoder=false`.

### Smoke test (token shapes)

The smoke test instantiates the vision encoder and prints the **number of image tokens** produced per camera at a given resolution. This is useful because self-attention cost scales roughly quadratically with sequence length (about O(S^2) in the number of tokens S), and token count can vary a lot across backbones.

```
python scripts/smoke_test_vision_encoders.py \
  --dinov3-variant vitb16 \
  --dinov3-weights /ABS/PATH/TO/dinov3_vitb16_pretrain_*.pth \
  --h 480 --w 640 --num-cameras 2 --batch-size 2
```

Tip: token count is roughly:
- **ViT (DINOv3, patch=16)**: (H/16) * (W/16)
- **ConvNeXt (DINOv3)**: typically (H/32) * (W/32) (effective stride ~32)

So at 480x640, ViT/16 gives 30 * 40 = 1200 tokens per camera, while ConvNeXt gives 15 * 20 = 300 tokens per camera (about 4x fewer tokens).

## Quick Start

### Train a RewACT policy

You can train a RewACT policy for any standard LeRobot dataset using the `scripts/train.py` script. This script basically copies the standard LeRobot training script so it should work directly with any command or `launch.json` config you use in LeRobot.

Note: `job_name` and `output_dir` default to the model name from `policy.repo_id` (e.g. `so100_test` → `outputs/train/so100_test`).

```
python scripts/train.py \
--dataset.repo_id=danaaubakirova/so100_task_2 \
--dataset.episodes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] \
--policy.type=rewact \
--policy.repo_id=<your-hf-user>/so100_resnet \
--batch_size=32 \
--eval_freq=-1 \
--log_freq=50 \
--save_freq=1000 \
--steps=10000
```

To use a DINOv3 vision backbone, add these flags (example shown with placeholders):

```
python scripts/train.py \
--dataset.repo_id=danaaubakirova/so100_task_2 \
--dataset.episodes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] \
--policy.type=rewact \
--policy.repo_id=<your-hf-user>/so100_dinov3 \
--batch_size=1 \
--eval_freq=-1 \
--log_freq=50 \
--save_freq=500 \
--steps=1000 \
--policy.vision_encoder_type=dinov3 \
--policy.dinov3.variant=<vitb16|vitl16|convnext_base|convnext_large> \
--policy.dinov3.weights=/ABS/PATH/TO/<dinov3_checkpoint>.pth
```

### Evaluating the trained RewACT policy

To avoid overfitting, I highly recommend evaluating the trained policy using a validation or test dataset in the same training distribution.

```
python scripts/visualise_reward_predictions.py \
--dataset-repo-id "danaaubakirova/so100_task_2" \
--episode-id 24 \
--policy-path "outputs/train/so100_test/checkpoints/last/pretrained_model"
```

Note: `scripts/visualise_reward_predictions.py` currently writes to `outputs/reward_visualization.mp4` and will overwrite it between runs (rename/move it if you want to keep multiple outputs).

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

You can use rewACT with any pre-existing LeRobot dataset. Just train an ACT model using the default training script - the reward prediction is integrated into the loss function and will be optimised as part of the training process. You can follow this in Wandb as usual.

You can test the reward prediction on an existing dataset using `scripts/visualise_reward_predictions.py` - the reward value will be rendered in the output video. You should use unseen data for this test - either a different dataset for the same task, or episodes not used for training if you only have 1 dataset (you can restruct these using the `dataset.episodes` param, for example `--dataset.episodes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]`).

You can use the rewACT model like any ACT model, you only need to add a second return value to the model call (i.e. replace `action = policy.select_action(...) with action, reward = policy.select_action(...)`). **You can use the predicted reward value for many purposes, such as detecting when the task is complete.**

## But does it _really_ work? It feels like this shouldn't work...

Kind of. Here's a demo from one of my datasets - the rewACT model was trained on 25 episodes, with the episode in the demo being unseen during training.

![Pepsi stacking demo](https://github.com/villekuosmanen/rewACT/blob/main/videos/pepsi_cans_rewards.gif?raw=true)

Because this method is supervised, it doesn't generalise well outside distribution (like ACT in general). A reward model trained with successful episodes only isn't able to handle failed scenarios well either - to do this we need data with failures (e.g. eval data) labelled with moments where the reward falls to zero. This process requires more human engineering effort but can deliver cool results like this.

![Dice tower demo](https://github.com/villekuosmanen/rewACT/blob/main/videos/dice_tower_rewards.gif?raw=true)

Reward models also struggle to tell between a success and failure if the beginning and end of the task look the same. An example of this is a self-resetting task like the one in the above demo.

