# rewact_tools

RewACT Tools: Plugins and utilities for reward-based learning with LeRobot

## Installation

```bash
pip install rewact_tools
```

Or install from source:

```bash
cd rewact_tools
pip install -e .
```

## Features

This package provides:

- **Plugins for robocandywrapper**: Reward calculation, advantage computation, and control mode tracking
- **Factory functions**: Utilities for creating pre/post processors for RewACT and ACTvantage policies
- **Dataset utilities**: KeypointReward and LeRobotDatasetWithReward for managing reward labels

## Plugins

- `DenseRewardPlugin`: Adds dense reward calculation with keypoint-based interpolation
- `PiStar0_6CumulativeRewardPlugin`: Computes cumulative rewards for pi*0.6 style training
- `PiStar0_6AdvantagePlugin`: Computes advantages for advantage-conditioned policies
- `ControlModePlugin`: Tracks control mode (human vs autonomous) in datasets

## Usage

```python
from rewact_tools import DenseRewardPlugin, KeypointReward, make_pre_post_processors

# Use plugins with robocandywrapper
# Use factory functions for policy processors
# Use KeypointReward for defining reward keypoints
```

