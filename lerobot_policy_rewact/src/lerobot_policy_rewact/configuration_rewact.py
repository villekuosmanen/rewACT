# Copyright 2024 Tony Z. Zhao, The HuggingFace Inc. team, and Ville Kuosmanen. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("rewact")
@dataclass
class RewACTConfig(ACTConfig):
    # Reward prediction head
    use_reward_head: bool = True # Deprecated - unused
    reward_loss_weight: float = 0.1
    num_value_bins: int = 100  # Number of bins for distributional value function
    value_min: int = 0
    value_max: int = 1
