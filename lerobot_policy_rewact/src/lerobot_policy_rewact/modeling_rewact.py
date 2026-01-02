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

from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .configuration_rewact import RewACTConfig
from .vision import make_vision_encoder


class RewACTPolicy(PreTrainedPolicy):
    """
    Reward prediction wrapper for ACT.
    """

    config_class = RewACTConfig
    name = "rewact"

    def __init__(
        self,
        config: RewACTConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            kwargs: unused.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = RewACT(config)

        # Pre-compute bin edges for discretizing continuous values during training
        self.register_buffer(
            "bin_edges",
            torch.linspace(config.value_min, config.value_max, config.num_value_bins + 1)
        )

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def _value_to_bin(self, values: Tensor) -> Tensor:
        """Convert continuous values to bin indices.
        
        Args:
            values: (B,) or (B, 1) continuous values in [value_min, value_max]
            
        Returns:
            bin_indices: (B,) long tensor with bin indices in [0, num_bins-1]
        """
        values = values.squeeze()  # (B,)
        # Clip to valid range
        values = torch.clamp(values, self.config.value_min, self.config.value_max)
        
        # Find which bin each value belongs to
        # torch.bucketize returns indices where values would be inserted
        # We subtract 1 because bucketize returns the right edge
        bin_indices = torch.bucketize(values, self.bin_edges) - 1
        
        # Clamp to valid bin range [0, num_bins-1]
        bin_indices = torch.clamp(bin_indices, 0, self.config.num_value_bins - 1)
        
        return bin_indices


    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        backbone_prefixes = ("model.vision_encoder", "model.backbone")
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith(backbone_prefixes) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith(backbone_prefixes) and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], force_model_run: bool = False) -> tuple[Tensor, Tensor]:
        """Select a single action given environment observations."""
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions, reward_output = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            # Use expected value for inference
            reward_pred = reward_output['expected_value'][0, 0]
            reward_pred = torch.clamp(reward_pred, 0.0, 1.0)
            return action, reward_pred

        current_reward_pred = 0.0
        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions, reward_output = self.predict_action_chunk(batch)
            actions = actions[:, : self.config.n_action_steps]

            # Use expected value for inference
            current_reward_pred = reward_output['expected_value'][0, 0]
            current_reward_pred = torch.clamp(current_reward_pred, 0.0, 1.0)
            self._action_queue.extend(actions.transpose(0, 1))

        elif force_model_run:
            _, reward_output = self.predict_action_chunk(batch)
            current_reward_pred = reward_output['expected_value'][0, 0]
            current_reward_pred = torch.clamp(current_reward_pred, 0.0, 1.0)

        return self._action_queue.popleft(), current_reward_pred


    def get_reward_pred(self, batch: dict[str, Tensor]) -> Tensor:
        """Get the reward prediction for the current batch."""
        _, reward_output = self.predict_action_chunk(batch)
        current_reward_pred = reward_output['expected_value'][0, 0]
        current_reward_pred = torch.clamp(current_reward_pred, 0.0, 1.0)
        return current_reward_pred

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
            # Past images for VJEPA2 temporal context
            if self.config.vision_encoder_type == "vjepa2":
                past_keys = [k.replace("observation.images.", "observation.images_past.") for k in self.config.image_features]
                if all(k in batch for k in past_keys):
                    batch["observation.images_past"] = [batch[k] for k in past_keys]

        actions, reward_output, _ = self.model(batch)
        return actions, reward_output

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
            # Past images for VJEPA2 temporal context
            if self.config.vision_encoder_type == "vjepa2":
                past_keys = [k.replace("observation.images.", "observation.images_past.") for k in self.config.image_features]
                if all(k in batch for k in past_keys):
                    batch["observation.images_past"] = [batch[k] for k in past_keys]

        actions_hat, reward_output, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # For action loss, we calculate a combined mask using the use_action_mask and action_is_pad, episode_outcome, as well as control_mode != "policy"
        # action_is_pad and use_action_mask have shape (B, chunk_size), episode_outcome and control_mode have shape (B,)
        # We need to expand all to (B, chunk_size, 1) for proper broadcasting with action predictions
        action_mask = (~batch["action_is_pad"].unsqueeze(-1)) & batch["use_action_mask"].unsqueeze(-1).unsqueeze(-1)
        # if "episode_outcome" in batch:
        #     # Expand (B,) -> (B, 1, 1) to broadcast across chunk_size dimension
        #     action_mask = action_mask & (batch["episode_outcome"].unsqueeze(-1).unsqueeze(-1))
        if "control_mode_autonomous" in batch:
            # Expand (B,) -> (B, 1, 1) to broadcast across chunk_size dimension
            action_mask = action_mask & (~batch["control_mode_autonomous"].squeeze().unsqueeze(-1).unsqueeze(-1))

        # Action loss - only compute for samples where use_action_mask is True
        # Compute mean only over valid (non-masked) elements to avoid under-estimating the loss
        masked_l1_loss = F.l1_loss(batch[ACTION], actions_hat, reduction="none") * action_mask
        l1_loss = masked_l1_loss.sum() / (action_mask.sum() + 1e-8)  # Add epsilon to avoid division by zero

        # Distributional value prediction loss - use cross-entropy
        reward_targets = batch["reward"]  # (B, 1) - continuous values in [0, 1]
        
        # Convert continuous targets to bin indices
        target_bins = self._value_to_bin(reward_targets)  # (B,) - discrete bin indices
        
        # Compute cross-entropy loss
        reward_logits = reward_output['logits']  # (B, num_bins)
        reward_loss = F.cross_entropy(
            reward_logits,      # (B, num_bins)
            target_bins,        # (B,)
            reduction='mean'
        )

        loss_dict = {
            "l1_loss": l1_loss.item(),
            "reward_loss": reward_loss.item() * self.config.reward_loss_weight,
        }

        # Log expected value MSE for comparison with old approach
        expected_values = reward_output['expected_value'].squeeze()
        mse_for_logging = F.mse_loss(expected_values, reward_targets.squeeze())
        loss_dict["reward_mse"] = mse_for_logging.item()
        
        # Log entropy (measure of uncertainty)
        reward_dist = reward_output['distribution']
        entropy = -(reward_dist * torch.log(reward_dist + 1e-8)).sum(dim=-1).mean()
        loss_dict["reward_entropy"] = entropy.item()
        
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + self.config.reward_loss_weight * reward_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss + self.config.reward_loss_weight * reward_loss

        return loss, loss_dict


class RewACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: RewACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        ## Vision encoder (image -> encoder tokens).
        if self.config.image_features:
            self.vision_encoder = make_vision_encoder(config)

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        # Distributional value prediction head: predicts distribution over bins
        self.reward_head = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model // 2),
            nn.ReLU(),
            nn.Linear(config.dim_model // 2, config.dim_model // 4),
            nn.ReLU(),
            nn.Linear(config.dim_model // 4, config.num_value_bins),  # Output: distribution over bins
        )
        
        # Pre-compute bin values for converting distribution to continuous value
        # Shape: (num_value_bins,)
        self.register_buffer(
            "bin_values",
            torch.linspace(config.value_min, config.value_max, config.num_value_bins)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        Expected `batch` keys (actual keys used by this implementation):
        Inputs (inference + training):
        - "observation.state": (B, state_dim) robot proprioceptive state. Required if `config.robot_state_feature` is set.
        - "observation.images": optional, list of camera tensors, each (B, C, H, W). Present when using vision inputs.
          Note: `RewACTPolicy` constructs this from per-camera keys like "observation.images.top", etc.
        - "observation.environment_state": optional, (B, env_dim). Present when using env-state inputs.
        Training-only (needed when `config.use_vae` and `self.training` is True):
        - "action": (B, chunk_size, action_dim) action chunk used by the VAE encoder.
        - "action_is_pad": (B, chunk_size) boolean padding mask for the action sequence (True means pad).
        
        Returns:
        - actions: (B, chunk_size, action_dim)
        - reward_preds: (B, 1, 1) if `config.use_reward_head` else None (only predicts reward for the first step)
        - (mu, log_sigma_x2): both (B, latent_dim) if using VAE in training, else (None, None)
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # Prepare transformer encoder inputs.
        encoder_input_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_input_pos_embeds = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_input_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_input_tokens.append(
                self.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            )

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for cam_idx, img in enumerate(batch["observation.images"]):
                # For VJEPA2: stack past + current frame into video tensor
                if self.config.vision_encoder_type == "vjepa2" and "observation.images_past" in batch:
                    past_img = batch["observation.images_past"][cam_idx]
                    img = torch.stack([past_img, img], dim=2)  # (B, 3, 2, H, W)
                img_tokens, img_pos_tokens = self.vision_encoder(img, cam_idx=cam_idx)
                # Extend immediately instead of accumulating and concatenating.
                encoder_input_tokens.extend(list(img_tokens))
                encoder_input_pos_embeds.extend(list(img_pos_tokens))

        # Stack all tokens along the sequence dimension.
        encoder_input_tokens = torch.stack(encoder_input_tokens, axis=0)
        encoder_input_pos_embeds = torch.stack(encoder_input_pos_embeds, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_input_tokens, pos_embed=encoder_input_pos_embeds)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_input_pos_embeds.dtype,
            device=encoder_input_pos_embeds.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_input_pos_embeds,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        # Predict distribution over value bins for the current timestep
        reward_logits = self.reward_head(decoder_out[:, 0, :])  # (B, num_bins)
        reward_dist = F.softmax(reward_logits, dim=-1)  # (B, num_bins)
        
        # Convert distribution to expected value (for inference/logging)
        # reward_preds: (B, 1)
        reward_preds = (reward_dist * self.bin_values).sum(dim=-1, keepdim=True)
        
        # Return both the distribution (logits) and expected value
        # We'll use logits for loss computation, expected value for inference
        reward_output = {
            'logits': reward_logits,      # (B, num_bins) - for training
            'distribution': reward_dist,   # (B, num_bins) - for analysis
            'expected_value': reward_preds # (B, 1) - for inference
        }

        return actions, reward_output, (mu, log_sigma_x2)

