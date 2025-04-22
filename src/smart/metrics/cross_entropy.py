# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional

import torch
from torch import Tensor, tensor
from torch.nn.functional import cross_entropy
from torchmetrics.metric import Metric
from torch.nn import functional as F

from .utils import get_euclidean_targets, get_prob_targets


class CrossEntropy(Metric):

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        use_gt_raw: bool,
        gt_thresh_scale_length: float,  # {"veh": 4.8, "cyc": 2.0, "ped": 1.0}
        label_smoothing: float,
        rollout_as_gt: bool,
        angular_acc_weight: float=0.0, # angular acceleration loss weight
        beta: float=0.25, # commitment loss weight
    ) -> None:
        super().__init__()
        self.use_gt_raw = use_gt_raw
        self.gt_thresh_scale_length = gt_thresh_scale_length
        self.label_smoothing = label_smoothing
        self.rollout_as_gt = rollout_as_gt
        self.add_state("loss_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("cross_entropy_loss", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss_commitment_dictionary", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss_vqvae", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss_var", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss_angular_acc", default=tensor(0.0), dist_reduce_fx="sum")
        
        self.beta = beta                              # commitment loss weight
        self.angular_acc_weight = angular_acc_weight  # angular acceleration loss weight
        print('CrossEntropy: beta: ', beta, "angular_acc_weight: ", angular_acc_weight)

    def update(
        self,
        # ! action that goes from [(10->15), ..., (85->90)]
        next_token_logits: Tensor,  # [n_agent, 16, n_token]
        next_token_valid: Tensor,  # [n_agent, 16]
        # ! for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
        pred_pos: Tensor,  # [n_agent, 18, 2]
        pred_head: Tensor,  # [n_agent, 18]
        pred_valid: Tensor,  # [n_agent, 18]
        # ! for step {5, 10, ..., 90}
        gt_pos_raw: Tensor,  # [n_agent, 18, 2]
        gt_head_raw: Tensor,  # [n_agent, 18]
        gt_valid_raw: Tensor,  # [n_agent, 18]
        # or use the tokenized gt
        gt_pos: Tensor,  # [n_agent, 18, 2]
        gt_head: Tensor,  # [n_agent, 18]
        gt_valid: Tensor,  # [n_agent, 18]
        # ! for tokenization
        token_agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2]
        # ! for filtering intersting agent for training
        train_mask: Optional[Tensor] = None,  # [n_agent]
        # ! for rollout_as_gt
        next_token_action: Optional[Tensor] = None,  # [n_agent, 16, 3]
        # ! vqvae loss
        loss_commitment_dictionary: Optional[Tensor] = None,  # [1]
        f_hat_per_level: Optional[Tensor] = None,  # List[Tensor[n_agent, n_emb, 18, 2]]
        n_points_per_level: Optional[Tensor] = None,  # List[int]
        f_BCt2: Optional[Tensor] = None,  # [n_agent, n_emb, 18, 2] # encoder output
        f_BCt2_reconstructed: Optional[Tensor] = None,  # [n_agent, n_emb, 18, 2] or [n_agent, 18, n_token=vocab_size] # decoder output
        # ! var loss
        var_loss: Optional[Tensor] = None,  # [1]
        var_logits_BLV: Optional[Tensor] = None,  # [B, L, V]
        var_gt_BL: Optional[Tensor] = None,  # [B, L]
        **kwargs,
    ) -> None:
        # ! use raw or tokenized GT
        if self.use_gt_raw:
            gt_pos = gt_pos_raw
            gt_head = gt_head_raw
            gt_valid = gt_valid_raw

        # ! GT is valid if it's close to the rollout.
        if self.gt_thresh_scale_length > 0:
            dist = torch.norm(pred_pos - gt_pos, dim=-1)  # [n_agent, n_step]
            _thresh = token_agent_shape[:, 1] * self.gt_thresh_scale_length  # [n_agent]
            gt_valid = gt_valid & (dist < _thresh.unsqueeze(1))  # [n_agent, n_step]

        # ! get prob_targets, transform_to_local gt to pred frame
        euclidean_target, euclidean_target_valid = get_euclidean_targets(
            pred_pos=pred_pos,
            pred_head=pred_head,
            pred_valid=pred_valid,
            gt_pos=gt_pos,
            gt_head=gt_head,
            gt_valid=gt_valid,
        ) # hk: [n_agent, 16, 3] x,y,yaw, [n_agent, 16]
        if self.rollout_as_gt and (next_token_action is not None):
            euclidean_target = next_token_action
        # use contour to compute one-hot prob_target
        prob_target = get_prob_targets(
            target=euclidean_target,  # [n_agent, n_step, 3] x,y,yaw in local
            token_agent_shape=token_agent_shape,  # [n_agent, 2]
            token_traj=token_traj,  # [n_agent, n_token, 4, 2]
        )  # [n_agent, n_step, n_token] prob, last dim sum up to 1
        # n_token = n_classes = token vocabulary size
        loss = cross_entropy(
            next_token_logits.transpose(1, 2),  # [n_agent, n_token, n_step], logits: [B, n_classes, dim1]
            prob_target.transpose(1, 2),  # [n_agent, n_token, n_step], prob
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # [n_agent, n_step=16]

        # ! weighting final loss [n_agent, n_step]
        loss_weighting_mask = next_token_valid & euclidean_target_valid
        if self.training:
            loss_weighting_mask &= train_mask.unsqueeze(1)  # [n_agent, n_step]

        # self.loss_sum += (loss * loss_weighting_mask).sum()
        self.count += (loss_weighting_mask > 0).sum()
        self.cross_entropy_loss += (loss * loss_weighting_mask).sum()
  
        assert ((loss_weighting_mask == 1) | (loss_weighting_mask == 0)).all(), loss_weighting_mask

        # Commitment and dictionary loss
        if loss_commitment_dictionary is not None:
            self.loss_commitment_dictionary += loss_commitment_dictionary 
        elif f_BCt2 is not None:
            f_BCt2 = f_BCt2[:, :, 1:-1, :]  # [n_agent, n_emb, n_step=16, 2]
            assert loss_weighting_mask.shape == (f_BCt2.shape[0], f_BCt2.shape[2]), (loss_weighting_mask.shape, f_BCt2.shape)
            f_BCt2 = f_BCt2 * loss_weighting_mask[:, None, :, None].float()
            f_no_grad = f_BCt2.detach()
            assert not torch.isclose(f_hat_per_level[0], f_hat_per_level[1]).all()
            for f_hat in f_hat_per_level:
                f_hat = f_hat[:, :, 1:-1, :]  # [n_agent, n_emb, n_step=16, 2]
                f_hat = f_hat * loss_weighting_mask[:, None, :, None].float()
                # Commitment loss: make sure encoder output f_BChw  ze(x)  commits to the quantized embeddings (f_hat.data) (.data detach from graph), sg[e]. 
                # Dictionary Loss: ensures the quantized embeddings f_hat accurately reconstruct the encoder output feature map f_no_grad sg[ze(x)]
                self.loss_commitment_dictionary += F.mse_loss(f_hat.data, f_BCt2).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            assert len(n_points_per_level) == len(f_hat_per_level), (len(n_points_per_level), len(f_hat_per_level))
            # self.loss_commitment_dictionary *= 1. / len(n_points_per_level)
        
        # self.loss_sum += loss_commitment_dictionary # shape [1] <- [1]

        if f_BCt2_reconstructed is not None:
            _, _, W_dim = f_BCt2_reconstructed.shape
            assert W_dim != 2
            if W_dim == 2: # traj x,y
                f_BCt2_reconstructed = f_BCt2_reconstructed[:, 1:-1, :]  # [n_agent, n_step=16, 2]
                f_BCt2_reconstructed = f_BCt2_reconstructed * loss_weighting_mask[:, :, None].float()
                loss_vqvae += F.mse_loss(f_BCt2_reconstructed, euclidean_target[:, :, 0:2])
            else: # W_dim == n_token = 2048
                # cross entropy loss
                loss_vqvae = cross_entropy(
                    f_BCt2_reconstructed[:, 1:-1, :].transpose(1, 2),  # [n_agent, n_token, n_step], logits: [B, class, dim1]
                    prob_target.transpose(1, 2),  # [n_agent, n_token, n_step], prob
                    reduction="none",
                    label_smoothing=self.label_smoothing,
                )  # [n_agent, n_step=16]
                loss_vqvae = (loss_vqvae * loss_weighting_mask).sum()
            self.loss_vqvae += loss_vqvae
            # self.loss_sum += loss_vqvae

        if var_loss is not None:
            self.loss_var += var_loss
            
        # Calculate angular velocity and angular acceleration
        angular_velocity = pred_head[:, 1:] - pred_head[:, :-1]  # [n_agent, 17]
        angular_acceleration = angular_velocity[:, 1:] - angular_velocity[:, :-1]  # [n_agent, 16]

        # Compute angular acceleration loss
        self.loss_angular_acc += torch.mean(angular_acceleration ** 2)

        # print('\n')
        # print('self.loss_sum / self.count', self.loss_sum / self.count)
        # print('self.loss_commitment_dictionary / self.count', self.loss_commitment_dictionary / self.count)
        # print('self.cross_entropy_loss / self.count', self.cross_entropy_loss / self.count)
        # print('self.loss_L2_traj / self.count', self.loss_L2_traj / self.count)

    def compute(self) -> Tensor:
        return {'loss': self.cross_entropy_loss / self.count + self.loss_commitment_dictionary + 
                self.loss_vqvae / self.count + self.loss_var + self.angular_acc_weight * self.loss_angular_acc / self.count,
            # 'loss': self.loss_sum / self.count,
                'cross_entropy_loss': self.cross_entropy_loss / self.count, 
                'loss_commitment_dictionary': self.loss_commitment_dictionary, 
                'loss_vqvae': self.loss_vqvae / self.count,
                'loss_var': self.loss_var,
                'loss_angular_acc': self.loss_angular_acc / self.count * self.angular_acc_weight}