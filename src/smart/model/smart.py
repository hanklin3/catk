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

import math
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler

from src.smart.metrics import (
    CrossEntropy,
    TokenCls,
    WOSACMetrics,
    WOSACSubmission,
    minADE,
)
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils.finetune import set_model_for_finetuning
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts

from .var import VAR


class SMART(LightningModule):

    def __init__(self, model_config) -> None:
        super(SMART, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = SMARTDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
        )
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        self.wosac_metrics = WOSACMetrics("val_closed")
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = CrossEntropy(**model_config.training_loss)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling\
        
        # Important: This property (False) activates manual optimization.
        self.automatic_optimization = True
        
        # Gradient clipping and new scale for overflow
        self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11, growth_interval=1000)
        if not self.automatic_optimization:
            self.gradient_clip_val = model_config.gradient_clip_val
            self.gradient_clip_algorithm = model_config.gradient_clip_algorithm
            self.overflow_new_scale = model_config.overflow_new_scale

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)
        elif self.training_rollout_sampling.criterium == "next_scale": # behavior cloning
            pred = self.encoder.open_next_scale(
                tokenized_map, 
                tokenized_agent, 
                sampling_scheme=self.training_rollout_sampling,
                train_mask=data["agent"]["train_mask"],  # [n_agent]
            )
        elif self.training_rollout_sampling.criterium == "next_scale_var": # openshot next-scale training
            pred = self.encoder.next_scale_var(
                tokenized_map, 
                tokenized_agent, 
                sampling_scheme=self.training_rollout_sampling
            )
        elif self.training_rollout_sampling.criterium == "next_scale_autoreg": # autoregressive next-scale
            pred = self.encoder.next_scale_autoreg(
                tokenized_map, 
                tokenized_agent, 
                sampling_scheme=self.training_rollout_sampling
            )
        else:
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme=self.training_rollout_sampling,
            )
            
        loss_dict = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
                train_mask=data["agent"]["train_mask"],  # [n_agent]
                current_epoch=self.current_epoch,
            )
        loss = loss_dict["loss"]

        if not self.automatic_optimization:
            opt = self.optimizers()
            # scheduler = self.lr_schedulers()
            opt.zero_grad()

            # early clip gradients
        
            self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, 
                                gradient_clip_algorithm=self.gradient_clip_algorithm)
            
            # self.manual_backward(loss)
            # opt.step()
            
            self.scaler.scale(loss).backward()
            
            # import pdb; pdb.set_trace()
            
            scaler_sc: float = self.scaler.get_scale()
            overflow_found = False
            if scaler_sc > self.overflow_new_scale: # fp16 will overflow when >65536, so multiply 32768 could be dangerous
                print("OVERFLOW found: scaler scaler_sc > 32768, reducing scale to 32768")
                overflow_found = True
            #     self.scaler.update(new_scale=32768.)
            
            nan_grad_found = False
            for name, param in self.encoder.named_parameters():
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    nan_grad_found = True
                    print(f"NaN gradient found in {name}")
                    break
                


            if overflow_found or nan_grad_found:
                # fp16 will overflow when >65536, so multiply 32768 could be dangerous
                print("nan grad found, reducing scale")
                self.scaler.update(new_scale=self.overflow_new_scale)
            else:
                self.scaler.step(opt)
                self.scaler.update()
            
        # scheduler.step()

        self.log("train/loss", loss_dict["loss"], on_step=True, batch_size=1)
        self.log("train/loss_cross_entropy", loss_dict["cross_entropy_loss"], on_step=True, batch_size=1)
        self.log("train/loss_vq", loss_dict["loss_commitment_dictionary"], on_step=True, batch_size=1)
        self.log("train/loss_vqvae", loss_dict["loss_vqvae"], on_step=True, batch_size=1)
        self.log("train/loss_var", loss_dict["loss_var"], on_step=True, batch_size=1)
        self.log("train/loss_angular_acc", loss_dict["loss_angular_acc"], on_step=True, batch_size=1)
        
        print('\n')
        print('self.loss_sum / self.count', loss_dict["loss"])
        print('self.loss_vq', loss_dict["loss_commitment_dictionary"])
        print('self.cross_entropy_loss / self.count', loss_dict["cross_entropy_loss"])
        print('self.loss_vqvae / self.count', loss_dict["loss_vqvae"])
        print('self.loss_var', loss_dict["loss_var"])
        print('self.loss_angular_acc', loss_dict["loss_angular_acc"])
        
        assert not torch.isnan(loss_dict["loss_var"]), loss_dict["loss_var"]
        
        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ! open-loop vlidation
        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)
            loss_dict  = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            )
            loss = loss_dict["loss"]
            self.TokenCls.update(
                # action that goes from [(10->15), ..., (85->90)]
                pred=pred["next_token_logits"],  # [n_agent, 16, n_token]
                pred_valid=pred["next_token_valid"],  # [n_agent, 16]
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log(
                "val_open/acc",
                self.TokenCls,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        # ! closed-loop vlidation
        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                if self.training_rollout_sampling.criterium == "next_scale_autoreg": # autoregressive next-scale
                    pred = self.encoder.next_scale_autoreg(
                        tokenized_map, tokenized_agent, sampling_scheme=self.training_rollout_sampling
                    )
                else:
                    pred = self.encoder.inference(
                        tokenized_map, tokenized_agent, self.validation_rollout_sampling
                    )
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
            pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

            # ! WOSAC
            scenario_rollouts = None
            if self.wosac_submission.is_active:  # ! save WOSAC submission
                self.wosac_submission.update(
                    scenario_id=data["scenario_id"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                    global_rank=self.global_rank,
                )
                _gpu_dict_sync = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in _gpu_dict_sync.keys():  # single gpu fix
                        if type(_gpu_dict_sync[k]) is list:
                            _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                    scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
                    self.wosac_submission.aggregate_rollouts(scenario_rollouts)
                self.wosac_submission.reset()

            else:  # ! compute metrics, disable if save WOSAC submission
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][
                        :, self.num_historical_steps :, : pred_traj.shape[-1]
                    ],
                    target_valid=data["agent"]["valid_mask"][
                        :, self.num_historical_steps :
                    ],
                )

                # WOSAC metrics
                if batch_idx < self.n_batch_wosac_metric:
                    device = pred_traj.device
                    scenario_rollouts = get_scenario_rollouts(
                        scenario_id=get_scenario_id_int_tensor(
                            data["scenario_id"], device
                        ),
                        agent_id=data["agent"]["id"],
                        agent_batch=data["agent"]["batch"],
                        pred_traj=pred_traj,
                        pred_z=pred_z,
                        pred_head=pred_head,
                    )
                    self.wosac_metrics.update(data["tfrecord_path"], scenario_rollouts)

            # ! visualization
            if self.global_rank == 0 and batch_idx < self.n_vis_batch:
                if scenario_rollouts is not None:
                    for _i_sc in range(self.n_vis_scenario):
                        # # if model_config.n_vis_batch > data.val_batch_size  
                        # if _i_sc >= len(data["tfrecord_path"]):
                        #     continue
                        # try:
                        #     data["tfrecord_path"][_i_sc]
                        # except:
                        #     print('error tfrecord_path', len(data["tfrecord_path"]), data["tfrecord_path"])
                        #     print('_i_sc', _i_sc)
                        #     continue
                        _vis = VisWaymo(
                            scenario_path=data["tfrecord_path"][_i_sc],
                            save_dir=self.video_dir
                            / f"batch_{batch_idx:02d}-scenario_{_i_sc:02d}",
                        )
                        _vis.save_video_scenario_rollout(
                            scenario_rollouts[_i_sc], self.n_vis_rollout
                        )
                        for _path in _vis.video_paths:
                            self.logger.log_video(
                                "/".join(_path.split("/")[-3:]), [_path]
                            )

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()
                epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
                if self.global_rank == 0:
                    epoch_wosac_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    self.logger.log_metrics(epoch_wosac_metrics)

                self.wosac_metrics.reset()
                self.minADE.reset()

            if self.global_rank == 0:
                if self.wosac_submission.is_active:
                    self.wosac_submission.save_sub_file()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return (
                    self.lr_min_ratio
                    + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
                )
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps)
                        / (self.lr_total_steps - self.lr_warmup_steps),
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ! only closed-loop vlidation
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            if self.training_rollout_sampling.criterium == "next_scale_autoreg": # autoregressive next-scale
                pred = self.encoder.next_scale_autoreg(
                    tokenized_map, tokenized_agent, sampling_scheme=self.training_rollout_sampling
            )
            else:
                pred = self.encoder.inference(
                    tokenized_map, tokenized_agent, self.validation_rollout_sampling
                )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])

        pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
        pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
        pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

        # ! WOSAC submission save
        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            global_rank=self.global_rank,
        )
        _gpu_dict_sync = self.wosac_submission.compute()
        if self.global_rank == 0:
            for k in _gpu_dict_sync.keys():  # single gpu fix
                if type(_gpu_dict_sync[k]) is list:
                    _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
            scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()
            
    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {self.current_epoch}: LR = {current_lr}")
