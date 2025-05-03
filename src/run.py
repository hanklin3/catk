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

from typing import List
import os

import hydra
import lightning as L
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig
from lightning.pytorch.plugins.precision import MixedPrecisionPlugin
from torch.cuda.amp import GradScaler


from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    print_config_tree,
)

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("high")
# torch.set_float32_matmul_precision("medium")


def run(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    # setup model watching
    for _logger in logger:
        if isinstance(_logger, WandbLogger):
            _logger.watch(model, log="all")
            
    # # Initialize GradScaler
    # scaler = GradScaler(init_scale=2. ** 11, growth_interval=1000)

    # # Initialize MixedPrecisionPlugin with the scaler
    # amp_plugin = MixedPrecisionPlugin(precision="16-mixed", device="cuda", scaler=scaler)


    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    log.info("Logging hyperparameters!")
    log_hyperparameters(
        {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }
    )
    
    print('cfg.ckpt_path: ', cfg.ckpt_path)
    log.info(f"Resuming from ckpt: cfg.ckpt_path={cfg.ckpt_path}")
    if cfg.action == "fit":
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    elif cfg.action == "finetune":
        log.info("Starting finetuning!")
        model.load_state_dict(torch.load(cfg.ckpt_path, weights_only=False)["state_dict"], strict=False)
        trainer.fit(model=model, datamodule=datamodule)
    elif cfg.action == "validate":
        log.info("Starting validating!")
        trainer.validate(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )
    elif cfg.action == "test":
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


@hydra.main(config_path="../configs/", config_name="run.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_printoptions(precision=3)

    log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
    print_config_tree(cfg, resolve=True, save_to_file=True)

    # copy agent_decoder.py to logs/experiment_name
    os.system(f"cp src/smart/modules/agent_decoder.py logs/{cfg.task_name}/")
    os.system(f"cp src/smart/model/smart.py logs/{cfg.task_name}/")
    os.system(f"cp src/smart/metrics/cross_entropy.py logs/{cfg.task_name}/")
    os.system(f"cp src/smart/model/quant.py logs/{cfg.task_name}/")
    os.system(f"cp src/smart/model/vqvae.py logs/{cfg.task_name}/")
    os.system(f"cp src/smart/model/basic_vae.py logs/{cfg.task_name}/")
    os.system(f"cp src/smart/model/var.py logs/{cfg.task_name}/")

    run(cfg)  # train/val/test the model

    log.info("Closing wandb!")
    wandb.finish()
    log.info(f"Output dir: {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
    log.info("run.py DONE!!!")
