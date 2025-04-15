from typing import Dict, List, Optional, Union

import json
from src.smart.minigrid.maze_loader import MazeTrajectoryDataset
from src.smart.model.vqvae import VQVAE
from src.smart.model.var import VAR
import torch.nn.functional as F
import torch.nn as nn
import os

import glob

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from src.smart.minigrid.utils import (Config, load_config)

import wandb

from src.smart.utils import (
    angle_between_2d_vectors,
    sample_next_token_traj,
    transform_to_global,
    weight_init,
    wrap_angle,
)

from tqdm import tqdm

class NullCtx:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class MazeVQVAEVARTrainer(nn.Module):
    def __init__(self, n_vq_emb=32, # vqvae quantized vector emb size
                 vq_vocab_size= 4096, # vqvae quantized vector latent vocab size
                 vqvae_ch=160, traj_per_epoch=10000, batch_size=4, n_token_agent=108, #n_token_agent=11664, 
                #  v_patch_nums=[1, 2, 3, 5, 9, 16, 18],
                 v_patch_nums=[1, 2, 3, 5, 8],
                 var_precision: str = "bfloat16", # float16 or bfloat16 or None
                 ckpt_dir="./output_minigrid/model1", # checkpoint directory
                 per_point_dim=2, # 2D point
                 name='model1',
                 target_traj=True,
                 lr= 8e-5,
                 scheduler_step=1000,
                 ):
        super(MazeVQVAEVARTrainer, self).__init__()
                
        self.vqvae_ch = vqvae_ch
        self.v_patch_nums = v_patch_nums
        self.target_traj = target_traj
        # make n_points_per_level into [(i, per_point_dim) for i in n_points_per_level]
        if self.target_traj == 'traj':
            self.n_points_per_level = [(i, per_point_dim) for i in v_patch_nums]
        elif self.target_traj == 'map':
            self.n_points_per_level = [(i, i) for i in v_patch_nums]
        else:
            raise ValueError("target_traj must be 'traj' or 'map'")
        print('n_points_per_level:', self.n_points_per_level)
        
        self.vq_vocab_size = vq_vocab_size
        self.n_vq_emb = n_vq_emb
        self.n_token_agent = n_token_agent
        self.device = device
        lr = lr
        
        
        # Dataset
        self.dataset = MazeTrajectoryDataset(num_trajectories=traj_per_epoch, num_points=v_patch_nums[-1])
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        if wandb.run is not None:
            wandb.init(project="maze-vqvae-var", 
                       name=name, 
                       config={
                "vq_vocab_size": self.vq_vocab_size,
                "n_token_agent": self.n_token_agent,
                "batch_size": self.dataloader.batch_size,
                "patch_nums": self.v_patch_nums,
                "n_points_per_level": self.n_points_per_level,
                "lr": lr
            })
        
        build_vqvae = True
        build_var = True
        
        self.vae = self.quantize = None
        if build_vqvae:
            self.use_xy_as_output = True
            if self.use_xy_as_output:
                in_out_emb_channels = 1
            else:
                in_out_emb_channels = n_token_agent
            
            using_znorm = True # True cosine similarity
            # self.quantize: VectorQuantizer2 = VectorQuantizer2(
            #     vq_vocab_size=self.vq_vocab_size, Cvae=self.n_vq_emb, using_znorm=using_znorm, beta=self.beta,
            #     default_qresi_counts=default_qresi_counts, v_patch_nums=self.n_points_per_level, 
            #     quant_resi=quant_resi, share_quant_resi=share_quant_resi,
            # )

            share_quant_resi=4     # use 4 \phi layers for K scales: partially-shared \phi
            self.vae = VQVAE(vocab_size=self.vq_vocab_size, z_channels=self.n_vq_emb, ch=self.vqvae_ch, test_mode=False, 
                            share_quant_resi=share_quant_resi, v_patch_nums=self.n_points_per_level, #v_patch_nums=v_patch_nums,
                            using_znorm=using_znorm, coder_in_channels=in_out_emb_channels,
                            coder_ch_mult=(1, 1, 2, 2, 4), W=-1, #W=per_point_dim
                            ) #.half()
            self.quantize = self.vae.quantize

        self.apply(weight_init)

        self.var_wo_ddp = None
        if build_var:
            depth = 16
            heads = depth
            width = depth * 64
            dpr = 0.1 * depth/24
            self.var_wo_ddp = VAR(
                vae_local=self.vae,
                num_classes=self.n_token_agent, depth=16, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., 
                drop_path_rate=dpr, norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=True, patch_nums=self.n_points_per_level,
                flash_if_available=False, fused_if_available=False,
            ) #.half() #.to(device)
            self.var_wo_ddp.init_weights(init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1)

            # VAR training
            self.prog_it = 0
            self.last_prog_si = -1
            self.first_prog = True
            self.loss_weight = torch.ones(1, self.var_wo_ddp.L, device=self.device) / self.var_wo_ddp.L
            self.train_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
            
        self.amp_ctx = NullCtx()
        assert var_precision in (None, 'float16', 'bfloat16'), var_precision
        if var_precision is not None or var_precision == "float16" or var_precision == "bfloat16":        
            self.amp_ctx = torch.autocast('cuda', enabled=True, 
                                          dtype=torch.float16 if var_precision == "float16" else torch.bfloat16, 
                                          cache_enabled=True)
            
            

        self.ckpt_dir = ckpt_dir
        
        self.vae.to(self.device)
        self.var_wo_ddp.to(self.device)
        self.optimizer = torch.optim.Adam(list(self.vae.parameters()) + list(self.var_wo_ddp.parameters()), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=0.1)
        
    def get_label(self, maze, start, goal):
        grid_H, grid_W = maze.shape[-2:]      # Height and width of the maze grid
        if self.target_traj == 'traj':
            label_B = (start[:, 0] * grid_W + start[:, 1]) * (grid_H * grid_W) + (goal[:, 0] * grid_W + goal[:, 1])
        elif self.target_traj == 'map':
            label_B = (goal[:, 0] * grid_W + goal[:, 1])
        else:
            raise ValueError("target_traj must be 'traj' or 'map'")
        return label_B

    def train_epoch(self):
        self.vae.train()
        self.var_wo_ddp.train()
        
        for batch in self.dataloader:
            maze = self.dataset[0]["maze"]  # Pick one example
            # free_mask = (maze == 0)
            # N = free_mask.sum().item()
            # max_vocab_size = N * N
            grid_H, grid_W = maze.shape[-2:]
            # max_vocab_size = grid_H * grid_W * grid_H * grid_W
            # print(f"Max vocab size needed for unique (start, goal): {max_vocab_size}")
            max_vocab_size = grid_H * grid_W
            print(f"Max vocab size needed for unique (goal): {max_vocab_size}")
            
            maze_img = batch["maze"].unsqueeze(1).float().to(self.device)  # [B, 1, H, W]
                      
            traj_BCT2 = batch["trajectory"].to(self.device)  # [B, T, 2]
            traj_BCT2 = traj_BCT2.unsqueeze(1) # [B, 1, T, 2]
            
            if self.target_traj == 'traj':
                f_BCt2 = traj_BCT2
            elif self.target_traj == 'map':
                f_BCt2 = maze_img
            else:
                raise ValueError("target_traj must be 'traj' or 'map'")

            # VQVAE forward
            result = self.vae(f_BCt2)
            recon = result["f_BCt2_reconstructed"]  # [B, C, T, 2]

            mse_loss = F.mse_loss(recon, f_BCt2, reduction="mean")
            vq_loss = result["loss_commitment_dictionary"]

            # VAR forward
            gt_idx_Bl: List[torch.LongTensor] = self.vae.img_to_idxBl(
                f_BCt2, v_patch_nums=self.var_wo_ddp.patch_nums) # List[B, patch_h*patch_w] codebook indices, multi-scale tokens R
            # print('gt_idx_Bl: size', len(gt_idx_Bl), 'patch', gt_idx_Bl[0].shape)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)  # (B, L), ground truth quantized indices for the input image batch
            x_BLCv_wo_first_l: torch.Tensor = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl) # (B, L, Cv), quantized indices to var input
            
            prog_si=-1
            prog_wp_it=20
            prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
            if prog_si == len(self.var_wo_ddp.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
            
            # Create unique label for each (start, goal) pair as int
            start = batch["start"].to(self.device)  # [B, 2]
            goal = batch["goal"].to(self.device)    # [B, 2]
            maze = batch["maze"].to(self.device)  # [B, H, W]
            label_B = self.get_label(maze, start, goal)  # [B]
            B, V = label_B.shape[0], self.vae.vocab_size
            # print('label_B: ', max(label_B), 'V:', V)
            assert max(label_B) < self.n_token_agent, "label_B {} >= n_token_agent {}".format(max(label_B), self.n_token_agent)
            
            with torch.autocast('cuda', enabled=True): #, dtype=torch.float16):
                # with self.amp_ctx:
                self.var_wo_ddp.forward # 2. Train transformer to predict tokens
                logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l) # (B, L, V), logits for the input image batch, V is the vocab size
                assert not torch.isnan(logits_BLV).any(), logits_BLV
                loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1) # (B, L). logits shape is (B, L, V), gt shape is (B, L)
                if prog_si >= 0:    # in progressive training - start with coarse scales
                    bg, ed = self.vae.begin_ends[prog_si]
                    assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                    lw = self.loss_weight[:, :ed].clone()
                    lw[:, bg:ed] *= min(max(prog_wp, 0), 1)  # Gradually increase weight for finer scales
                else:               # not in progressive training
                    lw = self.loss_weight
                var_loss = loss.mul(lw).sum(dim=-1).mean()
            

            total_loss = mse_loss + vq_loss + var_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            print(f"Loss: {total_loss.item():.4f} | VQ: {vq_loss.item():.4f} | VAR: {var_loss.item():.4f}")
            
            return total_loss, vq_loss, var_loss

    def save_checkpoint(self, epoch):
        torch.save({
            'vae': self.vae.state_dict(),
            'var': self.var_wo_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }, os.path.join(self.ckpt_dir, f"epoch_{epoch:05d}.pt"))

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae'])
        self.var_wo_ddp.load_state_dict(checkpoint['var'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint from {path} at epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    
    def generate_and_plot(self, traj_np, maze_np, start, goal, label_B, groundtruth_traj, title=''):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Cannot Generate inferece plot. "
                  "Matplotlib is not installed. Please install it to plot the generated trajectory.")
            return
        plt.figure()
        plt.imshow(maze_np.T, origin='lower', cmap='gray_r')
        if self.target_traj == 'traj':
            plt.plot(traj_np[:, 0], traj_np[:, 1], marker='o', color='blue', label='Generated')
        plt.plot(groundtruth_traj[:, 0], groundtruth_traj[:, 1], marker='o', color='red', label='Ground Truth')
        if start is not None:
            plt.plot(start[0], start[1], marker='s', color='green', markersize=10, label='Start')
        if goal is not None:
            plt.plot(goal[0], goal[1], marker='*', color='yellow', markersize=12, label='Goal')
        plt.legend()
        plt.title(title)
        plt.grid(True)
        path = os.path.join(self.ckpt_dir, f"generated_label_{label_B}_{title}.png")
        plt.savefig(path)
        print("Saved generated trajectory plot to", path)
        plt.close()
        
    def test_inference(self, num_samples=5, ckpt_filename=""):
        for i, batch in enumerate(self.dataloader):
            if i >= num_samples:
                break
            maze = batch["maze"]#[0].to(self.device)
            start = batch["start"]#[0].tolist()
            goal = batch["goal"]#[0].tolist()
            groundtruth_traj = batch["trajectory"].cpu().numpy()
            print('maze:', maze.shape, 'start:', start, 'goal:', goal)
            label_B = self.get_label(maze, start, goal)  # [B]
            print('test label_B:', label_B)
            
            with torch.no_grad():
                traj_recon = self.var_wo_ddp.autoregressive_infer_cfg(
                    B=1,
                    label_B=torch.tensor([label_B], device=self.device),
                    top_k=5,
                    top_p=0.95,
                    cfg=1.0
                )  # [1, 1, T, 2]
                print('traj_recon:', traj_recon.shape)
                traj_np = traj_recon[0, 0].cpu().numpy()
                print(traj_np.shape)
                maze_np = maze.cpu().numpy() if torch.is_tensor(maze) else maze
            
            if self.target_traj == 'traj':
                self.generate_and_plot(traj_np, maze_np[0], start[0], goal[0], label_B[0], 
                                       groundtruth_traj[0], title='VAR Generated Trajectory_' + ckpt_filename)   
            else:
                self.generate_and_plot(traj_np, maze_np[0], start[0], goal[0], label_B[0], 
                                       groundtruth_traj[0], title='Groundtruth_maze_' + ckpt_filename)
                self.generate_and_plot(traj_np, traj_np, start[0], goal[0], label_B[0], 
                                       groundtruth_traj[0], title='Generated_maze_' + ckpt_filename)


    def train(self, epochs=100, resume_path=None):
        start_epoch = 0
        if resume_path:
            print("Resuming Training from checkpoint:", resume_path)
            start_epoch = self.load_checkpoint(resume_path) + 1

        epoch = None
        for epoch in range(start_epoch, epochs):
            print(f"=== Epoch {epoch + 1} ===")
            total_loss, vq_loss, var_loss = self.train_epoch()
            if epoch % 250 == 0:
                self.save_checkpoint(epoch)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Current LR: {current_lr}")
            
            if wandb.run is not None:
                wandb.log({
                    "loss/total": total_loss.item(),
                    "loss/vq": vq_loss.item(),
                    "loss/var": var_loss.item(),
                    "lr": current_lr
                })
                
        if epoch:
            # save last checkpoint
            self.save_checkpoint(epoch)
            
            
def get_latest_checkpoint(ckpt_dir):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found.")
    # return max(ckpts, key=os.path.getctime)
    return ckpts[-1]  # Return second to last checkpoint


if __name__ == "__main__":
    

        
    # Create a dict for config and save to disk
    cfg = load_config("src/smart/minigrid/config.yaml")
    name = cfg.model.name
    
    ckpt_dir= os.path.join("./output_minigrid/", name)
    resume_path = None
    try:
        resume_path = get_latest_checkpoint(ckpt_dir)
    except FileNotFoundError:
        print("No checkpoints found, starting from scratch.")
        
    cfg.model.ckpt_dir = ckpt_dir
    cfg.trainer.resume_path = resume_path
    cfg.model.lr = float(cfg.model.lr)
    
    print("Config:", cfg)
    # name = '07_classify_NoGoal_YesStart_maze'
    
        
    # make ckpt_dir if does not exist
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        os.system(f"cp src/smart/minigrid/train_vqvae_var.py ./{ckpt_dir}/")
        os.system(f"cp src/smart/minigrid/maze_loader.py ./{ckpt_dir}/")
        cfg.save(f"{ckpt_dir}/config.yaml")
        
    print('Running name', name)
    
    # %%
    os.environ["WANDB_DISABLED"] = "false"
    trainer = MazeVQVAEVARTrainer(**cfg.model.to_dict())
    trainer.train(**cfg.trainer.to_dict())
    
    
    # %%
    # ckpt_dir="./output_minigrid/01_regression/"
    os.environ["WANDB_DISABLED"] = "true"
    cfg.model.batch_size = 1
    trainer = MazeVQVAEVARTrainer(**cfg.model.to_dict(),
        # ckpt_dir=ckpt_dir,  # batch_size=1
        )
    
    # Load latest checkpoint
    latest_ckpt = get_latest_checkpoint(trainer.ckpt_dir)
    print('Loading latest_ckpt:', latest_ckpt)
    trainer.load_checkpoint(latest_ckpt)
    
    # get just the filename
    ckpt_filename = os.path.basename(latest_ckpt)

    for _ in range(5):
        # Run test-time inference and plot
        trainer.test_inference(num_samples=1, ckpt_filename=ckpt_filename)
# %%
