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

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph
from torch.nn import functional as F

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    sample_next_token_traj,
    transform_to_global,
    weight_init,
    wrap_angle,
)

from typing import Tuple
import torch.nn as nn
import torch.cuda.amp as amp

from src.smart.model.quant import VectorQuantizer2
from src.smart.model.var import VAR
from src.smart.model.vqvae import VQVAE

class NullCtx:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    

def build_vae_var(
    # Shared args
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # VAR args
    num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
) -> Tuple[VQVAE, VAR]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums) #.to(device)
    var_wo_ddp = VAR(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ) #.to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp


class SMARTAgentDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int, # num_agent_layers = 6
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int, 
        n_points_per_level: List[int],
        build_vqvae: bool,
        build_var: bool,
        n_vq_emb: int,  # vqvae quantized vector emb size
        vq_vocab_size: int, # vqvae quantized vector latent vocab size
        var_precision: str = "bfloat16", # float16 or bfloat16 or None
        finetune_vqvae: bool = True, # fintune vqvae during var training. (SMART is still frozen)
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps # hk: 11
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.shift = 5
        self.hist_drop_prob = hist_drop_prob
        self.n_token_agent = n_token_agent # 2048, number of classes of agent tokens (representing relative positiions)

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.vq_vocab_size = vq_vocab_size #2048 # number of next-scale cookbook embeddings
        # assert n_token_agent == self.vq_vocab_size, "{n_token_agent} != {self.vq_vocab_size}"
        self.use_xy_as_output = False
        if self.use_xy_as_output:
            in_out_emb_channels = 1
        else:
            in_out_emb_channels = n_token_agent

        self.n_vq_emb = Cvae = n_vq_emb # 32 #1024 # each single cookbook vocab embedding dim for VQ-VAE. VQ-VAE uses 32 as default.
        # self.vqvae_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=in_out_emb_channels, out_channels=self.n_vq_emb, kernel_size=3, stride=1, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=self.n_vq_emb, out_channels=self.n_vq_emb, kernel_size=3, stride=1, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=self.n_vq_emb, out_channels=self.n_vq_emb, kernel_size=3, stride=1, padding='same')
        # )
        # self.vqvae_decoder = nn.Sequential(
        #     nn.Conv2d(in_channels=self.n_vq_emb, out_channels=self.n_vq_emb, kernel_size=3, stride=1, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=self.n_vq_emb, out_channels=self.n_vq_emb, kernel_size=3, stride=1, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=self.n_vq_emb, out_channels=in_out_emb_channels, kernel_size=3, stride=1, padding='same')
        # )

        # self.embedding = nn.Embedding(self.vq_vocab_size, self.n_vq_emb) # codebook/lookup table. Output: Each row of the embedding matrix represents a codebook embedding.
        # self.quant_resi_ratio = quant_resi = 0.5
        self.beta=0.25         # commitment loss weight
        quant_resi=0.5         # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4     # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0 # if is 0: automatically set to len(v_patch_nums)
        # if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
        #     self.quant_resi = PhiNonShared([
        #         (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) 
        #         for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        # elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
        #     self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        # else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
        #     self.quant_resi = PhiPartiallyShared(nn.ModuleList([
        #         (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) 
        #         for _ in range(share_quant_resi)]))
            
        self.n_points_per_level = n_points_per_level
        
        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pt2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.token_emb_veh = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_ped = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_cyc = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.fusion_emb = MLPEmbedding(
            input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim
        )

        self.t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
        )

        self.vae = self.quantize = None
        if build_vqvae:
            using_znorm = False
            # self.quantize: VectorQuantizer2 = VectorQuantizer2(
            #     vq_vocab_size=self.vq_vocab_size, Cvae=self.n_vq_emb, using_znorm=using_znorm, beta=self.beta,
            #     default_qresi_counts=default_qresi_counts, v_patch_nums=self.n_points_per_level, 
            #     quant_resi=quant_resi, share_quant_resi=share_quant_resi,
            # )

            ch=160
            self.vae = VQVAE(vocab_size=self.vq_vocab_size, z_channels=self.n_vq_emb, ch=ch, test_mode=False, 
                            share_quant_resi=share_quant_resi, v_patch_nums=self.n_points_per_level,
                            using_znorm=using_znorm, coder_in_channels=in_out_emb_channels,
                            coder_ch_mult=(1, 1, 2, 2, 4)) #.half()
            self.quantize = self.vae.quantize

        self.apply(weight_init)

        # self.vae, self.var = build_vae_var(
        #     V=self.vq_vocab_size, Cvae=self.n_vq_emb, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        #     patch_nums=self.n_points_per_level,
        #     num_classes=self.n_token_agent, depth=16, shared_aln=False,
        # )
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
                flash_if_available=True, fused_if_available=True,
            ) #.half() #.to(device)
            self.var_wo_ddp.init_weights(init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1)

            # VAR training
            self.prog_it = 0
            self.last_prog_si = -1
            self.first_prog = True
            self.loss_weight = torch.ones(1, self.var_wo_ddp.L, device='cuda') / self.var_wo_ddp.L
            self.train_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
            
        self.amp_ctx = NullCtx()
        assert var_precision in (None, 'float16', 'bfloat16'), var_precision
        if var_precision is not None or var_precision == "float16" or var_precision == "bfloat16":        
            self.amp_ctx = torch.autocast('cuda', enabled=True, dtype=torch.float16 if var_precision == "float16" else torch.bfloat16, cache_enabled=True)
            
        self.finetune_vqvae = finetune_vqvae


    def agent_token_embedding(
        self,
        agent_token_index,  # [n_agent, n_step]
        trajectory_token_veh,  # [n_token, 8]
        trajectory_token_ped,  # [n_token, 8]
        trajectory_token_cyc,  # [n_token, 8]
        pos_a,  # [n_agent, n_step, 2]
        head_vector_a,  # [n_agent, n_step, 2]
        agent_type,  # [n_agent]
        agent_shape,  # [n_agent, 3]
        inference=False,
    ):
        n_agent, n_step, traj_dim = pos_a.shape
        _device = pos_a.device

        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2
        #  [n_token, hidden_dim]
        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh)
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped)
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc)
        agent_token_emb = torch.zeros(
            (n_agent, n_step, self.hidden_dim), device=_device, dtype=pos_a.dtype
        )
        agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]].to(agent_token_emb)
        agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]].to(agent_token_emb)
        agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]].to(agent_token_emb)

        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(agent_token_index.shape[0], 1, traj_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # [n_agent, n_step, 2]
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )  # [n_agent, n_step, 2]
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]  # List of len=2, shape [n_agent, hidden_dim]

        x_a = self.x_a_emb( # FourierEmbedding cat with categorical_embs
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],
        )  # [n_agent*n_step, hidden_dim]
        x_a = x_a.view(-1, n_step, self.hidden_dim)  # [n_agent, n_step, hidden_dim]

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)  # mlp

        if inference:
            return (
                feat_a,  # [n_agent, n_step, hidden_dim]
                agent_token_emb,  # [n_agent, n_step, hidden_dim]
                agent_token_emb_veh,  # [n_agent, hidden_dim]
                agent_token_emb_ped,  # [n_agent, hidden_dim]
                agent_token_emb_cyc,  # [n_agent, hidden_dim]
                veh_mask,  # [n_agent]
                ped_mask,  # [n_agent]
                cyc_mask,  # [n_agent]
                categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
            )
        else:
            return feat_a  # [n_agent, n_step, hidden_dim]

    def build_temporal_edge(
        self,
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2],
        mask,  # [n_agent, n_step]
        inference_mask=None,  # [n_agent, n_step]
    ):
        pos_t = pos_a.flatten(0, 1) # [n_agent*n_step, 2]
        head_t = head_a.flatten(0, 1) # [n_agent*n_step]
        head_vector_t = head_vector_a.flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & _mask_keep

        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1) # [n_agent, n_step, n_step] <- [a, b, 1] & [a, 1, c] = [a, b, c]
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]  # [2, 228], [0]: [(source, target), num_edges] 
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]  # edges flow forward in time, no "backward" temporal dependencies
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift  # (self.time_span / self.shift) = 6 = 30/5 ~ num_historical_steps/5?
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]] # [n_agent*n_step, 2]
        rel_pos_t = rel_pos_t[:, :2]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t, p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t
                ),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(
        self,
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        batch_s,  # [n_agent*n_step]
        mask,  # [n_agent, n_step]
    ):
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(
        self,
        pos_pl,  # [n_pl, 2]
        orient_pl,  # [n_pl]
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        mask,  # [n_agent, n_step]
        batch_s,  # [n_agent*n_step]
        batch_pl,  # [n_pl*n_step]
    ):
        n_step = pos_a.shape[1]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = pos_pl.repeat(n_step, 1)
        orient_pl = orient_pl.repeat(n_step)
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s, # hk: batch values y query for x are [0, 0, 1, 1, 2, 2, ...] where search is only within the same indices
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def forward(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mask = tokenized_agent["valid_mask"] # hk: [n_agent, n_step], n_step=18
        pos_a = tokenized_agent["sampled_pos"] # hk: [n_agent, n_step, 2], n_step=18
        head_a = tokenized_agent["sampled_heading"] # hk: [n_agent, n_step], n_step=18
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1) # hk: [n_agent, n_step, 2], n_step=18
        n_agent, n_step = head_a.shape

        # ! get agent token embeddings
        feat_a = self.agent_token_embedding(
            agent_token_index=tokenized_agent["sampled_idx"],  # [n_ag, n_step]
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
        )  # feat_a: [n_agent, n_step, hidden_dim]

        # ! build temporal, interaction and map2agent edges
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_t: [2, n_edge_t], r_t: [n_edge_t, hidden_dim]

        batch_s = torch.cat(
            [
                tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_agent*n_step]
        batch_pl = torch.cat(
            [
                map_feature["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_pl*n_step]

        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            batch_s=batch_s,  # [n_agent*n_step]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],  # [n_pl, 2]
            orient_pl=map_feature["orientation"],  # [n_pl]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
            batch_s=batch_s,  # [n_agent*n_step]
            batch_pl=batch_pl,  # [n_pl*n_step]
        )

        # ! attention layers
        # [n_step*n_pl, hidden_dim]
        feat_map = (
            map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
        )

        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)  # [n_agent*n_step, hidden_dim]
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t) # [n_step*n_agent, hidden_dim]
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[i](
                (feat_map, feat_a), r_pl2a, edge_index_pl2a  # [((k,v),q), r, edge_index]
            )
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)

        # ! final mlp to get outputs
        next_token_logits = self.token_predict_head(feat_a) # [n_agent, n_step, n_token], n_step=18

        return {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": next_token_logits[:, 1:-1],  # [n_agent, 16, n_token]   # len(range(10, 90, 5))=16  # start=10, end=90 (exclusive), step=5
            "next_token_valid": tokenized_agent["valid_mask"][:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": tokenized_agent["sampled_pos"],  # [n_agent, 18, 2]
            "pred_head": tokenized_agent["sampled_heading"],  # [n_agent, 18]
            "pred_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
        }

    def inference(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, torch.Tensor]:
        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_step_future_10hz = self.num_future_steps  # 80
        n_step_future_2hz = n_step_future_10hz // self.shift  # 16 = 80 // 5
        step_current_10hz = self.num_historical_steps - 1  # 10
        step_current_2hz = step_current_10hz // self.shift  # 2 = 10 // 5

        pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone() # hk: [n_agent, n_steps=18, 2] --> [n_agent, 2, 2]. Current? [0, 5]
        head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pred_idx = tokenized_agent["gt_idx"].clone()
        (
            feat_a,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb_veh,  # [n_agent, hidden_dim]
            agent_token_emb_ped,  # [n_agent, hidden_dim]
            agent_token_emb_cyc,  # [n_agent, hidden_dim]
            veh_mask,  # [n_agent]
            ped_mask,  # [n_agent]
            cyc_mask,  # [n_agent]
            categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
        ) = self.agent_token_embedding(
            agent_token_index=tokenized_agent["gt_idx"][:, :step_current_2hz],
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
            inference=True,
        )

        if not self.training:
            pred_traj_10hz = torch.zeros(
                [n_agent, n_step_future_10hz, 2], dtype=pos_a.dtype, device=pos_a.device
            ) # hk: [n_agent, 80, 2]
            pred_head_10hz = torch.zeros(
                [n_agent, n_step_future_10hz], dtype=pos_a.dtype, device=pos_a.device
            ) # hk: [n_agent, 80]

        pred_valid = tokenized_agent["valid_mask"].clone() # hk: [n_agent, 18]
        next_token_logits_list = []
        next_token_action_list = []
        feat_a_t_dict = {}
        for t in range(n_step_future_2hz):  # 0 -> 15
            t_now = step_current_2hz - 1 + t  # 1 -> 16
            n_step = t_now + 1  # 2 -> 17
            # print('t_now:', t_now, 'n_step:', n_step, "pos_a", pos_a.shape, "head_a", head_a.shape, "head_vector_a", head_vector_a.shape, "pred_valid", pred_valid.shape)
            # print('tokenized_agent["num_graphs"]', tokenized_agent["num_graphs"])
            # print('tokenized_agent["batch"]', tokenized_agent["batch"])
            if t == 0:  # init
                hist_step = step_current_2hz # 2
                batch_s = torch.cat(
                    [
                        tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t # hk: [n_agent] + 2 * t;  tokenized_agent["batch"] has values [0, 1]?
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                batch_pl = torch.cat(
                    [
                        map_feature["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                inference_mask = pred_valid[:, :n_step]
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a, # hk: [n_agent, n_step, 2], n_step grows from 2 to 17
                    head_a=head_a, # hk: [n_agent, n_step], n_step grows from 2 to 17
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                )
            else:
                hist_step = 1 # hk: 1 because we only infer the current step
                batch_s = tokenized_agent["batch"] # hk: [n_agent]
                batch_pl = map_feature["batch"] # hk: [n_pl]
                inference_mask = pred_valid[:, :n_step].clone()
                inference_mask[:, :-1] = False
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                    inference_mask=inference_mask,
                )
                edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1
            # print('batch_s', batch_s.shape, batch_s)
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                pos_pl=map_feature["position"],  # [n_pl, 2]
                orient_pl=map_feature["orientation"],  # [n_pl]
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
                batch_s=batch_s,  # [n_agent*hist_step], hist_step=1 or 2
                batch_pl=batch_pl,  # [n_pl*hist_step]
            )
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                batch_s=batch_s,  # [n_agent*hist_step]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
            ) # hk: [2, n_edge_a2a], [n_edge_a2a, hidden_dim]

            # ! attention layers
            for i in range(self.num_layers):
                # [n_agent, n_step, hidden_dim]
                _feat_temporal = feat_a if i == 0 else feat_a_t_dict[i] # # hk: feat_a: [n_agent, n_step=step_current_2hz=2, hidden_dim]
                    # hk:feat_a_t_dict[i]: [n_agent, n_step, hidden_dim], t=0, n_step=2; t>0, n_step=2+t, until n_step=17. (2 historical + 15 predicted steps)
                # print('t', t, 'i', i); print('_feat_temporal', _feat_temporal.shape); print('feat_a_t_dict', feat_a_t_dict[i].shape) if i > 1 else None
                if t == 0:  # init, process hist_step together
                    _feat_temporal = self.t_attn_layers[i](
                        _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    ).view(n_agent, n_step, -1) # hk: [n_agent, n_step=2, hidden_dim]
                    _feat_temporal = _feat_temporal.transpose(0, 1).flatten(0, 1) #  hk: [n_step*n_agent, hidden_dim]

                    # [hist_step*n_pl, hidden_dim]
                    _feat_map = (
                        map_feature["pt_token"]
                        .unsqueeze(0)
                        .expand(hist_step, -1, -1)
                        .flatten(0, 1)
                    )

                    _feat_temporal = self.pt2a_attn_layers[i](
                        (_feat_map, _feat_temporal), r_pl2a, edge_index_pl2a # [((k,v),q), r, edge_index]; [((src=(k,v), dst=q), pos_emb, edge_index]
                    ) # hk: [n_step*n_agent, hidden_dim]
                    _feat_temporal = self.a2a_attn_layers[i](
                        _feat_temporal, r_a2a, edge_index_a2a
                    )
                    _feat_temporal = _feat_temporal.view(n_step, n_agent, -1).transpose(
                        0, 1
                    ) # hk: [n_agent, n_step=2, hidden_dim]
                    feat_a_now = _feat_temporal[:, -1]  # [n_agent, hidden_dim], hk: -1 means the last step, current step

                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = _feat_temporal

                else:  # process one step
                    feat_a_now = self.t_attn_layers[i](  # hk: [n_agent, hidden_dim], where n_step=-1 means the last step, current step
                        (_feat_temporal.flatten(0, 1), _feat_temporal[:, -1]), # hk: [n_step*n_agent, hidden_dim] and [n_agent, hidden_dim]
                        r_t,
                        edge_index_t,
                    )   
                    # * give same results as below, but more efficient
                    # feat_a_now = self.t_attn_layers[i](
                    #     _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    # ).view(n_agent, n_step, -1)[:, -1]

                    feat_a_now = self.pt2a_attn_layers[i](
                        (map_feature["pt_token"], feat_a_now), r_pl2a, edge_index_pl2a
                    ) # hk: [n_agent, hidden_dim]
                    feat_a_now = self.a2a_attn_layers[i](
                        feat_a_now, r_a2a, edge_index_a2a
                    ) # hk: [n_agent, hidden_dim]

                    # [n_agent, n_step, hidden_dim]
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = torch.cat(
                            (feat_a_t_dict[i + 1], feat_a_now.unsqueeze(1)), dim=1
                        ) #hk: feat_a_t_dict: [n_agent, n_step=T-1, hidden_dim], feat_a_now: [n_agent, 1, hidden_dim]. [A, T-1, D] and [A, 1, D] -> [A, T, D]
                        # print('t', t, 'i', i, 'feat_a_now', feat_a_now.shape, 'feat_a_t_dict', feat_a_t_dict[i + 1].shape) 
            # import pdb; pdb.set_trace()
            # ! get outputs
            next_token_logits = self.token_predict_head(feat_a_now) # hk: [n_agent, n_token], feat_a_now: [n_agent, hidden_dim]
            next_token_logits_list.append(next_token_logits)  # [n_agent, n_token]

            next_token_idx, next_token_traj_all = sample_next_token_traj(
                token_traj=tokenized_agent["token_traj"], # [n_agent, n_token, 4, 2]
                token_traj_all=tokenized_agent["token_traj_all"], # [n_agent, n_token, 6, 4, 2]
                sampling_scheme=sampling_scheme,
                # ! for most-likely sampling
                next_token_logits=next_token_logits,
                # ! for nearest-pos sampling
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
                pos_next_gt=tokenized_agent["gt_pos_raw"][:, n_step],  # [n_agent, 2]
                head_next_gt=tokenized_agent["gt_head_raw"][:, n_step],  # [n_agent]
                valid_next_gt=tokenized_agent["gt_valid_raw"][:, n_step],  # [n_agent]
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_token, 2]
            )  # next_token_idx: [n_agent], next_token_traj_all: [n_agent, 6, 4, 2]

            diff_xy = next_token_traj_all[:, -1, 0] - next_token_traj_all[:, -1, 3]
            next_token_action_list.append(
                torch.cat(
                    [
                        next_token_traj_all[:, -1].mean(1),  # [n_agent, 2]
                        torch.arctan2(diff_xy[:, [1]], diff_xy[:, [0]]),  # [n_agent, 1]
                    ],
                    dim=-1,
                )  # [n_agent, 3]
            )

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
            )[0].view(*next_token_traj_all.shape) # [n_agent, 6, 4, 2]

            if not self.training:
                pred_traj_10hz[:, t * 5 : (t + 1) * 5] = token_traj_global[:, 1:].mean(
                    2 # reduce mean over 4 box corners
                ) # [n_agent, 5, 2]
                diff_xy = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head_10hz[:, t * 5 : (t + 1) * 5] = torch.arctan2(
                    diff_xy[:, :, 1], diff_xy[:, :, 0]
                )

            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
            pred_idx[:, n_step] = next_token_idx

            # ! update tensors for for next step
            pred_valid[:, n_step] = pred_valid[:, t_now]
            # pred_valid[:, n_step] = pred_valid[:, t_now] | mask_spawn
            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
            head_vector_a_next = torch.stack(
                [head_a_next.cos(), head_a_next.sin()], dim=-1
            )
            head_vector_a = torch.cat(
                [head_vector_a, head_vector_a_next.unsqueeze(1)], dim=1
            )

            # ! get agent_token_emb_next
            agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
            agent_token_emb_next[veh_mask] = agent_token_emb_veh[
                next_token_idx[veh_mask]
            ]
            agent_token_emb_next[ped_mask] = agent_token_emb_ped[
                next_token_idx[ped_mask]
            ]
            agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[
                next_token_idx[cyc_mask]
            ]
            agent_token_emb = torch.cat(
                [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
            )

            # ! get feat_a_next
            motion_vector_a = pos_a[:, -1] - pos_a[:, -2]  # [n_agent, 2]
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a[:, -1], nbr_vector=motion_vector_a
                    ),
                ],
                dim=-1,
            )
            # [n_agent, hidden_dim]
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            # [n_agent, 1, 2*hidden_dim]
            feat_a_next = torch.cat((agent_token_emb_next, x_a), dim=-1).unsqueeze(1)
            feat_a_next = self.fusion_emb(feat_a_next)
            feat_a = torch.cat([feat_a, feat_a_next], dim=1)

        out_dict = {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": torch.stack(next_token_logits_list, dim=1), # hk: [n_agent, 16, n_token]
            "next_token_valid": pred_valid[:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)] # hk: len(list(range(5, 95, 5)))=18
            "pred_pos": pos_a,  # [n_agent, 18, 2]
            "pred_head": head_a,  # [n_agent, 18]
            "pred_valid": pred_valid,  # [n_agent, 18]
            "pred_idx": pred_idx,  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for shifting proxy targets by lr
            "next_token_action": torch.stack(next_token_action_list, dim=1),
        }

        if not self.training:  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_traj_10hz"] = pred_traj_10hz # hk: [n_agent, 80, 2]
            out_dict["pred_head_10hz"] = pred_head_10hz # hk: [n_agent, 80]
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1]) # hk: [n_agent, 80]

        return out_dict
    
    def open_next_scale_smart(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        inference=False,
    ) -> Dict[str, torch.Tensor]:
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        n_agent, n_step = head_a.shape

        # ! get agent token embeddings
        agent_token_embed_out = self.agent_token_embedding(
            agent_token_index=tokenized_agent["sampled_idx"],  # [n_ag, n_step]
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
            inference=inference,
        )  # feat_a: [n_agent, n_step, hidden_dim]
        
        agent_token_emb = agent_token_emb_veh = agent_token_emb_ped = agent_token_emb_cyc = None
        veh_mask = ped_mask = cyc_mask = None
        categorical_embs = None
        if inference:
            (
                feat_a,  # [n_agent, step_current_2hz, hidden_dim]
                agent_token_emb,  # [n_agent, step_current_2hz, hidden_dim]
                agent_token_emb_veh,  # [n_agent, hidden_dim]
                agent_token_emb_ped,  # [n_agent, hidden_dim]
                agent_token_emb_cyc,  # [n_agent, hidden_dim]
                veh_mask,  # [n_agent]
                ped_mask,  # [n_agent]
                cyc_mask,  # [n_agent]
                categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
            ) = agent_token_embed_out
        else:
            feat_a = agent_token_embed_out

        # ! build temporal, interaction and map2agent edges
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_t: [2, n_edge_t], r_t: [n_edge_t, hidden_dim]

        batch_s = torch.cat(
            [
                tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_agent*n_step]
        batch_pl = torch.cat(
            [
                map_feature["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_pl*n_step]

        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            batch_s=batch_s,  # [n_agent*n_step]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],  # [n_pl, 2]
            orient_pl=map_feature["orientation"],  # [n_pl]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
            batch_s=batch_s,  # [n_agent*n_step]
            batch_pl=batch_pl,  # [n_pl*n_step]
        )

        # ! attention layers
        # [n_step*n_pl, hidden_dim]
        feat_map = (
            map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
        )

        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)  # [n_agent*n_step, hidden_dim]
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            # [n_step*n_agent, hidden_dim]
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[i](
                (feat_map, feat_a), r_pl2a, edge_index_pl2a  # [((k,v),q), r, edge_index]
            )
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a) 
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1) # [n_agent, n_step, hidden_dim]

        # ! final mlp to get outputs
        next_token_logits = self.token_predict_head(feat_a) # [n_agent, n_step, n_token], n_step=18
    
        return {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": next_token_logits[:, 1:-1],  # [n_agent, 16, n_token]   # len(range(10, 90, 5))=16  # start=10, end=90 (exclusive), step=5
            "next_token_logits_raw": next_token_logits, # [n_agent, 18, n_token]
            "next_token_valid": tokenized_agent["valid_mask"][:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": tokenized_agent["sampled_pos"],  # [n_agent, 18, 2]
            "pred_head": tokenized_agent["sampled_heading"],  # [n_agent, 18]
            "pred_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # inference
            "feat_a": feat_a,  # [n_agent, n_step, hidden_dim]
            "agent_token_emb": agent_token_emb,  # [n_agent, n_step, hidden_dim]
            "agent_token_emb_veh": agent_token_emb_veh,  # [n_agent, hidden_dim]
            "agent_token_emb_ped": agent_token_emb_ped,  # [n_agent, hidden_dim]
            "agent_token_emb_cyc": agent_token_emb_cyc,  # [n_agent, hidden_dim]
            "veh_mask": veh_mask,  # [n_agent]
            "ped_mask": ped_mask,  # [n_agent]
            "cyc_mask": cyc_mask,  # [n_agent]
            "categorical_embs": categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
        }
    
    
    def open_next_scale(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
        train_mask: torch.Tensor=None, # [n_agent]
        inference=False,
    ) -> Dict[str, torch.Tensor]:
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        n_agent, n_step = head_a.shape

        # ! get agent token embeddings
        enable_grad = not inference
        # enable_grad = True
        with torch.set_grad_enabled(enable_grad): # VAR mode, inference=True should be no grad, freeze smart
            smart_dict = self.open_next_scale_smart(tokenized_agent, map_feature, inference)
        next_token_logits = smart_dict["next_token_logits_raw"] # [n_agent, 18, n_token]

        n_agent, n_step, n_token = next_token_logits.shape
        
        if self.use_xy_as_output:
            # tokenized_agent["token_traj"] = [n_agent, n_token, 4, 2], n_token=2048
            # tokenized_agent["token_traj_all"] = [n_agent, n_token, 6, 4, 2], n_token=2048

            # topk_logits, topk_indices = torch.topk(
            #     next_token_logits, sampling_scheme.num_k, dim=-1, sorted=False
            # ) # [n_agent, n_step=18, num_k=1], [n_agent, n_step=18, num_k=1]

            # # Get token x, y from topk_indices
            token_traj = tokenized_agent["token_traj"] # [n_agent, n_token_vocab=2048, 4, 2]
            # range_a = torch.arange(next_token_logits.shape[0]).to(next_token_logits.device) # [n_agent]

            # token_world_sample = token_traj[range_a[:, None, None], topk_indices] # hk: [n_agent, n_step, num_k, 4, 2]

            #### soft indexing
            # Replace hard indexing with soft weights
            soft_weights = F.softmax(next_token_logits, dim=-1)  # Create soft attention weights, [n_agent, n_step, n_token] 

            # Use soft weights to combine token_traj
            # Flatten the trajectory data for batch matrix multiplication
            flattened_token_traj = token_traj.mean(dim=-2) # hk: [n_agent, n_token_vocab, 2]
            # flattened_token_traj = token_traj.view(n_agent, n_token, -1)  # Shape: [n_agent, n_token, 8]
            token_world_sample = torch.einsum('bst,btc->bsc', soft_weights, flattened_token_traj)  # Shape: [n_agent, n_step, 2]
            ####


            ### transform to global
            # token logit seems to be local
            # step_current_10hz = self.num_historical_steps - 1  # 10
            # step_current_2hz = step_current_10hz // self.shift  # 2 = 10 // 5
            # pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone() # hk: [n_agent, n_step=2, 2]. gt_pos: [n_agent, n_step=18, 2]
            # head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
            # t = 0
            # t_now = step_current_2hz - 1 + t  # 1 -> 16

            # token_world_sample = transform_to_global(
            #     pos_local=token_world_sample.flatten(1, 3),
            #     head_local=None,
            #     pos_now=pos_a[:, t_now],  # [n_agent, 2]
            #     head_now=head_a[:, t_now],  # [n_agent]
            # )[0].view(*token_world_sample.shape)
            ###

            # # compute valid mask if we want to compute the loss
            # gt_valid = tokenized_agent["valid_mask"] # [n_agent, n_step=18]
            # gt_last_valid = gt_valid.roll(shifts=-1, dims=1)  # [n_agent, 18]
            # gt_last_valid[:, -1:] = False  # [n_agent, 18]
            # pred_valid = tokenized_agent["valid_mask"]  # [n_agent, 18]
            # target_valid = pred_valid & gt_last_valid  # [n_agent, 18]
            # # truncate [(5->10), ..., (90->5)] to [(10->15), ..., (85->90)]
            # target_valid = target_valid[:, 1:-1]  # [n_agent, 16]
            # next_token_valid = tokenized_agent["valid_mask"][:, 1:-1]  # [n_agent, 16]
            # loss_weighting_mask = next_token_valid & target_valid
            # if not train_mask.all():
            #     loss_weighting_mask = loss_weighting_mask & train_mask[:, None] # [n_agent, 16]

            # # compute the center point from the 4 corner points of the box for token_world_sample
            # token_world_sample = token_world_sample.mean(dim=-2) # hk: [n_agent, n_step, num_k, 2]
            # assert token_world_sample.shape[2] == 1, "only supports num_k=1"
            # token_world_sample = token_world_sample[:, 1:-1] # hk: [n_agent, n_step=16, 2]
            # token_world_sample = token_world_sample.squeeze(2) # [n_agent, n_step, 2]
            f_BCt2 = token_world_sample[:, None, :, :] # [n_agent, n_emb=1, n_step, 2]
        else:
            # next_token_logits = [n_agent, n_step, n_token], n_step=18
            f_BCt2 = next_token_logits.permute(0, 2, 1)[:, :, :, None] # [n_agent, n_token, n_step, 1]
        ################# Algorithm 1 of VAR ################
        # # encoder, quantize, and decoder
        # f_BCt2_reconstructed, usages, mean_vq_loss = self.vae(f_BCt2, ret_usages=False)
        # f_hat = None
        # f_hat_per_level = None

        
        # -------------------------------------
        # # [n_agent, n_emb=1, n_step, 2] --> [n_agent, n_emb=self.n_vq_emb, n_step, 2] or 
        # # [n_agent, n_emb=n_token, n_step, 1] --> [n_agent, n_emb=self.n_vq_emb, n_step, 1]
        # f_BCt2 = self.vqvae_encoder(f_BCt2) 

        # --qunatize--

        #### Algorithm 2: Take R cookbook and decode back to the original space ####
        # Problem: gradient cannot flow through zeros.
        # f_hat_recons = idx_Bhw_list[0].new_zeros(B, self.n_vq_emb, H, W, dtype=torch.float32)
        # assert len(idx_Bhw_list) == len(n_points_per_level)
        # for si, pn in enumerate(n_points_per_level): # from small to large
        #     idx_Bhw = idx_Bhw_list[si] # (B, h, w)
        #     h_BChw = self.embedding(idx_Bhw).permute(0, 3, 1, 2) # embedding(B,pn,2,n_emb) --> permute (B, n_emb, pn, 2)
        #     if si < len(n_points_per_level) - 1:
        #         h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
        #     h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
        #     f_hat_recons.add_(h_BChw)
        # f_hat = f_hat_recons

        # ----------------------------------------
        quantized_dict = self.vae(f_BCt2, ret_usages=False)

        f_hat = quantized_dict['f_hat']
        mean_vq_loss = quantized_dict['loss_commitment_dictionary']
        # f_hat_per_level = quantized_dict['f_hat_per_level']
        f_hat_per_level = None

        # f_BCt2_reconstructed = self.vqvae_decoder(f_hat) # [n_agent, n_emb, n_step=18, W]

        f_BCt2_reconstructed = quantized_dict['f_BCt2_reconstructed']
        assert f_BCt2_reconstructed.shape[2] == n_step

        if self.use_xy_as_output:
            assert f_BCt2_reconstructed.shape[1] == 1, "f_BCt2_reconstructed.shape[1] should be equal to 1"
            assert f_BCt2_reconstructed.shape[3] == 2
            f_BCt2_reconstructed = f_BCt2_reconstructed.squeeze(1) # [n_agent, 1,  n_step=18, 2] --> [n_agent,  n_step=18, 2] 
        else:
            # next_token_logits = [n_agent, n_step, n_token], n_step=18
            # H=n_step, W = 1
            assert f_BCt2_reconstructed.shape[1] == self.n_token_agent
            assert f_BCt2_reconstructed.shape[3] == 1
            f_BCt2_reconstructed = f_BCt2_reconstructed.squeeze(-1) # [n_agent, n_emb=2048, n_step=18, 1] --> [n_agent, n_emb, n_step=18]
            f_BCt2_reconstructed = f_BCt2_reconstructed.permute(0, 2, 1) # [n_agent, n_step=18, n_token]
            assert f_BCt2_reconstructed.shape[2] == self.n_token_agent, "f_BCt2_reconstructed.shape[2] should be equal to n_token"

        # margin = tdist.get_world_size() * (f_BCt2.numel() / f_BCt2.shape[1]) / self.vq_vocab_size * 0.08
        return {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": next_token_logits[:, 1:-1],  # [n_agent, 16, n_token]   # len(range(10, 90, 5))=16  # start=10, end=90 (exclusive), step=5
            "next_token_logits_raw": next_token_logits, # [n_agent, 18, n_token]
            "next_token_valid": tokenized_agent["valid_mask"][:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": tokenized_agent["sampled_pos"],  # [n_agent, 18, 2]
            "pred_head": tokenized_agent["sampled_heading"],  # [n_agent, 18]
            "pred_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # vqvae loss
            "loss_commitment_dictionary": mean_vq_loss, # [1]
            "f_hat": f_hat, # [n_agent, n_emb, 18, 2] # final reconstructed feature map # think it is alg 2
            "f_hat_per_level": f_hat_per_level, # List[Tensor[n_agent, n_emb, 18, 2]] # reconstructed feature map at each level
            "f_BCt2": f_BCt2, # [n_agent, n_emb, 18, 2] # encoder output
            "n_points_per_level": self.n_points_per_level, # List[int]
            "f_BCt2_reconstructed": f_BCt2_reconstructed, # [n_agent, 18, 2] # reconstructed input (traj) from the quantized embeddings, or [n_agent, n_step=18, n_token]
            # inference
            "feat_a": smart_dict["feat_a"],  # [n_agent, n_step, hidden_dim]
            "agent_token_emb": smart_dict["agent_token_emb"],  # [n_agent, n_step, hidden_dim]
            "agent_token_emb_veh": smart_dict["agent_token_emb_veh"],  # [n_agent, hidden_dim]
            "agent_token_emb_ped": smart_dict["agent_token_emb_ped"],  # [n_agent, hidden_dim]
            "agent_token_emb_cyc": smart_dict["agent_token_emb_cyc"],  # [n_agent, hidden_dim]
            "veh_mask": smart_dict["veh_mask"],  # [n_agent]
            "ped_mask": smart_dict["ped_mask"],  # [n_agent]
            "cyc_mask": smart_dict["cyc_mask"],  # [n_agent]
            "categorical_embs": smart_dict["categorical_embs"],  # List of len=2, shape [n_agent, hidden_dim]
        }


    def next_scale_var(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
        inference=False
    ) -> Dict[str, torch.Tensor]:
        """
        Train the autoregressive next-scale transformer model.
        """
        Ten = torch.Tensor
        FTen = torch.Tensor
        ITen = torch.LongTensor
        BTen = torch.BoolTensor
        
        with torch.set_grad_enabled(self.finetune_vqvae):
            open_loop_results = self.open_next_scale(
                tokenized_agent,
                map_feature,
                sampling_scheme,
                inference=inference,
            )

        with torch.no_grad():
            prog_si=-1
            prog_wp_it=20
            # if progressive training
            self.var_wo_ddp.prog_si = self.vae.quantize.prog_si = prog_si
            if self.last_prog_si != prog_si:
                if self.last_prog_si != -1: self.first_prog = False
                self.last_prog_si = prog_si
                self.prog_it = 0
            self.prog_it += 1
            prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
            if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
            if prog_si == len(self.var_wo_ddp.patch_nums) - 1: prog_si = -1    # max prog, as if no prog

            inp_B3HW: torch.Tensor = open_loop_results["f_BCt2"] #.to(self.vae.quantize.embedding.weight) #.half()
            # label_B: Union[torch.LongTensor, torch.Tensor] = torch.ones(B, dtype=torch.long)
            # label_B use the argmax of the logits at last step of n_step in  "next_token_logits_raw"
            label_B = open_loop_results["next_token_logits_raw"][:, -1, :] # last step, [n_agent, n_token]
            label_B = torch.argmax(label_B, dim=-1) #.half() # [n_agent]
            assert not torch.isnan(inp_B3HW).any()
            # forward
            #Train step 2: VAR Training (after VQVAE is trained):
            # VQVAE is frozen
            # Use Algorithm 1 to get ground truth tokens
            # Train transformer to predict these tokens
            # Uses teacher forcing (showing ground truth tokens during training)
            B, V = label_B.shape[0], self.vae.vocab_size
            self.var_wo_ddp.require_backward_grad_sync = stepping = False
            # 1. Get ground truth tokens using Algorithm 1
            gt_idx_Bl: List[torch.LongTensor] = self.vae.img_to_idxBl(inp_B3HW, 
                                                                    v_patch_nums=self.var_wo_ddp.patch_nums) # List[B, patch_h*patch_w] codebook indices, multi-scale tokens R
            gt_BL = torch.cat(gt_idx_Bl, dim=1) # (B, L), ground truth quantized indices for the input image batch
            x_BLCv_wo_first_l: torch.Tensor = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl) # (B, L, Cv), quantized indices to var input
            # teacher forcing input, "wo" means without, and "first_l" refers to the first token in the sequence, Used to predict the next token during training.
            assert not torch.isnan(gt_BL).any()
        
        # with amp.autocast(enabled=True): #, dtype=torch.float16):
        with self.amp_ctx:
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
            loss = loss.mul(lw).sum(dim=-1).mean()

        close_loop_result = open_loop_results
        close_loop_result["var_loss"] = loss
        close_loop_result["var_logits_BLV"] = logits_BLV
        close_loop_result["var_gt_BL"] = gt_BL
        return close_loop_result
    
    
    def next_scale_autoreg(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
        inference: bool = True # compute autoregressive pos
    ) -> Dict[str, torch.Tensor]:
        
        # pytorch lightening calls no grad during validation and test, no need to be explicit
        close_loop_result = self.next_scale_var(tokenized_agent, map_feature, sampling_scheme, 
                                                inference=inference)
            
        # self.training = not inference
        
        head_a = tokenized_agent["sampled_heading"]
        n_agent, n_step = head_a.shape
        
        inp_B3HW: torch.Tensor = close_loop_result["f_BCt2"] #.half()
        # label_B: Union[torch.LongTensor, torch.Tensor] = torch.ones(B, dtype=torch.long)
        # label_B use the argmax of the logits at last step of n_step in  "next_token_logits_raw"
        label_B = close_loop_result["next_token_logits_raw"][:, -1, :] # last step, [n_agent, n_token]
        label_B = torch.argmax(label_B, dim=-1) #.half() # [n_agent]
        B, V = label_B.shape[0], self.vae.vocab_size
        
        next_token_logits = self.var_wo_ddp.autoregressive_infer_cfg(B, label_B,
            g_seed= None, cfg=1.5, top_k=sampling_scheme.num_k, top_p=0.0, more_smooth=False) # [n_agent, n_emb=2048, n_step=18, 1]
        
        # print('decoder_out:', next_token_logits.shape)
        
        assert next_token_logits.shape[2] == n_step
        assert not torch.isnan(next_token_logits).any()

        if self.use_xy_as_output:
            assert next_token_logits.shape[1] == 1, "next_token_logits.shape[1] should be equal to 1"
            assert next_token_logits.shape[3] == 2
            next_token_logits = next_token_logits.squeeze(1) # [n_agent, 1,  n_step=18, 2] --> [n_agent,  n_step=18, 2] 
        else:
            # next_token_logits = [n_agent, n_step, n_token], n_step=18
            # H=n_step, W = 1
            assert next_token_logits.shape[1] == self.n_token_agent
            assert next_token_logits.shape[3] == 1
            next_token_logits = next_token_logits.squeeze(-1) # [n_agent, n_emb=2048, n_step=18, 1] --> [n_agent, n_emb, n_step=18]
            next_token_logits = next_token_logits.permute(0, 2, 1) # [n_agent, n_step=18, n_token]
            assert next_token_logits.shape[2] == self.n_token_agent, "next_token_logits.shape[2] should be equal to n_token"

        next_token_logits = next_token_logits # [n_agent, 18, n_token]
        
        # print('next_token_logits', next_token_logits.shape)
        
        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_step_future_10hz = self.num_future_steps  # 80
        n_step_future_2hz = n_step_future_10hz // self.shift  # 16
        step_current_10hz = self.num_historical_steps - 1  # 10
        step_current_2hz = step_current_10hz // self.shift  # 2
        
        pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone() # hk: [n_agent, n_steps=18, 2] --> [n_agent, 2, 2]. Current? [0, 5]
        head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pred_idx = tokenized_agent["gt_idx"].clone()
        
        if not self.training:
            pred_traj_10hz = torch.zeros(
                [n_agent, n_step_future_10hz, 2], dtype=pos_a.dtype, device=pos_a.device
            ) # hk: [n_agent, 80, 2]
            pred_head_10hz = torch.zeros(
                [n_agent, n_step_future_10hz], dtype=pos_a.dtype, device=pos_a.device
            ) # hk: [n_agent, 80]
            
        pred_valid = tokenized_agent["valid_mask"].clone() # hk: [n_agent, 18]
        next_token_logits_list = []
        next_token_action_list = []
        feat_a_t_dict = {}
        
        sampling_scheme_catk = sampling_scheme.copy()
        sampling_scheme_catk.criterium = "topk_prob_sampled_with_dist" # {topk_dist_sampled_with_prob, topk_prob, topk_prob_sampled_with_dist}
        sampling_scheme_catk.num_k = sampling_scheme_catk.num_k_catk # for k nearest neighbors, set to -1 to turn-off closed-loop training
        sampling_scheme_catk.temp = 1e-5 # catk = topk_prob_sampled_with_dist with temp=1e-5
        
        feat_a = close_loop_result["feat_a"] # [n_agent, n_step, hidden_dim]
        agent_token_emb = close_loop_result["agent_token_emb"] # [n_agent, n_step, hidden_dim]
        agent_token_emb_veh = close_loop_result["agent_token_emb_veh"] # [n_agent, hidden_dim]
        agent_token_emb_ped = close_loop_result["agent_token_emb_ped"] # [n_agent, hidden_dim]
        agent_token_emb_cyc = close_loop_result["agent_token_emb_cyc"] # [n_agent, hidden_dim]
        veh_mask = close_loop_result["veh_mask"] # [n_agent]
        ped_mask = close_loop_result["ped_mask"] # [n_agent]
        cyc_mask = close_loop_result["cyc_mask"] # [n_agent]
        categorical_embs = close_loop_result["categorical_embs"] # List of len=2, shape [n_agent, hidden_dim]
        
        # import pdb; pdb.set_trace()
        for t in range(n_step_future_2hz):  # 0 -> 15
            t_now = step_current_2hz - 1 + t  # 1 -> 16
            n_step = t_now + 1  # 2 -> 17
            # print('t_now:', t_now, 'n_step:', n_step, "pos_a", pos_a.shape, "head_a", head_a.shape, "head_vector_a", head_vector_a.shape, "pred_valid", pred_valid.shape)
            # print('tokenized_agent["num_graphs"]', tokenized_agent["num_graphs"])
            # print('tokenized_agent["batch"]', tokenized_agent["batch"])
            if t == 0:  # init
                hist_step = step_current_2hz # 2
                
            next_token_idx, next_token_traj_all = sample_next_token_traj(
                token_traj=tokenized_agent["token_traj"], # [n_agent, n_token, 4, 2]
                token_traj_all=tokenized_agent["token_traj_all"], # [n_agent, n_token, 6, 4, 2]
                sampling_scheme=sampling_scheme_catk,
                # ! for most-likely sampling
                next_token_logits=next_token_logits[:, n_step, :], # hk: [n_agent, n_token]
                # ! for nearest-pos sampling
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
                pos_next_gt=tokenized_agent["gt_pos_raw"][:, n_step],  # [n_agent, 2]
                head_next_gt=tokenized_agent["gt_head_raw"][:, n_step],  # [n_agent]
                valid_next_gt=tokenized_agent["gt_valid_raw"][:, n_step],  # [n_agent]
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_token, 2]
            )  # next_token_idx: [n_agent], next_token_traj_all: [n_agent, 6, 4, 2]

            diff_xy = next_token_traj_all[:, -1, 0] - next_token_traj_all[:, -1, 3]
            next_token_action_list.append(
                torch.cat(
                    [
                        next_token_traj_all[:, -1].mean(1),  # [n_agent, 2]
                        torch.arctan2(diff_xy[:, [1]], diff_xy[:, [0]]),  # [n_agent, 1]
                    ],
                    dim=-1,
                )  # [n_agent, 3]
            )

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
            )[0].view(*next_token_traj_all.shape) # [n_agent, 6, 4, 2]

            if not self.training:
                pred_traj_10hz[:, t * 5 : (t + 1) * 5] = token_traj_global[:, 1:].mean(
                    2 # reduce mean over 4 box corners
                ) # [n_agent, 5, 2]
                diff_xy = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head_10hz[:, t * 5 : (t + 1) * 5] = torch.arctan2(
                    diff_xy[:, :, 1], diff_xy[:, :, 0]
                )
                
                # import pdb; pdb.set_trace()
                
            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
            pred_idx[:, n_step] = next_token_idx

            # ! update tensors for for next step
            pred_valid[:, n_step] = pred_valid[:, t_now]
            # pred_valid[:, n_step] = pred_valid[:, t_now] | mask_spawn
            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
            head_vector_a_next = torch.stack(
                [head_a_next.cos(), head_a_next.sin()], dim=-1
            )
            head_vector_a = torch.cat(
                [head_vector_a, head_vector_a_next.unsqueeze(1)], dim=1
            )

            # ! get agent_token_emb_next
            agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
            agent_token_emb_next[veh_mask] = agent_token_emb_veh[
                next_token_idx[veh_mask]
            ].to(agent_token_emb_next)
            agent_token_emb_next[ped_mask] = agent_token_emb_ped[
                next_token_idx[ped_mask]
            ].to(agent_token_emb_next)
            agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[
                next_token_idx[cyc_mask]
            ].to(agent_token_emb_next)
            agent_token_emb = torch.cat(
                [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
            )

            # ! get feat_a_next
            motion_vector_a = pos_a[:, -1] - pos_a[:, -2]  # [n_agent, 2]
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a[:, -1], nbr_vector=motion_vector_a
                    ),
                ],
                dim=-1,
            )
            # [n_agent, hidden_dim]
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            # [n_agent, 1, 2*hidden_dim]
            feat_a_next = torch.cat((agent_token_emb_next, x_a), dim=-1).unsqueeze(1)
            feat_a_next = self.fusion_emb(feat_a_next)
            feat_a = torch.cat([feat_a, feat_a_next], dim=1)
        
        out_dict = {
            # action that goes from [(10->15), ..., (85->90)]
            # "next_token_logits": torch.stack(next_token_logits_list, dim=1), # hk: [n_agent, 16, n_token]
            "next_token_logits": next_token_logits[:, 1:-1, :], # hk: [n_agent, 16, n_token]
            "next_token_valid": pred_valid[:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)] # hk: len(list(range(5, 95, 5)))=18
            "pred_pos": pos_a,  # [n_agent, 18, 2]
            "pred_head": head_a,  # [n_agent, 18]
            "pred_valid": pred_valid,  # [n_agent, 18]
            "pred_idx": pred_idx,  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for shifting proxy targets by lr
            "next_token_action": torch.stack(next_token_action_list, dim=1),
            # vqvae loss
            "loss_commitment_dictionary": close_loop_result["loss_commitment_dictionary"], # [1]
            "f_hat": close_loop_result["f_hat"], # [n_agent, n_emb, 18, 2] # final reconstructed feature map # think it is alg 2
            "f_hat_per_level": close_loop_result["f_hat_per_level"], # List[Tensor[n_agent, n_emb, 18, 2]] # reconstructed feature map at each level
            "f_BCt2": close_loop_result["f_BCt2"], # [n_agent, n_emb, 18, 2] # encoder output
            "n_points_per_level": close_loop_result["n_points_per_level"], # List[int]
            "f_BCt2_reconstructed": close_loop_result["f_BCt2_reconstructed"], # [n_agent, 18, 2] # reconstructed input (traj) from the quantized embeddings, or [n_agent, n_step=18, n_token]
            # VAR loss
            "var_loss": close_loop_result["var_loss"],
            "var_logits_BLV": close_loop_result["var_logits_BLV"],
            "var_gt_BL": close_loop_result["var_gt_BL"],
        }

        if not self.training:  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_traj_10hz"] = pred_traj_10hz # hk: [n_agent, 80, 2]
            out_dict["pred_head_10hz"] = pred_head_10hz # hk: [n_agent, 80]
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1]) # hk: [n_agent, 80]

        return out_dict