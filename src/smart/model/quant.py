from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

# import src.smart.utils.dist as dist


# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,  # share_quant_resi: args.qsr
        W=1 # BCHW: [n_agent, n_emb, n_step, 2], n_step=18. or [n_agent, n_token, n_step, 1]
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
         
        self.patch_hws = [(pn, W) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (self.v_patch_nums)]    # from small to large
        H = self.v_patch_nums[-1]
        if W != -1: 
            assert self.patch_hws[-1][0] == H and self.patch_hws[-1][1] == W, f'{self.patch_hws[-1]=} != ({H=}, {W=})'
        print('VectorQuantizer2.init self.patch_hws:', self.patch_hws)
        
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae) #.half() # codebook/lookup table. Output: Each row of the embedding matrix represents a codebook embedding.
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1

        self.W = W
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    # ===================== `forward` is only used in VAE training =====================
    def forward_original(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        f_rest = f_no_grad.clone() # f_rest is the residual feature map, which starts as the full feature map f produced by the encoder.
        f_hat = torch.zeros_like(f_rest)
        
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                # find the nearest embedding
                if self.using_znorm: # using_znorm=True (cosine similarity)
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C) # (B, h, w, C) -> (B*h*w, C)
                    rest_NC = F.normalize(rest_NC, dim=-1) # Normalizes the feature map points rest_NC along the embedding dimension C (L2 norm).
                    # Normalizes the codebook embeddings along each row, dot product with normalized rest_NC which gives cosine similarity scores.
                    # Finds the index of the embedding k with the highest cosine similarity for each feature point.
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1) # (B*h*w, C) @ (C, vocab_size) -> (B*h*w, vocab_size) -> argmax -> (B*h*w)
                else: # using_znorm=False (Euclidean distance).
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C) # (B, h, w, C) -> (B*h*w, C)
                    # ||x - e_k||^2 = ||x||^2 + ||e_k||^2 - 2 * x * e_k
                    # ||x||^2 -> (B*h*w, C) ==> ||x||^2 = sum(x^2, dim=1, keepdim=True) -> (B*h*w, 1) 
                    # ||e_k||^2 -> (vocab_size, C) ==> ||e_k||^2 = sum(e_k^2, dim=1, keepdim=False) -> (vocab_size,)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False) # (B*h*w, 1) + (vocab_size,) -> (B*h*w, vocab_size)
                    # -2 * x * e_k = -2 * x @ e_k^T. addmm_ is in-place operation, input = beta * input + alpha * mat1 @ mat2
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, C) @ (C, vocab_size) -> (B*h*w, vocab_size), addmm_ is in-place operation
                    idx_N = torch.argmin(d_no_grad, dim=1) # (B*h*w)
                
                hit_V = idx_N.bincount(minlength=self.vocab_size).float() # count the number of occurrences of each value in the tensor idx_N
                if self.training:
                    if dist.initialized(): handler = tdist.all_reduce(hit_V, async_op=True)
                
                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn) # (B*h*w) -> (B, h, w)
                # h_BChw: quantized lookup embedding. interpolate/upscale the embeddings (h,w) to the same full size (H, W) as the feature map, then permute the tensor to (B, C, H, W), 
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)  # A refinement step is applied to the quantized embeddings h_BChw for the current scale.
                                                             # The refinement step is a convolutional layer with kernel size 3x3 and stride 1.
                f_hat = f_hat + h_BChw # Accumulates the reconstructed feature map from quantized embeddings, approximate feature map f_BChw. shape (B, C, H, W)
                f_rest -= h_BChw # Updates the residual f_rest by removing the contribution of the current scale's embeddings h_BChw.
                                 # passes the remaining unexplained features to the next finer scale.
                if self.training and dist.initialized():
                    handler.wait() # The codebook vectors are updated via EMA, which aligns them with the latent space without relying on gradient updates.
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V) # Exponential Moving Average (EMA). hit_V histogram of embeddings
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1)) # EMA = 0.9 * EMA + 0.1 * hit_V
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01)) # EMA = 0.99 * EMA + 0.01 * hit_V
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
                # Commitment loss: make sure encoder output f_BChw  ze(x)  commits to the quantized embeddings (f_hat.data) (.data detach from graph), sg[e]. 
                # Dictionary Loss: ensures the quantized embeddings f_hat accurately reconstruct the encoder output feature map f_no_grad
            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        
        margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        else: usages = None
        return f_hat, usages, mean_vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Dict[str, torch.Tensor]: #Tuple[torch.Tensor, List[float], torch.Tensor]:
        f_BCt2 = f_BChw

        dtype = f_BCt2.dtype
        if dtype != torch.float32: f_BCt2 = f_BCt2.float()
        B, C, H, W = f_BCt2.shape # [n_agent, n_emb, n_step, 2], n_step=18. or [n_agent, n_token, n_step, 1]
        # assert H == 18, "n_step should be 18"
        f_no_grad = f_BCt2.detach()
        
        f_rest = f_no_grad.clone() # f_rest is the residual feature map, which starts as the full feature map f produced by the encoder.
        f_hat = torch.zeros_like(f_rest)
        f_hat_per_level = []
        # idx_Bhw_list = []
        # print('f_rest.shape:', f_rest.shape)

        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hist_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BCt2.device)

            SN = len(self.patch_hws)
            for si, (ph, pw) in enumerate(self.patch_hws): # from small to large
                # find the nearest embedding
                # print('processing scale:', si, 'with n_points_per_level:', pn)
                if self.using_znorm: # using_znorm=True (cosine similarity)  
                    # Interpolate spatial and temporal dimensions are interpolated to the target number of points pn.
                    f_B2tC_upsampled = F.interpolate(
                        f_rest, size=(ph, pw), mode='bicubic', align_corners=True
                    ).permute(0, 2, 3, 1).reshape(-1, C)  if (si != SN-1) else f_BCt2.permute(0, 2, 3, 1).reshape(-1, C) 
                    # interpolate: [n_agents, n_embedding, n_upsampled_steps, 2] --> permute/reshape: [n_agents * 2* n_upsampled_steps, n_embedding]
                    f_B2tC_upsampled = F.normalize(f_B2tC_upsampled, dim=-1)  # Normalizes the feature map points f_B2tC_upsampled along the embedding dimension n_embedding (L2 norm).
                    # Normalizes the codebook embeddings along each row, dot product with normalized rest_NC which gives cosine similarity scores.
                    # Finds the index of the embedding k with the highest cosine similarity for each feature point.
                    idx_N = torch.argmax(f_B2tC_upsampled @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1) # (B*h*w, C) @ (C, vocab_size) -> (B*h*w, vocab_size) -> argmax -> (B*h*w)
                else: # self.using_znorm=False (Euclidean distance).
                    # Interpolate spatial and temporal dimensions are interpolated to the target number of points pn.
                    f_B2tC_upsampled = F.interpolate(
                        f_rest, size=(ph, pw), mode='bicubic', align_corners=True
                    ).permute(0, 2, 3, 1).reshape(-1, C)  if (si != SN-1) else f_BCt2.permute(0, 2, 3, 1).reshape(-1, C)
                    # interpolate: [n_agents, n_embedding, n_upsampled_steps, 2] --> permute/reshape: [n_agents * 2* n_upsampled_steps, n_embedding]
                    # ||x - e_k||^2 = ||x||^2 + ||e_k||^2 - 2 * x * e_k
                    # ||x||^2 -> (B*h*w, C) ==> ||x||^2 = sum(x^2, dim=1, keepdim=True) -> (B*h*w, 1) 
                    # ||e_k||^2 -> (vocab_size, C) ==> ||e_k||^2 = sum(e_k^2, dim=1, keepdim=False) -> (vocab_size,)
                    d_no_grad = torch.sum(f_B2tC_upsampled.square(), dim=1, keepdim=True) + \
                        torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False) # (B*h*w, 1) + (vocab_size,) -> (B*h*w, vocab_size)
                    # -2 * x * e_k = -2 * x @ e_k^T. addmm_ is in-place operation, input = beta * input + alpha * mat1 @ mat2
                    d_no_grad.addmm_(f_B2tC_upsampled, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, C) @ (C, vocab_size) -> (B*h*w, vocab_size), addmm_ is in-place operation
                    # d_no_grad = 1.0 * d_no_grad.half() + -2.0 *  f_B2tC_upsampled.half() @ self.embedding.weight.data.T.half()
                    idx_N = torch.argmin(d_no_grad, dim=1) # (B*h*w)
                
                hist_V = idx_N.bincount(minlength=self.vocab_size).float() # count the number of occurrences of each value in the tensor idx_N
                # if self.training:
                #     if dist.initialized(): handler = tdist.all_reduce(hist_V, async_op=True)
                
                # calc loss
                # print('ph:', ph, 'pw:', pw, 'idx_N.shape:', idx_N.shape)
                idx_Bhw = idx_N.view(B, ph, pw) # (B*h*w) -> (B, h, w)
                # idx_Bhw_list.append(idx_Bhw)
                # h_BCt2: idx_Bhw(B, h, w) -> embedding -> (B,pn,2,n_emb) --> permute (B, n_emb, pn, 2) --> interpolate to HW (B, C, H=18, W=2)
                h_BCt2 = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else \
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BCt2 = self.quant_resi[si/(SN-1)](h_BCt2)  # A refinement step is applied to the quantized embeddings h_BCt2 for the current scale.
                    # [B, n_emb, 18, 2]  # Hierarchical refinement: weights are shared across scales, weight-sum of raw quantized embeddings and the refinement.
                f_hat = f_hat + h_BCt2 # Accumulates the reconstructed feature map from quantized embeddings, approximate feature map f_BChw. shape (B, C, H, W), [n_agent, n_emb, 18, 2]
                f_rest -= h_BCt2 # Updates the residual f_rest by removing the contribution of the current scale's embeddings h_BCt2.
                                 # passes the remaining unexplained features to the next finer scale.
                # if self.training and dist.initialized():
                #     handler.wait() # The codebook vectors are updated via EMA, which aligns them with the latent space without relying on gradient updates.
                #     if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hist_V) # Exponential Moving Average (EMA). hist_V histogram of embeddings
                #     elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hist_V.mul(0.1)) # EMA = 0.9 * EMA + 0.1 * hist_V
                #     else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hist_V.mul(0.01)) # EMA = 0.99 * EMA + 0.01 * hist_V
                #     self.record_hit += 1
                vocab_hist_V.add_(hist_V) # [vocab_size]
                mean_vq_loss += F.mse_loss(f_hat.data, f_BCt2).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
                # mean_vq_loss *= loss_weighting_mask # make sure token_world_sample[:, 1:-1] # hk: [n_agent, n_step=16, 2]
                # Commitment loss: make sure encoder output f_BChw  ze(x)  commits to the quantized embeddings (f_hat.data) (.data detach from graph), sg[e]. 
                # Dictionary Loss: ensures the quantized embeddings f_hat accurately reconstruct the encoder output feature map f_no_grad sg[ze(x)]
                f_hat_per_level.append(f_hat)
            mean_vq_loss *= 1. / SN
            # print('mean_vq_loss:', mean_vq_loss)
            # preserve gradients: z_q = z + (z_q - z).detach(). gradient flow from decoder -> f_hat -> f_BCt2 (-> encoder)
            f_hat = (f_hat.data - f_no_grad).add_(f_BCt2) # [n_agent, n_emb, 18, 2]

        # margin = tdist.get_world_size() * (f_BCt2.numel() / f_BCt2.shape[1]) / self.vocab_size * 0.08
        margin = 1 * (f_BCt2.numel() / f_BCt2.shape[1]) / self.vocab_size * 0.08
        ## margin = pn*pn / 100
        if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        else: usages = None

        return {
            # vqvae loss
            "loss_commitment_dictionary": mean_vq_loss, # [1]
            "f_hat": f_hat, # [n_agent, n_emb, 18, 2] # final reconstructed feature map # think it is alg 2
            "f_hat_per_level": f_hat_per_level, # List[Tensor[n_agent, n_emb, 18, 2]] # reconstructed feature map at each level
            "f_BCt2": f_BCt2, # [n_agent, n_emb, 18, 2] # encoder output
            "n_points_per_level": self.v_patch_nums, # List[int]
            # "f_BCt2_reconstructed": f_BCt2_reconstructed, # [n_agent, 18, 2] # reconstructed input (traj) from the quantized embeddings, or [n_agent, n_step=18, n_token]
            "vocab_hist_usages": usages, # List[float]
        }
    # ===================== `forward` is only used in VAE training =====================

    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        # H = W = self.v_patch_nums[-1]
        H, W = self.patch_hws[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            # f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                # f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                f_hat = F.interpolate(f_hat, size=(pn, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw) # The refined embeddings are added to the feature map f_hat.
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        """
        v_patch_nums: can be list of tuples for rectangle [(1, W), (3, W), (5, W)...], where W=1 or 2

        to_fhat: True for f_hat, False for idx_Bl

        return f_hat_or_idx_Bl: fhat: List[BChw], idx_Bl: List[Bl]/List[(B*h*w)]
        """
        
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone() # Initialize residual feature map
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []  # Store tokens R for each scale
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(self.patch_hws)
        for si, (ph, pw) in enumerate(self.patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1) # (B*h*w, C) @ (C, vocab_size) -> (B*h*w, vocab_size) -> argmax -> (B*h*w)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, ph, pw) # (B*h*w) -> (B, h, w)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw) # Refinement network
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            # print('B, ph, pw:', B, ph, pw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw)) # (B, ph*pw)
        
        return f_hat_or_idx_Bl
    
    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """
        gt_ms_idx_Bl[0]: [B, L]
        return BLC: [B, L, C], where L = sum([ph*pw for ph, pw in self.v_patch_nums]), C = self.Cvae
        """
        # print('idxBl_to_var_input gt_ms_idx_Bl:', gt_ms_idx_Bl[0].shape, gt_ms_idx_Bl[-1].shape)
        
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0] 
        C = self.Cvae # n_emb
        # H = W = self.v_patch_nums[-1]
            
        H, W = self.patch_hws[-1]
        SN = len(self.v_patch_nums)
        # print('B, C H W:', B, C, H, W)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        pn_nextH, pn_nextW = self.patch_hws[0]
        for si in range(SN-1):
            
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # progressive training: not supported yet, prog_si always -1
            # h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            # h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, W), size=(H, W), mode='bicubic')
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_nextH, pn_nextW), size=(H, W), mode='bicubic')
            
            # same as this block         
            # h_BCt2: idx_Bhw(B, h, w) -> embedding -> (B,pn,2,n_emb) --> permute (B, n_emb, pn, 2) --> interpolate to HW (B, C, H=18, W=2)
            # print('gt_ms_idx_Bl[si]', gt_ms_idx_Bl[si].shape)
            # idx_Bhw = gt_ms_idx_Bl[si].view(B, pn_next, W) # (B*h*w) -> (B, h, w)
            # h_BCt2 = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else \
            #     self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            # h_BChw = h_BCt2
            # h_BCt2 = self.quant_resi[si/(SN-1)](h_BCt2)  # A refinement step is applied to the quantized embeddings h_BCt2 for the current scale.
            # [B, n_emb, 18, 2]  # Hierarchical refinement: weights are shared across scales, weight-sum of raw quantized embeddings and the refinement.
        
            # in-place operation, no gradient
            # f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw)) # upscale the embeddings to the full size (H, W) of the feature map, run through conv layers
            
            # Allow gradient
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)  # A refinement step is applied to the quantized embeddings h_BCt2 for the current scale.
            # [B, n_emb, 18, 2]  # Hierarchical refinement: weights are shared across scales, weight-sum of raw quantized embeddings and the refinement.
            f_hat = f_hat + h_BChw # Accumulates the reconstructed feature map from quantized embeddings, approximate feature map f_BChw. shape (B, C, H, W), [n_agent, n_emb, 18, 2]
  
            
            # pn_next = self.v_patch_nums[si+1] # Update pn_next to the next scale’s resolution.
            pn_nextH, pn_nextW = self.patch_hws[si+1]
            # next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
            # next_scales.append(F.interpolate(f_hat, size=(pn_next, W), mode='area').view(B, C, -1).transpose(1, 2))
            next_scales.append(F.interpolate(f_hat, size=(pn_nextH, pn_nextW), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        # HW = W = self.v_patch_nums[-1]
        HW, W = self.patch_hws[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, W), mode='bicubic'))     # conv after upsample
            f_hat.add_(h)
            # return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
            return f_hat, F.interpolate(f_hat, size=(self.patch_hws[si+1][0], self.patch_hws[si+1][1]), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw): # h_BChw * (1 - resi_ratio) + Conv2d(h_BChw) * resi_ratio
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi) # Defines evenly spaced tick marks across the range [0, 1], used to index the available Phi instances (qresi_ls).
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K) # [0.08333333, 0.36111111, 0.63888889, 0.91666667] if
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
