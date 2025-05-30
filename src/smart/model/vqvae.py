"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2


class VQVAE(nn.Module):
    def __init__(
        self, vocab_size=4096, z_channels=32, ch=128, dropout=0.0,
        beta=0.25,              # commitment loss weight
        using_znorm=False,      # whether to normalize when computing the nearest neighbors
        quant_conv_ks=3,        # quant conv kernel size
        quant_resi=0.5,         # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4,     # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0, # if is 0: automatically set to len(v_patch_nums)
        # v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        v_patch_nums=[1, 2, 3, 5, 9, 16, 18], # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        test_mode=False,
        coder_in_channels=3,    # encoder decoder input channel
        coder_ch_mult=(1, 1, 2, 2, 4),  # encoder decoder channel multiplier
        W=1,   # width of the next-scale image
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        # ddconfig is copied from https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/vq-f16/config.yaml
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,  # ch is encoder internal channels, z_channels is the output channels
            in_channels=coder_in_channels, ch_mult=coder_ch_mult, num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
            W=W,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp, ret_usages=False):   # -> rec_B3HW, idx_N, loss
        # Algorithm 1: Multi-scale VQVAE Encoding
        # 1. Encode image: f = E(im)
        # 2. self.quantize: Get tokens and embeddings at multiple scales
        # 3. Decode back to reconstructed image
        VectorQuantizer2.forward
        # f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        # return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss

        # inp: [n_agent, n_emb=n_token=2048, n_step, 1]
        # self.encoder(inp) [n_agent, Cvae=64, n_step=18, 1]
        result_dict = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        f_hat, usages, vq_loss = result_dict["f_hat"], result_dict["vocab_hist_usages"], result_dict["loss_commitment_dictionary"]

        # self.decoder(self.post_quant_conv(f_hat)).shape [n_agent, 2048, 18, 1]
        result_dict["f_BCt2_reconstructed"] = self.decoder(self.post_quant_conv(f_hat)) # [n_agent, n_emb, n_step=18, W]
        return result_dict
    # ===================== `forward` is only used in VAE training =====================
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        # return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
        return self.decoder(self.post_quant_conv(f_hat))
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad)).to(self.quantize.embedding.weight.dtype)
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Algorithm 2: Multi-scale VQVAE Reconstruction
        Inputs: multi-scale token maps R (ms_idx_Bl)
        Hyperparameters: steps K, resolutions (h_k, w_k)

        This reconstruction is deterministic (tokens → image), Already have all tokens, just reconstruct. 
        In generation autoregressive_infer_cfg is probabilistic (predict tokens → image), no tokens, just given class label.
        """
        # Initialize reconstruction
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        # For k = 1,...,K do
        for idx_Bl in ms_idx_Bl: # Each idx_Bl represents r_k = queue_pop(R)
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            # Step 6: z_k = lookup(Z, r_k) - get embeddings from codebook
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
       
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Complete steps 7-9 of Algorithm 2:
        - Interpolate embeddings
        - Accumulate reconstructions
        - Decode final image

        ms_h_BChw: List[B, C, h, w] for each scale
        """
        if last_one:
            # Steps 7-8: Interpolate and accumulate f̂ = f̂ + φ_k(z_k), embed_to_fhat()
            # Step 9: îm = D(f̂) - decode to reconstructed image
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
