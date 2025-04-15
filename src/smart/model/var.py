import math
from functools import partial
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
# from huggingface_hub import PyTorchModelHubMixin

import src.smart.utils.dist as dist
from .basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from .helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from .vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=2048, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 5, 9, 16, 18),   # max 18 n_steps by default
        flash_if_available=True, fused_if_available=True,
        W=1 # BCHW: [n_agent, n_emb, n_step, 2], n_step=18. or [n_agent, n_token, n_step, 1]
    ):
        """
        num_classes = n_tokens, use the endpoint of a trajectory.
        """
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        # self.patch_nums: Tuple[int] = patch_nums
        patch_nums = [(pn, W) if isinstance(pn, int) else (pn[0], pn[1]) for pn in patch_nums]
        self.patch_nums: List[Tuple[int, int]] = patch_nums # rectangular patches
        # self.L = sum(pn ** 2 for pn in self.patch_nums)  
        # self.first_l = self.patch_nums[0] ** 2 # Number of tokens at the first scale
        self.L = sum(ph * pw for (ph, pw) in self.patch_nums) # Total length of the sequence for all scales
        # if patch_nums[0][1] == 1:
        #     assert self.L == 54, f"self.L {self.L} != 54"
        self.first_l = self.patch_nums[0][0] * self.patch_nums[0][1] # Number of tokens at the first scale
        self.begin_ends = []
        cur = 0
        # for i, pn in enumerate(self.patch_nums):
            # self.begin_ends.append((cur, cur+pn ** 2))
            # cur += pn ** 2
        for si, (ph, pw) in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + ph * pw))
            cur += ph * pw
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding from VQVAE
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C) # Embed VQVAE tokens. Input BLCvae, output BLC
        
        # 2. class embedding for each sample (like image)
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C) # Class embedding. [B, C] <- [B]. initial token [B,first_l,C]: sos=class_embed+pos_start 
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        # hk: VAR uses both absolute positional embeddings (fixed positions within the sequence) 
        # and level embeddings (hierarchical scale distinctions).
        pos_1LC = []
        # for i, pn in enumerate(self.patch_nums):
        #     pe = torch.empty(1, pn*pn, self.C)
        #     nn.init.trunc_normal_(pe, mean=0, std=init_std)
        #     pos_1LC.append(pe)
        for si, (ph, pw) in enumerate(self.patch_nums):
            pe = torch.empty(1, ph*pw, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # [1, L, C]
        assert tuple(pos_1LC.shape) == (1, self.L, self.C), f"pos_1LC shape {pos_1LC.shape} != (1, {self.L}, {self.C})" # L = sum(ph*pw for pn in patch_nums)
        self.pos_1LC = nn.Parameter(pos_1LC) # Position embedding for all levels where lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1][0]*self.patch_nums[si+1][1]] is the position embedding for level i
        # LEVEL embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C) # Level embedding, [n_levels, C]
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
            # workflow Add positional embeddings to level embeddings: x = level_emb + pos_1LC[:, :ed]
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([ # Transformer blocks
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        """
        # 1. Create d tensor:
        # For patch_nums=(1,2,3):
        # Scale 0 (1x1): 1 token    -> index 0
        # Scale 1 (2x2): 4 tokens   -> index 1
        # Scale 2 (3x3): 9 tokens   -> index 2
        d = [[0, 1,1,1,1, 2,2,2,2,2,2,2,2,2]]  # 1 + 4 + 9 = 14 tokens total
        dT = [[0],
            [1],
            [1],
            [1],
            [1],
            [2],
            [2],
            [2],
            [2],
            [2],
            [2],
            [2],
            [2],
            [2]]

        # attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        [[ 0  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞]  # Scale 0 token: can only attend to itself
        [ 0   0   0   0   0  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞]  # Scale 1 tokens: can attend to scale 0,1
        [ 0   0   0   0   0  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞]  # Scale 1
        [ 0   0   0   0   0  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞]  # Scale 1
        [ 0   0   0   0   0  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞  -∞]  # Scale 1
        [ 0   0   0   0   0   0   0   0   0   0   0   0   0   0]  # Scale 2 tokens: can attend to all scales
        [ 0   0   0   0   0   0   0   0   0   0   0   0   0   0]  # Scale 2
        [ 0   0   0   0   0   0   0   0   0   0   0   0   0   0]  # Scale 2
        [ 0   0   0   0   0   0   0   0   0   0   0   0   0   0]  # Scale 2
        [ 0   0   0   0   0   0   0   0   0   0   0   0   0   0]  # Scale 2
        [ 0   0   0   0   0   0   0   0   0   0   0   0   0   0]  # Scale 2
        [ 0   0   0   0   0   0   0   0   0   0   0   0   0   0]  # Scale 2
        [ 0   0   0   0   0   0   0   0   0   0   0   0   0   0]  # Scale 2
        [ 0   0   0   0   0   0   0   0   0   0   0   0   0   0]] # Scale 2
        - Each row represents a token that's doing the attending (source)
        - Each column represents a token that can be attended to (target)
        """
        # d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1) # (1, L, 1)
        d: torch.Tensor = torch.cat([torch.full((ph*pw,), i) for i, (ph, pw) in enumerate(self.patch_nums)]).view(1, self.L, 1) # (1, L, 1)
        # Example with patch_nums=[1,2,3]: d = [[0,1,1,1,1,2,2,2,2,2,2,2,2,2]] where 0 is scale 0 (1x1) and 1 is scale 1 (2x2) and 2 is scale 2 (3x3)
        dT = d.transpose(1, 2)    # dT: 1 x 1 x L
        lvl_1L = dT[:, 0].contiguous() # 1 x L
        self.register_buffer('lvl_1L', lvl_1L) # shape [1, L]
        # attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L) # -inf so softmax will be 0 in attention
        attn_bias_for_masking = torch.where(d >= dT, 0., -1e5).reshape(1, 1, self.L, self.L) # -inf so softmax will be 0 in attention
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        # Register the bias as a buffer (persistent state) for the module, so it's reused during training but doesn't require gradients.

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
            # If label_B is an integer, creates a tensor with the same class label for all samples in the batch.
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0)) #  sos [2B, C]
        # Get class embedding as start token.
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC # ( lvl_pos: [1, L, C] <- embed(lvl_1L [1,L])  ) + pos_1LC [1, L, C]
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        # next_token_map: [2B, first_l, C], first_l = self.patch_nums[0][0] * self.patch_nums[0][1]
        cur_L = 0
        # f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]) # Get class embedding as start token
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1][0], self.patch_nums[-1][1]) # Get class embedding as start token
        
        for b in self.blocks: b.attn.kv_caching(True)
        # for si, pn in enumerate(self.patch_nums):   # si: i-th segment
        for si, (ph, pw) in enumerate(self.patch_nums):   # si: i-th segment
            """
            # Process each scale sequentially
            cur = 0  # Current position in sequence
            for si, (ph, pw) in enumerate(self.patch_nums):
                l = ph * pw  # Number of tokens at this scale
                logits_BLV[:, cur:cur+l] = ...
                cur += l  # Move to next scale's starting position

            Scale 0 (1x1): → 1 token,  Scale 1 (2x2):   → 4 tokens, Scale 2 (3x3):  → 9 tokens

            Sequence: [0|1 1 1 1|2 2 2 2 2 2 2 2 2]  → 14 tokens total
                       ↑ ↑       ↑
                  Scale0 Scale1  Scale2     
            """
            ratio = si / self.num_stages_minus_1
            ## last_L = cur_L
            # cur_L += pn*pn
            cur_L += ph * pw
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD) # [B, 6*C] # Output from shared_ada_lin. if using shared_aln: [B, 1, 6, C] # Reshaped adaptive layer norm parameters
            x = next_token_map
            assert not torch.isnan(x).any(), x
            AdaLNSelfAttn.forward
            for b in self.blocks: # 1. Use transformer to predict next tokens
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)  # x: [B, L, C], L is total sequence length (1^2 + 2^2 + 3^2 = 14)
                # attn_bias: [1, 1, L, L] # Attention mask (None during inference)
                # # or when expanded for multiple heads:
                # attn_bias: [B*num_heads, L, L]

            logits_BlV = self.get_logits(x, cond_BD)
            # 2. Apply classifier-free guidance
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            # 3. Sample next tokens. Sample from these logits (rather than taking the argmax) for a few key reasons: 
            # Quality (staying close to the most likely tokens), Diversity (allowing some randomness)
            idx_Bl = sample_with_top_k_top_p_(logits_BlV,     
                rng=rng,      # Random generator for sampling
                top_k=top_k,  # Only sample from the k most likely tokens
                top_p=top_p,  # (nucleus sampling): Only sample from tokens that comprise the top p% of probability mass
                num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            # 4. Get embeddings and prepare for next scale
            # h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, ph, pw)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            assert not torch.isnan(f_hat).any(), f_hat
            assert not torch.isnan(next_token_map).any(), next_token_map
            # f_hat: [B, Cvae, H, W], next_token_map: [B, Cvae, h, w], where h = w = v_patch_nums[si+1]
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2) # hk: [B, L, Cvae]
                # next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                # print('lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1][0] * self.patch_nums[si+1][1]]', 
                #       lvl_pos[:, cur_L:cur_L + ph * pw].shape)
                # print('self.word_embed(next_token_map)', self.word_embed(next_token_map).shape)
                # import pdb; pdb.set_trace()
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1][0] * self.patch_nums[si+1][1]]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        # return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
        return self.vae_proxy[0].fhat_to_img(f_hat) #.add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        During Training process:
        1. Get ground truth tokens from VQVAE encoder (Algorithm 1)
        2. Feed tokens to transformer autoregressively
        3. Train to predict next token in sequence

        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, B: batch, L: Sequence length n_tokens, V is vocab_size
        """

        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B) # Get class embedding as start token, shape [B, C]
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1) # class Combine with position embeddings
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos_1LC: [1LC] add emb up to end level ed
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD) # cond_BD [B, D]: The conditioning embeddings derived from class labels (default behavior).
        #A layer or module (e.g., nn.Linear) that transforms cond_BD into gss parameters for Adaptive Layer Normalization.
                                                      # gss (gamma-scale-shift) [B, 6*C] or [B, 1, 6, C]: Adaptive layer normalization params gamma1, gamma2, scale1, scale2, shift1, shift2
        temp = x_BLC.new_ones(8, 8) # shape [8, 8]
        main_type = torch.matmul(temp, temp).dtype # shape [8, 8] = [8, 8] * [8, 8]
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks): # Pass through transformer blocks, 2 layers
            assert not torch.isnan(x_BLC).any(), x_BLC
            assert not torch.isnan(cond_BD_or_gss).any(), cond_BD_or_gss
            assert not torch.isnan(attn_bias).any(), attn_bias
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            assert not torch.isnan(x_BLC).any(), x_BLC
        x_BLC = self.get_logits(x_BLC.float(), cond_BD) # Predict next token logits
        assert not torch.isnan(x_BLC).any(), x_BLC
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
                print('s', s)
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


# class VARHF(VAR, PyTorchModelHubMixin):
#             # repo_url="https://github.com/FoundationVision/VAR",
#             # tags=["image-generation"]):
#     def __init__(
#         self,
#         vae_kwargs,
#         num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
#         norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
#         attn_l2_norm=False,
#         patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
#         flash_if_available=True, fused_if_available=True,
#     ):
#         vae_local = VQVAE(**vae_kwargs)
#         super().__init__(
#             vae_local=vae_local,
#             num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
#             norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
#             attn_l2_norm=attn_l2_norm,
#             patch_nums=patch_nums,
#             flash_if_available=flash_if_available, fused_if_available=fused_if_available,
#         )
