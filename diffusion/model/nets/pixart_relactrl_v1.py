import re
import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.nn import Module, Linear, init
import torch.nn.functional as F
from typing import Any, Mapping
from timm.models.vision_transformer import PatchEmbed, Mlp

from diffusion.model.nets import PixArtMSBlock, PixArtMS, PixArt
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed, WindowAttention
from diffusion.model.utils import auto_grad_checkpoint


class TDSM_token(nn.Module):
    def __init__(self, moe_hidden_size, token_dim=4096, **block_kwargs):
        super(TDSM_token, self).__init__()

        self.moe_hidden_size = moe_hidden_size
        self.att_moe_token = 4 

        self.attn = WindowAttention(
            moe_hidden_size, num_heads=4, qkv_bias=True, **block_kwargs
        )


    def forward(self, x, expertCoef=1):
        N, L, D = x.size()
        x_flattened = x.view(N, -1)

        # Randomly shuffled index
        total_length = L * D
        shuffle_indices = torch.randperm(total_length, device=x.device)
        shuffled = x_flattened[:, shuffle_indices]

        segment_length = total_length // self.att_moe_token
        split_tensors = []
        for i in range(self.att_moe_token):
            start = i * segment_length
            end = start + segment_length
            split_tensors.append(shuffled[:, start:end].view(N, segment_length // D, D))
        processed_segments = [self.attn(segment) for segment in split_tensors]
        processed_flattened = torch.cat([segment.view(N, -1) for segment in processed_segments], dim=1)

        # Restore in original index order
        unshuffle_indices = torch.argsort(shuffle_indices.float()).long()
        output_flattened = processed_flattened[:, unshuffle_indices]
        output_tensor = output_flattened.view(N, L, D)
        return expertCoef * output_tensor


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def distribute_dim_4_TDSM_token(dim, n, num_heads=4, multi=False, decay_factor=0.9):
    base_dim = dim // n
    remainder = dim % n
    dim_list = [base_dim + 1 if i < remainder else base_dim for i in range(n)]

    if multi:
        for i in range(1, n):
            dim_list[i] = max(1, int(dim_list[i - 1] * decay_factor))

        total_dim = sum(dim_list)
        if total_dim != dim:
            dim_list[-1] += (dim - total_dim)

    block_size_multiple = 8 * num_heads
    dim_list = [elem + (block_size_multiple - elem % block_size_multiple) % block_size_multiple for elem in dim_list]
    total_dim = sum(dim_list)
    diff = dim - total_dim

    i = 0
    while diff != 0 and i < n:
        adjustment = block_size_multiple if diff > 0 else -block_size_multiple
        if (dim_list[i] + adjustment > 0) and ((dim_list[i] + adjustment) % block_size_multiple == 0):
            dim_list[i] += adjustment
            diff -= adjustment
        i = (i + 1) % n
    return dim_list


class TDSM(Module):
    def __init__(self, block_index, dim=1152, att_moe_num=6 , drop_path=0., **block_kwargs):
        super().__init__()

        self.block_index = block_index
        self.hidden_sizes_attn = distribute_dim_4_TDSM_token(dim = dim, n = att_moe_num, multi=False)
        if sum(self.hidden_sizes_attn) != dim:
            raise ValueError(f"The sum of the hidden_sizes of the experts must be {dim}, and currently sums to {sum(self.hidden_sizes_attn)}")
        else:
            print(f"The attn expert of block {self.block_index} is set to {self.hidden_sizes_attn}")
        self.experts_attn = nn.ModuleList([TDSM_token(moe_hidden_size = hidden_sizes_attn) for hidden_sizes_attn in self.hidden_sizes_attn])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / (dim ** 0.5))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        
    def forward(self, c, y, t, mask=None):

        B, N, dim = c.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        c_ori = c
        c = t2i_modulate(self.norm(c), shift_msa, scale_msa)
        moe_num_attn = len(self.hidden_sizes_attn)
        selected_indices = []
        split_tensors = []

        # Randomly select channel index
        all_indices = torch.randperm(c.size(-1)).tolist()
        start_index = 0

        for hidden_size in self.hidden_sizes_attn:
            end_index = start_index + hidden_size
            selected_indices.append(all_indices[start_index:end_index])
            split_tensors.append(c[:, :, torch.tensor(selected_indices[-1], dtype=torch.long)])
            start_index = end_index

        for i in range(moe_num_attn):
            split_tensors[i] = self.experts_attn[i](split_tensors[i])
        output_tensor = torch.empty_like(c)
        for i, indices in enumerate(selected_indices):
            output_tensor[:, :, indices] = split_tensors[i].to(output_tensor.dtype)

        c = c_ori + self.drop_path(gate_msa * output_tensor.reshape(B, N, dim))
        return c
        
class RGLC_Block(Module):
    def __init__(self, dim=1152, block_index=0 ,drop_path=0., **block_kwargs):
        super().__init__()

        self.block_index = block_index

        prior2 = [6, 2, 6, 4, 2, 2, 4, 4, 6, 6, 6]        
        relactrl_prior2 = prior2[block_index]

        self.process = TDSM(self.block_index, dim, relactrl_prior2)

        self.before_proj = Linear(dim, dim)
        init.zeros_(self.before_proj.weight)
        init.zeros_(self.before_proj.bias)

        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)
        
    def forward(self, x, y, t, mask=None, c=None):
        if self.block_index == 0:
            c = self.before_proj(c)
            c = self.process(x + c, y, t, mask)
            c_skip = self.after_proj(c)
        else:
            x_trans = self.before_proj(x)
            c = self.process(c + x_trans, y, t, mask)
            c_skip = self.after_proj(c)
        return c, c_skip


# The implementation of ControlPixArtHalf net
class ControlPixArtHalf_RelaCtrl(Module):
    # only support single res model
    def __init__(self, base_model: PixArt, copy_blocks_num: int = 13) -> None:
        super().__init__()
        self.base_model = base_model.eval()
        self.hidden_size = 1152 # x.size = torch.Size([6, 4096, 1152])
        self.controlnet = []
        self.copy_blocks_num = copy_blocks_num
        self.total_blocks_num = len(base_model.blocks)
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        # Copy first copy_blocks_num block
        for i in range(self.copy_blocks_num):
            self.controlnet.append(RGLC_Block(self.hidden_size, block_index=i))
        self.controlnet = nn.ModuleList(self.controlnet)
    
    def __getattr__(self, name: str) -> Tensor or Module:
        if name in ['forward', 'forward_with_dpmsolver', 'forward_with_cfg', 'forward_c', 'load_state_dict']:
            return self.__dict__[name]
        elif name in ['base_model', 'controlnet']:
            return super().__getattr__(name)
        else:
            return getattr(self.base_model, name)

    def forward_c(self, c):
        # print(c.size())
        self.h, self.w = c.shape[-2]//self.patch_size, c.shape[-1]//self.patch_size
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(c.device).to(self.dtype)
        return self.x_embedder(c) + pos_embed if c is not None else c


    def forward(self, x, timestep, y, mask=None, data_info=None, c=None, **kwargs):
        # modify the original PixArtMS forward function
        if c is not None:
            c = c.to(self.dtype)
            c = self.forward_c(c)
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        c_adding_index = [0, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19]
        # define the first layer
        x = auto_grad_checkpoint(self.base_model.blocks[0], x, y, t0, y_lens, **kwargs)  # (N, T, D) #support grad checkpoint

        if c is not None:
            # update c
            for index in range(1, self.total_blocks_num):
                if index - 1 in c_adding_index:
                    controlnet_index = c_adding_index.index(index - 1)
                    c, c_skip = auto_grad_checkpoint(self.controlnet[controlnet_index], x, y, t0, y_lens, c, **kwargs)
                x = auto_grad_checkpoint(self.base_model.blocks[index], x + c_skip, y, t0, y_lens, **kwargs)
        else:
            for index in range(1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, t, y, data_info, c, **kwargs):
        model_out = self.forward(x, t, y, data_info=data_info, c=c, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, t, y, cfg_scale, data_info, c, **kwargs):
        return self.base_model.forward_with_cfg(x, t, y, cfg_scale, data_info, c=self.forward_c(c), **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if all((k.startswith('base_model') or k.startswith('controlnet')) for k in state_dict.keys()):
            return super().load_state_dict(state_dict, strict)
        else:
            new_key = {}
            for k in state_dict.keys():
                new_key[k] = re.sub(r"(blocks\.\d+)(.*)", r"\1.base_block\2", k)
            for k, v in new_key.items():
                if k != v:
                    print(f"replace {k} to {v}")
                    state_dict[v] = state_dict.pop(k)

            return self.base_model.load_state_dict(state_dict, strict)
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    @property
    def dtype(self):
        return next(self.parameters()).dtype


# The implementation for PixArtMS_Half + 1024 resolution
class ControlPixArtMSHalf_RelaCtrl(ControlPixArtHalf_RelaCtrl):
    # support multi-scale res model (multi-scale model can also be applied to single reso training & inference)
    def __init__(self, base_model: PixArtMS, copy_blocks_num: int = 11) -> None:
        super().__init__(base_model=base_model, copy_blocks_num=copy_blocks_num)

    def forward(self, x, timestep, y, mask=None, data_info=None, c=None, **kwargs):
        # modify the original PixArtMS forward function
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        if c is not None:
            c = c.to(self.dtype)
            c = self.forward_c(c)
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size

        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(x.device).to(self.dtype)
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)  # (N, D)
        csize = self.csize_embedder(c_size, bs)  # (N, D)
        ar = self.ar_embedder(ar, bs)  # (N, D)
        t = t + torch.cat([csize, ar], dim=1)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        c_adding_index = [0, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19]
        # define the first layer
        x = auto_grad_checkpoint(self.base_model.blocks[0], x, y, t0, y_lens, **kwargs)  # (N, T, D) #support grad checkpoint

        if c is not None:
            # update c
            for index in range(1, self.total_blocks_num):
                if index - 1 in c_adding_index:
                    controlnet_index = c_adding_index.index(index - 1)
                    c, c_skip = auto_grad_checkpoint(self.controlnet[controlnet_index], x, y, t0, y_lens, c, **kwargs)
                x = auto_grad_checkpoint(self.base_model.blocks[index], x + c_skip, y, t0, y_lens, **kwargs)
        else:
            for index in range(1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x
