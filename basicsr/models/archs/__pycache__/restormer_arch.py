## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

#add new dependencies
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # print(B, H, W, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

#steal from swinIR
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution=128, num_heads=6, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # # mlp_hidden_dim = int(dim * mlp_ratio)
        # if self.shift_size > 0:
        #     attn_mask = self.calculate_mask(self.input_resolution)
        # else:
        #     attn_mask = None

        # self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        # H, W = x_size

        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        # print(B,L,C,H)
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # if self.input_resolution == x_size:
        #     attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # else:
        #     attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
        attn_windows = self.attn(x_windows, mask=None)
            

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # # reverse cyclic shift
        # if self.shift_size > 0:
        #     x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # else:
        #     x = shifted_x
        x = shifted_x
        
        x = x.view(B, H * W, C)

        # # FFN
        # x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    # def extra_repr(self) -> str:
    #     return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
    #            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

# ##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
## MDMLP
class MDMLP(nn.Module):
    def __init__(self, dim, height, width, bias, LayerNorm_type):
        super(MDMLP, self).__init__()
     
        # hidden_features = int(dim*ffn_expansion_factor)

        # self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv_c = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv_h = nn.Conv2d(height, height, kernel_size=3, stride=1, padding=1, groups=height, bias=bias)
        self.dwconv_w = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, groups=width, bias=bias)
        self.norm_c = LayerNorm(dim, LayerNorm_type)
        self.norm_h = LayerNorm(height, LayerNorm_type)
        self.norm_w = LayerNorm(width, LayerNorm_type)

        # self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        #input : b c h w
        x = x + self.dwconv_c(self.norm_c(x))
        x = rearrange(x, ' b c h w -> b h w c')
        x = x + self.dwconv_h(self.norm_h(x))
        x = rearrange(x, ' b h w c -> b w c h')
        x = x + self.dwconv_w(self.norm_w(x))
        x = rearrange(x, ' b w c h -> b c h w')

        # x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x = F.gelu(x1) * x2
        # x = self.project_out(x)
        return x
## pallal multi-scale feature learning PMFLFFN
class PMFLFFN(nn.Module):
    def __init__(self, dim,bias,):
        super(PMFLFFN, self).__init__()
     
        # hidden_features = int(dim*ffn_expansion_factor)

        # self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.dwconv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        # self.norm_c = LayerNorm(dim, LayerNorm_type)
        # self.norm_h = LayerNorm(height, LayerNorm_type)
        # self.norm_w = LayerNorm(width, LayerNorm_type)
        self.relu = nn.ReLU()
        self.project_out = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias)


        # self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        #input : b c h w
        x_3 = self.dwconv_3(x)
        x_5 = self.dwconv_5(x)
        x_7 = self.dwconv_7(x)

        #planA- mul on each other

        #planB-pallal-concat then flatten and 1x1 pro
        x_cat = torch.cat([x_3, x_5, x_7], dim=1)
        x = self.project_out(self.relu(x_cat))

        #planC-pallal-concat then flatten and point-wise 

        #planD-sequence- then 1x1 pro

        #PlanE- sequnence then point-wise fuse

        #1x1 pro can be replaced by point-wise 

        # x = x + self.dwconv_c(x)
        # x = rearrange(x, ' b c h w -> b h w c')
        # x = x + self.dwconv_h(self.norm_h(x))
        # x = rearrange(x, ' b h w c -> b w c h')
        # x = x + self.dwconv_w(self.norm_w(x))
        # x = rearrange(x, ' b w c h -> b c h w')

        # x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x = F.gelu(x1) * x2
        # x = self.project_out(x)
        return x
    
# #borrow from FasterNet
# class Partial_conv3(nn.Module):

#     def __init__(self, dim, n_div, forward):
#         super().__init__()
#         self.dim_conv3 = dim // n_div
#         self.dim_untouched = dim - self.dim_conv3
#         self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

#         if forward == 'slicing':
#             self.forward = self.forward_slicing
#         elif forward == 'split_cat':
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError

#     def forward_slicing(self, x: Tensor) -> Tensor:
#         # only for inference
#         x = x.clone()   # !!! Keep the original input intact for the residual connection later
#         x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

#         return x

#     def forward_split_cat(self, x: Tensor) -> Tensor:
#         # for training/inference
#         x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
#         x1 = self.partial_conv3(x1)
#         x = torch.cat((x1, x2), 1)

#         return x
    
class MPartial_conv3InFFN(nn.Module):

    def __init__(self, dim, n_div=4):
        super().__init__()
        # self.dim_conv3 = dim // n_div
        self.dim_conv = dim // 4
        self.dim_untouched = dim - self.dim_conv * 3
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        self.partial_conv5 = nn.Conv2d(self.dim_conv, self.dim_conv, 5, 1, 2, bias=False)
        self.partial_conv7 = nn.Conv2d(self.dim_conv, self.dim_conv, 7, 1, 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # for training/inference
        x1, x2, x3, x4 = torch.split(x, [self.dim_conv,self.dim_conv,self.dim_conv,self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x2 = self.partial_conv5(x2)
        x3 = self.partial_conv7(x3)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = x + self.project_out(x)#new add

        return x
########### feed-forward network #############
class LeFF(nn.Module):
    def __init__(self, dim=32, ffn_expansion_factor=1, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = int(dim*ffn_expansion_factor)
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        # self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()
        self.act_1 = nn.GELU()
        # self.act_2 = 

    def forward(self, x):
        # bs x hw x c
        b,c,h,w = x.shape
        # hh = int(math.sqrt(hw))
        x = rearrange(x, ' b c h w -> b h w c ')

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b h w c -> b c h w ')
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b h w c')

        x = self.linear2(x)

        x = rearrange(x, ' b h w c -> b c h w ')

        # x = self.eca(x)

        return x
##########################################################################
## MLP
# class MLP(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(MLP, self).__init__()

#         hidden_features = int(dim*ffn_expansion_factor)

#         self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

#         self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x)
#         # x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         # x = F.gelu(x1) * x2
#         x = self.dwconv(x)
#         x = self.project_out(x)
#         return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class SimpleFFN(nn.Module):
    def __init__(self, dim, bias):
        super(SimpleFFN, self).__init__()
        # self.dwconv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # self.dwconv_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # self.concat_conv = nn.Conv2d(dim*2, dim*2, kernel_size=1, bias=False)

    def forward(self, x):
        # x = self.concat_conv(torch.cat([x1, x2], dim=1))
        input= x
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = input + self.conv2(x1*x2)
        # x1 = self.dwconv_1(x1) * self.dwconv_2(x2)
        # return self.dwconv_1(x1) * self.dwconv_2(x2)
        return x



##########################################################################
#borrow from poolformer
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=(1, 1), bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, (1, 1), bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, (kernel_size, kernel_size), padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, ratio=16):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes, ratio=ratio)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, pool_size, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention(dim, num_heads, bias)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = MLP(dim, ffn_expansion_factor, bias)
        self.ffn = LeFF(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
class TransformerBlock_v0(nn.Module):
    #cancel the token-mixer part
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_v0, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        # self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = MLP(dim, ffn_expansion_factor, bias)
        # self.ffn = LeFF(dim, ffn_expansion_factor)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # x = x + self.token_mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
##########################################################################
class TransformerBlock_v00(nn.Module):
    #cancel the token-mixer part
    def __init__(self,h,w, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_v00, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        # self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = MLP(dim, ffn_expansion_factor, bias)
        # self.ffn = LeFF(dim, ffn_expansion_factor)
        # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.ffn =MDMLP(dim,h,w,bias,LayerNorm_type)
        # self.ffn =PMFLFFN(dim,bias)
        self.ffn = MPartial_conv3InFFN(dim,bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # x = x + self.token_mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        # x = x + self.ffn(x)

        return x
class TransformerBlock_v11(nn.Module):
    #cancel the token-mixer part
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_v11, self).__init__()

        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention(dim, num_heads, bias)
        # self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = MLP(dim, ffn_expansion_factor, bias)
        # self.ffn = LeFF(dim, ffn_expansion_factor)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # x = x + self.attn(self.norm1(x))
        # x = x + self.token_mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
class TransformerBlock_v111(nn.Module):
    #cancel the token-mixer part/ new MDMLP
    def __init__(self,h,w, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_v111, self).__init__()

        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention(dim, num_heads, bias)
        # self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = MLP(dim, ffn_expansion_factor, bias)
        # self.ffn = LeFF(dim, ffn_expansion_factor)
        # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.ffn =MDMLP(dim,h,w,bias,LayerNorm_type)
        # self.ffn =PMFLFFN(dim,bias)
        self.ffn = MPartial_conv3InFFN(dim,bias)

    def forward(self, x):
        # x = x + self.attn(self.norm1(x))
        # x = x + self.token_mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        # x = x + self.ffn(x)

        return x
        

##########################################################################
class TransformerBlock_v1(nn.Module):
    #cancel the token-mixer part
    def __init__(self, dim, pool_size, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_v1, self).__init__()

        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention(dim, num_heads, bias)
        # self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = MLP(dim, ffn_expansion_factor, bias)
        self.ffn = LeFF(dim, ffn_expansion_factor)

    def forward(self, x):
        # x = x + self.token_mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    
class TransformerBlock_v2(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_v2, self).__init__()

        #B, C, H, W for cbam
        # self.att_c = BasicBlock(dim, dim, ratio=16)


        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)

        # self.window_size = 8
        # self.norm_wmsa = LayerNorm(dim, LayerNorm_type)
        # self.attn_w = SwinTransformerBlock(dim=dim, input_resolution=128,
        #                          num_heads=num_heads, window_size=8,
        #                          shift_size=0,
        #                          qkv_bias=True, 
        #                          norm_layer=nn.LayerNorm)
        # self.attn_w = WindowAttention(dim, window_size=to_2tuple(8), num_heads=6)#default setting in swinIR

        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.ffn = LeFF(dim, ffn_expansion_factor)
        # self.concat_conv = nn.Conv2d(dim*2, dim, kernel_size=1, bias=False)
        self.ffn = SimpleFFN(dim,bias)
        # self.ffn =MDMLP(dim,h,w,bias,LayerNorm_type)

    def forward(self, x):
        # x_bak  = x
        x = x + self.attn(self.norm1(x))
        
        #x: b c h w 
        # h,w = x_bak.shape[-2:]
        # x_bak = x_bak + self.att_c((x_bak))#seq add cbam
        # print("x shape(before w/c token-mixer):",x_bak.shape)
        # x_bak = to_3d(x_bak)
        # x = to_3d(x) + self.attn_w(to_3d(x),(h,w))#seq add w-msa
        # x_bak = self.attn_w(to_3d(self.norm2(x_bak)),(h,w))#seq add w-msa

        # x = to_4d(x, h, w)
        # x = x+x_bak
         
        # x = self.ffn(x)
        # print("x.size:",x.shape)
        # print("x_bak.size:",x_bak.shape)
        # x = self.concat_conv(torch.cat([x, x_bak], dim=1))


        x = self.ffn(self.norm2(x))
        # x = self.ffn(x)
        # print("x.size:",x.shape)
        # x = self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        input_res = 128,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()
        h = w = input_res

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # self.encoder_level1 = nn.Sequential(*[TransformerBlock_v1(dim=dim, pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1 = nn.Sequential(*[TransformerBlock_v111(dim=dim,h=h,w=w, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
                
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        h = w = input_res//2
        # self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        # self.encoder_level2 = nn.Sequential(*[TransformerBlock_v1(dim=int(dim*2**1), pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2 = nn.Sequential(*[TransformerBlock_v111(dim=int(dim*2**1),h=h,w=w, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
                
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        h = w = input_res//4
        # self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        # self.encoder_level3 = nn.Sequential(*[TransformerBlock_v1(dim=int(dim*2**2), pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3 = nn.Sequential(*[TransformerBlock_v111(dim=int(dim*2**2),h=h,w=w, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        h = w = input_res//8
        # self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent = nn.Sequential(*[TransformerBlock_v00(dim=int(dim*2**3),h=h,w=w, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        h = w = input_res//4
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        # self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.decoder_level3 = nn.Sequential(*[TransformerBlock_v00(dim=int(dim*2**2),h=h,w=w, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        h = w = input_res//2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        # self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2 = nn.Sequential(*[TransformerBlock_v00(dim=int(dim*2**1),h=h,w=w, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])


        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        h = w = input_res

        # self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1 = nn.Sequential(*[TransformerBlock_v00(dim=int(dim*2**1),h=h,w=w, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
       
        # self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), pool_size=3, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement = nn.Sequential(*[TransformerBlock_v00(dim=int(dim*2**1),h=h,w=w, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
 
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # print("input size:",inp_img.shape)

        inp_enc_level1 = self.patch_embed(inp_img)
        # print("inp_enc_level1 size:",inp_enc_level1.shape)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # print("out_enc_level1 size:",out_enc_level1.shape)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        # print("inp_enc_level2 size:",inp_enc_level2.shape)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        # print("out_enc_level2 size:",out_enc_level2.shape)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        # print("inp_enc_level3 size:",inp_enc_level3.shape)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        # print("out_enc_level3 size:",out_enc_level3.shape)

        inp_enc_level4 = self.down3_4(out_enc_level3)     
        # print("inp_enc_level4 size:",inp_enc_level4.shape)   
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

