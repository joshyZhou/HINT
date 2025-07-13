## HINT
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.cluster import SpectralClustering
import warnings

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



##########################################################################
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
    
class Inter_CacheModulation(nn.Module):
    def __init__(self, in_c=3):
        super(Inter_CacheModulation, self).__init__()

        self.align = nn.AdaptiveAvgPool2d(in_c)
        self.conv_width = nn.Conv1d(in_channels=in_c, out_channels=2*in_c, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=1)

    def forward(self, x1,x2):
        C = x1.shape[-1]
        x2_pW = self.conv_width(self.align(x2)+x1)
        scale,shift = x2_pW.chunk(2, dim=1)
        x1_p = x1*scale+shift
        x1_p = x1_p * F.gelu(self.gatingConv(x1_p))
        return x1_p


class Intra_CacheModulation(nn.Module):
    def __init__(self,embed_dim=48):
        super(Intra_CacheModulation, self).__init__()

        self.down = nn.Conv1d(embed_dim, embed_dim//2, kernel_size=1)
        self.up = nn.Conv1d(embed_dim//2, embed_dim, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)


    def forward(self, x1,x2):
        x_gated = F.gelu(self.gatingConv(x2+x1)) * (x2+x1)
        x_p = self.up(self.down(x_gated))  
        return x_p

class ReGroup(nn.Module):
    def __init__(self, groups=[1,1,2,4]):
        super(ReGroup, self).__init__()
        self.gourps = groups

    def forward(self, query,key,value):
        C = query.shape[1]
        channel_features = query.mean(dim=0)
        correlation_matrix = torch.corrcoef(channel_features)

        mean_similarity = correlation_matrix.mean(dim=1)
        _, sorted_indices = torch.sort(mean_similarity, descending=True) 

        query_sorted = query[:, sorted_indices, :]
        key_sorted = key[:, sorted_indices, :]
        value_sorted = value[:, sorted_indices, :]

        query_groups = []
        key_groups = []
        value_groups = []
        start_idx = 0
        total_ratio = sum(self.gourps)
        group_sizes = [int(ratio / total_ratio * C) for ratio in self.gourps]

        for group_size in group_sizes:
            end_idx = start_idx + group_size
            query_groups.append(query_sorted[:, start_idx:end_idx, :])  
            key_groups.append(key_sorted[:, start_idx:end_idx, :])  
            value_groups.append(value_sorted[:, start_idx:end_idx, :])  
            start_idx = end_idx

        return query_groups,key_groups,value_groups


def CalculateCurrentLayerCache(x,dim=128,groups=[1,1,2,4]):
    lens = len(groups)
    ceil_dim = dim #* max_value // sum_value 
    for i in range(lens):
        qv_cache_f = x[i].clone().detach()
        qv_cache_f=torch.mean(qv_cache_f,dim=0,keepdim=True).detach()
        update_elements = F.interpolate(qv_cache_f.unsqueeze(1), size=(ceil_dim, ceil_dim), mode='bilinear', align_corners=False)
        c_i = qv_cache_f.shape[-1]
                
        if i==0:
            qv_cache = update_elements * c_i // dim
        else:
            qv_cache = qv_cache + update_elements * c_i // dim
                
    return qv_cache.squeeze(1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.group =[1,2,2,3] 

        self.intra_modulator = Intra_CacheModulation(embed_dim=dim)

        self.inter_modulator1 = Inter_CacheModulation(in_c=1*dim//8)
        self.inter_modulator2 = Inter_CacheModulation(in_c=2*dim//8)
        self.inter_modulator3 = Inter_CacheModulation(in_c=2*dim//8)
        self.inter_modulator4 = Inter_CacheModulation(in_c=3*dim//8)
        self.inter_modulators = [self.inter_modulator1,self.inter_modulator2,self.inter_modulator3,self.inter_modulator4]

        self.regroup = ReGroup(self.group)
        self.dim=dim

    def forward(self, x ,qv_cache=None):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
    
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        qu,ke,va = self.regroup(q,k,v)
        attScore = []
        tmp_cache=[]
        for index in range(len(self.group)):

            query_head = qu[index]
            key_head   = ke[index]

            query_head = torch.nn.functional.normalize(query_head, dim=-1)
            key_head = torch.nn.functional.normalize(key_head, dim=-1)

            attn = (query_head @ key_head.transpose(-2, -1)) * self.temperature[index,:,:]
            attn = attn.softmax(dim=-1)

            attScore.append(attn)#CxC
            t_cache = query_head.clone().detach()+key_head.clone().detach()
            tmp_cache.append(t_cache)
        
        tmp_caches = torch.cat(tmp_cache, 1)
        # Inter Modulation
        out=[]
        if qv_cache is not None:
            if qv_cache.shape[-1]!=c:
                
                qv_cache = F.adaptive_avg_pool2d(qv_cache,c)
        for i in range(4):
            if qv_cache is not None:
                inter_modulator = self.inter_modulators[i]
                attScore[i] = inter_modulator(attScore[i],qv_cache)+attScore[i]
                out.append(attScore[i] @ va[i])
            else:
                out.append(attScore[i] @ va[i])
                
        update_factor=0.9
        if qv_cache is not None:
            
            update_elements = CalculateCurrentLayerCache(attScore,c,self.group)
            qv_cache = qv_cache*update_factor + update_elements*(1-update_factor)
        else:
            qv_cache = CalculateCurrentLayerCache(attScore,c,self.group)
            qv_cache = qv_cache*update_factor

        out_all = torch.concat(out, 1)
        # Intra Modulation
        out_all = self.intra_modulator(out_all,tmp_caches)+out_all

        out_all = rearrange(out_all, 'b  c (h w) -> b c h w', h=h, w=w)
        out_all = self.project_out(out_all)
        return [out_all,qv_cache]



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, isAtt):
        super(TransformerBlock, self).__init__()
        self.isAtt = isAtt
        if self.isAtt:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self,inputs):
        x = inputs[0]
        qv_cache = inputs[1]
        if self.isAtt:
            x_tmp = x
            [x_att,qv_cache] = self.attn(self.norm1(x),qv_cache=qv_cache)
            x = x_tmp + x_att
        x = x + self.ffn(self.norm2(x))

        return [x,qv_cache]




class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x




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




class HINT(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        dual_pixel_task = False,
        qv_cache=None
    ):

        super(HINT, self).__init__()

        self.qv_cache=qv_cache

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=False) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=False) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=False) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[1])])
        
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_refinement_blocks)])
        

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)

            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1,self.qv_cache = self.encoder_level1([inp_enc_level1,self.qv_cache])
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2,self.qv_cache = self.encoder_level2([inp_enc_level2,self.qv_cache])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3,self.qv_cache = self.encoder_level3([inp_enc_level3,self.qv_cache]) 

        inp_enc_level4 = self.down3_4(out_enc_level3) 
        latent,self.qv_cache = self.latent([inp_enc_level4,self.qv_cache])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3,self.qv_cache = self.decoder_level3([inp_dec_level3,self.qv_cache]) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2,self.qv_cache = self.decoder_level2([inp_dec_level2,self.qv_cache]) 


        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1,self.qv_cache = self.decoder_level1([inp_dec_level1,self.qv_cache])

        
        out_dec_level1,self.qv_cache = self.refinement([out_dec_level1,self.qv_cache])

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

