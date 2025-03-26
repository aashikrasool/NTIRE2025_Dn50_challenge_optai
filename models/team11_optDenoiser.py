import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]



class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map
    

class chrominance_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=6, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(chrominance_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img, ch):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        #mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,ch], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map



class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks1 = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks1.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

        self.blocks2 = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks2.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x1 = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks1:
            x1 = attn(x1, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x1
            x1 = ff(x1) + x1
        out = x1#.permute(0, 3, 1, 2)

        x2 = out#.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks2:
            x2 = attn(x2, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x2
            x2 = ff(x2) + x2
        out2 = x2.permute(0, 3, 1, 2)

        


        return out2 + x


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out




class Denoiser2img(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser2img, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        #xcat = (x+x2)#torch.cat((x, x2), 1)
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) #+ x + x2

        return out
    

class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img,illu_fea)

        return output_img

class RetinexFormer_Single_Refine(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Refine, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.estimator2 = Illumination_Estimator(n_feat)
        self.att = nn.Conv2d(80, 40 , 1, 1, bias=False)
        self.denoiser = Denoiser2img(in_dim=3,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img1, img2, inp):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img1)
        input_img = img1 * illu_map + img1

        illu_fea2, illu_map2 = self.estimator2(img2)
        input_img2 = img2 * illu_map2 + img2

        #input_img = torch.cat((input_img, input_img2), 1)
        illu_fea = torch.cat((illu_fea, illu_fea2), 1)
        illu_fea = self.att(illu_fea)
        output_img = self.denoiser(input_img, input_img2, illu_fea)

        return (output_img  + img1 + img2 + inp)


class RetinexFormer_AOAB(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_AOAB, self).__init__()
        
        self.estimator = Illumination_Estimator(n_feat)
        self.estimator2 = Illumination_Estimator(n_feat)
        self.aoab = AOA(40)
        self.input1_proj = nn.Conv2d(3, 40 , 3, 1, 1)
        self.input1_proj2 = nn.Conv2d(3, 40 , 3, 1, 1)
        self.att = nn.Conv2d(6, 3 , 7, 1,3, bias=False)
        self.denoiser = Denoiser2img(in_dim=3,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img1, img2):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img1)
        input_img = img1 * illu_map + img1

        illu_fea2, illu_map2 = self.estimator2(img2)
        input_img2 = img2 * illu_map2 + img2

        input_img1_proj = self.input1_proj(input_img)
        input_img2_proj = self.input1_proj2(input_img2)
        #print(input_img1_proj.shape, illu_fea2.shape)
        aoa = self.aoab(input_img1_proj, input_img2_proj, illu_fea, illu_fea2)
        # print(aoa.shape, illu_fea.shape)

        input_img = torch.cat((input_img, input_img2), 1)
        #illu_fea = torch.cat((illu_fea, illu_fea2), 1)
        #print(input_img.shape)
        input_boost = self.att(input_img)
        #print(input_boost.shape, aoa.shape)
        
        output_img = self.denoiser(input_boost, aoa )
        #print(output_img.shape)
        return (output_img  + img1 + img2 )
    


class RetinexFormer_2stageAOAB(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, stage=1, num_blocks=[1, 1, 1]):
        super(RetinexFormer_2stageAOAB, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
        
        self.estimator = Illumination_Estimator(n_feat)
        self.estimator2 = Illumination_Estimator(n_feat)
        self.aoab = AOA(40)
        self.input1_proj = nn.Conv2d(3, 40 , 3, 1, 1)
        self.input1_proj2 = nn.Conv2d(3, 40 , 3, 1, 1)
        self.att = nn.Conv2d(6, 3 , 7, 1,3, bias=False)
        self.denoiser = Denoiser2img(in_dim=3,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img1):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        recon = self.body(img1)

        illu_fea, illu_map = self.estimator(img1)
        input_img = img1 * illu_map + img1

        illu_fea2, illu_map2 = self.estimator2(recon)
        input_img2 = recon * illu_map2 + recon

        input_img1_proj = self.input1_proj(input_img)
        input_img2_proj = self.input1_proj2(input_img2)
        #print(input_img1_proj.shape, illu_fea2.shape)
        aoa = self.aoab(input_img1_proj, input_img2_proj, illu_fea, illu_fea2)
        # print(aoa.shape, illu_fea.shape)

        input_img = torch.cat((input_img, input_img2), 1)
        #illu_fea = torch.cat((illu_fea, illu_fea2), 1)
        #print(input_img.shape)
        input_boost = self.att(input_img)
        #print(input_boost.shape, aoa.shape)
        
        output_img = self.denoiser(input_boost, aoa )
        #print(output_img.shape)
        return (recon + img1), (output_img  + recon)
    


class AOA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=40,
            heads=1,
            num_blocks=1,
    ):
        super().__init__()
        self.chrom = IG_MSA(dim, dim_head, heads)
        self.c_norm =  nn.LayerNorm(dim)#PreNorm(dim, FeedForward(dim=dim))
        self.lum = IG_MSA(dim, dim_head, heads)
        self.l_norm =  nn.LayerNorm(dim)#PreNorm(dim, FeedForward(dim=dim))
        self.cross = IG_MSA(dim, dim_head, heads)
        self.cr_norm =  nn.LayerNorm(dim)#PreNorm(dim, FeedForward(dim=dim))

    def forward(self, lum, chrom,  illu_fea, chrom_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        
        c = self.c_norm(self.chrom(chrom.permute(0, 2, 3, 1), chrom_fea.permute(0, 2, 3, 1)))
        l = self.l_norm(self.lum(lum.permute(0, 2, 3, 1), illu_fea.permute(0, 2, 3, 1)))
        out = self.cr_norm(self.cross(c,l))
        #print(lum.shape, out.shape)
        return out.permute(0,3, 1, 2)
    



class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1]):
        super(RetinexFormer, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        return out


if __name__ == '__main__':
    #from fvcore.nn import FlopCountAnalysis
    model = RetinexFormer_2stageAOAB(n_feat=40,num_blocks=[1,2,2]).cuda()
    #print(model)
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    outputs = model(inputs, inputs)
    #flops = FlopCountAnalysis(model,inputs)
    #n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    #print(f'GMac:{flops.total()/(1024*1024*1024)}')
    #rint(f'Params:{n_param}')