import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from timm.models.layers import DropPath, trunc_normal_
from thop import profile
from torch.autograd import Variable
import numpy as np
import copy
import unfoldNd
# helpers


def exists(val):
    return val is not None


def conv_output_size(image_size, kernel_size, stride, padding=0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num


# classes
class RearrangeImage1(nn.Module):
    def __init__(self, data_type):
        super().__init__()
        self.data_type=data_type

    def forward(self, x):
        # print(x.shape, pow(x.shape[1],1/2))
        if self.data_type == "ESCC":
            return rearrange(x, 'b (d h w) c -> b c d h w', h=round(pow(int(x.shape[1]*4/9),1/3)*3/2),
                         w=round(pow(int(x.shape[1]*4/9),1/3)*3/2))     #(32,48,48))

class unfd(nn.Module):
    def __init__(self,kernel_size,stride,padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        #self.unfold = unfoldNd.unfoldNd(x,kernel_size,stride,padding)

    def forward(self,x):
        return unfoldNd.unfoldNd(x,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)

def pair(t):
    # 把t变成一对输出
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, dropout=0.):
        super().__init__()
        # 这个前传过程其实就是几层全连接
        """ print(in_dim)
        print(hidden_dim)
        print(out_dim) """
        if out_dim is None:
            out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        #print(x.shape)
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, out_dim=None, dropout=0., se=0):
        super().__init__()
        # dim_head是每个头的特征维度
        # 多个头的特征是放在一起计算的
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.out_dim = out_dim
        self.dim = dim

        self.attend = nn.Softmax(dim=-1)
        # 这个就是产生QKV三组向量因此要乘以3
        #print(dim,inner_dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if out_dim is None:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, out_dim),
            ) if project_out else nn.Identity()

        self.to_out_dropout = nn.Dropout(dropout)

        self.se = se
        if self.se > 0:
            self.se_layer = SE(dim)

    def forward(self, x):
        # b是batch size h 是注意力头的数目 n 是图像块的数目
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)

        if self.se:
            out = self.se_layer(out)

        out = self.to_out_dropout(out)

        if self.out_dim is not None and self.out_dim != self.dim:
            # 这个时候需要特殊处理，提前做一个残差
            """ print(1)
            print(v.squeeze(1).shape)
            print(out.shape) """
            out = out + v.squeeze(1)

        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_out_dim=None, ff_out_dim=None, dropout=0., se=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attn_out_dim = attn_out_dim
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                                       dim_head=dim_head, out_dim=attn_out_dim, dropout=dropout, se=se)),
                PreNorm(dim if not attn_out_dim else attn_out_dim,
                        FeedForward(dim if not attn_out_dim else attn_out_dim, mlp_dim, out_dim=ff_out_dim,
                                    dropout=dropout))
            ]))

    def forward(self, x):
        # print("syc--x.shape(input) = {}\t",x.shape) #syc自己加的
        for attn, ff in self.layers:
            # 都是残差学习
            if self.attn_out_dim is not None and self.dim != self.attn_out_dim:
                # print(x.shape)
                x = attn(x)
                # print(x.shape)
            else:
                x = attn(x) + x
            #print(x.shape)
            x = ff(x) + x
        # print("syc--x.shape(output) = {}",x.shape)
        return x


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        # print(x.shape)
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x

class T2T(nn.Module):
    def __init__(self, *,
                 image_size,
                 dim,
                 channels=1,
                 dropout=0.,
                 emb_dropout=0.,
                 time_emb="One", #"One"代表每个时间点仅有一个时间变量， "All"代表每个时间点每个维度都有一个时间变量， "No"代表没有时间嵌入
                 t2t_layers=((7, 4, 2), (3, 2,1), (3, 1,1)),
                 data_type = "ESCC"
                 ):
        super().__init__()

        layers = []
        if data_type == "ESCC":
            layer_dim = [channels * 7 * 7 * 7, 64 * 3 * 3 * 3]
        output_image_size = image_size
        # print(output_image_size)
        # print(layer_dim)
        for i, (kernel_size, stride,padding) in enumerate(t2t_layers):
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(
                output_image_size, kernel_size, stride, padding)
            layers.extend([
                RearrangeImage1(data_type) if not is_first else nn.Identity(),
                unfd(kernel_size, stride, padding),
                Rearrange('b c n -> b n c'),
                Transformer(dim=layer_dim[i], heads=1, depth=1, dim_head=64,
                            mlp_dim=64, attn_out_dim=64, ff_out_dim=64,
                            dropout=dropout) if not is_last else nn.Identity(),
            ])
        # print(layer_dim[1])
        layers.append(nn.Linear(layer_dim[1], dim))
        layers.append(SE(dim,16))

        layers_before = copy.deepcopy(layers)            
        self.to_patch_embedding_before = nn.Sequential(*layers_before)  # 不共用
           
####T2T结束
        # print(output_image_size)
        if data_type == "ESCC":
            self.pos_embedding_before = nn.Parameter(torch.randn(1, output_image_size * 9  , dim))
            self.scale = nn.Sequential(nn.Linear(2 * output_image_size * 9  , 2))
            # print(self.pos_embedding_before.shape)

        self.time_emb = time_emb
        if self.time_emb == "One":
            self.time_embedding = nn.Parameter(torch.randn(1))
        elif self.time_emb == "All":
            self.time_embedding = nn.Parameter(torch.randn(dim))

        # self.scale = nn.Sequential(nn.Linear(2 * output_image_size * 9  , 2))
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, before_x):
        before_x = self.to_patch_embedding_before(before_x)
        # print(before_x.shape)
###T2T end
        before_x += self.pos_embedding_before
        
        if self.time_emb:
            before_x += self.time_embedding
                
        before_x = self.dropout(before_x)
        return before_x

if __name__ == '__main__':
    from thop import profile
    net = T2T(image_size=32,
                dim=128,
                channels=1,
                dropout=0.,
                emb_dropout=0.,
                time_emb="One",
                t2t_layers=((7, 4,2), (3, 2,1), (3, 2,1)),
                data_type = "ESCC"
                )
    input = torch.randn(8, 1, 32, 48, 48) #ESCC
    #out = net(input,input)
    #print(out)
    flops, params = profile(net, (input,))
    print('flops: ', flops, 'params: ', params)