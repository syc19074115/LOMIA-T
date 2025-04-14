
"""
Modifed from Timm. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block
import copy


class TRRN(nn.Module):  # Tumor Region Representation Network
    def __init__(self, t2t_image_size, t2t_dim, channels, time_emb, TRRN_depth, TRRN_dropKey, use_DropKey, data_type):
        super().__init__()
        # from .T2T import T2T
        import sys
        sys.path.append("") #添加自己指定的搜索路径
        from main_model.T2T import T2T
        self.t2t = T2T(image_size = t2t_image_size, #64
                        dim = t2t_dim,  #128
                        channels = channels,
                        dropout = 0.,
                        emb_dropout = 0.,
                        time_emb = time_emb,
                        t2t_layers = ((7, 4,2), (3, 2,1), (3, 2,1)),
                        data_type = data_type,
                        )
        dpr = [x.item() for x in torch.linspace(0, TRRN_dropKey, TRRN_depth)]  # stochastic depth decay rule
        layers_encoder = []
        for idx , block_cfg in enumerate(dpr):
            layers_encoder.append(
                                Block(dim=t2t_dim, num_heads=8, mlp_ratio=2, qkv_bias=None, 
                                proj_drop=0., attn_drop=0., drop_path=block_cfg, norm_layer=nn.LayerNorm,
                                use_DropKey=use_DropKey,mask_ratio=TRRN_dropKey,)
                                )
        self.TRRN_encoder = nn.Sequential(*copy.deepcopy(layers_encoder))

    def forward(self, x):
        x = self.t2t(x)
        # print(x.shpe)
        # print(x.shape) #不加入SE是torch.Size([8, 196, 576])
        x = self.TRRN_encoder(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Cross_Fusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.pre_fusion = CrossAttentionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=nn.LayerNorm, has_mlp=False)
        self.post_fusion = CrossAttentionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=nn.LayerNorm, has_mlp=False)

    def forward(self, t_pre, t_post):
        tmp_pre = torch.cat((t_pre[:,0:1],t_post[:,1:,...]),dim=1)
        tmp_post = torch.cat((t_post[:,0:1],t_pre[:,1:,...]),dim=1)
        cls_pre = self.pre_fusion(tmp_pre)
        cls_post = self.post_fusion(tmp_post)
        pre_cross = torch.cat((cls_pre[:,0:1], tmp_pre[:, 1:, ...]), dim=1)
        post_cross = torch.cat((cls_post[:,0:1], tmp_post[:, 1:, ...]), dim=1)
        return pre_cross, post_cross

class DFFM(nn.Module): #Deep Feature Fusion Module
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.Cross_Fusion = Cross_Fusion(embed_dim=dim)
        self.DFFM_encoder_pre = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=None, proj_drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
        self.DFFM_encoder_post = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=None, proj_drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)

    def forward(self, t_pre, t_post):
        pre_cross, post_cross = self.Cross_Fusion(t_pre, t_post)
        t_crosspre = self.DFFM_encoder_pre(pre_cross)
        # t_crosspre = self.DFFM_encoder_pre(t_crosspre) #
        t_crosspost = self.DFFM_encoder_post(post_cross)
        # t_crosspost = self.DFFM_encoder_post(t_crosspost) #
        return t_crosspre, t_crosspost

class LOMIA_T(nn.Module):
    """ 
    LOMIA-T including Tumor Region Representation Network & Deep Feature Fusion Module
    """
    def __init__(self, image_size, dim, channels, time_emb, TRRN_depth, TRRN_dropKey, use_DropKey, #t2t parameters
                mode,num_classes,
                data_type = "ESCC" ,#"OAI" or "ESCC"
                 ):
        super().__init__()
        self.TRRN_pre = TRRN(t2t_image_size=image_size, t2t_dim=dim, channels=channels, time_emb=time_emb, TRRN_depth=TRRN_depth, TRRN_dropKey=TRRN_dropKey, use_DropKey=use_DropKey, data_type=data_type)
        self.TRRN_post = TRRN(t2t_image_size=image_size, t2t_dim=dim, channels=channels, time_emb=time_emb, TRRN_depth=TRRN_depth, TRRN_dropKey=TRRN_dropKey, use_DropKey=use_DropKey, data_type=data_type)
        self.Cross_module = DFFM(dim=dim)
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for i in range(2)])
        self.head = nn.ModuleList([nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity() for i in range(2)])
        self.dropout = nn.Dropout(0.6)
        self.mode = mode
        self.softmax = nn.Softmax(dim=1)
        """ from utilis.loss import ContrastiveLoss_euc,FocalLoss
        self.tcl_m = TCL_m          #TCL
        self.fcl_alpha = FCL_alpha  #Focal alpha
        self.fcl_gamma = FCL_gamma  #Focal gamma
        self.TCL = ContrastiveLoss_euc(margin=TCL_m)
        FCL_alpha = torch.tensor([0.5, 0.5])
        self.FCL = FocalLoss(class_num=num_classes, alpha=FCL_alpha, gamma=FCL_gamma)
        self.softmax = nn.Softmax(dim=1) """

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, pre, post):
        t_pre = self.TRRN_pre(pre)
        t_post = self.TRRN_post(post)
        
        t_crosspre, t_crosspost = self.Cross_module(t_pre, t_post)

        if self.mode == 'pre_post':     
            xs = [t_crosspre, t_crosspost]
        elif self.mode == 'only_pre':   
            xs = [t_crosspre, t_crosspre]
        elif self.mode == 'only_post':  
            xs = [t_crosspost, t_crosspost]
        elif self.mode == 'crosspre_post':  
            xs = [t_crosspre, t_post]
        elif self.mode == 'pre_crosspost':  
            xs = [t_pre, t_crosspost]
        elif self.mode == 'crosspre_pre':  
            xs = [t_crosspre, t_pre]
        elif self.mode == 'crosspost_post': 
            xs = [t_crosspost, t_post]

        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        cls_tokens = [x[:, 0] for x in xs]
        
        # 分类
        cls_tokens = [self.dropout(x) for i, x in enumerate(cls_tokens)]
        logits = [self.head[i](x) for i, x in enumerate(cls_tokens)] #原先的写法
        logits = torch.mean(torch.stack(logits, dim=0), dim=0) #[batch,2]
        
        # prob = self.softmax(logits)

        return logits, t_pre, t_post, t_crosspre, t_crosspost
    

if __name__ == '__main__':
    from thop import profile
    net = LOMIA_T(image_size=32, dim=128, channels=1, time_emb="One", TRRN_depth=7, TRRN_dropKey=0., use_DropKey=False,mode="pre_post",num_classes=5, data_type="ESCC") #ESCC
    input = torch.randn(8, 1, 32, 48, 48)

    # net = LOMIA_T(image_size=224, dim=576, channels=1, time_emb="One", TRRN_depth=7, TRRN_dropKey=0., use_DropKey=False,mode="pre_post",num_classes=5, data_type="OAI") #OAI
    # input = torch.randn(8, 1, 224, 224)

    """ from torchvision import models
    model = models.vgg19(weights="VGG19_Weights.DEFAULT")
    weight = model.features[0].weight.data
    bias = model.features[0].bias.data

    # Compute the mean value of the weights along the channel dimension
    mean_weight = weight.mean(dim=1, keepdim=True)

    # Create new weights and biases for the first convolutional layer
    # The number of input channels is changed to 1
    new_weight = mean_weight.repeat(1, 1, 3, 3)  # Repeat the mean weight along the channel dimension
    new_bias = bias

    # Set the new weights and biases to the first convolutional layer
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    model.features[0].weight.data = new_weight
    model.features[0].bias.data = new_bias
    num_ftrs = model.classifier[6].in_features
    feature_model = list(model.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(num_ftrs, 5))
    model.classifier = nn.Sequential(*feature_model)
    net = model """

    flops, params = profile(net, (input,input, ))
    # flops, params = profile(net, (input, ))
    print('flops: ', flops, 'params: ', params)