import os
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import copy
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import torch.nn.functional as F
import warnings
from IJEPA.video_dataset import VideoFrameDataset, ImglistToTensor


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

 
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class EarlyStop:
    def __init__(self, patience, loss=False):
        self.patience = patience
        self.best_value = np.inf if loss else 0
        self.best_epoch = 0
        self.loss = loss

    def step(self, current_value, current_epoch):
        print("Current:{} Best:{}".format(current_value, self.best_value))
        if self.loss:
            if current_value < self.best_value:
                self.best_value = current_value
                self.best_epoch = current_epoch
        else:
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_epoch = current_epoch

    def stop_training(self, current_epoch) -> bool:
        return current_epoch - self.best_epoch > self.patience


class CustomDataParallel(nn.DataParallel):
    """
    Wrapper for scoring with nn.DataParallel object containing LTRModel.
    """

    def forward(self, x):
        return self.module.forward(x)  # type: ignore


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = img_size, img_size
        patch_size = patch_size, patch_size
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x # xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x + res_temporal # xt = x[:,1:,:] + res_temporal

            ## Spatial
            # init_cls_token = x[:,0,:].unsqueeze(1)
            # cls_token = init_cls_token.repeat(1, T, 1)
            # cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,w=W,t=T)
            # xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            # cls_token = res_spatial[:,0,:]
            # cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            # cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            # res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = x + res # x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    """
    def __init__(self,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=False,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm,
                attention_type='divided_space_time',
                dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_features = self.embed_dim = embed_dim

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, B, T, W):

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        # x = rearrange(x, 'b (h w t) m -> b t (h w) m',b=B,t=T,w=W)
        return x
    
    
class IJEPA_base(nn.Module):
    def __init__(self,
                img_size=224,
                patch_size=16,
                in_chans=3,
                norm_layer=nn.LayerNorm,
                num_frames=22,
                attention_type='joint_space_time',
                dropout=0.,
                mode="train",
                r=0.5,
                embed_dim=768,
                device='cuda',
                # encoder parameters
                enc_depth=12,
                enc_num_heads=12,
                enc_mlp_ratio=4.,
                enc_qkv_bias=False,
                enc_qk_scale=None,
                enc_drop_rate=0.,
                enc_attn_drop_rate=0.,
                enc_drop_path_rate=0.1,
                # predictor parameters
                pred_depth=12,
                pred_num_heads=12,
                pred_mlp_ratio=4.,
                pred_qkv_bias=False,
                pred_qk_scale=None,
                pred_drop_rate=0.,
                pred_attn_drop_rate=0.,
                pred_drop_path_rate=0.1,
                # positional and spacial embedding parameters
                pos_drop_rate=0.,
                time_drop_rate=0.):
        super().__init__()
        self.mode = mode
        self.dropout = dropout
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)
        self.r = r # number of masked frames

        self.norm_layer = norm_layer
        self.norm = norm_layer(embed_dim)

        self.attention_type = attention_type

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=time_drop_rate)

        self.teacher_encoder = VisionTransformer(
            embed_dim=embed_dim,
            num_heads=enc_num_heads,
            depth=enc_depth, 
            dropout=self.dropout,
            norm_layer=self.norm_layer,
            mlp_ratio=enc_mlp_ratio,
            attention_type=attention_type,
            qkv_bias=enc_qkv_bias,
            qk_scale=enc_qk_scale,
            drop_rate=enc_drop_rate,
            attn_drop_rate=enc_attn_drop_rate,
            drop_path_rate=enc_drop_path_rate
        )

        self.student_encoder = copy.deepcopy(self.teacher_encoder)
        self.predictor = VisionTransformer(
            embed_dim=embed_dim,
            num_heads=pred_num_heads,
            depth=pred_depth, 
            dropout=self.dropout,
            norm_layer=self.norm_layer,
            mlp_ratio=pred_mlp_ratio,
            attention_type=attention_type,
            qkv_bias=pred_qkv_bias,
            qk_scale=pred_qk_scale,
            drop_rate=pred_drop_rate,
            attn_drop_rate=pred_attn_drop_rate,
            drop_path_rate=pred_drop_path_rate
        )
        

    @torch.no_grad() 
    ### get the target block
    def get_target_block(self, target_encoder, x, B, T, W):  
        #get the target block
        target_encoder = target_encoder.eval()
        x = target_encoder(x, B, T, W) # input in format 'b (t h w) m',output in format 'b (t h w) m' (batch frames n_patches embed_dim)
        x = self.norm(x)
        tn = x.shape[1]

        #randomly select M frames to mask in x
        p = int(tn * self.r)
        mask_indices = (torch.randperm(tn))[:p]
        
        #mask the selected frames in the context block
        target_block = x[:,mask_indices] #get portion of random patches from rnadom frames
        #all_patches = x
        return target_block, mask_indices

    ### get the context block
    def get_context_block(self, x, mask_indices):
      #reshape x to format 'b t (h w) m'
      # x = rearrange(x, 'b (t h w) m -> b t (h w) m',b=B,t=T,w=W)
      #select all frames which are not masked
      index = torch.ones(x.shape[1], dtype=bool)
      index[mask_indices] = False
      context_block = x[:,index]
      # context_block = rearrange(context_block, 'b t (h w) m -> b (t h w) m',b=B,t=(T-self.M),w=W)
      return context_block
    
    def get_patch_embeddings(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.attention_type != 'space_only':
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            
        return x, B, T, W
    
    def forward(self, x):
        #get the patch embeddings
        x, B, T, W = self.get_patch_embeddings(x)

        #if mode is test, we get return full embedding:
        if self.mode == 'test':
            encoding = self.student_encoder(x, B, T, W) # input in format 'b (t h w) m',output in format 'b t (h w) m' (batch frames n_patches embed_dim)
            encoding = self.norm(encoding)
            b, n, m = encoding.shape
            # encoding = rearrange(encoding, 'b t (h w) m -> b (t h w) m',b=B,t=T,w=W)
            #add 11 mask tokens to the end of the embedding
            target_masks = self.mask_token.repeat(B, n, 1)
            target_pos_embedding = self.pos_embed.unsqueeze(1)
            # Add time embedding
            target_time_embed = self.time_embed.unsqueeze(2)[:,11:]

            target_embeddings = target_pos_embedding + target_time_embed
            target_embeddings = rearrange(target_embeddings, 'b t n m -> b (t n) m')
            target_masks = target_masks + target_embeddings
            
            # target_masks = rearrange(target_masks, 'b t (h w) m -> b (t h w) m',b=B,t=11,w=W)
            encoding = torch.cat((encoding, target_masks), dim=1)
            return self.predictor(encoding, B, T+11, W) # predict the masked frames
        
        # #get target embeddings
        # input in format 'b (t h w) m', output in format (1) 'b 11 (h w) m' and (2) 'b t (h w) m'
        target_blocks, mask_indices = self.get_target_block(self.teacher_encoder,x,B,T,W)

        #get context embeddings
        context_block = self.get_context_block(x, mask_indices)

        context_encoding = self.student_encoder(context_block, B, T, W)
        context_encoding = self.norm(context_encoding)
        # context_encoding = rearrange(context_encoding, 'b t (h w) m -> b (t h w) m',b=B,t=T-self.M,w=W)

        #n = h x w
        b, p, m = target_blocks.shape
        target_masks = self.mask_token.repeat(B, p, 1)

        # Add time embedding and position embedding
        target_pos_embedding = self.pos_embed.unsqueeze(1)
        target_time_embed = self.time_embed.unsqueeze(2)
        target_embeddings = target_pos_embedding + target_time_embed
        target_embeddings = rearrange(target_embeddings, 'b t n m -> b (t n) m')[:, mask_indices]


        target_masks = target_masks + target_embeddings
        # target_masks = rearrange(target_masks, 'b t (h w) m -> b (t h w) m',b=B,t=self.M,w=W)
        prediction_cat = torch.cat((context_encoding, target_masks), dim = 1)
        # make sure that the preds are actually at the end
        prediction_blocks = self.predictor(prediction_cat,B, T, W)

        prediction_blocks = prediction_blocks[:,-p:]
        return prediction_blocks, target_blocks, context_encoding





















def Projector(embed_dim, sizes, num_frames):
        sizes = [embed_dim] + sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.BatchNorm2d(num_frames))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(sizes[-1], embed_dim, bias=False))
        return nn.Sequential(*layers)

class IJEPA_Vic(nn.Module):
    def __init__(self,
                img_size=224,
                patch_size=16,
                in_chans=3,
                norm_layer=nn.LayerNorm,
                num_frames=22,
                attention_type='divided_space_time',
                dropout=0.,
                mode="train",
                M=4,
                embed_dim=768,
                device='cuda',
                sizes = [512, 1024, 512],
                # encoder parameters
                enc_depth=12,
                enc_num_heads=12,
                enc_mlp_ratio=4.,
                enc_qkv_bias=False,
                enc_qk_scale=None,
                enc_drop_rate=0.,
                enc_attn_drop_rate=0.,
                enc_drop_path_rate=0.1,
                # predictor parameters
                pred_depth=12,
                pred_num_heads=12,
                pred_mlp_ratio=4.,
                pred_qkv_bias=False,
                pred_qk_scale=None,
                pred_drop_rate=0.,
                pred_attn_drop_rate=0.,
                pred_drop_path_rate=0.1,
                # positional and spacial embedding parameters
                pos_drop_rate=0.,
                time_drop_rate=0.):
        super().__init__()
        self.mode = mode
        self.dropout = dropout
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)
        self.M = M # number of masked frames

        self.embed_dim = embed_dim
        self.sizes = sizes
        self.norm_layer = norm_layer
        self.norm = norm_layer(embed_dim)
        # self.norm_proj = norm_layer(sizes[-1])

        self.attention_type = attention_type

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=time_drop_rate)

        self.teacher_encoder = VisionTransformer(
            embed_dim=embed_dim,
            num_heads=enc_num_heads,
            depth=enc_depth, 
            dropout=self.dropout,
            norm_layer=self.norm_layer,
            mlp_ratio=enc_mlp_ratio,
            attention_type=attention_type,
            qkv_bias=enc_qkv_bias,
            qk_scale=enc_qk_scale,
            drop_rate=enc_drop_rate,
            attn_drop_rate=enc_attn_drop_rate,
            drop_path_rate=enc_drop_path_rate
        )

        self.student_encoder = copy.deepcopy(self.teacher_encoder)
        self.predictor = VisionTransformer(
            embed_dim=embed_dim,
            num_heads=pred_num_heads,
            depth=pred_depth, 
            dropout=self.dropout,
            norm_layer=self.norm_layer,
            mlp_ratio=pred_mlp_ratio,
            attention_type=attention_type,
            qkv_bias=pred_qkv_bias,
            qk_scale=pred_qk_scale,
            drop_rate=pred_drop_rate,
            attn_drop_rate=pred_attn_drop_rate,
            drop_path_rate=pred_drop_path_rate
        )

        self.student_projector = Projector(embed_dim, sizes, num_frames-M)
        self.teacher_projector = Projector(embed_dim, sizes, num_frames)
        

    @torch.no_grad() 
    ### get the target block
    def get_target_block(self, target_encoder, projector, x, B, T, W):  
        #get the target block
        target_encoder = target_encoder.eval()
        x = projector(target_encoder(x, B, T, W)) # input in format 'b (t h w) m',output in format 'b t (h w) m' (batch frames n_patches embed_dim)
        x = self.norm(x)

        #randomly select M frames to mask in x
        mask_indices = (torch.randperm(11)+11)[:self.M]
        
        #mask the selected frames in the context block
        target_block = x[:,mask_indices] #get 4 random frames from the last 11 frames
        #all_patches = x
        return target_block, mask_indices

    ### get the context block
    def get_context_block(self, x, B, T, W, mask_indices):
      #reshape x to format 'b t (h w) m'
      x = rearrange(x, 'b (t h w) m -> b t (h w) m',b=B,t=T,w=W)
      #select all frames which are not masked
      index = torch.ones(x.shape[1], dtype=bool)
      index[mask_indices] = False
      context_block = x[:,index]
      context_block = rearrange(context_block, 'b t (h w) m -> b (t h w) m',b=B,t=(T-self.M),w=W)
      return context_block
    
    def get_patch_embeddings(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.attention_type != 'space_only':
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            
        return x, B, T, W
    
    def forward(self, x):
        #get the patch embeddings
        x, B, T, W = self.get_patch_embeddings(x)

        #if mode is test, we get return full embedding:
        if self.mode == 'test':
            # Check if projector gives error, if so it's because it receives 11 frames and not 18? to solve change projector batchnorm 
            encoding = self.student_projector(self.student_encoder(x, B, T, W)) # input in format 'b (t h w) m',output in format 'b t (h w) m' (batch frames n_patches embed_dim)
            encoding = self.norm(encoding)
            n = encoding.shape[2]
            encoding = rearrange(encoding, 'b t (h w) m -> b (t h w) m',b=B,t=T,w=W)
            #add 11 mask tokens to the end of the embedding
            target_masks = self.mask_token.repeat(B, 11, n, 1)
            target_pos_embedding = self.pos_embed.unsqueeze(1)
            target_masks = target_masks + target_pos_embedding
            
            # Add time embedding
            target_time_embed = self.time_embed.unsqueeze(2)[:,11:]
            target_masks = target_masks + target_time_embed
            
            target_masks = rearrange(target_masks, 'b t (h w) m -> b (t h w) m',b=B,t=11,w=W)
            encoding = torch.cat((encoding, target_masks), dim=1)
            return self.predictor(encoding, B, T+11, W) # predict the masked frames
        
        # #get target embeddings
        # input in format 'b (t h w) m', output in format (1) 'b 11 (h w) m' and (2) 'b t (h w) m'
        target_blocks, mask_indices = self.get_target_block(self.teacher_encoder, self.teacher_projector,x,B,T,W)

        #get context embeddings
        context_block = self.get_context_block(x, B, T, W, mask_indices)

        context_encoding = self.student_projector(self.student_encoder(context_block, B, T-self.M, W))
        context_encoding = self.norm(context_encoding)
        context_encoding = rearrange(context_encoding, 'b t (h w) m -> b (t h w) m',b=B,t=T-self.M,w=W)

        #n = h x w
        n = target_blocks.shape[2]
        target_masks = self.mask_token.repeat(B, self.M, n, 1)
        target_pos_embedding = self.pos_embed.unsqueeze(1)
        target_masks = target_masks + target_pos_embedding
        
        # Add time embedding
        target_time_embed = self.time_embed.unsqueeze(2)[:,mask_indices]
        target_masks = target_masks + target_time_embed
        
        target_masks = rearrange(target_masks, 'b t (h w) m -> b (t h w) m',b=B,t=self.M,w=W)
        prediction_cat = torch.cat((context_encoding, target_masks), dim = 1)
        # make sure that the preds are actually at the end
        prediction_blocks = self.predictor(prediction_cat,B, T, W)

        prediction_blocks = prediction_blocks[:,-self.M:]
        return prediction_blocks, target_blocks