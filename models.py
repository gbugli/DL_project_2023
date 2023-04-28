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
from video_dataset import VideoFrameDataset, ImglistToTensor
from x_transformers import Encoder, Decoder


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
    
class Predictor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        
        self.predictor = Decoder(dim = embed_dim, depth = depth, heads = num_heads)
    def forward(self, context_encoding, target_masks):
        x = torch.cat((context_encoding, target_masks), dim = 1)
        x = self.predictor(x)
        return x
        #return last len(target_masks) tokens
        # l = x.shape[1]
        # return x[:, l - target_masks.shape[1]:, :]
    

class IJEPA_base(nn.Module):
    def __init__(self,
                img_size=224,
                patch_size=16,
                in_chans=3,
                attention_type='joint_space_time',
                norm_layer=nn.LayerNorm,
                num_frames=22,
                dropout=0.,
                mode="train",
                M=11,
                embed_dim=768,
                device='cuda',
                # encoder parameters
                enc_depth=12,
                enc_num_heads=12,
                # predictor parameters
                pred_depth=12,
                pred_num_heads=12):
        super().__init__()
        self.mode = mode
        self.dropout = dropout
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        self.M = M # number of masked frames

        self.norm_layer = norm_layer
        self.attention_type = attention_type

        self.post_emb_norm = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        self.teacher_encoder = Encoder(
            dim=embed_dim,
            heads=enc_num_heads,
            depth=enc_depth, 
            layer_dropout=self.dropout,
        )  

        self.student_encoder = copy.deepcopy(self.teacher_encoder)
        self.predictor = Predictor(embed_dim, pred_num_heads, pred_depth, layer_dropout=self.dropout)
        

    @torch.no_grad() 
    ### get the target block
    def get_target_block(self, target_encoder, x, B, T, W):  
        #get the target block
        target_encoder = target_encoder.eval()
        x = target_encoder(x) # input in format 'b (t h w) m',output in format 'b t (h w) m' (batch frames n_patches embed_dim)
        x = self.norm(x)

        #select last M frames to mask in x
        mask_indices = torch.arange(22-self.M, 22)
        
        #mask the selected frames in the context block
        target_block = x[:,mask_indices] #get last M frames
        #all_patches = x
        return target_block, mask_indices

    ### get the context block
    def get_context_block(self, x, mask_indices):
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

        if self.attention_type != 'space_only':
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            
        return x, B, T, W
    
    def forward(self, x):
        x, B, T, W = self.get_patch_embeddings(x)

        x = self.post_emb_norm(x)

        #if mode is test, we get return full embedding:
        if self.mode == 'test':
            encoding = self.student_encoder(x) # input in format 'b (t h w) m',output in format 'b t (h w) m' (batch frames n_patches embed_dim)
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
            return self.predictor(encoding) # predict the masked frames
        
        # #get target embeddings
        # input in format 'b (t h w) m', output in format (1) 'b 11 (h w) m' and (2) 'b t (h w) m'
        target_blocks, mask_indices = self.get_target_block(self.teacher_encoder,x,B,T,W)

        #get context embeddings
        context_block = self.get_context_block(x, B, T, W, mask_indices)

        context_encoding = self.student_encoder(context_block)
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
        prediction_blocks = self.predictor(prediction_cat)

        prediction_blocks = prediction_blocks[:,-self.M:]
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