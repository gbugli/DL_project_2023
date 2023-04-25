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
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
from models import trunc_normal_

# Totally naive decoder implementation for testing, takes in frames x embed_dim and outputs frames x h x w x num_classes
# When running a batch, we should rearrange to (b f) x embed_dim before passing in
# Input is of shape b t (h w) m, 
class BaselineFCDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, H, W, num_classes, num_hidden_layers=2,dropout=0.1):
        super(BaselineFCDecoder, self).__init__()

        self.layers = nn.Sequential()
        self.layers.add_module("fc_in", nn.Linear(embed_dim, hidden_dim))

        for i in range(num_hidden_layers):
            self.layers.add_module(f"hidden_{i+1}", nn.Linear(hidden_dim, hidden_dim))
            self.layers.add_module(f"relu_{i+1}", nn.ReLU())
            self.layers.add_module(f"dropout_{i+1}", nn.Dropout(p=dropout))

        self.layers.add_module("fc_out", nn.Linear(hidden_dim,  H * W * num_classes))

        self.reshape_output = nn.Sequential(
            nn.Unflatten(2, (num_classes, H, W)) # Wrong? the tensor is (b n H * W * num_classes) where n is the number of patches
        )

    def forward(self, x):
        self.layers(x)
        return self.reshape_output(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_hidden_layers=2):
        super(Decoder, self).__init__()

        self.decoder = BaselineFCDecoder(input_dim, hidden_dim, num_classes, num_hidden_layers) # What are H and W?

    def forward(self, x):
        x = rearrange(x, 'b f e -> (b t) c h w') # the input is of shape (b t n m), also what is c? in order for the function to work some values need to be specified
        return self.decoder(x)
    

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        # attns = []
        # for mod in self.layers:
        #     output, attn = mod(output, memory, tgt_mask=tgt_mask,
        #                  memory_mask=memory_mask,
        #                  tgt_key_padding_mask=tgt_key_padding_mask,
        #                  memory_key_padding_mask=memory_key_padding_mask)
            # attns.append(attn)

        output, attn = self.layers[0](output, memory, tgt_mask=tgt_mask,
                          memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

def accuracy(pred, target, topk=1, thresh=None, ignore_index=None):
    """Calculate accuracy according to the prediction and target.
    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        ignore_index (int | None): The label index to be ignored. Default: None
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.
    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    if ignore_index is not None:
        correct = correct[:, target != ignore_index]
    res = []
    eps = torch.finfo(torch.float32).eps
    for k in topk:
        # Avoid causing ZeroDivisionError when all pixels
        # of an image are ignored
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        if ignore_index is not None:
            total_num = target[target != ignore_index].numel() + eps
        else:
            total_num = target.numel() + eps
        res.append(correct_k.mul_(100.0 / total_num))
    return res[0] if return_single else res

class ATMHead(nn.Module):
    def __init__(
            self,
            img_size,
            H,
            W,
            in_channels,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=3,
            use_proj=True,
            CE_loss=False,
            crop_train=False,
            #shrink_ratio=None,
            #**kwargs,

            num_classes=48
    ):
        
        super().__init__()

        # super(ATMHead, self).__init__()
         #   in_channels=in_channels, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.image_size = img_size
        self.H = H
        self.W = W

        self.use_stages = use_stages
        self.crop_train = crop_train
        nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []
        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
            decoder = TPN_Decoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            atm_decoders.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders
        self.q = nn.Embedding(self.num_classes + 1, dim) # nn.Embedding(self.num_classes, dim)

        self.class_embed = nn.Linear(dim, self.num_classes + 1)
        self.CE_loss = CE_loss
        # delattr(self, 'conv_seg')

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, inputs):
        x = []
        # for stage_ in inputs[:self.use_stages]:
        #     x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)

        x.append(self.d4_to_d3(inputs) if inputs.dim() > 3 else inputs)

        x.reverse()
        bs = x[0].size()[0]

        laterals = []
        attns = []
        maps_size = []
        qs = []
        q = self.q.weight.repeat(bs, 1, 1).transpose(0, 1)

        for idx, (x_, proj_, norm_, decoder_) in enumerate(zip(x, self.input_proj, self.proj_norm, self.decoder)):
            lateral = norm_(proj_(x_))
            # if idx == 0:
            if True:
                laterals.append(lateral)
            else:
                if laterals[idx - 1].size()[1] == lateral.size()[1]:
                    laterals.append(lateral + laterals[idx - 1])
                else:
                    # nearest interpolate
                    l_ = self.d3_to_d4(laterals[idx - 1])
                    l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
                    l_ = self.d4_to_d3(l_)
                    laterals.append(l_ + lateral)

            
            q, attn = decoder_(q, lateral.transpose(0, 1))
            attn = attn.transpose(-1, -2)
            if self.crop_train and self.training:
                blank_attn = torch.zeros_like(attn)
                blank_attn = blank_attn[:, 0].unsqueeze(1).repeat(1, (self.image_size//16)**2, 1)
                blank_attn[:, inputs[-1]] = attn
                attn = blank_attn
                self.crop_idx = inputs[-1]
            attn = self.d3_to_d4(attn)
            maps_size.append(attn.size()[-2:])
            qs.append(q.transpose(0, 1))
            attns.append(attn)
        qs = torch.stack(qs, dim=0)
        outputs_class = self.class_embed(qs)
        out = {"pred_logits": outputs_class[-1]}

        outputs_seg_masks = []
        size = maps_size[-1]

        for i_attn, attn in enumerate(attns):
            if i_attn == 0:
                outputs_seg_masks.append(F.interpolate(attn, size=size, mode='bilinear', align_corners=False))
            else:
                outputs_seg_masks.append(outputs_seg_masks[i_attn - 1] +
                                         F.interpolate(attn, size=size, mode='bilinear', align_corners=False))

        out["pred_masks"] = F.interpolate(outputs_seg_masks[-1],
                                          #size=(self.image_size, self.image_size),
                                          size=(self.H, self.W),
                                          mode='bilinear', align_corners=False)

        out["pred"] = self.semantic_inference(out["pred_logits"], out["pred_masks"])

        if self.training:
            # [l, bs, queries, embed]
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_seg_masks
            )
        else:
            return out["pred"]

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    # @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        if self.CE_loss:
            return super().losses(seg_logit["pred"], seg_label)

        if isinstance(seg_logit, dict):
            # atm loss
            seg_label = seg_label.squeeze(1)
            if self.crop_train:
                # mask seg_label by crop_idx
                bs, h, w = seg_label.size()
                mask_label = seg_label.reshape(bs, h//16, 16, w//16, 16)\
                    .permute(0, 1, 3, 2, 4).reshape(bs, h*w//256, 256)
                empty_label = torch.zeros_like(mask_label) + self.ignore_index
                empty_label[:, self.crop_idx] = mask_label[:, self.crop_idx]
                seg_label = empty_label.reshape(bs, h//16, w//16, 16, 16)\
                    .permute(0, 1, 3, 2, 4).reshape(bs, h, w)
            loss = self.loss_decode(
                seg_logit,
                seg_label,
                ignore_index=self.ignore_index)

            loss['acc_seg'] = accuracy(seg_logit["pred"], seg_label, ignore_index=self.ignore_index)
            return loss