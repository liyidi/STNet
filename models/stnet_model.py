from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
from tools import ops
import cv2
import math
__all__ = ['AlexNetV1']

class STNet(nn.Module):
    def __init__(self, net_path_vi= None, net_path_au= None):
        super(STNet, self).__init__()
        self.net1 = Net(
            backbone=AlexNetV1(),
            head=SiamFC())
        ops.init_weights(self.net1)
        if net_path_vi is not None:
            self.net1.load_state_dict(torch.load(
                    net_path_vi, map_location=lambda storage, loc: storage))#模型是CPU，预加载的训练参数是GPU
        self.net2 = audioNet(
            net_head = AlexNetV1_au(),
            predNet = GCFpredictor())
        ops.init_weights(self.net2)
        if net_path_au is not None:
            self.net2.load_state_dict(torch.load(
                    net_path_au, map_location=lambda storage, loc: storage)['model'])
        self.netMHA = MultiHeadAttention()
        ops.init_weights(self.netMHA)
        self.predNet = Predictor()
        ops.init_weights(self.predNet)
        self.evlNet = evlNet()
        ops.init_weights(self.evlNet)
        self.PE_vi = PositionEmbeddingSine()
        self.PE_au = PositionEmbeddingSine()

        self.LN1_vi = nn.LayerNorm([256, 35, 35], eps=1e-6)
        self.LN1_au = nn.LayerNorm([256, 35, 35], eps=1e-6)
        self.LN_vi = nn.LayerNorm([1225, 256], eps=1e-6)
        self.LN_au = nn.LayerNorm([1225, 256], eps=1e-6)
        self.LN_av = nn.LayerNorm([1225, 256], eps=1e-6)
        self.LN2 = nn.LayerNorm([1225,256], eps=1e-6)

    def forward(self, ref, img0,img1,img2, auFr):
        Fvi1 = self.net1(ref, img0,img1,img2) # [b,c,h,w]
        Fau1 = self.net2(auFr).permute(0, 3, 1, 2)# [b,c,h,w]

        b, c = Fvi1.shape[0], Fvi1.shape[1]
        Fvi1 = self.LN1_vi(Fvi1)
        Fau1 = self.LN1_au(Fau1)
        Fvi_pe = Fvi1 + self.PE_vi(Fvi1)
        Fau_pe = Fau1 + self.PE_au(Fau1)

        Fvi_pe = Fvi_pe.permute(0, 2, 3, 1).view(b, -1, c)  # [b,N=h*w,c]
        Fau_pe = Fau_pe.permute(0, 2, 3, 1).view(b, -1, c)  # [b,N=h*w,c]
        Fvi2 = self.LN_vi(Fvi_pe)
        Fau2 = self.LN_au(Fau_pe)

        out_vi = self.netMHA(q=Fau2, k=Fvi2, v=Fvi2)
        out_au = self.netMHA(q=Fvi2, k=Fau2, v=Fau2)
        out_av = out_vi + out_au
        out_av = self.LN_av(out_av)

        out_pred = self.predNet(out_av)
        out_pred = torch.squeeze(out_pred)
        out_av2 = Fvi2 + Fau2 + out_av
        out_av2 = self.LN2(out_av2)

        out_evl = self.evlNet(out_av2)

        return out_pred, out_evl

class audioNet(nn.Module):
    def __init__(self,net_head, predNet):
        super(audioNet, self).__init__()
        self.net_head = net_head
        self.predNet = predNet

    def forward(self, x):
        x = self.net_head(x).permute(0, 2, 3, 1)  #[b,h,w,c]
        x = self.predNet(x)
        return x

class GCFpredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU())

    def forward(self, x):
        x = self.fc1(x)
        return x

class Net(nn.Module):
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def updownsample(self, x):
        return F.interpolate(x,size=(35,35),mode='bilinear',align_corners=False)

    def forward(self, z, x0, x1, x2):
        z = self.backbone(z)

        fx0 = self.backbone(x0)
        fx1 = self.backbone(x1)
        fx2 = self.backbone(x2)

        h0 = self.head(z, fx0)
        h1 = self.head(z, fx1)
        h2 = self.head(z, fx2)

        n0 = self.updownsample(h0)
        n1 = self.updownsample(h1)
        n2 = self.updownsample(h2)

        return n0+n1+n2

class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        z0 = z[0]
        x0 = x[0]
        out = F.conv2d(x0.unsqueeze(0), z0.unsqueeze(1), groups=c)

        for i in range(1, nz):
            zi = z[i]
            xi = x[i]
            outi = F.conv2d(xi.unsqueeze(0), zi.unsqueeze(1), groups=c)
            out = torch.cat([out, outi], dim=0)

        return out


class _AlexNet(nn.Module):

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class _AlexNet_au(nn.Module):

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class AlexNetV1(_AlexNet):
    output_stride = 8
    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


class AlexNetV1_au(_AlexNet_au):
    output_stride = 8
    def __init__(self):
        super(AlexNetV1_au, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 6, 1, groups=2))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head=8, d_model=256, d_k=32, d_v=32, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = k

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=(1225, 1), stride=1)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU())
        self.fc3 = nn.Linear(256, 2, bias=False)

    def forward(self, x):
        x = self.maxPool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LN(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm([1225,256], eps=1e-6)

    def forward(self, x):
        x = self.layer_norm(x)
        return x

class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):# [b,c,h,w]
        b,h,w = x.shape[0],x.shape[2],x.shape[3]
        mask = torch.ones((b, h, w), dtype=torch.bool).to(x.device)
        assert mask is not None
        not_mask = mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class evlNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=(1225, 1), stride=1)
        self.avgPool = nn.AvgPool2d(kernel_size=(1225, 1), stride=1)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.ReLU())
        self.fc3 = nn.Linear(128, 1, bias=False)

    def forward(self, x):
        x1 = self.maxPool(x)
        x2 = self.avgPool(x)
        x = x1+x2
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

class GCFpredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU())

    def forward(self, x):
        x = self.fc1(x)
        return x