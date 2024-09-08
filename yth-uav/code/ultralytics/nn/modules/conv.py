# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
#down temp
from torch import einsum
from einops.layers.torch import Rearrange
from einops import rearrange

__all__ = ('Conv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv', 'CARAFE', 'CWM', 'SWM',
           'SELayer','SpatialGate','Mul','Sub','GAU','BiFPN_Concat3','BiFPN_Concat2')
#
# class CARAFE(nn.Module):
#     # CARAFE: Content-Aware ReAssembly of FEatures       https://arxiv.org/pdf/1905.02188.pdf
#     def __init__(self, c1, c2, kernel_size=3, up_factor=2):
#         super(CARAFE, self).__init__()
#         self.kernel_size = kernel_size
#         self.up_factor = up_factor
#         self.down = nn.Conv2d(c1, c1 // 4, 1)
#         self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
#                                  self.kernel_size, 1, self.kernel_size // 2)
#         self.out = nn.Conv2d(c1, c2, 1)
#
#     def forward(self, x):
#         N, C, H, W = x.size()
#         # N,C,H,W -> N,C,delta*H,delta*W
#         # kernel prediction module
#         kernel_tensor = self.down(x)  # (N, Cm, H, W)
#         kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
#         kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
#         kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
#         kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
#         kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
#         kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W,
#                                               self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
#         kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)
#
#         # content-aware reassembly module
#         # tensor.unfold: dim, size, step
#         x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
#                           self.kernel_size // 2, self.kernel_size // 2),
#                   mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
#         x = x.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
#         x = x.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
#         x = x.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
#         x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)
#
#         out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
#         out_tensor = out_tensor.reshape(N, H, W, -1)
#         out_tensor = out_tensor.permute(0, 3, 1, 2)
#         out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
#         out_tensor = self.out(out_tensor)
#         # print("up shape:",out_tensor.shape)
#         return out_tensor


class SpatialGate(nn.Module):
    def __init__(self,
                 gate_channel,
                 reduction_ratio=8,
                 dilation_conv_num=2,
                 dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()

        self.gate_s.add_module(
            'gate_s_conv_reduce0',
            nn.Conv2d(gate_channel,
                      gate_channel // reduction_ratio,
                      kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0',
                               nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())

        # 进行多个空洞卷积，丰富感受野
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                'gate_s_conv_di_%d' % i,
                nn.Conv2d(gate_channel // reduction_ratio,
                          gate_channel // reduction_ratio,
                          kernel_size=3,
                          padding=dilation_val,
                          dilation=dilation_val))
            self.gate_s.add_module(
                'gate_s_bn_di_%d' % i,
                nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())

        self.gate_s.add_module(
            'gate_s_conv_final',
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, x):
        return self.gate_s(x).expand_as(x)


class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)


class Sub(nn.Module):
    def __init__(self, c1):
        super(Sub, self).__init__()

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        out = x1 - x2
        return out

class Mul(nn.Module):
    def __init__(self, c1):
        super(Mul, self).__init__()

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        out = x1 * x2
        return out



class GAU(nn.Module):
    def __init__(self, c1):
        super(GAU, self).__init__()
        self.conv_mask = nn.Conv2d(c1, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU(True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        low_feature = x[0]
        high_feature = x[1]
        # 获取深层特征图的全局信息
        batch, channel, height, width = high_feature.size()
        input_x = high_feature
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(high_feature)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        # 为浅层特征图提供指导
        #out = self.relu(context * low_feature) + high_feature
        out = self.relu(context * low_feature)
        #out = torch.cat((out1,high_feature),1)
        #out = self.max_pool(out)
        return out


params = {
    "global_context": {
        "weighted_gc": True,
        "gc_reduction": 1,
        "compete": True,
        "head": 1,
    },
    "spatial_mixer": {
        "use_globalcontext": True,
        "useSecondTokenMix": True,
        "mix_size_1": 5,
        "mix_size_2": 7,
        "fc_factor": 8,
        "fc_min_value": 16,
        "useSpatialAtt": True
    },
}


class GlobalContext(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU, params=params):
        super().__init__()
        # bottleneck information
        # "Compete to compute." NeurIPS 2013
        self.compete = params["global_context"]["compete"]
        if self.compete:
            self.fc1 = nn.Linear(dim, 2 * dim // params["global_context"]["gc_reduction"])
            self.fc2 = nn.Linear(dim // params["global_context"]["gc_reduction"], dim)
        else:
            self.fc = nn.Sequential(
                nn.Linear(dim, dim // params["global_context"]["gc_reduction"]),
                act_layer(),
                nn.Linear(dim // params["global_context"]["gc_reduction"], int(dim * 0.5))
            )
        self.weight_gc = params["global_context"]["weighted_gc"]
        if self.weight_gc:
            self.head = params["global_context"]["head"]
            self.scale = (dim // self.head) ** -0.5
            self.rescale_weight = nn.Parameter(torch.ones(self.head))
            self.rescale_bias = nn.Parameter(torch.zeros(self.head))
            self.epsilon = 1e-5

    def _get_gc(self, gap):  # gap [b,c]
        if self.compete:
            b, c = gap.size()
            gc = self.fc1(gap).reshape([b, 2, -1])
            gc, _ = gc.max(dim=1)
            gc = self.fc2(gc)
            return gc
        else:
            return self.fc(gap)

    def forward(self, x):
        if self.weight_gc:
            b, c, w, h = x.size()
            x = rearrange(x, "b c x y -> b c (x y)")
            gap = x.mean(dim=-1, keepdim=True)
            q, g = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h=self.head), [x, gap])  # [b,head, hdim, n]
            sim = einsum('bhdi,bhjd->bhij', q, g.transpose(-1, -2)).squeeze(dim=-1) * self.scale  # [b,head, w*h]
            std, mean = torch.std_mean(sim, dim=[1, 2], keepdim=True)
            sim = (sim - mean) / (std + self.epsilon)
            sim = sim * self.rescale_weight.unsqueeze(dim=0).unsqueeze(dim=-1) + self.rescale_bias.unsqueeze(
                dim=0).unsqueeze(dim=-1)
            sim = sim.reshape(b, self.head, 1, w, h)  # [b, head, 1, w, h]
            gc = self._get_gc(gap.squeeze(dim=-1)).reshape(b, self.head, -1).unsqueeze(dim=-1).unsqueeze(
                dim=-1)  # [b, head, hdim, 1, 1]
            gc = rearrange(sim * gc, "b h d x y -> b (h d) x y")  # [b, head, hdim, w, h] - > [b,c,w,h]
        else:
            gc = self._get_gc(x.mean(dim=-1).mean(dim=-1)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return gc  # [b,c,w,h] for weighted or [b,c,1,1]


class DeepWise_PointWise_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeepWise_PointWise_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch,
            dilation = 1,
            bias = False
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            dilation = 1,
            bias = False
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class SELayer(nn.Module):
    def __init__(self, c1, r=16, act_layer=nn.GELU, params=params):
        super(SELayer, self).__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.context = GlobalContext(dim=c1, act_layer=act_layer, params=params)
        self.conv3 = nn.Conv2d(c1 , c1 , 1, 1, bias=False)
        self.dw1 = DeepWise_PointWise_Conv(c1, c1)
        self.conv1 = nn.Conv2d(c1, c1 // r, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c1 // r)
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c1 // r, c1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.pconv = Partial_conv3(c1)
        # self.dconv = DWConv(c1, c1, 1)
        # self.dpconv = DeepWise_PointWise_Conv(c1, c1)
        # self.conv3 = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.l2 = nn.Linear(c1 // r, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print(x.size())
        # b, c, _, _ = x.size()
        # y = self.avgpool(x)
        v = self.context(x)
        y = x + self.conv3(v)
        y = self.dw1(y)
        y = self.conv1(y)
        y = self.bn(y)
        # y = self.l1(y)
        y = self.relu(y)
        y = self.conv2(y)
        wei = self.sig(y)
        z = x * wei
        # print(z.size())
        # z = self.pconv(z)
        # z = self.dconv(z)
        # z = self.dpconv(z)
        # z = self.conv3(z)
        # y = self.l2(y)
        # y = self.sig(y)
        # y = y.view(b, c, 1, 1)
        return z



#上面temp
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


# [[C,X],1,quan,[yuan 通道数]]
class CWM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x, y1=0):
        x, y1 = x
        # print(x.shape, 'and11', y1.shape)
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        y = torch.mul(y1, atten)
        y = y + y1
        return self.conv(y)


class SWM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        kernel_size = 7
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x, y=0):
        x, y = x
        # print(x.shape, 'and', y.shape)
        atten = self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
        y1 = torch.mul(y, atten)
        y1 = y1 + y
        return self.conv(y1)


class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X
