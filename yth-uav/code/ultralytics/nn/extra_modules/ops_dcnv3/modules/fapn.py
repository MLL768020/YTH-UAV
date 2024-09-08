# __package__ = '/home/liyihang/lyhredetr/ultralytics/nn/extra_modules/ops_dcnv3/modules/fapn.py'

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
from ultralytics.nn.extra_modules.ops_dcnv3.modules.dcnv3 import DCNv3


class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2, g * 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)


class FAM1(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c1, c2)
        self.offset = nn.Conv2d(c2 * 2, c2, 1, bias=False)
        self.conv = nn.Conv2d(c2 // 2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv4(c2, c2, 3, 1, 1, 8)

    def forward(self, feat_l, feat_s=0):
        feat_l, feat_s = feat_l
        # feat_arm = self.lateral_conv(feat_l)
        # feat_s = self.conv(feat_s)
        offset = self.offset(torch.cat([feat_arm, feat_s * 2], dim=1))
        feat_align = F.relu(self.dcpack_l2(feat_s, offset))
        return feat_align + feat_arm

#
# class CWM(nn.Module):
#     def __init__(self, c1, c2):
#         super().__init__()
#         self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
#         self.conv = nn.Conv2d(c1, c2, 1, bias=False)
#
#     def forward(self, x,y1):
#         atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
#         y = torch.mul(y1, atten)
#         y = y + y1
#         return self.conv(y)
#
#
# class SWM(nn.Module):
#     def __init__(self, c1, c2):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
#         self.conv = nn.Conv2d(c1, c2, 1, bias=False)
#
#     def forward(self, x, y):
#         atten = self.fc(x).sigmoid()
#         y1 = torch.mul(y, atten)
#         y1 = y1 + y
#         return self.conv(y1)
#
#
# if __name__ == '__main__':
#     # 实例化一个 SWM 对象
#     c1 = 3  # 输入通道数
#     c2 = 64  # 输出通道数
#     cwm = CWM(c2, c2)
#
#     # 准备输入数据
#     batch_size = 1
#     height, width = 32, 32
#     x = torch.randn(batch_size, c2, height, width)
#     y = torch.randn(batch_size, c2, height, width)
#
#     # 将输入数据传递给 SWM 模块
#     output = cwm(x, y)
#
#     # 输出结果
#     print("输出d 张量大小:", output.shape)
#
#
# class ChannelAttention(nn.Module):
#     """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""
#
#     def __init__(self, channels: int) -> None:
#         """Initializes the class and sets the basic configurations and instance variables required."""
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
#         return x * self.act(self.fc(self.pool(x)))
#
#
# class SpatialAttention(nn.Module):
#     """Spatial-attention module."""
#
#     def __init__(self, kernel_size=7):
#         """Initialize Spatial-attention module with kernel size argument."""
#         super().__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         """Apply channel and spatial attention on input for feature recalibration."""
#         return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))