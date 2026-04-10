import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class SpatialAARMAdapter(nn.Module):
    """
    将AARM概念适配到空间特征图的处理
    """

    def __init__(self, c1, c2, *args):  # 接受额外参数以适配YOLOv5参数传递
        super(SpatialAARMAdapter, self).__init__()

        self.c1 = c1
        self.c2 = c2

        # 特征转换层
        self.query_conv = nn.Conv2d(c1, c2, kernel_size=1)
        self.key_conv = nn.Conv2d(c1, c2, kernel_size=1)
        self.value_conv = nn.Conv2d(c1, c2, kernel_size=1)

        # 输出转换层
        self.output_conv = nn.Conv2d(c2, c2, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        # 生成查询、键和值
        query = self.query_conv(x).view(batch_size, self.c2, -1)  # B x C x HW
        key = self.key_conv(x).view(batch_size, self.c2, -1).permute(0, 2, 1)  # B x HW x C
        value = self.value_conv(x).view(batch_size, self.c2, -1)  # B x C x HW

        # 计算注意力得分和权重
        attention = torch.bmm(query, key)  # B x C x C
        attention = F.softmax(attention, dim=2)

        # 重构特征
        out = torch.bmm(attention, value)  # B x C x HW
        out = out.view(batch_size, self.c2, H, W)
        out = self.output_conv(out)

        # 残差连接
        out = out + x

        return out


class AARM(nn.Module):
    """
    Attention-based Appearance Reconstruction Module (AARM)
    根据提供的架构图重新设计的模块，用于重构外观特征以增强区分能力
    """

    def __init__(self, c1, c2, *args):  # 修改参数列表以适配YOLOv5
        """
        初始化AARM模块

        参数:
            c1: 输入通道数/特征维度
            c2: 输出通道数/特征维度
            args: 额外参数
        """
        super(AARM, self).__init__()

        # 特征映射卷积层（适用于空间特征图）
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)

        # 使用SpatialAARMAdapter处理卷积特征图
        self.spatial_aarm = SpatialAARMAdapter(c2, c2)

    def forward(self, x):
        """前向传播"""
        # 应用卷积调整通道数
        x = self.conv(x)

        # 应用空间AARM模块
        out = self.spatial_aarm(x)

        return out


class AARMC3(nn.Module):
    """
    AARM模块与YOLOv5 C3模块的融合
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, *args):
        """
        初始化AARMC3模块

        参数:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck重复次数
            shortcut: 是否使用shortcut连接
            g: 分组卷积的组数
            e: 隐藏层通道数比例
        """
        super(AARMC3, self).__init__()

        c_ = int(c2 * e)  # 隐藏通道数

        # 确保通道数能被分组数整除
        if g > 1 and c_ % g != 0:
            c_ = math.ceil(c_ / g) * g

        # C3模块的组件
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # 创建Bottleneck序列，确保通道数兼容
        m = []
        for _ in range(n):
            m.append(Bottleneck(c_, c_, shortcut, g, e=1.0))
        self.m = nn.Sequential(*m)

        # AARM模块
        self.spatial_aarm = SpatialAARMAdapter(c2, c2)

    def forward(self, x):
        """前向传播"""
        # C3模块的处理
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        out = self.cv3(torch.cat((y1, y2), 1))

        # AARM模块处理
        out = self.spatial_aarm(out)

        return out


class AARMSPPF(nn.Module):
    """
    AARM模块与SPPF模块的融合
    """

    def __init__(self, c1, c2, k=5, *args):
        """
        初始化AARMSPPF模块

        参数:
            c1: 输入通道数
            c2: 输出通道数
            k: SPPF的kernel size
        """
        super(AARMSPPF, self).__init__()

        c_ = c1 // 2  # 隐藏通道数

        # SPPF模块的组件
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        # 确保池化保持空间尺寸不变
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 空间AARM模块
        self.spatial_aarm = SpatialAARMAdapter(c2, c2)

    def forward(self, x):
        """前向传播"""
        # SPPF模块的处理
        x = self.cv1(x)

        # 保存输入尺寸
        _, _, h, w = x.shape

        # 应用最大池化，并确保输出尺寸与输入相同
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制torch 1.9.0 max_pool2d()警告
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)

            # 确保所有特征具有相同的空间尺寸
            if y1.shape[2:] != x.shape[2:]:
                y1 = F.interpolate(y1, size=(h, w), mode='nearest')
            if y2.shape[2:] != x.shape[2:]:
                y2 = F.interpolate(y2, size=(h, w), mode='nearest')
            if y3.shape[2:] != x.shape[2:]:
                y3 = F.interpolate(y3, size=(h, w), mode='nearest')

            # 连接特征
            out = self.cv2(torch.cat((x, y1, y2, y3), 1))

        # AARM模块处理
        out = self.spatial_aarm(out)

        return out

class C2f_AARM(nn.Module):
    """
    C2f模块与AARM的融合版本
    结合C2f的高效特征融合和AARM的注意力增强机制
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """
        初始化C2f_AARM模块

        参数:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck重复次数
            shortcut: 是否使用shortcut连接
            g: 分组卷积的组数
            e: 隐藏层通道数比例
        """
        super(C2f_AARM, self).__init__()

        self.c_ = int(c2 * e)  # 隐藏通道数

        # 确保通道数能被分组数整除
        if g > 1 and self.c_ % g != 0:
            self.c_ = math.ceil(self.c_ / g) * g

        # C2f的核心组件
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)  # 输入分支卷积
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)  # 输出融合卷积

        # Bottleneck序列
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g) for _ in range(n))

        # AARM注意力增强模块 - 应用于最终输出
        self.aarm = SpatialAARMAdapter(c2, c2)

        # 可选：在中间特征上也应用轻量级注意力
        self.intermediate_aarm = SpatialAARMAdapter(self.c_, self.c_)
        self.use_intermediate_aarm = True  # 控制是否在中间特征使用AARM

    def forward(self, x):
        """前向传播"""
        # C2f的特征分割和处理
        y = list(self.cv1(x).chunk(2, 1))  # 分割为两个分支

        # 逐步添加Bottleneck处理的特征
        for i, m in enumerate(self.m):
            if self.use_intermediate_aarm and i == len(self.m) // 2:
                # 在中间层应用AARM增强
                enhanced_feature = self.intermediate_aarm(y[-1])
                y.append(m(enhanced_feature))
            else:
                y.append(m(y[-1]))

        # 融合所有特征
        fused_features = self.cv2(torch.cat(y, 1))

        # 应用AARM进行最终的注意力增强
        enhanced_output = self.aarm(fused_features)

        return enhanced_output


class C2f_AARM_Lite(nn.Module):
    """
    C2f_AARM的轻量化版本
    仅在输出层应用AARM，减少计算开销
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """
        初始化C2f_AARM_Lite模块

        参数同C2f_AARM
        """
        super(C2f_AARM_Lite, self).__init__()

        self.c_ = int(c2 * e)  # 隐藏通道数

        # 确保通道数能被分组数整除
        if g > 1 and self.c_ % g != 0:
            self.c_ = math.ceil(self.c_ / g) * g

        # C2f的核心组件
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g) for _ in range(n))

        # 仅在输出应用AARM
        self.aarm = SpatialAARMAdapter(c2, c2)

    def forward(self, x):
        """前向传播"""
        # 标准C2f处理
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        fused_features = self.cv2(torch.cat(y, 1))

        # 应用AARM增强
        enhanced_output = self.aarm(fused_features)

        return enhanced_output


class C2f_AARM_Adaptive(nn.Module):
    """
    C2f_AARM的自适应版本
    根据特征尺寸动态调整AARM的应用策略
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """
        初始化C2f_AARM_Adaptive模块
        """
        super(C2f_AARM_Adaptive, self).__init__()

        self.c_ = int(c2 * e)

        # 确保通道数能被分组数整除
        if g > 1 and self.c_ % g != 0:
            self.c_ = math.ceil(self.c_ / g) * g

        # C2f组件
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g) for _ in range(n))

        # 多尺度AARM模块
        self.aarm_output = SpatialAARMAdapter(c2, c2)

        # 可学习的权重参数，用于平衡原始特征和增强特征
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """前向传播"""
        # C2f处理
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        fused_features = self.cv2(torch.cat(y, 1))

        # 自适应特征增强
        enhanced_features = self.aarm_output(fused_features)

        # 可学习的特征融合
        output = self.alpha * enhanced_features + (1 - self.alpha) * fused_features

        return output



