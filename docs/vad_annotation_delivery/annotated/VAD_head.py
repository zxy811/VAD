# -*- coding: utf-8 -*-
# 【中文阅读注释：只新增 # 注释，原始可执行语句与 docstring 不变】
# 维度符号：B=batch 大小；L=图片队列长度；N=相机数；D=特征通道；A=检测查询数；
# P=预测地图实例槽位数；S=每个地图实例的点数；K=运动模式数；T=未来时间步数；
# G=当前样本真实目标数；M=当前样本真实地图实例数；R=解码器层数；Q=某处查询数。
# 常见 Tiny 配置：B=1，L=3，N=6，D=256，A=300，P=100，S=20，K=T=6，R=3。
# 这些是配置示例而非本文件所有分支的固定值；图像 H/W、G/M、筛选后的 Q 会变化。
# 模型参数：Embedding、Linear、Conv、LayerNorm 等可训练部分；权重在创建/初始化时产生，
# 训练 forward 计算场景特征，loss.backward 与 optimizer.step 在框架 Hook 中更新参数。
# 数据/GT/标定/mask/阈值不是可训练参数。中间特征虽不是独立参数，仍可承载梯度。
# 512 维查询嵌入可拆成 256 维 query_pos 和 256 维 query 内容；位置嵌入不是二维坐标。
# 坐标注明归一化或米制；不同坐标系不能直接相减。具体以当前函数运算为准。
# 语法：self 是当前对象；self.layer(...) 对 nn.Module 对象触发 __call__，再进入 forward。
# *args 展开/接收位置参数；**kwargs 展开/接收关键字参数；super 复用继承链方法。
# 注意：原作者部分英文 docstring 来自通用 DETR 模板，维度/名称可能与实际实现不同；
# 新增中文注释按本文件可执行代码及 Tiny 配置解释，不把模板中的 4D 框说法当成事实。
# 职责：创建查询、分类回归头与交互模块；由图像特征生成 BEV/检测/地图，再做运动和规划。
# 训练入口 VAD.forward_pts_train 调用本类 forward，随后单独调用 loss；测试调用 get_bboxes 解码。
# forward 的主要输出：[R,B,A,10] 框编码、[R,B,A,10] 类别 logits、[R,B,P,S,2] 地图点、
# [R,B,A,K,2T] 运动位移、[R,B,A,K] 模式 logits、[B,3,T,2] 自车命令分支位移。
# 检测框 10 维是编码形式，与 Dataset 框对象的 9 维不同；地图点主要为归一化二维坐标。
# 地图/检测/运动/规划都有语义可解释几何输出；中间 256 维是学习特征，没有逐维固定物理名称。
# 训练总损失由父类 BaseDetector._parse_losses 汇总；优化器 Hook 在本文件外进行 backward/step。

import copy
from math import pi, cos, sin

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.VAD.utils.traj_lr_warmup import get_traj_warmup_loss_weight
from projects.mmdet3d_plugin.VAD.utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)

# 【类】MLP：构造时执行__init__；实例可通过方法或nn.Module调用机制参与计算。
class MLP(nn.Module):
    # 【函数入口】MLP.__init__
    # 建立 Linear -> LayerNorm -> ReLU；Linear W=[hidden_unit,in_channels]，b=[hidden_unit]。
    # 这些参数在点/实例/batch之间共享；构造层不等于已经对输入数据执行一次forward。
    def __init__(self, in_channels, hidden_unit, verbose=False):
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        super(MLP, self).__init__()
        # 创建线性层：最后维in->out，W=[out,in]、b=[out]，在全部batch/序列位置共享；这里创建层而非执行forward。
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    # 【函数入口】MLP.forward
    # 输入 [...,in_channels]，线性层只变最后一维为hidden_unit，前面的batch/实例/点轴不变。
    # Linear计算 y=xW^T+b；LayerNorm按最后特征轴归一化；ReLU逐元素激活。
    def forward(self, x):
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        x = self.mlp(x)
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return x

# 【类】LaneNet：构造时执行__init__；实例可通过方法或nn.Module调用机制参与计算。
class LaneNet(nn.Module):
    # 【函数入口】LaneNet.__init__
    # 建立若干点级MLP。每层点特征与该实例的最大池化特征concat，因此下一层输入是2*hidden_unit。
    # 当前地图调用 LaneNet(256,128,3)，最终实例特征256维；历史编码分支可接不同输入通道。
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        super(LaneNet, self).__init__()
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.num_subgraph_layers = num_subgraph_layers
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        self.layer_seq = nn.Sequential()
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    # 【函数入口】LaneNet.forward
    # 地图输入 [B,P,S,D]，或历史分支的相应点序列。
    # 每层MLP将最后轴变128 -> 沿S取max=[B,P,128] -> 复制回每点 -> concat=[B,P,S,256]。
    # 末尾再次沿点轴max，输出[B,P,256]。池化丢掉点轴，保留实例归属，不会跨实例混池化。
    # 注意这里聚合学习的点特征，不是直接把20个(x,y)当作256维特征。
    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for name, layer in self.layer_seq.named_modules():
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                # 沿dim做极值归约，返回(极值,索引)；[0]取数值、[1]取索引，是否保留轴由keepdim决定。
                x_max = torch.max(x, -2)[0]
                # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
                x = torch.cat([x, x_max], dim=-1)
        # 沿dim做极值归约，返回(极值,索引)；[0]取数值、[1]取索引，是否保留轴由keepdim决定。
        x_max = torch.max(x, -2)[0]
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return x_max


# 【类】VADHead：构造时执行__init__；实例可通过方法或nn.Module调用机制参与计算。
# @...register_module()把类登记为名称->类的映射；导入时注册，不会在这里实例化或训练。
@HEADS.register_module()
class VADHead(DETRHead):
    """Head of VAD model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """
    # 【函数入口】VADHead.__init__
    # 从配置接收尺寸、阈值、各解码器与loss设置；函数参数默认值可能被Tiny配置覆盖。
    # 调用DETRHead父类时会构建Transformer/分类loss等并动态调用本类_init_layers。
    # 随后构建地图匹配器/采样器及地图、运动、规划loss。阈值/损失系数是手设常量，不训练。
    # code_weights/map_code_weights虽包装成Parameter，但requires_grad=False，不参与学习。
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 map_bbox_coder=None,
                 map_num_query=900,
                 map_num_classes=3,
                 map_num_vec=20,
                 map_num_pts_per_vec=2,
                 map_num_pts_per_gt_vec=2,
                 map_query_embed_type='all_pts',
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v0',
                 map_dir_interval=1,
                 map_code_size=None,
                 map_code_weights=None,
                loss_map_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_map_pts=dict(
                    type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0
                 ),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 tot_epoch=None,
                 use_traj_lr_warmup=False,
                 motion_decoder=None,
                 motion_map_decoder=None,
                 use_pe=False,
                 motion_det_score=None,
                 map_thresh=0.5,
                 dis_thresh=0.2,
                 pe_normalization=True,
                 ego_his_encoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanAgentDisLoss', loss_weight=0.1),
                 loss_plan_dir=dict(type='PlanMapThetaLoss', loss_weight=0.1),
                 ego_agent_decoder=None,
                 ego_map_decoder=None,
                 query_thresh=None,
                 query_use_fix_pad=None,
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 **kwargs):

        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.bev_h = bev_h
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.bev_w = bev_w
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.fp16_enabled = False
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.fut_ts = fut_ts
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.fut_mode = fut_mode
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.tot_epoch = tot_epoch
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.use_traj_lr_warmup = use_traj_lr_warmup
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.motion_decoder = motion_decoder
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.motion_map_decoder = motion_map_decoder
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.use_pe = use_pe
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.motion_det_score = motion_det_score
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_thresh = map_thresh
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.dis_thresh = dis_thresh
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.pe_normalization = pe_normalization
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.ego_his_encoder = ego_his_encoder
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.ego_fut_mode = ego_fut_mode
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.ego_agent_decoder = ego_agent_decoder
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.ego_map_decoder = ego_map_decoder
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.query_thresh = query_thresh
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.query_use_fix_pad = query_use_fix_pad
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.valid_fut_ts = valid_fut_ts

        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if loss_traj_cls['use_sigmoid'] == True:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.traj_num_cls = 1
        else:
          # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
          self.traj_num_cls = 2

        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.with_box_refine = with_box_refine
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.as_two_stage = as_two_stage
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.as_two_stage:
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            transformer['as_two_stage'] = self.as_two_stage
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if 'code_size' in kwargs:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.code_size = kwargs['code_size']
        else:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.code_size = 10
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if code_weights is not None:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.code_weights = code_weights
        else:
            # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if map_code_size is not None:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.map_code_size = map_code_size
        else:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.map_code_size = 10
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if map_code_weights is not None:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.map_code_weights = map_code_weights
        else:
            # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
            self.map_code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.bbox_coder = build_bbox_coder(bbox_coder)
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.pc_range = self.bbox_coder.pc_range
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.real_w = self.pc_range[3] - self.pc_range[0]
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.real_h = self.pc_range[4] - self.pc_range[1]
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.num_cls_fcs = num_cls_fcs - 1

        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.map_bbox_coder = build_bbox_coder(map_bbox_coder)
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_query_embed_type = map_query_embed_type
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_transform_method = map_transform_method
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        map_num_query = map_num_vec * map_num_pts_per_vec
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_num_query = map_num_query
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_num_classes = map_num_classes
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_num_vec = map_num_vec
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_num_pts_per_vec = map_num_pts_per_vec
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_dir_interval = map_dir_interval

        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if loss_map_cls['use_sigmoid'] == True:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.map_cls_out_channels = map_num_classes
        else:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.map_cls_out_channels = map_num_classes + 1

        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get('class_weight', None)
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if map_class_weight is not None and (self.__class__ is VADHead):
            # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
            assert isinstance(map_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(map_class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            map_bg_cls_weight = loss_map_cls.get('bg_cls_weight', map_class_weight)
            # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
            assert isinstance(map_bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(map_bg_cls_weight)}.'
            # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({'class_weight': map_class_weight})
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if 'bg_cls_weight' in loss_map_cls:
                loss_map_cls.pop('bg_cls_weight')
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.map_bg_cls_weight = map_bg_cls_weight
        
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.traj_bg_cls_weight = 0

        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        super(VADHead, self).__init__(*args, transformer=transformer, **kwargs)
        # Parameter把Tensor登记为模型参数；是否更新还由requires_grad及是否参与loss决定。
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        # Parameter把Tensor登记为模型参数；是否更新还由requires_grad及是否参与loss决定。
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)
        
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if kwargs['train_cfg'] is not None:
            # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
            assert 'map_assigner' in kwargs['train_cfg'], 'map assigner should be provided '\
                'when train_cfg is set.'
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            map_assigner = kwargs['train_cfg']['map_assigner']
            # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
            assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
            assert loss_map_bbox['loss_weight'] == map_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
            assert loss_map_iou['loss_weight'] == map_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
            assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be' \
                'exactly the same.'

            # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
            self.map_assigner = build_assigner(map_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
            self.map_sampler = build_sampler(sampler_cfg, context=self)
        
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_traj = build_loss(loss_traj)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_traj_cls = build_loss(loss_traj_cls)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_map_bbox = build_loss(loss_map_bbox)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_map_cls = build_loss(loss_map_cls)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_map_iou = build_loss(loss_map_iou)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_map_pts = build_loss(loss_map_pts)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_map_dir = build_loss(loss_map_dir)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_plan_reg = build_loss(loss_plan_reg)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_plan_bound = build_loss(loss_plan_bound)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_plan_col = build_loss(loss_plan_col)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.loss_plan_dir = build_loss(loss_plan_dir)

    # 【函数入口】VADHead._init_layers
    # 创建检测、地图、轨迹、模式分类、规划网络以及查询表。函数内暂存的Python list不是模型特征。
    # ModuleList/Sequential将层注册为子模块，参数才会随模型移动、保存和被优化器发现。
    # with_box_refine=True使用deepcopy为各解码层创建独立预测头；False列表复用同一对象，参数共享。
    # Embedding是可训练表，初始通常由PyTorch随机初始化；init_weights还可能重置部分参数。
    # Tiny query_embedding=[300,512]，BEV=[10000,256]，map_instance=[100,512]，map_pts=[20,512]。
    # 位置Linear(2,256) W=[256,2]、b=[256]，在所有查询之间共享，不是每个点一个W。
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        cls_branch = []
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for _ in range(self.num_reg_fcs):
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            cls_branch.append(nn.ReLU(inplace=True))
        # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        reg_branch = []
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for _ in range(self.num_reg_fcs):
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            reg_branch.append(nn.ReLU())
        # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        traj_branch = []
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for _ in range(self.num_reg_fcs):
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            traj_branch.append(nn.ReLU())
        # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
        traj_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
        traj_branch = nn.Sequential(*traj_branch)

        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        traj_cls_branch = []
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for _ in range(self.num_reg_fcs):
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            traj_cls_branch.append(nn.ReLU(inplace=True))
        # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
        traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        map_cls_branch = []
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for _ in range(self.num_reg_fcs):
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            map_cls_branch.append(nn.ReLU(inplace=True))
        # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        map_reg_branch = []
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for _ in range(self.num_reg_fcs):
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            map_reg_branch.append(nn.ReLU())
        # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)


        # 【函数入口】VADHead._get_clones
        # 内部辅助函数：deepcopy模块N次，各副本参数独立；不同于把同一module引用重复放进列表。
        def _get_clones(module, N):
            # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        motion_num_pred = (num_motion_decoder_layers + 1) if \
            self.as_two_stage else num_motion_decoder_layers
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.with_box_refine:
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.cls_branches = _get_clones(cls_branch, num_pred)
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.reg_branches = _get_clones(reg_branch, num_pred)
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.traj_branches = nn.ModuleList(
                [traj_branch for _ in range(motion_num_pred)])
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.traj_cls_branches = nn.ModuleList(
                [traj_cls_branch for _ in range(motion_num_pred)])
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if not self.as_two_stage:
            # 创建可学习查找表[num_embeddings,embedding_dim]；表值是参数，初值不表示已找到真实目标。
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            # 创建可学习查找表[num_embeddings,embedding_dim]；表值是参数，初值不表示已找到真实目标。
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.map_query_embed_type == 'all_pts':
                # 创建可学习查找表[num_embeddings,embedding_dim]；表值是参数，初值不表示已找到真实目标。
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                    self.embed_dims * 2)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            elif self.map_query_embed_type == 'instance_pts':
                # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
                self.map_query_embedding = None
                # 创建可学习查找表[num_embeddings,embedding_dim]；表值是参数，初值不表示已找到真实目标。
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                # 创建可学习查找表[num_embeddings,embedding_dim]；表值是参数，初值不表示已找到真实目标。
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)
        
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.motion_decoder is not None:
            # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
            self.motion_decoder = build_transformer_layer_sequence(self.motion_decoder)
            # 创建可学习查找表[num_embeddings,embedding_dim]；表值是参数，初值不表示已找到真实目标。
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)	
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.motion_mode_query.weight.requires_grad = True
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.use_pe:
                # 创建线性层：最后维in->out，W=[out,in]、b=[out]，在全部batch/序列位置共享；这里创建层而非执行forward。
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
        else:
            # 显式抛异常：表示该配置/类型分支不受支持，正常主线不应进入此处。
            raise NotImplementedError('Not implement yet')

        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.motion_map_decoder is not None:
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.lane_encoder = LaneNet(256, 128, 3)
            # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
            self.motion_map_decoder = build_transformer_layer_sequence(self.motion_map_decoder)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.use_pe:
                # 创建线性层：最后维in->out，W=[out,in]、b=[out]，在全部batch/序列位置共享；这里创建层而非执行forward。
                self.pos_mlp = nn.Linear(2, self.embed_dims)
        
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.ego_his_encoder is not None:
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.ego_his_encoder = LaneNet(2, self.embed_dims//2, 3)
        else:
            # 创建可学习查找表[num_embeddings,embedding_dim]；表值是参数，初值不表示已找到真实目标。
            self.ego_query = nn.Embedding(1, self.embed_dims)	

        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.ego_agent_decoder is not None:
            # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
            self.ego_agent_decoder = build_transformer_layer_sequence(self.ego_agent_decoder)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.use_pe:
                # 创建线性层：最后维in->out，W=[out,in]、b=[out]，在全部batch/序列位置共享；这里创建层而非执行forward。
                self.ego_agent_pos_mlp = nn.Linear(2, self.embed_dims)

        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.ego_map_decoder is not None:
            # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
            self.ego_map_decoder = build_transformer_layer_sequence(self.ego_map_decoder)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.use_pe:
                # 创建线性层：最后维in->out，W=[out,in]、b=[out]，在全部batch/序列位置共享；这里创建层而非执行forward。
                self.ego_map_pos_mlp = nn.Linear(2, self.embed_dims)

        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        ego_fut_decoder = []
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        ego_fut_dec_in_dim = self.embed_dims*2 + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims*2
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for _ in range(self.num_reg_fcs):
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            ego_fut_decoder.append(nn.ReLU())
        # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, self.ego_fut_mode*self.fut_ts*2))
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        # 创建线性层：最后维in->out，W=[out,in]、b=[out]，在全部batch/序列位置共享；这里创建层而非执行forward。
        self.agent_fus_mlp = nn.Sequential(
            nn.Linear(self.fut_mode*2*self.embed_dims, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

    # 【函数入口】VADHead.init_weights
    # 在模型初始化阶段调用：Transformer/交互模块Xavier，模式Embedding正交初始化，分类bias设先验。
    # 普通Embedding/Linear尚未被这里覆盖的部分沿用构造时默认初始化；不能笼统说所有bias都是0。
    # 这里无显式return，所以打印本函数调用结果通常为None，但模型内部权重已经改变。
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        self.transformer.init_weights()
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.loss_map_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for m in self.map_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.loss_traj_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for m in self.traj_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.map_reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.map_reg_branches[0][-1].bias.data[2:], 0.)
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.motion_decoder is not None:
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for p in self.motion_decoder.parameters():
                # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
                if p.dim() > 1:
                    # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                    nn.init.xavier_uniform_(p)
            # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
            nn.init.orthogonal_(self.motion_mode_query.weight)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.use_pe:
                # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.motion_map_decoder is not None:
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for p in self.motion_map_decoder.parameters():
                # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
                if p.dim() > 1:
                    # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                    nn.init.xavier_uniform_(p)
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for p in self.lane_encoder.parameters():
                # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
                if p.dim() > 1:
                    # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                    nn.init.xavier_uniform_(p)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.use_pe:
                # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                xavier_init(self.pos_mlp, distribution='uniform', bias=0.)
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.ego_his_encoder is not None:
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for p in self.ego_his_encoder.parameters():
                # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
                if p.dim() > 1:
                    # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                    nn.init.xavier_uniform_(p)
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.ego_agent_decoder is not None:
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for p in self.ego_agent_decoder.parameters():
                # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
                if p.dim() > 1:
                    # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                    nn.init.xavier_uniform_(p)
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.ego_map_decoder is not None:
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for p in self.ego_map_decoder.parameters():
                # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
                if p.dim() > 1:
                    # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                    nn.init.xavier_uniform_(p)

    # @auto_fp16(apply_to=('mlvl_feats'))
    # 【函数入口】VADHead.forward
    # 输入mlvl_feats为尺度列表，每项[B,N,D,h,w]；当前Tiny只有一个尺度。
    # only_bev=True直接返回Transformer.get_bev_features；False继续感知解码、运动和规划。
    # object/map query的512维嵌入在Transformer中拆成内容与位置，各256维；这里不是512个坐标。
    # 输出字典含原始logits/编码坐标；训练交给loss，测试交给get_bboxes。
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
            ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        # 解包首尺度[B,N,D,h,w]，只使用batch与相机数；_是被忽略的局部变量名，不是特殊轴运算。
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        dtype = mlvl_feats[0].dtype
        # 取可学习查询表[A,2D]，Tiny[300,512]；.to(dtype)匹配特征精度，不改变实例槽位。
        object_query_embeds = self.query_embedding.weight.to(dtype)
        
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.map_query_embed_type == 'all_pts':
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        elif self.map_query_embed_type == 'instance_pts':
            # 点位置表[S,2D]->[1,S,2D]，在P个地图实例间共享这套点嵌入。
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
            # 实例表[P,2D]->[P,1,2D]，同一实例的S个点共享该实例分量。
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
            # 广播相加[1,S,2D]+[P,1,2D]=[P,S,2D]；flatten(0,1)->[P*S,2D]，归属由轴索引确定。
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

        # 可学习BEV查询，经车辆状态注入；不是已经生成的鸟瞰RGB图片
        # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
        bev_queries = self.bev_embedding.weight.to(dtype)

        # 构造[B,Hbev,Wbev]位置编码输入mask，当前全0；用于LearnedPositionalEncoding，不是目标类别分割GT。
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        # BEV行/列位置查表并拼接，输出[B,D,Hbev,Wbev]，Tiny[B,256,100,100]；位置表可训练。
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
            
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            # 调用本项目VAD_transformer.py的VADPerceptionTransformer.forward；返回BEV、检测/地图逐层特征与参考点的tuple。
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                map_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
                map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        # 按Transformer返回tuple顺序解包；hs=[R,A,B,D]、map_hs=[R,P*S,B,D]；名称/顺序由调用双方约定。
        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs

        # 检测解码特征[R,A,B,D]->[R,B,A,D]，后续hs[lvl]是当前层[B,A,D]。
        hs = hs.permute(0, 2, 1, 3)
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        outputs_classes = []
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        outputs_coords = []
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        outputs_coords_bev = []
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        outputs_trajs = []
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        outputs_trajs_classes = []

        # 地图解码特征[R,P*S,B,D]->[R,B,P*S,D]，可view为[B,P,S,D]。
        map_hs = map_hs.permute(0, 2, 1, 3)
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        map_outputs_classes = []
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        map_outputs_coords = []
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        map_outputs_pts_coords = []
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        map_outputs_coords_bev = []

        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for lvl in range(hs.shape[0]):
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if lvl == 0:
                # 当前解码层使用的上一阶段参考坐标；经inverse_sigmoid进入logit空间再加残差
                reference = init_reference
            else:
                # 当前解码层使用的上一阶段参考坐标；经inverse_sigmoid进入logit空间再加残差
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                reference = inter_references[lvl - 1]
            # 旧参考点从[0,1]变为logit；后续加回归残差再sigmoid，不是简单在归一化坐标上直接线性相加。
            reference = inverse_sigmoid(reference)
            # 检测分类MLP把[B,A,D]最后轴变10，输出[B,A,10] logits；不是softmax概率。
            outputs_class = self.cls_branches[lvl](hs[lvl])
            # 该检测解码层共享回归头把[B,A,D]变[B,A,10]；10维含坐标、log尺寸、yaw sin/cos及速度编码。
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
            assert reference.shape[-1] == 3
            # 仅xy残差加旧参考点logit：[B,A,2]；z在下面使用编码索引4，不能把前三维一概当xyz。
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            # sigmoid逐元素映射到(0,1)，shape不变；类别分数各自独立，不保证沿类别/模式之和为1。
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            # 保存[B,A,2]归一化xy副本并detach，供运动/规划位置编码；不是米制框中心，也不携带这条坐标的梯度。
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            # 检测编码第4维是z；加旧三维参考点的第2维logit，保持[B,A,1]。
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            # sigmoid逐元素映射到(0,1)，shape不变；类别分数各自独立，不保证沿类别/模式之和为1。
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            # 检测x归一化值映射到[xmin,xmax]，Tiny x=-15+30*u；不是直接乘图像宽度。
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            # 检测y归一化值映射到[ymin,ymax]，Tiny y=-30+60*v；x/y轴物理跨度不同。
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            # 检测z参考值映射到[zmin,zmax]，Tiny z=-2+4*w；仍在当前局部坐标系。
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            # 当前层检测框10维编码[B,A,10]；x/y/z已映射局部米制，其他分量有自己的编码约定
            outputs_coord = tmp
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            outputs_classes.append(outputs_class)
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            outputs_coords.append(outputs_coord)
        
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for lvl in range(map_hs.shape[0]):
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if lvl == 0:
                # 当前解码层使用的上一阶段参考坐标；经inverse_sigmoid进入logit空间再加残差
                reference = map_init_reference
            else:
                # 当前解码层使用的上一阶段参考坐标；经inverse_sigmoid进入logit空间再加残差
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                reference = map_inter_references[lvl - 1]
            # 旧参考点从[0,1]变为logit；后续加回归残差再sigmoid，不是简单在归一化坐标上直接线性相加。
            reference = inverse_sigmoid(reference)
            # 点特征[B,P*S,D]先view[B,P,S,D]并沿S均值为[B,P,D]，分类输出[B,P,3]。
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs,self.map_num_vec, self.map_num_pts_per_vec,-1).mean(2)
            )
            # 地图回归头逐点D->2，输出[B,P*S,2]坐标残差，与同索引地图参考点配对。
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            # TODO: check the shape of reference
            # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
            assert reference.shape[-1] == 2
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            tmp[..., 0:2] += reference[..., 0:2]
            # sigmoid逐元素映射到(0,1)，shape不变；类别分数各自独立，不保证沿类别/模式之和为1。
            tmp = tmp.sigmoid() # cx,cy,w,h
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            # 保存[B,P,S,2]归一化地图点副本并detach，用于筛选和相对位置；阈值距离不是直接米制。
            map_outputs_coords_bev.append(map_outputs_pts_coord.clone().detach())
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            map_outputs_classes.append(map_outputs_class)
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            map_outputs_coords.append(map_outputs_coord)
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            map_outputs_pts_coords.append(map_outputs_pts_coord)
            
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.motion_decoder is not None:
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
            # motion_query
            # 最后检测层[B,A,D]->[A,B,D]；当前运动分支使用场景更新后的检测查询特征。
            motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
            # 取可学习模式表[K,D]=[6,256]；所有目标共享同一套模式嵌入，各目标内容不同。
            mode_query = self.motion_mode_query.weight  # [fut_mode, D]
            # [M, B, D], M=A*fut_mode
            # [A,1,B,D]+[1,K,1,D]->[A,K,B,D]，flatten(0,1)->[A*K,B,D]；加法不是concat。
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.use_pe:
                # 检测分支detach的归一化xy；复制模式后与A*K个查询对应
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                # 归一化检测坐标[B,A,2]经共享Linear(2,D)得到[B,A,D]位置嵌入，不是从BEV重新取图像特征。
                motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
                # [B,A,D]->[B,A,1,D]->[B,A,K,D]->[B,A*K,D]，同一目标K模式共用位置嵌入。
                motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                # 运动注意力的位置特征，最后维D=256；不是二维采样偏移或预测轨迹
                # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
                motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]
            else:
                # 运动注意力的位置特征，最后维D=256；不是二维采样偏移或预测轨迹
                motion_pos = None

            # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
            if self.motion_det_score is not None:
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                motion_score = outputs_classes[-1]
                # 沿dim做极值归约，返回(极值,索引)；[0]取数值、[1]取索引，是否保留轴由keepdim决定。
                max_motion_score = motion_score.max(dim=-1)[0]
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                # flatten合并指定连续轴，乘积为新轴长度；不会执行特征提取或池化。
                invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
            else:
                invalid_motion_idx = None

            # 普通自注意力：输入Q/K内容加位置，V为motion_query；输出[A*K,B,D]，Tiny8头每头32维。
            motion_hs = self.motion_decoder(
                query=motion_query,
                key=motion_query,
                value=motion_query,
                query_pos=motion_pos,
                key_pos=motion_pos,
                key_padding_mask=invalid_motion_idx)

            # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
            if self.motion_map_decoder is not None:
                # map preprocess
                # 检测分支detach的归一化xy；复制模式后与A*K个查询对应
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                # 目标位置[B,A,2]复制到K模式并展平为[B,A*K,2]，顺序与运动查询相同。
                motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                # 地图内容特征；具体阶段为点序列[B,P,S,D]或池化/筛选后实例序列，参照本函数入口
                # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
                map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
                # 按地图实例聚合点特征[B,P,S,D]->[B,P,D]；LaneNet内部点MLP与max，实例间不混池化。
                map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
                # 地图类别logits或在筛选函数内转为sigmoid分数，最后类别轴3
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                map_score = map_outputs_classes[-1]
                # 地图几何位置；不同阶段为归一化点/代表点/相对点，不是256维内容特征
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                map_pos = map_outputs_coords_bev[-1]
                # 为每个运动查询准备地图key/value；输出[B*A*K,P1,D]及坐标/布尔mask，随后改为序列优先。
                map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(
                    motion_coords, map_query, map_score, map_pos,
                    map_thresh=self.map_thresh, dis_thresh=self.dis_thresh,
                    pe_normalization=self.pe_normalization, use_fix_pad=True)
                # 地图内容特征；具体阶段为点序列[B,P,S,D]或池化/筛选后实例序列，参照本函数入口
                # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
                map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]
                # [A*K,B,D]->[B,A*K,D]->[B*A*K,D]->[1,B*A*K,D]；每个运动模式作为独立batch读取地图。
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

                # position encoding
                # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
                if self.use_pe:
                    # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                    (num_query, batch) = ca_motion_query.shape[:2] 
                    # 运动注意力的位置特征，最后维D=256；不是二维采样偏移或预测轨迹
                    # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
                    motion_pos = torch.zeros((num_query, batch, 2), device=motion_hs.device)
                    # 运动注意力的位置特征，最后维D=256；不是二维采样偏移或预测轨迹
                    # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
                    motion_pos = self.pos_mlp(motion_pos)
                    # 地图几何位置；不同阶段为归一化点/代表点/相对点，不是256维内容特征
                    # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
                    map_pos = map_pos.permute(1, 0, 2)
                    # 地图几何位置；不同阶段为归一化点/代表点/相对点，不是256维内容特征
                    # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
                    map_pos = self.pos_mlp(map_pos)
                else:
                    motion_pos, map_pos = None, None
                
                # Q为[1,B*A*K,D]运动内容，K/V为[P1,B*A*K,D]地图；Q/K加各自位置，V不加位置。
                ca_motion_query = self.motion_map_decoder(
                    query=ca_motion_query,
                    key=map_query,
                    value=map_query,
                    query_pos=motion_pos,
                    key_pos=map_pos,
                    key_padding_mask=key_padding_mask)
            else:
                # [A*K,B,D]->[B,A*K,D]->[B*A*K,D]->[1,B*A*K,D]；每个运动模式作为独立batch读取地图。
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            batch_size = outputs_coords_bev[-1].shape[0]
            # [A*K,B,D]->[B,A*K,D]->[B,A,K,D]，unflatten把目标与模式轴恢复，不改变数据个数。
            motion_hs = motion_hs.permute(1, 0, 2).unflatten(
                dim=1, sizes=(num_agent, self.fut_mode)
            )
            # [1,B*A*K,D]->[B*A*K,D]->[B,A,K,D]；squeeze(0)只删除指定轴，避免误删B=1等其他轴。
            ca_motion_query = ca_motion_query.squeeze(0).unflatten(
                dim=0, sizes=(batch_size, num_agent, self.fut_mode)
            )
            # 沿最后轴拼接两类交互特征：[B,A,K,D]+[B,A,K,D]->[B,A,K,2D]；不是逐元素相加。
            motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]
        else:
            # 显式抛异常：表示该配置/类型分支不受支持，正常主线不应进入此处。
            raise NotImplementedError('Not implement yet')

        # 共享轨迹MLP对每个目标模式最后轴2D->2T，得到[B,A,K,12]逐步二维位移。
        outputs_traj = self.traj_branches[0](motion_hs)
        # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
        outputs_trajs.append(outputs_traj)
        # 每模式2D特征映射为1个logit：[B,A,K,1]，squeeze(-1)后[B,A,K]。
        outputs_traj_class = self.traj_cls_branches[0](motion_hs)
        # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
        outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        (batch, num_agent) = motion_hs.shape[:2]
             
        # 各地图解码层的[B,P,3]沿新层轴堆叠为[R,B,P,3]，R=3。
        map_outputs_classes = torch.stack(map_outputs_classes)
        # 地图各层包围框[B,P,4]->[R,B,P,4]，不是检测框的10维编码。
        map_outputs_coords = torch.stack(map_outputs_coords)
        # 地图点[B,P,S,2]逐层堆叠为[R,B,P,S,2]；点坐标归一化，训练loss负责与GT坐标对齐。
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

        # 各检测层类别[B,A,10]->[R,B,A,10]，最后层通过[-1]取出。
        outputs_classes = torch.stack(outputs_classes)
        # 各检测层框[B,A,10]->[R,B,A,10]；解码与损失都要遵循编码约定。
        outputs_coords = torch.stack(outputs_coords)
        # 运动这里只产生一套[B,A,K,2T]，先stack得到[1,B,A,K,2T]；返回字典时repeat到R层。
        outputs_trajs = torch.stack(outputs_trajs)
        # 运动模式[B,A,K]先stack为[1,B,A,K]；重复输出层轴不代表又运行R次交互网络。
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)

        # planning
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        (batch, num_agent) = motion_hs.shape[:2]
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.ego_his_encoder is not None:
            # 若有历史编码则是历史特征；否则是学习的自车查询[B,1,D]，不是实际读取历史轨迹
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
        else:
            # 当前Tiny无历史编码器，使用可学习表[1,D]，扩展为[B,1,D]；传入的ego_his_trajs在此分支未被编码。
            ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
        # Interaction
        # 自车内容查询，当前分支来源为可训练ego_query表；后续注意力读取目标/地图
        ego_query = ego_his_feats
        # 位置为当前局部原点的二维零坐标，再经共享Linear编码为D维
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        ego_pos = torch.zeros((batch, 1, 2), device=ego_query.device)
        # 自车位置嵌入[B,1,D]，来自Linear(2,D)，不是米制坐标
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)
        # 目标类别logits[B,A,10]，筛选函数内部sigmoid后取类别最大值
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        agent_conf = outputs_classes[-1]
        # 把模式与2D特征合并：[B,A,K,2D]->[B,A,K*2D]，Tiny最后轴3072。
        agent_query = motion_hs.reshape(batch, num_agent, -1)
        # 共享MLP融合六模式：最后轴3072->256，输出[B,A,D]，供自车查询读取。
        agent_query = self.agent_fus_mlp(agent_query) # [B, A, fut_mode, 2*D] -> [B, A, D]
        # 目标检测归一化xy[B,A,2]；规划位置编码用，不是未来位置
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        agent_pos = outputs_coords_bev[-1]
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        agent_query, agent_pos, agent_mask = self.select_and_pad_query(
            agent_query, agent_pos, agent_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        # 目标位置的D维嵌入[B,A,D]，在目标交叉注意力中加给K
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        agent_pos_emb = self.ego_agent_pos_mlp(agent_pos)
        # ego <-> agent interaction
        # 自车-目标普通交叉注意力：Q自车[1,B,D]，K目标内容+位置，V目标内容；输出[1,B,D]。
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=agent_pos_emb.permute(1, 0, 2),
            key_padding_mask=agent_mask)

        # ego <-> map interaction
        # 位置为当前局部原点的二维零坐标，再经共享Linear编码为D维
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        ego_pos = torch.zeros((batch, 1, 2), device=agent_query.device)
        # 自车位置嵌入[B,1,D]，来自Linear(2,D)，不是米制坐标
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        ego_pos_emb = self.ego_map_pos_mlp(ego_pos)
        # 地图内容特征；具体阶段为点序列[B,P,S,D]或池化/筛选后实例序列，参照本函数入口
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
        # 按地图实例聚合点特征[B,P,S,D]->[B,P,D]；LaneNet内部点MLP与max，实例间不混池化。
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_conf = map_outputs_classes[-1]
        # 地图几何位置；不同阶段为归一化点/代表点/相对点，不是256维内容特征
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_pos = map_outputs_coords_bev[-1]
        # use the most close pts pos in each map inst as the inst's pos
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        batch, num_map = map_pos.shape[:2]
        # 每个实例每个点相对当前坐标原点的距离[B,P,S]；这里输入仍归一化，原点不是每辆车。
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        # 每地图实例代表点的索引，flatten后[B*P]；不是地图实例的类别
        # flatten合并指定连续轴，乘积为新轴长度；不会执行特征提取或池化。
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        # flatten合并指定连续轴，乘积为新轴长度；不会执行特征提取或池化。
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        # 高级索引为每个[B,P]实例选一个代表点；argmin是在归一化点距原点上求，不是每车重新求最近点。
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        map_query, map_pos, map_mask = self.select_and_pad_query(
            map_query, min_map_pos, map_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        map_pos_emb = self.ego_map_pos_mlp(map_pos)
        # Q是已融合目标的自车特征，K地图内容+位置，V地图内容；输出[1,B,D]。
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=map_pos_emb.permute(1, 0, 2),
            key_padding_mask=map_mask)

        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.ego_his_encoder is not None and self.ego_lcf_feat_idx is not None:
            # 拼接哪些分量由上方if/elif决定；当前ego_his_encoder=None且ego_lcf_feat_idx=None时[B,1,D]+[B,1,D]->[B,1,512]。
            ego_feats = torch.cat(
                [ego_his_feats,
                 ego_map_query.permute(1, 0, 2),
                 ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
                dim=-1
            )  # [B, 1, 2D+2]
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        elif self.ego_his_encoder is not None and self.ego_lcf_feat_idx is None:
            # 拼接哪些分量由上方if/elif决定；当前ego_his_encoder=None且ego_lcf_feat_idx=None时[B,1,D]+[B,1,D]->[B,1,512]。
            ego_feats = torch.cat(
                [ego_his_feats,
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is not None:                
            # 拼接哪些分量由上方if/elif决定；当前ego_his_encoder=None且ego_lcf_feat_idx=None时[B,1,D]+[B,1,D]->[B,1,512]。
            ego_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2),
                 ego_map_query.permute(1, 0, 2),
                 ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
                dim=-1
            )  # [B, 1, 2D+2]
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is None:                
            # 拼接哪些分量由上方if/elif决定；当前ego_his_encoder=None且ego_lcf_feat_idx=None时[B,1,D]+[B,1,D]->[B,1,512]。
            ego_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2),
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]  

        # Ego prediction
        # 规划MLP按最后轴映射为E*T*2=36个数值，随后reshape为[B,3,6,2]；是命令分支而非模式分数。
        outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], 
                                                      self.ego_fut_mode, self.fut_ts, 2)

        # 输出字典：检测[R,B,A,10]、地图[R,B,P,S,2]、运动[R,B,A,K,2T]、自车[B,E,T,2]；其中enc_*为None。
        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_traj_preds': outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1),
            'all_traj_cls_scores': outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1),
            'map_all_cls_scores': map_outputs_classes,
            'map_all_bbox_preds': map_outputs_coords,
            'map_all_pts_preds': map_outputs_pts_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'map_enc_cls_scores': None,
            'map_enc_bbox_preds': None,
            'map_enc_pts_preds': None,
            'ego_fut_preds': outputs_ego_trajs,
        }

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return outs

    # 【函数入口】VADHead.map_transform_box
    # 输入地图点[B,P*S,2]，重排[B,P,S,2]；按x/y分别取min/max得到实例包围框[B,P,4]。
    # 返回归一化(cx,cy,w,h)框和点序列。包围框由点计算，未引入另一套独立地图框回归器。
    # y_first只控制输入坐标分量次序，不改变点与实例的索引对应关系。
    def map_transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                                self.map_num_pts_per_vec,2)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.map_transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            # 沿dim做极值归约，返回(极值,索引)；[0]取数值、[1]取索引，是否保留轴由keepdim决定。
            xmin = pts_x.min(dim=2, keepdim=True)[0]
            # 沿dim做极值归约，返回(极值,索引)；[0]取数值、[1]取索引，是否保留轴由keepdim决定。
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            # 沿dim做极值归约，返回(极值,索引)；[0]取数值、[1]取索引，是否保留轴由keepdim决定。
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            # 沿dim做极值归约，返回(极值,索引)；[0]取数值、[1]取索引，是否保留轴由keepdim决定。
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            # 显式抛异常：表示该配置/类型分支不受支持，正常主线不应进入此处。
            raise NotImplementedError
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return bbox, pts_reshape

    # 【函数入口】VADHead._get_target_single
    # 处理一个样本、一层检测输出：cls=[A,C]、bbox=[A,10]；GT框[G,9]、attrs[G,34]。
    # attrs前12列是未来位移，接着6列是有效mask；GT框是米制，loss前另做编码。
    # assigner配对预测与GT；PseudoSampler仅提取正负索引，不是DataLoader的样本Sampler。
    # pos_inds是预测槽位索引；pos_assigned_gt_inds是对应GT行索引；两种索引不能互换。
    # 同一个检测匹配用于框、类别与未来轨迹，运动不会重新随机匹配另一辆车。
    # 返回每个预测的target、weight、mask及正负索引，未匹配轨迹权重为0。
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        # 从[G,34]取前2T列，Tiny[G,12]；每行与同索引GT框/类别对应。
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts*2]
        # 取第2T到3T列，即[G,6]有效时间mask；是数据标注，不是模型推理时预测的有效性。
        gt_fut_masks = gt_attr_labels[:, self.fut_ts*2:self.fut_ts*3]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # 匹配到GT的预测槽位索引，长度Npos；不是GT列表中的行索引
        pos_inds = sampling_result.pos_inds
        # 未匹配GT的预测槽位索引，分类作为背景；回归权重通常为0
        neg_inds = sampling_result.neg_inds

        # label targets
        # 每个预测槽位的类别target，未匹配为背景编号（检测10/地图3）
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        # 把匹配GT类别写到预测正槽位，未匹配仍是背景类别编号。
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # 分类项权重[Nquery]，与预测槽位一一对应
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # 预测槽位对应GT框，只有正槽位填匹配GT，未匹配保持零占位
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        # 框回归权重，正槽位为1，背景槽位为0，再按编码分量权重缩放
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        bbox_weights = torch.zeros_like(bbox_pred)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        traj_weights = torch.zeros_like(traj_targets)
        # 预测正样本槽位pos_inds对应GT行pos_assigned_gt_inds；复制同一目标未来，不能直接用pos_inds索引GT。
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        traj_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        # GT时间有效性mask，来源标注；取样缺失未来时为0，不是推理预测的可见性
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks

        # Extra future timestamp mask for controlling pred horizon
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                   dtype=torch.float32, device=bbox_pred.device)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask

        # DETR
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return (
            labels, label_weights, bbox_targets, bbox_weights, traj_targets,
            traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )

    # 【函数入口】VADHead._map_get_target_single
    # 单样本地图cls=[P,3]、pts=[P,S,2]；GT等价排列[M,O,S,2]，O随模式/补齐方案而定。
    # MapHungarianAssigner3D返回实例匹配和每个预测-GT组合的最佳点顺序order_index[P,M]。
    # 用pos_inds和GT行索引共同取assigned_shift，再取GT的那一套点排列监督预测。
    # 预测点不需要先聚类才能知道实例：P维槽位已固定归属，每槽S个点。
    def _map_get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        gt_c = gt_bboxes.shape[-1]
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # 匹配到GT的预测槽位索引，长度Npos；不是GT列表中的行索引
        pos_inds = sampling_result.pos_inds
        # 未匹配GT的预测槽位索引，分类作为背景；回归权重通常为0
        neg_inds = sampling_result.neg_inds
        # label targets
        # 每个预测槽位的类别target，未匹配为背景编号（检测10/地图3）
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        # 把匹配GT类别写到预测正槽位，未匹配仍是背景类别编号。
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # 分类项权重[Nquery]，与预测槽位一一对应
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        # 预测槽位对应GT框，只有正槽位填匹配GT，未匹配保持零占位
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        # 框回归权重，正槽位为1，背景槽位为0，再按编码分量权重缩放
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        bbox_weights = torch.zeros_like(bbox_pred)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        bbox_weights[pos_inds] = 1.0
        # pts targets
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if order_index is None:
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            # 对每个正预测，用(预测槽位,GT行)查最佳点排列编号；不是类别编号或时间索引。
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        # 地图每预测槽的GT点序列[P,S,2]，已选择等价排列，未匹配为零占位
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        # 地图点回归权重[P,S,2]，匹配实例有效；不等于注意力权重
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        pts_weights = torch.zeros_like(pts_targets)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        pts_weights[pos_inds] = 1.0
        # DETR
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        # 按[GT行,等价排列编号,:,:]取[S,2]点序列写入对应预测槽，保持实例与点顺序对齐。
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    # 【函数入口】VADHead.get_targets
    # 用multi_apply对batch中的每个样本调用_get_target_single；可变GT数用list保持。
    # 返回各样本的target列表及正/负样本总数，用于分类归一化和轨迹权重。
    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, traj_targets_list, traj_weights_list,
         gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
         )
        # batch内正匹配数，用作loss归一化；多卡reduce_mean不是目标类别平均
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        # 背景预测槽位数，分类平均因子按配置决定是否计入
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                traj_targets_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    # 【函数入口】VADHead.map_get_targets
    # 与get_targets类似，但调用_map_get_target_single，保留每个实例的S个点及最佳GT排列。
    # 跨batch拼接发生在后面的map_loss_single；这里没有把不同地图实例的点混成一条线。
    def map_get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        # batch内正匹配数，用作loss归一化；多卡reduce_mean不是目标类别平均
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        # 背景预测槽位数，分类平均因子按配置决定是否计入
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    # 【函数入口】VADHead.loss_planning
    # ego_fut_preds=[B,E,T,2]，E=3命令分支；GT=[B,T,2]、mask=[B,T]、cmd=[B,E]。
    # 把同一专家GT复制给3分支，再用cmd与mask使仅对应命令/有效时间步的回归项生效。
    # 边界/碰撞/方向约束使用预测地图和目标运动，具体实现VAD/utils/plan_loss.py。
    # 训练约束是可微loss，不等于强化学习reward；不需要在这里运行车辆仿真交互。
    def loss_planning(self,
                      ego_fut_preds,
                      ego_fut_gt,
                      ego_fut_masks,
                      ego_fut_cmd,
                      lane_preds,
                      lane_score_preds,
                      agent_preds,
                      agent_fut_preds,
                      agent_score_preds,
                      agent_fut_cls_preds):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """

        # 同一专家GT[B,T,2]复制到[B,E,T,2]；是否监督该分支由cmd权重控制，不代表有3条不同GT。
        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        # cmd[B,E]扩展[B,E,1,1]，mask[B,T]扩展[B,1,T,1]，广播相乘得[B,E,T,1]。
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        # repeat按各轴次数复制数值；复制出的场景特征不是新建独立可学习参数。
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        # 用[B,E,T,2]命令×时间权重监督自车逐步位移，非对应命令分支的这一回归权重为0。
        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        # 用选中命令的自车预测与预测道路边界计算约束，具体几何在VAD/utils/plan_loss.py。
        loss_plan_bound = self.loss_plan_bound(
            ego_fut_preds[ego_fut_cmd==1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        # 用自车与预测其他目标未来轨迹计算碰撞相关约束；同时使用目标/模式分数，细节在plan_loss.py。
        loss_plan_col = self.loss_plan_col(
            ego_fut_preds[ego_fut_cmd==1],
            agent_preds,
            agent_fut_preds,
            agent_score_preds,
            agent_fut_cls_preds,
            weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
        )

        # 用预测地图方向约束规划方向；这不是调用车辆仿真后的强化学习奖励。
        loss_plan_dir = self.loss_plan_dir(
            ego_fut_preds[ego_fut_cmd==1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
            loss_plan_bound = torch.nan_to_num(loss_plan_bound)
            loss_plan_col = torch.nan_to_num(loss_plan_col)
            loss_plan_dir = torch.nan_to_num(loss_plan_dir)
        
        loss_plan_dict = dict()
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_plan_dict['loss_plan_reg'] = loss_plan_l1
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_plan_dict['loss_plan_bound'] = loss_plan_bound
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_plan_dict['loss_plan_col'] = loss_plan_col
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_plan_dict['loss_plan_dir'] = loss_plan_dir

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return loss_plan_dict
    
    # 【函数入口】VADHead.loss_single
    # 一层检测/运动loss：先get_targets匹配GT，再把B与A合并计算分类与回归。
    # normalize_bbox把GT9维编码为10维（包含尺寸log、yaw的sin/cos），不是所有分量都归一化到[0,1]。
    # 运动预测reshape=[B*A,K,T,2]，按选中模式做带mask的逐步位移L1；模式标签做focal分类。
    # 非正样本运动回归权重为0；get_traj_cls_target对负样本设背景编号K。
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    traj_preds,
                    traj_cls_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # 每个预测槽位的类别target，未匹配为背景编号（检测10/地图3）
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        labels = torch.cat(labels_list, 0)
        # 分类项权重[Nquery]，与预测槽位一一对应
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        label_weights = torch.cat(label_weights_list, 0)
        # 预测槽位对应GT框，只有正槽位填匹配GT，未匹配保持零占位
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        bbox_targets = torch.cat(bbox_targets_list, 0)
        # 框回归权重，正槽位为1，背景槽位为0，再按编码分量权重缩放
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        traj_targets = torch.cat(traj_targets_list, 0)
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        traj_weights = torch.cat(traj_weights_list, 0)
        # GT时间有效性mask，来源标注；取样缺失未来时为0，不是推理预测的可见性
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        # 分类loss的平均因子，至少1以避免除零，不是需要训练的系数
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.sync_cls_avg_factor:
            # 分类loss的平均因子，至少1以避免除零，不是需要训练的系数
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        # 分类loss的平均因子，至少1以避免除零，不是需要训练的系数
        cls_avg_factor = max(cls_avg_factor, 1)
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        # batch内正匹配数，用作loss归一化；多卡reduce_mean不是目标类别平均
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        # batch内正匹配数，用作loss归一化；多卡reduce_mean不是目标类别平均
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # GT框编码；检测normalize_bbox包含log尺寸/sin-cos yaw，不保证每维在0到1
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        # 有限数值筛选mask，防止编码GT中的NaN/Inf污染loss
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        # 框回归权重，正槽位为1，背景槽位为0，再按编码分量权重缩放
        bbox_weights = bbox_weights * self.code_weights
        # 检测编码预测与编码GT做L1，应用正样本和分量权重后归一化，返回可微标量。
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # traj regression loss
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

        # 未匹配GT的预测槽位索引，分类作为背景；回归权重通常为0
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        neg_inds = (bbox_weights[:, 0] == 0)
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks, neg_inds)

        # 被选模式逐步位移[Nbox,2T]与GT比较，权重包含正匹配、未来有效性及valid_fut_ts。
        loss_traj = self.loss_traj(
            best_traj_preds[isnotnan],
            traj_targets[isnotnan],
            traj_weights[isnotnan],
            avg_factor=num_total_pos)

        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.use_traj_lr_warmup:
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
            loss_traj = loss_scale_factor * loss_traj

        # traj classification loss
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.traj_bg_cls_weight
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.sync_cls_avg_factor:
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        # K个运动模式logits与最佳模式编号做分类loss；不是每个未来时间步分类，也不是sigmoid结果直接argmax监督。
        loss_traj_cls = self.loss_traj_cls(
            traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
        )

        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    # 【函数入口】VADHead.get_best_fut_preds
    # 输入[Nbox,K,T,2]、GT[Nbox,T,2]、mask[Nbox,T]。
    # 先沿时间cumsum得到相对起点累计位置，计算距离后乘mask，然后直接取[..., -1]选argmin模式。
    # 重要边界：末步mask=0可能让模式距离都为0；这不是自动选择最后一个有效时间步。
    # 只用终点距离选择模式；选中后返回该模式原始逐步位移[Nbox,2T]，不是累计位置。
    def get_best_fut_preds(self,
             traj_preds,
             traj_targets,
             gt_fut_masks):
        """"Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        # 沿时间累计预测位移得到相对当前起点位置[Nbox,K,T,2]，未自动加目标框中心
        # cumsum沿指定轴累加；这里轨迹时间轴的逐步位移变成相对起点累计位置，shape不变。
        cum_traj_preds = traj_preds.cumsum(dim=-2)
        # GT位移累计为相对起点位置[Nbox,T,2]，与预测使用同一参考系
        # cumsum沿指定轴累加；这里轨迹时间轴的逐步位移变成相对起点累计位置，shape不变。
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        # GT[Nbox,1,T,2]与预测[Nbox,K,T,2]广播相减并沿xy范数，输出[Nbox,K,T]。
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        # [Nbox,K,T]乘[Nbox,1,T]，无GT的时间步距离置零；这一步没有改变轨迹数或自动删时间步。
        dist = dist * gt_fut_masks[:, None, :]
        # 只取时间轴最后一步的距离，得到[Nbox,K]；末步mask无效时可能全0，原实现不自动找有效末步。
        dist = dist[..., -1]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        # 对每个检测预测的K种模式取终点距离最小编号[Nbox]；tolist变Python列表用于后续高级索引。
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        # 高级索引一一配对box_idx和该box的最佳mode_idx：[Nbox,K,T,2]->[Nbox,T,2]->[Nbox,2T]。
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts*2)

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return best_traj_preds

    # 【函数入口】VADHead.get_traj_cls_target
    # 使用与get_best_fut_preds一致的累计终点规则得到模式编号[Nbox]；负预测编号设K作为背景。
    # sigmoid focal中背景表示所有K个模式都是负类；不是给每个未来时间步单独分类。
    # 同样保留末步无效的实现边界；这段索引标签构造不是可训练参数。
    def get_traj_cls_target(self,
             traj_preds,
             traj_targets,
             gt_fut_masks,
             neg_inds):
        """"Get Trajectory mode classification target.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            traj_labels (Tensor): traj cls labels (num_box_preds,).
        """

        # 沿时间累计预测位移得到相对当前起点位置[Nbox,K,T,2]，未自动加目标框中心
        # cumsum沿指定轴累加；这里轨迹时间轴的逐步位移变成相对起点累计位置，shape不变。
        cum_traj_preds = traj_preds.cumsum(dim=-2)
        # GT位移累计为相对起点位置[Nbox,T,2]，与预测使用同一参考系
        # cumsum沿指定轴累加；这里轨迹时间轴的逐步位移变成相对起点累计位置，shape不变。
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        # GT[Nbox,1,T,2]与预测[Nbox,K,T,2]广播相减并沿xy范数，输出[Nbox,K,T]。
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        # [Nbox,K,T]乘[Nbox,1,T]，无GT的时间步距离置零；这一步没有改变轨迹数或自动删时间步。
        dist = dist * gt_fut_masks[:, None, :]
        # 只取时间轴最后一步的距离，得到[Nbox,K]；末步mask无效时可能全0，原实现不自动找有效末步。
        dist = dist[..., -1]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        # 每预测槽取一个模式编号[Nbox]；训练标签来自当前预测与GT距离，不是预先固定左转/右转编号。
        traj_labels = torch.argmin(dist, dim=-1)
        # 未匹配检测查询设置背景编号K；对K类sigmoid focal意味着K个模式都为负，不是第K+1个输出logit。
        traj_labels[neg_inds] = self.fut_mode

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return traj_labels

    # 【函数入口】VADHead.map_loss_single
    # 地图单层cls=[B,P,3]、box=[B,P,4]、pts=[B,P,S,2]。
    # 匹配得到米制GT点，normalize_2d_pts后与归一化预测计算点L1；方向项先反归一化再作差。
    # 预测/GT点数不同则插值对齐。相邻点方向差：[B*P,S-interval,2]。
    # 某些bbox/IoU项虽计算，配置loss_weight为0时不贡献总loss；不应据函数存在认定已启用。
    def map_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
 
        # 每个预测槽位的类别target，未匹配为背景编号（检测10/地图3）
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        labels = torch.cat(labels_list, 0)
        # 分类项权重[Nquery]，与预测槽位一一对应
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        label_weights = torch.cat(label_weights_list, 0)
        # 预测槽位对应GT框，只有正槽位填匹配GT，未匹配保持零占位
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        bbox_targets = torch.cat(bbox_targets_list, 0)
        # 框回归权重，正槽位为1，背景槽位为0，再按编码分量权重缩放
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # 地图每预测槽的GT点序列[P,S,2]，已选择等价排列，未匹配为零占位
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        pts_targets = torch.cat(pts_targets_list, 0)
        # 地图点回归权重[P,S,2]，匹配实例有效；不等于注意力权重
        # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        # 分类loss的平均因子，至少1以避免除零，不是需要训练的系数
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.sync_cls_avg_factor:
            # 分类loss的平均因子，至少1以避免除零，不是需要训练的系数
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        # 分类loss的平均因子，至少1以避免除零，不是需要训练的系数
        cls_avg_factor = max(cls_avg_factor, 1)
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        # batch内正匹配数，用作loss归一化；多卡reduce_mean不是目标类别平均
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        # batch内正匹配数，用作loss归一化；多卡reduce_mean不是目标类别平均
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # GT框编码；检测normalize_bbox包含log尺寸/sin-cos yaw，不保证每维在0到1
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        # 有限数值筛选mask，防止编码GT中的NaN/Inf污染loss
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        # 框回归权重，正槽位为1，背景槽位为0，再按编码分量权重缩放
        bbox_weights = bbox_weights * self.map_code_weights

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan,:4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        # 地图米制GT点按pc_range缩放至归一化坐标，与预测坐标比较
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
            pts_preds = pts_preds.permute(0,2,1)
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # 归一化地图预测点与已按最佳排列对齐的GT点比较；每实例点轴S保留，按pts_weights和正样本数归一化。
        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)

        # 相邻点差分对应权重，时间/点数轴缩短map_dir_interval
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        dir_weights = pts_weights[:, :-self.map_dir_interval,0]
        # 地图归一化预测点恢复为局部米制点，供几何方向损失
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        # 米制预测地图相邻点向量差[Npos_or_all,S-interval,2]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        denormed_pts_preds_dir = denormed_pts_preds[:,self.map_dir_interval:,:] - \
            denormed_pts_preds[:,:-self.map_dir_interval,:]
        # GT地图相邻点向量差，与预测方向在同一米制坐标中比较
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        pts_targets_dir = pts_targets[:, self.map_dir_interval:,:] - pts_targets[:,:-self.map_dir_interval,:]

        # 预测与GT的米制相邻点差向量比较方向，点数S减少interval；不是车辆未来朝向loss。
        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan,:,:],
            pts_targets_dir[isnotnan,:,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    # 【函数入口】VADHead.loss
    # 总训练loss入口。检测GT框先取gravity_center恢复几何中心，与编码预测对齐。
    # 逐解码层multi_apply调用loss_single/map_loss_single，保存最终层和辅助层loss。
    # 地图GT对象在此读取bbox/固定采样点/等价排列属性并搬到预测设备。
    # 运动前向只计算一套最后检测层交互，输出里按层repeat；辅助层并非独立运动网络。
    # 规划输入坐标、类别、目标轨迹来自预测；专家自车GT只用于规划监督与mask。
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # 创建副本；clone默认保留梯度联系，deepcopy用于独立容器/模块，不能把两者都说成detach。
        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        all_cls_scores = preds_dicts['all_cls_scores']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        all_bbox_preds = preds_dicts['all_bbox_preds']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        all_traj_preds = preds_dicts['all_traj_preds']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        enc_cls_scores = preds_dicts['enc_cls_scores']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        ego_fut_preds = preds_dicts['ego_fut_preds']

        num_dec_layers = len(all_cls_scores)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        device = gt_labels_list[0].device

        # 每样本框对象取gravity_center[G,3]与其余属性[G,6]拼回[G,9]，转到预测设备；恢复几何中心。
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        losses_cls, losses_bbox, loss_traj, loss_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_bboxes_ignore_list)
        

        num_dec_layers = len(map_all_cls_scores)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        device = map_gt_labels_list[0].device

        # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.map_gt_shift_pts_pattern == 'v0':
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        elif self.map_gt_shift_pts_pattern == 'v1':
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        elif self.map_gt_shift_pts_pattern == 'v2':
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        elif self.map_gt_shift_pts_pattern == 'v3':
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        elif self.map_gt_shift_pts_pattern == 'v4':
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]
        else:
            # 显式抛异常：表示该配置/类型分支不受支持，正常主线不应进入此处。
            raise NotImplementedError
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        map_losses_cls, map_losses_bbox, map_losses_iou, \
            map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        # 按名称收集各分支标量loss；包含loss的键将由BaseDetector._parse_losses用于汇总
        loss_dict = dict()
        # loss from the last decoder layer
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_cls'] = losses_cls[-1]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_bbox'] = losses_bbox[-1]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_traj'] = loss_traj[-1]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_traj_cls'] = loss_traj_cls[-1]
        # loss from the last decoder layer
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # Planning Loss
        # [B,1,T,2]->[B,T,2]；只去指定维度，不删batch轴B=1。
        ego_fut_gt = ego_fut_gt.squeeze(1)
        # [B,1,1,T]->[B,T]，两次squeeze(1)逐步去格式化轴。
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        # [B,1,1,E]->[B,E]，命令用于选择对应规划分支。
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        batch, num_agent = all_traj_preds[-1].shape[:2]
        # 目标多模式未来逐步位移[B,A,K,T,2]，规划碰撞loss会按实现进一步累加
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
        # 目标模式分数[B,A,K]；传规划loss时使用sigmoid，非softmax归一化和为1
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)
        # sigmoid逐元素映射到(0,1)，shape不变；类别分数各自独立，不保证沿类别/模式之和为1。
        loss_plan_input = [ego_fut_preds, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
                           map_all_pts_preds[-1], map_all_cls_scores[-1].sigmoid(),
                           all_bbox_preds[-1][..., 0:2], agent_fut_preds,
                           all_cls_scores[-1].sigmoid(), agent_fut_cls_preds.sigmoid()]

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        loss_planning_dict = self.loss_planning(*loss_plan_input)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_plan_reg'] = loss_planning_dict['loss_plan_reg']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_plan_bound'] = loss_planning_dict['loss_plan_bound']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_plan_col'] = loss_planning_dict['loss_plan_col']
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        loss_dict['loss_plan_dir'] = loss_planning_dict['loss_plan_dir']

        # loss from other decoder layers
        num_dec_layer = 0
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i in zip(
            map_losses_cls[:-1],
            map_losses_bbox[:-1],
            map_losses_iou[:-1],
            map_losses_pts[:-1],
            map_losses_dir[:-1]
        ):
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if enc_cls_scores is not None:
            # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 gt_bboxes_ignore)
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict['enc_loss_cls'] = enc_loss_cls
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if map_enc_cls_scores is not None:
            # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i])
                for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            map_enc_loss_cls, map_enc_loss_bbox, map_enc_loss_iou, \
                 map_enc_loss_pts, map_enc_loss_dir = \
                self.map_loss_single(
                    map_enc_cls_scores, map_enc_bbox_preds,
                    map_enc_pts_preds, map_gt_bboxes_list,
                    map_binary_labels_list, map_gt_pts_list,
                    map_gt_bboxes_ignore
                )
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict['enc_loss_map_cls'] = map_enc_loss_cls
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict['enc_loss_map_bbox'] = map_enc_loss_bbox
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict['enc_loss_map_iou'] = map_enc_loss_iou
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict['enc_loss_map_pts'] = map_enc_loss_pts
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            loss_dict['enc_loss_map_dir'] = map_enc_loss_dir

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return loss_dict

    # 【函数入口】VADHead.get_bboxes
    # 推理后处理，调用检测bbox_coder和map_bbox_coder.decode，主要用最后解码层结果。
    # 当前NMS-free coder是按分数/范围筛选，不应按旧docstring说这里进行了常规NMS。
    # 检测decoded框z由几何中心减半高成为底部中心，再按metadata中的box_type_3d包装。
    # 每样本返回框、分数、类别、运动位移、地图框/分数/类别/点，数目由coder筛选决定。
    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        # 进入core/bbox/coders/nms_free_coder.py解码最后检测层，按分数/范围筛选并同步取对应运动模式。
        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        # 进入core/bbox/coders/map_nms_free_coder.py；地图归一化点反归一化到局部米制坐标，按分数选择。
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
        assert len(det_preds_dicts) == len(map_preds_dicts), \
             'len(preds_dict) should be equal to len(map_preds_dicts)'
        # 每个batch样本的解码结果列表；各样本实例数可不同
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        ret_list = []
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for i in range(num_samples):
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            preds = det_preds_dicts[i]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            bboxes = preds['bboxes']
            # decoded框z中心减0.5*高度转为底面中心，与LiDARInstance3DBoxes内部约定一致；x/y不变。
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            code_size = bboxes.shape[-1]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            scores = preds['scores']
            # 每个预测槽位的类别target，未匹配为背景编号（检测10/地图3）
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            labels = preds['labels']
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            trajs = preds['trajs']

            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            map_preds = map_preds_dicts[i]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            map_bboxes = map_preds['map_bboxes']
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            map_scores = map_preds['map_scores']
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            map_labels = map_preds['map_labels']
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            map_pts = map_preds['map_pts']

            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            ret_list.append([bboxes, scores, labels, trajs, map_bboxes,
                             map_scores, map_labels, map_pts])

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return ret_list

    # 【函数入口】VADHead.select_and_pad_pred_map
    # 运动-地图交互准备。实际forward传入motion_pos=[B,A*K,2]，这里num_agent=A*K而非真实目标数G。
    # 输入map_query=[B,P,D]、map_score为logits[B,P,3]、map_pos=[B,P,S,2]。
    # 注意实现先取每个实例中距当前坐标原点最近的点为代表点，不是为每个agent重新求最近点。
    # 按sigmoid后最大类别分数筛选实例；batch内补齐到共同P1；True padding mask表示忽略该键。
    # 再为每个运动查询复制地图，以代表点与该运动查询坐标之差计算距离/相对位置。
    # 这些坐标来自归一化预测，因此dis_thresh=0.2不是直接0.2米；x/y物理跨度也不同。
    # 输出[B*A*K,P1(+1),D]、同前两轴的位置[...,2]与mask；最后虚拟零特征槽可避免所有键被mask。
    def select_and_pad_pred_map(
        self,
        motion_pos,
        map_query,
        map_score,
        map_pos,
        map_thresh=0.5,
        dis_thresh=None,
        pe_normalization=True,
        use_fix_pad=False
    ):
        """select_and_pad_pred_map.
        Args:
            motion_pos: [B, A, 2]
            map_query: [B, P, D].
            map_score: [B, P, 3].
            map_pos: [B, P, pts, 2].
            map_thresh: map confidence threshold for filtering low-confidence preds
            dis_thresh: distance threshold for masking far maps for each agent in cross-attn
            use_fix_pad: always pad one lane instance for each batch
        Returns:
            selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
            selected_map_pos: [B*A, P1(+1), 2]
            selected_padding_mask: [B*A, P1(+1)]
        """
        
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if dis_thresh is None:
            # 显式抛异常：表示该配置/类型分支不受支持，正常主线不应进入此处。
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        batch, num_map = map_pos.shape[:2]
        # 每个实例每个点相对当前坐标原点的距离[B,P,S]；这里输入仍归一化，原点不是每辆车。
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        # 每地图实例代表点的索引，flatten后[B*P]；不是地图实例的类别
        # flatten合并指定连续轴，乘积为新轴长度；不会执行特征提取或池化。
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        # flatten合并指定连续轴，乘积为新轴长度；不会执行特征提取或池化。
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        # 高级索引为每个[B,P]实例选一个代表点；argmin是在归一化点距原点上求，不是每车重新求最近点。
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        # 地图类别logits或在筛选函数内转为sigmoid分数，最后类别轴3
        # sigmoid逐元素映射到(0,1)，shape不变；类别分数各自独立，不保证沿类别/模式之和为1。
        map_score = map_score.sigmoid()
        # 沿类别轴取每实例最高sigmoid分数[B,P]，不是沿20个点取最大，更不是GT置信度。
        map_max_score = map_score.max(dim=-1)[0]
        # 按地图预测最大类别置信度构成的布尔筛选[B,P]
        map_idx = map_max_score > map_thresh
        # 初始化batch内最大有效地图数量的计数器；随后扫描每个样本更新，用于padding长度。
        batch_max_pnum = 0
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for i in range(map_score.shape[0]):
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            pnum = map_idx[i].sum()
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if pnum > batch_max_pnum:
                # batch内置信筛选后的最大地图数，用于补齐；不是固定100，也不是真实地图数M
                batch_max_pnum = pnum

        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for i in range(map_score.shape[0]):
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            dim = map_query.shape[-1]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            valid_pnum = map_idx[i].sum()
            # 当前样本置信筛选后的地图实例内容特征，保留原始实例顺序
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            valid_map_query = map_query[i, map_idx[i]]
            # 与筛选实例对应的代表点坐标，和valid_map_query按行对齐
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            # 当前样本补齐位置标记，前面真实项False，后面补齐项True
            # repeat按各轴次数复制数值；复制出的场景特征不是新建独立可学习参数。
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if pad_pnum != 0:
                # 当前样本置信筛选后的地图实例内容特征，保留原始实例顺序
                # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
                # 与筛选实例对应的代表点坐标，和valid_map_query按行对齐
                # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                padding_mask[valid_pnum:] = True
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            selected_map_query.append(valid_map_query)
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            selected_map_pos.append(valid_map_pos)
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            selected_padding_mask.append(padding_mask)

        # stack在指定位置新增一轴，要求各Tensor形状一致；默认dim=0，常用于堆叠解码层或batch项。
        selected_map_query = torch.stack(selected_map_query, dim=0)
        # stack在指定位置新增一轴，要求各Tensor形状一致；默认dim=0，常用于堆叠解码层或batch项。
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        # 筛选/补齐后的key mask，True忽略，False允许读取；不是GT未来时间mask
        # stack在指定位置新增一轴，要求各Tensor形状一致；默认dim=0，常用于堆叠解码层或batch项。
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        num_agent = motion_pos.shape[1]
        # [B,P1,D]->[B,1,P1,D]->[B,A*K,P1,D]，每模式复制同批地图内容，不创建新训练参数。
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        # [B,P1,2]->[B,A*K,P1,2]，位置随后减各模式所属目标坐标，形状保持。
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        # 筛选/补齐后的key mask，True忽略，False允许读取；不是GT未来时间mask
        # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        # [B,A*K,P1,2]-[B,A*K,1,2]得到相对归一化坐标；减法仅改位置，不改地图内容特征。
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if pe_normalization:
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        # 对每个运动查询与地图代表点的归一化相对位移求L2，得到[B,A*K,P1]以进行距离mask。
        map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        # 把补齐项与距离过远项的忽略条件合并；布尔True表示该map key不可被注意力读取。
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        # [B,A*K,P1,D]->[B*A*K,P1,D]；随后注意力把每个运动query作为独立batch项读取地图。
        selected_map_query = selected_map_query.flatten(0, 1)
        # [B,A*K,P1,2]->[B*A*K,P1,2]，顺序与selected_map_query同样展平。
        selected_map_pos = selected_map_pos.flatten(0, 1)
        # [B,A*K,P1]->[B*A*K,P1]，与地图key维P1配对，True会被注意力忽略。
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        num_batch = selected_padding_mask.shape[0]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        feat_dim = selected_map_query.shape[-1]
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if use_fix_pad:
            # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            # 额外虚拟键故意设False，使全部真实地图被mask时仍有一个可读取零特征槽，缓解全mask数值问题。
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            # 筛选/补齐后的key mask，True忽略，False允许读取；不是GT未来时间mask
            # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return selected_map_query, selected_map_pos, selected_padding_mask


    # 【函数入口】VADHead.select_and_pad_query
    # 供自车读取目标/地图时筛选。query=[B,Q,D]、pos=[B,Q,2]、score=[B,Q,C] logits。
    # 按sigmoid类别最大值筛选；每样本保留数量不同，补齐为batch共同Qmax，mask=True处不读取。
    # use_fix_pad=True额外添加一个未mask的零特征键；当前Tiny query_thresh=0通常保留所有有限logit槽。
    # 不是按GT筛选，也没有改变原检测/地图监督的匹配关系。
    def select_and_pad_query(
        self,
        query,
        query_pos,
        query_score,
        score_thresh=0.5,
        use_fix_pad=True
    ):
        """select_and_pad_query.
        Args:
            query: [B, Q, D].
            query_pos: [B, Q, 2]
            query_score: [B, Q, C].
            score_thresh: confidence threshold for filtering low-confidence query
            use_fix_pad: always pad one query instance for each batch
        Returns:
            selected_query: [B, Q', D]
            selected_query_pos: [B, Q', 2]
            selected_padding_mask: [B, Q']
        """

        # select & pad query for different batch using score_thresh
        # sigmoid逐元素映射到(0,1)，shape不变；类别分数各自独立，不保证沿类别/模式之和为1。
        query_score = query_score.sigmoid()
        # 沿dim做极值归约，返回(极值,索引)；[0]取数值、[1]取索引，是否保留轴由keepdim决定。
        query_score = query_score.max(dim=-1)[0]
        # 预测查询置信筛选mask[B,Q]，不是采样器提供的Dataset索引
        query_idx = query_score > score_thresh
        # 初始化batch内最大有效query数；是控制流计数，不是训练参数。
        batch_max_qnum = 0
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for i in range(query_score.shape[0]):
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            qnum = query_idx[i].sum()
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if qnum > batch_max_qnum:
                # batch内最大保留查询数，用于统一key长度
                batch_max_qnum = qnum

        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for i in range(query_score.shape[0]):
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            dim = query.shape[-1]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            valid_qnum = query_idx[i].sum()
            # 当前样本按预测分数筛选的内容特征[Qvalid,D]，没有使用GT来筛选
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            valid_query = query[i, query_idx[i]]
            # 与筛选内容同索引的位置[Qvalid,2]，保持配对
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            # 当前样本补齐位置标记，前面真实项False，后面补齐项True
            # repeat按各轴次数复制数值；复制出的场景特征不是新建独立可学习参数。
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if pad_qnum != 0:
                # 当前样本按预测分数筛选的内容特征[Qvalid,D]，没有使用GT来筛选
                # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                # 与筛选内容同索引的位置[Qvalid,2]，保持配对
                # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0)
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                padding_mask[valid_qnum:] = True
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            selected_query.append(valid_query)
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            selected_query_pos.append(valid_query_pos)
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            selected_padding_mask.append(padding_mask)

        # stack在指定位置新增一轴，要求各Tensor形状一致；默认dim=0，常用于堆叠解码层或batch项。
        selected_query = torch.stack(selected_query, dim=0)
        # stack在指定位置新增一轴，要求各Tensor形状一致；默认dim=0，常用于堆叠解码层或batch项。
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        # 筛选/补齐后的key mask，True忽略，False允许读取；不是GT未来时间mask
        # stack在指定位置新增一轴，要求各Tensor形状一致；默认dim=0，常用于堆叠解码层或batch项。
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        num_batch = selected_padding_mask.shape[0]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        feat_dim = selected_query.shape[-1]
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if use_fix_pad:
            # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            # 固定虚拟query键不mask；False表示保留，不是无效标签。
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            # 筛选/补齐后的key mask，True忽略，False允许读取；不是GT未来时间mask
            # cat沿指定已有轴连接，其他轴必须相同；dim=-1增加特征维，dim=0常合并batch与查询列表。
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return selected_query, selected_query_pos, selected_padding_mask
