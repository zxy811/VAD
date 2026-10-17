# -*- coding: utf-8 -*-
# 中文阅读版：跟着“一组数据”看计算，不需要先读完所有框架代码。
# 本例固定 B=1：一个 batch 只有一个训练样本；一个样本不是一张图片。
# Tiny训练队列L=3：较早历史帧、较近历史帧、当前帧，每帧有6个相机，因此共18张图片。
# 三帧先按队列索引0/1/2理解，不保证严格紧邻；场景起点可能复制帧来补齐。
# 输入img=[1,3,6,3,384,640]：batch/队列/相机/颜色通道/高/宽。
# 每个Tensor都要问：属于哪个阶段？轴代表什么？是数据、参数、内容特征还是坐标？
# 本注释的具体数字对应Tiny常见配置：BEV=100x100，D=256，检测槽300，地图槽100，
# 每地图20点，运动6模式，每模式6个未来步，自车3命令分支；改配置后数字随之变化。
# GT个数不固定：用G表示真实目标数，M表示真实地图实例数；举例可为G=10、M=15。
#
# 先接上你熟悉的LSS：
# LSS通常从图像像素出发，预测深度分布，把图像特征沿深度lift，再按空间位置汇聚到BEV。
# 这里的BEVFormer路线从BEV格子出发：每个格子先有一个待更新的256维查询特征，
# 用该格子的3D参考位置和相机标定投影到图像，围绕投影位置读取特征，更新这个格子的内容。
# 两者都使用图像特征/相机几何得到BEV，但本实现没有在此使用LSS式逐像素深度分布lift。
# 查询(query)先理解为“一个待填内容的槽位”：BEV槽对应格子，检测槽对应待发现目标，
# 地图槽对应待发现地图实例/点。初始查询表是跨样本共享的可训练参数，不是GT或随机坐标。
# 模型处理当前图片后，查询变成包含当前场景信息的中间特征；不应再称它为固定参数表。
# 256维位置嵌入、256维内容、二维/三维参考坐标是三个不同东西，下面会在出现处拆开说明。
#
# 阅读时遇到 self.layer(...)：这是调用模块对象，经PyTorch __call__进入它的forward。
# __init__创建层；init_weights初始化/加载权重；forward用当前输入计算；loss由外部Hook反传。
# 常见语法：a[-1]取最后项；a[:,i]选第i个时间/槽轴；None索引增加轴；permute仅交换轴；
# view/reshape重排但元素总数不变；cat拼已有轴；stack新增轴；detach只切断该分支梯度。
# 本文件只新增#注释，不改变原始语句/docstring；原英文有通用模板，维度以本例中文和实际代码为准。
#
# 阅读地图：
# 由VAD.py调用本类forward。先构造BEV/目标/地图查询，再根据only_bev选择：
# 历史帧：only_bev=True -> 【B2】只编码BEV -> 返回VAD.py继续下一历史帧。
# 当前帧：only_bev=False -> 【B3】感知Transformer -> 【B4】检测/地图几何
# -> 【B5】运动查询交互 -> 【B6】自车规划 -> 返回outs -> 【B7】loss匹配监督。
# 这个Head不只是框检测：它串起地图、其他目标未来和自车规划；名字pts不是点云输入证明。
# 不要把三种“3”混淆：图片队列3帧、感知解码器3层、自车3个命令分支互不相同。
# GT与预测的数量不同：300预测目标匹配G个GT，100地图槽匹配M个GT，不能按相同行号直接配。

# 损失阅读路线（本次重写）：先看【L0】loss，再顺着下方函数编号追踪。
#   VAD.forward_pts_train -> 本类forward产生outs -> 本类loss(预测+GT) -> 返回loss_dict。
#   【D1】loss_single -> 【D2】get_targets -> 【D3】_get_target_single -> 返回匹配后的监督。
#   D1再算检测分类/框L1、调用【T1】get_best_fut_preds与【T2】get_traj_cls_target算运动损失。
#   【M1】map_loss_single -> 【M2】map_get_targets -> 【M3】_map_get_target_single -> 地图监督。
#   【P1】loss_planning -> 模仿轨迹L1 + 地图边界/碰撞/方向几何约束。
#   返回后由mmdet的train_step/_parse_losses汇总，Runner的OptimizerHook反传、裁剪、step。
#
# 固定最小例子：B=1，A=300检测槽，V=100地图槽，T=6未来步，K=6目标运动模式，E=3自车命令。
# 用G表示当前样本真实目标数(举例10)，M表示真实地图数(举例15)，与300/100没有相等要求。
# 单层预测：类别[B,A,10]，框[B,A,10]，目标轨迹[B,A,K,12]，模式评分[B,A,K]。
# 单层地图：类别[B,V,3]、辅助框[B,V,4]、节点[B,V,20,2]；自车位移[B,E,T,2]。
# GT传入loss时：每batch一个框对象/地图线对象，目标类别[G]，目标属性[G,34]。
# GT真实未来由数据集生成，不是loss里重新运行模型创造出来；loss只对齐/编码/比较它们。
#
# 三种数量/索引必须分开：
#   预测槽编号，例如17；GT行编号，例如3；类别编号，例如car=0。
#   匹配17->GT3后，labels[17]=gt_labels[3]，bbox_targets[17]=gt_boxes[3]，traj_targets同取GT3。
#   assigned_gt_inds内部正数从1开始；PseudoSampler的pos_assigned_gt_inds已转成从0开始的GT行号。
#   batch列表长度B不是6个相机数；loss_single里的single指一个解码层，不代表只算一个目标。
#
# cost与loss分工：Hungarian代价矩阵负责决定配对，detach后到CPU求解，不对匹配索引求梯度。
# 分类/回归loss在选定配对上计算可微误差；配对不反传不等于预测张量整体detach。
# PseudoSampler仅提取配对正负索引，不是DataLoader的Sampler，也不在这里随机抽图片。
#
# 三个不同的权重：
#   label_weights：哪些槽参加分类；本实现背景也参加分类，通常全1。
#   bbox_weights/traj_weights/pts_weights：哪些槽/字段/未来步参加几何回归，未匹配槽通常为0。
#   loss_weight：配置给整项任务乘的系数，在对应loss模块内部使用，不是上述逐元素mask。
# avg_factor指定聚合除数；sum(weight*error)/正样本数，不一定是对所有元素取普通平均。
# 规划L1未传avg_factor，按其默认mean规约；因此不能把它说成自动除以有效mask个数。
#
# 当前Tiny stage_2的loss系数，仅供对应源码(实际以运行cfg为准)：
#   检测分类0.8，三维框L1 0.1，目标轨迹L1 0.2，轨迹模式Focal 0.2。
#   地图分类0.8，地图辅助框L1 0，地图GIoU 0，地图点L1 0.4，地图方向0.005。
#   规划轨迹L1 1，边界1，碰撞1，方向0.5。
# 权重0的地图框/IoU路径仍可能执行计算，但该项没有训练贡献；不是所有函数都实际启用为非零。
#
# 注意输出层轴：检测/地图有3层不同预测；运动只基于最后检测层运行一次，repeat成3份适配接口。
# D1虽随3层调用而计算3份运动loss，L0最终只保留[-1]那份；早期辅助项只保留检测/地图。
# 同一运动预测配上不同层的检测匹配，算出的运动loss不一定相同，不能误称三次独立运动预测。
#
# 英文docstring有旧二维DETR模板：检测GT实际带速度为[G,9]，编码后[G,10]，不是[G,4]。
# normalize_bbox只做框编码(log尺寸、sin/cos角度)，不会把所有分量都缩到0..1。
# 地图normalize_2d_pts/bbox才按pc_range将米制几何映射到0..1；两种normalize含义不同。
# 本次仅改#注释与空白，原语句、原docstring、保留的边界行为均不改变。
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

class MLP(nn.Module):
    # 【构造点特征小网络】创建Linear/LayerNorm/ReLU。Linear把最后维in变成hidden，W=[hidden,in]。
    # 同一层W作用于所有batch/地图实例/点，不是每个点单独建一层。此时还未处理具体样本。
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    # 【LaneNet内部一步】例如输入[1,100,20,256]，当前Linear(256,128)后[1,100,20,128]。
    # LayerNorm和ReLU只处理特征值，不改前3轴。返回给LaneNet后再做每实例点池化。
    def forward(self, x):
        x = self.mlp(x)
        return x

class LaneNet(nn.Module):
    # 【准备地图实例聚合器】当前调用LaneNet(256,128,3)，建立3层点MLP。
    # 每层输出128维再拼上128维实例摘要，所以各层下一次输入256，最终实例摘要256维。
    # 它不是分割网络，也不是从GT坐标中直接读20个点来代替感知。
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    # 【从20个点特征得到一条线特征】本例输入[1,100,20,256]。
    # 每层：点MLP->[1,100,20,128]；沿20点max->[1,100,128]；复制摘要回20点并concat->[1,100,20,256]。
    # 最后沿20点max得到[1,100,256]，每个地图槽一个内容向量，便于运动/自车读取。
    # 池化只在同一个实例内发生，不把100条地图混成一条；max的[0]取数值，[1]才是点索引。
    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                # 当前层逐点256->128：[1,100,20,256] -> [1,100,20,128]。
                # 同一MLP作用于全部点，尚未把20点聚合；每个点保留自己的局部内容。
                x = layer(x)
                # 沿倒数第二轴（20点）取max，得到每条线的摘要[1,100,128]。
                # 只在同一个地图实例内部取，不跨100个实例。循环末尾再次max时最后轴已256，输出[1,100,256]。
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                # 把每点128维局部信息与复制的128维整线摘要拼在最后轴，得到[1,100,20,256]。
                # 这是concat；若写加法仍是128维，会丢掉“分别保留局部和实例摘要”的结构。
                x = torch.cat([x, x_max], dim=-1)
        # 最后一次max的输入已是[1,100,20,256]，输出[1,100,256]，不是128通道。
        # 按当前实现最后一轮cat=[局部128,重复的全局128]，沿点max后两半均为最后的全局摘要。
        x_max = torch.max(x, -2)[0]
        return x_max


# 登记类名供配置type查找；导入本文件时登记，不在此创建样本或执行前向。
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
    # 【准备模型组件】先保存配置，再调DETRHead父类构造，父类会动态调用本类_init_layers。
    # 下方参数默认值不等于Tiny最终值：配置覆盖为BEV100x100、100地图槽、每槽20点、D256。
    # loss/阈值/坐标范围在这里配置；匹配器决定预测槽对应哪个GT，真实几何在loss才进入。
    # code_weights虽然用Parameter包装但requires_grad=False，不训练；普通查询表则可训练。
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

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.tot_epoch = tot_epoch
        self.use_traj_lr_warmup = use_traj_lr_warmup
        self.motion_decoder = motion_decoder
        self.motion_map_decoder = motion_map_decoder
        self.use_pe = use_pe
        self.motion_det_score = motion_det_score
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.pe_normalization = pe_normalization
        self.ego_his_encoder = ego_his_encoder
        self.ego_fut_mode = ego_fut_mode
        self.ego_agent_decoder = ego_agent_decoder
        self.ego_map_decoder = ego_map_decoder
        self.query_thresh = query_thresh
        self.query_use_fix_pad = query_use_fix_pad
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts

        if loss_traj_cls['use_sigmoid'] == True:
            self.traj_num_cls = 1
        else:
          self.traj_num_cls = 2

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.map_bbox_coder = build_bbox_coder(map_bbox_coder)
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        map_num_query = map_num_vec * map_num_pts_per_vec
        self.map_num_query = map_num_query
        self.map_num_classes = map_num_classes
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        self.map_dir_interval = map_dir_interval

        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1

        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get('class_weight', None)
        if map_class_weight is not None and (self.__class__ is VADHead):
            assert isinstance(map_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(map_class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            map_bg_cls_weight = loss_map_cls.get('bg_cls_weight', map_class_weight)
            assert isinstance(map_bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(map_bg_cls_weight)}.'
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({'class_weight': map_class_weight})
            if 'bg_cls_weight' in loss_map_cls:
                loss_map_cls.pop('bg_cls_weight')
            self.map_bg_cls_weight = map_bg_cls_weight
        
        self.traj_bg_cls_weight = 0

        # 进入mmdet DETRHead.__init__：构建Transformer、基本分类loss/匹配器等，并调用本类_init_layers。
        # 从父类返回时，后面forward用的查询/预测层已创建；不是此处就处理当前batch。
        super(VADHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)
        
        # 地图assigner与loss权重在这里检查一致性，防止匹配侧与监督侧配置不配套。
        # train_cfg由MVXTwoStageDetector提取model.train_cfg.pts后放入Head配置。
        # 【损失准备：匹配器与loss对象】下面检查地图cost与loss配置的系数一致，随后构造匹配器。
        # 匹配器算[预测槽,GT]代价来选配对；loss模块在配对后计算可微误差，二者用途不同。
        # 匹配代价weight与训练loss_weight可能数值相同，但不是在这里先计算一次loss再重复加总。
        if kwargs['train_cfg'] is not None:
            assert 'map_assigner' in kwargs['train_cfg'], 'map assigner should be provided '\
                'when train_cfg is set.'
            map_assigner = kwargs['train_cfg']['map_assigner']
            assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_bbox['loss_weight'] == map_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_map_iou['loss_weight'] == map_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be' \
                'exactly the same.'

            self.map_assigner = build_assigner(map_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.map_sampler = build_sampler(sampler_cfg, context=self)
        
        # build_loss根据type查LOSSES注册表并实例化模块；self.loss_xxx(...)会进入该模块forward。
        # self.loss_cls/self.loss_bbox等检测基础模块在父类DETRHead中建立，不是本文件漏掉了定义。
        # 目标轨迹/框/自车回归L1实际类在mmdet/models/losses/smooth_l1_loss.py的L1Loss。
        # 类别/模式Focal实际类在mmdet/models/losses/focal_loss.py；地图点/方向在VAD/utils/CD_loss.py。
        # 规划边界/碰撞/方向类在VAD/utils/plan_loss.py；下方P1在各调用处解释具体公式和筛选。
        # 保存的是可调用loss对象，不是当前样本已经计算好的loss标量；到forward后有了预测才可比较。
        self.loss_traj = build_loss(loss_traj)
        self.loss_traj_cls = build_loss(loss_traj_cls)
        self.loss_map_bbox = build_loss(loss_map_bbox)
        self.loss_map_cls = build_loss(loss_map_cls)
        self.loss_map_iou = build_loss(loss_map_iou)
        self.loss_map_pts = build_loss(loss_map_pts)
        self.loss_map_dir = build_loss(loss_map_dir)
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        self.loss_plan_col = build_loss(loss_plan_col)
        self.loss_plan_dir = build_loss(loss_plan_dir)

    # 【把要用的工具先建好，尚未读取图片】
    # 分类头D->类别数，检测回归D->10编码，地图点回归D->2，运动头512->12，自车头512->36。
    # 共享是指“同一层作用全部槽位”；不同分支不是全部共用一层。with_box_refine=True时各解码层头用deepcopy独立。
    # 最重要的四种初始查询：BEV[10000,256]、检测[300,512]、地图实例[100,512]+点[20,512]、模式[6,256]。
    # 这些是学习的槽位初始值，每个batch都会读取同一参数表；经过图片更新后的内容才代表当前场景。
    # 位置Linear(2,256)只是把二维位置编码成256维；它不产生另一个真实世界位置。
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)


        # 内部工具：deepcopy层N次，各副本有独立权重；重复引用同一module则共享权重。
        # 用于不同检测/地图解码层的预测头，不是复制训练样本。
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        motion_num_pred = (num_motion_decoder_layers + 1) if \
            self.as_two_stage else num_motion_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        # 当前Tiny启用逐层框细化：三层分类/回归头用deepcopy独立参数。
        # else中重复同一个module引用才是跨层共享；二者都对每个查询位置共享该层权重。
        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.traj_branches = nn.ModuleList(
                [traj_branch for _ in range(motion_num_pred)])
            self.traj_cls_branches = nn.ModuleList(
                [traj_cls_branch for _ in range(motion_num_pred)])
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

        if not self.as_two_stage:
            # BEV查询表[10000,256]，每行一个鸟瞰格子的初始内容。
            # 它不是图片特征、不是深度概率、不是二维坐标。训练让这些初始值更适于从图像读取内容。
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            # 目标查询表[300,512]，300个候选目标槽；512将在C2拆为位置256+内容256。
            # 槽位没有预先绑定某辆车，训练时用匈牙利匹配确定谁负责哪个GT。
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            if self.map_query_embed_type == 'all_pts':
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                    self.embed_dims * 2)
            elif self.map_query_embed_type == 'instance_pts':
                self.map_query_embedding = None
                # 100个地图实例初始嵌入[100,512]，每个实例下面再结合20个共享点嵌入[20,512]。
                # 同一实例20点共享实例分量，点序号0..19提供点间差异；不会先从分割图中随机挑第一个点。
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)
        
        if self.motion_decoder is not None:
            self.motion_decoder = build_transformer_layer_sequence(self.motion_decoder)
            # 6种未来模式的初始嵌入[6,256]，所有300目标共用这套模式表。
            # 模式不是预先固定“左转/右转”语义；由多模态轨迹与分类监督逐渐学出不同未来假设。
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)	
            self.motion_mode_query.weight.requires_grad = True
            if self.use_pe:
                # 位置编码Linear(2,256)：W=[256,2]、bias=[256]，所有300目标/6模式共享。
                # 二维坐标进来，得到适合与256维内容相加的位置特征；没有改变真实目标的位置。
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
        else:
            raise NotImplementedError('Not implement yet')

        if self.motion_map_decoder is not None:
            # 地图点特征需要压成每实例一个向量，才能被运动/规划作为100个key读取；具体过程见本文件LaneNet。
            # 不是另一个从相机预测车道的模型，输入已是地图解码器产生的场景点特征。
            self.lane_encoder = LaneNet(256, 128, 3)
            self.motion_map_decoder = build_transformer_layer_sequence(self.motion_map_decoder)
            if self.use_pe:
                self.pos_mlp = nn.Linear(2, self.embed_dims)
        
        if self.ego_his_encoder is not None:
            self.ego_his_encoder = LaneNet(2, self.embed_dims//2, 3)
        else:
            # 当前不启用历史编码器，所以自车初始查询为[1,256]可学习表。
            # 每组数据取一行后扩成[1,1,256]，后续读取目标和地图，变成当前场景的自车特征。
            self.ego_query = nn.Embedding(1, self.embed_dims)	

        if self.ego_agent_decoder is not None:
            self.ego_agent_decoder = build_transformer_layer_sequence(self.ego_agent_decoder)
            if self.use_pe:
                self.ego_agent_pos_mlp = nn.Linear(2, self.embed_dims)

        if self.ego_map_decoder is not None:
            self.ego_map_decoder = build_transformer_layer_sequence(self.ego_map_decoder)
            if self.use_pe:
                self.ego_map_pos_mlp = nn.Linear(2, self.embed_dims)

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims*2 + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims*2
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, self.ego_fut_mode*self.fut_ts*2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        # 运动交互每目标[6,512]，展平3072后融合到256，供一个自车查询读取300个目标。
        # 这是从“多模式内容”到“每目标摘要”，并不是从六条坐标轨迹直接线性外推。
        self.agent_fus_mlp = nn.Sequential(
            nn.Linear(self.fut_mode*2*self.embed_dims, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

    # 【训练开始前初始化】Transformer等用Xavier，模式查询正交初始化，分类bias设低正类先验。
    # 其他未在此覆盖的参数沿用构造默认初始化。初始化不是每个forward再随机一次。
    # 函数无显式return，所以print(init_weights())输出None并不代表模型没有权重。
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_map_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.map_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_traj_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.traj_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.map_reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.map_reg_branches[0][-1].bias.data[2:], 0.)
        if self.motion_decoder is not None:
            for p in self.motion_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(self.motion_mode_query.weight)
            if self.use_pe:
                xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)
        if self.motion_map_decoder is not None:
            for p in self.motion_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.lane_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            if self.use_pe:
                xavier_init(self.pos_mlp, distribution='uniform', bias=0.)
        if self.ego_his_encoder is not None:
            for p in self.ego_his_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_agent_decoder is not None:
            for p in self.ego_agent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_map_decoder is not None:
            for p in self.ego_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # @auto_fp16(apply_to=('mlvl_feats'))
    # 【B1：一帧图像特征来到Head】输入mlvl_feats=[Tensor(1,6,256,12,20)]。
    # 历史/当前都会先准备查询表；only_bev=True会在【B2】提前return，目标/地图查询虽取出但不参与历史解码。
    # only_bev=False才进入【B3】完整感知，随后【B4】几何输出、【B5】运动、【B6】规划。
    # 当前帧GT和真实未来不在本函数参数中；自车历史/状态是否使用由对应配置开关决定。
    # 这里的query不是“先随机采样一个目标坐标”，而是一个等待场景信息填充的特征槽。
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
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        # 【B1.1】从模型参数取目标初始表[300,512]，.to(dtype)仅匹配精度。
        # 当前还没有300个真实目标，也没从图片找到它们；完整感知C2会让槽位读取BEV。
        object_query_embeds = self.query_embedding.weight.to(dtype)
        
        # 地图查询有两种构造；当前Tiny用instance_pts走下面elif。
        # all_pts直接给2000点各自一行参数；instance_pts用实例表+点表构造，更明确表达100组x20点。
        if self.map_query_embed_type == 'all_pts':
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        elif self.map_query_embed_type == 'instance_pts':
            # 点序号表[20,512]加实例轴->[1,20,512]，实例表[100,512]加点轴->[100,1,512]。
            # 这里unsqueeze只增加长度1轴，方便广播，不是在空间中采样新点。
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
            # 广播相加[1,20,512]+[100,1,512] -> [100,20,512]，展平前两轴为[2000,512]。
            # 索引0..19归实例0，20..39归实例1；attention交流不会改这种索引归属。
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

        # 【B1.2】取BEV格子的初始内容[10000,256]，相当于100x100个“等待填入场景信息”的槽。
        # 与LSS不同：LSS先lift图像特征再聚合；这里先准备BEV槽，再从图片里查询内容。
        # 注意这还不是prev_bev：prev_bev是已看过历史图片的内容，而这里是共享初始参数表。
        bev_queries = self.bev_embedding.weight.to(dtype)

        # 给位置编码器的占位mask[1,100,100]，全0表示完整网格。
        # 不是道路分割真值，也不是被遮挡图片区域；下一行得到[1,256,100,100]行列位置嵌入。
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
            
        # 【B2：历史帧在这里提前返回】跳到VAD_transformer.py的C1，只完成相机特征->BEV。
        # 返回[1,10000,256]给A2，A2再把它作为下一历史帧的prev_bev。
        # 当前帧only_bev=False不进入这里，走else的B3；所以不要把历史调用误读为每帧都预测完整轨迹。
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
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
            # 【B3：当前帧完整感知】进入VAD_transformer.py的C2：先C1编码BEV，再检测/地图解码。
            # 输入当前相机特征、历史prev_bev、三类查询；返回tuple后在下面按顺序解包。
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

        # 【从C2返回】BEV=[10000,1,256]；检测hs=[3,300,1,256]；地图map_hs=[3,2000,1,256]。
        # 其余返回值是初始/逐层参考点，不是256维位置嵌入；Head将用它们计算最终框与地图点。
        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs

        # 检测[层3,槽300,batch1,特征256] -> [3,1,300,256]；只是交换轴，便于逐层分类/回归。
        # 与之对应地图[3,2000,1,256] -> [3,1,2000,256]。permute不改变特征含义或数值个数。
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        outputs_trajs = []
        outputs_trajs_classes = []

        map_hs = map_hs.permute(0, 2, 1, 3)
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        # 【B4.1：检测几何输出】三次循环分别处理三个解码层，不是三帧图片。
        # 每层hs[lvl]=[1,300,256]是已从BEV提取当前目标信息的内容，接分类/回归头。
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            # 把当前层使用的参考坐标转成logit，再与该层回归残差相加，sigmoid回0..1。
            # 这是“在上个锚点附近修正”，不是每层独立从零预测。第一层用初始锚点，后面用前层细化锚点。
            reference = inverse_sigmoid(reference)
            # 分类：[1,300,256] -> [1,300,10] logits；每槽对10类评分，未经过sigmoid不应称概率。
            # 同一分类头作用于300个槽，权重共享；三个解码层是否共享由with_box_refine决定。
            outputs_class = self.cls_branches[lvl](hs[lvl])
            # 框回归：[1,300,256] -> [1,300,10]框编码；只有中心xyz与参考点作残差修正。
            # 10维包含x/y、log尺寸、z、yaw sin/cos、速度；GT原始9维在loss里编码到兼容格式。
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            # 保存归一化xy副本[1,300,2]给运动/规划做位置编码，然后detach。
            # 接下来tmp的x/y转米制，但这个列表里仍保留0..1坐标；同名“位置”要区分用途和单位。
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            # 检测中心从归一化映射到局部米制：x=-15+30*u；下面y=-30+60*v，z=-2+4*w。
            # 坐标值与当前LiDAR参考系关联，不是相机像素。其他编码分量不在这里全部sigmoid。
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        # 【B4.2：地图几何输出】每层地图内容[1,2000,256]，每20个点属于一个实例。
        # 三层分别有监督；它不是先输出像素分割，再从分割边界采样地图点。
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            # 把当前层使用的参考坐标转成logit，再与该层回归残差相加，sigmoid回0..1。
            # 这是“在上个锚点附近修正”，不是每层独立从零预测。第一层用初始锚点，后面用前层细化锚点。
            reference = inverse_sigmoid(reference)
            # 先view[1,100,20,256]并沿20点mean，得到每实例摘要[1,100,256]。
            # 分类MLP输出[1,100,3]：分隔线/人行横道/道路边界的logits；这里mean分类与后面LaneNet聚合用途不同。
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs,self.map_num_vec, self.map_num_pts_per_vec,-1).mean(2)
            )
            # 点回归：[1,2000,256] -> [1,2000,2]，与同索引旧参考点logit相加再sigmoid。
            # 每个点由自己的内容预测几何，但共享同一层回归W；“自己的结果”不等于“独有一套参数”。
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid() # cx,cy,w,h
            # 把2000点恢复[1,100,20,2]，并推导100个包围框[1,100,4]；详情在map_transform_box。
            # map_outputs_pts_coord仍为归一化坐标；append进outputs_coords_bev的副本用于后面地图距离筛选。
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_coords_bev.append(map_outputs_pts_coord.clone().detach())
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)
            
        # 【B5：从感知转到运动预测】到这里才有已读过BEV的目标/地图特征与几何。
        # 检测300槽全部可产生未来假设，不代表当前有300个真实动态目标；训练GT匹配与权重会区分。
        if self.motion_decoder is not None:
            batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
            # motion_query
            # 取最后检测层[1,300,256]，换成[300,1,256]。
            # 这是当前场景内容，不是最初的query_embedding表；位置另外来自检测xy预测。
            motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
            # 取六模式初始表[6,256]，为同一目标提供六种不同未来假设的可学习起点。
            # 它在所有300槽间共享；不同目标内容+同一模式表仍得到不同运动查询。
            mode_query = self.motion_mode_query.weight  # [fut_mode, D]
            # [M, B, D], M=A*fut_mode
            # 目标[300,1,1,256]+模式[1,6,1,256] -> [300,6,1,256] -> [1800,1,256]。
            # 先加法再展平；1800是300x6，不是1800个真实车辆，也不是batch里1800组图片。
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
            if self.use_pe:
                # 拿最后层归一化目标xy[1,300,2]。Linear(2,256)后的位置嵌入说明“该槽当前在哪里”。
                # 内容特征负责“场景里是什么”，显式位置给注意力补几何关系；不是再次读取原始BEV。
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
                # [1,300,256] -> [1,300,6,256] -> [1,1800,256]，再permute为[1800,1,256]。
                # 同一目标的6模式共享当前位置；它们未来假设不同，但起点是同一辆车。
                motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]
            else:
                motion_pos = None

            if self.motion_det_score is not None:
                motion_score = outputs_classes[-1]
                max_motion_score = motion_score.max(dim=-1)[0]
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
            else:
                invalid_motion_idx = None

            # 运动自注意力：Q/K为运动内容+位置，V为运动内容；输入输出均[1800,1,256]。
            # 作用是让不同目标/模式交换交通上下文。8头每头32维，再融合回256；不输出坐标。
            # 这里是普通QK点积注意力，与从BEV采样的可变形注意力不同。
            motion_hs = self.motion_decoder(
                query=motion_query,
                key=motion_query,
                value=motion_query,
                query_pos=motion_pos,
                key_pos=motion_pos,
                key_padding_mask=invalid_motion_idx)

            if self.motion_map_decoder is not None:
                # map preprocess
                # 拿最后层归一化目标xy[1,300,2]。Linear(2,256)后的位置嵌入说明“该槽当前在哪里”。
                # 内容特征负责“场景里是什么”，显式位置给注意力补几何关系；不是再次读取原始BEV。
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                # 当前位置[1,300,2]复制到6模式并展平为[1,1800,2]，顺序与motion_query保持一致。
                # 每模式都要独立读地图，但同目标模式共享同一个当前起点。
                motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
                # 地图内容[1,100,20,256] -> LaneNet -> [1,100,256]，每地图实例一个内容向量。
                # 为什么聚合？后面运动/自车查询需要读取“哪条线”的信息，而不是把20点当20条不同道路。
                map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
                map_score = map_outputs_classes[-1]
                map_pos = map_outputs_coords_bev[-1]
                # 按预测地图类别置信度与距离筛选、补齐。若保留p条，则输出[1800,p+1,256]与位置/mask。
                # 为什么第一轴1800？把每个运动查询暂当独立batch，让它有自己的相对地图位置和距离mask。
                # +1是虚拟零键防止全mask，不是新发现一条道路；距离是归一化坐标，见select_and_pad_pred_map。
                map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(
                    motion_coords, map_query, map_score, map_pos,
                    map_thresh=self.map_thresh, dis_thresh=self.dis_thresh,
                    pe_normalization=self.pe_normalization, use_fix_pad=True)
                map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]
                # [1800,1,256] -> [1,1800,256] -> [1800,256] -> [1,1800,256]。
                # 现在轴含义换成[查询长度1,临时batch1800,特征256]；数据相同，但注意力将每个模式单独读取自己的地图。
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

                # position encoding
                if self.use_pe:
                    (num_query, batch) = ca_motion_query.shape[:2] 
                    motion_pos = torch.zeros((num_query, batch, 2), device=motion_hs.device)
                    motion_pos = self.pos_mlp(motion_pos)
                    map_pos = map_pos.permute(1, 0, 2)
                    map_pos = self.pos_mlp(map_pos)
                else:
                    motion_pos, map_pos = None, None
                
                # 运动-地图普通交叉注意力：Q=[1,1800,256]；K/V=[p+1,1800,256]。
                # Q加目标原点位置嵌入，K加相对地图位置嵌入，V是地图内容；输出仍[1,1800,256]。
                # 这一步让未来运动内容知道道路几何/语义，尚未输出六步轨迹坐标。
                ca_motion_query = self.motion_map_decoder(
                    query=ca_motion_query,
                    key=map_query,
                    value=map_query,
                    query_pos=motion_pos,
                    key_pos=map_pos,
                    key_padding_mask=key_padding_mask)
            else:
                # [1800,1,256] -> [1,1800,256] -> [1800,256] -> [1,1800,256]。
                # 现在轴含义换成[查询长度1,临时batch1800,特征256]；数据相同，但注意力将每个模式单独读取自己的地图。
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

            batch_size = outputs_coords_bev[-1].shape[0]
            # 将自注意力结果恢复[1,300,6,256]；下面交叉结果也恢复同shape。
            # unflatten只是恢复目标/模式索引，并没有再运行注意力或更改模式身份。
            motion_hs = motion_hs.permute(1, 0, 2).unflatten(
                dim=1, sizes=(num_agent, self.fut_mode)
            )
            ca_motion_query = ca_motion_query.squeeze(0).unflatten(
                dim=0, sizes=(batch_size, num_agent, self.fut_mode)
            )
            # [1,300,6,256]自注意力内容与同shape地图交互内容，沿最后轴concat为[1,300,6,512]。
            # 为什么保留两份？预测头同时读取交通交互和地图交互信息；加法只有256维，和当前实现不同。
            motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]
        else:
            raise NotImplementedError('Not implement yet')

        # 轨迹MLP最后轴512->12，得到[1,300,6,12]；reshape可看[1,300,6模式,6时间,2位移]。
        # 这是相邻未来位置的逐步位移；cumsum才变相对当前起点的位置，注意训练标签也是位移。
        # 【接到后面的D1/T1】这里只运行一次目标轨迹回归，输入来自最后检测层生成的运动交互特征。
        # 每模式512维->12个逐步位移数，得到[B,300,6模式,12]，并未在此使用真实未来GT。
        # append只放入一项，随后stack成为[1,B,300,6,12]；这个首轴1不是batch，而是只有一组运动输出。
        outputs_traj = self.traj_branches[0](motion_hs)
        outputs_trajs.append(outputs_traj)
        # 模式分类头512->1，得[1,300,6,1]；squeeze(-1)后[1,300,6]。
        # 每条候选轨迹一个logit，不是六时间步各一个分数；sigmoid后六分数不必和为1。
        outputs_traj_class = self.traj_cls_branches[0](motion_hs)
        outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
        (batch, num_agent) = motion_hs.shape[:2]
             
        # 结束分支列表收集：stack默认新增层轴0。
        # 地图类别[3,1,100,3]，地图框[3,1,100,4]，地图点[3,1,100,20,2]。
        # 检测类别/框[3,1,300,10]；运动先stack一套为[1,1,300,6,12]，返回outs时repeat到3层。
        map_outputs_classes = torch.stack(map_outputs_classes)
        map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = torch.stack(outputs_trajs)
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)

        # planning
        (batch, num_agent) = motion_hs.shape[:2]
        # 【B6：自车规划】当前配置ego_his_encoder=None，所以走else读取学习的ego_query[1,1,256]。
        # 不是把专家未来输入给规划器；这个查询需要从预测目标/地图特征里读出当前可行驾驶信息。
        if self.ego_his_encoder is not None:
            ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
        else:
            ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
        # Interaction
        ego_query = ego_his_feats
        ego_pos = torch.zeros((batch, 1, 2), device=ego_query.device)
        ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)
        agent_conf = outputs_classes[-1]
        # 每目标六模式的512维内容展平：[1,300,6,512] -> [1,300,3072]。
        # 后面agent_fus_mlp压到[1,300,256]，让自车把每个目标当一个key读取。
        agent_query = motion_hs.reshape(batch, num_agent, -1)
        agent_query = self.agent_fus_mlp(agent_query) # [B, A, fut_mode, 2*D] -> [B, A, D]
        agent_pos = outputs_coords_bev[-1]
        agent_query, agent_pos, agent_mask = self.select_and_pad_query(
            agent_query, agent_pos, agent_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        agent_pos_emb = self.ego_agent_pos_mlp(agent_pos)
        # ego <-> agent interaction
        # Q是自车[1,1,256]，K/V是筛选目标[q,1,256]。Q/K加各自当前位置编码，V为目标内容。
        # 输出[1,1,256]已经包含与目标未来有关的场景信息；读取的是运动交互特征，不只外推几个框数字。
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=agent_pos_emb.permute(1, 0, 2),
            key_padding_mask=agent_mask)

        # ego <-> map interaction
        ego_pos = torch.zeros((batch, 1, 2), device=agent_query.device)
        ego_pos_emb = self.ego_map_pos_mlp(ego_pos)
        map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
        # 地图内容[1,100,20,256] -> LaneNet -> [1,100,256]，每地图实例一个内容向量。
        # 为什么聚合？后面运动/自车查询需要读取“哪条线”的信息，而不是把20点当20条不同道路。
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        map_conf = map_outputs_classes[-1]
        map_pos = map_outputs_coords_bev[-1]
        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]
        map_query, map_pos, map_mask = self.select_and_pad_query(
            map_query, min_map_pos, map_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        map_pos_emb = self.ego_map_pos_mlp(map_pos)
        # Q改为上一步ego_agent_query，K/V来自地图实例；输出[1,1,256]再含道路约束信息。
        # 因此这不是“重新独立做一次规划”，而是在已融合目标的自车特征上继续加地图上下文。
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=map_pos_emb.permute(1, 0, 2),
            key_padding_mask=map_mask)

        # 按配置选择拼接项；当前两个开关都None，走最后一个elif。
        # 拼接ego_agent_query与ego_map_query：[1,1,256]+[1,1,256] -> [1,1,512]。
        # 历史/状态数组虽然作为参数传到本函数，但当前这个分支没有把它们拼进规划特征。
        if self.ego_his_encoder is not None and self.ego_lcf_feat_idx is not None:
            ego_feats = torch.cat(
                [ego_his_feats,
                 ego_map_query.permute(1, 0, 2),
                 ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
                dim=-1
            )  # [B, 1, 2D+2]
        elif self.ego_his_encoder is not None and self.ego_lcf_feat_idx is None:
            ego_feats = torch.cat(
                [ego_his_feats,
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is not None:                
            ego_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2),
                 ego_map_query.permute(1, 0, 2),
                 ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
                dim=-1
            )  # [B, 1, 2D+2]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is None:                
            ego_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2),
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]  

        # Ego prediction
        # 规划MLP [1,1,512]->[1,1,36]，再reshape为[1,3,6,2]。
        # 3是右转/左转/直行命令分支，不是运动的6种模式；训练按cmd监督对应分支，推理按导航命令选择。
        outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], 
                                                      self.ego_fut_mode, self.fut_ts, 2)

        # 【B6结束：返回给A4】这里还是原始预测，不是训练总loss。
        # all_cls_scores/all_bbox_preds=[3,1,300,10]；map_all_pts_preds=[3,1,100,20,2]；
        # all_traj_preds=[3,1,300,6,12]（同一运动结果repeat）；ego_fut_preds=[1,3,6,2]。
        # A4随后将这些预测与GT传入loss的B7。测试则调用get_bboxes解码，没有训练匹配。
        # 【输出与loss接口对齐】检测/地图保留3层不同预测，但目标运动只有一组。
        # 下面运动repeat(检测层数,...)仅把[1,B,300,6,12]复制成[3,B,300,6,12]，模式评分同样复制。
        # 原因：L0的multi_apply逐层调用D1，同时需要分类/框/轨迹/模式四路长度都为3。
        # 它不创建三套运动网络，也不产生三组不同轨迹；L0最终只把[-1]运动loss放入loss_dict。
        # 这是一种复用联合loss_single接口的组织方式，前两份临时运动loss不会自动加进总loss。
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

        return outs

    # 【B4内部：点组成地图实例】输入[1,2000,2]归一化点，view为[1,100,20,2]。
    # 同一实例的20点固定来自同一个槽位，不需要预测后再按距离聚类。
    # 取每槽xmin/ymin/xmax/ymax后转(cx,cy,w,h)，返回[1,100,4]框和[1,100,20,2]点。
    # 框是由点推导的辅助几何，不是又有100个独立地图框查询。
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
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                                self.map_num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    # 【D3：一个样本、一个检测层，为300个预测槽生成监督target】由D2 get_targets调用。
    # 输入cls_score[300,10]、bbox_pred[300,10]、gt_labels[G]、gt_bboxes[G,9]、gt_attr_labels[G,34]。
    # 本方法尚未计算最终Focal/L1标量；先解决300个预测如何对应G个GT，再把GT散射到预测槽上。
    # 输出labels[300]，bbox_targets[300,9]/weights[300,10]，traj_targets/weights[300,12]，
    # 未来mask[300,6]与正负槽索引；回到D1后编码框、计算损失。
    # 英文docstring的二维4维框模板不适用于本三维检测。
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

        # D3.1 num_bboxes=300是预测槽数；GT个数从gt_attr_labels行数取，可能G=10。
        # 变量名bbox不代表已确认存在目标，未匹配槽也占一行。
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        # 属性前12列=6步*(dx,dy)逐步位移，得到[G,12]；接着6列是未来有效mask[G,6]。
        # 后面的目标方向/状态/yaw等列，不在这里直接作为这项位移L1的target。
        # gt_bbox_c=9(带速度)，gt_traj_c=12；这两个c是字段数，不是GT目标数。
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts*2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts*2:self.fut_ts*3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        # D3.2 进入core/bbox/assigners/hungarian_assigner_3d.py的HungarianAssigner3D.assign。
        # 分类代价[300,G] + 前8维框编码L1代价[300,G] -> 总代价[300,G] -> 一对一匹配。
        # 这里匹配不使用未来轨迹，不偷看未来来决定当前检测框负责哪个目标；速度末2维也未进该框匹配代价。
        # 当前分类/回归cost系数为0.8/0.1；cost用于选择配对，不是直接加入最终loss_dict的误差项。
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        # PseudoSampler不随机挑正负样本：读取assign_result，把已匹配预测列为正槽，其余列为负槽。
        # 没有从当前场景重新读图，sampler不是训练DataLoader的索引采样器。
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # D3.3 pos_inds/neg_inds都是预测槽的行号，例如正槽[17,42]、其他槽为负。
        # pos_assigned_gt_inds是这些正槽对应的GT行号，例如[3,0]，表示预测17->GT3、42->GT0。
        # 以后类别/框/轨迹/时间mask都必须按同一组配对取，不能把预测17直接拿去减GT17。
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        # D3.4 先给全部300槽设背景标签10，再只覆盖正槽的真实类别0..9。
        # 背景10是监督标签的约定，不代表分类头输出第11个数；10维sigmoid Focal会把背景展开为全0标签。
        # 例如17->GT3且GT3是car：labels[17]=0；未匹配槽labels仍为10。
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # 分类权重[300]全1，所以负槽也要学习“所有目标类别都不应高分”。
        # 不要把回归权重为0误理解成背景完全不参与训练。
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # D3.5 建立按预测槽排列的GT框target[300,9]，未匹配行先填0占位。
        # bbox_weights[300,10]按预测编码字段建，全0后仅正槽设1。
        # 一个GT框9维，而编码yaw拆成sin/cos后10维，所以target与weights此刻末维不同是有意的。
        # 真正正槽框target在下面bbox_targets[pos_inds]统一填入。
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        # D3.6 轨迹target[300,12]也按预测槽组织；weights同形状。
        # 17->GT3时，traj_targets[17]=GT3的12个未来位移，正槽初始权重1，背景权重0。
        # 背景target中的0只是占位，不是要求背景预测一条静止轨迹。
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        traj_weights = torch.zeros_like(traj_targets)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        # D3.7 把GT未来有效mask[G,6]复制到xy两个分量，得到[G,6,2]->[G,12]。
        # 再按预测-GT配对填入traj_masks[300,12]；背景无配对，mask保持0。
        # 例如[1,1,1,1,0,0]展开为前8个分量1、后4个分量0：只监督前4步的x/y。
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        # 轨迹回归权重=正槽标记*该槽GT的有效时间标记。
        # mask表示记录是否存在，不是有没有运动：真实停车位移为0且有记录时，mask仍为1。
        # 缺失尾部填充的0不可当作真实停车GT，因此权重为0。
        traj_weights = traj_weights * traj_masks

        # Extra future timestamp mask for controlling pred horizon
        # D3.8 再应用“配置允许监督几步”的开关：[300,6,2]，前valid_fut_ts步设1。
        # 当前valid_fut_ts=6，所以这层开关不缩短监督；它不同于数据缺失mask。
        # 最后traj_weights=正槽*GT有效时间*配置时域，不等于所有预测步都无条件参与。
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                   dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask

        # DETR
        # 按同一匹配取真实框[G,9]并散射到正预测槽，形成D1所需的[300,9]GT。
        # 到这里仍没有反向传播：这里只构造“每个预测应该学什么”和“哪些元素要算误差”。
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        # 按顺序返回给D2：类别及权重、框及权重、轨迹及权重、时间mask、正槽、负槽。
        # traj_masks[300,12]->[300,6,2]再取[...,0]恢复每步一份mask[300,6]；xy共用同一标记。
        # 注意返回的时间mask不额外包含valid_fut_ts配置开关，该开关已经乘进traj_weights。
        return (
            labels, label_weights, bbox_targets, bbox_weights, traj_targets,
            traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )

    # 【M3：一个样本、一个地图解码层，为100个地图槽生成监督】由M2调用。
    # 输入分类[100,3]，辅助框[100,4]，归一化节点[100,20,2]；GT类别[M]、米制xyxy框[M,4]。
    # gt_shifts_pts=[M,O,20,2]是同一GT几何的O种合法节点排列；O不是目标运动6模式。
    # 输出100槽的标签/辅助框/点target及各自权重，供M1计算可微损失。
    # 地图GT实例M不必等于100；通过配对确定哪些预测槽负责这些真实几何。
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
        # num_bboxes=100地图候选数；gt_c=4是GT辅助框的xmin/ymin/xmax/ymax，均为米制。
        # 此处不是前面检测的10维三维框；同名bbox在不同任务中含义不同。
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # M3.1 进入core/bbox/assigners/map_hungarian_assigner_3d.py的assign。
        # 先算每个预测与每个GT每种排列的节点代价[100,M,O]，沿O取最小得到节点代价和order_index[100,M]。
        # 再加分类/辅助框/IoU代价，经Hungarian选择预测-GT配对；Tiny框/IoU代价系数0，主要是分类+点几何。
        # order_index[i,j]记录“预测i对GTj的哪种节点顺序最合适”，它不是匹配到的GT编号。
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        # M3.2 PseudoSampler提取正/负地图槽及对应GT行号，不重新生成地图。
        # 例如预测槽7->GT2，pos_inds包含7，pos_assigned_gt_inds对应项是2。
        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        # 地图类别0/1/2对应divider/ped_crossing/boundary；背景标签用3。
        # 当前输出只有3个sigmoid logits，背景3对应三类全为负，并非额外第4类输出。
        # label_weights全1，未匹配地图槽也参加分类学习。
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        # M3.3 辅助框target[100,4]先0占位，weights同形状，正槽权重1、负槽0。
        # 即使当前地图框loss_weight=0仍保留这条接口，点/方向监督使用另建的pts权重。
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        # M3.4 选节点排列编号。正常有GT且使用当前OrderedPtsL1Cost时走else。
        # 没有order_index的旧兼容分支直接取匹配GT类别当排列编号；按原实现保留，不把它解释成几何最优顺序。
        # 正常else对“预测槽索引、GT行索引”逐项配对取order_index，得到每个正槽应使用的排列。
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        # M3.5 按100预测槽建立点target[100,20,2]，仍用米制GT坐标；点weights同形状。
        # 正槽20点xy全部权重1，负槽0；这里只初始化容器，还未执行坐标归一化。
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        # 同时用GT行和已选排列编号索引gt_shifts_pts[M,O,20,2]。
        # 例如预测7->GT2且最佳排列5：pts_targets[7]=gt_shifts_pts[2,5,:,:]。
        # 取的是这条GT地图真实节点的一种等价顺序，不能把预测第7条硬配GT第7条。
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    # 【D2：把一个检测层的batch拆成逐样本监督，再收集回来】由D1 loss_single调用。
    # cls_scores_list长度B，每项[300,10]；GT列表同长，每项的真实目标数G可以不同。
    # multi_apply对每个样本调用D3，不是对每个相机调用，也不创建新的GT。
    # 返回各类target的batch列表，以及本GPU这一batch的正/负预测槽总数。
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
        # 当前实现只支持无忽略框列表；这是GT忽略接口，不是未来时间mask或注意力padding mask。
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        # None改成长度B的[None,...]，使下面multi_apply每个参数都能逐样本取一项。
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # D2.1 multi_apply等价于：对b=0..B-1运行_get_target_single(第b份预测,第b份GT)。
        # 再按返回项分组：所有样本的labels放labels_list，所有bbox_targets放bbox_targets_list等。
        # 括号多变量赋值按位置解包；不是把9个输出拼成一个张量。B=1时每个列表仍只有一项。
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, traj_targets_list, traj_weights_list,
         gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
         )
        # D2.2 numel()统计索引里有几个槽，跨batch求和得到正槽/负槽数。
        # 若B=1、300预测与10个GT正常一对一匹配，则正槽10、负槽290；不是把300全当正样本。
        # 这些计数供D1归一化损失，真正的权重数组仍单独返回。
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                traj_targets_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    # 【M2：一个地图层，逐batch调用M3生成target】输入是长度B的预测与GT列表。
    # 每样本预测固定100个候选，GT数量M可变；不能把列表长度当100或相机数6。
    # 输出标签/框/节点target等列表与正负地图槽计数，供M1展平后计算损失。
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
        # 当前地图GT忽略列表也必须None；后面变成[None]*B是为了逐样本调用接口。
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # M2.1 对每个样本调用_map_get_target_single，并按返回位置收集8类结果。
        # 例如pts_targets_list=[样本0的[100,20,2],样本1的[100,20,2],...]，不是GT原始[M,20,2]列表。
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        # M2.2 统计本batch匹配上的地图槽与背景槽；若M=15正常配对，则100候选中15正、85负。
        # 计数用于损失分母；是否参与几何回归仍由每槽weights决定。
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    # 【P1：自车规划损失，不是再次预测自车轨迹】由L0组装参数调用。
    # 自车预测ego_fut_preds[B,3命令,6步,2]，真实未来ego_fut_gt[B,6,2]，mask[B,6]，cmd[B,3]。
    # 其余输入来自本次预测：地图节点[B,100,20,2]归一化，地图分数[B,100,3]已sigmoid；
    # 目标中心[B,300,2]米制，目标候选位移[B,300,6模式,6步,2]、类别分数[B,300,10]、模式分数[B,300,6]。
    # 只有一条实际发生的自车GT轨迹；3个命令分支不是3条专家GT。命令索引按转换脚本为右/左/直行。
    # 4项：P1.2位移模仿L1、P1.3道路边界距离、P1.4预测车辆接近惩罚、P1.5地图方向约束。
    # 后三项的几何参照是预测的地图/目标，不是又从GT地图中直接读取答案；各自仍受时间mask。
    # 公式、筛选条件见下方调用前注释，实际函数均在VAD/utils/plan_loss.py。
    # 这些是软损失，不是已经保证满足交通规则的硬约束，也没有在此单列红灯状态/停车规则loss。
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

        # P1.1 同一条GT位移[B,6,2]加命令轴并复制成[B,3,6,2]，只为与预测形状对齐。
        # 复制不意味着真实存在3条未来路线；下一行cmd会选择真正需要监督的那一条分支。
        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        # 命令[B,3]->[B,3,1,1]，时间mask[B,6]->[B,1,6,1]，相乘->[B,3,6,1]。
        # 例cmd=[0,1,0]且mask=[1,1,1,1,0,0]：仅左转分支前4步的回归权重1，其他为0。
        # mask由数据记录可用性生成，不由模型学习；真实停车且有记录时零位移的mask仍为1。
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        # 将同一分支/时间权重复制给x和y：[B,3,6,1]->[B,3,6,2]。
        # 这次repeat扩的是坐标轴，不是前面运动输出对齐3个检测层的repeat，含义不能混淆。
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        # P1.2 自车模仿L1：逐元素abs(预测逐步位移-GT逐步位移)，乘cmd*时间mask，再由模块规约。
        # 这里未传avg_factor，mmdet L1Loss默认对完整[B,3,6,2]误差张量取mean，再乘配置系数1。
        # 因此分母不是自动统计有效步/有效命令数；无效元素先置零但仍在默认mean的总元素数内。
        # GT只有前4步有效时后2步不贡献这项直接GT误差；完整6步记录会监督全部6步。
        # 缺失尾部填充的零与真实停车零位移必须用mask区分，不能人为惩罚“预测了固定6步”。
        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        # P1.3 道路边界软约束：ego_fut_preds[cmd==1]选每样本实际命令分支->[B,6,2]。
        # 进入PlanMapBoundLoss.forward -> plan_map_bound_loss；按boundary类(索引2)分数>=0.5保留地图，
        # 将其节点从0..1还原米制，自车位移cumsum成位置后找最近边界节点距离d。
        # 基本距离项为max(dis_thresh-d,0)，当前阈值1m；先得到[B,6]再乘时间mask规约。
        # 源码另检查自车段与最近边界折线相交，并把相交时刻及之后的该项设0；不能说它是无限大的越界硬惩罚。
        # 默认perception_detach=False，几何分支没有统一detach；硬阈值筛选本身不对选择索引求梯度。
        loss_plan_bound = self.loss_plan_bound(
            ego_fut_preds[ego_fut_cmd==1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        # P1.4 进入PlanCollisionLoss.forward -> plan_col_loss，选命令分支自车[B,6,2]。
        # 外部类按检测置信度>=0.5且最高类别索引0..4筛车辆，并按预测模式评分argmax选每车1条未来轨迹。
        # 注意训练轨迹L1用GT误差选候选(T1)，这里规划碰撞用预测评分选候选，两种规则不同。
        # 自车位移累计为位置；目标位移累计后加目标当前米制中心，才能在同一局部坐标系比较。
        # 距离大于3m的候选被置远；每步分别找最近x/y距离，计算max(1.5-dx,0)、max(3.0-dy,0)。
        # x/y最小值可能来自不同目标，所以这不是精确三维框碰撞IoU，而是代码实现的轴向距离软约束。
        # 返回逐步xy误差[B,6,2]再乘展开的时间mask规约，整项系数1。
        # 外部类会对传入agent_fut_preds原地标记无效预测为1e6，本处不额外clone，保留原实现行为。
        loss_plan_col = self.loss_plan_col(
            ego_fut_preds[ego_fut_cmd==1],
            agent_preds,
            agent_fut_preds,
            agent_score_preds,
            agent_fut_cls_preds,
            weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
        )

        # P1.5 进入PlanMapDirectionLoss.forward -> plan_map_dir_loss；按divider类(索引0)置信度>=0.5筛地图。
        # 自车位移cumsum后，每步寻找最近地图实例及节点，用相邻点形成地图方向，与自车预测轨迹方向比较。
        # 实际以源码的角度折叠规则得到abs(yaw_diff)，不是这段Head直接做目标GT的方向回归。
        # 附近没有地图(距离>2m)或轨迹被判静止(末点与首预测点距离<1m)时，该项按实现置0。
        # 返回[B,6]后乘有效时间mask、默认mean、再乘整项系数0.5；运动GT真实未来没有传给该项做直接比较。
        # 方向估计等时序运算可能跨步依赖；mask0只说明对应输出误差权重0，不是承诺所有相关张量均无间接梯度。
        loss_plan_dir = self.loss_plan_dir(
            ego_fut_preds[ego_fut_cmd==1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        # P1.6 对4项已算出的标量做nan_to_num数值兜底；不新增交通规则或GT。
        # 4个loss模块已在各自内部应用配置loss_weight，这里不再重复乘一次。
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
            loss_plan_bound = torch.nan_to_num(loss_plan_bound)
            loss_plan_col = torch.nan_to_num(loss_plan_col)
            loss_plan_dir = torch.nan_to_num(loss_plan_dir)
        
        # 返回含4个标量Tensor的字典给L0；字典key包含loss，之后框架会把这些训练项汇总。
        # 这里没有optimizer.step，不会在每调用一项loss后分别更新模型。
        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_reg'] = loss_plan_l1
        loss_plan_dict['loss_plan_bound'] = loss_plan_bound
        loss_plan_dict['loss_plan_col'] = loss_plan_col
        loss_plan_dict['loss_plan_dir'] = loss_plan_dir

        return loss_plan_dict
    
    # 【D1：一个检测解码层的检测+运动损失】由L0的multi_apply逐层调用。
    # 输入cls_scores[B,300,10]、bbox_preds[B,300,10]、traj_preds[B,300,6,12]、模式logits[B,300,6]。
    # 执行顺序：D2/D3匹配与target -> Focal类别 -> 框L1 -> T1选运动候选并算位移L1 -> T2模式Focal。
    # single是单个decoder层：本函数仍处理整个batch与300个预测槽，不是仅计算一个框。
    # 以下用B=1、G=10举例；展平后的N=B*300=300。实际正槽数以匹配为准。
    # 运动输入来自最后检测层生成的一份预测，被repeat到3层；L0只保留最后层匹配下的运动损失。
    # 本函数返回4个可微标量，不执行backward/optimizer.step，原英文二维框说明是旧模板。
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
        # D1.1 num_imgs=cls_scores.size(0)=B，不是相机数6；将每个batch张量拆成列表给D2。
        # cls_scores_list长度B，每项[300,10]；bbox_preds_list同样每项[300,10]。
        # GT列表长度也为B，但每项的GT行数可以不同。
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # D1.2 进入get_targets(D2)->_get_target_single(D3)，完成Hungarian配对与监督填充。
        # 返回target已经按300个预测槽排列，而不再是原始G行GT；这是后面能逐元素计算误差的原因。
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        # 按D2的返回顺序解包：类别/权重、框/权重、轨迹/权重、时间mask、正负槽总数。
        # 变量名targets表示“期望预测成什么”；weights/masks表示“哪些元素需要监督”。
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # D1.3 沿第0轴cat各样本：B份[300] -> [N]，N=B*300。
        # 其余变量同样合并：bbox_targets[N,9]、bbox_weights[N,10]，
        # traj_targets/traj_weights[N,12]、gt_fut_masks[N,6]，顺序必须与预测reshape一致。
        # B=1时数值仍300槽，但仍使用统一batch实现。
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        # D1.4 类别预测[B,300,10]->[N,10]，与labels[N]逐槽对齐。
        # labels0..9是目标类别，10为背景；FocalLoss内部把它转换为sigmoid监督，不要在调用前再sigmoid。
        # 例labels[17]=car时，希望该行car高分、其他类别低分；背景行希望10类均低分。
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        # 分类归一化分母=正槽数 + 背景槽数*bg_cls_weight，最终至少1。
        # 当前sigmoid Focal的bg_cls_weight通常0，因此G=10时分母通常10，而不是300*10。
        # 这个bg_cls_weight出现在分母，不代表背景分类项被屏蔽；label_weights全1，背景仍贡献分类误差。
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        # 分布式训练时reduce_mean同步分母计数，避免各卡目标数不同导致归一化尺度不一致。
        # 单卡仍可使用这套接口；这里只处理计数，不是跨卡平均图像内容或预测坐标。
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        # D1.5 进入mmdet/models/losses/focal_loss.py的FocalLoss.forward。
        # 传入原始logits[N,10]、类别编号[N]、逐槽权重[N]、归一化分母。
        # 核心是带alpha/gamma调制的sigmoid二分类交叉熵：降低易样本权重，强调难样本。
        # 当前gamma2、alpha0.25，整项再乘配置loss_weight0.8；不是QK注意力softmax，也不是GT生成。
        # 返回一个标量Tensor，梯度可回到分类头及其上游内容特征。
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        # D1.6 框/轨迹回归用正槽数作分母：计数转设备上的Tensor，跨卡reduce_mean后夹到>=1。
        # .item()把计数变为普通数值；分母不需要学习，不会因此切断预测本身的梯度。
        # 这不是“自动除以有效未来步数”；运动mask只控制分子哪些元素贡献。
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        # D1.7 框预测[B,300,10]->[N,10]。
        # GT target此时[N,9]，下一行编码到[N,10]才能与预测逐字段比较。
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # 进入core/bbox/util.py的normalize_bbox：
        # [x,y,z,w,l,h,yaw,vx,vy] -> [x,y,logw,logl,z,logh,sin(yaw),cos(yaw),vx,vy]。
        # 名字normalize在这里主要指编码，不是全字段除pc_range；中心仍米制，速度仍为预测/GT对应单位。
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        # 逐行检查10个GT编码字段是否全部有限，返回[N]布尔值；同时排除NaN和Inf。
        # 背景target原本填0，尺寸log(0)为-inf，所以常会被此条件排除；它原来的回归权重也为0。
        # 名称虽叫isnotnan，实际torch.isfinite比只检查NaN更严格。
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        # 正负槽回归权重再乘字段权重code_weights：默认前8项1，速度两项0.2。
        # 例如正槽中心x权重1、vx权重0.2；背景仍全0。这与整项loss_weight0.1是两层不同权重。
        bbox_weights = bbox_weights * self.code_weights
        # D1.8 三维框L1：对有限GT行，计算sum(逐字段权重*abs(预测编码-GT编码))/正槽分母。
        # mmdet/models/losses/smooth_l1_loss.py中的L1Loss再乘loss_weight0.1。
        # 预测与GT采用同样编码，所以比较的是中心/对数尺寸/sin-cos朝向/速度；不是三维框IoU。
        # 当前这个函数没有另外调用self.loss_iou计算检测IoU损失，不能凭构造参数名想象存在该项。
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # traj regression loss
        # D1.9 进入T1 get_best_fut_preds选运动候选：
        # 预测[B,300,6,12]->[N,6模式,6步,2]，GT[N,12]->[N,6步,2]，mask[N,6]。
        # 选择依据是按源码处理后的末时间步位置误差；不是用模式分类分数来选训练回归分支。
        # 返回[N,12]的“被选模式原始逐步位移”，随后仍对所有有效时间步算L1，不只是监督终点。
        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

        # 这里neg_inds是[N]布尔mask，不是D3里的负槽整数索引列表。
        # 通过bbox_weights[:,0]==0找未参与框回归的槽，供T2把它们的轨迹模式标签设为背景6。
        # 当前默认code_weights第0项为1，所以这个测试能反映正/负匹配；不要任意推广到第0项设0的配置。
        neg_inds = (bbox_weights[:, 0] == 0)
        # D1.10 T2按同一末步规则给正槽选最佳模式编号0..5；负槽标为6(背景)。
        # 它是从当前候选轨迹与匹配GT误差导出的训练标签，不是数据集预先写好的“模式0=左转”。
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks, neg_inds)

        # D1.11 目标未来位移L1：预测[N,12]、GT[N,12]、权重[N,12]。
        # 有效贡献=正槽*GT时间有效*valid_fut_ts开关，再乘abs(预测位移-GT位移)，求和除正槽数。
        # 当前整项系数0.2；有记录的静止GT位移0、mask1仍受监督，缺记录的填充值mask0不监督。
        # 只将T1选出的一个候选送入这项回归，未选候选不从这项L1获得同样的直接监督。
        loss_traj = self.loss_traj(
            best_traj_preds[isnotnan],
            traj_targets[isnotnan],
            traj_weights[isnotnan],
            avg_factor=num_total_pos)

        # 可选的轨迹loss系数预热，额外乘随epoch变化的标量；Tiny当前False，不执行。
        # 实现位于VAD/utils/traj_lr_warmup.py；不是学习率优化器step，也不修改GT有效mask。
        if self.use_traj_lr_warmup:
            loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
            loss_traj = loss_scale_factor * loss_traj

        # traj classification loss
        # D1.12 模式logits[B,300,6]->[N,6]，每条模式一个分数，非6时间步分别分类。
        # traj_cls_avg_factor按正槽/背景系数计算；当前traj_bg_cls_weight=0影响分母，不取消负槽分类。
        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        # 模式Focal：输入6个logits、T2生成的最佳模式标签、label_weights[N]、归一化分母。
        # 正槽标签k表示第k种轨迹最符合当前GT；负槽标签6表示六模式都不应是正类。
        # 输出只有6维，没有第7个背景logit；sigmoid后6分数不要求和为1。当前loss_weight0.2。
        loss_traj_cls = self.loss_traj_cls(
            traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
        )

        # D1.13 支持的PyTorch版本下，对已经规约成标量的loss做nan_to_num。
        # 这是数值兜底，不是GT配对策略；不能据此认为所有上游无效数据/模式选择问题都已被纠正。
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        # 返回本检测层4个标量：检测分类、框回归、轨迹回归、模式分类。
        # L0接收三层各一份；最终检测前两项保留三层，运动后两项只保留最后层对应结果。
        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    # 【T1：从每槽6条运动候选中选一条，供位移L1监督】由D1调用。
    # 输入traj_preds[N,6模式,6步,2]，traj_targets[N,6步,2]，gt_fut_masks[N,6步]。
    # 先累计为位置计算候选与GT的末步距离，argmin选择模式；返回的是原位移[N,12]，不是累计位置。
    # 这是硬选择：模式索引不求梯度，但被gather选中的预测数值仍可接收后续L1梯度。
    # 本函数名字best不表示已证明全时域最优；以下严格说明实际末步规则及缺失尾帧边界。
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

        # T1.1 cumsum(dim=-2)沿时间步累计，不沿模式或xy轴。
        # 例位移[(1,0),(1,.5)] -> 相对起点位置[(1,0),(2,.5)]；GT同样累计。
        # 此处无需加目标当前中心，因为预测与匹配GT都比较相对起点位移轨迹。
        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        # T1.2 GT[N,6步,2]加模式轴->[N,1,6步,2]，广播与6模式预测相减。
        # 沿xy轴取L2范数得到[N,6模式,6步]，每个数是对应时间的位置误差，不是预测概率。
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        # GT有效时间mask[N,6步]->[N,1,6步]，同一个目标的6候选共享该mask。
        # 这一步只是把缺失时间的位置误差置零；不对有效时间求平均，也不寻找最后有效步。
        dist = dist * gt_fut_masks[:, None, :]
        # T1.3 关键：只取固定的最后一个时间步，得到[N,6模式]，按第6步距离选模式。
        # 若第6步mask=0且数值有限，六个模式距离都会变0，argmin通常选索引0。
        # 因此部分未来缺失时，本实现不是按“最后有效步”挑最佳模式；注释不暗改这一行为。
        # 之后L1仍会用选中模式与前面有效步GT比较，所以选模式规则和回归时间mask要分开理解。
        dist = dist[..., -1]
        # 原实现尝试将NaN项乘0；但NaN*0仍是NaN，不能把这行解释为可靠的NaN清零。
        # 这里保留原表达式，仅说明其实际边界；本次任务不修改模式选择算法。
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        # T1.4 沿6模式取argmin，得到每个预测槽的模式编号[N]；tolist转为Python索引列表。
        # box_idxs=[0,1,...,N-1]与min_mode_idxs逐项配对，不是所有槽读取同一个模式。
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        # 高级索引逐槽取：槽i读取自己的第min_mode_idxs[i]种模式。
        # [N,6模式,6步,2]->[N,6步,2]->[N,12]；从traj_preds取的是原始逐步位移，不是cum_traj_preds。
        # 例槽17最佳模式2，则该行使用traj_preds[17,2,:,:]。
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts*2)

        return best_traj_preds

    # 【T2：为每槽的6个轨迹模式logits生成分类标签】由D1调用。
    # 输入与T1相同，另有neg_inds[N]布尔背景mask；输出traj_labels[N]整型编号。
    # 正槽标签为0..5的候选模式编号，负槽为6(背景)，不是每步一份类别，也不是自车3命令。
    # 标签按预测轨迹与GT距离动态生成，并非人工预定义左转/右转六类别。
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

        # T2.1 与T1同样将预测/GT的逐步位移累计为相对位置，沿xy求距离得到[N,6模式,6步]。
        # 目标当前位置无须再加一次；这一步只用于选择模式标签。
        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        # T2.2 先乘时间有效mask再取固定最后步；不是取有效步均值。
        # 第6步无效且数值有限时，模式误差并列0，会把正槽模式标签置为argmin默认的0。
        # 因此不能把无效尾部mask0理解为“模式分类监督也自动完全忽略这个目标”。
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        # 同T1的原表达式：NaN乘0不保证变成0，保留源码，不声称已正确清理NaN。
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        # 沿模式轴选最小距离，输出[N]，不是选检测类别，也不是概率最大值。
        # argmin生成整数标签，不反传这个选择过程；分类loss仍会训练traj_cls_branches输出的logits。
        traj_labels = torch.argmin(dist, dim=-1)
        # T2.3 未匹配检测槽的模式标签改成self.fut_mode=6。
        # 对于输出6个sigmoid logits，背景标签6会展开为全零正类目标，而不是增加第7个网络输出。
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels

    # 【M1：一个地图解码层的分类/几何损失】由L0逐地图层调用。
    # 输入类别[B,100,3]，辅助框[B,100,4](归一化cxcywh)，节点[B,100,20,2](归一化xy)。
    # GT列表每样本M个实例：辅助框[M,4]米制xyxy，等价排列节点[M,O,20,2]米制。
    # 流程：M2/M3配对与选GT排列 -> 地图Focal -> 辅助框L1 -> 节点L1 -> 线段方向余弦 -> 辅助框GIoU。
    # 所有几何项只监督匹配正槽；背景仍参与分类。以下N=B*100，本例100。
    # Tiny当前辅助框L1/GIoU整项系数0，分类/点/方向非零；不能把“被调用”都当成“非零训练贡献”。
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
        # M1.1 将本层batch预测按样本拆列表：分类每项[100,3]，框每项[100,4]，点每项[100,20,2]。
        # num_imgs是B，不是相机数。GT列表的每项有可变M行，不能直接与100预测逐行相减。
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        # M1.2 进入map_get_targets(M2)->_map_get_target_single(M3)。
        # 先Hungarian匹配地图实例，再为每个正槽选择该GT的合法节点排列，最后把target排到100槽上。
        # 这一步返回的pts_targets已经采用选定顺序，后面的L1不再对O种顺序重复求最小。
        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        # 按M2顺序解包目标与权重；pts_targets_list每项[100,20,2]米制，weights相同形状。
        # 例如预测7->GT2最佳顺序5，该预测槽监督的是GT2顺序5，而不是原始GT第7条。
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
 
        # M1.3 合并batch到N=B*100：labels[N]、bbox_targets/weights[N,4]、pts_targets/weights[N,20,2]。
        # labels正类0/1/2、背景3；点和框回归权重仅正槽为1，分类权重通常所有槽都为1。
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        # M1.4 分类logits[B,100,3]->[N,3]，最后3是地图类别，不是(x,y,z)。
        # FocalLoss接收原始logits与类别索引，不要先在这里手动softmax或sigmoid。
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        # 分母=正地图槽数+负槽数*map_bg_cls_weight；当前sigmoid Focal的背景分母系数为0。
        # 有M个地图正槽时用其数量作尺度，夹至少1；背景分类仍通过label_weights参与误差。
        # 分布式reduce_mean同步计数，不是在混合不同样本的地图实例。
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        # M1.5 地图类别Focal，当前loss_weight0.8；监督“这个实例是divider/ped_crossing/boundary还是背景”。
        # 每实例一个类别，而不是20个点各自给一个地图类别；返回标量。
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        # M1.6 后面的地图几何项以跨卡平均正槽数作为avg_factor，夹>=1。
        # 它不是节点数20，也不是总候选100，更不是对所有无效/有效节点重新按mask计数。
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        # M1.7 辅助地图框预测[B,100,4]->[N,4]，是由预测20点的min/max计算的归一化cxcywh。
        # 它不是独立三维目标框，也不包含高度z或车辆速度。
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # GT辅助框米制xyxy -> utils/map_utils.py的normalize_2d_bbox -> 归一化cxcywh[N,4]。
        # 这与检测normalize_bbox不同：此处真的根据pc_range平移/缩放坐标和宽高。
        # isnotnan随后逐行检查4个target字段有限；负槽几何仍由零权重屏蔽。
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        # 正负槽mask再乘map_code_weights四个字段权重；当前配置四项均1。
        # 这个逐字段权重不是整项loss_map_bbox的loss_weight，后者Tiny为0。
        bbox_weights = bbox_weights * self.map_code_weights

        # M1.8 辅助框L1=sum(weights*abs(归一化预测框-归一化GT框))/正槽数，再乘整项系数。
        # 当前loss_weight=0，所以计算路径存在但该标量通常为0，不贡献框几何训练梯度。
        # 地图主要由节点L1与方向项监督，不能把这个辅助框当作真正地图折线输出。
        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan,:4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        # M1.9 GT节点[N,20,2]米制 -> 归一化xy，与网络输出节点使用相同坐标单位。
        # 以范围[-15,-30]~[15,30]为例：u=(x+15)/30，v=(y+30)/60。
        # 归一化不改变节点顺序；顺序已经由M3结合order_index选定。
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        # 预测节点[B,100,20,2]->[N,20,2]，匹配target也已是N行，所以可逐点比较。
        # 原始GT可能只有M=15条，但target已经散射成100行，未匹配的行权重0。
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        # 若预测点数与GT采样点数不同，沿节点序列线性插值对齐数量，不改变实例对应关系。
        # 当前预测/GT都是20点，此分支跳过；permute只是为了满足interpolate的[N,通道,长度]接口。
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # M1.10 进入VAD/utils/CD_loss.py的PtsL1Loss.forward -> pts_l1_loss。
        # 逐节点逐xy计算abs(归一化预测-GT)，乘pts_weights，求和除正地图槽数，再乘系数0.4。
        # 该项不是Chamfer距离：当前已选好一一对应的节点顺序，所以直接按同索引L1比较。
        # 若预测某节点相差.01归一化x，在当前30m横向跨度中对应0.3m；loss本身这里比较归一化值。
        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)

        # M1.11 方向比较使用节点之间的线段，而不是每个节点单独的坐标。
        # 间隔k=map_dir_interval=1时，20点组成19段；weights[N,19]取前19点的x权重。
        # xy共享同一正槽标记，因此取第0坐标的权重即可，不是只监督x方向。
        dir_weights = pts_weights[:, :-self.map_dir_interval,0]
        # 将预测节点从0..1还原为米制[N,20,2]，再计算方向。
        # x/y跨度不同(30m/60m)，若直接在归一化坐标计算夹角，会扭曲物理线段方向。
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        # 预测方向向量=节点i+k - 节点i，GT用相同相减方式；k=1时均为[N,19,2]。
        # 这里是线段向量，不是yaw标量，也不是未来运动位移；时间轴不存在于这条地图点序列。
        denormed_pts_preds_dir = denormed_pts_preds[:,self.map_dir_interval:,:] - \
            denormed_pts_preds[:,:-self.map_dir_interval,:]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:,:] - pts_targets[:,:-self.map_dir_interval,:]

        # M1.12 进入CD_loss.py的PtsDirCosLoss -> pts_dir_cos_loss。
        # 每段用CosineEmbeddingLoss(label=+1)鼓励预测向量与GT向量同向，非零向量时误差=1-cos夹角。
        # 乘dir_weights后求和除正地图槽数，再乘0.005；线段长度不作为这项的主要直接比较对象。
        # 它与后面的规划方向约束PlanMapDirectionLoss不同：此处是预测地图对GT地图的监督。
        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan,:,:],
            pts_targets_dir[isnotnan,:,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        # M1.13 将预测辅助框转回米制xyxy，与原GT框坐标一致后计算GIoU。
        # 不要将归一化cxcywh直接与米制xyxy相交比较。
        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        # GIoULoss通常用1-GIoU形式衡量辅助框重叠，正槽权重生效；当前整项系数0。
        # 因此当前地图几何训练的非零项是节点L1与方向，而不是该IoU项。
        # 外部实现会按其规则把框字段权重规约为每框权重，本次不改模块内部算法。
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        # M1.14 将已计算出的5项标量做数值兜底，不改变原始GT几何或匹配顺序。
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        # 按固定顺序返回5个标量：地图分类、辅助框L1、辅助框GIoU、节点L1、节点方向。
        # L0对3个地图层都保留这些项(其中当前系数0的项仍为零)，形成主loss和d0/d1辅助loss。
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    # 【B7/L0：监督总入口，建议从这里开始读，再按函数编号进入子函数】
    # 由VAD.py/forward_pts_train在Head.forward返回outs后调用；GT来自Dataset，不是此处新预测。
    # 调用实参依次对应：目标GT框列表、类别列表、地图GT对象列表、地图类别列表、preds_dicts=outs，
    # 自车未来GT/有效mask/命令、目标属性GT；列表长度均按batch，不按6相机或3历史帧。
    # 流程：L0.1取预测 -> L0.2检测GT中心统一 -> L0.3逐检测层D1 -> L0.4地图GT采样/等价顺序
    # -> L0.5逐地图层M1 -> L0.6主层loss -> L0.7规划P1 -> L0.8辅助层loss -> 返回字典。
    # 本例输出字段：检测[3,B,300,...]、地图[3,B,100,...]、自车[B,3命令,6步,2]。
    # 注意3种“3”：解码层数3、自车命令3、输入历史队列3，不代表同一件事。
    # 原英文docstring只保留了部分二维DETR模板；真实检测是三维框，且地图/运动/规划字段也都存在。
    # @force_fp32是对指定预测输入做精度管理的封装，不负责匹配、生成GT或改变张量轴。
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
        # 当前实现不支持传入忽略框，必须None；这不影响未来GT mask或注意力mask的使用，它们是不同接口。
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # L0.1 复制地图几何对象列表，后面读取其辅助框、固定20点采样与等价排列属性。
        # 这不是按预测100槽补齐GT：GT仍是每样本M条真实几何，先保留对象再提取张量。
        # 变量名map_gt_bboxes_list容易误导：其元素实际含LiDARInstanceLines等地图几何，不只是一个4维框Tensor。
        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        # 取检测类别logits[3,B,300,10]；3是解码层数、10是目标类别数，未sigmoid。
        all_cls_scores = preds_dicts['all_cls_scores']
        # 取检测框编码[3,B,300,10]：cx,cy,logw,logl,cz,logh,sin-yaw,cos-yaw,vx,vy。
        # 中心xyz已米制，其他分量仍编码；不能按英文模板误读为全部0..1的二维(cx,cy,w,h)。
        all_bbox_preds = preds_dicts['all_bbox_preds']
        # 取目标候选位移[3,B,300,6模式,12]，12=6步*xy；首轴是同一份运动预测repeat适配检测层的结果。
        all_traj_preds = preds_dicts['all_traj_preds']
        # 取模式logits[3,B,300,6]，每候选一分数，不是每时间步类别；同样沿首轴复制。
        # 当前sigmoid Focal，6个候选评分不要求softmax和为1。
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        # 编码器候选分类预留字段；当前forward明确给None，不是实际BEV内容bev_embed。
        enc_cls_scores = preds_dicts['enc_cls_scores']
        # 编码器候选框预留字段，当前None；as_two_stage=False不执行后面的enc_loss残留分支。
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        # 地图类别logits[3,B,100,3]，每实例一类别；最后3类为divider/ped_crossing/boundary。
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        # 地图辅助框[3,B,100,4]，为预测20节点min/max推导的归一化cxcywh，不是独立三维检测框。
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        # 地图节点[3,B,100,20,2]，归一化xy；100为预测容量，与每样本真实地图数M不同。
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        # 地图编码器类别预留字段，当前None；不是map_all_cls_scores。
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        # 地图编码器辅助框预留字段，当前None。
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        # 地图编码器点预留字段，当前None，不能拿模板形状想象当前确实运行了该预测头。
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']
        # 自车3个命令分支的逐步位移[B,3,6,2]；这里没有3分支概率或动态轨迹结束标记。
        # 固定输出6步；GT有效mask说明哪些未来记录真实存在，不说明车辆是否应该停止。
        ego_fut_preds = preds_dicts['ego_fut_preds']

        # len(Tensor)取首轴长度，得到检测解码层数3，不是GT数量G。
        # device读取GT类别Tensor所在设备，后续将框/地图GT搬到相应设备后比较。
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        # L0.2 统一GT框中心定义：框对象内部用底部中心z，预测监督用包围框几何中心z。
        # gravity_center[G,3]拼原尺寸/朝向/速度[G,6] -> GT[G,9]，再搬设备。
        # 例底部z=0、高1.6，则几何中心z=.8；不转换会把正确预测.8与底部0错误比较。
        # 这是同一框的表示原点转换，不是换成另一个世界坐标系；之后D1才编码为10维监督。
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        # L0.3 把同一份batch GT列表放入3个层位置，方便逐层对齐；标签/属性/忽略项同样处理。
        # 这几行是重复引用列表，不是加载三组不同图片、三个不同GT场景，也没有将真实目标数变为3G。
        # 每层自己的预测可产生不同Hungarian匹配，但比较的是同一样本的真实标注。
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # 依层调用D1 loss_single：第0/1/2层检测各算一份分类和框loss。
        # 运动输入虽同值repeat，也随D1算三次，且各层匹配可能不同，故运动loss数值不一定相同。
        # 输出4个长度3的列表：losses_cls、losses_bbox、loss_traj、loss_traj_cls；不是这里就将它们求总和。
        # 后面只保留loss_traj[-1]/loss_traj_cls[-1]，不把三份运动loss加三遍。
        losses_cls, losses_bbox, loss_traj, loss_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_bboxes_ignore_list)
        

        # 切换到地图分支，重用局部变量num_dec_layers：现在表示地图decoder层数，当前也为3。
        # 它不改变前面已算好的检测层结果。device改取地图GT类别的设备。
        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        # L0.4 从每个地图GT对象读取辅助xyxy框[M,4]米制，用于匹配与可选框损失。
        # 列表仍长B，不是固定100；例如一个样本M=15，得到[15,4]。
        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        # 读取固定节点采样属性，每条GT几何采样20点，得到每样本[M,20,2]米制坐标。
        # GT来自数据集地图标注，不是模型从map_all_pts_preds复制出来。
        # 这个列表在当前主地图loss中不直接传入：下面实际传的是包含等价顺序的map_gt_shifts_pts_list。
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        # 同一折线/闭合轮廓可有合法的反向/起点顺序，具体允许哪些排列由属性版本实现决定。
        # 例如闭合轮廓ABCD与CDAB形状相同；只按固定节点编号比较会产生人为顺序误差。
        # 准备[M,O,20,2]的GT等价排列；O是顺序候选数量，不是6运动模式，也不是额外GT地图实例。
        # Tiny当前选v2，对应datasets/nuscenes_vad_dataset.py中LiDARInstanceLines属性。
        if self.map_gt_shift_pts_pattern == 'v0':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v1':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v2':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v3':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v4':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]
        else:
            raise NotImplementedError
        # 将同一个batch的地图GT框/类别/点/顺序列表重复到地图层轴，不制造3份不同场景。
        # map_all_gt_pts_list虽建立，当前主调用传的是map_all_gt_shifts_pts_list，配对时再选一种顺序。
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # L0.5 逐地图层调用M1：M2/M3做实例匹配和顺序选择，再计算5项loss。
        # 返回5个长度3列表；map_losses_bbox/IoU当前系数0，其他项根据配置非零。
        # 这是监督重复用于多层，不是把100个预测实例扩为300个独立地图GT。
        map_losses_cls, map_losses_bbox, map_losses_iou, \
            map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        # L0.6 创建最终要交给训练框架的字典；只有真正放入这个字典的loss才从此接口返回。
        # 此前临时算过的值若未加入，不能直接认为都参与最终优化。
        loss_dict = dict()
        # loss from the last decoder layer
        # 保存最后检测层分类与框回归loss，key不加d前缀；[-1]指最后解码层，不是最后一辆车。
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        # 只保留最后检测层匹配条件下的目标轨迹回归loss与模式分类loss。
        # 早两层的临时运动loss未加入字典；当前不是3套运动网络，也不是运动监督乘3。
        loss_dict['loss_traj'] = loss_traj[-1]
        loss_dict['loss_traj_cls'] = loss_traj_cls[-1]
        # loss from the last decoder layer
        # 保存最后地图层5项loss：分类、辅助框L1、辅助框GIoU、节点L1、节点方向。
        # 权重已在模块内部应用，不在此重复乘；即便某项系数0仍保留其日志key。
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # Planning Loss
        # [1,1,6,2]->[1,6,2]；mask[1,1,1,6]->[1,6]；cmd[1,1,1,3]->[1,3]。
        # L0.7 整理自车监督的包装轴：本例GT[1,1,6,2]->[1,6,2]。
        # 只删除指定位置的长度1轴，不改变GT数值，也不删除真实batch轴。
        ego_fut_gt = ego_fut_gt.squeeze(1)
        # mask[1,1,1,6]->[1,6]，每个未来时间步是否有真实记录；不是车辆是否运动、不是可学习概率。
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        # cmd[1,1,1,3]->[1,3]，one-hot决定监督右/左/直行哪个规划分支。
        # 同一真实未来不是同时要求3个分支都拟合；P1用cmd*时间mask控制监督范围。
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        # 取最后检测层对应的运动副本，batch=B、num_agent=300；后面的view不再运行运动网络。
        # 目标位移[B,300,6模式,12]->[B,300,6模式,6步,2]；评分[B,300,6]不变。
        batch, num_agent = all_traj_preds[-1].shape[:2]
        agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
        agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)
        # 按P1函数签名顺序组装10个实参，*loss_plan_input随后逐项展开，不是把全部张量cat。
        # 1自车预测[B,3,6,2]；2真实位移[B,6,2]；3有效mask[B,6]；4命令[B,3]。
        # 5预测地图点[B,100,20,2]归一化；6地图分数[B,100,3]经sigmoid。
        # 7预测目标中心[B,300,2]米制；8目标候选位移[B,300,6模式,6步,2]。
        # 9检测类别分数[B,300,10]、10轨迹模式分数[B,300,6]均在这里sigmoid供几何筛选使用。
        # 地图/目标几何来自模型预测而不是GT；P1模仿L1使用自车GT，另外3项使用预测场景形成软约束。
        loss_plan_input = [ego_fut_preds, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
                           map_all_pts_preds[-1], map_all_cls_scores[-1].sigmoid(),
                           all_bbox_preds[-1][..., 0:2], agent_fut_preds,
                           all_cls_scores[-1].sigmoid(), agent_fut_cls_preds.sigmoid()]

        # 进入P1 loss_planning，返回4项规划标量后逐项放入最终loss_dict。
        # 几何预测可能经这些约束接收梯度；分数的阈值/argmax选择是硬索引，不意味着对选择过程可微。
        loss_planning_dict = self.loss_planning(*loss_plan_input)
        loss_dict['loss_plan_reg'] = loss_planning_dict['loss_plan_reg']
        loss_dict['loss_plan_bound'] = loss_planning_dict['loss_plan_bound']
        loss_dict['loss_plan_col'] = loss_planning_dict['loss_plan_col']
        loss_dict['loss_plan_dir'] = loss_planning_dict['loss_plan_dir']

        # loss from other decoder layers
        num_dec_layer = 0
        # L0.8 把检测前两层分类/框loss作为辅助监督加入字典：d0.loss_cls、d0.loss_bbox、d1.*。
        # zip逐项配对两个列表；[:-1]排除已保存的最后层，不是排除背景目标。
        # 没有添加d0/d1.loss_traj或模式loss，所以前面运动repeat不会导致运动监督自动加三遍。
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
        # 地图前两层同样添加5项辅助loss，key如d0.loss_map_pts、d1.loss_map_dir。
        # 检测/地图都用同一批输入和GT，各层可以有不同匹配；不是又运行两次Dataset或forward。
        # 字段名中的d0/d1是层编号，与自车3个命令分支、历史3帧都无关。
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i in zip(
            map_losses_cls[:-1],
            map_losses_bbox[:-1],
            map_losses_iou[:-1],
            map_losses_pts[:-1],
            map_losses_dir[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        # L0.9 通用DETR编码器proposal监督的残留分支，当前outs里该字段明确None，跳过。
        # as_two_stage是模型内部proposal开关，不是作者stage_1/stage_2训练流程。
        # 这条旧调用未按当前扩展的运动loss_single接口完整传参，不能据此认为只改flag就支持它；保留原码不修改。
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # 地图编码器proposal残留路径同样因None跳过，原作者也保留TODO。
        # 主线地图监督需要等价排列GT，这个分支的旧调用不是当前Tiny实际使用的数据流。
        if map_enc_cls_scores is not None:
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i])
                for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            map_enc_loss_cls, map_enc_loss_bbox, map_enc_loss_iou, \
                 map_enc_loss_pts, map_enc_loss_dir = \
                self.map_loss_single(
                    map_enc_cls_scores, map_enc_bbox_preds,
                    map_enc_pts_preds, map_gt_bboxes_list,
                    map_binary_labels_list, map_gt_pts_list,
                    map_gt_bboxes_ignore
                )
            loss_dict['enc_loss_map_cls'] = map_enc_loss_cls
            loss_dict['enc_loss_map_bbox'] = map_enc_loss_bbox
            loss_dict['enc_loss_map_iou'] = map_enc_loss_iou
            loss_dict['enc_loss_map_pts'] = map_enc_loss_pts
            loss_dict['enc_loss_map_dir'] = map_enc_loss_dir

        # L0.10 返回各项已加配置权重的标量Tensor，而不是预测框/轨迹。
        # 外部VAD.forward_pts_train/forward_train交回train_step，BaseDetector._parse_losses整理日志并汇总名称含loss的项。
        # 随后Runner的OptimizerHook对总loss执行backward，按配置裁剪梯度并optimizer.step，一次迭代统一更新。
        # 本文件各loss函数不自行step；GT/匹配索引不学习，可微预测及生成它们的网络参数接受梯度。
        return loss_dict

    # 【推理输出】检测/地图coder用最后层输出做解码和分数/范围筛选；NMS-free不是常规NMS。
    # decoded检测框z减半高转为底部中心，包装成metadata指定的框类；运动轨迹随选中检测索引取出。
    # 每样本输出框、类别/分数、六模式位移、地图点等；数量是筛选后的预测数，不是GT的G/M。
    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        # 分别进入core/bbox/coders/nms_free_coder.py、map_nms_free_coder.py读取最后层预测。
        # 按分数/topk/范围筛选且反编码；地图点恢复局部米制，检测轨迹随同一个预测索引选取。
        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), \
             'len(preds_dict) should be equal to len(map_preds_dicts)'
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            # 检测解码框的几何中心z减半高，改为LiDARInstance3DBoxes约定的底部中心。
            # 这是框表示原点转换，不是把整场景换到其他坐标系。
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds['map_bboxes']
            map_scores = map_preds['map_scores']
            map_labels = map_preds['map_labels']
            map_pts = map_preds['map_pts']

            ret_list.append([bboxes, scores, labels, trajs, map_bboxes,
                             map_scores, map_labels, map_pts])

        return ret_list

    # 【B5内部：每个运动查询看哪些地图】实际motion_pos=[1,1800,2]（300目标x6模式），不是G个真实目标。
    # 先以每个地图实例中“距归一化原点最近的点”为代表点，不是为每辆车重新选最近点。
    # 按预测最大类别置信度保留p条；本例B=1不需补齐别的样本，但仍执行统一逻辑。
    # 复制给1800查询，再减查询坐标形成相对位置；距离用归一化xy，阈值0.2不是0.2米。
    # 返回[1800,p+1,256]内容、[1800,p+1,2]位置、[1800,p+1]mask；最后+1是未mask的虚拟零特征槽。
    # 这样即使某目标所有真实地图都被mask，也保留一个可读取键；True表示忽略，False表示可读取。
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
        
        if dis_thresh is None:
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        # [1,100,20,2]归一化点对原点求距离->[1,100,20]，argmin沿20点选每实例一个代表点。
        # 实现不是“每辆车到线的真实最短距离”；理解这个选择后再决定是否值得修改。
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        # 原map_score=[1,100,3] logits先sigmoid，再取类别最大值->[1,100]。
        # 筛的是整条地图实例置信度，不是20个采样点的置信度，GT没有参与筛选。
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        # 假设置信筛选保留p条：[1,p,256]->[1,1800,p,256]。
        # 复制内容给每个模式，几何相对位置在下面分别计算，所以同地图对不同车辆的相对关系不同。
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        # 代表点[1,1800,p,2]减各运动起点[1,1800,1,2]，广播得相对位置。
        # 当前是归一化xy，dis_thresh=0.2不等于米；pe_normalization=True时也将相对位置用于key位置编码。
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        # [1,1800,p,256]->[1800,p,256]；坐标/mask也同步展平，保证相同索引是同一个运动模式。
        # 后面append虚拟槽后p变p+1，交叉注意力前permute成[p+1,1800,256]。
        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        # 这里forward显式传True，所以补一条零内容键和未屏蔽mask=False。
        # 它不是GT道路，只是防止某个查询所有key都被屏蔽而产生不稳定注意力。
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask


    # 【B6内部：自车要读取的目标/地图key】输入[1,Q,256]、位置[1,Q,2]、logits[1,Q,C]。
    # 按sigmoid最大类别分数筛选，补齐不同batch长度；输出[1,q,256]及位置/mask。
    # 当前Tiny query_thresh=0通常保留全部有限logit槽；没有按GT类别筛选，不改变训练匹配。
    # use_fix_pad=True才多添未mask零槽；当前配置开关决定，不能把所有pad都说成有效真实地图。
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
        # 目标/地图筛选共用：logits[1,Q,C]->sigmoid->类别max[1,Q]，再与阈值比。
        # 当前规划query_thresh=0时一般保留全部有限预测，但它们并不都是真实目标；后续特征/分数表达可信程度。
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0)
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        return selected_query, selected_query_pos, selected_padding_mask
