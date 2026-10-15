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
        # 沿倒数第二轴（20点）取max，得到每条线的摘要[1,100,128]。
        # 只在同一个地图实例内部取，不跨100个实例。循环末尾再次max时最后轴已256，输出[1,100,256]。
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
            # 框回归：[1,300,256] -> [1,300,10]编码残差。
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

    # 【B7检测配对：一组数据中的一层输出】预测300槽；GT有G个，本例可为10。
    # 分类[300,10]、预测框[300,10]、GT[G,9]、属性[G,34]，先匹配再生成300槽的监督。
    # pos_inds例如[17,42]表示预测槽；pos_assigned_gt_inds例如[3,0]表示GT行：17->GT3，42->GT0。
    # 框/类别/未来轨迹都沿同一配对复制；未来GT不会配给别的车辆。
    # 这里self.sampler是PseudoSampler提取匹配正负索引，不是DataLoader按索引取图片的Sampler。
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
        # 当前一组GT属性[G,34]：前12列位移，接着6列mask；后面的方向/状态/朝向不都直接用于运动L1。
        # 取slice得到[G,12]和[G,6]，行号仍对应同一GT框。
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts*2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts*2:self.fut_ts*3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        # 300预测与G个GT构造[300,G]成本，Hungarian匹配；进入core/bbox/assigners/hungarian_assigner_3d.py。
        # 分类/框代价决定“谁负责谁”，这次匹配本身detach后在CPU求解，不反传匹配索引。
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pos_inds是预测槽索引，neg_inds是未匹配预测槽。sampling_result.pos_assigned_gt_inds是对应GT行索引。
        # 例如预测17匹配GT3：写labels[17]=gt_labels[3]，轨迹也写traj_targets[17]=gt_fut_trajs[3]。
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        # 为300个预测先全部设背景编号10，正槽再覆盖成真实类别0..9。
        # 背景不会占输出的第11维：当前检测用10类sigmoid focal，背景表示10类都为负。
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        # target按预测槽建立[300,12]，不是直接拿[G,12]和300预测硬相减。
        # 未匹配槽填0仅是占位，权重也0，所以不把背景硬拉成原点轨迹。
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        traj_weights = torch.zeros_like(traj_targets)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        # [G,6] -> [G,6,1] -> [G,6,2] -> [G,12]，让每时间步的x/y共用同一个有效标记。
        # 按同一GT配对复制到300槽，再乘正样本权重，缺标注时间步不会参与位移回归。
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks

        # Extra future timestamp mask for controlling pred horizon
        # 额外的预测时域开关[300,6,2]，valid_fut_ts=6时六步全允许。
        # 它与GT mask不同：GT mask说“数据有没有”，这里说“配置要监督多长”。二者相乘。
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                   dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        # 额外的预测时域开关[300,6,2]，valid_fut_ts=6时六步全允许。
        # 它与GT mask不同：GT mask说“数据有没有”，这里说“配置要监督多长”。二者相乘。
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels, label_weights, bbox_targets, bbox_weights, traj_targets,
            traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )

    # 【B7地图配对】预测100槽、GT有M条，例如15条；每条20点。
    # GT等价排列[M,O,20,2]，O是闭合循环起点/开口正反等排列数量（含实现补齐）。
    # map_assigner给出实例配对和order_index[100,M]；用预测槽+GT行共同查询最佳排列，再取20点作为target。
    # 同一几何换起点/顺序不应被当成完全不同道路，因此不是硬按原始JSON点序做L1。
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
        gt_c = gt_bboxes.shape[-1]
        # 100地图槽与M个真实线匹配，cost会比较每条线的等价点顺序。
        # 返回order_index[100,M]；同一对预测/GT选正向还是反向/哪个循环起点，由这个索引记录。
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            # 两个一一对应的整数数组做高级索引，取每个正预测与对应GT的最佳排列号。
            # 例如预测槽6->GT2且order_index[6,2]=4，下面取gt_shifts_pts[2,4,:,:]作为该槽20个点的target。
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        # 目标按100个预测实例建[100,20,2]；仅匹配正槽填GT和weight=1。
        # pts_targets[pos_inds]与GT行/排列共同索引，点序列归属不会因自注意力交流而改变。
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    # 【B7的batch工具】对每个样本调用_get_target_single。本例B=1，但仍返回各样本列表。
    # G可变所以GT用列表；target按预测槽300构建，后面loss_single再cat合并batch。
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
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # multi_apply相当于对batch各样本调用_get_target_single，再把每种返回项分别收集成列表。
        # 这里batch=1，但统一接口仍返回列表，不能把list长度1又当成一个Tensor轴。
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, traj_targets_list, traj_weights_list,
         gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
         )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                traj_targets_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    # 【B7的地图batch工具】对每个样本调用_map_get_target_single。
    # 保留每预测地图槽对应的20个GT点和权重，不把不同实例的点串成一条长折线。
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
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    # 【B7规划监督】预测[1,3,6,2]，专家GT[1,6,2]，mask[1,6]，cmd[1,3]。
    # 同一GT复制到3分支，用cmd让对应命令分支生效；mask让缺失未来的时间步权重为0。
    # 预测地图/其他目标运动还参与边界、碰撞、方向约束，具体几何在utils/plan_loss.py。
    # 约束是训练loss，不是和环境交互得到强化学习reward；参数更新仍由OptimizerHook执行。
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

        # GT[1,6,2]->[1,3,6,2]复制到三个命令分支，下一行用cmd只让对应分支回归权重非零。
        # 不是因为存在三条不同专家路线；一组记录通常只有实际行驶的一条。
        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        # cmd[1,3,1,1] × mask[1,1,6,1] -> [1,3,6,1]，再repeat坐标轴为[1,3,6,2]。
        # 这样监督条件=对应命令且有效时间。cmd/mask不是可训练参数。
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        # cmd[1,3,1,1] × mask[1,1,6,1] -> [1,3,6,1]，再repeat坐标轴为[1,3,6,2]。
        # 这样监督条件=对应命令且有效时间。cmd/mask不是可训练参数。
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        # 以下三项对cmd==1选出的自车[1,6,2]计算边界/碰撞/方向约束。
        # 地图与其他目标未来来自预测，因此显式几何输出除了自身监督，还能服务规划安全约束。
        loss_plan_bound = self.loss_plan_bound(
            ego_fut_preds[ego_fut_cmd==1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        loss_plan_col = self.loss_plan_col(
            ego_fut_preds[ego_fut_cmd==1],
            agent_preds,
            agent_fut_preds,
            agent_score_preds,
            agent_fut_cls_preds,
            weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
        )

        loss_plan_dir = self.loss_plan_dir(
            ego_fut_preds[ego_fut_cmd==1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
            # 以下三项对cmd==1选出的自车[1,6,2]计算边界/碰撞/方向约束。
            # 地图与其他目标未来来自预测，因此显式几何输出除了自身监督，还能服务规划安全约束。
            loss_plan_bound = torch.nan_to_num(loss_plan_bound)
            loss_plan_col = torch.nan_to_num(loss_plan_col)
            loss_plan_dir = torch.nan_to_num(loss_plan_dir)
        
        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_reg'] = loss_plan_l1
        loss_plan_dict['loss_plan_bound'] = loss_plan_bound
        loss_plan_dict['loss_plan_col'] = loss_plan_col
        loss_plan_dict['loss_plan_dir'] = loss_plan_dir

        return loss_plan_dict
    
    # 【B7一层检测+运动loss】匹配完成后将batch与300槽合并。
    # GT9维框经normalize_bbox转10维编码，包含log尺寸/sin-cos yaw，并非每个分量都缩到0..1。
    # 运动预测[300,6,6,2]选一个模式，回到[300,12]做带时间mask的位移L1；模式分数做focal分类。
    # 未匹配槽的运动回归权重0。只有最后检测层生成一次运动前向，输出层轴repeat不代表三套运动网络。
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
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        # [1,300,10] -> [300,10]，把batch与预测槽合并；labels也按相同顺序cat成[300]。
        # 分类loss按匹配的正类/背景计算，仍保留可微logits。
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # 函数名normalize_bbox在这里是“编码框”，不是把所有字段都缩到0..1。
        # GT9维 -> 10维：中心坐标、log尺寸、yaw sin/cos、速度，与网络输出格式配套。
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # traj regression loss
        # 进入get_best_fut_preds：每个预测槽的6条候选与匹配GT比较累计终点，选一个模式。
        # 返回[300,12]原始逐步位移，用整个有效时域计算L1，不能把选模式叫“只监督终点”。
        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

        neg_inds = (bbox_weights[:, 0] == 0)
        # 进入get_traj_cls_target，给每个预测槽一个最佳模式编号；背景槽有背景标记。
        # 轨迹回归学“怎么走”，模式分类学“哪种走法更符合当前示范”，二者监督用途不同。
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks, neg_inds)

        loss_traj = self.loss_traj(
            best_traj_preds[isnotnan],
            traj_targets[isnotnan],
            traj_weights[isnotnan],
            avg_factor=num_total_pos)

        if self.use_traj_lr_warmup:
            loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
            loss_traj = loss_scale_factor * loss_traj

        # traj classification loss
        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        loss_traj_cls = self.loss_traj_cls(
            traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    # 【B7：六种未来中选监督模式】输入[300,6,6,2]，GT[300,6,2]，mask[300,6]。
    # 先cumsum把位移变相对起点位置；逐步距离乘mask，再直接取最后时间步，argmin选模式。
    # 选模式后取出的仍是原始位移[300,12]，不是累计坐标；终点选模式，整条有效轨迹回归。
    # 边界：末步无效会使最后距离全0，原实现没有自动找最后有效步；此注释保留行为，不偷偷修算法。
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

        # [300,6模式,6时间,2]沿倒数第二轴累加：例如(1,0),(1,0.5)变(1,0),(2,0.5)。
        # 得到相对起点的位置，不是再加一次当前目标框中心；GT也同样累计后比较。
        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        # [300,6模式,6时间] -> [300,6模式]，只保留mask后最后一步距离。
        # 末步无效时会全0，argmin常落首模式；这是代码边界，不是自动找有效末步。
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        # box_idxs与min_mode_idxs按元素配对，各预测取自己选中的模式：[300,6,6,2]->[300,6,2]->[300,12]。
        # 返回的是traj_preds原位移；用于loss时依然与逐步位移GT比较。
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts*2)

        return best_traj_preds

    # 【B7：模式分数该学什么】按同样终点规则得到每预测的模式编号0..5。
    # 负检测槽设编号6（背景）；对于6个sigmoid输出，背景意味着六模式都作为负类，不额外输出第七个logit。
    # 分类标签按整条模式给一个，不对每个未来时间步分别分类；同样存在末步无效的边界。
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

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        # 负检测槽设背景6；六模式sigmoid focal将其展开为全零正类标签。
        # 模式不是固定左右转标签；自车3命令与这里6模式是两套机制。
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels

    # 【B7地图loss】预测点[1,100,20,2]在0..1，GT米制点先normalize到相同范围再算位置L1。
    # 方向项则先把预测恢复米制，用相邻点差向量算方向；20点、interval=1时得到19个方向。
    # 点数若不同会插值到GT点数；当前两者20相同。bbox/IoU某些loss虽保留，权重0则不贡献总loss。
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
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
 
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights

        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan,:4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        # GT点米制[100,20,2]按pc_range映射为归一化点；预测已经是0..1，必须同单位再比较。
        # 例如x=0米映射u=0.5，y=0米映射v=0.5；不是把它们映射到图像像素。
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval,0]
        # 方向项先恢复米制再算相邻点差，避免x跨度30米、y跨度60米造成归一化方向失真。
        # interval=1时20点变19个向量[100,19,2]，再与GT方向比较。
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.map_dir_interval:,:] - \
            denormed_pts_preds[:,:-self.map_dir_interval,:]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:,:] - pts_targets[:,:-self.map_dir_interval,:]

        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan,:,:],
            pts_targets_dir[isnotnan,:,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    # 【B7：outs与GT在这里第一次作为监督配对】由VAD.forward_pts_train在Head.forward返回后调用。
    # 检测GT框取gravity_center，避免Dataset底部中心与预测几何中心错位；地图对象此时读取采样和等价排列。
    # 各检测/地图解码层都可有辅助loss，最后层名无d前缀，早期层用d0/d1等；不是再加载不同图片。
    # 自车未来先squeeze掉格式化轴，再按命令/mask监督预测。返回loss字典给BaseDetector._parse_losses。
    # 不在这里optimizer.step；外部OptimizerHook执行backward、裁剪、AdamW更新。
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
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # 复制地图GT对象，后面读取bbox/采样点/等价排列属性；原输入仍是每样本一个几何对象。
        # GT有M条线，不是预测100条；target构造时才按匹配填到100个预测槽。
        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_traj_preds = preds_dicts['all_traj_preds']
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']
        ego_fut_preds = preds_dicts['ego_fut_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        # Dataset框对象内部z是底部中心；取gravity_center恢复几何中心[G,3]再拼尺寸/朝向/速度。
        # 转到类别Tensor的设备后才与预测编码比较，不能把cpu_only框理解为所有框loss都在CPU上算。
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # 逐解码层调用loss_single，列表/tuple按返回位置解包；本例检测三层各有监督。
        # multi_apply是批量调用工具，不是新建三个数据样本或在这里执行optimizer.step。
        losses_cls, losses_bbox, loss_traj, loss_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_bboxes_ignore_list)
        

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        # 此处把地图几何实际采样为每条20点，本例若M=15则[15,20,2]，并搬到预测设备。
        # 后面v2等属性进一步生成等价顺序[15,O,20,2]，O不是运动模式数6。
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
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
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        map_losses_cls, map_losses_bbox, map_losses_iou, \
            map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_traj'] = loss_traj[-1]
        loss_dict['loss_traj_cls'] = loss_traj_cls[-1]
        # loss from the last decoder layer
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # Planning Loss
        # [1,1,6,2]->[1,6,2]；mask[1,1,1,6]->[1,6]；cmd[1,1,1,3]->[1,3]。
        # 只删格式化多出的轴，保留batch=1。准备传入loss_planning，不是把GT传入预测前向。
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        batch, num_agent = all_traj_preds[-1].shape[:2]
        agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
        agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)
        loss_plan_input = [ego_fut_preds, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
                           map_all_pts_preds[-1], map_all_cls_scores[-1].sigmoid(),
                           all_bbox_preds[-1][..., 0:2], agent_fut_preds,
                           all_cls_scores[-1].sigmoid(), agent_fut_cls_preds.sigmoid()]

        loss_planning_dict = self.loss_planning(*loss_plan_input)
        loss_dict['loss_plan_reg'] = loss_planning_dict['loss_plan_reg']
        loss_dict['loss_plan_bound'] = loss_planning_dict['loss_plan_bound']
        loss_dict['loss_plan_col'] = loss_planning_dict['loss_plan_col']
        loss_dict['loss_plan_dir'] = loss_planning_dict['loss_plan_dir']

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
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

        # 回到A4->A1->BaseDetector.train_step，_parse_losses汇总总loss/log_vars。
        # Runner保存self.outputs后，OptimizerHook才做backward、梯度裁剪、AdamW.step，再开始下个batch。
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
