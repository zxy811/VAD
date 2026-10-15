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
# 由VAD_head.py进入。历史和当前都需要【C1】get_bev_features；只有完整当前帧才继续【C2】forward的检测/地图解码。
# BEV编码前的数据：相机特征[1,6,256,12,20]；学习的格子查询[10000,256]；历史prev_bev可能为None。
# 编码输出[1,10000,256]，也就是100x100个格子的内容，不是10000个检测目标。
# 具体读取图片/融合历史的注意力在同目录VAD/modules/encoder.py、spatial_cross_attention.py、
# temporal_self_attention.py；这里会说明这些调用的输入输出和它们相对LSS的角色。
# 读完本文件的感知forward后返回VAD_head.py继续运动/规划；不要误以为这里已经输出驾驶控制量。

import torch
import numpy as np
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.utils import ext_loader
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from torchvision.transforms.functional import rotate
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.VAD.modules.decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.VAD.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.VAD.modules.spatial_cross_attention import MSDeformableAttention3D


ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

# 【坐标更新工具】输入0..1坐标r，返回log(r/(1-r))，形状不变。
# 例r=0.5时logit=0；加delta后sigmoid得到新参考点。clamp避免r=0/1出现无穷。
# 不是把256维内容变坐标；256->2/3由其他共享Linear执行。
def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    # 先限制0..1，再用eps避免log(0)；形状完全不变。
    # r=0.5时返回0，若回归delta=0则sigmoid(0)=0.5，解释“零残差保持旧参考点”。
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# 登记类名供配置type查找；导入本文件时登记，不在此创建样本或执行前向。
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapDetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    # 父类按配置创建三层地图解码层；return_intermediate=True保存每层特征/参考点供辅助监督。
    # 创建发生在构造阶段，下面forward才处理本例2000个地图点查询。
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapDetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    # 【C3地图解码】query=[2000,1,256]，reference_points=[1,2000,2]，value是BEV[10000,1,256]。
    # 这时BEV已经从图片生成，任务变成“2000个点槽到BEV中找相关道路内容”。
    # 每层先更新256维内容，再由该层回归头输出2维残差更新参考点；下一层围绕新点读BEV。
    # 普通自注意力允许跨实例点交流；点归属仍由100x20轴索引固定，不会因为attention读别人就换实例。
    # 三层返回[3,2000,1,256]和[3,1,2000,2]；Head随后才整理100条地图几何。
    # 参考点detach仅切断跨层坐标更新路径，不冻结内容特征和回归头的当前层监督。
    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        # C3起点：2000个待更新点槽[2000,1,256]；reference_points是它们的二维锚点[1,2000,2]。
        # 内容256维描述“读到了什么”，参考2维决定“下一次围绕哪里读”，不要把它们看成同一个变量。
        output = query
        intermediate = []
        intermediate_reference_points = []
        # lid=0/1/2代表三次细化，不是三历史帧；每次layer按配置self_attn->cross_attn->FFN更新。
        # 每次都保留2000点槽身份，最后Head才恢复100实例x20点。
        for lid, layer in enumerate(self.layers):

            # [1,2000,2]增加BEV特征尺度轴为[1,2000,1,2]；1是尺度数，BEV只有一层空间特征。
            # 这不是增加一个预测时间步，也不是增加一个注意力头。
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            # 先更新内容[2000,1,256]，不直接用本行把参考点变成最终地图输出。
            # 普通自注意力Q/K=内容+位置、V=内容；可变形交叉注意力以参考点为中心读BEV value。
            # 偏移/权重由256维查询通过共享线性层预测，再分头采样，不是每点独立学习一个线性层。
            # 内容shape始终256；采样位置属于注意力内部临时几何，不等于20个最终地图节点。
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            # [2000,1,256]->[1,2000,256]，便于该层回归头逐点处理最后维。
            # 回归之后再permute回[2000,1,256]供下一层注意力使用。
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                # 当前层点内容[1,2000,256]经共享回归MLP->残差[1,2000,2]。
                # 参考点会变，但256维query_pos嵌入在这条路径里不被原地替换为新坐标；两者用途不同。
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                # r_new=sigmoid(delta+logit(r_old))，shape[1,2000,2]。
                # 如果某点r_old=(0.5,0.5)、delta=(0,0)，它保持原位；有非零delta则移动下一层读取中心。
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                # 下一层使用更新坐标的数值，但切断“跨层参考坐标”这条梯度路径。
                # 不表示该点256维内容不训练；内容链与各层回归loss仍可反传，注意力/回归参数仍更新。
                reference_points = new_reference_points.detach()

            # [2000,1,256]->[1,2000,256]，便于该层回归头逐点处理最后维。
            # 回归之后再permute回[2000,1,256]供下一层注意力使用。
            output = output.permute(1, 0, 2)
            # 当前True：记录本层内容[2000,1,256]与更新参考[1,2000,2]，供三层输出和辅助监督。
            # 如果False，实际返回最后一个内容Tensor与参考点，不自动增加一个长度1的层轴。
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        # 当前True：记录本层内容[2000,1,256]与更新参考[1,2000,2]，供三层输出和辅助监督。
        # 如果False，实际返回最后一个内容Tensor与参考点，不自动增加一个长度1的层轴。
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


# 登记类名供配置type查找；导入本文件时登记，不在此创建样本或执行前向。
@TRANSFORMER.register_module()
class VADPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    # 【准备BEV生成器与两个读取器】encoder负责图像->BEV；decoder负责BEV->目标；map_decoder负责BEV->地图。
    # 三者由配置type创建，位置/相机/尺度参数在init_layers建立。本例D256、六相机、一图像尺度。
    # 地图100槽/20点来自配置，不是这个函数默认50/10就一定执行50/10。
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 map_decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 map_num_vec=50,
                 map_num_pts_per_vec=10,
                 **kwargs):
        super(VADPerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        if decoder is not None:
            self.decoder = build_transformer_layer_sequence(decoder)
        else:
            self.decoder = None
        if map_decoder is not None:
            self.map_decoder = build_transformer_layer_sequence(map_decoder)
        else:
            self.map_decoder = None

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.init_layers()

    # 【准备可学习参数，不处理图片】相机表[6,256]、尺度表[1,256]用于区别特征来源。
    # 检测位置投影Linear(256,3)，地图Linear(256,2)，共享于300/2000个槽，不是每点独立W。
    # 车辆状态MLP 18->128->256，输出用于给每个BEV格子加入同一帧的运动上下文。
    # 这里不是LSS的深度预测头，也不输出每个像素的深度概率。
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        # 尺度表[1,256]与相机表[6,256]是可学习参数。图像展平后靠它们区分“来自哪个相机/尺度”。
        # 它们不是相机内参/外参；真实投影由metadata中的lidar2img负责。
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        # 共享Linear：检测256维位置嵌入->3维参考坐标logit；地图256->2。
        # W分别[3,256]、[2,256]，所有检测槽/地图点共用。sigmoid后才是0..1空间锚点。
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.map_reference_points = nn.Linear(self.embed_dims, 2)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    # 【训练前一次初始化】Xavier初始化矩阵，normal初始化相机/尺度表；注意力有自己的offset初始化。
    # 框架调用model.init_weights时执行；每个新样本不会重置权重。Linear/Embedding参数跨样本共享并训练。
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.map_reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    # 【C1：最关键的图像->BEV阶段；历史和当前都会进来】
    # 本例输入：mlvl_feats=[Tensor(1,6,256,12,20)]；bev_queries=[10000,256]；bev_pos=[1,256,100,100]。
    # prev_bev=None表示没有更早记忆；否则是上一时刻[1,10000,256]（本函数也接受转置布局）。
    # 第1步：按车辆运动旋转/平移对齐历史BEV；第2步：将车辆状态加入当前格子查询；
    # 第3步：相机特征展平并标记相机/尺度；第4步：encoder逐格从历史BEV与当前图片读取内容。
    # 与LSS的联系在第4步：这里BEV格子先有参考空间位置，再反投影到图片读取；不是像素预测深度后往BEV撒点。
    # encoder里的point_sampling用lidar2img将10000格子的高度参考点投到六相机，mask屏蔽看不见位置。
    # 高度参考点数4是柱内几何锚点数，不是逐像素深度概率分布；空间注意力采样数与解码器采样数也不同。
    # 这里不展开外部CUDA，但返回形状确定为[1,10000,256]，格子内容已经包含本帧图像信息。
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        # 【C1.1：准备格子】mlvl_feats[0]取唯一图像尺度[1,6,256,12,20]，所以bs=1。
        # bev_queries初始[10000,256]加batch轴并repeat->[10000,1,256]，每个格子仍有自己的初始内容。
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        # 行列位置图[1,256,100,100] -> [1,256,10000] -> [10000,1,256]。
        # 与10000个格子内容查询形状相同，但角色不同：内容等待场景信息，位置告诉注意力自己是哪一格。
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        # 【C1.2：先处理历史几何】从这帧metadata读取与上一队列帧的位移及角度。
        # 历史内容在上帧参考系中，自车移动后同一障碍物对应的当前格子变了，必须对齐后再融合。
        # 这是时序BEV的运动补偿，不是LSS深度推断；没有prev_bev时仍可生成当前BEV。
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        # 本例y每格60/100=0.6米，x每格30/100=0.3米。
        # 位移先除每格米数再除格子数，得到归一化网格平移，不是学习的采样偏移。
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        # [shift_x数组,shift_y数组]先构造[2,1]，permute后[1,2]。
        # 例沿某网格轴实际平移0.6米，对应归一化偏移还需按该轴格长/100换算；方向由上面的角度公式确定。
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        # 没有历史则跳过整个旋转块。历史only_bev通常传[1,10000,256]，完整测试缓存可能[10000,1,256]。
        # 先统一为[10000,1,256]，两种形状表示相同格子/内容，只是batch和序列轴互换。
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    # 取第i=0个样本：[10000,256] -> [100,100,256] -> [256,100,100]。
                    # 为什么变回二维图？torchvision.rotate按空间图像旋转，256通道一起旋转；不是旋转256维向量本身。
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    # 按车辆帧间yaw差（单位度）旋转历史格子内容，旋转中心来自rotate_center配置。
                    # 旋转后展平回[10000,1,256]写回prev_bev；平移则主要通过shift参考点交给时间注意力处理。
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        # 【C1.3：注入自车状态】把一个样本18维状态变[1,18]，通过18->128->256 MLP得[1,256]。
        # [None,:,:]增加序列轴成[1,1,256]，广播加到10000格子查询[10000,1,256]，形状不变。
        # 每格得到相同帧级状态信息，但各自原始内容和行列位置不同，不会因此变成同一格子。
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        # 【C1.4：整理相机内容供BEV读取】Tiny只有lvl=0、feat=[1,6,256,12,20]。
        # 12x20是相机透视特征平面，与100x100鸟瞰平面不是同一个网格。接下来仅展平，不做LSS lift。
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # [1,6,256,12,20] -> [1,6,256,240] -> [6,1,240,256]。
            # 240=12x20，只改变轴排列，不丢相机身份；稍后加相机/尺度嵌入进一步标记来源。
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        # 将各尺度位置沿序列轴相连。只有1尺度时仍是[6,1,240,256]。
        # 不是将六个相机拼成BEV，也不是沿深度展开到240个深度层。
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        # 尺度空间shape=[[12,20]]，位置总数240；唯一尺度起点=[0]。
        # 可变形注意力在展平位置数组中采样时，需要知道每个尺度的二维尺寸和起始位置。
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # [6,1,240,256] -> [6,240,1,256]，调整为encoder要求的相机/空间/batch/通道布局。
        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # 【C1.5：真正从图像生成BEV】跳到VAD/modules/encoder.py的BEVFormerEncoder.forward。
        # 传入格子查询[10000,1,256]、相机key/value[6,240,1,256]、历史BEV、位置编码、运动shift。
        # 内部先为10000格子建立固定几何参考位置（每柱4个高度），经lidar2img投到六相机。
        # BEVFormerLayer先以时间可变形注意力读取历史内容，再以空间可变形注意力读取当前相机特征，然后FFN。
        # 这是“BEV格子到图像找信息”，与你熟悉的LSS“图像特征按深度lift到空间”方向不同。
        # 没有历史时TemporalSelfAttention用当前查询构造初始value；不是读一个不存在的全零历史GT。
        # 几何参考形状可对照外部encoder：3D参考[1,4,10000,3]，投到相机后[6,1,10000,4,2]。
        # 最后一维2是每个投影点的图像位置，不是256维特征；mask说明哪些相机看得到该高度点。
        # 时间注意力内部会暂将“历史value+当前查询”组织成[2,10000,256]，这里2不是两组原始样本。
        # 无历史时用当前查询复制两份初始化value；有历史时带来真正的上一时刻场景内容。
        # 空间注意力围绕图像投影位置采样，预测offset/weight后加权图像value，再写回对应BEV格子。
        # 它的图像采样点配置与后面解码器“8头x4点读BEV”不同，不可把两者都当同一种32点采样。
        # 编码器按配置执行3层，输出[1,10000,256]，每格已经含当前场景内容。
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        # 【C1返回】历史调用回到Head B2 -> A2，作为下一帧prev_bev。
        # 当前调用回到本文件C2继续检测/地图解码；因此相同函数的返回者不同，要看调用栈而非只看变量名。
        return bev_embed

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    # 【C2：当前帧完整感知】先调用C1生成BEV[1,10000,256]，再分别给检测/地图槽读取。
    # 输入的object_query_embed=[300,512]、map_query_embed=[2000,512]是共享初始参数表。
    # 先拆512为位置/内容各256：位置嵌入确定初始采样锚点，内容在注意力中逐步获得场景信息。
    # 初始检测参考[1,300,3]，地图参考[1,2000,2]；这些会实际决定去BEV哪里读取，并非仅可视化。
    # 检测/地图解码共享同一BEV但查询不同，互不先生成对方的GT；本函数返回tuple给Head B4。
    # 注意此函数返回BEV是[10000,1,256]，而only_bev返回[1,10000,256]；差别只是permute轴顺序。
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                map_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                map_reg_branches=None,
                map_cls_branches=None,                
                prev_bev=None,            
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        # 【C2.1：先让当前图像变BEV】进入C1时prev_bev是A2算出的较近历史记忆。
        # 返回[1,10000,256]是融合了历史与当前图像的“当前BEV”，此后所有感知查询从它读取。
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        # 【C2.2：检测槽开始工作】[300,512]拆为两个[300,256]：query_pos位置嵌入、query内容。
        # 位置嵌入不是(x,y,z)，内容也不是已经知道的车辆属性；它们是学习好的初始槽位参数。
        # 增加/广播batch轴后各为[1,300,256]。expand不为每样本新建独立参数表。
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        # 位置嵌入[1,300,256]经共享Linear(256,3)->[1,300,3]，sigmoid后是0..1的三维初始参考坐标。
        # 为什么需要坐标？可变形交叉注意力要知道围绕BEV哪个位置采样，再由查询预测局部偏移。
        # 最开始由学习的位置嵌入决定，不是随机从GT采点；训练让这些初始锚点与内容查询更有效。
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # 【C2.3：地图点槽开始工作】[2000,512]->位置/内容各[2000,256]，扩batch后[1,2000,256]。
        # 100x20归属在Head构造时已确定；每个位置嵌入再用共享Linear(256,2)+sigmoid得到[1,2000,2]参考点。
        # 地图点是预测折线/轮廓的节点，不是BEV编码器10000个固定网格中心。
        map_query_pos, map_query = torch.split(
            map_query_embed, self.embed_dims, dim=1)
        map_query_pos = map_query_pos.unsqueeze(0).expand(bs, -1, -1)
        map_query = map_query.unsqueeze(0).expand(bs, -1, -1)
        map_reference_points = self.map_reference_points(map_query_pos)
        map_reference_points = map_reference_points.sigmoid()
        map_init_reference_out = map_reference_points        

        # 进入解码器前改成序列优先：检测[1,300,256]->[300,1,256]；地图[1,2000,256]->[2000,1,256]。
        # 这些permute为接口整理轴，未改内容数值，更没有把256维位置嵌入当成坐标。
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        map_query = map_query.permute(1, 0, 2)
        map_query_pos = map_query_pos.permute(1, 0, 2)
        # BEV[1,10000,256]->[10000,1,256]，作为两个解码器共享的value。
        # 这就是为什么完整Head返回BEV与历史only_bev返回BEV轴序不同；后续代码兼容两种。
        bev_embed = bev_embed.permute(1, 0, 2)

        if self.decoder is not None:
            # [L, Q, B, D], [L, B, Q, D]
            # 【C2.4：检测解码】去VAD/modules/decoder.py的DetectionTransformerDecoder。
            # query=[300,1,256]，位置同shape，参考[1,300,3]，value为BEV[10000,1,256]，key=None。
            # 每层普通自注意力让目标槽交流；可变形交叉注意力围绕锚点读BEV，然后FFN更新内容。
            # 交叉注意力不先做300x10000的QK点积；从查询预测offset/weight，常见8头每头4个位置，读取投影后的BEV value。
            # 回归分支细化参考坐标供下一层；三层返回[3,300,1,256]内容和逐层参考点。
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                **kwargs)
            inter_references_out = inter_references
        else:
            inter_states = query.unsqueeze(0)
            inter_references_out = reference_points.unsqueeze(0)

        if self.map_decoder is not None:
            # [L, Q, B, D], [L, B, Q, D]
            # 【C2.5：地图解码】进入本文件C3，输入2000个点查询，不是100个类别查询。
            # 与检测读取同一份BEV，但独立查询/解码器。输出[3,2000,1,256]和[3,1,2000,2]参考点。
            # 最终地图类别和点坐标还要回Head B4.2计算；这里不是先做鸟瞰实例分割再矢量化。
            map_inter_states, map_inter_references = self.map_decoder(
                query=map_query,
                key=None,
                value=bev_embed,
                query_pos=map_query_pos,
                reference_points=map_reference_points,
                reg_branches=map_reg_branches,
                cls_branches=map_cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=map_query.device),
                level_start_index=torch.tensor([0], device=map_query.device),
                **kwargs)
            map_inter_references_out = map_inter_references
        else:
            map_inter_states = map_query.unsqueeze(0)
            map_inter_references_out = map_reference_points.unsqueeze(0)

        # 【C2返回B3】tuple顺序必须与Head的bev_embed,hs,init_reference,...解包一致。
        # 本函数没有生成运动/自车轨迹；回VAD_head.py的B4/B5/B6继续。
        return (
            bev_embed, inter_states, init_reference_out, inter_references_out,
            map_inter_states, map_init_reference_out, map_inter_references_out)


# 登记类名供配置type查找；导入本文件时登记，不在此创建样本或执行前向。
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    # 【准备运动/规划交互容器】父类根据配置建立一层BaseTransformerLayer，常见8头、D256。
    # 这里用于已提取的目标/地图/自车特征交互，不再直接读取原始图片或LSS深度体。
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(CustomTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    # 【B5/B6共用交互接口】query/key/value分别来自调用处，不同调用任务不同：
    # 运动自注意力：内容[1800,1,256]；Q/K加运动位置嵌入，V为运动内容。
    # 运动-地图：Q=[1,1800,256]，K/V=[p+1,1800,256]；这里1800是人为合并的独立查询batch，不是1800组原始样本。
    # 自车-目标：Q=[1,1,256]，K/V=[q,1,256]；自车-地图同样Q一条，K/V为地图实例。
    # 普通注意力实际Q/K/V还过共享投影，8头各32维；内容+位置是加法，最后concat发生在Head中。
    # return_intermediate=False时保持[查询数,batch,256]，不会凭空多一个层轴；原英文额外1轴说法不适用。
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                *args,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        # 当前交互通常只有一层。调用方提供Q/K/V和位置，本容器不再给它们新建独立语义。
        # BaseTransformerLayer在普通多头注意力内部做投影：每头Q/K/V末维32，8头合成256。
        # mask=True的key被忽略；注意它是地图/目标key mask，不是GT未来时间mask。
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs)

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        # 交互后返回给Head B5或B6，保持调用约定的[序列,batch,256]。
        # 返回的内容不是轨迹坐标，后面预测MLP才将内容转为12或36个几何数值。
        return query