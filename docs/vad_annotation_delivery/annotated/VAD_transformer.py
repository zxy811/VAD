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
# 职责：将多相机图像特征和历史 BEV 融合成当前 BEV，再分别运行检测和地图解码器。
# 核心：VADHead.forward -> VADPerceptionTransformer.forward -> get_bev_features
# -> VAD/modules/encoder.py 的 BEVFormerEncoder -> 检测/地图解码。
# BEV 时间/空间可变形注意力实现在 VAD/modules/temporal_self_attention.py、spatial_cross_attention.py；
# 检测可变形注意力在 VAD/modules/decoder.py；本文件负责组织它们，未逐行展开外部 CUDA 内核。
# 可变形交叉注意力不是查询对每个 BEV 格子的全量 QK 点积：查询预测采样偏移和权重，采样 V。
# 普通 MultiheadAttention 则使用投影后的 Q/K 点积和 V 加权；不要混淆两种计算。

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


# 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

# 【函数入口】inverse_sigmoid
# 对[0,1]参考坐标做logit：log(x/(1-x))；clamp避免0/1导致无穷。
# 输入输出形状相同；用于参考点残差更新 r_new=sigmoid(delta+logit(r_old))。
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
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
    return torch.log(x1 / x2)


# 【类】MapDetectionTransformerDecoder：构造时执行__init__；实例可通过方法或nn.Module调用机制参与计算。
# @...register_module()把类登记为名称->类的映射；导入时注册，不会在这里实例化或训练。
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapDetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    # 【函数入口】MapDetectionTransformerDecoder.__init__
    # 父类按配置构建self.layers，return_intermediate控制是否堆叠每层特征/参考点。
    # 同一个对象被forward调用多次，层参数在所有batch/点查询之间共享。
    def __init__(self, *args, return_intermediate=False, **kwargs):
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        super(MapDetectionTransformerDecoder, self).__init__(*args, **kwargs)
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.return_intermediate = return_intermediate
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.fp16_enabled = False

    # 【函数入口】MapDetectionTransformerDecoder.forward
    # 地图query=[P*S,B,D]，reference_points=[B,P*S,2]，BEV value=[10000,B,D]。
    # 每层先用当前参考点做注意力/FFN更新特征，再用该层reg_branch预测二维残差更新参考点。
    # reference_points_input=[B,P*S,1,2]，其中1是BEV特征层数维，不是时间维。
    # 参考点detach阻断该坐标更新路径向上一层反传，但当前层内容特征与回归头仍可被各自loss训练。
    # return_intermediate=True返回[R,P*S,B,D]和[R,B,P*S,2]；否则返回未额外加层轴的output与参考点。
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
        # 当前解码层内容特征，参考点更新和特征更新是两条不同变量路径
        output = query
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        intermediate = []
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        intermediate_reference_points = []
        # enumerate同时给出序号与元素；序号可能表示解码层/尺度/样本，按循环对象区分，不都代表时间。
        for lid, layer in enumerate(self.layers):

            # 截取二维参考坐标再unsqueeze(2)：[B,P*S,2] -> [B,P*S,1,2]；1是BEV尺度轴。
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            # 当前解码层内容特征，参考点更新和特征更新是两条不同变量路径
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            # 当前解码层内容特征，参考点更新和特征更新是两条不同变量路径
            # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
            output = output.permute(1, 0, 2)

            # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
            if reg_branches is not None:
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                tmp = reg_branches[lid](output)

                # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
                assert reference_points.shape[-1] == 2

                # 由旧参考点logit加回归残差再sigmoid得到的新归一化坐标，与query的D维特征不同
                # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
                new_reference_points = torch.zeros_like(reference_points)
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                # 由旧参考点logit加回归残差再sigmoid得到的新归一化坐标，与query的D维特征不同
                # sigmoid逐元素映射到(0,1)，shape不变；类别分数各自独立，不保证沿类别/模式之和为1。
                new_reference_points = new_reference_points.sigmoid()

                # 保留更新后的数值，切断跨层参考点这条梯度链；不是冻结所有Transformer参数或内容特征。
                reference_points = new_reference_points.detach()

            # 当前解码层内容特征，参考点更新和特征更新是两条不同变量路径
            # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
            output = output.permute(1, 0, 2)
            # 是否返回所有解码层的中间结果；开启时多一层轴R，否则只返回末层结果。
            if self.return_intermediate:
                # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
                intermediate.append(output)
                # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
                intermediate_reference_points.append(reference_points)

        # 是否返回所有解码层的中间结果；开启时多一层轴R，否则只返回末层结果。
        if self.return_intermediate:
            # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return output, reference_points


# 【类】VADPerceptionTransformer：构造时执行__init__；实例可通过方法或nn.Module调用机制参与计算。
# @...register_module()把类登记为名称->类的映射；导入时注册，不会在这里实例化或训练。
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

    # 【函数入口】VADPerceptionTransformer.__init__
    # 接收encoder/decoder/map_decoder配置，通过build_transformer_layer_sequence创建对象。
    # encoder是BEVFormerEncoder；decoder为目标检测；map_decoder为地图；init_layers另建共享投影层。
    # num_cams与实际输入/标定应匹配，默认6；减少图片数量不等于自动适配完整模型。
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
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        super(VADPerceptionTransformer, self).__init__(**kwargs)
        # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
        self.encoder = build_transformer_layer_sequence(encoder)
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if decoder is not None:
            # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
            self.decoder = build_transformer_layer_sequence(decoder)
        else:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.decoder = None
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if map_decoder is not None:
            # 根据配置type经相应注册器创建对象；构建函数返回模块/工具，不是在此完成整个训练。
            self.map_decoder = build_transformer_layer_sequence(map_decoder)
        else:
            # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
            self.map_decoder = None

        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.embed_dims = embed_dims
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.num_feature_levels = num_feature_levels
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.num_cams = num_cams
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.fp16_enabled = False
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.rotate_prev_bev = rotate_prev_bev
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.use_shift = use_shift
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.use_can_bus = use_can_bus
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.can_bus_norm = can_bus_norm
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.use_cams_embeds = use_cams_embeds
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.two_stage_num_proposals = two_stage_num_proposals
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.rotate_center = rotate_center
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_num_vec = map_num_vec
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.map_num_pts_per_vec = map_num_pts_per_vec
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        self.init_layers()

    # 【函数入口】VADPerceptionTransformer.init_layers
    # 建立level_embeds=[F,D]、cams_embeds=[N,D]可学习表；它们区分尺度和相机。
    # reference_points: Linear(D,3)，map_reference_points: Linear(D,2)，权重在所有查询间共享。
    # can_bus_mlp:18->128->256，输出用于广播相加到所有BEV查询；不是给每个BEV格子独立一个MLP。
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        # Parameter把Tensor登记为模型参数；是否更新还由requires_grad及是否参与loss决定。
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        # Parameter把Tensor登记为模型参数；是否更新还由requires_grad及是否参与loss决定。
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        # 创建线性层：最后维in->out，W=[out,in]、b=[out]，在全部batch/序列位置共享；这里创建层而非执行forward。
        self.reference_points = nn.Linear(self.embed_dims, 3)
        # 创建线性层：最后维in->out，W=[out,in]、b=[out]，在全部batch/序列位置共享；这里创建层而非执行forward。
        self.map_reference_points = nn.Linear(self.embed_dims, 2)
        # 创建线性层：最后维in->out，W=[out,in]、b=[out]，在全部batch/序列位置共享；这里创建层而非执行forward。
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    # 【函数入口】VADPerceptionTransformer.init_weights
    # 训练开始前重置参数。矩阵用Xavier；相机/尺度嵌入用normal；坐标投影与CAN bus MLP用Xavier且bias=0。
    # 可变形注意力还执行自己的特殊偏移/权重初始化，具体在VAD/modules/对应类。
    # 不是每个forward重新随机初始化。
    def init_weights(self):
        """Initialize the transformer weights."""
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for p in self.parameters():
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if p.dim() > 1:
                # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                nn.init.xavier_uniform_(p)
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for m in self.modules():
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                    m.init_weight()
                except AttributeError:
                    # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
                    m.init_weights()
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        normal_(self.level_embeds)
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        normal_(self.cams_embeds)
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        xavier_init(self.map_reference_points, distribution='uniform', bias=0.)
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    # 【函数入口】VADPerceptionTransformer.get_bev_features
    # 输入尺度列表每项[B,N,D,h,w]、BEV查询[Hbev*Wbev,D]、位置图[B,D,Hbev,Wbev]。
    # 先由metadata车辆运动对齐历史BEV，再把18维车辆状态嵌入加入BEV查询。
    # 图片位置展平并添加相机/尺度嵌入，送BEVFormerEncoder；输出通常[B,Hbev*Wbev,D]。
    # Tiny单尺度h*w=12*20=240，总BEV位置10000；相机图像特征不是同一张BEV二维网格。
    # self.encoder(query,key,value)中key/value来自图像特征；内部执行时间/空间可变形注意力。
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

        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        bs = mlvl_feats[0].size(0)
        # BEV查询[Hbev*Wbev,D]->[Hbev*Wbev,1,D]->[Hbev*Wbev,B,D]；repeat复制数据，源表参数仍共享。
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        # 位置图[B,D,Hbev,Wbev]->[B,D,Hbev*Wbev]->[Hbev*Wbev,B,D]，对齐BEV查询序列。
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        # BEV每格y方向物理跨度=real_h/Hbev，Tiny约0.6米
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        grid_length_y = grid_length[0]
        # BEV每格x方向物理跨度=real_w/Wbev，Tiny约0.3米
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        # 车辆运动换算为BEV归一化y偏移，供时序对齐
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        # 车辆运动换算为BEV归一化x偏移，供时序对齐；不等于注意力学习的offset
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        # 车辆运动换算为BEV归一化y偏移，供时序对齐
        shift_y = shift_y * self.use_shift
        # 车辆运动换算为BEV归一化x偏移，供时序对齐；不等于注意力学习的offset
        shift_x = shift_x * self.use_shift
        # 按[shift_x,shift_y]构造[2,B]再permute成[B,2]；是归一化BEV运动偏移，不是学习的采样偏移。
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if prev_bev is not None:
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if prev_bev.shape[1] == bev_h * bev_w:
                # 历史BEV特征Tensor；与metadata内同名布尔标志不同
                # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
                prev_bev = prev_bev.permute(1, 0, 2)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.rotate_prev_bev:
                # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    # 取当前样本can_bus最后一项的帧间yaw角度（度），给torchvision.rotate使用；倒数第二项是绝对yaw弧度。
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    # 单样本BEV[HW,D]->[Hbev,Wbev,D]->[D,Hbev,Wbev]，按特征图进行旋转对齐。
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    # 旋转后的[D,Hbev,Wbev]->[Hbev,Wbev,D]->[HW,1,D]，恢复序列布局。
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        # 把B个metadata内18维车辆状态列表变[B,18]浮点Tensor，设备/类型跟随BEV查询；原始状态不是可训练参数。
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        # 车辆状态[B,18]经共享MLP变[B,D]，[None,:,:]增加序列轴为[1,B,D]。
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        # 广播[1,B,D]到全部10000个BEV位置再相加；数值融合，不改变查询shape或给每格创建独立MLP。
        bev_queries = bev_queries + can_bus * self.use_can_bus

        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        feat_flatten = []
        # 每个图像尺度的[h,w]元数据[F,2]，Tiny为[[12,20]]
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        spatial_shapes = []
        # enumerate同时给出序号与元素；序号可能表示解码层/尺度/样本，按循环对象区分，不都代表时间。
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # 图像特征[B,N,D,h,w]->[B,N,D,h*w]->[N,B,h*w,D]，保留相机轴与每个视角的空间序列。
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.use_cams_embeds:
                # 相机嵌入[N,D]扩展[N,1,1,D]，广播到当前相机所有batch/像素；不同相机可学习区分。
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            # 当前尺度嵌入[1,1,1,D]广播相加到[N,B,h*w,D]，尺度id=lvl，不是相机id。
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            spatial_shapes.append(spatial_shape)
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            feat_flatten.append(feat)

        # 把多尺度图像位置拼在第2轴：[N,B,sum(h_l*w_l),D]；Tiny单尺度sum=240。
        feat_flatten = torch.cat(feat_flatten, 2)
        # 每个图像尺度的[h,w]元数据[F,2]，Tiny为[[12,20]]
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        # 各尺度h*w累加后右移一位，得到[F]起始索引；单尺度为[0]，用于可变形采样定位。
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # [N,B,sum_hw,D]->[N,sum_hw,B,D]，按BEVFormerEncoder接口布置；没有产生新特征通道。
        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # 调用BEVFormerEncoder.forward：query=BEV查询，key/value=展平相机特征，输出通常[B,Hbev*Wbev,D]。
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

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return bev_embed

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    # 【函数入口】VADPerceptionTransformer.forward
    # 完整感知：先get_bev_features，再分别准备检测/地图查询并调用两个解码器。
    # 检测Embedding[A,2D]拆为query_pos和query，各[A,D]，扩展batch后送解码器。
    # 初始检测参考点由位置嵌入Linear(D,3)+sigmoid得到[B,A,3]；地图为[B,P*S,2]。
    # 这些参考点是采样几何锚点，会实际影响读取BEV的位置，不是仅供可视化。
    # 返回BEV[R之外的序列布局]、每层检测特征/参考点、每层地图特征/参考点；Head再回归坐标。
    # 检测/地图decoder调用key=None，value=BEV；这条可变形交叉注意力没有全量QK点积键。
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

        # 当前BEV内容特征，内部按调用可能是[B,HW,D]或[HW,B,D]；空间HW=10000
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        bs = mlvl_feats[0].size(0)
        # 检测Embedding[A,2D]沿最后轴等分：位置[A,D]、内容[A,D]；torch.split返回两个Tensor。
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        # 查询位置嵌入，与query形状相同；Q/K可加它，但V通常只用内容
        # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        # 当前内容特征序列；最后维D，具体Q/B轴由当前函数入口和permute确定
        # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
        query = query.unsqueeze(0).expand(bs, -1, -1)
        # 检测位置嵌入[B,A,D]经共享Linear(D,3)输出[B,A,3] logits，后续sigmoid成为归一化参考点。
        reference_points = self.reference_points(query_pos)
        # sigmoid逐元素映射到(0,1)，shape不变；类别分数各自独立，不保证沿类别/模式之和为1。
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # 地图Embedding[P*S,2D]等分为位置和内容[P*S,D]；每点并非512个空间坐标。
        map_query_pos, map_query = torch.split(
            map_query_embed, self.embed_dims, dim=1)
        # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
        map_query_pos = map_query_pos.unsqueeze(0).expand(bs, -1, -1)
        # 地图内容特征；具体阶段为点序列[B,P,S,D]或池化/筛选后实例序列，参照本函数入口
        # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
        map_query = map_query.unsqueeze(0).expand(bs, -1, -1)
        # 地图位置嵌入[B,P*S,D]经Linear(D,2)得到[B,P*S,2]；sigmoid后定位BEV采样中心。
        map_reference_points = self.map_reference_points(map_query_pos)
        # sigmoid逐元素映射到(0,1)，shape不变；类别分数各自独立，不保证沿类别/模式之和为1。
        map_reference_points = map_reference_points.sigmoid()
        map_init_reference_out = map_reference_points        

        # 当前内容特征序列；最后维D，具体Q/B轴由当前函数入口和permute确定
        # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
        query = query.permute(1, 0, 2)
        # 查询位置嵌入，与query形状相同；Q/K可加它，但V通常只用内容
        # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
        query_pos = query_pos.permute(1, 0, 2)
        # 地图内容特征；具体阶段为点序列[B,P,S,D]或池化/筛选后实例序列，参照本函数入口
        # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
        map_query = map_query.permute(1, 0, 2)
        # permute按给定轴编号重排，元素数不变；例如(1,0,2)交换前两轴，不是矩阵乘法。
        map_query_pos = map_query_pos.permute(1, 0, 2)
        # BEV [B,Hbev*Wbev,D]->[Hbev*Wbev,B,D]，适配两个解码器value布局；不是重算BEV。
        bev_embed = bev_embed.permute(1, 0, 2)

        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.decoder is not None:
            # [L, Q, B, D], [L, B, Q, D]
            # 检测解码器读取BEV，返回[R,A,B,D]内容特征及逐层[B,A,3]参考点集合；R=3。
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
            # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
            inter_states = query.unsqueeze(0)
            # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
            inter_references_out = reference_points.unsqueeze(0)

        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.map_decoder is not None:
            # [L, Q, B, D], [L, B, Q, D]
            # 地图解码器读取同一BEV，返回[R,P*S,B,D]及[R,B,P*S,2]参考点；与检测并列而非由检测框生成地图。
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
            # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
            map_inter_states = map_query.unsqueeze(0)
            # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
            map_inter_references_out = map_reference_points.unsqueeze(0)

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return (
            bev_embed, inter_states, init_reference_out, inter_references_out,
            map_inter_states, map_init_reference_out, map_inter_references_out)


# 【类】CustomTransformerDecoder：构造时执行__init__；实例可通过方法或nn.Module调用机制参与计算。
# @...register_module()把类登记为名称->类的映射；导入时注册，不会在这里实例化或训练。
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    # 【函数入口】CustomTransformerDecoder.__init__
    # 父类根据配置构建self.layers；当前运动/规划交互常配单层BaseTransformerLayer。
    # return_intermediate=False时不会额外增加解码层轴。
    def __init__(self, *args, return_intermediate=False, **kwargs):
        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        super(CustomTransformerDecoder, self).__init__(*args, **kwargs)
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.return_intermediate = return_intermediate
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.fp16_enabled = False

    # 【函数入口】CustomTransformerDecoder.forward
    # 通用交互层容器：query=[Q,B,D]、key/value=[J,B,D]，mask=[B,J]。
    # 这里仅转发给BaseTransformerLayer，具体self_attn/cross_attn/FFN顺序由配置决定。
    # 运动自注意力：Q/K输入内容加位置、V内容；运动-地图交叉：Q运动、K地图+位置、V地图。
    # 规划自车-目标/地图同理；普通8头D=256时每头32维，线性投影共享于全部序列位置。
    # return_intermediate=True堆叠[R,Q,B,D]；False直接返回[Q,B,D]，旧英文注释的额外1轴不适用。
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
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        intermediate = []
        # enumerate同时给出序号与元素；序号可能表示解码层/尺度/样本，按循环对象区分，不都代表时间。
        for lid, layer in enumerate(self.layers):
            # 当前内容特征序列；最后维D，具体Q/B轴由当前函数入口和permute确定
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

            # 是否返回所有解码层的中间结果；开启时多一层轴R，否则只返回末层结果。
            if self.return_intermediate:
                # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
                intermediate.append(query)

        # 是否返回所有解码层的中间结果；开启时多一层轴R，否则只返回末层结果。
        if self.return_intermediate:
            # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
            return torch.stack(intermediate)

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return query
