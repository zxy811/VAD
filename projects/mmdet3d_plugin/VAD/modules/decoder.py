# 【当前BEV生成之后，检测/地图查询读取BEV】本文件不是图片->BEV的编码器。
# DetectionTransformerDecoder负责300检测查询三层迭代；地图容器在上层VAD_transformer.py。
# CustomMSDeformableAttention被检测/地图交叉注意力复用：300或2000查询读10000个BEV位置。
# 本例每头4采样点，8头合计32位置，读取后的内容仍256维；地图最终每实例20节点是另一种点数。
#
# 中文阅读路线：以一组数据B=1为例，先理解当前阶段，再看维度和语法。
# 训练三帧中，每帧六相机；CNN/FPN后当前帧特征[1,6,256,12,20]，历史逐帧也以此形状编码。
# BEV=100x100格，内容256维；初始BEV query=[10000,256]，不是坐标，也不是已经生成的RGB鸟瞰图。
# 位置嵌入bev_pos为[1,256,100,100]；固定参考点是xyz/xy坐标；三者不要混淆。
# 你熟悉的LSS：图像像素+深度分布->lift到空间->汇聚BEV。
# 这里BEVFormer：先有BEV格子查询和几何参考位置->投到图像->可变形采样->更新格子内容。
# prev_bev是上一历史时刻已经提取的内容特征，非GT；无历史为None，有历史先按自车运动对齐。
# 可变形注意力：查询经共享Linear预测offset与weight，围绕参考点读V；不对所有格子做全量QK点积。
# 普通注意力：内容+位置形成Q/K投影，QK点积得到权重，V加权；不预测二维采样偏移。
# 本例编码器10000查询，检测300查询，地图2000点查询；具体数字可由配置改变。
# 多头D=256、8头时每头32维。符号Z本例=4个高度锚点；F=1尺度；不要把这里Z与特征维D混用。
# reshape/view重排不改元素总数；permute换轴；None/unsqueeze加轴；cat拼旧轴；stack加新轴。
# 参数共享：同一Linear用于所有位置，W是参数，每次预测的offset是中间结果，不是每位置一套新W。
# 初始化构造在训练开始前；forward每组数据计算一次；backward/optimizer.step在框架Hook里。
# 仅增加#注释，不修改原有代码、docstring或作者保留的兼容/边界行为。

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import mmcv
import cv2 as cv
import copy
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


# 【坐标细化工具】logit(r)=log(r/(1-r))，clamp防止0/1无穷；输入输出shape一致。
# 新参考=sigmoid(logit(旧参考)+回归残差)，零残差保持原坐标；不是256维内容解码本身。
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
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    # 【准备300个检测槽的多层读取】父类建3层DetrTransformerDecoderLayer，每层普通self_attn+可变形cross_attn+FFN。
    # return_intermediate=True保存三层特征及细化参考点；三层不是历史三帧。
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    # 【BEV->目标】query=[300,1,256]，value在kwargs里为BEV[10000,1,256]，初始参考=[1,300,3]。
    # 每层：普通自注意力让目标槽交流；可变形交叉注意力围绕xy读BEV；FFN后回归头细化xyz供下一层。
    # 注意坐标2维采样只用xy，检测参考仍保留xyz三维；编码回归tmp第4索引对应z，不是前三维都为xyz。
    # 返回[3,300,1,256]内容与[3,1,300,3]参考；回VADHead再生成框/类别，不直接输出最终检测结果。
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
        output = query
        intermediate = []
        intermediate_reference_points = []
        # 当前BEV已经生成；这里300检测槽反复读取它，不再回到六相机。
        # 每层普通self_attn让检测槽互相交流，再用可变形cross_attn从BEV采样。
        for lid, layer in enumerate(self.layers):

            # 检测参考本为[1,300,3]归一化xyz，这里只取xy并加尺度轴->[1,300,1,2]。
            # 读BEV不需要z采样轴，但z仍保留在后面的检测回归参考里。
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                # 回归残差加在logit参考坐标上，再sigmoid回0..1；不是把偏移直接加到米制框。
                # xy用tmp[:2]，z用tmp[4:5]，由框编码布局决定，不应假定tmp[:3]就是xyz。
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                # 下一层用细化后的参考，但detach切断跨层参考更新链的反传。
                # 各层当前输出仍可通过各自监督训练；detach不等于所有检测网络停止学习。
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class CustomMSDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    # 【建立BEV采样器】当前D256，8头，1尺度，每头4点。被300检测槽或2000地图点槽复用。
    # offset Linear(256,64)，W[64,256]；weight Linear(256,32)，value/output各Linear(256,256)。
    # 同层对全部查询位置共享W，预测的offset则随输入内容变化；不是每个位置独有一套参数。
    # 配置num_levels=1覆盖默认4；这里采样点4与图像MSDeformableAttention3D的8不同。
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        # 检查每头通道是否为2的幂：本例256/8=32，利于CUDA效率。
        # n & (n-1)是整数位运算；这个函数只检查配置，没有修改或采样特征。
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    # 【起始采样模式】offset权重初始化0、bias按8头方向与4个半径排列；不是所有offset完全随机。
    # weight层权重/bias为0，softmax后起始较均匀；value/output投影Xavier。只在初始化执行，训练随后改变参数。
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        # 先把偏移Linear的W设为0，再用不同头的方向/逐点半径初始化bias。
        # 起始采样不是随机散落；训练后W更新，偏移才随不同输入query改变。
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    # 【Q到BEV哪里读取】默认batch_first=False，检测传query[300,1,256]，地图传[2000,1,256]。
    # 内部转[1,Q,256]；value=[1,10000,256]投影分头[1,10000,8,32]。
    # offset=[1,Q,8,1,4,2]，weight=[1,Q,8,1,4]，参考=[1,Q,1,2]，位置=参考+offset/(100,100)。
    # 内核逐头采样、加权后[1,Q,256]，输出投影/残差再恢复[Q,1,256]。
    # 这里不先构造Qx10000的点积K矩阵；Q/K/V一词来自接口，实际这条路径没有独立全量点积K。
    # 输出是内容，不是坐标；参考点的迭代更新在调用者Detection/MapDecoder的回归分支执行。
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # 共享256->256通道投影，位置数不变；随后reshape分8头，每头32维。
        # 不同头各读自己的32维value并采样，核把8头拼回256维；不是把8头求和成32维。
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # 检测Q=300或地图Q=2000，Linear(256,64)预测8头x1尺度x4点x2偏移。
        # 权重Linear(256,32)后沿4点softmax；与时间注意力不同，这里不拼两个时间来源。
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            # 参考[1,Q,1,2]加偏移/[100,100]，得到[1,Q,8,1,4,2]实际BEV采样位置。
            # 100x100是BEV特征尺寸，不是原始图像384x640，也不是米制pc_range。
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            # 进入本目录multi_scale_deformable_attn_function.py的CUDA自动求导接口。
            # 采样坐标与权重已在Python算好；核负责双线性插值与加权和，返回[临时B,Q,256]。
            # 采样坐标可以是小数，梯度经插值传回offset预测Linear。
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
