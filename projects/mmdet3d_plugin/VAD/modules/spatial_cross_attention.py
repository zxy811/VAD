# 【当前图像特征->BEV内容】从encoder.py的BEVFormerLayer.cross_attn进入。
# SpatialCrossAttention按可见相机挑格子、汇总结果；MSDeformableAttention3D做每相机局部采样。
# Tiny：图像value每相机240位置，每位置256；BEV10000格，每柱4高度投影锚点。
# 本例每头总8采样点分配到4锚点（每锚2点），8头共64点/可见相机；不是4x8x8=256。
# 采样偏移与归一化权重由查询Linear生成；相机可见mask来自标定几何，并不是预测道路分割。
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
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    # 【图像采样的相机调度层】建立内层MSDeformableAttention3D与输出投影，保存相机数6。
    # 外层负责哪些格子在哪个相机可见；内层负责每个相机中的偏移/权重/插值采样。
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    # 仅初始化外层output_proj(256,256)；名字是单数init_weight，和内层init_weights要区分。
    # 可变形offset/weight由内层自己的初始化负责。
    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    # 【当前六相机->鸟瞰格子】query=[1,10000,256]，key/value=[6,240,1,256]，投影参考=[6,1,10000,4,2]。
    # 对每相机选至少一个高度锚点可见的BEV格子，补齐到最长J，临时把6相机合并为batch=6。
    # 内层query=[6,J,256]，value=[6,240,256]，reference=[6,J,4,2]；J随视角和样本变化，不固定10000。
    # 内层返回后按原格子索引散射回slots[1,10000,256]；按可见相机数平均，再投影+残差。
    # 这里只融合相机，不修改固定BEV网格位置；新内容仍256维。选择索引实现使用首个batch的可见性，扩B需留意。
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        # 只为可见格子计算该相机的图像采样，减少无效开销。
        # 每格有4高度，只要至少一个高度投影可见就进入该相机索引；J是六相机中最多可见格子数。
        # 这里按mask_per_img[0]取索引，是首batch样本；本注释固定B=1解释实际运行。
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        # 临时缓存[1,6,J,256]与参考[1,6,J,4,2]。
        # 每相机可见格子数不同，补到同一J才能批量计算；不是所有格子都真实在6相机里可见。
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        # 图像输入[6,240,1,256]->[1,6,240,256]->[6,240,256]。
        # 把相机合并到临时batch；6是相机数，240=12x20是每张图像特征位置数。
        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        # 进入MSDeformableAttention3D：Q=[6,J,256]，V=[6,240,256]。
        # 4个高度投影中心=[6,J,4,2]，围绕它们预测二维偏移并读图像，得到[6,J,256]。
        # 恢复[1,6,J,256]后按相机索引写回原来10000个格子。
        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                # 散射累加：index_query_per_img把压缩后的第k个查询映回原BEV格子编号。
                # 同一格子可能在多相机可见，先累加各相机内容，再除可见相机数量。
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        # count=[1,10000]，不是采样点权重；表示每格可见相机数。
        # clamp>=1避免完全不可见格子除0；这些格子的采样增量为0，仍可保留残差内容。
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
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

    # 【图像特征采样器】D256，8头，1图像尺度，每头总8点；每格有4个高度投影锚点。
    # offset Linear(256,128)，weight Linear(256,64)；value Linear(256,256)，每位置共享。
    # 8点会重排成2x4分配到4锚点，不是每锚8点；原英文注释的乘法描述应以实际view为准。
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
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

        self.init_weights()

    # offset层W=0，bias按头方向与半径初始化；weight层W/b=0，value投影Xavier。
    # 初始化给出结构化附近采样，后续任务loss训练偏移/权重；不需要单独的偏移GT。
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

    # 【投影附近找图像内容】临时batch=6相机，query[6,J,256]，value[6,240,256]，reference[6,J,4,2]。
    # 每头32维value；offset[6,J,8,1,8,2]、weight[6,J,8,1,8]。softmax沿尺度x采样点归一化。
    # 参考4锚点广播后，offset的8点拆成每锚2点，再加到相应投影中心；最终仍每头8点。
    # 采样核返回[6,J,256]，没有在本类输出投影/残差，外层SpatialCrossAttention负责汇总。
    # 这是BEV位置投到相机读取，与LSS逐像素预测深度后lift不同；不形成逐像素深度概率体。
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
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
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
        # [6,J,256]经Linear输出128，view为[6,J,8,1,8,2]。
        # 权重Linear输出64，softmax沿1x8点；偏移来源是BEV查询内容，不是从图片手工算光流。
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
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            # 4高度锚点：参考[6,J,4,2]扩为[6,J,1,1,1,4,2]。
            # 8采样点拆成2组x4锚点，偏移重排[6,J,8,1,2,4,2]，加各自投影中心。
            # 最后合回[6,J,8,1,8,2]；每头总8点，没有变成8x4点。
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
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
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
