# 【历史BEV->当前格子查询】由BEVFormerLayer的self_attn调用；self_attn这里是时间可变形注意力。
# B=1，当前query=[1,10000,256]；value有历史+当前两个来源，暂组织[2,10000,256]。
# query_pos是上层行列位置嵌入bev_pos，256维相加后仍256；再与历史内容concat变512，用于预测偏移/权重。
# 车辆运动给出对齐参考位置；本模块另学局部采样偏移。前者来自位姿，后者来自训练的Linear。
# 每来源8头x4点=32次采样，两来源共64；来源2不是两组原始batch，也不是三帧队列长度3。
# 无历史时用当前query复制两份构造value；没有凭空生成真实历史。返回内容仍[1,10000,256]。
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

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class TemporalSelfAttention(BaseModule):
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
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    # 【建立时间采样器】当前1尺度、8头、每来源4点、2来源，参数层共享全部10000格。
    # 当前256维query加pos后仍256；拼上历史256变512，所以offset Linear(512,128)、weight Linear(512,64)。
    # value/output投影仍256->256。2来源是历史+当前，非队列3帧；多历史帧由上层逐帧递归折叠为prev_bev。
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
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
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    # offset W=0、bias为头方向/半径结构，权重层W/b=0；value/output投影Xavier。
    # 这些是模型参数初始化，之后每组输入算不同offset，但不会重新随机重置参数。
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
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    # 【一个BEV格子怎样读过去】输入query=[1,10000,256]，有历史时value=[2,10000,256]。
    # 先query+query_pos（bev_pos）仍256，再concat历史内容->512预测offset/weight。
    # 每格偏移128=2来源x8头x1尺度x4点x2坐标；权重64不乘坐标2。
    # 来源维移到batch后采样，再还原两个来源并平均，投影+残差，返回[1,10000,256]。
    # vehicle shift来自位姿做粗对齐，sampling_offsets来自线性层做局部查找，两者不是同一个偏移。
    # 当前第一编码层的query是初始格子内容，后层query已读过当前图片；不要总把它叫完整当前BEV。
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

        # 第一帧无历史：复制当前query得到[2,10000,256]，保持“两来源”接口。
        # 这只是无历史的退化输入，不意味着模型看到了不存在的过去图片。
        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)

            # value = torch.cat([query, query], 0)

        if identity is None:
            identity = query
        if query_pos is not None:
            # 逐元素加位置提示：[1,10000,256]+[1,10000,256]仍256维。
            # pos不是额外坐标轴；它帮助同一采样预测层区分各格子的空间身份。
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs,  num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        # 【偏移如何预测的核心】拼历史value[:bs]与带位置的当前query：256+256=512。
        # 随后Linear(512,128)输出两个来源各头各点的二维偏移；Linear(512,64)输出权重logits。
        # Linear共享所有格子；有偏移梯度传回，任务loss就能训练它，不需要偏移标注。
        query = torch.cat([value[:bs], query], -1)
        # 共享256->256通道投影，位置数不变；随后reshape分8头，每头32维。
        # 不同头各读自己的32维value并采样，核把8头拼回256维；不是把8头求和成32维。
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs*self.num_bev_queue,
                              num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        # [1,10000,128]->[1,10000,8,2,1,4,2]。
        # 依次为B、格子、头、历史/当前来源、尺度、采样点、xy偏移。
        # 最后的2是二维坐标；中间的2是来源数，两者含义不同。
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        # 沿每来源的尺度x点数做softmax：本例每来源每头4点之和=1。
        # 它没有在此产生一个历史0.7/当前0.3的总权重；两个来源采样后在后面mean融合。
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        # 将来源维移到batch供同一个采样核处理：权重[2,10000,8,1,4]。
        # 偏移对应[2,10000,8,1,4,2]；value对应[2,10000,8,32]，索引逐来源对齐。
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        if reference_points.shape[-1] == 2:
            # spatial_shapes存[H,W]，偏移分量为[x,y]，故归一化器反取[W,H]。
            # 位置=参考+offset/[100,100]；例如参考(.5,.5)，偏移(2,-1)读(.52,.49)。
            # 车辆位姿shift先给大致对应位置；此处Linear偏移再在附近学习找有效内容。
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

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        # 采样核原输出[2,10000,256]，此时已变[10000,256,1,2]。
        # 对最后两个来源取均值->[10000,256,1]，再恢复[1,10000,256]。
        # 最终output_proj混合通道，dropout后加原query残差。
        output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
