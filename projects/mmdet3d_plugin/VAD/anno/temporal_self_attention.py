# 时间注意力阅读副本：先看整条路线，再进入每一步的形状。
# 本文件按要求放在anno目录，供对照阅读；实际模型使用modules目录中的同名类。
# 原实现中的相对导入也原样保留，阅读副本不加入训练入口或包初始化。
#
# 【你最关心的问题：当前图片在哪里？】
# 当前六相机图片 -> VAD.py/extract_img_feat -> ResNet/FPN -> [1,6,256,12,20]。
# VAD_transformer.py/get_bev_features把图像特征整理为[6,240,1,256]，240=12*20。
# 同一入口准备BEV初始查询[10000,256]，加batch/车辆状态后交给encoder.py。
# encoder.py/BEVFormerLayer.forward按以下顺序更新query：
#   时间注意力(本文件) -> LN -> 空间注意力(spatial_cross_attention.py) -> LN -> FFN -> LN。
# 时间注意力不读取本帧六相机图片；下一步空间注意力才使用上面的图像value。
# 因此“当前来源”不等于“已由当前图片提取的完整BEV”：
#   第1编码层的query尚未通过本帧空间注意力，是初始格子查询经过上层准备后的内容。
#   第1层空间注意力后query含本帧图像信息；第2/3层query会继续携带这些更新。
# 有历史时，上层在进入编码层循环前组织value=[历史BEV, 本次初始query]。
# 层间更新的是query；这份两来源value不会在每层自动替换成最新query。
# 无历史时value=None，本函数才会把该次输入query复制为两个来源。
#
# 【固定一个具体例子，不同时混入B>1的情况】
# B=1组样本；N=10000个BEV格；C=256通道；M=8个头，每头D=32通道。
# T=2个来源(历史/当前)；F=1个尺度；P=4个采样点/头/来源。
# 输入query=[B,N,C]=[1,10000,256]，输入value=[B*T,N,C]=[2,10000,256]。
# 在B=1时value[0]是历史，value[1]是本次初始当前查询；首轴2不是batch=2。
# num_bev_queue=2不是训练queue_length=3；历史多帧由外层逐帧折叠进prev_bev。
#
# 【两条计算路线】
# 路线A：历史内容256 + (当前内容256 + 位置提示256) -> 拼接512 -> 预测偏移/权重。
#   位置提示是加法，所以括号里仍256；拼接历史后才512。
# 路线B：value内容256 -> 投影 -> 分8头各32 -> 按A的位置/权重读取 -> 合并8头256。
# 最后两个来源平均，输出投影，再加最初的query残差 -> [1,10000,256]。
# 固定参考点负责大致去哪里读，offset负责局部调整；二者都不是256维位置嵌入。
# 同一个Linear用于所有10000格；每格输出不同数值，不是每格创建独立网络。
# 下面每个注释中的数字来自当前Tiny配置；构造函数默认尺度4会被配置的1覆盖。
# 所有说明均以当前源码为准：保留原代码、原docstring，仅增加阅读注释。

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

    # 【准备阶段，不处理某一组图片】把配置数值变成可学习层与普通属性。
    # 父类BaseModule管理参数/初始化；self.xxx保存对象，供之后每次forward复用。
    # 默认参数不等于最终配置：Tiny传入num_levels=1；其余例子按C256/M8/T2/P4解释。
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
        # 256/8=32：多头把投影后的内容通道分成8组，每组32。
        # 头不代表相机/帧/目标类别；每组通道配一套采样偏移和权重。
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        # 检查每头通道32是否是2的幂，以便CUDA高效计算。
        # n & (n-1)是位运算；这里只检查配置，不会操作BEV特征。
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
        # 路线A的两个预测层：Linear只作用于输入最后一轴，其余轴保留。
        # 偏移：Linear(512,128)，128=T2*M8*F1*P4*xy2。
        # 权重：Linear(512,64)，64=T2*M8*F1*P4；每点只有1个权重，不乘xy2。
        # 以偏移层为例，W形状[128,512]、bias[128]，所有格子共享它们。
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        # 路线B：value_proj是256->256，先重新组合内容通道；output_proj也是256->256。
        # 它们不会把二维参考点变成内容，也不会增加BEV格子数。
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    # 【训练开始前初始化一次】偏移W=0，bias给不同头方向、不同点半径。
    # 初始权重logits全0，softmax后每来源每头4点各1/4；之后由任务loss学习。
    # 这里初始化的是Linear的参数；forward里预测的sampling_offsets是每次输入的中间结果。
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        # 8个头给8个平面方向；cos/sin组成[8,2]的二维方向。
        # repeat后为[8,T*F,4,2]=[8,2,4,2]；每个来源/尺度都具有初始采样方向。
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        # 4个点的半径分别乘1/2/3/4，避免所有采样点从完全同一位置开始。
        # 这些偏移bias与后续offset/[W,H]配合，表示特征网格尺度的初始局部位移。
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    # 【时间阶段入口】请先记住三个不同用途的对象：
    # query[1,10000,256]：需要更新的当前格子内容。
    # value[2,10000,256]：供读取的两个内容来源；无历史时为None。
    # reference_points[2,10000,1,2]：每来源每格的二维归一化采样中心。
    # query_pos[1,10000,256]是bev_pos展平换轴后的256维位置提示，不是上面的xy。
    # spatial_shapes=[[100,100]]，level_start_index=[0]；它们描述被读取的BEV平面。
    # 原英文docstring是通用模板；本例batch_first=True，实际输入输出按[B,N,C]解释。
    # key/flag以及kwargs传入的key_pos等未在这条实现中用于显式QK点积。
    # 返回的是时间更新后的格子内容；离开本函数后，空间注意力才读取当前图像。
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

        # T1：只有没有历史时才走这里。原query=[1,10000,256]。
        # stack(...,1)新增来源轴：[1,2,10000,256]；reshape合并B*T->[2,10000,256]。
        # 两份内容相同，且复制发生在query+pos之前；这不是两张新图片或真实历史。
        # 有历史时跳过此分支，value已由encoder.py在层循环外构造。
        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)

            # value = torch.cat([query, query], 0)

        # T2：保留本函数最初的query[1,10000,256]作为残差。
        # 后面query=query+pos、query=cat(...)创建新结果，不会把identity变成512维。
        # 最后将采样增量加回这个原始当前格子内容。
        if identity is None:
            identity = query
        if query_pos is not None:
            # T3：当前内容[1,10000,256] + 位置提示[1,10000,256]，结果仍[1,10000,256]。
            # 历史value此处没有显式加pos；它已在历史帧编码中受到位置提示影响，但不存独立pos段。
            # 本次query_pos帮助偏移预测层知道“我现在为哪个格子找信息”；实际去哪里读还要看reference_points。
            query = query + query_pos
        # 兼容[N,B,C]输入的分支；你当前batch_first=True，不执行。
        # 不要看到这两行permute就又把本例query当[10000,1,256]，本例始终先按[1,10000,256]往下读。
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        # 解包shape：bs=1，num_query=10000，embed_dims=256。
        # value的首轴是B*T=2，所以用_忽略；num_value=10000是每来源可读取的位置数。
        # assert检查100*100确实等于10000，并确认这份实现只处理两个来源。
        bs,  num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        # T4：现在变量query将换用途，由“格子内容”变成“采样参数预测输入”。
        # B=1时value[:bs]=value[:1]取历史但保留首轴：[1,10000,256]。
        # 与带pos的当前query沿最后一轴cat：[1,10000,256+256]=[1,10000,512]。
        # 第123格对应query[0,123,:]共512个数：前256历史内容，后256当前内容+位置提示。
        # 注意没有把value整体拼成512；value仍[2,10000,256]，下一行走独立内容路线。
        query = torch.cat([value[:bs], query], -1)
        # T5：路线B准备被采样的内容：[2,10000,256]->[2,10000,256]。
        # 最后256维经共享Linear变换，位置数/来源数没变；没有在此读取当前六相机图片。
        value = self.value_proj(value)

        # 若有无效位置mask，[...,None]把位置mask扩出通道轴，广播把对应256维内容置0。
        # 当前主线通常没有这个mask；masked_fill不改变value形状。
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        # T6：分头：[2,10000,256]->[2,10000,8,32]，-1自动算256/8=32。
        # 只拆分通道轴，元素总数不变。
        # 例如value[0,123,2,:]是历史来源第123位置、第2号头的32维内容。
        value = value.reshape(bs*self.num_bev_queue,
                              num_value, self.num_heads, -1)

        # T7：路线A预测偏移：[1,10000,512]->[1,10000,128]。
        # 这是Linear在每个格子上计算x@W.T+b，输出未经过sigmoid，可正可负，单位按特征网格偏移解释。
        # 下一行view只是把128个已预测数值按“头/来源/尺度/点/xy”分组，不再预测新数值。
        sampling_offsets = self.sampling_offsets(query)
        # 形状：[B,N,M,T,F,P,xy]=[1,10000,8,2,1,4,2]。
        # 两处2不同：中间2是时间来源，最后2是(dx,dy)。
        # 例offset[0,123,2,0,0,1,:]得到2个数：第123格/第2头/历史来源/第1点的偏移。
        # 每格128=8*2*1*4*2；这里的8是头数，不是8个空间坐标。
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
        # T8：预测权重logits，[1,10000,512]->[1,10000,64]。
        # 紧接着view->[1,10000,8,2,4]，最后4暂时合并F*P=1*4。
        # 每点一个分数，所以比偏移少xy这个轴。
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        # 只沿最后F*P轴归一化，本例每个格子/头/来源的4个点权重和为1。
        # 例如历史4点[.1,.2,.3,.4]，当前4点[.4,.3,.2,.1]，两组各自和为1。
        # 没有生成历史0.7/当前0.3这样的来源总权重；两个来源的结果后面用mean平均。
        attention_weights = attention_weights.softmax(-1)

        # 拆回独立尺度/点轴：[1,10000,8,2,4]->[1,10000,8,2,1,4]。
        # 这个view不改权重，仅把最后的4解释为F1*P4。
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        # T9：让两个来源各自调用同一种采样计算，把来源挪到B旁边再合并。
        # 权重：[1,10000,8,2,1,4]->[1,2,10000,8,1,4]->[2,10000,8,1,4]。
        # permute(0,3,1,2,4,5)是“新轴依次取旧第0/3/1/2/4/5轴”，不是改轴长度。
        # contiguous整理内存布局，不改变数据含义；此处临时首轴2对应value的历史/当前来源。
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        # 偏移做相同换轴：[1,10000,8,2,1,4,2]->[1,2,10000,8,1,4,2]
        # -> [2,10000,8,1,4,2]。value/偏移/权重现在都以相同顺序索引两个来源。
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        # T10：本例走二维参考点分支，reference_points=[2,10000,1,2]。
        # 这里是“来源、格子、尺度、xy坐标”，没有256维内容、没有头和采样点轴。
        # 参考由encoder.py生成；该处保留了作者注明的shift_ref_2d别名行为，不能假定两来源一定一移一不移。
        # 本函数仅以实际传入的参考为中心，再叠加学到的局部偏移。
        if reference_points.shape[-1] == 2:
            # spatial_shapes=[[H,W]]=[[100,100]]，偏移按(dx,dy)排列，故取[W,H]。
            # stack结果[1,2]，内容[[100,100]]；不是位置编码，也不是pc_range的米制跨度。
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            # None新增轴，不是取空值；参考[2,N,1,2]->[2,N,1,1,1,2]。
            # 偏移[2,N,8,1,4,2]；除数[1,1,1,1,1,2]，靠广播让8头/4点共用各自参考中心。
            # 最终坐标[2,N,8,1,4,2]，每点含归一化(x,y)。
            # 例参考(.50,.50)，偏移(2,-1)，除[100,100]后读(.52,.49)。
            # 这些位置是实际采样地址；不是把offset当作256维内容直接加到query。
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]

        # 兼容参考框(x,y,w,h)的分支：按框宽高缩放局部偏移；本例不走这里。
        # 它不是“BEV每格4个高度锚点”的分支；高度锚点在空间注意力中是另一个轴。
        elif reference_points.shape[-1] == 4:
            # None新增轴，不是取空值；参考[2,N,1,2]->[2,N,1,1,1,2]。
            # 偏移[2,N,8,1,4,2]；除数[1,1,1,1,1,2]，靠广播让8头/4点共用各自参考中心。
            # 最终坐标[2,N,8,1,4,2]，每点含归一化(x,y)。
            # 例参考(.50,.50)，偏移(2,-1)，除[100,100]后读(.52,.49)。
            # 这些位置是实际采样地址；不是把offset当作256维内容直接加到query。
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        # T11：偏移、坐标、权重已准备完，才开始真正读取value内容。
        # GPU走自定义CUDA核；否则用PyTorch实现，二者遵守相同输入输出轴约定。
        # 下面两个dtype分支实际都选fp32封装，用于稳定大量插值加权求和。
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            # 核输入：value[2,10000,8,32]、位置[2,10000,8,1,4,2]、权重[2,10000,8,1,4]。
            # 对“第123格/第2头/历史”来说：按4坐标双线性插值拿到4份32维向量，再乘4权重求和。
            # 4份32维 -> 1份32维；8头的结果拼接 -> 256维；两个来源各自保留。
            # 因此output=[2,10000,256]，采样点轴通过加权和消失，不是输出又有4个格子。
            # 位置梯度能通过插值传回偏移Linear，所以不需要额外标注“正确偏移”。
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:

            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        # T12：采样后output[2,10000,256]->[10000,256,2]。
        # 将来源轴移到最后，方便下一步恢复B与T并只对T求平均。
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        # [10000,256,2]->[10000,256,1,2]，依次为格子、通道、batch、来源。
        # B=1是本讲解范围；不要把形状解释直接推广成任意B的来源排列保证。
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        # 对最后T轴取均值：(历史采样结果+当前采样结果)/2。
        # [10000,256,1,2]->[10000,256,1]；前面的4点加权和与这里2来源平均是两次不同融合。
        output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        # 恢复batch在前：[10000,256,1]->[1,10000,256]。
        # 此时每个当前格子得到一份融合后的256维采样增量。
        output = output.permute(2, 0, 1)

        # T13：Linear(256,256)融合不同头的内容通道，形状不变。
        # 接着dropout训练时随机屏蔽部分增量，最后加T2保存的原始query残差。
        # 本函数里的512只用于预测参数；返回内容从来不需要保留512维。
        output = self.output_proj(output)

        # 兼容[N,B,C]输入的分支；你当前batch_first=True，不执行。
        # 不要看到这两行permute就又把本例query当[10000,1,256]，本例始终先按[1,10000,256]往下读。
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        # 返回[1,10000,256]，仍是10000格每格256维，更新了内容。
        # 调用者encoder.py接着LN，再进入spatial_cross_attention.py读取当前六相机图片！
        # 第1层的时间输出还没有经过本帧的空间读取；不能把这个return当作整个BEV生成结束。
        return self.dropout(output) + identity
