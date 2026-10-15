# 空间注意力阅读副本：在这里找到“当前图片如何变成BEV内容”。
# 本文件按要求置于anno目录供对照阅读；实际模型仍使用modules目录中的同名实现。
# 原实现相对导入保留，阅读副本不加入训练入口或包初始化。
#
# 【与上一份文件衔接】
# encoder.py/BEVFormerLayer.forward：时间注意力 -> LN -> 本文件空间注意力 -> LN -> FFN -> LN。
# 此时query[1,10000,256]是时间更新后的当前格子内容，需要继续从当前图片取信息。
# 当前图片在VAD.py/extract_img_feat中已走ResNet/FPN：[1,6,256,12,20]。
# VAD_transformer.py/get_bev_features把它展平、加入相机/尺度嵌入并换轴为[6,240,1,256]。
# 这就是本文件收到的key/value，不是上个文件里的历史/当前BEV两来源value！
# 两个forward都叫value，但来源不同：时间value读BEV网格；空间value读当前相机图像特征。
#
# 【与LSS的对应】
# LSS从每个像素出发，预测深度分布后把图像特征lift到3D，再汇聚到BEV。
# 这里从每个BEV格子出发，放4个高度参考点，用标定投到图像，再在投影附近采样。
# 无需先预测逐像素深度概率体；偏移学习“附近哪里更有用”，不是预测每像素深度。
#
# 【本例固定数值与轴】
# B=1；N=10000格(100*100)；C=256通道；6相机；图像位置L=240(12*20)。
# Z=4高度锚点；M=8注意力头、每头32通道；F=1尺度；P=8点/头/相机。
# J是6相机中最多可见格子数，随样本变化，不是固定10000，也不是240。
# 本文件分两层：
#   SpatialCrossAttention：筛选可见格子 -> 按相机打包 -> 调内层 -> 按格子索引写回 -> 多相机平均。
#   MSDeformableAttention3D：query预测偏移/权重 -> 按投影位置读取图像value -> 返回内容。
# 内层临时B'=B*6=6；这个6是相机，不是训练batch变为6，不是时间来源数。
# 每头总8采样点拆为2组*4高度锚点，即每锚2点；不是每锚8点。
# 读完当前图像后返回[1,10000,256]，进入下一编码层的query才携带这一层图像更新。
# 3个编码层结束后才得到本次最终BEV；不要把一个attention的返回当成整个模型输出。
# 下方原英文docstring含通用模板/旧描述，具体轴以下面逐步注释与真实view为准。
# 所有修改仅为注释，保留源码逻辑、docstring、边界分支。


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

    # 【外层装配】build_attention按配置type创建内层MSDeformableAttention3D实例。
    # 外层负责“相机调度/写回BEV”，内层负责“一个相机内的偏移预测和采样”。
    # 外层output_proj是256->256；配置最终决定内层num_levels=1、num_points=8。
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
        # 构造出来的是可调用的内层对象；后面self.deformable_attention(...)会经nn.Module.__call__进入其forward。
        # 查本文件下半部分的MSDeformableAttention3D，不要再去找名叫deformable_attention的普通函数。
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    # 初始化外层通道投影；名字是单数init_weight，与内层init_weights不同。
    # 不在这里生成BEV，也不在这里加载当前图片；forward才按每组数据运行。
    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    # 【S阶段入口：第一次真正接上当前图片内容】
    # query=[1,10000,256]：经过时间注意力+LN的格子查询，决定要为哪些BEV格子取信息。
    # key/value=[6,240,1,256]：当前6相机图像特征，240个位置/相机，256维/位置。
    # reference_points_cam=[6,1,10000,4,2]：各格子的4高度参考点在每相机中的归一化(u,v)。
    # bev_mask=[6,1,10000,4]：上述每高度参考点是否正深度且投影在图内，True为几何可见。
    # 这两个对象在encoder.py/point_sampling按lidar2img标定计算好；不是本文件预测的GT。
    # 本主线query_pos=None：BEV位置提示已经用于上一时间注意力，不会在这里必定再显式加一次。
    # 原英文模板写[N,B,C]，但本调用的query实际是[B,N,C]；以这里和真实调用为准。
    # @force_fp32对指定输入执行精度管理；不是改变轴顺序，也不负责图像特征提取。
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

        # S1：通用接口的缺省回退；当前主线key/value都已有六相机特征，跳过两个None分支。
        # 图片特征来源在本文件顶部有完整调用路径，不能因为这里没有CNN就认为模型没读图片。
        if key is None:
            key = query
        if value is None:
            value = key

        # S2：主线residual=None，保存原query[1,10000,256]供最后残差相加。
        # slots同形状、全0，用作当前图像信息写回BEV的累加槽；它不是最终BEV初始化权重。
        # 本源码inp_residual/slots只在此分支创建；此阅读按主线None解释，不额外修改非None边界行为。
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        # 本主线通常query_pos=None，此分支跳过。
        # 上一时间注意力显式使用bev_pos，这里不要脑补必定又加一次。
        # 若别的调用传pos，仍为256维逐元素相加，不是拼接512。
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        # S3：这里局部D=4，指每柱4个高度锚点，绝不是通道256或每头32！
        # reference_points_cam=[6,1,10000,4,2]：每相机每格4个不同高度点，各投影到一个二维位置。
        D = reference_points_cam.size(3)
        # 为每相机建立“哪些BEV格子能看到它”的索引表，而不是让10000格都读所有相机。
        # 例indexes[0]=[2,7,123,...]表示相机0可见原BEV第2/7/123号格。
        # 索引里的数字是原BEV格编号；列表长度是该相机可见格子数，二者不要混淆。
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            # mask_per_img=[1,10000,4]；[0]取本例唯一batch->[10000,4]。
            # sum(-1)数每格可见高度数->[10000]；nonzero找至少一个高度可见的格子->[n_i,1]。
            # squeeze(-1)去掉长度1轴->[n_i]，得到整数格子索引，不是256维内容。
            # 这里按首batch可见性建立索引，讲解只固定B=1，不把它误述为任意B独立筛选。
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        # J=max(n_0,...,n_5)，即六相机可见格子数的最大值。
        # 例如n_i分别2000/1800/...，J取最大，不是把6相机可见格子求和。
        # J随样本标定/范围变化，具体数值不能在未打印时伪造。
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        # S4：准备按相机打包的缓存query=[1,6,J,256]，投影参考=[1,6,J,4,2]。
        # 各相机长度不同，补0到同一J方便并行；只复制实际可见的n_i项。
        # 这新增的是相机轴，既不是时间轴，也不是注意力头轴。
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                # 用同一索引表从原query选格子、从投影参考选坐标，保证内容与几何地址一一对应。
                # 若indexes[i][2]=123，缓存queries_rebatch[0,i,2]对应原query[0,123]。
                # 对应reference_points_rebatch[0,i,2]为该格子在相机i中的4个投影点[4,2]。
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        # key输入[6,240,1,256]，解包得到num_cams=6、l=240、bs=1、embed_dims=256。
        # 这里l是小写L，表示图像特征位置数12*20，不是数字1。
        num_cams, l, bs, embed_dims = key.shape

        # S5：key/value都[6,240,1,256]->[1,6,240,256]->[6,240,256]。
        # permute(2,0,1,3)把batch移到前面；reshape合并真实batch和相机轴。
        # 临时首轴6=B*相机数；底层就能把六相机当六份独立特征图并行采样。
        # 每图240个位置保留；reshape没有把240个图像位置变成10000个BEV格子。
        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        # S6：进入本文件下半部分MSDeformableAttention3D.forward，完成真正图像内容采样。
        # query：[1,6,J,256]->[6,J,256]；value：[6,240,256]；参考：[6,J,4,2]。
        # query的J是“为多少BEV格子找信息”；value的240是“每相机有多少图像位置可读”，不要求相等。
        # 内层输出[6,J,256]，外层立即view还原[1,6,J,256]。
        # 只看某格子：它拿自己的查询预测采样偏移/权重，再从当前相机图像240个位置所在的12x20网格插值读取。
        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                # S7：采样结果写回原BEV格子，压缩索引反向对应恢复。
                # 例缓存第2项对应原格123，则把queries[0,i,2]加到slots[0,123]。
                # 同一格若在两相机可见，两份256维图像内容先累加；slots形状一直[1,10000,256]。
                # 缓存中补齐的无效尾部不写回，因为只取:len(index_query_per_img)。
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # S8：计算每格应按多少个相机平均，不是采样点权重，也不是可见高度总数。
        # bev_mask[6,1,10000,4]沿Z求和再>0 -> [6,1,10000]，每相机/格为可见布尔值。
        count = bev_mask.sum(-1) > 0
        # [6,1,10000]->[1,10000,6]，最后沿相机求和->[1,10000]。
        # 某格在两个相机可见，count就是2；每相机可见几个高度不会让该相机重复计数。
        count = count.permute(1, 2, 0).sum(-1)
        # 完全不可见格子的计数0改成1避免除0，不会把它凭空变成可见格子。
        # 其slots采样累加仍为0；后面投影bias若训练后非0，不能保证投影增量严格恒0。
        count = torch.clamp(count, min=1.0)
        # [1,10000,256]除以count[...,None]=[1,10000,1]，广播到每格全部256通道。
        # 若某格两相机内容分别a、b，则此时该格=(a+b)/2；仅一个相机则直接使用它。
        # 这次平均的是相机，前一文件平均的是时间来源，两次融合不要混为一谈。
        slots = slots / count[..., None]
        # S9：Linear256->256重新组合各头内容，输出仍[1,10000,256]。
        # 这些是从本帧图像读取的增量；下一行dropout后加时间阶段留下的query残差。
        slots = self.output_proj(slots)

        # 返回更新后的当前格子：[1,10000,256]，已融合历史路径与本层当前图片信息。
        # 离开后encoder.py继续LN/FFN/LN，作为下一编码层query；整个编码器结束才输出最终BEV。
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

    # 【内层装配】本例输入查询仍256维，没有历史256拼接，因此Linear输入不是512。
    # 偏移Linear(256,128)：8头*1尺度*8点*xy2；权重Linear(256,64)：8头*1尺度*8点。
    # 与时间注意力恰好都是128/64，但因子不同：这里没有两个时间来源，采样点从4变为8。
    # 构造函数默认num_levels=4，Tiny配置传1；下面形状均按实际配置1解释。
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
        # 内层不负责输出通道投影/残差，外层SpatialCrossAttention统一处理。
        # 保留原源码属性与初始化调用；不要在理解时额外想象内层也有一次256->256输出融合。
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        # 每头C/M=256/8=32，检查32是否为2的幂，利于CUDA实现效率。
        # 只检查配置，不会把32变成空间坐标或采样点数。
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
        # 偏移Linear(256,128)；权重Linear(256,64)；value投影Linear(256,256)。
        # 输入query256决定“去哪里、占多少”，图像value256是“读取什么”，两条路线分工与时间注意力相同。
        # 与上一文件不同：输入没有拼历史256，所以不出现512维预测输入。
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    # 初始化偏移/权重/内容投影参数，不是执行一次注意力采样。
    # offset的W初始0、bias按头方向和点半径设置；权重logits初始0，每头8点各1/8。
    # 训练时通过检测/地图/规划等任务的反传学习采样位置，不需要单独标注采样偏移。
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        # 8头8方向；grid_init按尺度1与点数8扩展为[8,1,8,2]。
        # 后续各点乘半径1..8形成初始采样图案；学习后偏移可随query改变。
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

    # 【I阶段入口：单个相机内采样，6相机并行】
    # bs在本函数是临时B'=6，不再是外层真实batch=1；num_query=J而非10000。
    # query=[6,J,256]，value=[6,240,256]，reference_points=[6,J,4,2]。
    # spatial_shapes=[[12,20]]描述被读取的图像特征网格，绝不是时间注意力的[[100,100]]！
    # 读图像value输出[6,J,256]，随后返回外层写回10000个格子。
    # 这里reference_points的4轴是高度锚点Z，不是尺度数；末轴2才是图像平面(u,v)。
    # key/identity保留通用接口；这条采样实现不执行全量QK点积，也不在此加入残差。
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

        # I1：主线传入value=[6,240,256]，不会回退到query。
        # 下面identity虽赋值，但本函数return没有加identity；残差在外层统一完成。
        if value is None:
            value = query
        if identity is None:
            identity = query
        # 内层主线没有单独传query_pos，保持输入query[6,J,256]。
        # BEV格子的几何位置通过reference_points传入，而不是通过把4高度坐标拼到query实现。
        if query_pos is not None:
            query = query + query_pos

        # 内层本例batch_first=True，不执行换轴；继续按[临时B,位置,C]解释。
        # 这里临时B=6，与外层真实B=1有区别。
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        # I2：bs=6，num_query=J；从value读取num_value=240。
        # assert检查spatial_shapes里的12*20=240，描述的是被采样图片，不是查询BEV的100*100。
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # I3：图像内容[6,240,256]经过共享Linear256->256，形状不变。
        # 每个图像特征位置的通道重新组合，随后拆成8头各32维。
        value = self.value_proj(value)
        # 有位置mask则沿通道广播把无效图像value置0；该分支不会改变位置/通道数量。
        # 外层当前调用未传这个mask；不要把它与筛可见BEV格子的bev_mask混淆。
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        # [6,240,256]->[6,240,8,32]，因为256=8*32。
        # 例value[0,10,2,:]是相机0图像第10位置、第2头的32维内容。
        # 图像位置的二维布局由spatial_shapes=[[12,20]]告诉底层，而非在此保留H/W轴。
        value = value.view(bs, num_value, self.num_heads, -1)
        # I4：query[6,J,256]->Linear->[6,J,128]->view->[6,J,8,1,8,2]。
        # 轴为：相机临时batch、查询格子、头、尺度、总采样点、xy偏移。
        # 每查询128=8头*1尺度*8点*2坐标；两处8分别是头数和每头采样点数。
        # 例offset[0,k,2,0,5,:]是相机0、第k个打包格子、第2头、第5点的(dx,dy)。
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        # I5：query[6,J,256]->Linear->[6,J,64]->view->[6,J,8,8]。
        # 最后8=F1*P8，每点一个权重，不乘xy2。
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        # 沿最后尺度*点数softmax：每个相机/格子/头的8点权重和为1。
        # 这里是每相机内采样点的权重；多相机之间的平均在外层count处进行。
        attention_weights = attention_weights.softmax(-1)

        # 拆回独立尺度轴：[6,J,8,8]->[6,J,8,1,8]。
        # 前一个8是头，后一个8是点；保持数值不变，供底层与采样位置逐项对应。
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        # I6：主线投影参考末轴2为图像(u,v)，进入此分支。
        # 下面原英文多行字符串对总点数的描述容易误导；实际代码总P=8，再拆成2*4高度。
        # 按实际view理解：每头每相机总8点，而不是4高度*每高度8点。
        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            # 图像特征尺寸[H,W]=[12,20]，所以归一化器是[[W,H]]=[[20,12]]。
            # 例如dx=1表示图像特征横向一格，归一化位移1/20；dy=1为纵向1/12。
            # 不能套用时间注意力的[100,100]：两者被采样的特征图不同。
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            # reference_points=[6,J,4,2]，解包得到Z=4、xy=2。
            # 这4个中心来自同一BEV柱不同高度点的图像投影，不是4张图片或4个尺度。
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            # [6,J,4,2]->[6,J,1,1,1,4,2]。
            # 新增头/尺度/每锚采样组的长度1轴，为后面8头、2组偏移广播准备。
            # None只加轴，不会预测更多参考坐标；同一几何参考可供各头读取不同内容。
            reference_points = reference_points[:, :, None, None, None, :, :]
            # 先把网格单位偏移除以[20,12]，得到归一化图像坐标位移。
            # 形状仍[6,J,8,1,8,2]，只有数值单位发生变化。
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            # I7：把总8点拆成“每锚2点组、4高度锚点”：
            # [6,J,8,1,8,2]->[6,J,8,1,2,4,2]，8=2*4。
            # 固定相机/格子/头后，原点索引0..3对应第1组4高度，4..7对应第2组4高度。
            # 例原第5点使用高度索引1的投影中心；它不是额外再给全部4锚点各采样一遍。
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            # 参考[6,J,1,1,1,4,2]+偏移[6,J,8,1,2,4,2]，广播得[6,J,8,1,2,4,2]。
            # 例某高度投影(.50,.50)，某点offset=(1,-1)，则位置=(.55,.4167)，因为除数[20,12]。
            # 加法得到实际采样坐标；并没有把二维坐标加到256维语义内容上。
            sampling_locations = reference_points + sampling_offsets
            # 注意同名局部变量num_points此时=2(每锚点组数)，self.num_points仍=8(总点数)。
            # assert验证num_all_points8=num_points2*num_Z_anchors4。
            # 同名变量在不同上下文含义不同，要看赋值来源，不能一律把num_points读成8。
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            # 再把2组*4高度合回总8点：[6,J,8,1,2,4,2]->[6,J,8,1,8,2]。
            # 底层只需最终8个采样坐标，不需要单独知道4个高度来自哪里。
            # 权重仍[6,J,8,1,8]，每个点的权重与这个合并后的点顺序对应。
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        # 本空间采样类不支持参考框分支，原代码assert False；主线二维投影不进入。
        # 不要把reference_points中的高度轴4误当成末轴4触发此分支，实际末轴一直是2。
        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        # I8：直到现在才进入插值采样核；此前只是决定怎么读，还没取图像内容。
        # GPU/CUDA或PyTorch两条实现同样接收value、采样坐标、权重；精度分支都选fp32封装。
        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            # 核输入：图像value[6,240,8,32]、坐标[6,J,8,1,8,2]、权重[6,J,8,1,8]。
            # 对“相机0/某BEV格/第2头”：从12x20网格按8坐标双线性插值读取8份32维内容。
            # 这8份分别乘对应权重后求和 -> 一份32维；8头拼接 -> 256维。
            # 因此output=[6,J,256]；8采样点通过加权和消失，J个待更新BEV格子保留。
            # 没有先计算J*240的完整QK点积矩阵；key参数不会在此参与普通注意力打分。
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        # 内层本例batch_first=True，不执行换轴；继续按[临时B,位置,C]解释。
        # 这里临时B=6，与外层真实B=1有区别。
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        # I9：返回[6,J,256]，包含来自本帧各相机图像的采样内容。
        # 回到外层S6后还原[1,6,J,256]，S7按原格子编号写回，S8多相机平均，S9投影+残差。
        # 只有完成这些外层步骤，才得到保持全BEV布局的[1,10000,256]。
        return output
