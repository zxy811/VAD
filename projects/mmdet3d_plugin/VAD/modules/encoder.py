# 【生成BEV的主入口】上层VAD_transformer.py的get_bev_features调用BEVFormerEncoder.forward。
# 建议顺序：forward看整个路线 -> get_reference_points看格子位置 -> point_sampling看投影
# -> BEVFormerLayer.forward看时间/空间/FFN -> temporal_self_attention.py / spatial_cross_attention.py。
# 返回[1,10000,256]当前BEV给上层；历史only_bev在Head提前return，当前帧继续检测/地图解码。
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
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    # 【装配编码器】父类按配置建3个BEVFormerLayer，保存pc_range与每柱高度锚点数4。
    # 本例return_intermediate=False，输出最后层[1,10000,256]；不是返回三帧BEV。
    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    # 【为固定BEV格子确定几何位置】这是staticmethod，不需要self，不访问学习参数。
    # 三维参考[1,4,10000,3]：100x100格、每格4高度；二维参考[1,10000,1,2]用于时间BEV采样。
    # 这些初始网格坐标归一化到0..1，由网格/高度范围确定，不是随机查询表，也不是GT车辆位置。
    # 本例物理x跨度30米、y跨度60米、z跨度4米，具体反归一化在point_sampling。
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        # 【先决定要找哪里】三维参考不是学习结果：xy取每格中心，z取柱内4个高度。
        # 本例每格宽0.3m、长0.6m；归一化(x,y)=(列+0.5,行+0.5)/100。
        # Z传入物理高度跨度4m；z归一化后在point_sampling还原成[-1.5,-0.5,0.5,1.5]m。
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            # 把同位置的x/y/z合并：最后一轴3表示坐标分量，非颜色。
            # 随后的permute/flatten把100x100压为10000，repeat复制到batch，最终[1,4,10000,3]。
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        # 时间注意力读的是BEV平面，故只需xy：[1,10000,1,2]。
        # 中间1是特征尺度数量，不是z高度；与上面4高度投图像的用途不同。
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    # 【空间位置投影到相机】输入归一化3D参考[1,4,10000,3]，先变当前LiDAR米制xyz。
    # 读取每样本六个lidar2img矩阵[1,6,4,4]，齐次坐标乘矩阵，除深度得到像素，再除图像宽高归一化。
    # 输出reference_points_cam=[6,1,10000,4,2]、bev_mask=[6,1,10000,4]；mask=True表示该相机可见该锚点。
    # 这是“BEV格子投影到图片找信息”，与LSS“像素预测深度后lift到BEV”路径不同。
    # 这里局部变量D是4个高度锚点，不是256特征维；对照下面赋值避免混淆。
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):

        # 标定从img_metas读取，而非从query预测；它已由数据集组合外参、内参、图像缩放。
        # 每相机矩阵4x4，合并后[1,6,4,4]，把当前LiDAR空间参考点投向图像。
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        # clone防止反归一化直接修改调用者ref_3d。
        # 接下来xyz分别乘30/60/4并加-15/-30/-2：0..1比例变成米制坐标。
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        # xyz补齐次分量1：[1,4,10000,3]->[1,4,10000,4]。
        # 这样4x4矩阵同时表达旋转和平移；这一步没有生成256维特征。
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        # 整理为[4,1,6,10000,4,1]，把同一空间点复制给6相机。
        # lidar2img随后扩为[4,1,6,10000,4,4]，批量matmul作用于最后两个轴。
        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        # 矩阵乘后[4,1,6,10000,4]；前两项为含深度比例的像素坐标。
        # 先要求depth>0，再xy/depth得真实像素；不是直接把前三项当(u,v,depth)。
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # 除图像宽/高得到0..1采样坐标；本实现取首样本首相机img_shape，依赖图像尺寸一致。
        # 后面mask同时检查正深度和0<u,v<1；表示投影可见性，并非标注或图像内容遮挡检测。
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        # 把相机轴放前面：坐标[6,1,10000,4,2]，可见mask[6,1,10000,4]。
        # 空间注意力按相机循环用这两个对象，坐标与mask通过同一个格子/高度索引对应。
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    # 【当前帧BEV编码总流程】输入query=[10000,1,256]、图像key/value=[6,240,1,256]，bev_pos同查询布局。
    # 1.固定格子参考点；2.三维参考投到图像；3.平移参考点及组织历史value；4.执行3层时间/空间/FFN。
    # prev_bev若有在上层已经旋转，并统一[10000,1,256]，这里组织成[2,10000,256]（历史+当前查询）。
    # 返回[1,10000,256]给上层；不是检测框或最终地图点。当前query第一层是初始内容，后层才逐渐含当前图像。
    # 原作者特意保留shift_ref_2d别名问题以复现，注释解释实际行为，不在这里暗改算法。
    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        # 【作者保留的别名行为】这里赋值没有复制，两个变量指向同一Tensor。
        # 下一行+=会同时改变ref_2d；因此实际两份参考均受shift影响。
        # 不能把这段讲成“一个只移历史、另一个严格不移”；按原代码保留，不改实验行为。
        shift_ref_2d = ref_2d  # .clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        # [10000,1,256]->[1,10000,256]；只是轴顺序切换，没增加/减少信息。
        # BEVFormerLayer内部batch_first=True，位置特征也同样换轴，才能逐元素相加。
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            # 历史内容[1,10000,256]和本次初始查询[1,10000,256]沿新来源轴stack。
            # [1,2,10000,256]->[2,10000,256]；2是两个时间来源，不是训练batch变2。
            # 这里历史特征已由上层旋转，平移体现在参考坐标shift中，非在这里搬移整张特征图。
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        # 【3层反复完善同一BEV】每层时间采样->图像采样->FFN，输出仍[1,10000,256]。
        # 下一层的query用新output；传入的prev_bev则保持本次编码开始时组织的两个来源。
        # 这3层是网络深度，不是三帧训练队列；三帧由上层逐帧调用本编码器。
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    # 【建立一个BEV更新层】复用MyCustomBaseTransformerLayer.__init__构建时间/空间注意力、3个LN和FFN。
    # 本例顺序时间注意力->LN->空间注意力->LN->FFN->LN；FFN 256->512->256。
    # attention内部预测采样坐标/权重，Layer容器负责按顺序调用并传递位置/参考点/mask。
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    # 【10000个格子更新一次】query=[1,10000,256]（本例batch_first=True）。
    # 时间阶段：Q为当前查询加bev_pos，V组织历史/当前内容；空间阶段：Q是更新后的格子，V是六相机特征。
    # 这里bev_pos显式传入时间注意力；空间调用传query_pos变量（默认None），不能说同一bev_pos必定又加一次。
    # 每个注意力返回都有残差；norm/FFN保持[1,10000,256]，最终成为下一编码层输入。
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            # 【先融合历史】调用attentions[0]（TemporalSelfAttention）。
            # query是当前格子内容；prev_bev作value；bev_pos提供位置提示，ref_2d提供采样中心。
            # 空间参考与256维位置提示各司其职，不可互换。
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            # 【再读取当前图片】调用attentions[1]（SpatialCrossAttention）。
            # value是六相机特征；reference_points_cam提供投影中心；bev_mask筛可见格子。
            # 调用结束内容仍[1,10000,256]，但已融合当前图像语义。
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            # 每个格子独立通过共享MLP 256->512->256并带残差。
            # 注意力负责跨位置取信息，FFN负责组合本位置通道；这里512是隐藏宽度，不是内容/pos拼接。
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
