# 【Python和CUDA之间的接口】上层已经算好value、采样坐标和权重，这里不再学习/预测offset。
# forward调用编译好的mmcv._ext，在特征图上按坐标插值、乘权重并求和；backward计算三类输入梯度。
# 举例检测：value=[1,10000,8,32]，locations=[1,300,8,1,4,2]，weights=[1,300,8,1,4]，
# 输出[1,300,256]。时间/空间注意力改变前面batch/query/value数，但核遵守同样轴约定。
# 这里是torch.autograd.Function，不是nn.Module；通过Class.apply调用，forward/backward由自动求导框架配对。
# 本文件只展示Python封装，CUDA内核位于安装的MMCV扩展中；不能说Python注释已展开所有CUDA实现。
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

import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class MultiScaleDeformableAttnFunction_fp16(Function):

    # 【执行采样核，不预测坐标】ctx是自动求导保存上下文，不是模型self；staticmethod由.apply触发。
    # value=[Bv,Nvalue,8,32]、locations=[Bv,Q,8,F,Pts,2]、weights=[Bv,Q,8,F,Pts]，输出[Bv,Q,256]。
    # 例如检测Bv=1、Nvalue=10000、Q=300、F=1、Pts=4；时间Bv=2，图像采样Bv=6。
    # 按照已给坐标双线性插值、权重乘和；ctx.save_for_backward保存反传需要的值，不是checkpoint文件。
    # custom_fwd设置AMP上下文转换精度；上层可选择fp32版本以稳定求和。
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    # 【梯度通过采样回到偏移预测层】grad_output与forward输出同shape，CUDA计算value/坐标/权重的梯度。
    # 返回项顺序对应forward输入：grad_value、None、None、grad_sampling_loc、grad_attn_weight、None。
    # shape/level索引/im2col_step是元数据，不求梯度；采样坐标梯度继续流向offset Linear，权重梯度流向weight Linear。
    # once_differentiable表示该自定义反传不支持继续求高阶导；本函数没有optimizer.step。
    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        # 6个返回值与forward的6个非ctx入参一一对应。
        # None表示该元数据不求梯度；采样位置有梯度，所以坐标预测层可被任务loss训练。
        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformableAttnFunction_fp32(Function):

    # 【执行采样核，不预测坐标】ctx是自动求导保存上下文，不是模型self；staticmethod由.apply触发。
    # value=[Bv,Nvalue,8,32]、locations=[Bv,Q,8,F,Pts,2]、weights=[Bv,Q,8,F,Pts]，输出[Bv,Q,256]。
    # 例如检测Bv=1、Nvalue=10000、Q=300、F=1、Pts=4；时间Bv=2，图像采样Bv=6。
    # 按照已给坐标双线性插值、权重乘和；ctx.save_for_backward保存反传需要的值，不是checkpoint文件。
    # custom_fwd设置AMP上下文转换精度；上层可选择fp32版本以稳定求和。
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    # 【梯度通过采样回到偏移预测层】grad_output与forward输出同shape，CUDA计算value/坐标/权重的梯度。
    # 返回项顺序对应forward输入：grad_value、None、None、grad_sampling_loc、grad_attn_weight、None。
    # shape/level索引/im2col_step是元数据，不求梯度；采样坐标梯度继续流向offset Linear，权重梯度流向weight Linear。
    # once_differentiable表示该自定义反传不支持继续求高阶导；本函数没有optimizer.step。
    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        # 6个返回值与forward的6个非ctx入参一一对应。
        # None表示该元数据不求梯度；采样位置有梯度，所以坐标预测层可被任务loss训练。
        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None
