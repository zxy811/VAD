# ============================================================================
# VAD Tiny 端到端配置：中文阅读注释版（只增加注释，保留原配置语义）
# ============================================================================
# 这不是网络forward源码，而是MMCV读取的Python配置：
# tools/train.py -> Config.fromfile -> build_model(cfg.model) -> VAD / VADHead。
# mmdet_train.py读取cfg.data/optimizer/runner等，创建loader/optimizer/Runner/Hooks。
# type="..."由Registry查找类；普通dict不会因为写了type就自动创建对象。
# Python语法：dict(a=1)等价于{'a':1}；[...]是列表；(...)常为元组/调用参数；
# (3,)是单元素元组；None表示不启用相应可选项；**img_norm_cfg展开字典命名参数。
# model中每一层缩进表示“属于哪个父字典”，不代表神经网络只按书写顺序前向。
# B=batch，D=256；A=300目标槽位；N=100地图实例；S=20点；K=6运动模式；T=6未来步。
# 候选槽位数不是GT数量；GT、位置坐标、内部特征、可训练参数必须分别理解。
#
# 实际模型树：
# VAD
#   |- ResNet50 -> FPN：图像特征[B,6,256,Hf,Wf]
#   `- VADHead
#       |- VADPerceptionTransformer
#       |   |- BEV encoder：历史/图像采样 -> [B,10000,256]
#       |   |- 检测decoder -> [B,300,256] / 类别与框
#       |   `- 地图decoder -> [B,100,20,256] / 类别与点
#       |- motion_decoder：1800运动查询之间交互
#       |- LaneNet + motion_map_decoder：运动读取地图
#       |- 目标位移[B,300,6,6,2] + 模式分数[B,300,6]
#       |- ego_agent_decoder -> ego_map_decoder：自车顺序读取动态/静态特征
#       `- 规划MLP：拼接为512维 -> 自车位移[B,3,6,2]
#
# 网络嵌入、Linear、attention、FFN等参数在模型构建后训练；
# 此文件的整数、阈值、学习率和loss系数是超参数，不由反向传播自动更新。
# 可变形attention不能套普通QK^T公式；普通注意力的Q/K/V表述均省略投影偏置。
# 配置可以有兼容字段/零权重分支：写在字典里不等于有效执行了相应训练监督。
# ============================================================================

# 【1 配置继承】MMCV Config.fromfile读取这两个基础配置，再合并本文件中的覆盖项。
# 相对路径以当前配置文件目录为基准；不是Python的类继承。
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
# 【2 插件注册】True让tools/train.py导入plugin_dir中的包，触发注册装饰器。
plugin = True
# 导入后type="VAD"、"VADHead"等字符串才能映射到自定义类。
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# 【3 统一空间范围】[xmin,ymin,zmin,xmax,ymax,zmax]，单位米。
# x跨度30m，y跨度60m；用于BEV参考点、坐标归一化、GT过滤及地图截取。
# 这是几何范围，不代表启用了LiDAR输入；修改时需同步相关模块范围。
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
# 框架兼容的体素尺寸配置；当前相机VAD不因此新增点云体素编码分支。
voxel_size = [0.15, 0.15, 4]

# 【4 图像预处理】当前为RGB均值/标准差，NormalizeMultiviewImage通过**展开。
# 复现旧公开VAD权重需匹配原训练预处理，参见docs/train_eval.md：
# mean=[103.530,116.280,123.675], std=[1,1,1], to_rgb=False。
# 这里保留原配置，不自动改预处理或替换权重。
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
# 【5 检测类别】列表顺序就是类别编号，数据标签、输出与评价必须一致。
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# len(list)得到类别数，此处为10，不是目标数量。
num_classes = len(class_names)

# map has classes: divider, ped_crossing, boundary
# 【6 地图类别】divider分隔线、ped_crossing人行横道轮廓、boundary道路边界。
map_classes = ['divider', 'ped_crossing', 'boundary']
# 100个地图实例候选槽位；不代表每帧一定有100条GT线。
map_num_vec = 100
# 每个GT地图实例等弧长采样20点；GT实例数量随样本变化。
map_fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
# 每个预测实例20点，所以地图点查询数量=100*20=2000。
map_fixed_ptsnum_per_pred_line = 20
# 地图评估使用相应固定GT采样策略；这是评价设置，不是网络参数。
map_eval_use_same_gt_sample_num_flag = True
# 地图类别数量为3，与地图实例数量100、每实例点数20分开理解。
map_num_classes = len(map_classes)

# 【7 数据/评估输入模态声明】下面的开关是配置元数据，实际输入还要看pipeline与模型。
input_modality = dict(
    # False：本配置不建立LiDAR特征分支；测试pipeline仍可能读点云用于辅助数据处理。
    use_lidar=False,
    # True：主要感知输入来自摄像头。
    use_camera=True,
    # False：不使用雷达感知输入。
    use_radar=False,
    # False：不把现成HD地图作为感知输入；训练仍有向量地图GT监督。
    use_map=False,
    # nuScenes模态/评估标记；不会凭这个字段自动创建外部特征网络。
    use_external=True)

# 【8 公共维度】D=256。下划线只是变量命名；这些整数不是可训练参数。
_dim_ = 256
# //是整数除法；128维行嵌入+128维列嵌入构成256维BEV位置编码。
_pos_dim_ = _dim_//2
# FFN隐藏维度512；通常是256->512->256，不是最终特征变为512。
_ffn_dim_ = _dim_*2
# FPN输出1个图像特征尺度；与BEV网格大小和解码层数不同。
_num_levels_ = 1
# BEV高100格；y跨度60m，因此每格对应约0.6m。
bev_h_ = 100
# BEV宽100格；x跨度30m，因此每格对应约0.3m。总格数10000。
bev_w_ = 100
# 每个训练样本包含3帧图像；batch后[B,3,6,3,H,W]。
# VAD.forward_train用前两帧生成历史BEV摘要（no_grad），最后一帧计算训练loss。
queue_length = 3 # each sequence contains `queue_length` frames.
# 完整遍历训练数据60轮；不是60个batch。e2e表示主任务联合训练。
total_epochs = 60

# 【9 模型构造字典】这里model是说明书，不是nn.Module。
# tools/train.py的build_model(cfg.model)才创建VAD对象；构造函数继续创建backbone/neck/head。
model = dict(
    # 注册类在projects/mmdet3d_plugin/VAD/VAD.py；VAD继承MVXTwoStageDetector。
    type='VAD',
    # 训练图像应用GridMask增强；不会创建额外的地图网格输出。
    use_grid_mask=True,
    # 测试时跨帧保存prev_bev；切换scene时重置历史。
    video_test_mode=True,
    # 图像骨干预训练来源，只初始化ResNet，不是完整VAD训练权重。
    pretrained=dict(img='torchvision://resnet50'),
    # 【9.1 图像骨干】外部MMDetection ResNet；每个摄像头使用同一套骨干参数。
    # 输入[B,6,3,H,W]通常先合并为[B*6,3,H,W]提取图像特征。
    img_backbone=dict(
        type='ResNet',
        # ResNet50的深度型号，不是50个Transformer层。
        depth=50,
        # 骨干有4个残差stage，stage索引0~3。
        num_stages=4,
        # (3,)是一元素元组，逗号不能省；0起始编号3表示输出最后一个残差stage。
        out_indices=(3,),
        # 按ResNet实现冻结stem及第一残差stage，其余可训练层继续学习。
        frozen_stages=1,
        # BatchNorm设置；requires_grad=False关闭BN仿射参数的梯度更新。
        # False关闭本BN仿射参数梯度，其他未冻结网络参数仍训练。
        norm_cfg=dict(type='BN', requires_grad=False),
        # 训练时BN仍用eval模式，固定running均值/方差；不是整个模型停止训练。
        norm_eval=True,
        # 外部ResNet的实现风格参数，按pytorch方式布置相应卷积stride。
        style='pytorch'),
    # 【9.2 FPN】外部FPN将最后stage的2048通道变为256，当前仅输出1个尺度。
    # 输出可记为[B,6,256,Hf,Wf]，具体空间尺寸取决于缩放/padding后的图像。
    img_neck=dict(
        type='FPN',
        # 输入通道列表/数值，应与前一级模块输出匹配。
        in_channels=[2048],
        # 输出通道256。
        out_channels=_dim_,
        # FPN从输入列表的第0个尺度开始处理；本例输入列表只有一个尺度。
        start_level=0,
        # 需要额外输出尺度时选其卷积位置；本例num_outs=1不额外增加尺度。
        add_extra_convs='on_output',
        # FPN输出尺度数1；即使配置add_extra_convs也不意味着本例额外增加尺度。
        num_outs=_num_levels_,
        # 额外尺度卷积前ReLU的选项；本例没有新增输出尺度。
        relu_before_extra_convs=True),
    # 【9.3 多任务Head】名字沿用点云检测框架，但当前传入的是图像特征。
    # VADHead负责地图、检测、运动、自车规划的前向和loss；实现在VAD/VAD_head.py。
    # 默认参数（本文件未显式写）：fut_mode=6、fut_ts=6、ego_fut_mode=3。
    pts_bbox_head=dict(
        type='VADHead',
        # 目标运动读取地图时，地图sigmoid最大类别分数>0.5才保留整个实例。
        map_thresh=0.5,
        # 运动->地图交互的距离mask阈值；在归一化xy上计算，0.2不是0.2米。
        dis_thresh=0.2,
        # 运动读取地图时使用地图代表坐标减目标坐标，编码相对位置。
        pe_normalization=True,
        # 记录总epoch，供可选训练调度使用。
        tot_epoch=total_epochs,
        # False关闭运动loss预热；与下方优化器学习率warmup是两个不同机制。
        use_traj_lr_warmup=False,
        # 自车规划读取目标/地图时的查询阈值；0通常保留所有有限logit候选。
        query_thresh=0.0,
        # 自车规划查询筛选不强制追加占位项；运动->地图代码另有固定占位。
        query_use_fix_pad=False,
        # None表示自车初始查询来自可训练Embedding(1,256)，不编码历史轨迹。
        ego_his_encoder=None,
        # None表示规划输出头不拼接额外自车状态；BEV仍可使用CAN bus，不能说全网络无状态输入。
        ego_lcf_feat_idx=None,
        # 训练目标轨迹的时间步限制；本配置6步都允许，仍需乘各样本GT有效mask。
        valid_fut_ts=6,
        # 【交互A 自车->目标】Q=自车内容+位置；K=目标运动特征+位置；V=目标运动特征。
        # 目标先融合6个模式：[B,300,6,512]->[B,300,3072]->[B,300,256]。
        # 每头Q[B,1,32]、K/V[B,A,32]，权重[B,1,A]；输出[B,1,256]。
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            # 此模块重复的Transformer层数；不是注意力头数或未来时间步数。
            num_layers=1,
            # True保留每层结果供中间监督；False只返回该解码器最终结果。
            return_intermediate=False,
            # 一个Transformer层的结构说明；由上面的num_layers重复构建。
            transformerlayers=dict(
                type='BaseTransformerLayer',
                # 层内注意力模块列表，按operation_order中的注意力步骤依次使用。
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        # 输入/输出特征通道D=256；不等于查询数量。
                        embed_dims=_dim_,
                        # 8头，每头256/8=32维；投影参数可训练，在所有查询间共享。
                        num_heads=8,
                        # 训练时随机丢弃部分注意力输出，概率0.1；eval模式不做该随机丢弃。
                        dropout=0.1),
                ],
                # FFN隐藏通道512，输出仍回到256维。
                feedforward_channels=_ffn_dim_,
                # FFN训练dropout概率，配置值不是可学习参数。
                ffn_dropout=0.1,
                # 元组指定层内执行顺序；norm为归一化，ffn为前馈网络。
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        # 【交互B 自车->地图】查询使用交互A输出，顺序读取地图实例。
        # Q=自车更新内容+位置；K=LaneNet地图特征+位置；V=地图特征。
        # 每头权重[B,1,N]；输出[B,1,256]，与交互A输出concat为[B,1,512]生成规划。
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            # 此模块重复的Transformer层数；不是注意力头数或未来时间步数。
            num_layers=1,
            # True保留每层结果供中间监督；False只返回该解码器最终结果。
            return_intermediate=False,
            # 一个Transformer层的结构说明；由上面的num_layers重复构建。
            transformerlayers=dict(
                type='BaseTransformerLayer',
                # 层内注意力模块列表，按operation_order中的注意力步骤依次使用。
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        # 输入/输出特征通道D=256；不等于查询数量。
                        embed_dims=_dim_,
                        # 8头，每头256/8=32维；投影参数可训练，在所有查询间共享。
                        num_heads=8,
                        # 训练时随机丢弃部分注意力输出，概率0.1；eval模式不做该随机丢弃。
                        dropout=0.1),
                ],
                # FFN隐藏通道512，输出仍回到256维。
                feedforward_channels=_ffn_dim_,
                # FFN训练dropout概率，配置值不是可学习参数。
                ffn_dropout=0.1,
                # 元组指定层内执行顺序；norm为归一化，ffn为前馈网络。
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        # 【交互C 运动查询间交互】300目标*6模式=1800运动查询。
        # 代码传query=key=value=motion_query，所以语义是自注意力，即使operation_order写cross_attn。
        # Q/K来源为运动内容+位置，V为运动内容；每头权重[B,1800,1800]。
        motion_decoder=dict(
            type='CustomTransformerDecoder',
            # 此模块重复的Transformer层数；不是注意力头数或未来时间步数。
            num_layers=1,
            # True保留每层结果供中间监督；False只返回该解码器最终结果。
            return_intermediate=False,
            # 一个Transformer层的结构说明；由上面的num_layers重复构建。
            transformerlayers=dict(
                type='BaseTransformerLayer',
                # 层内注意力模块列表，按operation_order中的注意力步骤依次使用。
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        # 输入/输出特征通道D=256；不等于查询数量。
                        embed_dims=_dim_,
                        # 8头，每头256/8=32维；投影参数可训练，在所有查询间共享。
                        num_heads=8,
                        # 训练时随机丢弃部分注意力输出，概率0.1；eval模式不做该随机丢弃。
                        dropout=0.1),
                ],
                # FFN隐藏通道512，输出仍回到256维。
                feedforward_channels=_ffn_dim_,
                # FFN训练dropout概率，配置值不是可学习参数。
                ffn_dropout=0.1,
                # 元组指定层内执行顺序；norm为归一化，ffn为前馈网络。
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        # 【交互D 运动->地图】每个目标模式独立读取筛选后的地图。
        # 展平batch为B*1800，Q[1,B*1800,256]、K/V[L,B*1800,256]。
        # Q来源为运动内容+位置；K为地图内容+相对位置；V为地图内容。每头权重[1,L]。
        # 恢复[B,300,6,256]后与交互C结果concat为512维，预测位移与模式分数。
        motion_map_decoder=dict(
            type='CustomTransformerDecoder',
            # 此模块重复的Transformer层数；不是注意力头数或未来时间步数。
            num_layers=1,
            # True保留每层结果供中间监督；False只返回该解码器最终结果。
            return_intermediate=False,
            # 一个Transformer层的结构说明；由上面的num_layers重复构建。
            transformerlayers=dict(
                type='BaseTransformerLayer',
                # 层内注意力模块列表，按operation_order中的注意力步骤依次使用。
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        # 输入/输出特征通道D=256；不等于查询数量。
                        embed_dims=_dim_,
                        # 8头，每头256/8=32维；投影参数可训练，在所有查询间共享。
                        num_heads=8,
                        # 训练时随机丢弃部分注意力输出，概率0.1；eval模式不做该随机丢弃。
                        dropout=0.1),
                ],
                # FFN隐藏通道512，输出仍回到256维。
                feedforward_channels=_ffn_dim_,
                # FFN训练dropout概率，配置值不是可学习参数。
                ffn_dropout=0.1,
                # 元组指定层内执行顺序；norm为归一化，ffn为前馈网络。
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        # 启用运动/规划交互位置嵌入；高维嵌入不是二维参考坐标。
        use_pe=True,
        bev_h=bev_h_,
        bev_w=bev_w_,
        # 300个检测槽位；初始Embedding[300,512]拆成256内容+256位置。
        num_query=300,
        # 类别数，不是查询槽位数量。
        num_classes=num_classes,
        # 输入通道列表/数值，应与前一级模块输出匹配。
        in_channels=_dim_,
        # 分布式训练同步分类loss归约用的样本计数。
        sync_cls_avg_factor=True,
        # 解码逐层回归并更新参考位置；不是训练后做额外路线后处理。
        with_box_refine=True,
        # False关闭检测Transformer的两阶段候选机制；与训练stage_1/stage_2无关。
        as_two_stage=False,
        map_num_vec=map_num_vec,
        map_num_classes=map_num_classes,
        map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
        map_num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
        # 实例Embedding[100,512]+点Embedding[20,512]广播相加为[100,20,512]再展开。
        # 初始2000个查询不是2000套完全独立Embedding。
        map_query_embed_type='instance_pts',
        # 对每实例20个点取xy最小/最大值生成辅助2D框。
        map_transform_method='minmax',
        # v2处理等价GT点序：开放线正反序，闭合轮廓循环起点；不是任意打乱点。
        map_gt_shift_pts_pattern='v2',
        # 方向监督比较相邻点，间隔1；几何遍历方向不等于车辆通行方向。
        map_dir_interval=1,
        # 地图点回归输出xy两维；不是map框只有两维。
        map_code_size=2,
        # 辅助地图框的4维权重；当前该框L1 loss为0，点坐标loss另有设置。
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        # 【9.4 感知Transformer】VAD_transformer.py组织BEV编码、检测解码、地图解码。
        # 最终BEV[B,10000,256]；检测特征[B,300,256]；地图特征[B,2000,256]。
        transformer=dict(
            type='VADPerceptionTransformer',
            map_num_vec=map_num_vec,
            map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
            # 按自车转角旋转历史BEV；旋转中心在Transformer实现中设置，不在此行给出。
            rotate_prev_bev=True,
            # 按自车平移调整历史读取参考位置，帮助时间对齐。
            use_shift=True,
            # 18维CAN bus元数据经MLP编码并加入BEV查询；与规划额外状态开关不同。
            use_can_bus=True,
            # 输入/输出特征通道D=256；不等于查询数量。
            embed_dims=_dim_,
            # 【BEV编码】每层依次时间采样、图像空间采样、FFN，总共3层。
            encoder=dict(
                type='BEVFormerEncoder',
                # 此模块重复的Transformer层数；不是注意力头数或未来时间步数。
                num_layers=3,
                # 模块使用的空间范围，与顶层point_cloud_range保持一致。
                pc_range=point_cloud_range,
                # 每个BEV格沿高度设置4个3D参考点，标定投影到各摄像头。
                num_points_in_pillar=4,
                # True保留每层结果供中间监督；False只返回该解码器最终结果。
                return_intermediate=False,
                # 一个Transformer层的结构说明；由上面的num_layers重复构建。
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    # 层内注意力模块列表，按operation_order中的注意力步骤依次使用。
                    attn_cfgs=[
                        dict(
                            # 时间可变形采样：查询预测偏移/权重，V来自历史与当前BEV；不做普通全格QK点积。
                            type='TemporalSelfAttention',
                            # 输入/输出特征通道D=256；不等于查询数量。
                            embed_dims=_dim_,
                            # 此注意力读取的特征尺度数量；BEV解码为1，图像空间采样用FPN尺度数。
                            num_levels=1),
                        dict(
                            # BEV查询读取可见摄像头特征；V来自图像，底层使用可变形采样。
                            type='SpatialCrossAttention',
                            # 模块使用的空间范围，与顶层point_cloud_range保持一致。
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                # 图像空间可变形采样；不要将这里num_points=8与BEV解码默认4混淆。
                                type='MSDeformableAttention3D',
                                # 输入/输出特征通道D=256；不等于查询数量。
                                embed_dims=_dim_,
                                # 图像采样每头每层共8个位置，分配给4个高度锚点；不是每个锚点再各采8个。
                                num_points=8,
                                # 此注意力读取的特征尺度数量；BEV解码为1，图像空间采样用FPN尺度数。
                                num_levels=_num_levels_),
                            # 输入/输出特征通道D=256；不等于查询数量。
                            embed_dims=_dim_,
                        )
                    ],
                    # FFN隐藏通道512，输出仍回到256维。
                    feedforward_channels=_ffn_dim_,
                    # FFN训练dropout概率，配置值不是可学习参数。
                    ffn_dropout=0.1,
                    # 元组指定层内执行顺序；norm为归一化，ffn为前馈网络。
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            # 【检测解码】300查询先互相交流，再围绕参考xy读取BEV，3层迭代。
            # 普通自注意力每头[300,300]；可变形BEV交互由query预测偏移/权重，不计算全格点积K。
            decoder=dict(
                type='DetectionTransformerDecoder',
                # 此模块重复的Transformer层数；不是注意力头数或未来时间步数。
                num_layers=3,
                # True保留每层结果供中间监督；False只返回该解码器最终结果。
                return_intermediate=True,
                # 一个Transformer层的结构说明；由上面的num_layers重复构建。
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    # 层内注意力模块列表，按operation_order中的注意力步骤依次使用。
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            # 输入/输出特征通道D=256；不等于查询数量。
                            embed_dims=_dim_,
                            # 8头，每头256/8=32维；投影参数可训练，在所有查询间共享。
                            num_heads=8,
                            # 训练时随机丢弃部分注意力输出，概率0.1；eval模式不做该随机丢弃。
                            dropout=0.1),
                        dict(
                            # 读取BEV：默认8头、每头4采样、1层，共32位置/查询。
                            # 偏移Linear 256->64，权重Linear 256->32；每头4点softmax；无独立点积K。
                            type='CustomMSDeformableAttention',
                            # 输入/输出特征通道D=256；不等于查询数量。
                            embed_dims=_dim_,
                            # 此注意力读取的特征尺度数量；BEV解码为1，图像空间采样用FPN尺度数。
                            num_levels=1),
                    ],
                    # FFN隐藏通道512，输出仍回到256维。
                    feedforward_channels=_ffn_dim_,
                    # FFN训练dropout概率，配置值不是可学习参数。
                    ffn_dropout=0.1,
                    # 元组指定层内执行顺序；norm为归一化，ffn为前馈网络。
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            # 【地图解码】2000点查询普通自注意力+可变形BEV读取，3层迭代修正坐标。
            # 自注意力覆盖全部2000查询，不只同实例20点；实例归属仍由固定分组保留。
            map_decoder=dict(
                type='MapDetectionTransformerDecoder',
                # 此模块重复的Transformer层数；不是注意力头数或未来时间步数。
                num_layers=3,
                # True保留每层结果供中间监督；False只返回该解码器最终结果。
                return_intermediate=True,
                # 一个Transformer层的结构说明；由上面的num_layers重复构建。
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    # 层内注意力模块列表，按operation_order中的注意力步骤依次使用。
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            # 输入/输出特征通道D=256；不等于查询数量。
                            embed_dims=_dim_,
                            # 8头，每头256/8=32维；投影参数可训练，在所有查询间共享。
                            num_heads=8,
                            # 训练时随机丢弃部分注意力输出，概率0.1；eval模式不做该随机丢弃。
                            dropout=0.1),
                         dict(
                            # 读取BEV：默认8头、每头4采样、1层，共32位置/查询。
                            # 偏移Linear 256->64，权重Linear 256->32；每头4点softmax；无独立点积K。
                            type='CustomMSDeformableAttention',
                            # 输入/输出特征通道D=256；不等于查询数量。
                            embed_dims=_dim_,
                            # 此注意力读取的特征尺度数量；BEV解码为1，图像空间采样用FPN尺度数。
                            num_levels=1),
                    ],
                    # FFN隐藏通道512，输出仍回到256维。
                    feedforward_channels=_ffn_dim_,
                    # FFN训练dropout概率，配置值不是可学习参数。
                    ffn_dropout=0.1,
                    # 元组指定层内执行顺序；norm为归一化，ffn为前馈网络。
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        # 【输出解码】检测分数top-k、框反归一化和空间过滤，同时携带目标未来轨迹。
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            # 解码后检测框中心xyz的米制过滤范围，比训练关注区稍宽。
            post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
            # 模块使用的空间范围，与顶层point_cloud_range保持一致。
            pc_range=point_cloud_range,
            # 解码最多保留100条检测候选输出；不是检测查询数（查询仍为300）。
            max_num=100,
            voxel_size=voxel_size,
            # 类别数，不是查询槽位数量。
            num_classes=num_classes),
        # 地图分数top-k、辅助框/点反归一化、空间过滤；它不执行地图匈牙利匹配。
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            # 地图minmax辅助框[xmin,ymin,xmax,ymax]的上下界过滤，前4个下界、后4个上界。
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            # 模块使用的空间范围，与顶层point_cloud_range保持一致。
            pc_range=point_cloud_range,
            # 解码最多保留50条地图候选输出；内部仍使用100个实例槽位。
            max_num=50,
            voxel_size=voxel_size,
            # 类别数，不是查询槽位数量。
            num_classes=map_num_classes),
        # 【BEV位置编码】学习行/列Embedding，各128维，拼成256；参数可训练。
        positional_encoding=dict(
            # 可训练的行/列位置嵌入，供BEV查询使用。
            type='LearnedPositionalEncoding',
            # 行/列各128维，二者拼接为256维。
            num_feats=_pos_dim_,
            # 可学习行位置表长度100。
            row_num_embed=bev_h_,
            # 可学习列位置表长度100。
            col_num_embed=bev_w_,
            ),
        # 【检测分类loss】预测目标类别，对匹配正样本及背景计算sigmoid focal。
        loss_cls=dict(
            type='FocalLoss',
            # 每类独立sigmoid；这里不是类别softmax概率总和为1。
            use_sigmoid=True,
            # focal聚焦参数2，降低容易样本的相对贡献。
            gamma=2.0,
            # focal正负样本平衡系数0.25。
            alpha=0.25,
            # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
            loss_weight=2.0),
        # 【检测框loss】GT匹配后对框编码算L1，权重0.25。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        # 【目标运动回归】GT终点minFDE选模式，再对该模式全部有效逐步位移算mask L1。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_traj=dict(type='L1Loss', loss_weight=0.2),
        # 【目标模式分类】GT选中的模式编号作为分类标签，六模式分别sigmoid focal。
        # 推理无GT，需要选择时用预测分数，不再计算minFDE。
        loss_traj_cls=dict(
            type='FocalLoss',
            # 每类独立sigmoid；这里不是类别softmax概率总和为1。
            use_sigmoid=True,
            # focal聚焦参数2，降低容易样本的相对贡献。
            gamma=2.0,
            # focal正负样本平衡系数0.25。
            alpha=0.25,
            # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
            loss_weight=0.2),
        # 检测IoU loss权重为0；兼容结构保留，不提供有效加权训练项。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        # 【地图分类】每实例的类别监督。
        loss_map_cls=dict(
            type='FocalLoss',
            # 每类独立sigmoid；这里不是类别softmax概率总和为1。
            use_sigmoid=True,
            # focal聚焦参数2，降低容易样本的相对贡献。
            gamma=2.0,
            # focal正负样本平衡系数0.25。
            alpha=0.25,
            # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
            loss_weight=2.0),
        # 地图辅助框L1权重0；不能误以为点坐标监督也关闭。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
        # 地图辅助框IoU权重0。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
        # 【地图点坐标】最佳等价点序与实例匹配后计算有序点L1。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        # 【地图形状方向】相邻预测点与GT点形成向量，计算方向余弦差；不是自车规划方向loss。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        # 【规划模仿】只监督导航cmd对应分支，逐步位移L1乘有效未来mask。
        # 自车输出[B,3,6,2]为右转/左转/直行三个分支；不是其他目标六个模式。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0),
        # 【规划边界】地图归一化坐标转米，与自车累计未来位置求距离。
        # 这里1.0是米制安全阈值，和运动地图筛选0.2不同。具体相交后置零等逻辑见plan_loss.py。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=1.0, dis_thresh=1.0),
        # 【规划碰撞】使用预测目标最高分模式的未来位置和自车未来位置求安全距离。
        # 具体类别过滤、距离阈值见plan_loss.py；不是精确旋转车辆框碰撞保证。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
        # 【规划道路方向】自车运动方向与附近divider几何方向的角度约束，权重0.5。
        # 所有这些规划loss在训练计算，推理不据此逐点优化路线。
        # 最终该loss项的乘法系数；不是通过训练自动学习的权重。
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.5)),
    # model training and testing settings
    # 【9.5 训练匹配配置】传给父类及head的训练设置；不是tools/train.py的命令行参数。
    # 框架沿用pts分支名；以下grid/voxel参数不自动开启LiDAR网络。
    train_cfg=dict(pts=dict(
        # 旧框架兼容的网格字段，不是BEV实际100x100格数。
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        # 旧检测训练配置兼容字段，不能据此推断FPN输出stride。
        out_size_factor=4,
        # 检测匈牙利匹配：选择哪个预测槽位对应哪个GT。
        # 匹配cost不是最终loss，但可以使用相同类型的误差；不要混淆weight与loss_weight。
        assigner=dict(
            type='HungarianAssigner3D',
            # 匹配cost的系数；用于确定对应关系，与最终loss系数分开。
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            # 匹配cost的系数；用于确定对应关系，与最终loss系数分开。
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # 匹配cost的系数；用于确定对应关系，与最终loss系数分开。
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            # 模块使用的空间范围，与顶层point_cloud_range保持一致。
            pc_range=point_cloud_range),
        # 地图匈牙利匹配；OrderedPtsL1Cost同时考虑等价GT点序的最小代价。
        map_assigner=dict(
            type='MapHungarianAssigner3D',
            # 匹配cost的系数；用于确定对应关系，与最终loss系数分开。
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            # 匹配cost的系数；用于确定对应关系，与最终loss系数分开。
            # 辅助框代价采用xywh格式；此代价权重0，地图主要按点序匹配。
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            # 辅助IoU代价形式giou，当前对应代价权重为0。
            # 匹配cost的系数；用于确定对应关系，与最终loss系数分开。
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            # 匹配cost的系数；用于确定对应关系，与最终loss系数分开。
            pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
            # 模块使用的空间范围，与顶层point_cloud_range保持一致。
            pc_range=point_cloud_range))))

# 【10 数据集类】由Dataset注册表找到datasets/nuscenes_vad_dataset.py中的类。
dataset_type = 'VADCustomNuScenesDataset'
# 相对于运行命令时工作目录的数据路径，通常在仓库根目录执行。
data_root = 'data/nuscenes/'
# 磁盘文件读取后端；这是配置字典，不是已打开的文件。
# 从本地磁盘读取文件。
file_client_args = dict(backend='disk')

# 【11 训练样本处理链】列表顺序就是执行顺序；Compose依次调用每个transform。
# 一个transform通常读写同一份样本字典。局部地图GT由Dataset.vectormap_pipeline额外构建。
train_pipeline = [
    # 读入图像后转浮点，后续进行颜色增强/归一化。
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # 训练颜色/亮度等增强；网络参数不在该transform中学习。
    dict(type='PhotoMetricDistortionMultiViewImage'),
    # 加载目标三维框GT。
    # 加载目标类别GT。
    # 加载扩展属性，含目标未来位移/mask等监督字段。
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    # 过滤空间范围外GT，同时过滤对应目标未来属性，保持对齐。
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    # 只保留class_names类别，标签与未来属性同步过滤。
    # Dataset类别编号顺序需与模型分类通道对应。
    dict(type='CustomObjectNameFilter', classes=class_names),
    # 将mean/std/to_rgb应用到多摄像头图像。
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # 按比例缩放图像，并同步修改图像投影相关矩阵。
    # 实际图像缩放比例0.4；对同一输入H/W按比例缩放。
    dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
    # 补齐图像尺寸，记录pad_shape等metadata。
    # 图像padding到高宽均可被32整除；padding不等于缩放。
    dict(type='PadMultiViewImage', size_divisor=32),
    # 图像/GT/自车字段转tensor与DataContainer，供collate组batch。
    # 格式化自车历史/未来/指令/状态字段，准备DataContainer。
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    # 收集模型字段与img_metas；metadata含相机标定、scene、CAN bus等。
    dict(type='CustomCollect3D',\
         # 只收集列出的数据字段与默认metadata，交给Dataset/loader/模型。
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels'])
]

# 【12 验证/测试样本处理链】这里带GT是为了开环指标；不等于规划前向使用未来GT。
# 此pipeline还读取点云用于数据/评价辅助，但model没有LiDAR特征提取分支。
test_pipeline = [
    # 读入图像后转浮点，后续进行颜色增强/归一化。
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # 测试pipeline确实读取点云；当前VAD.py的图像分支不把points送入点云特征网络。
    dict(type='LoadPointsFromFile',
         # 点云解释为LiDAR坐标系。
         coord_type='LIDAR',
         # 点云文件每点5个通道。
         load_dim=5,
         # 此步保留5个点云通道；模型use_lidar=False仍不构造点云编码器。
         use_dim=5,
         file_client_args=file_client_args),
    # 加载目标三维框GT。
    # 加载目标类别GT。
    # 加载扩展属性，含目标未来位移/mask等监督字段。
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    # 过滤空间范围外GT，同时过滤对应目标未来属性，保持对齐。
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    # 只保留class_names类别，标签与未来属性同步过滤。
    # Dataset类别编号顺序需与模型分类通道对应。
    dict(type='CustomObjectNameFilter', classes=class_names),
    # 将mean/std/to_rgb应用到多摄像头图像。
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        # 测试增强包装器，组织内部transforms；当前flip=False且只用一套缩放。
        type='MultiScaleFlipAug3D',
        # 测试增强包装器的尺度元数据；实际自定义缩放在scales=[0.4]那一步执行。
        img_scale=(1600, 900),
        # 测试包装器点云缩放元数据，值1表示不改变该比例。
        pts_scale_ratio=1,
        # False不执行这里的测试翻转增强。
        flip=False,
        # 测试包装器内部按顺序执行的transform列表。
        transforms=[
            # 按比例缩放图像，并同步修改图像投影相关矩阵。
            # 实际图像缩放比例0.4；对同一输入H/W按比例缩放。
            dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
            # 补齐图像尺寸，记录pad_shape等metadata。
            # 图像padding到高宽均可被32整除；padding不等于缩放。
            dict(type='PadMultiViewImage', size_divisor=32),
            # 图像/GT/自车字段转tensor与DataContainer，供collate组batch。
            # 测试格式化标签选项；前面仍加载GT用于开环评价。
            # 格式化自车历史/未来/指令/状态字段，准备DataContainer。
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_label=False, with_ego=True),
            # 收集模型字段与img_metas；metadata含相机标定、scene、CAN bus等。
            dict(type='CustomCollect3D',\
                 # 只收集列出的数据字段与默认metadata，交给Dataset/loader/模型。
                 keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                       'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                       'ego_lcf_feat', 'gt_attr_labels'])])
]

# 【13 Dataset与DataLoader配置】train/val/test描述Dataset，batch/worker/sampler描述数据加载。
data = dict(
    # 每张GPU一个时序样本，不是一张图；相机图像batch可为[B,3,6,3,H,W]。
    samples_per_gpu=1,
    # 每GPU数据加载工作进程4个；不是4张GPU，也不决定模型注意力头数。
    workers_per_gpu=4,
    # 训练Dataset：从temporal_train.pkl读取信息，pipeline处理图像/GT并组织历史队列。
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # 样本信息pkl路径；data_root字符串拼接不会自动检查文件存在。
        ann_file=data_root + 'vad_nuscenes_infos_temporal_train.pkl',
        # 引用前面定义的处理列表，不是重新手写同一份transform代码。
        pipeline=train_pipeline,
        # Dataset类别编号顺序需与模型分类通道对应。
        classes=class_names,
        # 传入前面的输入模态声明。
        modality=input_modality,
        # False按训练方式准备样本；验证/测试构造器另设置test_mode=True。
        test_mode=False,
        # 根据pkl的valid_flag过滤GT目标；与未来gt_fut_masks是两种不同有效性。
        use_valid_flag=True,
        # Dataset相关BEV尺寸，和模型100x100一致。
        bev_size=(bev_h_, bev_w_),
        # 模块使用的空间范围，与顶层point_cloud_range保持一致。
        pc_range=point_cloud_range,
        queue_length=queue_length,
        # 地图类别列表，需与地图分类通道对应。
        map_classes=map_classes,
        # GT每实例固定20点；实例数量不固定。
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        # 地图评价GT采样开关。
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        # GT框采用LiDAR坐标约定，不代表模型读取LiDAR特征。
        box_type_3d='LiDAR',
        # 自定义检测评估JSON配置名，datasets中有对应文件。
        custom_eval_version='vad_nusc_detection_cvpr_2019'),
    # 验证Dataset，使用验证pkl和test_pipeline；验证loader的batch设为1。
    val=dict(type=dataset_type,
             data_root=data_root,
             # 模块使用的空间范围，与顶层point_cloud_range保持一致。
             pc_range=point_cloud_range,
             # 样本信息pkl路径；data_root字符串拼接不会自动检查文件存在。
             ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
             # 引用前面定义的处理列表，不是重新手写同一份transform代码。
             # Dataset相关BEV尺寸，和模型100x100一致。
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             # Dataset类别编号顺序需与模型分类通道对应。
             # 传入前面的输入模态声明。
             classes=class_names, modality=input_modality, samples_per_gpu=1,
             # 地图类别列表，需与地图分类通道对应。
             map_classes=map_classes,
             # 地图评估GT缓存JSON，缺失时的生成/读取逻辑见Dataset；不是模型输入的HD地图。
             map_ann_file=data_root + 'nuscenes_map_anns_val.json',
             # GT每实例固定20点；实例数量不固定。
             map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
             # 地图评价GT采样开关。
             map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
             # 结果格式化使用pkl相关路径，方便地图/规划等扩展结果保存。
             use_pkl_result=True,
             # 自定义检测评估JSON配置名，datasets中有对应文件。
             custom_eval_version='vad_nusc_detection_cvpr_2019'),
    # 独立test.py也用验证pkl，便于开环评价；不是nuScenes无标注官方test集。
    test=dict(type=dataset_type,
              data_root=data_root,
              # 模块使用的空间范围，与顶层point_cloud_range保持一致。
              pc_range=point_cloud_range,
              # 样本信息pkl路径；data_root字符串拼接不会自动检查文件存在。
              ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
              # 引用前面定义的处理列表，不是重新手写同一份transform代码。
              # Dataset相关BEV尺寸，和模型100x100一致。
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              # Dataset类别编号顺序需与模型分类通道对应。
              # 传入前面的输入模态声明。
              classes=class_names, modality=input_modality, samples_per_gpu=1,
              # 地图类别列表，需与地图分类通道对应。
              map_classes=map_classes,
              # 地图评估GT缓存JSON，缺失时的生成/读取逻辑见Dataset；不是模型输入的HD地图。
              map_ann_file=data_root + 'nuscenes_map_anns_val.json',
              # GT每实例固定20点；实例数量不固定。
              map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
              # 地图评价GT采样开关。
              map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
              # 结果格式化使用pkl相关路径，方便地图/规划等扩展结果保存。
              use_pkl_result=True,
              # 自定义检测评估JSON配置名，datasets中有对应文件。
              custom_eval_version='vad_nusc_detection_cvpr_2019'),
    # 训练分布式分组/打乱索引，具体实现见datasets/samplers。
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    # 验证/测试的分布式索引分配，不会改变模型查询结构。
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# 【14 优化器】mmdet_train.py中build_optimizer(model,cfg.optimizer)创建实际对象。
optimizer = dict(
    type='AdamW',
    # 基础学习率2e-4=0.0002，实际值由学习率Hook调度。
    lr=2e-4,
    # 按参数名字匹配学习率倍率；这里只为img_backbone设置0.1。
    paramwise_cfg=dict(
        # img_backbone有效基础lr=2e-4*0.1=2e-5；未冻结的参数才会更新。
        custom_keys={
            # 骨干学习率倍率0.1。
            'img_backbone': dict(lr_mult=0.1),
        }),
    # AdamW权重衰减0.01，不是dropout概率。
    weight_decay=0.01)

# 【15 优化器Hook配置】梯度更新由MMCV Hook触发；这里设置L2梯度范数裁剪上限35。
# 反向传播后按全局梯度L2范数裁剪，避免过大的更新；不是裁剪轨迹坐标。
# 梯度裁剪最大范数35。
# 梯度范数类型2表示L2范数。
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# 【16 学习率Hook配置】预热后余弦衰减；不是手工在模型forward里改lr。
lr_config = dict(
    # 余弦学习率衰减策略，由MMCV创建相应Hook。
    policy='CosineAnnealing',
    # linear表示前期线性预热；500是iteration数量。
    warmup='linear',
    # 前500次训练迭代进行学习率预热。
    warmup_iters=500,
    # 预热起点为相应基础学习率的1/3，再逐步升高。
    warmup_ratio=1.0 / 3,
    # 余弦衰减最低比例为1e-3，不是最低学习率直接等于0.001。
    min_lr_ratio=1e-3)

# 【17 评价Hook】interval=60按当前epoch Runner在60轮间隔执行；不是每轮评价。
# metric="bbox"是此自定义Dataset的评价入口，map_metric指定地图距离评价。
# 此处等于total_epochs=60，训练结束附近才进行该定时验证。
# 引用前面定义的处理列表，不是重新手写同一份transform代码。
# 调用自定义Dataset bbox评价入口，包含该工程扩展指标。
# 地图匹配/评价使用Chamfer距离，不是训练点L1 loss。
evaluation = dict(interval=total_epochs, pipeline=test_pipeline, metric='bbox', map_metric='chamfer')

# 【18 训练循环】使用外部MMCV标准EpochBasedRunner，不是VAD/runner中的视频Runner。
# Runner最多60epoch，具体每epoch迭代数由数据和采样器决定。
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# 【19 日志Hook】训练过程按迭代间隔打印/写入日志。
log_config = dict(
    # 每100个训练iteration记录日志，不是每100个epoch。
    interval=100,
    # 日志Hook列表，训练到触发时刻才执行，不在配置加载时打印每批日志。
    hooks=[
        # 把loss、lr、耗时等写到文本日志。
        dict(type='TextLoggerHook'),
        # 记录TensorBoard事件；可用tensorboard查看训练曲线。
        dict(type='TensorboardLoggerHook')
    ])
# fp16 = dict(loss_scale=512.)
# find_unused_parameters = True
# 【20 权重保存Hook】每1个epoch保存，最多保留60份epoch checkpoint。
# 每1个epoch保存checkpoint，与evaluation.interval=60不同。
# 最多保留60份epoch checkpoint；实际保存目录由--work-dir或默认值决定。
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)


# 【21 自定义Hook】CustomSetEpochInfoHook在每轮开始调用model.set_epoch。
# 这里没有load_from；stage_2配置才显式加载stage_1 checkpoint。
# VAD/hooks/custom_hooks.py中注册；before_train_epoch调用模型set_epoch。
custom_hooks = [dict(type='CustomSetEpochInfoHook')]