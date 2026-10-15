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
# 训练从 forward(return_loss=True) -> forward_train 开始，先跳到标有【A1】的地方。
# 【A1】拆出历史/当前 -> 【A2】obtain_history_bev生成历史上下文 -> 【A3】当前图像提特征
# -> 【A4】forward_pts_train调用Head预测和loss。读历史时会跳进另外两个文件，再返回这里。
# prev_bev不是Dataset提供的GT，也不是一张RGB鸟瞰图，而是网络对历史图片生成的BEV特征。
# 它可理解成“每个鸟瞰格子已有的历史内容”，当前格子查询会参考它，结合当前图片再更新。
# 测试是另一条路：forward_test按时间顺序缓存上一帧BEV，不再在每次输入里塞训练的三帧队列。

import time
import copy

import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric


# 登记类名供配置type查找；导入本文件时登记，不在此创建样本或执行前向。
@DETECTORS.register_module()
class VAD(MVXTwoStageDetector):
    """VAD model.
    """
    # 【准备模型，还没有喂图片】配置type='VAD'找到本类，Python创建实例时执行这里。
    # 父类MVXTwoStageDetector在mmdet3d/models/detectors/mvx_two_stage.py中创建：
    # self.img_backbone=ResNet、self.img_neck=FPN、self.pts_bbox_head=VADHead。
    # super调用父类构造不是再建第二个VAD；它们为同一个self对象装配模块。
    # 下面的prev_frame_info是测试缓存；训练历史在obtain_history_bev中临时计算，不复用这个跨调用缓存。
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 fut_mode=6
                 ):

        # 先进入mmdet3d父类装配基础组件。img_backbone等现在还是字典，父类build_*后才是模块对象。
        # 所以后面self.img_backbone(img)不是调用一个字典，而是调用已经创建的ResNet。
        super(VAD,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head['valid_fut_ts']

        # temporal
        self.video_test_mode = video_test_mode
        # 只供测试：保存上次调用的BEV、场景和绝对位姿。开始时None表示尚无历史图像内容。
        # 训练的三帧队列另走A2，不在不同训练batch间混用这个缓存，因为训练样本可能被打乱。
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.planning_metric = None

    # 【A3的工具：图片变内容特征】一张图片从[3,384,640]变为[256,12,20]，六个视角独立过CNN。
    # 当前调用：输入[1,6,3,384,640] -> backbone输入[6,3,384,640] -> ResNet[6,2048,12,20]
    # -> FPN[6,256,12,20] -> 返回列表中的唯一Tensor[1,6,256,12,20]。
    # 历史调用：调用者暂把两帧当batch，输入[2,6,3,384,640]，这里返回[1,2,6,256,12,20]。
    # 因此len_queue=None表示当前提特征；len_queue=2表示最后需要恢复历史时间轴。
    # 返回值是尺度列表[Tensor]，不是Tensor本身；Tiny只有1个尺度，12/20是特征图空间尺寸。
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        # 记录输入首轴；当前B=1，历史批量提取时这里B暂时为2。下面恢复shape时会用它。
        # 不要看到局部变量B就认为任何时候都等于原始batch=1，它在历史调用里已合并时间。
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            # 当前[1,6,3,384,640]走这里，squeeze_后[6,3,384,640]。
            # 下划线表示原地操作。此处依赖六相机/RGB轴不是1；不说明任意单相机输入都自动兼容。
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            # 历史输入[2,6,3,384,640]走这里，合并前两轴得到[12,3,384,640]。
            # 12是两帧x六相机，不是12个独立训练样本。
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            # 可选图像遮挡增强，形状不变；不是BEV分割mask，也不是未来GT有效性mask。
            # 历史A2临时eval时该模块按自身实现不执行训练增强；当前帧train时按概率遮挡。
            if self.use_grid_mask:
                img = self.grid_mask(img)

            # 当前图片[6,3,384,640] -> ResNet最后stage [6,2048,12,20]，外层是尺度容器。
            # 空间缩小意味着提取高层特征，不是把6张图拼成一张。要看内部则进入mmdet/models/backbones/resnet.py。
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            # FPN对当前[6,2048,12,20]映射为[6,256,12,20]，保持视角与空间排列。
            # 历史则是[12,256,12,20]。此时还没有任何鸟瞰格子，需要到C1才能变BEV。
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            # 历史调用len_queue=2，局部B=2、BN=12，view为[1,2,6,256,12,20]。
            # 当前len_queue=None则走else，view为[1,6,256,12,20]；只是恢复索引，没有新建特征。
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        # 回到A2或A3：返回[Tensor]，方括号表示一个尺度，不等于Tensor又多了一轴。
        # 历史的Tensor含2帧轴；当前的不含。后续mlvl_feats[0]就是取这个唯一尺度。
        return img_feats_reshaped

    # 薄封装：把当前/历史图片交给上面的extract_img_feat，然后原样返回多尺度列表。
    # 装饰器只在启用对应fp16设置时进行转换，不会因为写了auto_fp16就必然使用半精度。
    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    # 【A4：当前帧特征交给Head】这里的pts_feats实际是[Tensor(1,6,256,12,20)]，不是点云。
    # 第一次调用Head生成outs；第二次调用Head.loss才传检测/地图GT、真实未来、mask、命令。
    # 这两个调用分开，说明模型生成未来预测时没有提前读取真实未来轨迹。返回的是各项loss字典。
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,                          
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          prev_bev=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        # 进入VAD_head.py的B1：传当前图像特征、当前metadata、A2返回的历史BEV。
        # 只把历史/状态的可选输入传给Head；真实未来ego_fut_trajs、gt_attr_labels不在这次预测调用中。
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        # Head预测完成才组装监督参数，顺序与VADHead.loss签名一一对应。
        # *loss_inputs将列表展开成位置实参。loss返回各标量项，交回Runner再反传；不是在Head调用时自动优化。
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels
        ]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    # 这是工具用的占位入口，不是正常训练/推理样本；dummy metadata=None不能替代真实标定。
    # 主线学习先跳过，训练从forward_train走。
    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    # 【模型的总开关】Runner继承的train_step执行self(**data)后，经nn.Module机制到这里。
    # return_loss=True默认训练；False为测试。它和model.train()/eval()控制BN/Dropout模式不是一回事。
    # **kwargs将batch字段按名称继续传入forward_train，不是重新创建模型。
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # 训练默认True，进入A1。测试入口显式return_loss=False，走下面forward_test。
        # 同一个VAD对象复用参数；不是每次forward重新调用__init__。
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    # 【A2：先建立历史记忆】此处实际输入[1,2,6,3,384,640]，metadata仍保留3帧，循环只取前2项。
    # 故事顺序：prev_bev=None -> 编码较早帧得到bev0 -> 编码较近帧并融合bev0得到bev1 -> 返回bev1。
    # bev0/bev1都是[1,10000,256]：100x100鸟瞰位置，每位置256维内容；不是2D坐标、GT或RGB图。
    # 为什么不把两张历史BEV直接concat后保留两倍通道？这里采用递归更新，上一份记忆融合新图像后仍是256维。
    # 与LSS联系：假如你已熟悉“历史LSS BEV通过运动对齐再融合”，这里历史记忆的用途类似；
    # 区别是每帧BEV由BEVFormer查询/投影/可变形注意力生成，不是LSS的深度lift。
    # with no_grad不构建历史反传图，临时eval关闭训练态随机层；最后恢复train，当前帧还会训练共享参数。
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        # 暂时切到评估态以生成稳定历史特征；这不代表本次是独立验证集评估。
        # 下面no_grad才负责不记录历史求导图，eval本身不会关闭梯度。
        self.eval()

        with torch.no_grad():
            # 历史循环起点：没有看过队列第一帧，所以还没有历史BEV。
            # None不是全零Tensor：C1/encoder会选择“没有可用历史”的逻辑，用当前查询初始化时间注意力。
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            # [1,2,6,3,384,640] -> [2,6,3,384,640]，把两历史帧暂当CNN的batch。
            # 这样共享ResNet一次提两帧特征，结果再恢复时间轴，不是把两帧像素相加。
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            # 进入A3工具extract_img_feat，再回来得到[Tensor(1,2,6,256,12,20)]。
            # 下一行for不是再提CNN，而是把已提好的特征逐帧送去生成BEV。
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            # 只有两次：i=0较早历史，i=1较近历史。
            # i=0：prev_bev=None -> 返回bev0[1,10000,256]。
            # i=1：传入bev0 + 第二历史图像 -> 返回bev1[1,10000,256]。
            # 变量名一直叫prev_bev，但每次赋值后内容更新；循环外拿到的是融合后的bev1。
            for i in range(len_queue):
                # 对原batch每个样本取队列第i帧metadata，本例列表长度1；里面含该帧六相机lidar2img。
                # metadata不是学习特征，它告诉编码器投影位置及自车如何相对上帧运动。
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                # 唯一尺度[1,2,6,256,12,20]选时间i，变成[1,6,256,12,20]。
                # [:,i]选的是历史时间，不是相机；其他轴由切片保留。
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                # 此刻跳到Head B1，再在B2只执行C1的图像->BEV；回来得到[1,10000,256]。
                # 10000=100x100格，256是格子内容。没有生成300框/100条线，所以历史开销比完整Head少。
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            # 历史两帧结束恢复训练模式；return把融合后的历史BEV交回A1，不会在此保存到全局测试缓存。
            self.train()
            return prev_bev

    # @auto_fp16(apply_to=('img', 'points'))
    # 【A1：从这里开始追一组真实训练数据】
    # img=[1,3,6,3,384,640]；img_metas是列表[一个字典]，该字典key为0/1/2，对应三帧。
    # 框/类别/轨迹GT只对应最后当前帧；例如G=10时框[10,9]、类别[10]、attrs[10,34]。
    # 地图GT是M个线几何对象；自车未来[1,1,6,2]、mask[1,1,1,6]，不是模型输入的未来线索。
    # 本函数只组织历史/当前数据，不在这里构造BEV查询，也不在这里调用optimizer.step。
    # 沿下面【A1.1】开始，遇到obtain_history_bev进入【A2】，返回后继续【A3】【A4】。
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        # 【A1.1】这次L=3，第二轴是队列。这里读尺寸，不对图片做任何计算。
        len_queue = img.size(1)
        # 【A1.2】[1,3,6,3,384,640]取前两帧 -> [1,2,6,3,384,640]。
        # :-1保留从头到倒数第一之前；...保留后面的相机/通道/空间轴。
        prev_img = img[:, :-1, ...]
        # 当前只取最后一帧：[1,6,3,384,640]；整数索引去掉了队列轴。
        # 它仍有六张相机图片，不是仅取最后一个相机。
        img = img[:, -1, ...]

        # metadata形态为[ {0:第一帧字典,1:第二帧字典,2:当前帧字典} ]。
        # deepcopy复制嵌套结构，供A2按0/1索引读取，不会自动挑历史帧或计算BEV。
        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        # 【A2调用点】此处调用上方历史函数，读完它再回到下一行。
        # 本例得到[1,10000,256]的历史记忆；如果队列只有1帧，则没有历史，直接传None。
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None

        # A2完成后，把metadata改为只含当前帧：[ {当前帧的标定/尺寸/运动信息} ]。
        # 列表推导式遍历batch，不是遍历六相机；len_queue-1=2选当前队列位置。
        img_metas = [each[len_queue-1] for each in img_metas]
        # 【A3调用点】当前[1,6,3,384,640] -> ResNet/FPN -> [Tensor(1,6,256,12,20)]。
        # 这里得到的是透视相机特征；A2的prev_bev是鸟瞰历史，两者形状/角色完全不同。
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        # 【A4调用点】进入forward_pts_train：当前相机特征+历史BEV预测当前场景，再与当前GT算loss。
        # 拿到losses_pts后update到losses，最后return给BaseDetector.train_step；仍未执行AdamW更新。
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore, prev_bev,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels)

        losses.update(losses_pts)
        return losses

    # 【测试支线】一帧一帧调用，外层可能有测试增强列表；通常只支持B=1。
    # 三帧训练是“一次输入含历史”；这里是“上次调用的BEV缓存在self.prev_frame_info”。
    # 场景token变化或video_test_mode=False会清空缓存，避免别的场景污染当前结果。
    # 给Head前把全局位姿变成相邻帧运动量；本次绝对位置先另存，下一次还要用它求差。
    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        # 这里[0][0]依次是测试增强索引、batch样本索引，不是训练队列第0帧。
        # 检测新场景就丢弃旧BEV：两个不相干道路场景不能共用历史记忆。
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        # 先保留本帧绝对全局位置/角度；下面can_bus改成差值后，不能再拿差值当下次的绝对基准。
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        # 一次当前帧预测：完整Head返回的BEV布局[10000,1,256]，保存给下次调用；C1也兼容这个布局。
        # 因此测试缓存和训练历史函数返回的[1,10000,256]看起来不同，只是轴顺序。
        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info['prev_bev'],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results

    # 【测试：提当前图片特征并组织输出】图片[1,6,3,H,W] -> CNN/FPN -> Head -> 解码。
    # 返回new_prev_bev供forward_test缓存，以及包含pts_bbox和metric_results的样本列表。
    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict

        return new_prev_bev, bbox_list

    # 【测试：先预测，后评估】先Head.forward再get_bboxes，之后才用GT算指标。
    # GT出现不表示它参与预测；no_grad块内按0.6分数筛选用于运动指标，再匹配GT。
    # 最终按ego_fut_cmd选择3个自车分支之一，逐步位移cumsum成位置后计算L2/碰撞。
    # 这里明确只支持测试batch=1，不能根据训练支持B>1就假设本评估也支持。
    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Test function"""
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]

        # 测试同样走Head B1-B6，先生成预测；后面的GT仅用于指标。
        # get_bboxes把编码/logits变成可评估/可视化的框、地图点与分数，不是再次运行CNN。
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = []
        for i, (bboxes, scores, labels, trajs, map_bboxes, \
                map_scores, map_labels, map_pts) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result['trajs_3d'] = trajs.cpu()
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        score_threshold = 0.6
        # 从这里开始是评估支线：复制预测、筛分数、匹配真实目标、算运动/规划指标。
        # 评估中使用GT不能反过来说模型提前输入真实未来；Head前向已在上面完成。
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)

            bbox_result = c_bbox_results[0]
            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])
            # filter pred bbox by score_threshold
            mask = bbox_result['scores_3d'] > score_threshold
            bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
            bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
            bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
            bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]

            matched_bbox_result = self.assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label)

            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result,
                matched_bbox_result, mapped_class_names)

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = bbox_result['ego_fut_preds']
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            # 3命令分支用命令选一条[T=6,2]位移，再cumsum成位置。
            # 这是数据提供的导航条件选择，不是运动分支6模式的argmax分数选择。
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)

        return outs['bev_embed'], bbox_results, metric_dict

    # 【测试结果包装】把解码后的地图框/分数/类别/点搬到CPU放入字典。
    # 没有再次生成地图或进行训练，Nmap是coder筛选数量，不是固定GT数M。
    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['map_attrs_3d'] = attrs.cpu()

        return result_dict

    # 【测试匹配】用局部米制框中心距离[Npred,G]配对，结果[G]保存每个GT的预测索引，-1无匹配。
    # 这是评估用的匹配，不是训练中分类cost+框cost的HungarianAssigner3D。
    # 动态类别与2米阈值在这里决定哪些配对计入运动指标；没有反向传播。
    def assign_pred_to_gt_vip3d(
        self,
        bbox_result,
        gt_bbox,
        gt_label,
        match_dis_thresh=2.0
    ):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """     
        dynamic_list = [0,1,3,4,6,7,8]
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result['boxes_3d'].center[:, :2]
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]

        return matched_bbox_result

    # 【测试运动指标】匹配一个目标后取6模式x6步位移，cumsum并加各自当前框中心，再算距离。
    # 累计car/pedestrian的ADE/FDE/MR及计数，最终由外部按计数汇总；不能把这里累加值当最终平均值。
    # 部分未来可算ADE；完整未来才计FDE等；这里与训练“直接取mask后末步”的选模式逻辑要分开。
    # 本函数会修改评估用类别Tensor，调用者先复制了结果，不是为训练修改原始标签。
    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        
        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met+'_'+cls] = 0.0

        veh_list = [0,1,3,4]
        ignore_list = ['construction_vehicle', 'barrier',
                       'traffic_cone', 'motorcycle', 'bicycle']

        for i in range(pred_bbox['labels_3d'].shape[0]):
            pred_bbox['labels_3d'][i] = 0 if pred_bbox['labels_3d'][i] in veh_list else pred_bbox['labels_3d'][i]
            box_name = mapped_class_names[pred_bbox['labels_3d'][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict['fp_'+box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts*2:self.fut_ts*3]
            num_valid_ts = sum(gt_fut_masks==1)
            if num_valid_ts == self.fut_ts:
                metric_dict['gt_'+box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict['cnt_ade_'+box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][:self.fut_ts*2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict['ADE_'+box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict['cnt_fde_'+box_name] += 1
                    metric_dict['FDE_'+box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict['hit_'+box_name] += 1
                    else:
                        metric_dict['MR_'+box_name] += 1

        return metric_dict

    ### same planning metric as stp3
    # 【测试规划指标】输入已累计的自车位置[1,6,2]，按2/4/6步评估1/2/3秒。
    # 由GT框及未来属性生成占据栅格，仅用于碰撞评估；这不是模型使用了栅格地图表示。
    # fut_valid_flag控制是否计入完整指标；函数没有训练参数更新。
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict

    # 【训练控制】CustomSetEpochInfoHook在每轮开始调用，保存epoch并告诉Head。
    # 用来支持可选loss预热等逻辑；不是sample时间，也不是又执行一轮训练。
    def set_epoch(self, epoch): 
        self.pts_bbox_head.epoch = epoch