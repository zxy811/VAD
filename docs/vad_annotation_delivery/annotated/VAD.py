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
# 职责：完整模型入口，拆分历史/当前图片，提取图像特征，调用 VADHead，组织 loss 或测试结果。
# 训练链：BaseDetector.train_step -> self(**data) -> VAD.forward -> forward_train
# -> obtain_history_bev/extract_feat -> forward_pts_train -> VADHead.forward/loss。
# 父类 MVXTwoStageDetector.__init__ 在 mmdet3d 中根据配置创建 img_backbone、img_neck、pts_bbox_head。
# pts_bbox_head 实际是 VADHead，不表示当前特征来自点云。测试 GT 用于指标，不提前输入未来给 Head。

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


# 【类】VAD：构造时执行__init__；实例可通过方法或nn.Module调用机制参与计算。
# @...register_module()把类登记为名称->类的映射；导入时注册，不会在这里实例化或训练。
@DETECTORS.register_module()
class VAD(MVXTwoStageDetector):
    """VAD model.
    """
    # 【函数入口】VAD.__init__
    # 构造模型对象；不是一次训练前向。父类按 type 配置创建 Head/ResNet/FPN，再添加时序缓存。
    # super(VAD,self).__init__ 调用继承链后续构造函数，父子类处理的是同一个 self 对象。
    # 输入 img_backbone/img_neck/pts_bbox_head 此时是配置，父类处理后成为 self 下的模块对象。
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

        # 原地初始化参数值；在模型初始化阶段执行，不是每次训练迭代重新随机设置。
        super(VAD,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.use_grid_mask = use_grid_mask
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.fp16_enabled = False
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.fut_ts = fut_ts
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.fut_mode = fut_mode
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.valid_fut_ts = pts_bbox_head['valid_fut_ts']

        # temporal
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.video_test_mode = video_test_mode
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.planning_metric = None

    # 【函数入口】VAD.extract_img_feat
    # 输入当前图片 [B,N,3,H,W]；处理历史时调用者先合并 B 与历史长度，首维为 B*Lh。
    # B=1 分支 squeeze_ 原地删除所有长度为1的轴；当前 Tiny N=6、RGB=3，得到 [N,3,H,W]。
    # 否则 reshape 为 [B*N,3,H,W]；Tiny ResNet 最后 stage 通道2048，常见空间12x20，FPN变256。
    # 返回多尺度列表；当前帧每项 [B,N,D,h,w]，历史特征每项 [B,Lh,N,D,h,w]。
    # list/dict 是多尺度容器，不是额外的 Tensor 维度；局部变量 H/W 会被特征图尺寸覆盖。
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if img.dim() == 5 and img.size(0) == 1:
                # 原地删除长度1的维度；当前B=1且N/C非1时[B,N,3,H,W]->[N,3,H,W]。单相机等极端形状需另查兼容性。
                img.squeeze_()
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                # 合并batch和相机轴：[B,N,3,H,W]->[B*N,3,H,W]，ResNet把每张视角图像作为一项处理。
                img = img.reshape(B * N, C, H, W)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if self.use_grid_mask:
                # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
                img = self.grid_mask(img)

            # 调用父类创建的ResNet.forward；Tiny out_indices=(3,)返回单尺度容器，常见[B*N,2048,12,20]。
            img_feats = self.img_backbone(img)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
            return None
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if self.with_img_neck:
            # 调用FPN.forward；Tiny唯一尺度通道2048->256，常见[B*N,256,12,20]，仍是尺度列表/元组。
            img_feats = self.img_neck(img_feats)

        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        img_feats_reshaped = []
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
            if len_queue is not None:
                # 恢复历史batch和时间：[B*Lh*N,D,h,w] -> [B,Lh,N,D,h,w]；局部B已是原batch乘Lh。
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                # 恢复当前batch和相机：[B*N,D,h,w]->[B,N,D,h,w]；append加入尺度列表，不是增加Tensor轴。
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return img_feats_reshaped

    # 【函数入口】VAD.extract_feat
    # 调用 extract_img_feat；当前相机路径没有在这里提取点云特征。
    # 装饰器 auto_fp16 受 fp16_enabled 控制，并不是看见装饰器就必然启用半精度训练。
    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return img_feats

    # 【函数入口】VAD.forward_pts_train
    # pts_feats 实际为图像特征列表，例如唯一尺度 [B,6,256,12,20]。
    # 先调用 self.pts_bbox_head(...) -> VADHead.forward；再把预测 outs 与 GT 交给 VADHead.loss。
    # GT 框/地图/目标未来/自车未来没有在这里提前作为真实未来输入给 Head。返回标量损失字典。
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

        # 调用VADHead对象，经nn.Module.__call__进入VADHead.forward；返回预测字典，未来GT未传入。
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels
        ]
        # *loss_inputs将列表逐项展开为位置参数；img_metas=显式关键字参数；在此预测才与GT比较。
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return losses

    # 【函数入口】VAD.forward_dummy
    # 用于工具接口的占位前向；dummy metadata=None 不是正常六相机推理样本。
    # 若外部 FLOPs 工具调用失败，应准备实际支持的 metadata，不能据此认为训练路径有问题。
    def forward_dummy(self, img):
        dummy_metas = None
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    # 【函数入口】VAD.forward
    # nn.Module.__call__ 进入这里；return_loss=True 走训练，False走测试。
    # 训练 img 是Tensor、img_metas是按样本排列的字典列表；测试还带一层测试增强容器。
    # return_loss 不是 model.training：一个控制接口分支，一个影响 Dropout/BN 等层行为。
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
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if return_loss:
            # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
            return self.forward_train(**kwargs)
        else:
            # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
            return self.forward_test(**kwargs)
    
    # 【函数入口】VAD.obtain_history_bev
    # 输入 imgs_queue=[B,Lh,N,3,H,W]，当前队列3时 Lh=2。
    # 临时 eval + no_grad 生成历史上下文，先批量提历史图像特征，再逐帧更新 prev_bev。
    # 每帧 Head 使用 only_bev=True，只运行 BEV 编码，不生成完整检测/地图/运动/规划输出。
    # 返回历史 BEV 通常 [B,10000,D]；当前帧路径仍可训练同一套参数；函数最后恢复 train 模式。
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        # with torch.no_grad作用域内不记录自动求导图；eval是层模式，两者不是同一个开关。
        with torch.no_grad():
            # 历史BEV特征Tensor；与metadata内同名布尔标志不同
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            # 把batch与历史时间合并：[B,Lh,N,3,H,W] -> [B*Lh,N,3,H,W]，便于一次提取历史特征。
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for i in range(len_queue):
                # 取每个样本的历史队列位置i，得到长度B的帧metadata列表，与选取的历史特征对齐。
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                # 逐尺度选历史时间i：[B,Lh,N,D,h,w] -> [B,N,D,h,w]，没有跨相机取第i张。
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                # 历史帧调用only_bev=True，仅得到BEV；没有运行完整检测/地图/运动/规划分支。
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
            return prev_bev

    # @auto_fp16(apply_to=('img', 'points'))
    # 【函数入口】VAD.forward_train
    # 训练模型入口：img=[B,L,N,3,H,W]；img_metas为长度B的列表，每项按队列索引0..L-1保存metadata。
    # 当前目标框/类别/属性/地图GT是最后一帧的监督；ego_fut_trajs=[B,1,T,2]、mask=[B,1,1,T]。
    # 历史图像与当前图像分流，复制metadata后取当前帧，返回 Head 的全部损失项。
    # 外部 BaseDetector.train_step 随后解析损失；本函数不调用 optimizer.step。
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
        
        # 图片队列长度L，不是batch数；用于分离历史和最后当前帧
        len_queue = img.size(1)
        # 切片[:, :-1, ...]保留除最后一帧外的队列：[B,L,N,3,H,W] -> [B,L-1,N,3,H,W]；...保留其余轴。
        prev_img = img[:, :-1, ...]
        # 索引-1取最后一帧并移除队列轴：[B,L,N,3,H,W] -> [B,N,3,H,W]。
        img = img[:, -1, ...]

        # 递归复制嵌套metadata；不是图片复制，不选择历史帧；两份信息在后续处理中可独立修改。
        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        # 历史BEV特征Tensor；与metadata内同名布尔标志不同
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None

        # 列表推导式遍历B个样本，每样本取队列最后一帧的字典；list[B]保持，时间字典层被去掉。
        img_metas = [each[len_queue-1] for each in img_metas]
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # 损失字典或调用者返回的loss容器；可微Tensor稍后在框架中汇总并反传
        losses = dict()
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore, prev_bev,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels)

        losses.update(losses_pts)
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return losses

    # 【函数入口】VAD.forward_test
    # 测试顶层：测试增强容器之外的 batch 通常只支持1；不是训练的三帧队列。
    # 按 scene_token 判断场景切换并清空历史BEV；把绝对车辆位置/角度转为相邻调用运动量。
    # 保存的 tmp_pos/tmp_angle 仍是本次绝对值，供下次计算差值；预测后缓存当前BEV。
    # 顺序推理需要按场景时间排列，不能随意打乱帧后继续复用缓存。
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
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for var, name in [(img_metas, 'img_metas')]:
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if not isinstance(var, list):
                # 显式抛异常：表示该配置/类型分支不受支持，正常主线不应进入此处。
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        img = [img] if img is None else img

        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            # 清空上一帧BEV缓存；场景变化或关闭时序时不能继续融合其他场景的历史。
            self.prev_frame_info['prev_bev'] = None
        # update idx
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
        if not self.video_test_mode:
            # 清空上一帧BEV缓存；场景变化或关闭时序时不能继续融合其他场景的历史。
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        # 创建副本；clone默认保留梯度联系，deepcopy用于独立容器/模块，不能把两者都说成detach。
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        # 创建副本；clone默认保留梯度联系，deepcopy用于独立容器/模块，不能把两者都说成detach。
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.prev_frame_info['prev_bev'] is not None:
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            img_metas[0][0]['can_bus'][-1] = 0
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            img_metas[0][0]['can_bus'][:3] = 0

        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
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
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.prev_frame_info['prev_pos'] = tmp_pos
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.prev_frame_info['prev_angle'] = tmp_angle
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.prev_frame_info['prev_bev'] = new_prev_bev

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return bbox_results

    # 【函数入口】VAD.simple_test
    # 取一帧图像特征，调用 simple_test_pts，给每个样本组织 pts_bbox 和 metric_results。
    # 测试增强外层已由 forward_test 取掉；这里 img通常 [B,N,3,H,W]。
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
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        bbox_list = [dict() for i in range(len(img_metas))]
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
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
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            result_dict['pts_bbox'] = pts_bbox
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            result_dict['metric_results'] = metric_dict

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return new_prev_bev, bbox_list

    # 【函数入口】VAD.simple_test_pts
    # Head.forward先预测；get_bboxes调用coder解码，得到可用的检测框/运动轨迹/地图。
    # 随后 CPU/no_grad 区域用GT计算运动与规划指标。GT参与评估，不参与前面Head的未来预测。
    # 该实现显式 assert batch_size=1；ego_fut_cmd选择评估分支；三个分支不是三种随机置信模式。
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
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]

        # 调用VADHead对象，经nn.Module.__call__进入VADHead.forward；返回预测字典，未来GT未传入。
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        bbox_results = []
        # enumerate同时给出序号与元素；序号可能表示解码层/尺度/样本，按循环对象区分，不都代表时间。
        for i, (bboxes, scores, labels, trajs, map_bboxes, \
                map_scores, map_labels, map_pts) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            bbox_result['trajs_3d'] = trajs.cpu()
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            # append把当前对象加入Python列表；没有立即堆叠Tensor，最终stack/cat时才形成规则轴。
            bbox_results.append(bbox_result)

        # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        score_threshold = 0.6
        # with torch.no_grad作用域内不记录自动求导图；eval是层模式，两者不是同一个开关。
        with torch.no_grad():
            # 创建副本；clone默认保留梯度联系，deepcopy用于独立容器/模块，不能把两者都说成detach。
            c_bbox_results = copy.deepcopy(bbox_results)

            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            bbox_result = c_bbox_results[0]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            gt_bbox = gt_bboxes_3d[0][0]
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            gt_label = gt_labels_3d[0][0].to('cpu')
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            fut_valid_flag = bool(fut_valid_flag[0][0])
            # filter pred bbox by score_threshold
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            mask = bbox_result['scores_3d'] > score_threshold
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]

            # 测试匹配结果[G]，每个GT对应一个预测索引，-1表示未匹配
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            matched_bbox_result = self.assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label)

            # 测试指标累计字典，不是训练loss；下游评估会按计数汇总
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result,
                matched_bbox_result, mapped_class_names)

            # ego planning metric
            # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            ego_fut_preds = bbox_result['ego_fut_preds']
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            ego_fut_trajs = ego_fut_trajs[0, 0]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]
            # 按命令选出的自车轨迹，cumsum后为[T,2]累计位置，用于测试指标
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            # 按命令选出的自车轨迹，cumsum后为[T,2]累计位置，用于测试指标
            # cumsum沿指定轴累加；这里轨迹时间轴的逐步位移变成相对起点累计位置，shape不变。
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            # cumsum沿指定轴累加；这里轨迹时间轴的逐步位移变成相对起点累计位置，shape不变。
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return outs['bev_embed'], bbox_results, metric_dict

    # 【函数入口】VAD.map_pred2result
    # 把一个样本的解码地图框、分数、类别、点坐标整理为 CPU 字典；每项实例数由coder筛选结果决定。
    # 这里没有重新训练/分割地图，.cpu/.to只是设备搬运，不改变几何维度。
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
        # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
        result_dict = dict(
            map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'))

        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if attrs is not None:
            # 转换设备或dtype，通常不改变shape；CPU测试搬运与训练梯度路径应分开理解。
            result_dict['map_attrs_3d'] = attrs.cpu()

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return result_dict

    # 【函数入口】VAD.assign_pred_to_gt_vip3d
    # 测试指标用的匹配，不是训练 HungarianAssigner3D 的分类+框成本。
    # 按预测与GT中心的米制距离构造 [Npred,G] 矩阵，屏蔽非动态类别及超过阈值的组合。
    # 返回 [G] 的预测索引，-1表示无匹配；此索引用于下面计算运动指标。
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
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        dynamic_list = [0,1,3,4,6,7,8]
        # 测试匹配结果[G]，每个GT对应一个预测索引，-1表示未匹配
        # 分配指定shape的常量Tensor，device/dtype与调用指定值一致；除非另设requires_grad，不是训练参数。
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        # 当前GT框中心的二维坐标[G,2]，测试匹配使用米制距离
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        gt_centers = gt_bbox.center[:, :2]
        # 解码预测框中心[Npred,2]，与GT处于同一局部参考系
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        pred_centers = bbox_result['boxes_3d'].center[:, :2]
        # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        dist[pred_not_dyn] = 1e6
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        dist[:, gt_not_dyn] = 1e6
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for i in range(len(r_list)):
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                matched_bbox_result[c_list[i]] = r_list[i]

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return matched_bbox_result

    # 【函数入口】VAD.compute_motion_metric_vip3d
    # 测试中累计 car/pedestrian 的计数、ADE/FDE/MR 等，后续由评估流程汇总归一化。
    # 先按匹配取目标，再累计逐步位移并加当前框中心，比较同一局部坐标下的未来位置。
    # ADE可使用有效前缀；FDE等完整未来指标仅在有效步数=T时计入；与训练选模式实现不同。
    # 此函数会改类别Tensor，调用处在copy后的测试结果上操作；不要把它当训练target函数。
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
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        motion_cls_names = ['car', 'pedestrian']
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        
        # 测试指标累计字典，不是训练loss；下游评估会按计数汇总
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        metric_dict = {}
        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for met in motion_metric_names:
            # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
            for cls in motion_cls_names:
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict[met+'_'+cls] = 0.0

        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        veh_list = [0,1,3,4]
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
        ignore_list = ['construction_vehicle', 'barrier',
                       'traffic_cone', 'motorcycle', 'bicycle']

        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for i in range(pred_bbox['labels_3d'].shape[0]):
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            pred_bbox['labels_3d'][i] = 0 if pred_bbox['labels_3d'][i] in veh_list else pred_bbox['labels_3d'][i]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            box_name = mapped_class_names[pred_bbox['labels_3d'][i]]
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if box_name in ignore_list:
                continue
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if i not in matched_bbox_result:
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict['fp_'+box_name] += 1

        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for i in range(gt_label.shape[0]):
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            box_name = mapped_class_names[gt_label[i]]
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if box_name in ignore_list:
                continue
            # GT时间有效性mask，来源标注；取样缺失未来时为0，不是推理预测的可见性
            # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
            gt_fut_masks = gt_attr_label[i][self.fut_ts*2:self.fut_ts*3]
            num_valid_ts = sum(gt_fut_masks==1)
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if num_valid_ts == self.fut_ts:
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict['gt_'+box_name] += 1
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict['cnt_ade_'+box_name] += 1
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                m_pred_idx = matched_bbox_result[i]
                # 目标逐步未来位移GT，所有未来都在当前帧固定参考坐标系下差分
                # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
                gt_fut_trajs = gt_attr_label[i][:self.fut_ts*2].reshape(-1, 2)
                # 目标逐步未来位移GT，所有未来都在当前帧固定参考坐标系下差分
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                # reshape/view重排形状而不改变元素总数；-1由其余轴自动推算。view通常要求相容的内存布局。
                pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                # 目标逐步未来位移GT，所有未来都在当前帧固定参考坐标系下差分
                # cumsum沿指定轴累加；这里轨迹时间轴的逐步位移变成相对起点累计位置，shape不变。
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                # cumsum沿指定轴累加；这里轨迹时间轴的逐步位移变成相对起点累计位置，shape不变。
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                # 目标逐步未来位移GT，所有未来都在当前帧固定参考坐标系下差分
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                # unsqueeze或None索引增加长度1轴，方便广播；不代表新增样本、目标或未来时间。
                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                # 沿dim做极值归约，返回(极值,索引)；[0]取数值、[1]取索引，是否保留轴由keepdim决定。
                ade = ade.min()

                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict['ADE_'+box_name] += ade
                # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
                if num_valid_ts == self.fut_ts:
                    # 沿dim做极值归约，返回(极值,索引)；[0]取数值、[1]取索引，是否保留轴由keepdim决定。
                    fde = dist[:, -1].min()
                    # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                    metric_dict['cnt_fde_'+box_name] += 1
                    # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                    metric_dict['FDE_'+box_name] += fde
                    # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
                    if fde <= match_dis_thresh:
                        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                        metric_dict['hit_'+box_name] += 1
                    else:
                        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                        metric_dict['MR_'+box_name] += 1

        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return metric_dict

    ### same planning metric as stp3
    # 【函数入口】VAD.compute_planner_metric_stp3
    # 测试规划：由GT目标构造占据栅格，仅用于ST-P3风格碰撞评估，不是VAD的栅格地图预测分支。
    # 输入累计自车位置 [1,T,2]；按前2/4/6步对应1/2/3秒评估L2与碰撞，fut_valid_flag控制计入。
    # 这些计量不是规划训练loss，也不参与反向传播。
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        # 测试指标累计字典，不是训练loss；下游评估会按计数汇总
        # 创建Python列表/字典容器，后续append或按key读取；它本身不是带统一shape的Tensor。
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
        # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        # assert校验当前实现的形状/取值假设；失败会抛异常，不能删掉断言就认为任意输入已受支持。
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        # 按可选参数/模块是否存在选择分支；None表示未提供，不是数值0，也不是空Tensor。
        if self.planning_metric is None:
            # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
            self.planning_metric = PlanningMetric()
        # 若调用对象是nn.Module，会经__call__进入对应forward；若是普通方法则直接调用，按左侧结果继续传递。
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        # 由真实目标生成的测试占据栅格，仅用于碰撞评估，不是感知网络的地图输出
        occupancy = torch.logical_or(segmentation, pedestrian)

        # for依次处理可迭代对象；循环中的i/lvl是索引，_表示不使用本次编号，层参数是否共享看构造方式。
        for i in range(future_second):
            # 条件分支：只执行满足条件的一侧；配置开关不同，实际走到的计算与shape也可能不同。
            if fut_valid_flag:
                cur_time = (i+1)*2
                # detach共享或复制后的数值不再沿这条分支反传；不等于永久冻结来源模块的全部参数。
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                # detach共享或复制后的数值不再沿这条分支反传；不等于永久冻结来源模块的全部参数。
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
            else:
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                # 按索引读取/写入对应项；整数索引通常去一轴，切片保留轴，布尔mask只保留条件为True的行。
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        # return结束当前函数并把结果交回调用处；逗号分隔通常是tuple，调用方按同一顺序解包。
        return metric_dict

    # 【函数入口】VAD.set_epoch
    # 由 CustomSetEpochInfoHook.before_train_epoch 调用，保存当前轮数并同步给 Head。
    # 这是训练控制状态，不是可学习参数，也不是样本时间戳。
    def set_epoch(self, epoch): 
        # 把配置或计算结果保存为当前对象属性，后续方法通过self访问；普通数值属性不是可训练参数。
        self.pts_bbox_head.epoch = epoch
