VAD 三文件中文注释包
====================

当前状态
--------
远程SSH端口连接被拒绝，未认证、未读取或修改远程服务器文件。
注释基于 /workspace/VAD 中已核对的VAD仓库源码，不声称与你服务器的调试版逐字一致。
因此提供安装前AST校验，发现可执行逻辑差异会拒绝覆盖，保留你的调试或修改。

文件
----
annotated/：新增中文注释的三个Python文件。
originals/：注释使用的原始基线，供比较，不需要手动覆盖服务器。
validation.json：函数数目、注释行数和AST一致性结果。
install_annotations.py：在服务器执行的备份和安装脚本。

注释范围
--------
VAD.py：模型注册/继承、图片特征提取、训练历史队列、metadata、Head/loss调用、
测试BEV缓存、结果解码、运动及规划评估指标。
VAD_head.py：查询/线性层构建、参数共享、前向维度、检测地图解码、LaneNet、运动自注意力、
运动-地图交互、自车规划、匹配索引、GT、mask、loss、地图筛选及坐标约定。
VAD_transformer.py：图像尺度展平、车辆运动对齐、BEV编码、检测与地图查询拆分、
参考坐标初始化/逐层更新、detach、通用交互层、Q/K/V及外部注意力实现的位置。
说明区分配置常量/可训练参数/中间特征；Tiny尺寸是示例，实际以配置和输入为准。
只增加 # 注释；原有语句和docstring保留，旧英文模板不准确处在中文注释中说明。

如何使用
--------
1. 解压后可直接在VS Code阅读annotated目录，不安装也不影响现有项目。
2. 若要替换服务器文件，通过MobaXterm/SFTP把整个目录上传到：
   /root/autodl-tmp/vad_annotation_delivery
3. 在服务器执行：
   conda activate vad
   python /root/autodl-tmp/vad_annotation_delivery/install_annotations.py --check-only
4. 三份文件均通过后执行：
   python /root/autodl-tmp/vad_annotation_delivery/install_annotations.py
5. 脚本先备份三份现有文件，再替换；备份路径打印在终端。
   如发生写入异常，尝试恢复已替换文件；没有通过校验时不会替换任何文件。
6. 如果提示可执行代码与基线不同，请保留服务器文件，提供当前源码重新注释。
   不要删除保护检查或强制覆盖；差异可能是你加的print或实际逻辑修改。
7. 已运行的Python进程不会自动重载修改；阅读/下一次运行使用新的文件。

验证边界
--------
已完成UTF-8读取、Python语法编译、AST结构比较及安装脚本隔离目录测试。
AST一致意味着可执行结构和docstring未改，不表示运行时源码行号保持不变。
未在你的AutoDL服务器运行训练或推理验证；本目录用于GitHub交付，安装脚本不会执行Git操作。
本包不包含登录密码、数据集、依赖或模型权重。
