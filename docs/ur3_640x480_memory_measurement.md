# 四卡 RTX 4090：640×480 实测显存报告

测量日期：2026-10-16。GPU0 的旧推理服务已按用户授权停止。使用四张空闲 RTX 4090，单卡驱动报告总显存 24564MiB。

## 结果

640×480 输入已在四卡真实训练中跑通，并不需要退回只训练动作专家。当前可行候选是局部解冻 VLM，并使用 FSDP2、BF16 和梯度检查点。

| 指标（显存取四卡最大值） | 最后两层解冻，无 offload | 最后四层解冻，activation offload |
|---|---:|---:|
| 可训练参数 | 2,139,952,183 | 2,341,813,815 |
| 每卡峰值 allocated | 20.313GiB | 18.729GiB |
| 每卡峰值 reserved | 22.816GiB | 21.943GiB |
| 驱动层采样占用峰值 | 24072MiB，23.508GiB | 23180MiB，22.637GiB |
| 最小采样显存余量 | 492MiB | 1384MiB |
| 优化器步数／每个 rank | 6 | 6 |
| 跳过前两步后的 StepTime 中位数 | 21.782 秒 | 29.113 秒 |
| 抽检梯度有限、参数全局发生更新 | 通过 | 通过 |
| OOM | 未发生 | 未发生 |

allocated 是 PyTorch 实际分配的张量内存；reserved 还包括 allocator 缓存。驱动层占用包含 CUDA/NCCL 等额外内存，因此更接近整张卡的占用。驱动每秒采样一次，可能漏掉更短的峰值，不能把采样余量当成严格保证。

两种配置同时改变了解冻层数与 activation offload；表格比较的是两套实际方案，不能将差值全部归因于 offload。

上述是保留第三个占位相机槽时的实测。按短跑中位数折算，5000 步纯训练约为 30.3 小时／40.4 小时，均不含保存、评估及长时间负载变化。后续双相机优化已经测通，目前优先推荐下文方案。

## 补充：双相机优化实测

去掉始终被 mask 的第三个相机槽，仅保留两路真实相机，VLM 输入变为 `[1,2,1200,1536]`。每路仍为完整 640×480、300 个视觉 token，四层解冻及所有辅助模块保持不变。以下两组各完成 6 步完整训练并通过参数更新检查：

| 指标 | 双相机，无 offload | 双相机，无 offload，expandable segments |
|---|---:|---:|
| 最大 allocated | 20.076GiB | 19.972GiB |
| 最大 reserved | 22.736GiB | 21.895GiB |
| 驱动采样峰值 | 23992MiB | 23130MiB |
| 最小采样余量 | 572MiB | 1434MiB |
| 中位 StepTime（排除前两步） | 17.834 秒 | 18.119 秒 |

目前首选 **双相机＋四层解冻＋关闭 offload＋`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**。与前三槽四层 offload 方案相比，短跑耗时减少约 38%，显存采样余量接近。allocator 设置需要在 Python 启动前生效，其减少碎片的机制见 [PyTorch 文档](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-memory-management)。

新结果目录分别为 `/data-teamA/outputs/ur3_memory_640x480_last4_two_views`、`/data-teamA/outputs/ur3_memory_640x480_last4_two_views_expandable`；均未测量 checkpoint 保存／恢复。正式候选参数和验证计划见 [推荐方案](ur3_4090_recommended_training.md)。

## 实际测量条件

- 原始模型：`/data-teamA/models/lingbot-vla-v2-6b`，每次从原始预训练权重开始，严格检查预训练参数加载和冻结状态。
- 数据：`/data-teamA/datasets/ur3_eef_relative_20261016`；训练划分 81 条、36688 帧。
- Manifest：`assets/training_data/ur3_20261016_split_seed42.json`；使用对应的新归一化统计。
- VLM 输入宽 640、高 480，没有先缩小到 256。训练与推理入口均增加了统一的高／宽配置支持。
- 每张图 `image_grid_thw=[1,30,40]`，合并后 300 个视觉 token。
- 模型输入 `images.shape=[1,3,1200,1536]`；两路实际相机＋一路 masked placeholder，与原生训练管线一致。
- 视觉编码器、词嵌入及未选择的语言层冻结；动作专家、state/action 投影及辅助头继续训练。
- FSDP2 全分片、BF16、梯度检查点开启；整模型 `use_compile=false`。原生 FlexAttention 仍使用其内部编译内核。
- 每卡 micro batch=1，梯度累积=4，全局 batch=16，action chunk=50。
- 保留 MoGe、LingBot-Depth、DINO 视频教师及当前／未来辅助损失，没有通过删除辅助模块降低显存。
- 四层 offload 方案启用现有 `enable_activation_offload`，`activation_gpu_limit=2.0`。该值控制重算阶段保存张量的策略，不是整个模型的显存上限；没有启用参数／优化器 CPU offload。
- AdamW；VLM LR=`1e-6`，动作专家 LR=`1e-5`，投影及辅助头 LR=`2e-5`，关闭 MoE 自动 LR 倍率。
- 测量沿用旧配置 weight decay=0；短跑预热 1 步。这是显存测试条件，不是正式 5000 步的完整超参数建议。

每组均执行 24 个 micro batch、6 个 optimizer step，包含真实前向、反向、梯度累积、裁剪及 Adam 状态初始化。记录了输入形状、损失、抽检梯度与参数更新。FSDP 的部分输出分片完全属于被 mask 的机器人维度，允许这些局部分片保持不变；参数更新在四卡之间汇总验证。

## 为非正方形输入补齐的适配

1. 数据加载与推理端统一使用 `data.image_height`／`data.image_width`，旧 `img_size` 正方形配置保持兼容。
2. 深度、未来视频和当前视频辅助头动态读取第一张图的完整视觉 token，不再固定截取 64 个 token。V2 的任务 query 位置原本已动态计算，继续使用原有逻辑。
3. 深度教师在 4:3 图像上产生约 14×18=252 个 patch 特征，而预训练辅助头输出 16×16=256 个。现将教师二维特征插值到 16×16，再计算损失；VLM 输入仍为 640×480，预训练头的形状和权重结构保持兼容。
4. UR3 包装入口支持 `--vlm-train-last-layers`，在 FSDP 分片前选择解冻范围，并使用独立的小学习率参数组。

## 限制与后续

这是短跑显存测量，不是 5000 步稳定性验证，也没有评估定位或真机抓取成功率。

本次在训练完成、准备保存 checkpoint 时有意停止，未保存这些短跑权重，也未测量 DCP／HF 导出峰值及断点恢复。正式长训前仍需要验证保存／恢复，并建议增加到 50～100 步检查稳定性。

没有测量整模型 `use_compile=true` 的显存、速度或兼容性，不应依据本报告宣称编译还能进一步省显存。

调试中遇到的非正方形深度特征形状错误，以及按局部分片判断更新的误报，均已修正；正式结果只取下面两套全部 rank 成功完成的运行。

## 文件与重现

- 两层结果目录：`/data-teamA/outputs/ur3_memory_640x480_last2_v3`。
- 四层 offload 结果目录：`/data-teamA/outputs/ur3_memory_640x480_last4_offload`。
- 各目录保留 `train.log`、`memory_rank0..3.json`、`weight_load_rank0..3.json`、`nvidia-smi.csv` 及实际运行配置。
- 汇总：[ur3_640x480_memory_measurement.json](ur3_640x480_memory_measurement.json)。
- 真实数据与辅助区间检查：[ur3_resolution_checks.json](ur3_resolution_checks.json)。

重测时使用新的输出目录，且四张 GPU 应空闲：

```bash
cd /home/teamA/lingbot-vla-v2

bash tools/run_ur3_memory_probe.sh \
  --output-dir /data-teamA/outputs/ur3_memory_640x480_last2_repeat \
  --steps 6 --height 480 --width 640 \
  --vlm-train-last-layers 2

bash tools/run_ur3_memory_probe.sh \
  --output-dir /data-teamA/outputs/ur3_memory_640x480_last4_offload_repeat \
  --steps 6 --height 480 --width 640 \
  --vlm-train-last-layers 4 --activation-offload
```

上述命令只测量显存，不是正式训练启动命令。
