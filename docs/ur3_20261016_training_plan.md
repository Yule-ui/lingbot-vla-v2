# 20261016 新数据检查与四卡 4090 训练计划

最新首选已根据四卡实测更新为双相机、640×480、四层解冻、关闭 offload 并启用 expandable segments，详见 [当前推荐方案](ur3_4090_recommended_training.md)。下文保留前期数据检查及训练路线推导。

## 结论与边界

新数据已通过数值、元数据、视频抽样以及真实 LeRobot/VLA 数据管线检查，格式和映射可用于 LingBot-VLA V2。正式训练前仍需核对采集端的控制增益、限幅、平滑和执行时序；数据文件本身无法证明动作命令的执行语义或全部演示的成功率。

已完成数据检查、选择性解冻与分组 LR 适配，并补齐非正方形输入支持。2026-10-16 已完成四卡 4090 的 640×480 短跑实测：最后两层无 offload、最后四层加 activation offload 均通过 6 个优化器步骤；详见 [实测报告](ur3_640x480_memory_measurement.md)。尚未启动长时间训练或验证新策略的 checkpoint 保存／恢复，不能直接把旧命令的 `train_expert_only` 改为 false 后开训。

## 数据检查

| 项目 | 结果 |
|---|---|
| 数据路径 | `/data-teamA/datasets/ur3_eef_relative_20261016` |
| 格式 | LeRobot v3，25 Hz |
| 轨迹与帧数 | 90 条、40847 帧，约 27.23 分钟 |
| 任务 | block 0、2、6，各 30 条 |
| 观测／动作 | 17 维 state、8 维相对 EEF action |
| 视频 | 两路 640×480 RGB、AV1，共 18 文件 |
| 数值检查 | 无非有限 state/action；索引、时间戳、episode 边界、任务文本一致 |
| 四元数 | 范数误差小于约 4e-8；rotvec 与 quaternion 对应一致；轨迹内无符号跳变 |
| 夹爪 | 所有轨迹均有明显开合目标变化；动作范围约 0.502～1.0 |
| 视频检查 | 所有文件元数据可读，每条轨迹每路首／中／末帧抽样，共解码 540 帧 |
| 实际模型特征检查 | 243 个训练边界、27 个验证边界通过；55 维 mask、8 维动作反归一化还原通过 |

检查不是逐帧视频损坏扫描。已目视检查三个任务各三条代表轨迹的接触图，抽样末帧可见对应目标方块位于目标区域；这不等于全部 90 条轨迹已经人工标注成功。

完整报告：[数据审计](ur3_20261016_data_audit.json)。接触图：[block 0](ur3_20261016_contact_task1.png)、[block 2](ur3_20261016_contact_task0.png)、[block 6](ur3_20261016_contact_task2.png)。

### 必须确认的控制比例

- action 平移向量范数中位数：6.18 mm／周期；TCP 实际相邻帧位移中位数：1.58 mm。
- 命令与实测位移在滞后 1 帧时相关系数约 0.9935，最小二乘比例约 0.254。
- 这提示存在执行缩放、servo 跟随或平滑，而不是直接证明标注错误。需要采集代码确认。不要依据相关分析直接将 action 乘 0.25 或修改训练标签。
- 推理必须复用采集时的 base 平移、局部 quaternion 旋转、夹爪绝对目标、控制频率、增益及限幅逻辑。额外增加或遗漏缩放都可能造成失败。

### 任务编号已经变化

```text
新 task_index 0 → block 2
新 task_index 1 → block 0
新 task_index 2 → block 6
```

旧数据中编号 0、1 对应顺序相反。必须从新 `meta/tasks.parquet` 读取任务文本，不能沿用手写编号映射。

## 已准备好的数据产物

- 划分：`assets/training_data/ur3_20261016_split_seed42.json`。
- 统计：`assets/norm_stats/ur3_robotiq_eef_relative_20261016.json`。
- 特征检查配置：`configs/vla/ur3_20261016/data_check.yaml`，仅用于数据检查，不是新训练策略的完整实现。
- 固定 seed=42，按任务各取 27／3 条，训练 81 条／36688 帧、验证 9 条／4159 帧。
- 验证 episode：14、15、23、40、44、49、73、74、84。
- 统计只使用训练帧，每帧一次；使用 float64 计算和固定参考四元数半球，避免接近常量的四元数分量方差计算失真。
- 保持旧 manifest、旧归一化统计和现有推理服务使用的文件独立。

## 上次实际训练范围

根据原始权重张量和冻结代码，上次冻结 Qwen 视觉语言主干，训练约 19.38 亿参数，其中动作专家约 17.87 亿、投影及辅助模块约 1.51 亿。它并非只有最后的线性 action head。

因此不能把效果不好直接归因于 action head 太小。需要同时检查执行语义、实时推理延迟、视觉／任务依赖和训练覆盖。该数据仍只有约半小时演示，扩大可训练范围并不保证泛化。

### 针对“空抓”的检查顺序

1. 在实际送入模型的图像上检查方块、编号和目标区是否清晰，确认全局／腕部相机顺序、RGB、有效 mask 和训练／推理预处理一致。当前数据加载会先把 640×480 缩放到 256×256；提高后续 processor 分辨率不能恢复首次缩放已经丢失的细节。全局视角包含较多机械臂和背景，可先验证固定工作区 ROI 能否保留所有方块与目标区，再决定是否进行同样预处理的重训。
2. 对照采集端与执行端的平移增益、平滑、限幅、四元数乘法方向及 25 Hz 时序。约 0.254 的位移比例仅是待追查的线索，不能直接据此修正 action。同步检查新 task_index 到文本的映射。
3. 用相近机械臂起始姿态、不同方块位置的观测比较固定随机种子下的预测首段轨迹；再改变任务文本，检查是否朝对应目标改变。只观察输出数值变化不足以证明定位正确。离线误差下降也不能替代不同摆放位置的闭环成功率。
4. 在这些条件成立后，比较动作专家训练与下述局部 VLM 解冻方案。若演示中方块位置缺乏变化，需要补充不同位置、干扰物和接近阶段的成功演示；解冻参数不能补足训练数据未覆盖的场景。

## 首选训练方案：动作专家＋视觉语言主干的最后四层

选择这一方案是为了在现有原生权重、FSDP2 和推理导出结构上做有限改动，并允许视觉语言表示适配当前场景。

这里解冻的是 VLM 内处理图像与文本 token 的语言解码器上层，而不是像素视觉编码器。代码在每层将 VLM 与动作专家的 Q/K/V 拼接做注意力，动作损失可以沿 VLM 的 K/V 路径反传；因此冻结 ViT 不等于动作模块看不到图像，局部解冻也有实际的视觉条件适配路径。但它不能保证修复已经丢失的图像细节或错误执行语义。

| 模块／参数 | 建议 |
|---|---|
| 起点 | `/data-teamA/models/lingbot-vla-v2-6b` 原始预训练权重 |
| Qwen vision encoder、token embedding、语言前 32 层 | 冻结 |
| Qwen language_model 第 32～35 层 | 全量解冻，学习率起点 `1e-6` |
| MoE 动作专家 | 保持可训练，学习率起点 `1e-5` |
| state/action 投影及辅助头 | 保持可训练，学习率起点 `2e-5` |
| MoE 专家 LR 自动倍率 | 关闭，采用明确的参数分组，避免意外再乘约 2.83 |
| 优化器 | AdamW；weight decay 可从 0.01 开始，bias/norm 不衰减 |
| 分布式与精度 | 四卡 FSDP2 全分片、BF16、梯度检查点、`init_device=meta` |
| batch | 每卡 1、梯度累积 4、全局 16 |
| 图像／动作窗口 | 首轮保持两路 256 输入和 chunk=50，以减少实验变量 |
| 学习率日程 | 预热后 constant，方便按阶段恢复；预热步数保持一致 |
| 深度／视频辅助模块 | 首轮保留原结构、教师和权重，不通过删模块破坏加载 |
| 编译 | 首轮关闭，先验证数值与显存 |

Qwen 最后四层共约 4.037 亿参数，比上次额外增加约 1.13 GiB／卡的梯度与 Adam 状态；反向激活和临时 all-gather 开销还要另计。原方案已实测 reserved 峰值约 20.85 GiB／卡，因此新方案必须通过真实四卡试跑后才能确认是否装入 24 GB。

如果试跑显存不足，先测试现有 activation offload 支持；它会增加 CPU 内存和传输时间。再考虑缩减到最后两层，或下面的 LoRA 备选。不能把四卡显存简单相加当成单卡可用显存。

若该方案仍然对方块位置缺乏正确响应，再单独试验视觉 merger，或 ViT 最后 2～4 层 attention 的 LoRA。视觉侧从更浅位置引入可训练参数，会让梯度继续穿过后面的冻结语言层；新增参数少不代表激活开销小，必须重新实测四卡峰值。首轮不同时解冻整套视觉编码器、扩大图像分辨率和扩大语言解冻范围，以便判断哪项改变有效。

### 更省训练状态的备选

只向 Qwen 语言模块最后八层的 q/k/v/o attention 投影注入 rank=16、alpha=32 的 LoRA，同时继续训练原动作专家和投影模块；这些 LoRA 新增约 262 万参数。

本仓库目前没有完整的 LoRA 训练与导出通路。需要实现精准的模块范围、FSDP2/meta 权重加载、adapter 初始化和梯度审计；不能让 PEFT 的默认冻结行为把动作专家也冻结。导出时合并 LoRA，才能沿用现有严格加载的推理服务。合并方法见 [PEFT 官方说明](https://huggingface.co/docs/peft/developer_guides/lora#merge-lora-weights-into-the-base-model)。

不建议直接对全部约 63.76 亿参数做普通 FP32 master＋AdamW 全量解冻：仅 master 权重、梯度和两份优化器状态，四卡分摊后就约 23.75 GiB／卡，还没有加入激活、教师和通信临时内存。

## 当前代码实施与验证状态

1. 已在 `tools/run_ur3.py` 增加 `--vlm-train-last-layers`，FSDP2 分片前冻结 Qwen 并选择解冻语言层，默认 0 保留旧策略。
2. 已扩展 `tools/ur3_adapters.py`：检查冻结范围、meta materialization 和严格预训练权重数值，按 rank 保存解冻层数及可训练参数量。
3. 已为选择性解冻增加三组 LR：VLM `1e-6`、动作专家 `1e-5`、投影／辅助模块使用基础 LR（本次为 `2e-5`），关闭原 MoE 自动倍率。
4. 已在实际短跑中检查抽样梯度有限、参数全局更新；保存／恢复后的策略和优化器分组一致性仍待验证。最后一层 VLM 的 K/V 可直接影响动作，但其输出投影／MLP 的最终 prefix 输出主要由深度／视频辅助损失训练，不能要求每个解冻参数都从 action loss 单独获得非零梯度。
5. 新运行的 checkpoint 保留规则尚未实施；独立 manifest、归一化和输出目录已准备。不要依赖目前日志中的自动评估计数，它没有执行真实评估。

`configs/vla/ur3_20261016/data_check.yaml` 仍是数据检查配置；显存测量通过专用入口明确覆盖分辨率、解冻范围、batch 和测量步数。正式长训仍需独立配置与保存／恢复验证。

## 训练预算、评估与磁盘

- 全局 batch=16 时，每遍训练集 **2293 步**；2000、5000、10000 步分别约 **0.87、2.18、4.36 遍**。
- 先进行 50～100 步四卡试跑，必须包含 backward、optimizer step、保存和恢复检查。只做模型加载不能证明显存够用。
- 然后按累计 **2000 → 5000 → 10000 步**推进。每段结束释放训练显存，另起 BF16 推理评估，按验证与真机表现决定是否继续。后两段从新运行的 checkpoint 恢复，不能每段重新从预训练起点训练。
- 建议每 1000 步保存供故障恢复，保留最近两份完整 checkpoint 和一份已评估的最佳 HF 权重。先验证新 checkpoint 完整，再清理同一新运行的旧文件；源模型保持独立。
- 用户要求清理旧 `/data-teamA/outputs/ur3` 训练结果后，数据盘剩余约 **293 GiB**（2026-10-16 再次检查）。新策略每套完整 DCP＋HF 预计约 **53 GiB**。建议保留最近两份完整 checkpoint 和一份最佳 HF；写入下一份完整 checkpoint 时预计约 171 GiB，仍需预留日志、缓存和实际大小误差。保存频率不能代替保留上限。
- 旧策略基准约 17.8 秒／步，5000 步约 25 小时。新策略尚未测速；若实测 25 秒／步，5000 步约 34.7 小时；若 30 秒／步，约 41.7 小时。用试跑预热后的中位数重新估算。
- 容器内存上限 200 GiB，大量现有占用为可回收文件缓存；仍需测量 activation offload 和 HF 导出峰值。
- GPU0 的旧推理服务已在 2026-10-16 按用户授权停止，以完成四卡显存测量。旧 checkpoint 已删除，不能依赖旧路径重启。

### 评估内容

使用新验证 manifest，对平移向量误差、单位四元数旋转误差和夹爪误差分别统计，并与不移动／保持夹爪基线比较。对任务替换、图像遮挡或替换做对照，检查策略是否使用视觉和任务信息；动作发生变化本身不证明目标选择正确。

按新物体摆放位置做小规模闭环验证，记录三项任务各自的成功率、错误物体、抓取偏差、夹爪时机和旋转异常。当前服务已启用编译，需要重新测量其端到端延迟并验证与 25 Hz 控制的动作时间线。此前约 0.77 秒仅为未编译版本的旧基准，不能当作当前服务延迟；训练范围变化也不会自动修复控制时序。

若固定工作台视角下目标数字太小，可另开对照实验，对全局相机做保留全部方块及目标区的固定 ROI，再缩放至原输入尺寸。ROI 必须在训练和推理的初次缩放之前一致应用；它是需要实现和验证的新预处理，当前数据检查配置没有启用 ROI。

## 重现本次数据检查

```bash
source /data-teamA/miniforge3/etc/profile.d/conda.sh
conda activate lingbotvla
cd /home/teamA/lingbot-vla-v2
export HF_HOME=/data-teamA/cache/huggingface
export HF_DATASETS_CACHE=/data-teamA/cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python tools/check_ur3_data.py \
  --manifest assets/training_data/ur3_20261016_split_seed42.json \
  --config configs/vla/ur3_20261016/data_check.yaml --split train

python tools/check_ur3_data.py \
  --manifest assets/training_data/ur3_20261016_split_seed42.json \
  --config configs/vla/ur3_20261016/data_check.yaml --split validation
```

后续评估及服务也必须传入新 manifest：

```text
--manifest assets/training_data/ur3_20261016_split_seed42.json
```

资源推导、参数统计和已通过检查见 [ur3_20261016_training_resources.json](ur3_20261016_training_resources.json)。原生配置参考：[官方训练说明](https://github.com/Robbyant/lingbot-vla-v2/blob/main/configs/vla/Training_Config.md)。
