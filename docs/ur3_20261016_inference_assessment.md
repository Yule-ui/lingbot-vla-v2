# UR3 第二次训练后的推理评估（2026-10-18）

## 结论与硬件选择

建议先采用 **单张服务器 RTX 4090 运行 LingBot-VLA V2，真机电脑使用 LeRobot 采集观测并执行控制**。训练用四卡，当前推理只需一卡。

真机电脑尚未实际接入，本报告暂按普通 RTX 4060 8GB 判断。NVIDIA 官方规格中 RTX 4060 为 8GB，RTX 4060 Ti 则有 8GB/16GB 版本，不能混为一谈。[官方规格](https://www.nvidia.com/en-eu/geforce/graphics-cards/40-series/rtx-4060-4060ti/)

| 部署硬件 | 当前完整 BF16 模型的判断 | 证据与限制 |
|---|---|---|
| RTX 4060 8GB / 其他 8GB 显卡 | 现有原生加载路径不能整模型驻留 GPU | 导出 BF16 张量本身约 11.876GiB，已经超过显存容量 |
| RTX 4060 Ti 16GB | 从当前占用看有容量可行性；尚不能确认控制实时性 | 4090 上 CUDA 峰值约 12.35GiB，但没有 4060 Ti 的实机速度、可用显存或系统内存数据 |
| RTX 4090 24GB | 已实测可行，推荐作为本轮实验基线 | 单卡严格加载、完整离线验证及本机 WebSocket 测试均通过 |

不是必须使用 4090 这一型号；现阶段明确不能直接使用的是 8GB 显卡上的完整 BF16 原生模型。降低分辨率、减少返回动作长度或启用 compile，都无法消除权重本身的 11.88GiB 下限。

量化、裁剪或 CPU offload 可能改变容量需求，但当前 `LingbotVLAv2Server` 使用整模型 `.to(bfloat16).cuda()`，未提供量化/自动分层卸载加载流程。模型还含自定义动作专家与融合算子，不能假设给标准语言模型加一个量化参数就能完成适配。若以后必须本地运行，需要单独实现并重新验证延迟和动作精度；本次没有修改量化或控制逻辑。

## 训练及权重检查

- 5000 个 optimizer step 连续完整，解析到的训练数值全部有限，跳过 batch 为 0；四卡完成记录齐全，最后 100 步平均 loss 为 0.111958。
- 2500/5000 步都有 DCP 模型、优化器及 extra_state，同时有完整 HF 导出；各 HF 导出均为 1708 个 BF16 张量、约 63.76 亿元素，张量数据约 11.876GiB。
- safetensors 索引、张量名称和分片文件长度一致；两个 checkpoint 均实际严格加载成功。
- 与初始权重转换到 BF16 后比较：语言第 0 层抽检参数一致，第 32、35 层及 state/action 投影抽检参数已更新。这验证了局部解冻确实产生更新，但不是每个参数都逐一比对。
- 训练日志中的 `eval_failed: 3` 是当前导出汇总把未提交评估的记录计为失败；同一汇总 `hf_failed: 0`、`failures: []`。本次独立评估已实际运行成功，不将该汇总字段解释为 checkpoint 损坏。

报告：[训练结果](ur3_20261016_training_result.json)、[参数抽检](ur3_20261016_checkpoint_parameter_check.json)。

## 完整离线动作评估

两个 checkpoint 使用相同 9 条独立验证轨迹、4159 帧、88 次查询。每次输入当时的真实图像及状态，返回 50×8 动作；BF16、不开整模型 compile、10 次去噪、按查询固定随机种子。尾部 padding 不计入误差，旋转指标先单位化并消除 q/-q 符号等价影响。

| 模型 | 平移向量 RMSE | 旋转平均误差 | 旋转 P95 | 夹爪 MAE |
|---|---|---|---|---|
| 2500 步 | 3.917mm | 0.2757° | 1.5153° | 0.03749 |
| 5000 步 | 3.709mm | 0.2314° | 1.4653° | 0.03715 |
| 不移动、保持姿态与当前夹爪基线 | 6.110mm | 0.2003° | 1.6648° | 0.12240 |

5000 步在这些汇总指标上均优于 2500 步，建议作为首个真机实验候选。其平移和夹爪优于基线，但旋转平均误差仍高于保持姿态基线。3.71mm 是**与示范动作的平移增量误差**，不是实际抓取定位误差；夹爪 MAE 沿用数据单位，不是长度单位。

这是给定真实观测的开环动作比较，没有用预测动作驱动物理系统或仿真环境；没有测闭环成功率，也没有证明目标位置变化或指令变化时一定能正确选择目标。两次训练的数据集不同，不能直接用新旧验证报告的数字判断 VLM 解冻带来的净收益。

报告：[2500 步](ur3_20261016_validation_step2500.json)、[5000 步](ur3_20261016_validation_step5000.json)。

## 单卡推理性能及接口测试

测量对象是本次 5000 步模型，双 RGB 640×480、batch=1、chunk=50、BF16、10 次去噪、compile=false；正式推理模块直接运行，未加载训练用深度/视频教师进行监督计算。

| 指标 | 本机结果 |
|---|---|
| 设备 | NVIDIA GeForce RTX 4090，一张卡 |
| 模型参数占用 | 11.876GiB |
| 加载后 CUDA allocated | 11.876GiB |
| 完整验证与接口测试的 CUDA allocated 峰值 | 12.354GiB |
| CUDA reserved 峰值 | 12.395GiB |
| 专项测试首次调用 | 1.624 秒，不含模型初始化 |
| 预热 3 次后的 9 次调用中位数 / P95 | 0.789 / 0.801 秒 |
| 完整验证集推理中位数 / P95 | 0.796 / 0.818 秒 |
| 本机 WebSocket 三种任务调用 | 0.793～0.798 秒；均返回有限 50×8 动作 |
| 接口检查 | `/healthz` 返回 OK；三种任务的 reset、图像/状态传入及动作反归一化返回通过 |
| 双 RGB 单次请求体 | 1,843,555 字节，约 1.84MB；当前原始数组未压缩 |
| 动作及计时返回体 | 约 1.7KB |

CUDA 指标不含全部驱动/上下文开销；不同显卡或共享桌面环境的余量可能不同。本机接口只临时监听 127.0.0.1 随机端口，测试完成后已关闭，没有向真机发送命令，也没有保留 8006 推理服务。

报告：[推理显存、延迟与接口](ur3_20261016_inference_benchmark.json)。可复现工具为 `tools/benchmark_ur3_inference.py`。

## 两端分工与实时性

| 端 | 职责 |
|---|---|
| 真机电脑 / LeRobot | 采集两路 RGB、17 维状态和任务文本；异步发送观测；维护动作缓冲；本地按 25Hz 调用硬件驱动 |
| 服务器 / LingBot-VLA V2 | 按本次配置处理观测、四元数半球与归一化；GPU 推理；返回已反归一化的 50×8 动作 |
| 网络 | 传输观测与动作；连接质量、时延和抖动尚待真机电脑实测 |

返回动作依次是 base 坐标系平移增量 xyz、局部旋转增量四元数 xyzw、绝对夹爪目标。客户端不能再次做 mean/std 反归一化；预测四元数执行前单位化。如果硬件驱动只接受绝对目标，则按采集时相同约定转换：`p_target = p_current + delta_p`、`q_target = q_current ⊗ delta_q`，避免重复转换。

25Hz 的单周期只有 40ms，而当前单次推理约 790ms，所以控制线程不能同步等待每次推理。50 步覆盖 2 秒，只是供给动作缓冲的预测窗口，不是要求盲目执行完整 2 秒；需要异步推理、观测/动作时间戳、过期动作处理和周期性重规划。当前服务返回切片不会减少模型内部预测的完整 chunk 计算。

单次上传约 14.75Mbit：若有效上传只有 10Mbps，纯上传理论下限约 1.47 秒，再加约 0.79 秒推理就超过 2 秒窗口，尚未计入 RTT 与抖动。建议先以有效上传约 50Mbps 及以上作为试验网络目标，再实测端到端 P95；这不是实时性保证。可以后续评估无损/有损图像压缩、编译或去噪步数优化，但本次没有改变图像编码及去噪配置。

## 真机连接前仍需完成

1. 在真机电脑确认显卡型号/显存、系统内存和 LeRobot 驱动接口；当前没有该电脑的实际运行输出。
2. 使用真机电脑回放验证数据，通过 SSH 隧道连接服务器，做不调用 `send_action` 的远程 dry-run；测端到端中位/P95、上传吞吐、断线和缓冲行为。
3. 核对真实相机 RGB 顺序、640×480 视角、17 维状态顺序，以及动作位移单位、base/local 参考系、四元数乘法和夹爪开合方向。
4. 核对采集与执行侧的增益、平滑、限幅及时序。数据审计曾观察命令位移与下一帧实测位移的比例约 0.254；这需要结合驱动解释，不能直接给动作乘 0.25。
5. 检查相同/相近机械臂起始状态下，不同目标摆位和不同任务文本能否产生朝正确目标变化的预测；仅观察输出变化不足以证明定位正确。
6. 接入真机后从受限动作实验开始，再分别记录三个任务的新摆位闭环成功率、目标选择、抓取偏差和夹爪时机。离线指标不代替这些实验。

## 后续服务启动命令

以下供后续启动正式服务器使用，本次评估未运行该常驻服务。请先激活 `lingbotvla` 并设置已有离线缓存和 CUDA 环境；必须显式使用新 manifest，不能沿用旧训练启动脚本或旧 checkpoint 路径。

```bash
cd /home/teamA/lingbot-vla-v2
source /data-teamA/miniforge3/etc/profile.d/conda.sh
conda activate lingbotvla
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
export HF_HOME=/data-teamA/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export QWEN3VL_PATH=/data-teamA/models/Qwen3-VL-4B-Instruct
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python -u tools/run_ur3.py --stage serve \
  --manifest assets/training_data/ur3_20261016_split_seed42.json \
  --model_path /data-teamA/outputs/ur3_20261016/train_two_views_640_last4_seed42/checkpoints/global_step_5000/hf_ckpt \
  --use_bf16 true --use_fp32 false --use_compile false \
  --chunk_ret true --use_length 50 --port 8006
```

推理加载器会从 checkpoint 的上级训练目录读取 `lingbotvla_cli.yaml`；迁移时需要保持相应目录结构，保留本次 norm stats、机器人映射和 manifest，并配置 Qwen processor/config 路径。

真机电脑建立隧道后连接 `127.0.0.1:8006`：

```bash
ssh -p 30044 -N -L 8006:127.0.0.1:8006 teamA@183.147.142.40
```

客户端输入使用 `observation.state`（17 维 float32）、`observation.images.camera_global` 与 `observation.images.camera_wrist`（RGB uint8，480×640×3），以及完整任务文本。新数据 task_index 为 0→block 2、1→block 0、2→block 6；优先传完整文本，避免混用旧数据编号。
