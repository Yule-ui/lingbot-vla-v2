# UR3 训练入口

环境：`/data-teamA/miniforge3/envs/lingbotvla`。
源数据：`/data-teamA/datasets/ur3_eef_relative`，不修改源数据及 info.json。

## 本次准备

- 固定 seed=42，三个任务各 27 条训练、3 条验证。训练 81 episode/38,822 帧；验证 9 episode/4,319 帧。
- 清单：`assets/training_data/ur3_split_seed42.json`。
- 验证 episode：10、14、19、44、45、53、73、74、84。
- 统计：`assets/norm_stats/ur3_robotiq_eef_relative.json`；仅使用训练帧，每帧一次。数值统计使用 float64，避免四元数近常量分量的方差因 float32 抵消而成为零。
- 状态四元数按清单中的固定参考四元数选择半球。统计、训练、验证和服务共用规则；动作仍为每周期 base 平移增量、局部旋转增量及绝对夹爪目标。
- `tools/run_ur3.py` 包装官方入口：显式选择 episode，修正过滤后行号与原始全局索引的转换；不修改官方模型、训练主程序或数据集。
- FSDP2 分片后严格加载完整 LingBot 预训练权重，并检查缺失/额外参数、冻结状态、meta tensor 和各 rank 的样本参数数值。报告保存在输出目录 `weight_load_rank*.json`。
- 243 组动作 chunk 边界检查通过；统计与原始训练帧直接计算结果一致；55 维 state/action mask 及 8 维动作反归一化还原检查通过。
- 验证集实际加载 9 episode/4,319 帧，另有 27 组轨迹边界检查及同样的 55 维映射还原检查通过。
- 四卡预训练权重加载检查通过，加载峰值约每卡 9.95 GiB；该数值不代表完整训练峰值。
- 实际完成 50 个优化器步骤：全部 loss/GradNorm 有限，loss 从 1.5865 到最后一步 0.2069；预热后中位数 17.208 秒/步。该下降来自训练 batch，不代表验证误差或真机成功率。
- 四 rank 均正常完成训练与保存；PyTorch 峰值 allocated=19.01 GiB、reserved=20.71 GiB。第 50 步分布式 checkpoint 大小约 38.22 GiB，位于 `/data-teamA/outputs/ur3/smoke_seed42/checkpoints/global_step_50`。
- 试跑过程中更新 launcher，导致已运行的 Bash 在训练成功后退出时出现读取错误；随后用新 launcher 完整运行 `check`，退出码为 0。训练、四 rank 完成记录和 checkpoint 均已成功保存。
- 当前配置下首轮 5,000 步按试跑速度约需 24 小时，实际受负载及 checkpoint 保存影响。基准结果见 `docs/ur3_smoke_result.json`。

## 命令

```bash
source /data-teamA/miniforge3/etc/profile.d/conda.sh
conda activate lingbotvla
cd /home/teamA/lingbot-vla-v2
```

重新准备配置、计算统计、检查数据：

```bash
python tools/prepare_ur3_training.py
bash tools/start_ur3.sh norm
bash tools/start_ur3.sh check
```

四卡模型加载检查和 50 步试跑：

```bash
bash tools/start_ur3.sh preflight
bash tools/start_ur3.sh smoke
```

试跑输出：`/data-teamA/outputs/ur3/smoke_seed42`。试跑不导出 HF 权重，结束时保存可恢复的分布式 checkpoint。

正式 5,000 步：

```bash
bash tools/start_ur3.sh train
```

正式训练从原始预训练权重开始，输出到 `/data-teamA/outputs/ur3/train_seed42`，与试跑隔离。当前数据盘容量下每 2,500 步保存完整 checkpoint 和 HF 导出，避免每 1,000 步保留五套大型模型/优化器/HF 文件占满磁盘。若修改保存策略，应先按实际 checkpoint 大小检查空间。

训练中断后恢复：

```bash
bash tools/start_ur3.sh train --train.enable_resume true
```

再次从头运行已有训练目录时应显式选择新的 `--train.output_dir`；不要将试跑的 checkpoint 用作首轮微调起点。

日志位于 `/data-teamA/outputs/ur3/logs/`。launcher 将临时文件、Hugging Face、Triton、Inductor 和 CUDA 缓存放在数据盘。

## 验证和部署

使用 exported `hf_ckpt` 目录，不要直接把分布式 checkpoint 目录传给推理服务。包装入口默认验证上述 9 个 held-out episode，并将原始全局索引转换为过滤后的行号。

```bash
export QWEN3VL_PATH=/data-teamA/models/Qwen3-VL-4B-Instruct
python tools/run_ur3.py --stage eval \
  --model_path /data-teamA/outputs/ur3/train_seed42/checkpoints/global_step_5000/hf_ckpt \
  --robo_name ur3_robotiq_eef_relative \
  --data_path /data-teamA/datasets/ur3_eef_relative \
  --use_bf16 --chunk_ret true --use_length 50 \
  --save_plot_path /data-teamA/outputs/ur3/validation_step5000
```

要比较第 2,500 步权重，修改 model_path 和图表目录。离线误差不能替代真实机器人成功率。

服务也要通过包装入口，保证训练与部署的状态四元数预处理一致：

```bash
python tools/run_ur3.py --stage serve \
  --model_path /data-teamA/outputs/ur3/train_seed42/checkpoints/global_step_5000/hf_ckpt \
  --use_bf16 true --use_compile false --chunk_ret true --use_length 50 --port 8006
```

LeRobot 客户端需在每次任务开始时发送 reset/robo_name；接收的是已经反归一化的相对 EEF 动作，按采集控制逻辑执行增量，归一化预测四元数并直接使用夹爪绝对目标。闭环 25 Hz、执行部分 chunk 后重规划的硬件适配仍需根据 UR3 驱动实施。

5000 步训练已完成，两套模型已完成真实离线推理和 WebSocket 检查。未编译推理实测约 0.77 秒，因此上述服务返回完整 50 步；客户端需要异步推理、延迟处理和动作缓冲。详见 [训练结果与推理](ur3_inference.md)。
