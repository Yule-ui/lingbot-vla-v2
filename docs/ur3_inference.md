# UR3 训练结果与推理

## 实际检查结果

2026-10-10 检查了 `/data-teamA/outputs/ur3/train_seed42`：

- 5000 个优化器步骤连续、完整；记录的训练 loss 和梯度范数均有限，无忽略 batch。
- 从 launcher 启动到最后一次 HF 导出完成耗时约 24 小时 54 分钟。
- 前 100 步平均训练 loss 0.670898，最后 100 步平均 0.123471。训练曲线见 [ur3_training_loss.png](ur3_training_loss.png)。
- 四 rank 完成记录完整；每卡峰值 allocated 约 19.02 GiB，reserved 约 20.85 GiB。
- 第 2500、5000 步都有 DCP 模型、优化器、extra_state 和 HF 导出。每套约 50.11 GiB，其中 HF 目录约 11.89 GiB。
- 两套 HF 权重均为 3 个 safetensors 分片、1708 个 BF16 tensor；索引、张量偏移和文件长度一致，实际推理严格加载成功。
- 日志中 `eval_failed: 3` 是 `AsyncHFCheckpointSaver.summary()` 把未执行评估的默认标志计为失败；该类当前没有执行 evaluator。`hf_failed: 0`，权重导出成功。最后一次同步 HF 导出执行了两次，最终文件有效。

使用相同验证划分、相同逐查询 diffusion seed、BF16、10 次 denoise、不编译，完整评估了 9 条轨迹的 4319 帧，共 92 次推理。去掉每条轨迹末尾重复 padding；旋转误差先将预测和真实四元数单位化，再计算与符号无关的 geodesic 角度。

| 模型 | 平移向量 RMSE | 旋转平均误差 | 旋转 P95 | 夹爪 MAE |
|---|---:|---:|---:|---:|
| 2500 步 | 3.866 mm | 0.1873° | 1.2477° | 0.03013 |
| 5000 步 | 3.783 mm | 0.1825° | 1.2479° | 0.02960 |
| 不移动、保持当前夹爪基线 | 6.233 mm | 0.1637° | 1.3896° | 0.14018 |

夹爪误差沿用数据中的数值单位，不是米。5000 步的主要平均指标略优于 2500 步，可以先用于部署验证；旋转平均误差仍未优于 identity rotation 基线。以上是离线动作误差，不是末端位置到目标物体的距离，也不代表闭环成功率。

报告：[训练](ur3_training_result.json)、[2500 步验证](ur3_validation_step2500.json)、[5000 步验证](ur3_validation_step5000.json)、[WebSocket 测试](ur3_websocket_smoke.json)。

## 启动服务

在服务器创建 tmux 会话：

```bash
tmux new-session -A -s ur3_infer
```

会话内执行：

```bash
source /data-teamA/miniforge3/etc/profile.d/conda.sh
conda activate lingbotvla
cd /home/teamA/lingbot-vla-v2
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export HF_HOME=/data-teamA/cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN3VL_PATH=/data-teamA/models/Qwen3-VL-4B-Instruct
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python tools/run_ur3.py --stage serve \
  --model_path /data-teamA/outputs/ur3/train_seed42/checkpoints/global_step_5000/hf_ckpt \
  --use_bf16 true --use_compile false \
  --chunk_ret true --use_length 50 --port 8006
```

按 Ctrl+b，松开，再按 d 可以保留会话；用 `tmux attach -t ur3_infer` 返回。

保持 checkpoint 与训练目录结构：服务需要读取 `train_seed42/lingbotvla_cli.yaml`。同时保留 UR3 robot_config、归一化 JSON、split manifest 和 Qwen 配置/processor；迁移到其他机器时修改对应绝对路径。通过 `tools/run_ur3.py` 启动可保证状态四元数使用与训练一致的半球处理。

## 本机接口测试

服务启动后，在另一个同环境终端运行以下命令。它只使用验证数据测试接口，不向机器人发送指令：

```bash
python tools/evaluate_ur3_checkpoint.py \
  --server_host 127.0.0.1 --port 8006 --use_length 50 \
  --traj_ids 10 44 73 --max_chunks_per_episode 2 \
  --report /data-teamA/outputs/ur3/websocket_check/metrics.json
```

检查时已实际使用临时本机服务完成三个任务的 reset、6 次请求、25×8 动作返回，所有输出有限；临时服务随后已停止。这里正式服务设置为 50×8，便于客户端保留足够的未来动作处理延迟。

## 连接机器人电脑

机器人电脑可以通过 SSH 隧道连接训练服务器上的服务：

```bash
ssh -p 30044 -N -L 8006:127.0.0.1:8006 teamA@183.147.142.40
```

随后客户端连接 `127.0.0.1:8006`。需要机器人电脑能访问该 SSH 地址，而且它连接的是运行推理服务的同一容器/主机。

接口使用 `deploy.websocket_client_policy.WebsocketClientPolicy`：每个新任务先 `policy.reset("ur3_robotiq_eef_relative")`，再发送：

- `observation.state`：17 维 float32，顺序与采集一致：6 joints、TCP xyz、TCP rotvec、TCP quaternion xyzw、实际 gripper。
- `observation.images.camera_global` 和 `observation.images.camera_wrist`：RGB uint8，H×W×3，视角与训练数据一致。
- `task`：训练时的任务描述。

`policy.infer(obs)["action"]` 返回 50×8 数组，已反归一化：base 平移增量 xyz、局部旋转增量四元数 xyzw、绝对夹爪目标。不要再次做 mean/std 反归一化或相对动作转换。执行端仍需将预测旋转四元数单位化；原来的 q/-q 半球选择不需要反变换。

若采集驱动支持 EEF relative，复用其执行逻辑。若驱动只收绝对 TCP，则逐周期使用 `p_target = p_current + delta_p`、`q_target = q_current ⊗ delta_q`，由驱动转换成需要的姿态格式；不要重复叠加。夹爪沿用采集时的开合方向和数值单位。

## 实时执行限制

本次未编译推理的单次调用中位数约 0.77 秒，P95 约 0.83 秒，GPU 峰值 reserved 约 12.18 GiB。WebSocket 本机请求中位数约 0.78 秒。该结果不能套用官方约 130 ms 的其他推理配置。

25 Hz 下 5 步只覆盖 0.2 秒，不能持续供给当前约 0.77 秒的同步推理。正式服务先返回完整 50 步（2 秒）动作；真机客户端必须将 25 Hz 控制线程与推理线程分离，记录观测时间，按延迟丢弃过期动作，管理缓冲不足和重新规划。50 步返回长度不是要求盲执行全部 50 步，也不能仅靠增大 chunk 保证闭环实时性。

目前训练环境中未找到采集端 UR3/Robotiq 驱动，尚未验证真实 `send_action` 的字段、单位和控制方式。下一步需要该驱动路径/代码来完成 LeRobot 客户端，先检查实际观测与动作输出，再进行有限幅度的真机测试。

## 重跑完整离线评估

```bash
python tools/evaluate_ur3_checkpoint.py \
  --model_path /data-teamA/outputs/ur3/train_seed42/checkpoints/global_step_5000/hf_ckpt \
  --report /data-teamA/outputs/ur3/validation_step5000/metrics.json
```

评估只使用验证 episode，不控制机器人。第 2500 步将两处 `5000` 改为 `2500`。使用同样的 dtype、denoise 步数与 query seed 比较结果。
