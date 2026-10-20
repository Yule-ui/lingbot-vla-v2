# UR3 数据集 → LingBot-VLA 2.0 训练交接

请继续完成数据准备、低显存训练适配、4 卡试跑和首轮微调，并给出可复制的运行指令。以下路径来自采集机器，服务器上请先定位实际路径。

## 当前情况

- 硬件：UR3 + Robotiq，手柄遥操；训练服务器有 4 张 NVIDIA RTX 4090，每卡通常为 24 GB，请实测确认。
- LeRobot 项目：`/home/robot/lerobot`。
- 数据集：`/home/robot/lerobot/dataset/ur3_eef_relative`，LeRobot v3.0，25 FPS。
- 已核对：90 个 episode、43,141 帧，约 29 分钟；下列任务各 30 条：
  - `Pick up block 0 and place it in the square area.`
  - `Pick up block 2 and place it in the square area.`
  - `Pick up block 6 and place it in the square area.`
- 当前 `meta/info.json` 的 train split 为 `0:90`，尚未划分验证集。
- LingBot 官方仓库：`https://github.com/Robbyant/lingbot-vla-v2`；采集机器已有 `/home/robot/lingbot-vla-v2`。
- 尚未确认服务器训练环境、预训练权重和教师权重路径；当前会话无法通过 `nvidia-smi` 获取 GPU 状态，未实测训练显存或速度。
- 用户偏好：尽量通过独立配置、脚本或包装入口适配，减少对官方主程序的修改。

## 数据语义和映射

`observation.state` 为 17 维，依次为：6 个关节角（rad）、TCP xyz（m）、TCP rotvec（rad）、TCP quaternion（xyzw）、夹爪实际位置。

`action` 为 8 维：`[delta_x, delta_y, delta_z, delta_qx, delta_qy, delta_qz, delta_qw, gripper]`。平移增量使用机器人 base 坐标系；旋转为局部相对四元数（xyzw）；夹爪为绝对目标值。动作已经是每个控制周期的相对增量，不能再减一次 state。

已有映射模板：`examples/ur3_robotiq/lingbot_vla_v2_eef_relative.yaml`，复制到 LingBot 的 `configs/robot_configs/ur3_robotiq_eef_relative.yaml` 后使用 `data_name: ur3_robotiq_eef_relative`。

- State：关节 `[0:6]` → `arm.position`；xyz `[6:9]` + quaternion `[12:16]` → `end.position`；夹爪 `[16:17]` → `effector.position`。
- Action：位姿 `[0:7]` → `end.position`；夹爪 `[7:8]` → `effector.position`；两项均为 `subtract_state: false`。
- 相机：`camera_global` → `camera_top`；`camera_wrist` → `camera_wrist_left`。
- 保留预训练模型的统一特征顺序、55 维内部 state/action 布局和缺失维度 mask，不要直接把模型输出层改为 8 维。核对 checkpoint 和训练配置的实际布局。
- 夹爪目标在现有数据中约为 0.502～1.0；核对采集代码的开合约定，并在部署中保持一致。

## 建议首轮配置（需要显存试跑验证）

从官方 `configs/vla/real_robot/real_robot.yaml` 派生专用配置，保留 checkpoint 对应的 MoE 结构。

```yaml
train:
  train_expert_only: true
  train_state_proj: true
  data_parallel_mode: fsdp2
  enable_full_shard: true
  vlm_fsdp: true
  enable_gradient_checkpointing: true
  enable_mixed_precision: true
  enable_fp32: false
  micro_batch_size: 1
  gradient_accumulation_steps: 4
  global_batch_size: 16
  optimizer: adamw
  lr: 2.0e-5
  lr_warmup_ratio: 0.03
  chunk_size: 50
  max_steps: 5000
  save_steps: 1000
  use_compile: false
```

此片段只列覆盖项，不是完整训练配置。`train_expert_only` 在本地 V2 实现中会冻结整个 Qwen-VL。官方完整配置的 4 卡示例约需每卡 49 GB；上述冻结、分片方案尚未验证能装入 24 GB。梯度累积扩大有效 batch，不能解决单次前向的显存不足。深度/视频教师的显存也要计入；不要直接删除 `align_params` 或把损失权重置零当作卸载教师，需核对加载条件和 checkpoint 结构。

## 已发现的权重加载问题

本地 `tasks/vla/train_lingbotvla.py` 调用 `build_parallelize_model` 时没有传入 `weights_path`；`lingbotvla/distributed/torch_parallelize.py` 的 FSDP2 `init_device=meta` 分支在缺少该参数时会 `to_empty` 并 `init_weights()`。因此不能仅设置 `init_device: meta` 就启动微调。

请根据服务器的实际版本核对并修复分片后的权重加载，保留 post-training 的键名映射、buffer、缺失权重检查和冻结状态；优先用独立包装入口。验证各 rank 确实加载同一份预训练权重，抽查参数与 checkpoint 一致。直接在 CUDA 上加载完整 FP32 模型也可能在分片前耗尽显存。

## 请按顺序继续执行

1. 检查 GPU、驱动、CUDA、RAM、磁盘、训练仓库版本及环境；使用官方 `tools/create_train_env.sh` 准备独立训练环境，核对 PyTorch/FlashAttention 要求及模型权重路径。
2. 只读验证数据完整性、两路视频、任务标签、动作/state、四元数范数及异常轨迹。保留源数据，按任务分层、固定 seed，以每任务 27/3 条划分训练/验证（81/9），记录 episode 清单。确保加载器实际采用划分；只改 `info.json` 的 splits 不足以保证过滤生效。
3. 准备机器人映射，用训练集运行官方 `scripts/compute_norm_stats.py`。优先采用 real-robot 的 `meanstd`；统计量文件由训练和部署共用，核对近常量维度的数值处理和 padding mask。
4. 修复低显存加载路径，生成专用配置及启动脚本；使用 `CUDA_VISIBLE_DEVICES=0,1,2,3` 启动 4 个进程。先跑 50 步，检查所有 rank 的权重、冻结参数、有限 loss、峰值显存和秒/步；试跑使用独立输出目录。
5. 试跑通过后从原始预训练权重开始首轮 5,000 步；比较每 1,000 步 checkpoint 的验证动作误差。验证数据量很小，后续还需每任务实机成功率评估；实机控制需另行安排。
6. 根据验证结果决定是否延长至 10,000～20,000 步；以实测速率估算时间。交付环境准备、统计量计算、训练、断点续训和离线评估命令，以及日志、最终配置、checkpoint 路径和结果说明。

部署衔接：输出反归一化后仍是每步相对 EEF 动作；按顺序执行增量，保持 base 平移和局部旋转约定，并归一化预测四元数。50 帧 chunk 在 25 FPS 下为 2 秒，闭环部署应重新规划，不应一次无反馈执行整段。

官方参考：[训练配置](https://github.com/Robbyant/lingbot-vla-v2/blob/main/configs/vla/Training_Config.md)、[自定义数据](https://github.com/Robbyant/lingbot-vla-v2/blob/main/lingbotvla/data/vla_data/README.md)。
