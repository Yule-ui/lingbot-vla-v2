# UR3 首轮训练检查记录

检查日期：2026-10-09。仓库版本：`e28c4f19d4327765382e3e8be6217c1c08863ed7`。

## 检查结果

| 项目 | 结果 |
| --- | --- |
| GPU | 沙箱外可见 4 张 RTX 4090；每卡 24,564 MiB，每卡空闲约 24,080 MiB；驱动 580.126.09。此前的 GPU 不可见由沙箱隔离造成。 |
| 训练环境 | 当前系统 Python 3.10.12；未找到可用 Conda 或独立训练环境；未安装 torch、lerobot、transformers、pyarrow、av。CUDA toolkit 目录为 12.6，具体训练 runtime 随安装的 PyTorch 决定。 |
| 磁盘与 RAM | `/data-teamA` 空闲约 343 GB；工作区所在系统盘约 4 GB；RAM 可用约 481 GiB。环境、包缓存、模型和 checkpoint 应使用数据盘。 |
| 数据 | `/data-teamA/datasets/ur3_eef_relative`，LeRobot v3.0，25 FPS，90 episode、43,141 帧，三个任务各 30 条。 |
| 数据数值 | 全量读取全部 Parquet：state 为 17 维，action 为 8 维，无 NaN/Inf；index 和每个 episode 内 frame_index 连续；时间间隔约 0.04 秒。 |
| 视频 | 20 个 AV1 文件，640×480、25 FPS；全部首帧和尾部抽检可解码；90 个 episode 的数据索引及两路视频时间范围无越界。没有逐帧解码全量视频，也没有验证相机的实际采集同步误差。 |
| 主模型 | `/data-teamA/models/lingbot-vla-v2-6b`；六个 safetensors 分片长度与文件头一致，1,708 个张量与索引匹配。所有张量为 FP32，总张量大小 25,503,630,044 字节。检查了头和长度，未验证下载源哈希或逐值扫描权重。 |
| 模型布局 | state_proj 为 [768,55]，action_out_proj 为 [55,768]，MoE 样本包含 32 个专家、expert hidden size=768、intermediate size=512。应保持 checkpoint 对应结构和 55 维输入/输出。 |
| 教师资产 | 存在 `depth/model.pt`、`dino_video/teacher_step_10000.pth` 和 `dino_video/config.yaml`；尚未实际加载教师文件。 |
| 缺少资产 | 已检查模型目录及 Hugging Face 缓存，未发现 Qwen3-VL-4B-Instruct 完整配置目录或 MoGe-2 权重。LingBot 根目录 config.json 仅有 vlm_family，不能替代 Qwen3-VL 模型配置。 |

## 数据语义与需要处理的事项

- 现有 YAML 对 17 维 state、8 维 action 和两路相机的映射与数据特征一致。action 已是每周期 EEF 增量，`subtract_state: false` 必须保留，夹爪是绝对目标。
- state quaternion 范数范围为 0.999999967～1.000000033；action quaternion 范数范围为 0.999999970～1.000000030。
- state quaternion 有 7 处相邻符号翻转，位于 episode 34、57、67、72、77、86、87；对应物理旋转变化仅约 0.0051～0.0084 rad。当前 q_w 已全部非负，简单使用 q_w≥0 的规则不能消除这些跨 180° 的不连续。建议用固定参考四元数选择同一半球，检查所有样本与参考的点积远离零，并让统计、训练、部署共用该规则；保留源数据。动作四元数无相邻符号翻转。
- 逐周期平移增量模长：中位数 6.12 mm，99 分位 9.27 mm，最大 10.20 mm。逐周期旋转增量：99 分位 0.0300 rad，最大 0.0351 rad。这些数值只是数据分布，不代表硬件安全限值。
- gripper action 范围 0.5019608～1.0，state 范围 0.5019608～0.9058824；开合约定仍需从采集机器 UR3/Robotiq 驱动核实。
- 当前只有 train=0:90。需按任务固定 seed 拆分 81/9，并显式把 episode 清单传入统计、训练和评估加载器。只修改 info.json 的 splits 不会让现有训练包装器自动筛选。
- 官方环境脚本固定 LeRobot v0.4.2；该版本通过 `HF_LEROBOT_HOME / repo_id` 解析 root，绝对 repo_id 在 Python Path 中会覆盖前缀。因此本地绝对路径不是已确认的必修故障，显式 root 属于兼容性改进。参考：https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/datasets/lerobot_dataset.py 。实际训练环境安装后仍需验证加载。

## 必要程序适配

1. **数据选择**：增加独立包装入口或最小加载器参数，让 episode 清单进入 LeRobotDataset；重用其绝对索引到过滤后索引的映射，验证 chunk 不跨 episode。训练集统计必须与训练集筛选一致。
2. **FSDP2 预训练权重加载**：`tasks/vla/train_lingbotvla.py` 的 build_parallelize_model 调用未传 weights_path；`lingbotvla/distributed/torch_parallelize.py` 的 meta 分支会 to_empty 后 init_weights。不能直接开启 init_device=meta 启动微调。独立包装入口应在分片后加载正确权重，并保留 post_training 严格键名/缺失检查、buffer 和冻结状态。不能仅传入路径后就假定加载正确。
3. **专用配置与启动脚本**：保持官方预训练 MoE/55 维结构；冻结 Qwen-VL，训练 action expert 与 state projection；FSDP2 全分片、BF16、梯度检查点、micro batch=1、累积=4、有效 batch=16、AdamW、lr=2e-5、chunk=50、关闭 compile，使用独立试跑输出目录。教师模型继续计入显存。

## 下一步执行顺序

1. 在数据盘创建 Python 3.12 训练环境，安装并核对官方依赖。官方 create_train_env.sh 依赖 Conda，目前没有可用 Conda；需先安装到数据盘，或编写等价 uv 环境安装入口，不能直接运行该脚本期待成功。
2. 补齐 Qwen3-VL 配置/处理器和 MoGe 权重。完整 LingBot checkpoint 已包含 Qwen-VL 参数；若沿用完整 post-training 加载路径，应避免把 vlm_repo_id 指向单独 Qwen 权重以覆盖它。是否需要下载独立 Qwen 权重应由实际加载路径决定。
3. 生成 81/9 清单、验证四元数符号预处理、安装机器人映射，然后用训练集计算 meanstd 统计。
4. 完成分片加载适配，逐 rank 抽查参数与 checkpoint 相符、冻结状态及所有 buffer 已物化；运行 50 步试跑，记录峰值显存、有限 loss 和秒/步。4×24 GB 能否运行尚未实测。
5. 试跑通过后从原始预训练权重开始 5,000 步，验证每 1,000 步 checkpoint；后续再接 LeRobot 真机控制适配层。

本轮仅检查资产并保存报告；未修改数据集、模型或训练主程序，未启动训练。临时检查依赖安装在 `/tmp/lingbot-inspect`，不是正式训练环境。详细数据检查结果见同目录 `ur3_data_audit.json`。
