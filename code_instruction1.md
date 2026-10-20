# LeRobot 端对接 LingBot-VLA V2 的代码修改说明

请在当前 LeRobot 仓库中完成 **UR3＋Robotiq 对接远程 LingBot-VLA V2 推理服务**。

先检查现有采集驱动和 LeRobot 版本，复用机器人、夹爪及相机接口。优先新增独立推理入口；保持采集程序和数据格式兼容，不修改 LingBot 模型权重或训练数据。

## 1. 已知部署情况

- LingBot 模型已经完成 5000 步训练，第 2500、5000 步权重均可正常加载。
- 已完成 9 条验证轨迹的离线推理，以及三个任务的 WebSocket reset 和动作返回测试。
- 首次部署使用第 5000 步模型：

  ```text
  /data-teamA/outputs/ur3/train_seed42/checkpoints/global_step_5000/hf_ckpt
  ```

- 服务端通过 LingBot 的 `tools/run_ur3.py --stage serve` 启动，保持训练与推理的预处理一致。
- 服务设置：BF16、`chunk_ret=true`、`use_length=50`、端口 `8006`。
- 未编译单次推理调用实测约 **0.77 秒**，不是每 40 ms 完成一次推理。
- 机器人端只负责通信和硬件控制，不加载模型权重。现有训练服务器使用 LeRobot 0.4.2；实施时以采集端实际版本和自定义驱动为准，不为对接而盲目升级环境。

## 2. 需要新增或修改的程序

以下文件名可根据本地仓库结构调整：

| 程序 | 职责 |
|---|---|
| `scripts/ur3_lingbot_infer.py` | 配置、连接机器人、启动推理线程与控制循环、退出清理 |
| `lingbot_remote/client.py` | WebSocket 连接、reset、发送观测、接收动作、超时与断线处理 |
| `lingbot_remote/adapter.py` | 将机器人观测拼接为训练格式；将返回动作转换为驱动字段；处理四元数 |
| `lingbot_remote/action_buffer.py` | 线程安全动作队列、请求时间线、过期动作处理、任务切换 |
| 现有 UR3／Robotiq 驱动 | 核对 `get_observation()`、`observation_features`、`action_features`、`send_action()`；必要时补充 EEF 相对控制 |

先通过独立入口调用 LeRobot 的 Robot 接口。如果后续需要接入标准 `--policy` 参数，再增加 RemotePolicy 注册及对应 processor。不要直接将 LingBot 的 `hf_ckpt` 当作 LeRobot 内置策略权重加载。

## 3. 通信协议

使用 **WebSocket 持久连接＋MsgPack NumPy 数组序列化**。

复用 LingBot 仓库中的两个通信模块，并保留匹配的数组编码方式：

```text
deploy/websocket_client_policy.py
deploy/msgpack_numpy.py
```

这两个文件在训练服务器上的路径为：

```text
/home/teamA/lingbot-vla-v2/deploy/websocket_client_policy.py
/home/teamA/lingbot-vla-v2/deploy/msgpack_numpy.py
```

如果采集端没有这些文件，先获取匹配版本并放入客户端包，调整相对导入。通信依赖包括 `numpy`、`msgpack`、`websockets`、`typing_extensions`；检查本地 WebSocket API 兼容性。上传值使用 NumPy 数组，不直接序列化 PyTorch Tensor。

服务连接建立后会先发送 metadata，现有 `WebsocketClientPolicy` 已处理该握手。

### 3.1 每个任务开始时 reset

```python
client.reset("ur3_robotiq_eef_relative")
```

实际 reset 消息为：

```python
{"reset": True, "robo_name": "ur3_robotiq_eef_relative"}
```

reset 初始化服务端机器人映射并清理策略状态，必须在该任务的首次推理之前完成。

### 3.2 上传观测

```python
request = {
    "observation.state": state17,                   # float32，形状 (17,)
    "observation.images.camera_global": rgb_global, # RGB uint8，H×W×3
    "observation.images.camera_wrist": rgb_wrist,
    "task": task_text,
}
response = client.infer(request)
```

`state17` 的拼接顺序：

| 索引 | 含义 | 单位／约定 |
|---|---|---|
| `[0:6]` | 六个关节角 | rad |
| `[6:9]` | TCP xyz | m，base 坐标系 |
| `[9:12]` | TCP rotvec | rad |
| `[12:16]` | TCP quaternion | xyzw，单位四元数 |
| `[16:17]` | 夹爪实际位置 | 与采集时相同的数值单位和方向 |

`get_observation()` 的原始字段不一定已经是上述数组，需要照采集代码拼接。图像保持与训练一致的相机视角、RGB 通道顺序和 uint8 范围，不要重复归一化。

任务描述使用训练中的原文：

```text
Pick up block 0 and place it in the square area.
Pick up block 2 and place it in the square area.
Pick up block 6 and place it in the square area.
```

### 3.3 接收动作

```python
actions = response["action"]  # float32，形状 (50, 8)
```

响应还可能包含 `server_timing` 等元信息，不要将这些字段当作机器人动作。客户端应检查动作形状、有限值和四元数有效性。

## 4. 动作语义与执行适配

每行动作的顺序为：

```text
[Δx, Δy, Δz, Δqx, Δqy, Δqz, Δqw, gripper]
```

- 平移增量使用 **base 坐标系**，单位 m。
- 旋转是 **局部相对四元数**，顺序 xyzw；执行前单位化。
- 夹爪是 **绝对目标**，复用采集时的单位和开合方向，不能假设 0 或 1 的开合含义。
- 服务已经完成反归一化，客户端不要再次归一化、反归一化或相对化。
- 训练时状态四元数的半球符号统一由服务处理；客户端不需要把符号反变换。
- 服务内部使用 55 维统一布局，线上客户端只上传原始 17 维状态、接收 8 维动作，不自行补齐 55 维。

如果驱动已经支持采集时的 `eef_relative` 执行模式，直接复用其逻辑。

如果驱动只接收绝对 TCP 位姿，则逐控制周期使用当时的当前位姿计算：

```text
Δq = Δq / ||Δq||
p_target = p_current + Δp
q_target = q_current ⊗ Δq
```

局部旋转增量右乘当前四元数。按驱动要求转换目标姿态格式，并将动作数组转换为符合 `action_features` 的字典。若驱动内部已经做了上述转换，适配层不要重复叠加。

## 5. 异步控制与动作缓冲

目标控制频率为 **25 Hz**，周期 **40 ms**。当前一次推理约 0.77 秒，因此不能把同步 `client.infer()` 放进每个控制周期。

### 控制线程

- 统一访问机器人驱动，避免多个线程并发调用不保证线程安全的硬件接口。
- 采集观测并维护线程安全的最新观测快照，记录采集时间及控制步数。
- 每 40 ms 从动作队列取一个有效动作，完成适配后调用 `robot.send_action()`。
- 保留驱动已有的增量、速度和工作空间限制。

### 推理线程

- 读取最新完整观测快照，独占 WebSocket 连接；同一连接只保留一个在途请求。
- 串行处理 reset 与 infer，避免多个线程同时 send／recv。
- 为请求记录任务代数、请求编号、观测时间、发送时间及接收时间。
- 校验返回动作，按明确的时间对齐规则更新动作队列。

### 时间线与异常处理

- 首次等待推理时保持机器人位姿，从首个有效动作建立执行时间线。
- 后续请求考虑等待期间实际已经执行的控制步，明确新旧 chunk 的对齐与替换规则。
- 不要固定丢弃 19 步作为所有请求的通用延迟补偿；也不要仅根据 `server_timing` 推断端到端延迟。
- 当前协议不回显客户端请求编号；保持单在途请求，在客户端维护请求与响应的对应关系即可。若扩展协议，应同时修改服务端并保留兼容性。
- 任务 reset 清空队列，拒绝上一任务的迟到响应。
- 队列为空、断网、超时或动作无效时停止移动／保持当前 TCP 位姿；夹爪保持已有目标。
- **不能重复最后一个相对增量来填补空队列**，也不能继续执行上一任务的残留动作。
- 返回 50 步是给客户端提供未来动作窗口，不要求盲执行全部 50 步；动作缓冲本身不能保证闭环实时性。

## 6. 配置与运行

入口至少支持：

```text
server_host
server_port
task
fps
dry_run
```

复用现有机器人连接配置，并提供可配置的通信超时和动作限制。默认 `dry_run=true`；在该模式下读取真实观测、请求推理并记录拟执行动作，但不发送会移动机器人或夹爪的指令。

服务端端口为 `8006`。机器人电脑可以建立 SSH 隧道：

```bash
ssh -p 30044 -N \
  -L 8006:127.0.0.1:8006 \
  teamA@183.147.142.40
```

隧道需要连接到实际运行推理服务的同一容器／主机。建立后客户端使用：

```text
server_host=127.0.0.1
server_port=8006
```

## 7. 验收与交付

先完成 dry-run，再开放真实执行：

1. 核对 17 维状态、两路相机和夹爪单位与采集代码一致。
2. 三个任务均能 reset、上传观测并返回形状为 50×8 的有限动作。
3. 验证新旧动作时间线、任务切换、队列空、超时及断线处理。
4. 记录实际控制周期、请求耗时、观测年龄、队列长度和丢弃动作数；dry-run 不实际移动硬件。
5. 明确区分接口测试、dry-run 和已完成的真机闭环测试，不将离线误差或队列机制当作真实成功率证明。

完成后提供：

- 修改／新增文件清单及各文件职责。
- 完整的环境准备、dry-run 和真实执行启动指令。
- 实际验证日志与结果。
- 所采用的延迟对齐和队列替换策略。
- 尚未验证的真机行为或需要用户补充的驱动信息。

参考：[LeRobot 硬件接口说明](https://huggingface.co/docs/lerobot/integrate_hardware)。
