# LingBot-VLA V2 三种训练策略对比

## 1. 整体理解

可以把模型分成三层：

```text
Vision Encoder
= “眼睛”：负责从图像中提取视觉特征

VLM 高层
= “大脑”：负责理解文本、视觉，并建立视觉-语言语义关系

Action Expert / Action Head
= “手”：根据语义和状态生成机器人动作
```

## 2. 方案一：只训练 Action Head / Action Expert

### 训练范围

```text
Vision Encoder        Freeze
VLM                   Freeze
Action Expert         Train
Projection / Aux      Train
```

### 原理

视觉和语言理解方式保持不变，只训练：

```text
已有视觉/语言表示
        ↓
Action Expert
        ↓
机器人动作
```

相当于：

> “眼睛和大脑不变，只训练手怎么动。”

### 适合解决

- 模型已经能正确识别目标
- 视觉-语言语义已经足够准确
- 主要问题是动作轨迹、抓取位置、控制策略不准确

### 局限

如果模型本身没有很好建立：

```text
文本中的 “2”
        ↕
图像中的数字2
```

那么 Action Expert 只能在固定的 VLM 特征上学习，难以从根本上改善目标语义绑定。

## 3. 方案二：解冻 VLM Last 4 + Action Expert

### 训练范围

```text
Vision Encoder        Freeze
VLM 前32层            Freeze
VLM 32~35层           Train
Action Expert         Train
Projection / Aux      Train
```

### 原理

Vision Encoder 仍然提供固定的基础视觉特征，但 VLM 最后四层可以根据机器人 Action Loss 调整高层视觉-语言表示。

例如：

```text
Instruction:
"Pick block 2"

        ↓

文本 token “2”
        ↕ Attention
图像中数字2对应的视觉 token

        ↓

Action Expert
        ↓
动作
```

训练后，模型可以逐渐强化：

```text
“2”文本
    ↕
数字2视觉区域
```

而降低对数字0、6等错误区域的关注。

相当于：

> “眼睛基本不变，但允许大脑重新学习应该关注哪个目标，同时训练手怎么动。”

### 适合解决

- Instruction 与视觉目标绑定不够强
- 例如 `Pick block 0 / 2 / 6`
- 希望模型真正根据指令选择对应数字积木，而不是依赖固定位置先验

### 优点

相比只训练 Action：

```text
视觉/语言高层语义可以适配当前机器人任务
```

相比解冻 Vision Encoder：

```text
显存更低
数据需求更小
过拟合风险更低
```

## 4. 方案三：解冻 Vision Encoder

### 训练范围

```text
Vision Encoder        Train
VLM                   部分或全部 Train
Action Expert         Train
```

### 原理

不仅调整高层语义关系，连最基础的视觉特征提取方式也进行训练。

例如原本：

```text
数字2和数字6
在视觉特征空间中比较接近
```

解冻 Vision Encoder 后，可以针对当前数据学习：

```text
数字2 → 更独立的视觉特征
数字6 → 更独立的视觉特征
```

即从图像最前端就增强 `0 / 2 / 6` 之间的视觉区分能力。

相当于：

> “连眼睛怎么看东西都重新训练，同时训练大脑和手。”

### 适合解决

- Vision Encoder 本身就经常分不清目标
- 数字、字体、光照、视角与预训练数据差异较大
- 即使解冻 VLM 高层，视觉目标仍然经常识别错误

### 缺点

- 显存占用明显增加
- 训练数据需求更高
- 更容易过拟合当前桌面、光照和相机视角
- 小数据集下可能破坏原有通用视觉能力

## 5. 三种方案对比

| 方案 | Vision Encoder | VLM | Action | 核心作用 |
|---|---|---|---|---|
| 只训练 Action | 冻结 | 冻结 | 训练 | 学“怎么动” |
| Last 4 + Action | 冻结 | 最后4层训练 | 训练 | 学“应该看哪个目标 + 怎么动” |
| 解冻 Vision Encoder | 训练 | 部分/全部训练 | 训练 | 连“怎么看图”都重新适配 |

## 6. 对当前数字积木任务的理解

当前任务：

```text
Instruction:
Pick block 2
```

模型需要完成：

```text
图像
 ↓
识别桌面中的0 / 2 / 6
 ↓
文本“2”与视觉中的数字2建立对应
 ↓
Action Expert选择正确目标
 ↓
生成抓取和放置动作
```

三种训练方式分别对应：

```text
只训练 Action
= 训练“手”
```

```text
解冻 Last 4
= 训练“大脑高层语义 + 手”
```

```text
解冻 Vision Encoder
= 连“眼睛”也一起训练
```

## 7. 当前推荐

对于目前：

```text
数据量：90 episodes
GPU：4 × RTX 4090
任务：0 / 2 / 6 数字积木抓取与放置
```

优先推荐：

```text
Vision Encoder        Freeze
VLM Layer 0~31        Freeze
VLM Layer 32~35       Train
Action Expert         Train
Projection / Aux      Train
```

原因：

```text
只训练 Action
→ 可能无法充分改善视觉-语言 grounding

解冻 Vision Encoder
→ 对当前数据量和显存要求过高

Last 4 + Action
→ 在语义适配、显存和过拟合风险之间更平衡
```
