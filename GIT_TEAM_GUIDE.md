# lingbot-vla-v2 最小 Git 协作指南

仓库：

```bash
https://github.com/Yule-ui/lingbot-vla-v2.git
```

## 核心规则

- `main` 是稳定主分支，禁止直接开发、直接 push。
- 每个人从最新 `main` 创建自己的功能分支。
- 开发完成后通过 Pull Request 合并到 `main`。
- PR 至少需要 1 人 Approve。
- PR 中的 review 对话必须处理完成。
- 合并前分支必须基于最新 `main`。
- 推荐使用 `Squash and merge`。
- 禁止对 `main` 强制推送。

---

## 1. 第一次 Clone

```bash
git clone https://github.com/Yule-ui/lingbot-vla-v2.git
cd lingbot-vla-v2
```

同步最新 `main`：

```bash
git switch main
git pull origin main
```

---

## 2. 创建自己的功能分支

统一命名：

```text
feature/<功能名>
fix/<问题名>
```

例如：

```bash
git switch -c feature/data-converter
```

第一次推送：

```bash
git push -u origin feature/data-converter
```

之后只在自己的分支开发。

---

## 3. 日常开发

修改代码后：

```bash
git status
git diff
git add .
git commit -m "feat: add data converter"
git push
```

常用提交格式：

```text
feat: 新功能
fix: 修复问题
refactor: 重构
docs: 文档
test: 测试
chore: 配置或杂项
```

不要使用：

```text
update
修改
test123
final
```

---

## 4. 提 PR 前同步最新 main

功能完成后：

```bash
git fetch origin
git rebase origin/main
```

如果没有冲突：

```bash
git push --force-with-lease
```

然后确认代码可以正常运行。

> `--force-with-lease` 只允许用于自己的功能分支，禁止用于 `main`。

---

## 5. 创建 Pull Request

GitHub：

```text
Pull requests
→ New pull request
```

选择：

```text
base: main
compare: feature/你的功能分支
```

PR 标题建议：

```text
feat: add xxx module
```

PR 描述至少写清：

```text
1. 做了什么
2. 修改了哪些主要文件/模块
3. 如何运行
4. 是否测试通过
```

然后提交 PR，并指定至少 1 名队友 Review。

---

## 6. Review 与合并

当前 `main` 已开启保护：

- 必须通过 PR 合并
- 至少 1 人 Approve
- Review 对话必须全部解决
- 合并前需要保持分支最新
- 使用线性历史
- 管理员也不能绕过规则

Review 通过后，推荐：

```text
Squash and merge
```

不要直接 push 到 `main`。

---

## 7. PR 出现冲突

先在自己的功能分支执行：

```bash
git fetch origin
git rebase origin/main
```

查看冲突：

```bash
git status
```

手动修改冲突文件后：

```bash
git add <冲突文件>
git rebase --continue
```

如果还有冲突，继续处理。

全部完成后：

```bash
git push --force-with-lease
```

如果处理错了：

```bash
git rebase --abort
```

重新开始。

原则：

> 谁提交 PR，谁负责解决自己的冲突。

---

## 8. PR 合并后

本地切回 `main`：

```bash
git switch main
git pull origin main
```

删除已经合并的本地分支：

```bash
git branch -d feature/data-converter
```

下一个功能重新从最新 `main` 创建：

```bash
git switch -c feature/new-feature
```

---

## 最小流程速查

### 第一次

```bash
git clone https://github.com/Yule-ui/lingbot-vla-v2.git
cd lingbot-vla-v2

git switch main
git pull origin main

git switch -c feature/xxx
git push -u origin feature/xxx
```

### 日常开发

```bash
git add .
git commit -m "feat: xxx"
git push
```

### 提 PR 前

```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
```

然后：

```text
feature/xxx
   ↓
Pull Request
   ↓
1 人 Review + Approve
   ↓
Squash and merge
   ↓
main
```

---

## 禁止事项

```bash
# 禁止
git push origin main

# 禁止
git push --force origin main
```

另外不要提交：

```text
datasets/
checkpoints/
outputs/
logs/
wandb/
*.pt
*.pth
*.ckpt
```

大型数据集、模型权重和训练结果放服务器共享目录，不进入 Git。
