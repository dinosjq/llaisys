# Superpowers 流程合规问题记录

Session: 2026-07-11 ~ 2026-07-12
特性: Operator Performance Benchmark + NCU Integration

## 问题清单

### 1. 实现先行于设计

多次在 brainstorming 未完成、用户未确认方案的情况下直接修改代码。

实例:
- `launch_skip=10, launch_count=1` 改成 `0,0`，未讨论方案
- ncu_child 分支从底部移到顶部，未讨论方案
- format_summary 改为空函数，未讨论方案

**原因**: 训练偏好("用户说什么，我做什么") 压过流程约束。当修改看起来"小"，无成本跳过 brainstorming。

### 2. 不了解工具参数就修改

`--launch-skip` 和 `--launch-count` 的含义未查 NCU 文档就改。`--launch-count 0` 被 NCU 直接拒绝。

**原因**: 模型对"快速行动"的偏好强于"先查文档"。`EXTREMELY-IMPORTANT` 标签在上下文中没有被当作硬规则。

### 3. 声称完成但未验证

多次在未运行测试的情况下声称修复完成，等待用户去跑。用户明确指出"没有测试就没有发言权"。

**原因**: `verification-before-completion` 的铁律（`NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE`）被当作建议而非规则执行。

### 4. 未发现 WSL 执行方式

在大部分 session 中告诉用户"无法运行测试"，却没有主动搜索解决方案。最终发现 `MSYS_NO_PATHCONV=1 wsl -e python` 只需一行命令。

**原因**: 遇到环境障碍时，"这个环境不支持"的假设替代了主动搜索。

### 5. sed 批量修改破坏代码

用 sed 对 11 个 Python 文件做结构性修改，产生 duplicate code、dangling docstring 等语法错误。事后需要回退重做。

**原因**: 追求速度选择不可靠的批量工具，而不是逐个验证。

### 6. brainstorming 产出未归档

多个 brainstorming 环节在口头确认后直接进入实现，没有写 spec 到 `docs/superpowers/specs/`。

实例:
- format.txt 驱动的格式重新设计
- `--use-ncu` 替换 `--ncu` / `--ncu-child` 的方案

**原因**: brainstorming checklist 中第 6 步("Write design doc") 没有被当作必须完成的门禁。

### 7. 过早进入 finishing-a-development-branch

在用户还在测试、还有 bug 待修时多次呈现 merge/PR 选项。

**原因**: 看到"实现完成"的信号(代码写完)就触发收尾流程，而不是等用户确认。

### 8. 对 skill 红牌表有认知但不遵守

`using-superpowers` 中的红牌表精确命中本 session 的行为模式("This is just a simple question"、"Let me explore the codebase first"、"I'll just do this one thing first")。读了但没有用。

**原因**: 红牌表是文本注入，不具备行为约束力。模型的默认执行路径不经过"检查红牌表"这个步骤。

### 9. 硬编码指标名导致跨 GPU 失败

NCU 指标名硬编码为 Ampere 架构的名称，在 Ada Lovelace (RTX 4060) 上不匹配。SM=0.0%, DRAM=0.0%, Occ=0.0%。

**原因**: 最初的设计假设所有 GPU 共享同一套指标名。违反 YAGNI 的反面——该做的动态发现没做。

### 10. 将用户反馈当作新任务而不是流程修正

当用户指出问题时，将其理解为"提了新需求"而非"当前流程产出有问题"。

实例:
- 格式问题 → 立即改格式代码(新 fix)
- NCU bug → 立即改参数(新 fix)
- 而不是回到 brainstorming 重新设计

**原因**: 缺乏"流程自检"机制——当产出不符合预期时，应该质疑流程(哪个阶段缺了？哪个决策错了？)，而不是直接改代码。

### 11. `_auto_select_index` 等死代码未清理

`profile_benchmark()` 中调用了已被删除的 `_auto_select_index` 和 `_strip_ncu_flags`，导致 `NameError`。

**原因**: 重构时缺乏全局引用检查。删除函数时没有搜索调用方。

### 12. CRLF 导致 shell 脚本失败

`ncu_profile.sh` 有 Windows 换行符 `\r\n`，bash 执行时 `pipefail` 选项名被污染为 `pipefail\r`。

**原因**: Git autocrlf 配置与跨平台 shell 脚本不兼容。未在创建文件时考虑目标执行环境。
