# 正确的批量推理语义

> 交付格式说明：Markdown 是可追溯的源文档；中文 Cursor Canvas 是持续更新的可视化概览。

## 已确认缺陷

- 前缀缓存冲突验证将每个已缓存 token 都与候选项索引 1 进行比较。
- 采样对 nucleus 候选项采用均匀处理，并且没有保留第一个使累计概率越过 `top_p` 的 token。
- 当队首 prompt 大于剩余 token 预算时，chunked prefill 在调度该 prompt 之前就停止了。
- 旧版基于 prompt hash 的请求 API 会让相同 prompt 共享同一个 `Sequence`。

## 设计与 API

`sampling.hpp` 集中实现有界 top-k、temperature softmax、nucleus filtering 和 weighted sampling。
每个 `Sequence` 拥有一个 RNG；每个 batch 先按请求中的最大 K 在设备上执行一次 top-k，
然后由 CPU sampling 对各行应用其专属的 K。

`LlaisysQwen2Request` 是一个 opaque handle，提供 submit、await、abort 和 release 入口。
它拥有独立的 `Sequence`；缓存的 KV block 仍然只能通过 `BlockManager` 共享。

## 测试

核心回归测试覆盖逐元素前缀匹配、完整 block 复用、确定性的 greedy/temperature/top-p/weighted sampling，
以及多轮 chunked prefill 的推进。在模型可用时，服务端集成测试还会验证并发相同 prompt，
以及取消操作之间的隔离。

## 行为与兼容性

Python 生成和服务端流式输出只使用 request handle，并在正常完成、异常和取消时释放它。
旧的 `llaisysQwen2ModelInfer/Abort` 及 prompt-hash `Sequence` 复用路径已删除；Prefix Cache
继续由 `BlockManager` 提供。

## 性能、限制与回滚

NVIDIA top-k 最终保留并行 radix kernel；评审期间出现过的串行 fallback 已被移除。最终 BF16
`(1, 151936), K=100` smoke 测试与 `torch.topk` 数值一致，延迟中位数为 `1.2815 ms`。
当前没有公共 request seed API；确定性 RNG 注入仅用于内部测试。回滚 Task 2 的 commits，
即可恢复此前的请求与采样行为。

后续 API 清理主动中断了旧 `llaisysQwen2ModelInfer/Abort` C ABI，不为仓库外旧调用者提供兼容层。

## 评审修复

取消操作现在会 shield 正在执行的阻塞式 `RequestAwait` task，并将其 opaque C handle 的释放推迟到
该 task 在 abort 后结束。两个服务端 endpoint 都会保留显式设置的 `temperature=0`。
构造 Sequence 时，会在解引用 token storage 前验证 prompt 非空指针且内容非空。回归测试套件还覆盖
每个请求使用不同 top-k 的混合场景，以及并发相同 prompt 之间的取消隔离。

## 第二轮评审修复

已移除串行 CUDA fallback。NVIDIA 实现已精确恢复为通过评审的 `960088c` 并行 radix kernel；
完整 NVIDIA top-k 测试套件经过三次全新构建并安装后均通过，借助现有通用覆盖验证了 FP32、FP16、BF16、
批量行以及 K。请求提交现在会先执行 model submit，再分配其 C wrapper；如果 wrapper 构造失败，
则释放已经提交的请求。

重新构建的隔离库针对 `(1, 151936), K=100` 的 BF16 smoke 测试使用了 10 次 warmup、
20 次 CUDA event 测量，并进行了显式同步：数值与 `torch.topk` 一致；延迟中位数为 `1.2815 ms`，
最小延迟为 `1.2780 ms`。
