# LLAISYS 项目优化 Roadmap

---

## 已完成

1) 流式输出 — Python callback 机制实现 token-level streaming（完成）

2) 链路 Profile — nsys + 单次 prefill + 单次 decode 的 profiling 链路（完成）

3) fp16 / bf16 算子迁移 — 替换为现有表示，更新全部算子（完成）

4) KV Cache 泄漏修复 — 通过 profile 系统识别并修复 KV cache 内存泄漏（完成）

5) Top-K 优化 — 内置插入排序（小数据暴力排序即可）；外部 sample 直接顺序计算前缀（k 值不固定，涉及 SM/寄存器限制，暂不进行采样侧优化）（完成）

6) OpenAI-Compatible API Server（完成）
   FastAPI + uvicorn，直接对标 vLLM / SGLang / TensorRT-LLM 产品形态。
   - `/v1/chat/completions`（SSE streaming + 非流式）、`/v1/completions`、`/v1/models`、`/health`
   - 真异步：C++ 侧 busy-wait 替换为 condition_variable 真休眠，Python 侧 asyncio.to_thread + async generator
   - Request cancellation：客户端断开 → CancelledError → C API Abort → scheduler 延迟清理（worker 线程释放 KV cache，无跨线程数据竞争）
   - Per-request 采样参数：top_k/top_p/temperature 经 C API 下沉到 Sequence（构造时固化）
   - 集成测试 test/test_server.py 6/6 通过；回归测试无性能与正确性回退
   - Logprobs 已暂缓：无条件 logits 快照（D2H ~300KB/步）造成吞吐回退（1.16x→1.09x），后续以请求参数按需触发的方式重新实现

---

## 待完成

7) Paged Attention 长序列优化（核心瓶颈）
   链路 profile 发现长序列下 attention 耗时线性增长（2ms → 18ms @ seq_len=1024）。
   - FlashDecoding 风格 KV 分块并行：将 totlen 切分为多个 chunk，各 block 独立计算 partial softmax，最后 reduction 合并
   - 解决 decode 阶段 grid 过小（仅 12 blocks）导致 SM 利用率低的问题
   - 考虑 warp specialization（FA3 思路）进一步提升 SM 利用率

8) Linear 算子优化
   Linear 耗时稳定（~27ms/step），但对整体延迟贡献大。当前使用 cuBLAS（`cublasGemmEx`）+ 独立 `add_bias` kernel。
   - Fused QKV projection：合并 q/k/v 三次 GEMM 为一次（weight 沿 N 维拼接）
   - Fused bias + GEMM（利用 cuBLAS 的 `beta` 参数或手写 fused kernel）
   - 针对 decode 的 small-M GEMM 优化（M=1 时 cuBLAS launch overhead 显著）

9) 算子融合
   当前每个算子独立 launch，中间结果经 global memory 往返，存在大量冗余访存。
   - QKV Projection 融合：3× GEMM + 3× bias → 1× GEMM (weight concat) + 1× fused bias（6→2 kernel launches）
   - RMS-Norm + Residual Add 融合：单 kernel 内完成 norm + add，省一次 global mem round-trip
   - SwiGLU + Linear(down) 融合：单 kernel 完成 silu×up + linear(down)，省一次中间结果写回
   - RoPE + KV Cache Move 融合：RoPE 后直接写入 KV cache block，省一次 global mem 读写

10) 异步 Tokenizer
   Tokenization 在 CPU 上执行，可能成为 decode 循环的瓶颈。将 tokenizer 移到独立线程，与 GPU 推理流水线化。

11) 算子通用性优化
    方便适配其他模型架构，减少硬编码假设（如 d=128、nkvh=2 等维度特化）。

12) 性能 Benchmark 报告
    当前有散落的 JSON profile 数据，需整理成可呈现的文档+图表。定量展示优化成果。
    - Throughput (tokens/sec) vs batch_size 曲线
    - TTFT vs prompt_length，TPOT vs seq_len
    - 每个算子的 micro-benchmark（耗时 + 带宽利用率）
    - Roofline 分析：各算子的 arithmetic intensity vs 理论峰值
    - 与 HuggingFace Transformers / vLLM 的加速比（同模型同硬件）
    - Ablation：各优化项的独立收益（on/off 对比）

13) Prefix Caching 增强
    已有基础 hash-based prefix caching，需升级。
    - Radix Tree（前缀树）管理缓存块 — 自动检测跨请求公共前缀，对标 SGLang
    - LRU / LFU eviction — 当前只"踢掉最老的序列"，无基于访问频率的淘汰策略
    - Cache hit rate metrics — 统计命中率、块复用率，量化缓存效果
    - System prompt 缓存 — 多轮对话中 system prompt 只计算一次

14) 适配新模型：Llama-3.2-3B-Instruct
    本地适配约占 6GB 显存，接近 RTX 4060 上限。
    https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
    在此基础上推进 15) Model-Agnostic Pipeline。

15) Model-Agnostic Pipeline
    当前模型层硬编码 Qwen2（28 layers, GQA ratio=6, SwiGLU+RMS-Norm+RoPE）。
    - JSON/YAML config 描述模型结构（layer 类型、参数、连接关系）
    - 类似 llama.cpp GGUF 格式或 vLLM model config
    - 统一 Layer 基类，组合式构建 pipeline
    - 新模型只需 config + safetensors 加载，无需改 C++ 代码

16) 模型量化
    权重量化：NF4 量化 + Hadamard 旋转（Tensor-Core 加速），预期跑更大模型且性能基本不变。
    KV Cache 量化：FP8 (E4M3) 或 INT8 K/V cache。写入时 cast → 读取时 paged_attention kernel 内 dequant。
    效果：同样显存支撑 2× 更长序列或 2× 更大 batch。对标 vLLM / TensorRT-LLM 标配技术。

17) CUDA Graph（Decode）
    Decode 阶段 kernel launch overhead 在高 batch 下明显。CUDA Graph 将整套 kernel 序列录制为 graph，单次 launch 执行。
    注意：paged attention 中 block_id table 动态变化，需处理 graph 的动态输入。

18) Speculative Decoding
    - Prompt Lookup Decoding：从 prompt 中匹配 n-gram 作为 draft token，无需额外模型，~100 行代码，~2× 加速。性价比极高。
    - MTP (Medusa-style)：多个预测头并行预测，需训练/加载额外参数。
    - Eagle-style：独立 draft model，精度最高。
    建议先实现 PLD，再考虑 MTP。

19) 多卡并行：TP + PP + DP
    预计 128GB 4×5090 可用，在多卡服务器上进行。
    前置依赖：NCCL 集成（`ncclAllReduce`、`ncclSend`/`ncclRecv`），当前项目无 NCCL。
    - TP (Tensor Parallelism)：同模型更快推理速度，降低 TTFT/TPOT，提高 throughput
    - PP (Pipeline Parallelism)：可跑更大模型（超过单卡显存）
    - DP (Data Parallelism)：相较单卡更高吞吐量

20) 模型部署 + Docker
    Docker 一键部署：`docker run -p 8000:8000 llaisys-server`，方便面试演示和开源展示。
    包含模型权重挂载、CUDA 环境配置。

21) Structured Output
    JSON mode / grammar-constrained decoding，OpenAI structured output 热门 feature。

22) LoRA 适配器
    基于 PEFT 的热插拔 LoRA，展示对模型服务生态的理解。

23) Beam Search
    当前仅支持 sampling，beam search 是基础能力的补齐。

24) 多轮对话 KV Cache 复用
    对话轮次间共享 prefix，减少重复计算。

---

## 收尾

25) 更新 README.md / README_ZN.md（补充 benchmark 数据与优化成果）
26) 完善 .gitignore
27) 架构设计文档：记录关键设计决策与 trade-off
28) GPU CI（当前 CI 仅 CPU，无法验证 CUDA 算子正确性）
