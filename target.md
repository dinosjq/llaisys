1) 流式输出 （完成）

2) 链路 profile：nsys + 1 prefill + 1 decode（完成）

nvidia 链路替换 fp16 bf16 为对应的现有表示，更新算子（完成）

基于 benchmark 与 profile 链路优化系统
5090 还是 4060 上做了

3) 适配新模型: 更大的模型

4) 模型部署

5) 模型量化

6) 张量并行 TP + 流水线并行 PP

7) MTP 投机解码


最后更新一下 README.md，然后更新 gitignore 收一下尾就可以了。