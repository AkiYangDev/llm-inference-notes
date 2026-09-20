# 推理机制的专项检查

仅在正文涉及通信、显存、Prefill/Decode 或 PD 分离时读取对应段落。这些是核验问题，不是目标框架已经验证的行为，也不要求文章增加这些主题。

通信能确认时明确 All-Reduce、All-Gather、Reduce-Scatter、All-to-All 或 Point-to-Point，解释参与方、输入/输出、为什么需要、在哪里发生。HCCL 是通信库名称，不是足够精确的 collective 类型。不能确认时标明未知，不猜原语，不把所有通信都描述成同步阻塞。

设备内存按实际情况拆开：Model Weights、Activations、KV Cache、Kernel Workspace、Runtime Buffer、Communication Buffer、Graph Capture / Static Buffer、Allocator Reserved Memory、Fragmentation。区分用途与分配器统计口径，避免把 reserved 与其中的活跃分配简单相加；不得统一称为“模型占用”。公式写清 batch/token 长度、层数、head 类型、dtype 字节数、分片或复制、额外开销等假设；标准 KV 公式不能直接套 MLA 或特殊压缩缓存。

涉及 Prefill/Decode 时区分：Prefill 处理 prompt，通常在每层对多个 prompt token 并行计算并建立相应 KV；Decode 使用已有 KV 继续生成，标准自回归情形每轮通常为每条活跃序列处理一个新 token。区分模型计算与采样，必要时指出第一枚输出 token 可由 Prefill 的 logits 采样得到。chunked prefill、投机解码和混合批处理按具体机制解释，不硬套单 token 模板。

解释两阶段计算、带宽和缓存访问特征如何影响调度及 PD 分离；不要把“Prefill 一定计算受限、Decode 一定带宽受限”当无条件定律。PD 分离的额外 KV 传输与调度成本也要在相关讨论中交代。

