# DeepSeek-V4 W8A8 在 Ascend 910C 上为什么不一定更快？从 Decode 小 M、FRACTAL_NZ 到 CANN Tiling

上一篇 [《DeepSeek-V4 W8A8 推理在 Ascend 910C 上到底发生了什么？从量化权重到 INT8 MatMul Kernel》](../ascend/deepseek-v4-w8a8-ascend-910c.md) 已经把执行链追到了：

~~~text
ModelSlim W8A8_DYNAMIC
        ↓
BF16 hidden
        ↓
Dynamic Quant
        ↓
INT8 activation + per-token scale
        ↓
npu_quant_matmul / npu_grouped_matmul
        ↓
ACLNN / CANN
        ↓
Ascend 910C AI Core
~~~

接下来真正值得问的不是“INT8 理论算力是不是更高”，而是：

> **这条 INT8 数据通路在当前 workload 下，究竟有没有把理论优势兑现成 TTFT、TPOT 和吞吐收益？**

对 DeepSeek-V4 这种同时包含 Attention、MoE、DeepEP、DPA 和投机推理的模型，答案不可能只由 dtype 决定。

更接近工程现实的性能模型是：

~~~text
W8A8 的实际收益
=
Quantized GEMM 节省的时间

-
Activation Quant 成本
-
额外 Scale / Launch / Epilogue 成本
-
小 M 下的低并行度
-
不理想 Layout / Tiling
-
MoE 小 Expert Batch
-
Attention / Communication / Runtime 中未被量化加速的时间
~~~

本文对四个最容易被经验性描述带偏的问题做严格源码 Review：

1. <code>npu_dynamic_quant</code> 在 Decode 小 M 时到底能从源码证明什么；
2. <code>aclnnQuantMatmulWeightNz</code> 对 M=1/2/4/8 是否真的存在固定 tiling bucket；
3. FRACTAL_NZ 对 DeepSeek-V4-Flash 的真实 N/K shape 到底意味着什么；
4. Expert-local <code>M_e</code> 与 <code>npu_grouped_matmul</code> 是否存在源码可证明的性能 cliff。

源码固定到：

| 组件 | 版本 |
| --- | --- |
| SGLang | <code>b63f8416b3b73bafdec029005c5db36bad207b44</code> |
| msModelSlim | <code>85d6c6f0266fd1fd77aa19ac26086670ca92d1db</code> |
| Ascend op-plugin | <code>e89cc608309341298193f957fa87c964ee561781</code> |
| CANN ops-nn GitHub mirror | <code>0026707d7ecb24d258e030d831c8121ac2d055bd</code> |
| CANN ops-transformer GitHub mirror | <code>bbb4dafe296119266a0a64133f17bc42b6597ed3</code> |
| DeepSeek-V4-Flash-0731 config | Hugging Face main，核对日期 2026-09-21 |

文中的 CANN GitHub 仓库用于源码检索便利；发布或部署时仍应以目标环境实际安装的 CANN 版本为准。

---

## 一、先建立正确的性能模型：W8A8 优化的是一部分时间，不是整个 Decode

一个普通 Dense Linear 可以写成：

~~~text
Y = XW

X: [M, K]
W: [K, N]
Y: [M, N]
~~~

如果是 BF16，粗略写成：

~~~text
T_BF16_linear
≈
T_BF16_GEMM
~~~

当前 DeepSeek-V4-Flash ModelSlim 主路径是 <code>W8A8_DYNAMIC</code>。普通 Tensor 输入进入 <code>NPUW8A8Int8DynamicLinearMethod.apply()</code> 后，执行链是：

~~~text
BF16 X
   ↓
npu_dynamic_quant
   ↓
INT8 X + per-token scale
   ↓
npu_quant_matmul
   ↓
BF16 Y
~~~

因此：

~~~text
T_W8A8_linear
≈
T_dynamic_quant
+
T_quant_matmul
+
T_other_epilogue
~~~

真正的 Linear 加速条件是：

~~~text
T_BF16_GEMM - T_INT8_GEMM
>
T_dynamic_quant + 新增的其它成本
~~~

到了整个 Decoder Layer，问题更明显：

~~~text
T_layer
=
T_norm
+
T_attention
+
T_dense
+
T_router
+
T_dispatch
+
T_expert
+
T_combine
+
T_collective
+
T_residual
+ ...
~~~

即使某个 QuantMatmul 快很多，也只能优化它自己占据的时间。

这就是为什么“INT8 峰值 TOPS 更高”不能直接推出“TPOT 按同样比例下降”。

### W8A8 还不是每个 Linear 都必然单独启动 DynamicQuant

第一版文章里最容易过度概括的一句话是：

~~~text
每个 W8A8 Dynamic Linear
=
npu_dynamic_quant
+
npu_quant_matmul
~~~

当前 SGLang 实际还有一个重要 fast path：

~~~python
if isinstance(x, tuple):
    original_dtype = torch.bfloat16
    quant_out, dynamic_scale = x
else:
    quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
~~~

也就是说，如果上游 fused/prolog kernel 已经产生：

~~~text
(INT8 payload, scale)
~~~

这个 Linear 不会再单独执行一次 <code>npu_dynamic_quant</code>。

因此分析真实 Timeline 时必须先回答：

> 当前这个 Linear 的输入是 BF16 Tensor，还是已经量化好的 tuple？

只有前者才存在独立 DynamicQuant Kernel 的成本。

---

## 二、Decode 小 M：DynamicQuant 的低并行度可以从 CANN 源码直接证明，但微秒数不能

这一点是本轮 Review 最重要的源码结论之一。

CANN arch35 的：

~~~text
quant/dynamic_quant/
└── op_host/
    └── dynamic_quant_tiling_arch35.cpp
~~~

在普通非 per-channel 路径会调用：

~~~text
CalculateCoreNum(context)
CalculateTilingData()
~~~

其中 <code>CalculateCoreNum()</code> 直接把输入最后一维当成 row width，把其余维度乘积当成 row 数：

~~~cpp
rowLen = x.shape[-1];

rowNum =
    product(x.shape[:-1]);

coreNum =
    max(
        min(vectorCoreNum, rowNum),
        1
    );
~~~

对于普通二维 Dense 输入：

~~~text
X.shape = [M, K]
~~~

于是：

~~~text
rowNum = M
rowLen = K
~~~

也就是：

~~~text
coreNum
=
min(Vector Core 数, M)
~~~

在没有触发“超长单 row 再沿尾轴拆分”的特殊路径时：

| M | DynamicQuant 最初可使用的 Vector Core 数上限 |
| ---: | ---: |
| 1 | 1 |
| 2 | 2 |
| 4 | 4 |
| 8 | 8 |

这不是性能猜测，而是当前 arch35 Host Tiling 的直接逻辑。

所以低并发 Decode 的问题可以更加准确地表述为：

> **Standalone per-token DynamicQuant 的 row-level parallelism 直接受 M 限制。M 很小时，即使芯片拥有更多 Vector Core，这个算子本身也没有足够多的 Token row 去铺满它们。**

### 一个重要例外：超长 row 可以沿尾轴进一步拆

源码没有简单停在 <code>coreNum=M</code>。

<code>CalculateTilingData()</code> 会计算一行在 UB 中需要的空间。如果一个 row 大到一个 UB 都放不下，而且满足：

~~~text
rowNum <= vectorCoreNum / 2
~~~

以及：

~~~text
calcSize >= 4 × maxUseUbSize
~~~

就可能进入：

~~~text
CalculateTilingForPertenLargeMulticore
~~~

把单行尾轴再拆给更多核。

因此更严格的说法不是：

> M=1 永远只能用一个 Core。

而是：

> **普通可整行处理的 per-token DynamicQuant，row 并行度由 M 决定；只有 row 本身足够大时，CANN 才有额外的尾轴多核拆分路径。**

DeepSeek-V4 的主要 W8A8 Dense K 值是 4096 或 1024，究竟在目标 CANN build 上走 full-load 还是特殊大-shape 路径，应读取实际 tiling log，而不是从模型 config 单独推断。

### 为什么仍然不能写“DynamicQuant M=1 = X μs”

源码能证明：

- 它是独立算子路径；
- 它要处理每一行并生成 per-token scale；
- Host Tiling 会根据 row 数分配 Vector Core；
- 最终 <code>SetBlockDim(coreNum)</code> 发布实际 block 数。

源码不能单独给出：

~~~text
M=1 → 8.3 μs
M=2 → 8.8 μs
...
~~~

真实设备时间还取决于 CANN build、Stream 状态、Kernel launch/enqueue、K、UB template、Graph replay 和前后算子融合。

所以本文能够严格确认的是：

> **小 M 会直接限制 standalone DynamicQuant 的可用 row parallelism。**

具体 launch + device duration 仍必须由 Ascend 910C profiler 给出。

---

## 三、aclnnQuantMatmulWeightNz 的 M=1/2/4/8：源码存在 Small-M 逻辑，但不存在一张固定 bucket 表

第二个 Review 点需要纠正一种常见写法。

我们很容易写：

~~~text
M=1 → Tiling A
M=2 → Tiling B
M=4 → Tiling C
M=8 → Tiling D
~~~

当前公开的 QuantBatchMatmulV3 arch35 源码并不支持这种简单结论。

### M 的确是 Tiling 的一等输入

在 <code>base_block_calculator.cpp</code> 中，默认 base block 直接使用：

~~~cpp
baseM =
    CeilAlign(
        min(mSize, 256),
        baseMAlign
    );
~~~

StreamK 初始化同样从真实 <code>inputParams_.mSize</code> 计算：

~~~text
baseM
mCnt
nCnt
preSplitKBlockCnt
~~~

然后决定是否进入：

~~~text
UpdateSmallMnStreamKBase()
~~~

公开代码里明确存在 SmallMN / StreamK 逻辑。

这说明：

> **M 改变会真实影响 baseM、M block 数、MN 并行度、StreamK 选择以及最终 used cores。**

这是源码事实。

### 但没有找到“1/2/4/8 四档固定分支”

对当前 arch35 QuantBatchMatmulV3 Tiling 代码做针对性检索后，没有发现类似：

~~~cpp
if (M == 1) ...
else if (M == 2) ...
else if (M == 4) ...
else if (M == 8) ...
~~~

这样的 A8W8 WeightNz Decode bucket 表。

相反，源码大量使用：

~~~text
mSize
CeilAlign
CeilDivision
baseM
mCnt
nCnt
coreNumMN
SmallMn
~~~

连续地推导分块。

这意味着：

> **M=1/2/4/8 当然可能得到不同 tiling 结果，但“不同”来自一套 shape-driven 计算，而不是当前公开源码里一张固定的四档查表。**

### 为什么现实中仍然可能看到 latency cliff

没有硬编码 1/2/4/8 bucket，不代表性能曲线一定平滑。

下面任何一个整数边界都可能造成离散变化：

~~~text
CeilAlign(M, alignment)
CeilDiv(M, baseM)
used core count
StreamK enable / disable
tail block
L1 / L0 capacity
Graph bucket / padding
~~~

所以 Profile 完全可能看到：

~~~text
M=7 → latency A
M=8 → latency B
~~~

但正确证据顺序应该是：

> 先观测 cliff，再用 tiling dump 解释。

而不是：

> 因为源码一定有 8 的 bucket，所以提前宣称这里必然 cliff。

### 最值得做的实际实验

固定一组真实 DeepSeek-V4 Linear，例如：

~~~text
wq_a:
K=4096
N=1024
~~~

只 sweep：

~~~text
M = 1, 2, 4, 8, 16, 32, 64
~~~

每个点记录：

~~~text
operator duration
tiling key
baseM / baseN / baseK
used cores
StreamK status
~~~

这样才能真正得到：

> **Ascend 910C + 当前 CANN build + 当前 DeepSeek-V4 shape 的小 M QuantMatmul 曲线。**

---

## 四、FRACTAL_NZ：DeepSeek-V4 的主要 W8A8 Dense Shape 天然对齐，但“对齐”不等于“已证明快多少”

当前 SGLang 对 W8A8 Weight 会尝试：

~~~python
weight =
    npu_format_cast(
        weight,
        ACL_FORMAT_FRACTAL_NZ
    )
~~~

对 INT8 的预检查是：

~~~text
K % 16 == 0
N % 32 == 0
~~~

不满足就保持 ND，并提示可能降低性能。

本轮 Review 的关键是把规则代入 DeepSeek-V4-Flash 的真实 shape。

### 官方模型的关键维度

DeepSeek-V4-Flash-0731 config：

~~~text
hidden_size            = 4096
q_lora_rank            = 1024
head_dim               = 512
num_attention_heads    = 64
o_lora_rank            = 1024
o_groups               = 8
moe_intermediate_size  = 2048
~~~

官方 ModelSlim W8A8 recipe 对 Attention include <code>*attn*</code>，但明确排除：

~~~text
*wo_a
*wo_b
*compressor.wgate
*compressor.wkv
*indexer.weights_proj
...
~~~

因此分析 Dense W8A8 FRACTAL_NZ 时，不应该拿被排除的 <code>wo_a/wo_b</code> 当主要样本。

真正典型的量化 Dense Linear 包括：

| Layer | 逻辑 K | 逻辑 N |
| --- | ---: | ---: |
| <code>wq_a</code> | 4096 | 1024 |
| <code>wkv</code> | 4096 | 512 |
| <code>wq_b</code> | 1024 | 32768 / <code>attn_tp_size</code> |

以官方 TP16/DP16 DPA 配置为例：

~~~text
attn_tp_size = 1
~~~

所以：

~~~text
wq_b:
K = 1024
N = 32768
~~~

这些典型 shape 全部满足当前 SGLang 的 INT8 FRACTAL_NZ alignment。

因此：

> **对官方主要 W8A8 Attention Dense 矩阵，SGLang 的 NZ alignment 预检查本身不是障碍。**

### FRACTAL_NZ 还会改变 op-plugin 的真实入口

当前 op-plugin：

~~~text
Weight = FRACTAL_NZ
        ↓
aclnnQuantMatmulWeightNz

Weight = ND
        ↓
aclnnQuantMatmulV5
~~~

所以 FRACTAL_NZ 不只是存储格式不同，而是直接改变下游 QuantMatmul dispatch。

### 但源码仍然不能告诉我们“快 X%”

到这里能够证明的是：

~~~text
这些 DSV4 shape 能通过 NZ alignment
        ↓
SGLang 会尝试 format cast
        ↓
成功后进入 WeightNz ACLNN 路径
~~~

不能证明：

~~~text
WeightNz 比 ND 快固定 X%
~~~

收益仍然受 M、K/N、CANN tiling、cache、HBM、Prefill/Decode 等影响。

因此最干净的验证方式，是对同一个 checkpoint 做 A/B：

~~~text
A:
默认
→ FRACTAL_NZ
→ aclnnQuantMatmulWeightNz

B:
SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
→ ND
→ aclnnQuantMatmulV5
~~~

保持模型、M/N/K、Batch、并行配置、CANN 和 Graph 配置全部不变。

然后比较：

~~~text
QuantMatmul duration
Dense layer duration
TTFT / TPOT
throughput
~~~

需要注意：这个环境变量影响 Weight load，因此 A/B 需要分别重启服务，不能在一个已加载模型进程里动态切换。

---

## 五、MoE 的 Expert-local M_e：CANN 确实有 128-token 调优阈值，但当前 SGLang 不能直接据此宣称有 128 cliff

DeepSeek-V4 MoE 一轮不是一个共享 Weight 的大 GEMM，而是：

~~~text
Expert 0: M_0
Expert 1: M_1
...
Expert E: M_e
~~~

当前 SGLang Ascend Runner 把：

~~~text
expert_tokens
~~~

作为：

~~~python
group_list=expert_tokens
~~~

传给 <code>npu_grouped_matmul</code>，所以 actual expert token distribution 是 GMM 的真实输入。

### CANN 源码真的出现了 128

在当前公开 GroupedMatmul tiling 源码中有非常明确的注释：

~~~text
实测当单专家 token 数低于 128 时
cube 算力不能完全发挥，
导致开启 2 个 vector 核可能会劣化
~~~

并定义：

~~~cpp
SMALL_TUNING_CONFIG_THRESHOLD = 128;
~~~

后面决定 AIV:AIC 比例时：

~~~cpp
has_sufficient_tuning =
    tuningConfig_ >= 128;
~~~

源码还包含一个 A8W8 Fixed-Axis 优化，要求 expected token / expert 处于：

~~~text
128 ~ 512
~~~

这说明：

1. CANN 开发者明确把“预期每 Expert Token 数”作为性能调优维度；
2. 128 在某些 A8W8 GroupedMatmul 优化策略中确实是一个有实测背景的阈值。

### 但关键边界是：SGLang 当前没有显式传 tuning_config

torch-npu 接口支持：

~~~python
npu_grouped_matmul(
    ...,
    tuning_config=None,
)
~~~

而当前 SGLang 的 <code>GroupedMatmul.forward()</code> 调用没有传这个参数。

op-plugin / CANN 未收到显式 tuning hint 时，对应：

~~~text
tuningConfig_ = 0
~~~

而不是 128、192 或其它预期每 Expert Token 数。

所以不能写：

> SGLang DeepSeek-V4 在每 Expert 128 Token 时一定切换 tiling，因此延迟一定有 cliff。

当前证据只支持：

> **CANN 有一套使用 expected-token hint 的调优机制，其中 128 是重要阈值；当前 SGLang 普通 GroupedMatmul wrapper 没有显式传这个 hint。**

### DeepSeek-V4 也不匹配源码里的 Fixed-Axis 特殊 shape

CANN 当前 Fixed-Axis A8W8 条件要求形如：

~~~text
(K,N)
=
(2048,7168)
or
(7168,4096)
~~~

而 DeepSeek-V4-Flash 的 Expert：

~~~text
W13:
K = 4096
N = 4096

W2:
K = 2048
N = 4096
~~~

因此这套特定 Fixed-Axis 优化不能直接套到 DeepSeek-V4 W8A8 Expert 上。

这是一个很具体的反例：

> **即使都是 DeepSeek 系 MoE，也不能把针对另一代 K/N shape 调出来的 GMM 优化直接当成 DeepSeek-V4 的现状。**

### 那 M_e 还重要吗？当然重要

没有显式 tuning hint，不代表 Expert Token 分布不重要。

实际 group list 已经决定了每个 Expert 真正计算多少 row，所以：

~~~text
总计算量
每个 Expert 的有效 GEMM M
空 Expert 数
最重 Expert
执行不均衡
~~~

都会随 <code>{M_e}</code> 改变。

但是“延迟曲线在哪些整数点发生 cliff”仍然是实验问题。

严谨的 sweep 应该覆盖：

~~~text
M_e =
1, 2, 4, 8, 16, 32, 64,
96, 127, 128, 129,
160, 192, 256
~~~

记录：

~~~text
GroupedMatmul duration
tiling key
AIV:AIC ratio
workspace
expert_tokens histogram
~~~

128 值得重点观察，因为 CANN 源码告诉我们它是一个调优阈值；但是否形成当前 SGLang 路径的真实 latency cliff，必须让 profiler 给答案。

---

## 六、严格 Review 之后，W8A8 性能应该怎么测

经过这轮源码核验，第一版文章里的 M sweep 可以升级成一套真正可判因果的实验矩阵。

### 实验 A：Standalone DynamicQuant 的小 M 成本

固定：

~~~text
dtype = BF16 → INT8
K = 1024 / 4096
~~~

sweep：

~~~text
M = 1, 2, 4, 8, 16, 32, 64
~~~

记录：

~~~text
DynamicQuant device duration
Host enqueue gap
blockDim / coreNum
tiling key
~~~

重点验证源码预测：

~~~text
普通二维 per-token path:
coreNum ≈ min(vectorCoreNum, M)
~~~

同时单独标记已经由上游 fused/prolog 产生 <code>(quant, scale)</code> tuple 的路径，因为那种情况下根本不存在 standalone DynamicQuant。

### 实验 B：WeightNz Quantmatmul 的 M sweep

选真实 DSV4 shape：

~~~text
wq_a:
K=4096, N=1024

wkv:
K=4096, N=512

wq_b:
K=1024, N=32768 / attn_tp_size
~~~

sweep：

~~~text
M = 1,2,4,8,16,32,64,128
~~~

每个点记录：

~~~text
tiling key
baseM/baseN/baseK
mCnt/nCnt
StreamK
used cores
kernel duration
~~~

目标不是寻找预设 1/2/4/8 bucket，而是从真实 dump 中发现哪些 M 真正改变了 tiling 结构。

### 实验 C：FRACTAL_NZ A/B

两次独立启动：

~~~text
A:
默认 FRACTAL_NZ

B:
SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
~~~

对同一组 M/N/K 比较：

~~~text
WeightNz vs V5
operator duration
TTFT
TPOT
throughput
~~~

只有这一步以后才应该写：

> FRACTAL_NZ 在 DeepSeek-V4 这个 shape 上带来 X% 收益。

### 实验 D：Expert-local M_e sweep

固定 Expert shape：

~~~text
W13: K=4096, N=4096
W2 : K=2048, N=4096
~~~

控制 M_e，并重点采样 128 两侧：

~~~text
1,2,4,8,16,32,64,
96,127,128,129,
192,256
~~~

记录：

~~~text
GMM duration
tiling key
AIV:AIC ratio
expert token histogram
empty expert ratio
~~~

如果 128 附近真的出现 cliff，再回头检查实际 op 参数和 tiling 是否发生变化，而不是看到源码常量 128 就提前下结论。

### 最后才看端到端

单算子 benchmark 只能回答“哪个 Kernel 快了”。

最终还要回到：

~~~text
TTFT
TPOT
output token throughput
request throughput
~~~

并拆清：

~~~text
Prefill
Decode
Target Verify
~~~

当前 SGLang 的 DeepSeek-V4-Flash W8A8 NPU CI 设置了：

~~~text
input_len        = 8000
output_len       = 1000
max_concurrency  = 160
TPOT gate        = 50 ms
output throughput gate = 3100 token/s
~~~

同时开启 ModelSlim W8A8、DPA、DeepEP INT8 和 DSpark。

这证明的是：

> 这一整套 W8A8 系统要达到一个绝对性能门槛。

它不是严格的 BF16 vs W8A8 控制变量实验，因此不能从这个 CI 门槛推导“W8A8 快 X%”。

---

## 结论：这四个 Review 点之后，哪些可以下结论，哪些不能

| 问题 | 严格结论 |
| --- | --- |
| DynamicQuant 小 M | 普通二维 per-token 路径的 row 数就是 M，初始 <code>coreNum=min(vectorCoreNum,M)</code>；小 M 明确限制 row-level 并行度 |
| DynamicQuant μs | **源码不能给出**，必须 profiler |
| Quantmatmul M=1/2/4/8 | M 会进入 baseM / block / SmallMN / StreamK 计算，但没有发现固定 1/2/4/8 查表分支 |
| FRACTAL_NZ 对齐 | DSV4 主要 W8A8 Dense shape <code>wq_a/wkv/wq_b</code> 天然满足当前 INT8 NZ alignment |
| FRACTAL_NZ 收益百分比 | **源码不能给出**，必须 NZ vs ND A/B |
| Expert <code>M_e</code> | actual group_list 决定真实 Expert workload；CANN 还提供 expected-token tuning 机制 |
| 128 token threshold | CANN 源码中真实存在，但属于 tuning-config 驱动的优化条件 |
| 当前 SGLang 是否必在 128 cliff | **不能这样说**：当前 wrapper 没有显式传 tuning_config |
| DSV4 是否命中 Fixed-Axis 特殊优化 | 当前 W13/W2 K/N 不匹配该源码分支的特定 shape |

因此，“DeepSeek-V4 W8A8 为什么不一定更快”最准确的答案不是一句：

~~~text
因为 INT8 有量化开销
~~~

而是：

~~~text
Decode 小 M
    ↓
DynamicQuant row parallelism 受限

真实 M/N/K
    ↓
Quantmatmul 重新计算 baseM / StreamK / used cores

Weight Layout
    ↓
NZ / ND 进入不同 ACLNN 路径

Router / DeepEP
    ↓
形成不同 Expert-local M_e

CANN Tiling
    ↓
决定每个算子的真实设备利用率

最后：
只有被这些算子占据的那部分时间
才能转化成 TTFT / TPOT / Throughput 收益
~~~

这也是性能工程里比“INT8 峰值算力高多少”更重要的判断：

> **dtype 只是性能输入之一；shape、layout、tiling 和 workload ownership 才决定这份理论算力到底有没有机会被用出来。**

---

## 源码入口

SGLang：

- [NPUW8A8Int8DynamicLinearMethod](https://github.com/sgl-project/sglang/blob/b63f8416b3b73bafdec029005c5db36bad207b44/python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py)
- [npu_format_cast / _is_nz_aligned](https://github.com/sgl-project/sglang/blob/b63f8416b3b73bafdec029005c5db36bad207b44/python/sglang/srt/hardware_backend/npu/utils.py)
- [Ascend GroupedMatmul wrapper](https://github.com/sgl-project/sglang/blob/b63f8416b3b73bafdec029005c5db36bad207b44/python/sglang/srt/hardware_backend/npu/moe/matmul.py)
- [W8A8 NPU performance CI](https://github.com/sgl-project/sglang/blob/b63f8416b3b73bafdec029005c5db36bad207b44/test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py)

Model / Quantization：

- [DeepSeek-V4-Flash-0731 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json)
- [ModelSlim DeepSeek-V4-Flash W8A8 recipe](https://github.com/Ascend/msmodelslim/blob/85d6c6f0266fd1fd77aa19ac26086670ca92d1db/lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml)

CANN：

- [DynamicQuant arch35 tiling](https://github.com/hicann/ops-nn/blob/0026707d7ecb24d258e030d831c8121ac2d055bd/quant/dynamic_quant/op_host/dynamic_quant_tiling_arch35.cpp)
- [QuantBatchMatmulV3 base block calculator](https://github.com/hicann/ops-nn/blob/0026707d7ecb24d258e030d831c8121ac2d055bd/matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/base_block_calculator.cpp)
- [QuantBatchMatmulV3 StreamK tiling](https://github.com/hicann/ops-nn/blob/0026707d7ecb24d258e030d831c8121ac2d055bd/matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/qbmm_streamk_tiling.cpp)
- [GroupedMatmul tiling](https://github.com/hicann/ops-transformer/blob/bbb4dafe296119266a0a64133f17bc42b6597ed3/gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp)

下一步如果要把本文从“源码严格推导”升级成“Ascend 910C 实测结论”，最值得做的不是继续读更多代码，而是跑四组最小实验：DynamicQuant M sweep、WeightNz M sweep、NZ/ND A/B、Expert <code>M_e</code> sweep，然后把 tiling dump 与 profiler 时间线一一对上。
