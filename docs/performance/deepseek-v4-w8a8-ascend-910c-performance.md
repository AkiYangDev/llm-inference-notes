# DeepSeek-V4 W8A8 性能分析：小 M、NZ 布局与测量方法

W8A8 把部分矩阵乘法换成低精度路径，同时增加激活动态量化、Scale 处理和相关算子开销。最终收益取决于同一负载下省下的 GEMM 时间，能否覆盖这些额外成本。

Decode 的小 M、MoE 的 Expert Batch 分散，以及 NZ 布局和 Tiling，都可能改变这个收支。下面沿具体算子分析这些因素，并给出对应测量方法。文中引用的 arch35 源码用于解释该分支的实现，910C 的实际路径仍需与目标环境对应。

<details markdown="1">
<summary>版本与源码入口</summary>

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

</details>

## 一、量化收益与额外开销怎样比较

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

~~~text
T_W8A8_linear
≈
T_dynamic_quant
+
T_quant_matmul
+
T_other_epilogue
~~~

 Linear 加速条件是：

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

端到端 TPOT 的变化，取决于这些成本在整轮执行中的占比。

### 复用上游已经量化的激活

SGLang 允许 Linear 直接消费上游产生的 INT8 数据与 Scale：

~~~python
if isinstance(x, tuple):
    original_dtype = torch.bfloat16
    quant_out, dynamic_scale = x
else:
    quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
~~~

也就是说，如果上游 fused/prolog kernel 已经产生：`(INT8 payload, scale)`

这个 Linear 不会再单独执行一次 <code>npu_dynamic_quant</code>。

对齐 Timeline 时，先确认输入形式：

> 当前这个 Linear 的输入是 BF16 Tensor，还是已经量化好的 tuple？

只有前者才存在独立 DynamicQuant Kernel 的成本。

## 二、DynamicQuant 小 M：arch35 源码对照

本节以 CANN **arch35** 分支演示按行分配计算核的实现；910C 对应的实际分支与 Tiling 尚待目标环境记录。下文公式均描述这个 arch35 对照分支。

对照文件是：

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

对于普通二维 Dense 输入：`X.shape = [M, K]`

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

这是该 arch35 Host Tiling 的行分配规则。

所以低并发 Decode 的问题可以更加准确地表述为：

> **Standalone per-token DynamicQuant 的 row-level parallelism 直接受 M 限制。M 很小时，即使芯片拥有更多 Vector Core，这个算子本身也没有足够多的 Token row 去铺满它们。**

### 一个重要例外：超长 row 可以沿尾轴进一步拆

源码没有简单停在 <code>coreNum=M</code>。

<code>CalculateTilingData()</code> 会计算一行在 UB 中需要的空间。如果一个 row 大到一个 UB 都放不下，而且满足：`rowNum <= vectorCoreNum / 2`

以及：`calcSize >= 4 × maxUseUbSize`

就可能进入：`CalculateTilingForPertenLargeMulticore`

把单行尾轴再拆给更多核。

普通 per-token DynamicQuant 按行并行；行足够长时，另有沿尾轴分给多核的路径。

DeepSeek-V4 的主要 W8A8 Dense K 值是 4096 或 1024，究竟在目标 CANN build 上走 full-load 还是特殊大-shape 路径，可以从目标 CANN build 的 Tiling Log 读取。

### 从分块规则到实际耗时还缺什么

这条算子路径的工作包括：

- 它是独立算子路径；
- 它要处理每一行并生成 per-token scale；
- Host Tiling 会根据 row 数分配 Vector Core；
- 最终 <code>SetBlockDim(coreNum)</code> 发布实际 block 数。

Tiling 描述怎样分配工作。实际耗时还受 CANN build、Stream、启动方式、K、UB template 与前后融合影响，因此下一步将这些分块信息与 Profiler 中的 Launch、Device Duration 对齐。

## 三、QuantMatmul 的小 M 分块与待测问题

接下来仍用 arch35 分支，观察 M 怎样进入 base block、SmallMN 与 StreamK 的分块计算。

### M 怎样参与 Tiling 计算

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

然后决定是否进入：`UpdateSmallMnStreamKBase()`

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

按 Shape 计算的 Tiling 也会在整数阈值附近发生切换。

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

将耗时变化对应到执行选择：

> 先观测 cliff，再用 tiling dump 解释。

而不是：

> 因为源码一定有 8 的 bucket，所以提前宣称这里必然 cliff。

### 对齐小 M 耗时与 Tiling 变化

固定一组真实 DeepSeek-V4 Linear，例如：

~~~text
wq_a:
K=4096
N=1024
~~~

只 sweep：`M = 1, 2, 4, 8, 16, 32, 64`

每个点记录：

~~~text
operator duration
tiling key
baseM / baseN / baseK
used cores
StreamK status
~~~

这样才能得到：

> **Ascend 910C + 当前 CANN build + 当前 DeepSeek-V4 shape 的小 M QuantMatmul 曲线。**

## 四、FRACTAL_NZ 对齐与实际矩阵形状

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

下面把布局规则代入 DeepSeek-V4-Flash 的具体 shape。

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

Dense W8A8 的 NZ 实验可以选取实际参与量化的投影层，例如：

| Layer | 逻辑 K | 逻辑 N |
| --- | ---: | ---: |
| <code>wq_a</code> | 4096 | 1024 |
| <code>wkv</code> | 4096 | 512 |
| <code>wq_b</code> | 1024 | 32768 / <code>attn_tp_size</code> |

以官方 TP16/DP16 DPA 配置为例：`attn_tp_size = 1`

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

这条路径中，FRACTAL_NZ 格式还参与了下游 QuantMatmul 的分发选择。因此，比较格式耗时时，需要一起确认实际调用的算子。

### 测量 NZ 与 ND 的耗时差异

这些 shape 对应的格式选择过程是：

~~~text
这些 DSV4 shape 能通过 NZ alignment
        ↓
SGLang 会尝试 format cast
        ↓
成功后进入 WeightNz ACLNN 路径
~~~

NZ 与 ND 的耗时差异随 M、K/N、CANN Tiling 和缓存访问变化。使用同一 checkpoint 分别加载两种格式，可以直接比较：

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

这个环境变量在 Weight Load 阶段生效。A/B 分别重启服务，确保两轮加载了对应格式。

## 五、Expert Token 数与 Grouped MatMul

DeepSeek-V4 MoE 的 Token 分给不同专家后，每位专家都有自己的权重和输入行数：

~~~text
Expert 0: M_0
Expert 1: M_1
...
Expert E: M_e
~~~

当前 SGLang Ascend Runner 把：`expert_tokens`

作为：

~~~python
group_list=expert_tokens
~~~

传给 <code>npu_grouped_matmul</code>，所以 actual expert token distribution 是 GMM 的真实输入。

### CANN 中的 128 阈值及其分支条件

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

源码还包含一个 A8W8 Fixed-Axis 优化，要求 expected token / expert 处于：`128 ~ 512`

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

op-plugin / CANN 未收到显式 tuning hint 时，对应：`tuningConfig_ = 0`

而不是 128、192 或其它预期每 Expert Token 数。

CANN 的 128 阈值属于 expected-token hint 驱动的优化机制。当前 SGLang 普通 GroupedMatmul wrapper 没有显式传入该 hint，分析执行分支时要从实际 `tuningConfig_` 与 group list 出发。

### Fixed-Axis 分支与 DeepSeek-V4 的矩阵形状

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

这组 K/N 与上面的 Fixed-Axis 条件不同，DeepSeek-V4 Expert 将按其实际形状进入相应分支。

### 用 Expert-local M_e 描述负载

实际 group list 直接给出各 Expert 的行数，进而决定：

~~~text
总计算量
每个 Expert 的有效 GEMM M
空 Expert 数
最重 Expert
执行不均衡
~~~

都会随 <code>{M_e}</code> 改变。

但是“延迟曲线在哪些整数点发生 cliff”仍然是实验问题。

Sweep 覆盖以下负载：

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

## 六、用四组实验测量实际成本

按前面的执行分支，可以将 M Sweep 分成四组实验，各自测量一项成本。

### 实验 A：Standalone DynamicQuant 的小 M 成本

固定：

~~~text
dtype = BF16 → INT8
K = 1024 / 4096
~~~

sweep：`M = 1, 2, 4, 8, 16, 32, 64`

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

sweep：`M = 1,2,4,8,16,32,64,128`

每个点记录：

~~~text
tiling key
baseM/baseN/baseK
mCnt/nCnt
StreamK
used cores
kernel duration
~~~

把耗时曲线与实际 dump 对齐，找出改变 Tiling 结构的 M。

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

这项 CI 给出 W8A8 路径的验收门槛；BF16 与 W8A8 的收益比较则使用同条件 A/B。

## 源码结论与待测问题

| 问题 | 源码支持的结论 |
| --- | --- |
| DynamicQuant 小 M | arch35 对照分支使用 <code>coreNum=min(vectorCoreNum,M)</code>；本文未确认该公式适用于 910C 的实际路径 |
| DynamicQuant μs | 用目标环境的 Profiler 测量 Launch 与 Device Duration |
| Quantmatmul M=1/2/4/8 | M 会进入 baseM / block / SmallMN / StreamK 计算，但没有发现固定 1/2/4/8 查表分支 |
| FRACTAL_NZ 对齐 | DSV4 主要 W8A8 Dense shape <code>wq_a/wkv/wq_b</code> 天然满足当前 INT8 NZ alignment |
| FRACTAL_NZ 收益百分比 | 同 checkpoint、同形状下比较 NZ 与 ND |
| Expert <code>M_e</code> | actual group_list 决定真实 Expert workload；CANN 还提供 expected-token tuning 机制 |
| 128 token threshold | CANN 源码中真实存在，但属于 tuning-config 驱动的优化条件 |
| SGLang 中的 128 阈值 | 当前 wrapper 未显式传 `tuning_config`，实际切换由对应分支决定 |
| DSV4 是否命中 Fixed-Axis 特殊优化 | 当前 W13/W2 K/N 不匹配该源码分支的特定 shape |

因此，“DeepSeek-V4 W8A8 为什么不一定更快”最准确的答案不是一句：`因为 INT8 有量化开销`

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

要形成 Ascend 910C 的实测结论，可以从前文四组实验开始：DynamicQuant M sweep、WeightNz M sweep、NZ/ND A/B、Expert <code>M_e</code> sweep，然后把 tiling dump 与 profiler 时间线一一对上。
