# DeepSeek-V4 W8A8 性能分析：小 M、NZ 布局与测量方法

W8A8 将部分矩阵乘改为低精度计算，也带来动态激活量化、Scale 和其他算子处理。是否更快，要在同一负载下看节省的 GEMM 时间能否覆盖新增工作。

Decode 的小 M、专家负载分散、NZ 布局及 Tiling 都会改变结果。下面沿这些执行分支设计测量，不给出目标环境尚未测得的收益。arch35 源码只作实现对照，不能直接当成 Ascend 910C 实际选择的路径。

<details markdown="1">
<summary>版本与源码入口</summary>

分析使用以下固定源码：

| 组件 | 版本 |
| --- | --- |
| SGLang | <code>b63f8416b3b73bafdec029005c5db36bad207b44</code> |
| msModelSlim | <code>85d6c6f0266fd1fd77aa19ac26086670ca92d1db</code> |
| Ascend op-plugin | <code>e89cc608309341298193f957fa87c964ee561781</code> |
| CANN ops-nn GitHub mirror | <code>0026707d7ecb24d258e030d831c8121ac2d055bd</code> |
| CANN ops-transformer GitHub mirror | <code>bbb4dafe296119266a0a64133f17bc42b6597ed3</code> |
| DeepSeek-V4-Flash-0731 config | Hugging Face main，核对日期 2026-09-21 |

CANN GitHub mirror 便于检索，部署行为仍需以目标环境安装的 CANN 版本核对。

</details>

## 一、量化收益与额外开销怎样比较

先从普通 Dense Linear 开始：

~~~text
Y = XW

X: [M, K]
W: [K, N]
Y: [M, N]
~~~

BF16 路径可粗略表示为：

~~~text
T_BF16_linear
≈
T_BF16_GEMM
~~~

固定 DeepSeek-V4-Flash ModelSlim 主路径使用 <code>W8A8_DYNAMIC</code>。普通 Tensor 进入 <code>NPUW8A8Int8DynamicLinearMethod.apply()</code> 后，按以下步骤执行：

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

这条 Linear 要加速，需要满足：

~~~text
T_BF16_GEMM - T_INT8_GEMM
>
T_dynamic_quant + 新增的其它成本
~~~

整个 Decoder Layer 还有更多组成部分：

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

这些公式用于分解工作，按串行成本近似。若通信与计算存在重叠，整层 elapsed 仍需沿依赖关系核算，不能把各流 duration 直接相加。

QuantMatmul 的单项提速，只能节省它在执行中占用的时间。端到端 TPOT 还取决于这些成本在整轮中的位置与占比。



### 复用上游已经量化的激活

若上游已经提供 INT8 激活与 Scale，Linear 可以直接复用：

~~~python
if isinstance(x, tuple):
    original_dtype = torch.bfloat16
    quant_out, dynamic_scale = x
else:
    quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
~~~

上游 fused/prolog 已输出 `(INT8 payload, scale)` tuple 时，这里不会再单独调用 <code>npu_dynamic_quant</code>。



对齐 Timeline 前应先查输入：普通 BF16 Tensor 有这次独立量化，已量化 tuple 则没有。





## 二、DynamicQuant 小 M：arch35 源码对照

先借 CANN arch35 的按行分核规则理解小 M。Ascend 910C 实际分支与 Tiling 仍待目标环境记录，本节公式只描述这个对照实现。

文件位置为：

~~~text
quant/dynamic_quant/
└── op_host/
    └── dynamic_quant_tiling_arch35.cpp
~~~

普通非 per-channel 路径调用：

~~~text
CalculateCoreNum(context)
CalculateTilingData()
~~~

<code>CalculateCoreNum()</code> 把末维作为 row width，其余维度乘积作为 row 数：

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

二维 Dense 输入为 `X.shape = [M, K]`，因此：



~~~text
rowNum = M
rowLen = K
~~~

初始分核可写为：

~~~text
coreNum
=
min(Vector Core 数, M)
~~~

未触发超长单行沿尾轴拆分时，可用 Vector Core 上限如下：

| M | DynamicQuant 最初可使用的 Vector Core 数上限 |
| ---: | ---: |
| 1 | 1 |
| 2 | 2 |
| 4 | 4 |
| 8 | 8 |

这说明 arch35 对照分支中的 standalone per-token DynamicQuant 按行并行，M 很小时即使有更多 Vector Core，也缺少足够输入行分给它们。





### 一个重要例外：超长 row 可以沿尾轴进一步拆

不过，这个规则还有超长 row 的例外。

<code>CalculateTilingData()</code> 会估算单行 UB 占用。单行放不下且满足 `rowNum <= vectorCoreNum / 2`，以及下一条件时：

`calcSize >= 4 × maxUseUbSize`

可能进入 `CalculateTilingForPertenLargeMulticore`，沿尾轴继续拆给多核。



因此，小 M 不必然等于只能使用 M 个核，还需核对行宽和所选模板。

DeepSeek-V4 主要 W8A8 Dense K 为 4096 或 1024。目标 build 采用 full-load 还是特殊大形状路径，需要从实际 Tiling Log 查明。

### 从分块规则到实际耗时还缺什么

这条独立算子逐行处理并生成 per-token scale，Host 根据行数安排 Vector Core，最后用 <code>SetBlockDim(coreNum)</code> 发布 block 数。



Tiling 说明工作如何分配，耗时还取决于 CANN build、Stream、启动方式、K、UB template 及前后融合。需要将这些信息和 Profiler 的 Launch、Device Duration 对齐，才能形成性能结论。

## 三、QuantMatmul 的小 M 分块与待测问题

接着仍以 arch35 为对照，观察 M 进入 base block、SmallMN 和 StreamK 计算的方式。

### M 怎样参与 Tiling 计算

<code>base_block_calculator.cpp</code> 的默认 base block 使用：

~~~cpp
baseM =
    CeilAlign(
        min(mSize, 256),
        baseMAlign
    );
~~~

StreamK 也由真实 <code>inputParams_.mSize</code> 计算：

~~~text
baseM
mCnt
nCnt
preSplitKBlockCnt
~~~

再判断是否进入 `UpdateSmallMnStreamKBase()`。

因此 M 会参与 baseM、M block 数、MN 并行度、StreamK 以及最终 used cores 的选择。







### 但没有找到“1/2/4/8 四档固定分支”

针对固定 arch35 QuantBatchMatmulV3 Tiling 检索，未找到按 `M=1/2/4/8` 逐项硬编码的分支，也未找到相应 A8W8 WeightNz Decode bucket 表。

其分块大量依赖：

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

通过真实形状计算分块结果。

M=1/2/4/8 当然可以得到不同 Tiling，但现有源码证据支持的是按形状推导，不能将差异解释成一张固定四档查表。



### 为什么现实中仍然可能看到 latency cliff

形状推导仍会在整数边界发生离散切换：



~~~text
CeilAlign(M, alignment)
CeilDiv(M, baseM)
used core count
StreamK enable / disable
tail block
L1 / L0 capacity
Graph bucket / padding
~~~

所以实测也可能出现：

~~~text
M=7 → latency A
M=8 → latency B
~~~

发现 latency cliff 后，需要用 Tiling dump 对照执行选择；不能先假定存在 8 的固定 bucket，再推断必然突变。







### 对齐小 M 耗时与 Tiling 变化

例如先固定一组真实 DeepSeek-V4 Linear：

~~~text
wq_a:
K=4096
N=1024
~~~

只改变 `M = 1, 2, 4, 8, 16, 32, 64`。

逐点记录：

~~~text
operator duration
tiling key
baseM / baseN / baseK
used cores
StreamK status
~~~

得到这些目标环境证据后，才能画出 Ascend 910C、当前 CANN build 和该 DeepSeek-V4 形状对应的小 M 曲线。



## 四、FRACTAL_NZ 对齐与实际矩阵形状

固定 SGLang 尝试将 W8A8 Weight 转为：

~~~python
weight =
    npu_format_cast(
        weight,
        ACL_FORMAT_FRACTAL_NZ
    )
~~~

INT8 预检查要求：

~~~text
K % 16 == 0
N % 32 == 0
~~~

不满足则保留 ND，并提示可能影响性能。

再把规则代入具体模型维度。

### 官方模型的关键维度

DeepSeek-V4-Flash-0731 的公开配置为：

~~~text
hidden_size            = 4096
q_lora_rank            = 1024
head_dim               = 512
num_attention_heads    = 64
o_lora_rank            = 1024
o_groups               = 8
moe_intermediate_size  = 2048
~~~

ModelSlim recipe 在 Attention include <code>*attn*</code>，同时排除：

~~~text
*wo_a
*wo_b
*compressor.wgate
*compressor.wkv
*indexer.weights_proj
...
~~~

因此可选实际量化的投影做 NZ 对照：

| Layer | 逻辑 K | 逻辑 N |
| --- | ---: | ---: |
| <code>wq_a</code> | 4096 | 1024 |
| <code>wkv</code> | 4096 | 512 |
| <code>wq_b</code> | 1024 | 32768 / <code>attn_tp_size</code> |

官方 TP16/DP16 DPA 的 `attn_tp_size = 1`，此时：

~~~text
wq_b:
K = 1024
N = 32768
~~~

这些主要 W8A8 Attention Dense 形状通过当前 INT8 FRACTAL_NZ alignment 预检查，至少不会在这一步受阻。





### FRACTAL_NZ 还会改变 op-plugin 的真实入口

op-plugin 还按权重格式选择入口：

~~~text
Weight = FRACTAL_NZ
        ↓
aclnnQuantMatmulWeightNz

Weight = ND
        ↓
aclnnQuantMatmulV5
~~~

格式参与下游 QuantMatmul 分发，所以比较 NZ、ND 耗时时，也需记录实际调用的算子。

### 测量 NZ 与 ND 的耗时差异

对于上面这些 DSV4 形状，选择过程为：

~~~text
这些 DSV4 shape 能通过 NZ alignment
        ↓
SGLang 会尝试 format cast
        ↓
成功后进入 WeightNz ACLNN 路径
~~~

NZ、ND 的实际耗时还受 M、K/N、Tiling、缓存访问影响。可用同一 checkpoint 分别加载两种格式：

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

对齐模型、M/N/K、Batch、并行配置、CANN 与 Graph。

同时比较：

~~~text
QuantMatmul duration
Dense layer duration
TTFT / TPOT
throughput
~~~

环境变量在权重加载阶段生效，因此这组 A/B 需要分别启动服务，确认各自加载了目标格式。

## 五、Expert Token 数与 Grouped MatMul

Token 路由到专家后，每组 GEMM 具有自己的输入行数：

~~~text
Expert 0: M_0
Expert 1: M_1
...
Expert E: M_e
~~~

Ascend Runner 将 `expert_tokens` 作为下列参数：



~~~python
group_list=expert_tokens
~~~

交给 <code>npu_grouped_matmul</code>，实际专家分布直接参与 GMM。

### CANN 中的 128 阈值及其分支条件

固定 GroupedMatmul Tiling 的注释记载：单专家 Token 数低于 128 时，Cube 算力不能完全发挥，开启两个 Vector 核可能劣化。代码使用 `SMALL_TUNING_CONFIG_THRESHOLD = 128`，并通过 `has_sufficient_tuning = tuningConfig_ >= 128` 参与 AIV:AIC 比例选择。另有 A8W8 Fixed-Axis 优化，要求预期 Token/Expert 在 `128 ~ 512`。

这里的 128 有上游实测背景，但它对应预期每专家行数的 tuning hint。能否用来解释 SGLang 的曲线，还需先看调用者传了什么。

<a id="sglang-tuning_config"></a>

torch-npu 接口提供可选 `tuning_config=None`，固定 SGLang 的 `GroupedMatmul.forward()` 没有显式传入它。op-plugin / CANN 缺少这份 hint 时对应 `tuningConfig_=0`，不会自动替换成实际 group list 中的 128、192 或其他行数。

因此，实际 `group_list` 与预期 Token hint 是两个输入。分析当前 wrapper 是否触发相关优化，需要核对 `tuningConfig_` 及所选分支，仅凭一个 128 常量还不能判断。

### Fixed-Axis 分支与 DeepSeek-V4 的矩阵形状

Fixed-Axis A8W8 还要求特定矩阵形状：

~~~text
(K,N)
=
(2048,7168)
or
(7168,4096)
~~~

本例 DeepSeek-V4-Flash Expert 则为：

~~~text
W13:
K = 4096
N = 4096

W2:
K = 2048
N = 4096
~~~

W13/W2 不匹配该 Fixed-Axis K/N，需按真实形状判断后续分支。

### 用 Expert-local M_e 描述负载

描述专家负载时，同时保留总计算量、各 Expert 的有效 GEMM M、空 Expert 数、AIV/AIC 调度和负载不均。即使总路由行数相同，分布也可能改变单专家形状与执行安排；不能只拿 batch 总 Token 数解释 GMM。

但延迟在哪个整数点出现突变，还需实测。

可覆盖下面的 M_e：

~~~text
M_e =
1, 2, 4, 8, 16, 32, 64,
96, 127, 128, 129,
160, 192, 256
~~~

并记录：

~~~text
GroupedMatmul duration
tiling key
AIV:AIC ratio
workspace
expert_tokens histogram
~~~

128 值得重点采样，因为源码有相应调优条件。当前 SGLang 路径会不会在此出现 latency cliff，仍要结合 Profiler 和实际 Tiling 判断。

## 六、用四组实验测量实际成本

将前面的待测问题分成四组，每组先隔离一种成本。这些是测量方案，本篇没有用尚未执行的对照填入收益。

### 实验 A：Standalone DynamicQuant 的小 M 成本

固定 BF16→INT8，分别取 `K=1024/4096`，改变 `M=1,2,4,8,16,32,64`。每点记录 DynamicQuant device duration、Host enqueue gap、blockDim / coreNum 和 tiling key。

对照目标分支是否采用普通二维 per-token 路径的 `coreNum ≈ min(vectorCoreNum, M)`，同时检查超长 row 例外。若上游 fused/prolog 已生成 `(quant, scale)` tuple，就没有这里的 standalone DynamicQuant，应单独列出。

### 实验 B：WeightNz Quantmatmul 的 M sweep

选实际参与量化的 DSV4 投影：`wq_a` 为 `K=4096,N=1024`，`wkv` 为 `K=4096,N=512`，`wq_b` 为 `K=1024,N=32768/attn_tp_size`。改变 `M=1,2,4,8,16,32,64,128`。

每点将 kernel duration 与 tiling key、baseM/baseN/baseK、mCnt/nCnt、StreamK、used cores 放在一起，观察哪些 M 真正改变执行结构。

### 实验 C：FRACTAL_NZ A/B

A 使用默认 FRACTAL_NZ，B 设置 `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1`。两组需分别启动、加载并确认权重格式；保持相同 M/N/K、checkpoint、并行配置、CANN 与 Graph。

对照实际进入的是 WeightNz 还是 V5，再比较 operator duration、TTFT、TPOT 和 throughput。只有拿到这些同条件结果，才能报告该形状的 NZ 收益。

### 实验 D：Expert-local M_e sweep

固定 W13 为 `K=4096,N=4096`，W2 为 `K=2048,N=4096`。控制专家行数，重点采样 128 两侧：

~~~text
M_e = 1,2,4,8,16,32,64,96,127,128,129,192,256
~~~

保留 GMM duration、tiling key、AIV:AIC ratio、expert token histogram 和 empty expert ratio。如果附近确有延迟突变，再检查实际参数与 Tiling 的切换位置；常量本身不能证明曲线形状。

### 最后才看端到端

单算子结果说明局部成本，最终仍需区分 Prefill、Decode、Target Verify，比较同条件的 TTFT、TPOT、输出 Token 吞吐和请求吞吐。

固定 SGLang 的 DeepSeek-V4-Flash W8A8 NPU CI 定义了下面这组门槛，并同时启用 ModelSlim W8A8、DPA、DeepEP INT8 和 DSpark：

~~~text
input_len        = 8000
output_len       = 1000
max_concurrency  = 160
TPOT gate        = 50 ms
output throughput gate = 3100 token/s
~~~

这是整套配置的绝对性能验收，没有同条件 BF16 对照。要判断量化带来多少相对收益，仍需单独做 BF16 / W8A8 A/B。

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

结合这些源码，W8A8 需要沿形状、布局和实际分支逐项分析。下图中的 DynamicQuant 行并行说明仍限于前文 arch35 对照，Ascend 910C 的对应路径需要另核。



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

dtype 提供了低精度计算的可能，形状、布局、Tiling 和专家负载则影响它能否在当前请求上发挥。可回到四组测量方案，核对局部收益最终是否缩短了整轮耗时。



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

形成 Ascend 910C 的实测结论时，需将 DynamicQuant M sweep、WeightNz M sweep、NZ/ND A/B、Expert <code>M_e</code> sweep 的结果，与对应 Tiling dump、Profiler 时间线一起留存。
