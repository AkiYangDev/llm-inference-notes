# DeepSeek-V4 W8A8 性能分析：小 M、NZ 布局与测量方法

`SMALL_TUNING_CONFIG_THRESHOLD = 128` 出现在 GroupedMatmul Tiling 中，固定 SGLang wrapper 却没有传 `tuning_config`，下游得到的 hint 是 0。专家实际收到多少行，不会自动填进这份 hint。只看一个 128 常量，还解释不了实际性能。

W8A8 的耗时还受激活量化、小 M、权重布局和专家分布影响。这里沿源码条件组织测量方案，不填入目标环境尚未测出的收益。arch35 只作实现对照，不代表 Ascend 910C 实际选中了相同路径。

<details markdown="1">
<summary>版本与源码入口</summary>

以下分析使用固定源码。

| 组件 | 版本 |
| --- | --- |
| SGLang | <code>b63f8416b3b73bafdec029005c5db36bad207b44</code> |
| msModelSlim | <code>85d6c6f0266fd1fd77aa19ac26086670ca92d1db</code> |
| Ascend op-plugin | <code>e89cc608309341298193f957fa87c964ee561781</code> |
| CANN ops-nn GitHub mirror | <code>0026707d7ecb24d258e030d831c8121ac2d055bd</code> |
| CANN ops-transformer GitHub mirror | <code>bbb4dafe296119266a0a64133f17bc42b6597ed3</code> |
| DeepSeek-V4-Flash-0731 config | Hugging Face main，核对日期 2026-09-21 |

CANN GitHub mirror 便于查代码，部署行为仍由目标环境安装的 CANN 版本决定。

</details>

## 一、量化收益与额外开销怎样比较

从普通 Dense Linear 的成本开始。

~~~text
Y = XW

X: [M, K]
W: [K, N]
Y: [M, N]
~~~

BF16 的主要成本近似如下。

~~~text
T_BF16_linear
≈
T_BF16_GEMM
~~~

固定 DeepSeek-V4-Flash ModelSlim 主线使用 <code>W8A8_DYNAMIC</code>。普通 Tensor 进入 <code>NPUW8A8Int8DynamicLinearMethod.apply()</code>，经历下面这些工作。

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

这条 Linear 的收益条件可以写成下面的比较。

~~~text
T_BF16_GEMM - T_INT8_GEMM
>
T_dynamic_quant + 新增的其它成本
~~~

放回整个 Decoder Layer，还包括这些部分。

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

这些是串行成本近似，多流计算、通信可能重叠，整层 elapsed 仍由依赖决定，各流 duration 不能直接相加。

QuantMatmul 快了多少，只能省去它原来占用的那段时间。端到端 TPOT 还取决于其他工作与它在整轮里的位置。

### 复用上游已经量化的激活

上游已有 INT8 激活和 Scale 时，Linear 还能复用这份结果。

~~~python
if isinstance(x, tuple):
    original_dtype = torch.bfloat16
    quant_out, dynamic_scale = x
else:
    quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
~~~

fused/prolog 已输出 `(INT8 payload, scale)` tuple，这里就不再单独调用 <code>npu_dynamic_quant</code>。

因此同样叫 W8A8，普通 BF16 Tensor 路径有这次独立量化，已量化 tuple 路径则没有，Timeline 中的开销要对应实际输入。

## 二、DynamicQuant 小 M：arch35 源码对照

CANN arch35 的按行分核规则，可以解释一种小 M 行为。下面的公式只属于这份对照实现，Ascend 910C 分支与实际 Tiling 另外记录。

对应文件如下。

~~~text
quant/dynamic_quant/
└── op_host/
    └── dynamic_quant_tiling_arch35.cpp
~~~

普通非 per-channel 分支进入下面的调用。

~~~text
CalculateCoreNum(context)
CalculateTilingData()
~~~

<code>CalculateCoreNum()</code> 用末维作 row width，其余维度乘积作 row 数。

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

二维 Dense 的 `X.shape = [M, K]`，对应下面的行数与行宽。

~~~text
rowNum = M
rowLen = K
~~~

初始分核关系如下。

~~~text
coreNum
=
min(Vector Core 数, M)
~~~

如果没有触发超长单行沿尾轴拆分，Vector Core 上限按下面的关系确定。

| M | DynamicQuant 最初可使用的 Vector Core 数上限 |
| ---: | ---: |
| 1 | 1 |
| 2 | 2 |
| 4 | 4 |
| 8 | 8 |

这个 standalone per-token DynamicQuant 分支按行并行。M 很小，即使硬件有更多 Vector Core，输入行也可能不够分。

### 一个重要例外：超长 row 可以沿尾轴进一步拆

单行特别宽时，还有另一条拆分路径。

<code>CalculateTilingData()</code> 估算单行 UB 占用，单行放不下、`rowNum <= vectorCoreNum / 2` 且满足下面的条件，才可能走这条例外。

`calcSize >= 4 × maxUseUbSize`

这时可能进入 `CalculateTilingForPertenLargeMulticore`，沿尾轴继续分给多核。

所以小 M 不保证最多只能用 M 个核，行宽与模板也参与选择。

DeepSeek-V4 主要 W8A8 Dense K 为 4096 或 1024，目标 build 是 full-load 还是特殊大形状路径，需要实际 Tiling Log。

### 从分块规则到实际耗时还缺什么

独立算子逐行处理、生成 per-token scale，Host 按行数分配 Vector Core，最后通过 <code>SetBlockDim(coreNum)</code> 发布 block 数。

分块解释工作怎样分，耗时还受 CANN build、Stream、启动方式、K、UB template 和前后融合影响。这些信息对应到 Profiler 的 Launch、Device Duration，才是性能证据。

## 三、QuantMatmul 的小 M 分块与待测问题

QuantMatmul 仍以 arch35 作对照，看 M 怎样进入 base block、SmallMN、StreamK 计算。

### M 怎样参与 Tiling 计算

<code>base_block_calculator.cpp</code> 的默认 base block 如下。

~~~cpp
baseM =
    CeilAlign(
        min(mSize, 256),
        baseMAlign
    );
~~~

StreamK 同样读取真实 <code>inputParams_.mSize</code>。

~~~text
baseM
mCnt
nCnt
preSplitKBlockCnt
~~~

然后再判断 `UpdateSmallMnStreamKBase()`。

baseM、M block 数、MN 并行度、StreamK 和最终 used cores，都受到 M 影响。

### 但没有找到“1/2/4/8 四档固定分支”

在这份 arch35 QuantBatchMatmulV3 Tiling 中，针对性检索没有找到 `M=1/2/4/8` 各自硬编码的分支，也没有找到相应 A8W8 WeightNz Decode bucket 表。

分块主要依赖下面这些输入。

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

它们按真实形状导出分块结果。

M=1/2/4/8 可以导出不同 Tiling，源码支持的是形状推导，尚没有证据将它解释为固定四档查表。

### 为什么现实中仍然可能看到 latency cliff

形状推导也可能在整数边界切换。

~~~text
CeilAlign(M, alignment)
CeilDiv(M, baseM)
used core count
StreamK enable / disable
tail block
L1 / L0 capacity
Graph bucket / padding
~~~

因此实测仍可能呈现下面的变化。

~~~text
M=7 → latency A
M=8 → latency B
~~~

出现 latency cliff，要把曲线对应到 Tiling dump 的实际切换。存在阶跃，不等于预先假定的固定 8 bucket 得到了证明。

### 对齐小 M 耗时与 Tiling 变化

例如选一组真正参与量化的 DeepSeek-V4 Linear。

~~~text
wq_a:
K=4096
N=1024
~~~

其他条件固定，只改变 `M = 1, 2, 4, 8, 16, 32, 64`。

每点保留以下记录。

~~~text
operator duration
tiling key
baseM / baseN / baseK
used cores
StreamK status
~~~

这些结果才能构成当前 CANN build、当前 DeepSeek-V4 形状在 Ascend 910C 上的小 M 曲线。

## 四、FRACTAL_NZ 对齐与实际矩阵形状

SGLang 尝试把 W8A8 Weight 转成下面的格式。

~~~python
weight =
    npu_format_cast(
        weight,
        ACL_FORMAT_FRACTAL_NZ
    )
~~~

INT8 的预检查条件如下。

~~~text
K % 16 == 0
N % 32 == 0
~~~

不满足就保留 ND，并提示可能的性能影响。

将这些条件代入模型维度。

### 官方模型的关键维度

DeepSeek-V4-Flash-0731 的公开配置如下。

~~~text
hidden_size            = 4096
q_lora_rank            = 1024
head_dim               = 512
num_attention_heads    = 64
o_lora_rank            = 1024
o_groups               = 8
moe_intermediate_size  = 2048
~~~

ModelSlim recipe 的 Attention include 是 <code>*attn*</code>，同时排除下面这些模块。

~~~text
*wo_a
*wo_b
*compressor.wgate
*compressor.wkv
*indexer.weights_proj
...
~~~

实际进入量化的投影，可以选来比较 NZ、ND。

| Layer | 逻辑 K | 逻辑 N |
| --- | ---: | ---: |
| <code>wq_a</code> | 4096 | 1024 |
| <code>wkv</code> | 4096 | 512 |
| <code>wq_b</code> | 1024 | 32768 / <code>attn_tp_size</code> |

官方 TP16/DP16 DPA 的 `attn_tp_size = 1`，对应下面的形状。

~~~text
wq_b:
K = 1024
N = 32768
~~~

这些主要 W8A8 Attention Dense 形状通过了当前 INT8 FRACTAL_NZ alignment 预检查，加载不会在这项条件上被挡回。

### FRACTAL_NZ 还会改变 op-plugin 的真实入口

op-plugin 也根据权重格式选择入口。

~~~text
Weight = FRACTAL_NZ
        ↓
aclnnQuantMatmulWeightNz

Weight = ND
        ↓
aclnnQuantMatmulV5
~~~

NZ、ND 因而不只存储不同，还会影响 QuantMatmul 分发，计时需要连实际算子一起保留。

### 测量 NZ 与 ND 的耗时差异

上述 DSV4 形状的选择过程如下。

~~~text
这些 DSV4 shape 能通过 NZ alignment
        ↓
SGLang 会尝试 format cast
        ↓
成功后进入 WeightNz ACLNN 路径
~~~

实际耗时还随 M、K/N、Tiling、缓存访问改变。同一个 Checkpoint 分别加载两种格式，便可形成下面的对照。

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

对照保持模型、M/N/K、Batch、并行配置、CANN 和 Graph 相同。

比较项如下。

~~~text
QuantMatmul duration
Dense layer duration
TTFT / TPOT
throughput
~~~

环境变量在权重加载时生效，这份 A/B 要分别启动服务，并确认各自实际加载的格式。

## 五、Expert Token 数与 Grouped MatMul

Token 路由到不同专家，各组 GEMM 的输入行数分别确定。

~~~text
Expert 0: M_0
Expert 1: M_1
...
Expert E: M_e
~~~

Ascend Runner 将 `expert_tokens` 通过下面的参数传入。

~~~python
group_list=expert_tokens
~~~

<code>npu_grouped_matmul</code> 因而直接使用本轮专家分布。

### CANN 中的 128 阈值及其分支条件

固定 GroupedMatmul Tiling 注释记载，单专家 Token 数低于 128 时 Cube 不易充分发挥，开启两个 Vector 核可能变差。实现以 `SMALL_TUNING_CONFIG_THRESHOLD = 128`、`has_sufficient_tuning = tuningConfig_ >= 128` 参与 AIV:AIC 比例选择，另有要求预期 Token/Expert 为 `128 ~ 512` 的 A8W8 Fixed-Axis 优化。

这个 128 带有上游实测背景，读取的却是预期每专家行数的 tuning hint。SGLang 当前传入了什么，决定能否用它解释曲线。

<a id="sglang-tuning_config"></a>

torch-npu 提供可选 `tuning_config=None`，固定 SGLang `GroupedMatmul.forward()` 没显式传它，下游 op-plugin / CANN 的 `tuningConfig_=0`。group list 里实际的 128、192 或其他行数，不会自动替代这项 hint。

实际 `group_list` 和预期 Token hint 是两个输入。是否触发优化，需要对应 `tuningConfig_` 与所选分支，常量本身没有给出当前 wrapper 的执行结果。

### Fixed-Axis 分支与 DeepSeek-V4 的矩阵形状

Fixed-Axis A8W8 还受下面的矩阵形状约束。

~~~text
(K,N)
=
(2048,7168)
or
(7168,4096)
~~~

本例 Flash Expert 的形状则如下。

~~~text
W13:
K = 4096
N = 4096

W2:
K = 2048
N = 4096
~~~

W13、W2 的 K/N 都不匹配该 Fixed-Axis 条件，后续仍按真实形状选择分支。

### 用 Expert-local M_e 描述负载

专家负载既包含总工作量，也包含每个 Expert 的有效 M、空 Expert 数、AIV/AIC 调度和不均衡程度。总路由行数相同，分布仍可能改变单专家矩阵形状，batch 总 Token 数不足以解释 GMM。

具体在哪个整数点出现延迟突变，还没有目标环境实测。

M_e 的采样可覆盖下面这些值。

~~~text
M_e =
1, 2, 4, 8, 16, 32, 64,
96, 127, 128, 129,
160, 192, 256
~~~

同时保留以下信息。

~~~text
GroupedMatmul duration
tiling key
AIV:AIC ratio
workspace
expert_tokens histogram
~~~

128 两侧值得密集采样，因为源码有相应调优条件。SGLang 这条调用路径是否出现 cliff，仍由实际 Tiling 与 Profiler 决定。

## 六、用四组实验测量实际成本

下面四组实验分别隔离一种成本，是测量方案，本文没有执行它们、填入收益。

### 实验 A：Standalone DynamicQuant 的小 M 成本

固定 BF16→INT8，取 `K=1024/4096`，改变 `M=1,2,4,8,16,32,64`，记录 DynamicQuant device duration、Host enqueue gap、blockDim / coreNum 与 tiling key。

检查目标分支是否采用普通二维 per-token 的 `coreNum ≈ min(vectorCoreNum, M)`，同时看超长 row 例外。上游 fused/prolog 已产出 `(quant, scale)` tuple 时，没有这次 standalone DynamicQuant，另行列出。

### 实验 B：WeightNz Quantmatmul 的 M sweep

选择实际量化的 DSV4 投影，`wq_a` 为 `K=4096,N=1024`，`wkv` 为 `K=4096,N=512`，`wq_b` 为 `K=1024,N=32768/attn_tp_size`，改变 `M=1,2,4,8,16,32,64,128`。

kernel duration 与 tiling key、baseM/baseN/baseK、mCnt/nCnt、StreamK、used cores 一起保留，观察哪些 M 改变了执行结构。

### 实验 C：FRACTAL_NZ A/B

A 用默认 FRACTAL_NZ，B 设 `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1`，分别启动、加载，确认 Weight 格式，保持相同 M/N/K、Checkpoint、并行配置、CANN 与 Graph。

确认实际走 WeightNz 还是 V5，再比较 operator duration、TTFT、TPOT、throughput。同条件的结果，才对应这个形状的 NZ 收益。

### 实验 D：Expert-local M_e sweep

固定 W13 `K=4096,N=4096`、W2 `K=2048,N=4096`，控制专家行数，密集采 128 两侧。

~~~text
M_e = 1,2,4,8,16,32,64,96,127,128,129,192,256
~~~

保留 GMM duration、tiling key、AIV:AIC ratio、expert token histogram、empty expert ratio。附近若出现突变，再对应实际参数与 Tiling 切换，不能由常量预定曲线形状。

### 最后才看端到端

单算子说明局部成本，回到服务还要分 Prefill、Decode、Target Verify，比较同条件 TTFT、TPOT、输出 Token 吞吐和请求吞吐。

固定 W8A8 NPU CI 定义下面的性能门槛，同时开启 ModelSlim W8A8、DPA、DeepEP INT8、DSpark。

~~~text
input_len        = 8000
output_len       = 1000
max_concurrency  = 160
TPOT gate        = 50 ms
output throughput gate = 3100 token/s
~~~

这是一整套配置的绝对性能验收，没有同条件 BF16 基线，相对量化收益仍需要 BF16 / W8A8 A/B。

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

下面将形状、布局与分支放回同一张图。DynamicQuant 行并行仍指前述 arch35 对照，Ascend 910C 对应路径另外确认。

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
