# 图捕获与 Tensor 存储小例子

配套文章：[图捕获以后，为什么还能改权重，却不能随便换张量？](../../docs/performance/graph-capture-tensor-storage.md)。

脚本分别检查原地 `copy_`、Python 重新绑定、`set_` 换存储、输出复用和 Python 控制流。
每项从独立状态开始。`set_` 一项显式保留旧存储，避免将地址绑定与失效存储混在一起。

在有可用设备的环境里，进入本目录后选择相应命令：

```bash
python graph_storage_demo.py --backend npu --device-index 0
```

或：

```bash
python graph_storage_demo.py --backend cuda --device-index 0
```

NPU 路径需要匹配设备环境的 PyTorch 和 torch_npu；CUDA 路径需要支持 CUDA 的 PyTorch。
脚本只创建少量单元素张量，不加载模型，也不创建通信组。成功时输出当前版本、设备名与各项实际结果的 JSON。
断言中的数字是文章解释的预期值，本次写作未执行设备示例；不能把这份脚本当成已经采集的真机日志。

文章核对的源码版本与接口依据见[配套说明](../../assets/graph-capture-tensor-storage/evidence.json)。
