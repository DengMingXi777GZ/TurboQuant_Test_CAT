# TurboQuant HTTP API 性能测试工具

通过 vLLM HTTP 服务进行严谨的性能测试，支持真实 TPOT、显存监控和日志记录。

## 📁 文件结构

```
test_framework/
├── test_tq_api.py          # HTTP API 测试脚本（推荐）
├── test_tq.py              # 子进程测试脚本（旧版）
├── requirements.txt        # Python 依赖
├── README.md               # 本文件
└── .gitignore
```

---

## 🚀 快速开始

### 方式1：HTTP API 测试（推荐）

适用于已启动 vLLM 服务的场景，可获取更精确的 TPOT 和显存指标。

#### 1. 环境准备

```bash
# 进入 test_framework 目录
cd test_framework

# 安装依赖
pip install -r requirements.txt
```

#### 2. 启动 vLLM 服务（Baseline）

```bash
# 使用提供的启动脚本（TurboQuant 关闭）
bash ../start_vllm_qwen35_2b_tq.sh

# 或使用环境变量方式
TQ_ENABLED=0 bash ../start_vllm_qwen35_2b_tq.sh
```

#### 3. 运行 Baseline 测试

```bash
python test_tq_api.py --url http://localhost:8000 --config baseline --test all
```

#### 4. 停止服务，启动 TQ 版本

```bash
# 停止当前服务 (Ctrl+C)，然后启动 TQ 版本
TQ_ENABLED=1 bash ../start_vllm_qwen35_2b_tq.sh
```

#### 5. 运行 TQ 测试

```bash
python test_tq_api.py --url http://localhost:8000 --config turboquant --test all
```

---

## 📊 测试指标说明

| 指标 | 说明 | 测量方式 |
|------|------|---------|
| **TTFT** | Time To First Token | 流式响应第一个 token 的时间 |
| **TPOT** | Time Per Output Token | (总时间 - TTFT) / (token数 - 1) |
| **Throughput** | 吞吐量 | tokens / 总时间(秒) |
| **显存** | GPU Memory Used | vLLM /metrics 端点获取 |
| **QA Accuracy** | 问答准确率 | 简单 QA 测试 |

---

## 🎯 严谨性保证

### 多次采样取平均

```bash
# 默认采样 5 次，可自定义
python test_tq_api.py --runs 10 --config baseline --test performance
```

### 真实 TPOT 计算

使用 **流式输出 (streaming)** 精确测量：
- 记录每个 token 的到达时间戳
- 计算相邻 token 的时间差
- 排除第一个 token (TTFT)，取平均

### 显存监控

通过 vLLM 的 `/metrics` 端点获取 Prometheus 格式的显存数据：
- `nvidia_gpu_memory_used_bytes` - 已使用显存
- `nvidia_gpu_memory_total_bytes` - 总显存

---

## 📋 输出文件

测试完成后生成：

```
logs/
├── test_tq_20250115_103045.log      # 详细日志（含每次请求）

results/
├── test_results_20250115_103045.json # 原始数据（JSON）
└── test_report_20250115_103045.md    # 可读报告（Markdown）
```

**日志内容包括：**
- 每次请求的 TTFT、TPOT、Throughput
- 显存采样数据
- QA 测试详情
- 错误信息

---

## 🔧 高级用法

### 只测性能（多次采样）

```bash
python test_tq_api.py \
    --url http://localhost:8000 \
    --config baseline \
    --test performance \
    --runs 10 \
    --max-tokens 512
```

### 只测显存

```bash
python test_tq_api.py --config baseline --test memory
```

### 只测质量

```bash
python test_tq_api.py --config baseline --test quality
```

---

## ⚙️ 修改启动脚本

### 修改模型路径

编辑 `start_vllm_qwen35_2b_tq.sh`：

```bash
MODEL_PATH="/your/model/path"  # 修改这里
```

### 修改 TQ 参数

```bash
TQ_KEY_BITS=4        # Key 精度: 3 或 4
TQ_VALUE_BITS=4      # Value 精度: 2 或 4
TQ_BUFFER_SIZE=256   # Buffer 大小: 64/128/256
```

### 修改测试模型名称

编辑 `test_tq_api.py`：

```python
MODEL_NAME = "YourModelName"  # 第48行
```

---

## 📈 对比测试完整流程

```bash
#!/bin/bash
# compare_tq.sh - 完整对比测试脚本

API_URL="http://localhost:8000"
OUTPUT_DIR="./comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "========== Baseline Test =========="
bash ../start_vllm_qwen35_2b_tq.sh &
SERVER_PID=$!
sleep 10  # 等待服务启动

python test_tq_api.py \
    --url $API_URL \
    --config baseline \
    --test all \
    --runs 5 \
    --output-dir $OUTPUT_DIR/baseline

kill $SERVER_PID
sleep 5

echo "========== TurboQuant Test =========="
TQ_ENABLED=1 bash ../start_vllm_qwen35_2b_tq.sh &
SERVER_PID=$!
sleep 10

python test_tq_api.py \
    --url $API_URL \
    --config turboquant \
    --test all \
    --runs 5 \
    --output-dir $OUTPUT_DIR/turboquant

kill $SERVER_PID

echo "========== Results =========="
echo "Baseline: $OUTPUT_DIR/baseline"
echo "TurboQuant: $OUTPUT_DIR/turboquant"
```

---

## 🐛 故障排查

### 服务未就绪

```
❌ 服务未就绪，请检查 vLLM 是否已启动
```

**解决：**
```bash
# 检查服务是否运行
curl http://localhost:8000/health

# 检查端口
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows
```

### 无法获取显存数据

```
Failed to get metrics: 404
```

**解决：**
- vLLM 需要开启 metrics 端点（默认开启）
- 检查是否可以访问 `http://localhost:8000/metrics`

### 测试超时

```
Streaming generation error: ReadTimeout
```

**解决：**
- 减少 `--max-tokens`
- 增加代码中的 `timeout` 参数

---

## 📦 依赖

```
requests>=2.28.0
```

**注意：** 由于通过 HTTP API 测试，无需安装 `vllm` 和 `torch`（只需在服务端安装）。

---

## 🔗 相关文件

- `start_vllm_qwen35_2b_tq.sh` - 启动脚本（支持 TQ 开关）
- `testvllm.py` - 简单连通性测试
- `test_tq.py` - 子进程测试（旧版，自行启动模型）

---

## 📝 注意事项

1. **服务必须已启动**：`test_tq_api.py` 不会启动 vLLM，需要先用 `start_vllm_qwen35_2b_tq.sh` 启动
2. **端口冲突**：确保 8000 端口未被占用
3. **显存监控**：需要 vLLM 开启 metrics 端点（默认开启）
4. **多次采样**：建议 `--runs >= 5` 以获得稳定结果

---

**有问题？** 查看日志文件 `logs/test_tq_*.log` 获取详细信息。
