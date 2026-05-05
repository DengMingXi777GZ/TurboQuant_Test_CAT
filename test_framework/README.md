# TurboQuant 效果验证工具

简洁的测试工具，用于对比 TurboQuant 开启前后的模型表现差异。

## 📁 文件结构

```
test_framework/
├── test_tq.py          # 主测试脚本（单文件，600行）
├── requirements.txt    # Python 依赖列表
└── README.md           # 本文件
```

---

## 🚀 快速开始（3步）

### 第1步：安装依赖

```bash
# 进入 test_framework 目录
cd test_framework

# 安装测试框架依赖
pip install -r requirements.txt

# 安装 turboquant 本体（在项目根目录执行）
cd ..
pip install -e .
```

**依赖说明：**
- `torch>=2.1.0` - PyTorch 深度学习框架
- `vllm>=0.16.0` - 大模型推理引擎（TQ 集成依赖）

**可选依赖：**
- `datasets` - 如需使用 HuggingFace 数据集
- `sentence-transformers` - 如需语义相似度评估

---

### 第2步：配置模型

**修改 `test_tq.py` 中的模型配置：**

打开 `test_tq.py`，找到 `MODELS` 字典（约第38行）：

```python
MODELS = {
    "qwen3.5-4b": ModelConfig(
        name="Qwen3.5-4B",
        path="Qwen/Qwen3.5-4B-Instruct",  # ← 修改这里
        max_length=131072
    ),
}
```

**根据你的环境修改：**

| 场景 | 修改方式 |
|------|---------|
| **使用 HuggingFace 模型** | `path="模型ID"` 如 `"Qwen/Qwen3.5-4B-Instruct"` |
| **使用本地模型** | `path="/绝对路径/到/模型目录"` |
| **使用 ModelScope** | 先设置环境变量 `export VLLM_USE_MODELSCOPE=True` |

**添加新模型示例：**

```python
"my-model": ModelConfig(
    name="My Model",
    path="/mnt/models/Qwen3.5-4B",  # 本地路径示例
    max_length=32768  # 根据模型支持调整
),
```

---

### 第3步：运行测试

```bash
# 回到 test_framework 目录
cd test_framework

# 完整测试（性能 + 显存 + 质量）
python test_tq.py --model qwen3.5-4b --test all

# 只测性能（指定上下文长度）
python test_tq.py --model qwen3.5-4b --test performance --length 32768

# 只测显存（4K → 128K 自动扫描）
python test_tq.py --model qwen3.5-4b --test memory

# 只测质量（简单 QA）
python test_tq.py --model qwen3.5-4b --test quality
```

---

## 🔧 高级配置

### 1. 修改 TQ 压缩参数

打开 `test_tq.py`，找到 `_create_tq_script` 方法（约第193行）：

```python
install_hooks(
    worker.model_runner,
    key_bits=3,        # Key 量化精度: 3 或 4
    value_bits=2,      # Value 量化精度: 2 或 4
    buffer_size=128,   # 保留 token 数: 64/128/256
    mode=MODE_HYBRID,  # 模式: MODE_HYBRID（推荐）
)
```

**配置建议：**

| 配置 | key_bits | value_bits | buffer_size | 适用场景 |
|------|----------|------------|-------------|---------|
| 速度优先 | 3 | 2 | 64 | 最大压缩，最快 |
| 平衡（默认）| 3 | 2 | 128 | 推荐配置 |
| 质量优先 | 4 | 4 | 256 | 近无损质量 |

### 2. 多 GPU 测试

如需多卡并行（Tensor Parallelism）：

```python
# 在 _create_baseline_script 和 _create_tq_script 中修改:
llm = LLM(
    model="...",
    tensor_parallel_size=2,  # ← 改为你的 GPU 数
    # ...
)
```

运行前设置可见 GPU：
```bash
export CUDA_VISIBLE_DEVICES=0,1  # Linux/Mac
set CUDA_VISIBLE_DEVICES=0,1     # Windows
```

### 3. 显存不足时的调整

如果显存不足（OOM）：

**方法1：降低上下文长度**
```bash
python test_tq.py --model qwen3.5-4b --test performance --length 8192
```

**方法2：降低 gpu_memory_utilization**
```python
# 在测试脚本中找到 LLM() 初始化，修改为:
llm = LLM(
    # ...
    gpu_memory_utilization=0.85,  # 从 0.90 降低
    # ...
)
```

**方法3：使用更小模型**
```bash
python test_tq.py --model qwen3.5-1b --test all  # 使用 1B 模型
```

---

## 📊 输出结果

测试完成后生成两个文件：

### 1. Markdown 报告 (`tq_report_YYYYMMDD_HHMMSS.md`)

```markdown
# TurboQuant 效果验证报告

**模型**: Qwen3.5-4B
**时间**: 2025-01-15 10:30:00

## 性能对比

| 指标 | Baseline | TurboQuant | 变化 |
|------|----------|------------|------|
| TTFT | 523ms | 489ms | -6.5% |
| Throughput | 125 tok/s | 129 tok/s | +2.7% |
| Memory | 32456MB | 22341MB | -10115MB |

## 结论
- TurboQuant 减少显存占用约 10GB
- 性能略有提升（2.7%）
```

### 2. JSON 原始数据 (`tq_results_YYYYMMDD_HHMMSS.json`)

包含详细的原始指标，可用于进一步分析或导入其他工具。

---

## 🐛 故障排查

### 问题1: `ModuleNotFoundError: No module named 'turboquant'`

**解决：** 在项目根目录运行 `pip install -e .`

```bash
cd /path/to/turboquant
pip install -e .
```

### 问题2: `ModuleNotFoundError: No module named 'vllm'`

**解决：** 安装 vllm

```bash
pip install vllm>=0.16.0
```

### 问题3: 模型下载失败 / 连接 HuggingFace 超时

**解决1：** 使用国内镜像
```bash
# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com

# Windows
set HF_ENDPOINT=https://hf-mirror.com
```

**解决2：** 使用本地模型路径
```python
# 修改 test_tq.py
"qwen3.5-4b": ModelConfig(
    name="Qwen3.5-4B",
    path="/本地/绝对/路径/Qwen3.5-4B-Instruct",  # ← 改这里
    max_length=131072
),
```

### 问题4: CUDA Out of Memory

**解决：** 参考上文"显存不足时的调整"

### 问题5: 测试超时

**解决：** 修改脚本中的 timeout 参数
```python
# 在 _run_subprocess 方法中
result = subprocess.run(
    # ...
    timeout=1200,  # 从 600 改为 1200 秒
)
```

---

## 📦 迁移到新电脑的完整步骤

假设你要把代码从电脑 A 迁移到电脑 B：

### 在电脑 B 上执行：

```bash
# 1. 克隆/复制代码
# 方式1: 如果从 GitHub
git clone https://github.com/yourusername/turboquant.git
cd turboquant

# 方式2: 如果直接复制文件夹
cd /path/to/turboquant

# 2. 安装 turboquant 本体
cd turboquant  # 进入包含 setup.py 的目录
pip install -e .

# 3. 进入测试框架目录
cd test_framework

# 4. 安装测试依赖
pip install -r requirements.txt

# 5. 修改模型配置（关键步骤！）
# 用编辑器打开 test_tq.py，修改 MODELS 字典中的 path
# - 如果使用 HuggingFace: path="模型ID"
# - 如果使用本地模型: path="/本地/路径"

# 6. 运行测试
python test_tq.py --model qwen3.5-4b --test all
```

---

## 💡 使用外部数据集（可选）

如需更全面的质量测试，可以使用 HuggingFace 数据集：

```bash
# 1. 安装 datasets
pip install datasets

# 2. 在 test_tq.py 中修改 test_quality 方法
# （参考 README 中的"高级：使用外部数据集"部分）
```

---

## 📝 .gitignore 建议

上传 GitHub 前，在项目根目录创建 `.gitignore`：

```gitignore
# 测试生成的文件
*.md
*.json
!requirements.txt
!README.md

# Python 缓存
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# 模型文件（如果放在项目目录）
*.bin
*.safetensors
*.model

# IDE
.vscode/
.idea/
*.swp
*.swo
```

---

## ❓ 常见问题

**Q: 支持哪些模型？**  
A: 理论上支持所有 vLLM 兼容的模型。已测试：Qwen2.5、Qwen3.5、Llama 系列。

**Q: 最低显存要求？**  
A: 取决于模型大小。例如 Qwen3.5-4B 需要约 10GB，Qwen3.5-7B 需要约 16GB。

**Q: 测试需要多长时间？**  
A: 完整测试（`--test all`）约 5-15 分钟，取决于模型和上下文长度。

**Q: 可以测试自己的模型吗？**  
A: 可以！在 `MODELS` 中添加你的模型配置即可。

---

**有问题？** 查看 `test_tq.py` 源码注释或参考项目根目录的 `README.md`
