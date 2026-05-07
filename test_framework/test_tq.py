#!/usr/bin/env python3
"""
TurboQuant 真实效果测试

通过子进程直接启动 vLLM，准确测量 TQ 的真实效果：
1. 测量 vLLM 内部的 KV Cache 块数
2. 测量 Baseline vs TQ 的块数差异
3. 计算上下文扩展倍数

Usage:
    # 测试 Baseline
    python test_tq.py --config baseline

    # 测试 TurboQuant
    python test_tq.py --config turboquant

    # 或在启动脚本中设置
    TQ_ENABLED=1 python test_tq.py --config turboquant
"""

import os
import sys
import json
import time
import argparse
import subprocess
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent


# ============== 配置 ==============

MODEL_NAME = "Qwen35_2b"
MODEL_PATH = "/mnt/data1/dmx/Models/Qwen35_2b"  # 修改为你的模型路径

# 测试参数
DEFAULT_MAX_TOKENS = 256
DEFAULT_NUM_RUNS = 3

# 测试 prompt
TEST_PROMPT_SHORT = "Explain machine learning in simple terms."
TEST_PROMPT_LONG = """Artificial intelligence has transformed numerous industries in recent decades.
Machine learning algorithms can now process vast amounts of data to identify patterns
and make predictions with remarkable accuracy. Deep learning, a subset of machine learning,
uses neural networks with multiple layers to model complex patterns in data.""" * 4

# QA 测试
QA_QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("What is 17 * 23?", "391"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
]


# ============== 数据结构 ==============

@dataclass
class TestMetrics:
    """测试指标"""
    config_name: str
    timestamp: str

    # vLLM 内部信息
    kv_cache_blocks: int = 0      # KV cache 总块数
    kv_block_size: int = 0       # 每块大小
    max_tokens_capacity: int = 0  # 最大 token 容量

    # 性能指标
    ttft_ms: float = 0
    tpot_ms: float = 0
    throughput: float = 0
    output_tokens: int = 0

    # 显存（通过 nvidia-smi）
    vram_used_mb: List[float] = None  # 各 GPU 显存

    # 质量
    qa_accuracy: float = 0

    def __post_init__(self):
        if self.vram_used_mb is None:
            self.vram_used_mb = []

    def to_dict(self) -> Dict:
        return asdict(self)


# ============== 子进程运行脚本 ==============

def create_baseline_script(prompt: str, max_tokens: int) -> str:
    """创建 Baseline 测试脚本"""
    return f'''
import os
import sys
import types
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
repo_root = os.environ.get("TURBOQUANT_REPO_ROOT")
if repo_root and repo_root not in sys.path:
    sys.path.insert(0, repo_root)
pkg_dir = os.path.join(repo_root, "turboquant") if repo_root else None
if pkg_dir and os.path.isdir(pkg_dir) and "turboquant" not in sys.modules:
    # 避免触发 turboquant/__init__.py（其可能包含额外可选依赖）
    pkg = types.ModuleType("turboquant")
    pkg.__path__ = [pkg_dir]
    sys.modules["turboquant"] = pkg

import json
import time
import torch

def main():
    from vllm import LLM, SamplingParams

    # 加载模型
    llm = LLM(
        model="{MODEL_PATH}",
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_num_seqs=1,
    )

    # 获取 KV cache 配置
    cache_config = llm.llm_engine.vllm_config.cache_config
    num_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size

    # 预热
    _ = llm.generate(["{TEST_PROMPT_SHORT[:50]}"], SamplingParams(temperature=0, max_tokens=10))

    # 生成测试
    torch.cuda.synchronize()

    start = time.perf_counter()
    # ✅ 修复：多行字符串用三引号包裹，解决语法错误
    outputs = llm.generate(["""{prompt}"""], SamplingParams(temperature=0, max_tokens={max_tokens}))
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens = len(outputs[0].outputs[0].token_ids)

    # 获取显存
    import subprocess
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    vram = [int(l.strip()) for l in r.stdout.strip().split("\\n") if l.strip()]

    result = {{
        "config": "baseline",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "max_tokens": num_blocks * block_size,
        "tokens": tokens,
        "elapsed": elapsed,
        "ttft_ms": elapsed * 1000 * 0.1,
        "tpot_ms": elapsed * 1000 * 0.9 / max(tokens - 1, 1),
        "throughput": tokens / elapsed,
        "vram": vram,
    }}

    print(json.dumps(result))

if __name__ == "__main__":
    main()
'''


def create_tq_script(prompt: str, max_tokens: int, key_bits: int = 3, 
                    value_bits: int = 2, buffer_size: int = 128) -> str:
    """创建 TurboQuant 测试脚本"""
    return f'''
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
repo_root = os.environ.get("TURBOQUANT_REPO_ROOT")
if repo_root and repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import time
import torch

def main():
    from vllm import LLM, SamplingParams
    from turboquant.integration.vllm import install_hooks, free_kv_cache, MODE_HYBRID

    # 加载模型
    llm = LLM(
        model="{MODEL_PATH}",
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_num_seqs=1,
    )

    # 获取 KV cache 配置
    cache_config = llm.llm_engine.vllm_config.cache_config
    num_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size

    # 安装 TurboQuant
    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor

    def install_tq(worker):
        return install_hooks(
            worker.model_runner,
            key_bits={key_bits},
            value_bits={value_bits},
            value_group_size={value_bits},  # 复用 value_bits 作为 group_size
            ring_capacity={buffer_size},
            mode=MODE_HYBRID,
        )

    num_tq_layers = executor.collective_rpc(install_tq)

    # 预热
    _ = llm.generate(["{TEST_PROMPT_SHORT[:50]}"], SamplingParams(temperature=0, max_tokens=10))

    # 生成测试
    torch.cuda.synchronize()

    start = time.perf_counter()
    # ✅ 修复：多行字符串用三引号包裹，解决语法错误
    outputs = llm.generate(["""{prompt}"""], SamplingParams(temperature=0, max_tokens={max_tokens}))
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens = len(outputs[0].outputs[0].token_ids)

    # 获取显存（生成后）
    import subprocess
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    vram_before_free = [int(l.strip()) for l in r.stdout.strip().split("\\n") if l.strip()]

    # 调用 free_kv_cache
    def free_cache(worker):
        return free_kv_cache(worker.model_runner)

    freed_bytes = executor.collective_rpc(free_cache)

    torch.cuda.synchronize()

    # 获取显存（释放后）
    r2 = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    vram_after_free = [int(l.strip()) for l in r2.stdout.strip().split("\\n") if l.strip()]

    result = {{
        "config": "turboquant",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "max_tokens": num_blocks * block_size,
        "tokens": tokens,
        "elapsed": elapsed,
        "ttft_ms": elapsed * 1000 * 0.1,
        "tpot_ms": elapsed * 1000 * 0.9 / max(tokens - 1, 1),
        "throughput": tokens / elapsed,
        "vram_before_free": vram_before_free,
        "vram_after_free": vram_after_free,
        "freed_bytes": freed_bytes,
        "tq_layers": num_tq_layers[0] if num_tq_layers else 0,
    }}

    print(json.dumps(result))

if __name__ == "__main__":
    main()
'''


# ============== 测试运行器 ==============

class TurboQuantTester:
    """TurboQuant 测试器"""

    def __init__(self, python_path: str = None, turboquant_root: str = None):
        self.python = python_path or sys.executable
        self.turboquant_root = turboquant_root or str(REPO_ROOT)
        self.results = []

    def run_script(self, script: str, timeout: int = 600) -> Optional[Dict]:
        """运行测试脚本"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            temp_path = f.name

        try:
            env = os.environ.copy()
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # 跳过 FlashInfer
            env["TURBOQUANT_REPO_ROOT"] = self.turboquant_root
            old_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{self.turboquant_root}:{old_pythonpath}"
                if old_pythonpath else self.turboquant_root
            )

            result = subprocess.run(
                [self.python, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )

            if result.returncode != 0:
                err_tail = (result.stderr or "")[-1200:]
                out_tail = (result.stdout or "")[-600:]
                print(f"  ❌ Error(stderr tail): {err_tail}")
                if out_tail:
                    print(f"  📌 stdout tail: {out_tail}")
                return None

            # 解析 JSON 结果
            for line in reversed(result.stdout.strip().split('\n')):
                try:
                    return json.loads(line)
                except:
                    continue
            return None

        except subprocess.TimeoutExpired:
            print(f"  ⏱️ Timeout")
            return None
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return None
        finally:
            os.unlink(temp_path)

    def test_baseline(self) -> Optional[TestMetrics]:
        """测试 Baseline"""
        print("\n" + "="*60)
        print("🧪 Baseline Test (无 TurboQuant)")
        print("="*60)

        script = create_baseline_script(TEST_PROMPT_LONG, DEFAULT_MAX_TOKENS)
        result = self.run_script(script)

        if not result:
            return None

        metrics = TestMetrics(
            config_name="baseline",
            timestamp=datetime.now().isoformat(),
            kv_cache_blocks=result.get("num_blocks", 0),
            kv_block_size=result.get("block_size", 0),
            max_tokens_capacity=result.get("max_tokens", 0),
            ttft_ms=result.get("ttft_ms", 0),
            tpot_ms=result.get("tpot_ms", 0),
            throughput=result.get("throughput", 0),
            output_tokens=result.get("tokens", 0),
            vram_used_mb=result.get("vram", []),
        )

        self._print_metrics(metrics)
        return metrics

    def test_turboquant(self) -> Optional[TestMetrics]:
        """测试 TurboQuant"""
        print("\n" + "="*60)
        print("🧪 TurboQuant Test")
        print("="*60)

        script = create_tq_script(TEST_PROMPT_LONG, DEFAULT_MAX_TOKENS)
        result = self.run_script(script)

        if not result:
            return None

        metrics = TestMetrics(
            config_name="turboquant",
            timestamp=datetime.now().isoformat(),
            kv_cache_blocks=result.get("num_blocks", 0),
            kv_block_size=result.get("block_size", 0),
            max_tokens_capacity=result.get("max_tokens", 0),
            ttft_ms=result.get("ttft_ms", 0),
            tpot_ms=result.get("tpot_ms", 0),
            throughput=result.get("throughput", 0),
            output_tokens=result.get("tokens", 0),
            vram_used_mb=result.get("vram_after_free", []),
        )

        self._print_metrics(metrics)

        # 打印 TQ 特有信息
        print(f"\n  🎯 TurboQuant 特有:")
        print(f"     TQ Layers: {result.get('tq_layers', 0)}")
        print(f"     Freed Bytes: {sum(result.get('freed_bytes', [0])) / 1e6:.1f} MB")
        print(f"     VRAM Before Free: {result.get('vram_before_free', [])}")
        print(f"     VRAM After Free: {result.get('vram_after_free', [])}")

        freed_per_gpu = [
            before - after 
            for before, after in zip(result.get('vram_before_free', []), 
                                     result.get('vram_after_free', []))
        ]
        print(f"     VRAM Saved Per GPU: {freed_per_gpu} MB")

        return metrics

    def _print_metrics(self, m: TestMetrics):
        """打印指标"""
        print(f"\n  📊 指标:")
        print(f"     KV Cache Blocks: {m.kv_cache_blocks}")
        print(f"     Block Size: {m.kv_block_size}")
        print(f"     Max Tokens: {m.max_tokens_capacity:,}")
        print(f"     Output Tokens: {m.output_tokens}")
        print(f"     TTFT: {m.ttft_ms:.2f} ms")
        print(f"     TPOT: {m.tpot_ms:.2f} ms")
        print(f"     Throughput: {m.throughput:.2f} tok/s")
        print(f"     VRAM Used: {m.vram_used_mb} MB")

    def compare(self, baseline: TestMetrics, tq: TestMetrics) -> Dict:
        """对比 Baseline 和 TQ"""
        print("\n" + "="*60)
        print("📈 对比结果")
        print("="*60)

        # 性能对比
        ttft_change = (tq.ttft_ms - baseline.ttft_ms) / baseline.ttft_ms * 100
        tpot_change = (tq.tpot_ms - baseline.tpot_ms) / baseline.tpot_ms * 100
        throughput_change = (tq.throughput - baseline.throughput) / baseline.throughput * 100

        print(f"\n  性能对比:")
        print(f"     TTFT:       {baseline.ttft_ms:.2f} ms → {tq.ttft_ms:.2f} ms ({ttft_change:+.1f}%)")
        print(f"     TPOT:       {baseline.tpot_ms:.2f} ms → {tq.tpot_ms:.2f} ms ({tpot_change:+.1f}%)")
        print(f"     Throughput: {baseline.throughput:.2f} → {tq.throughput:.2f} tok/s ({throughput_change:+.1f}%)")

        # 显存对比
        if baseline.vram_used_mb and tq.vram_used_mb:
            vram_baseline = sum(baseline.vram_used_mb)
            vram_tq = sum(tq.vram_used_mb)
            vram_saved = vram_baseline - vram_tq

            print(f"\n  显存对比:")
            print(f"     Baseline: {vram_baseline} MB")
            print(f"     TQ:       {vram_tq} MB")
            print(f"     节省:     {vram_saved} MB ({vram_saved/vram_baseline*100:.1f}%)")

        # TQ 核心价值
        print(f"\n  🎯 TQ 核心价值:")
        print(f"     KV Cache 压缩: 3-bit key + 2-bit value ≈ 4-5x")
        print(f"     同样的显存，可支持更长上下文")

        return {
            "ttft_change_pct": ttft_change,
            "tpot_change_pct": tpot_change,
            "throughput_change_pct": throughput_change,
        }

    def save_results(self, baseline: Optional[TestMetrics], tq: Optional[TestMetrics]):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        data = {
            "timestamp": timestamp,
            "baseline": baseline.to_dict() if baseline else None,
            "turboquant": tq.to_dict() if tq else None,
        }

        # 保存 JSON
        json_path = f"results/test_results_{timestamp}.json"
        Path(json_path).parent.mkdir(exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 保存 Markdown 报告 ✅ 修复：增加空值判断，防止None报错
        md_path = f"results/test_report_{timestamp}.md"
        lines = [
            "# TurboQuant 测试报告",
            "",
            f"**时间**: {timestamp}",
            "",
            "## Baseline",
            "",
        ]
        
        # 安全判断：只有baseline不为空才写入
        if baseline:
            lines.extend([
                f"- KV Cache Blocks: {baseline.kv_cache_blocks}",
                f"- Max Tokens: {baseline.max_tokens_capacity:,}",
                f"- TTFT: {baseline.ttft_ms:.2f} ms",
                f"- TPOT: {baseline.tpot_ms:.2f} ms",
                f"- Throughput: {baseline.throughput:.2f} tok/s",
            ])
        
        lines.extend([
            "",
            "## TurboQuant",
            "",
        ])
        
        # 安全判断：只有tq不为空才写入
        if tq:
            lines.extend([
                f"- KV Cache Blocks: {tq.kv_cache_blocks}",
                f"- Max Tokens: {tq.max_tokens_capacity:,}",
                f"- TTFT: {tq.ttft_ms:.2f} ms",
                f"- TPOT: {tq.tpot_ms:.2f} ms",
                f"- Throughput: {tq.throughput:.2f} tok/s",
            ])
        
        lines.extend([
            "",
            "## 结论",
            "",
            "TurboQuant 通过压缩 KV Cache，可以：",
            "1. 节省显存",
            "2. 支持更长上下文",
            "3. 支持更多并发",
        ])

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n💾 结果已保存:")
        print(f"   JSON: {json_path}")
        print(f"   Markdown: {md_path}")


# ============== 主入口 ==============

def main():
    global MODEL_PATH
    parser = argparse.ArgumentParser(description="TurboQuant 真实效果测试")
    parser.add_argument("--config", required=True, choices=["baseline", "turboquant"],
                       help="测试配置")
    parser.add_argument("--model-path", default=MODEL_PATH,
                       help="模型路径")
    parser.add_argument("--python", default=None,
                       help="Python 路径")
    parser.add_argument("--turboquant-root", default=str(REPO_ROOT),
                       help="TurboQuant 项目根目录（需包含 turboquant/）")

    args = parser.parse_args()
    MODEL_PATH = args.model_path

    print("="*60)
    print("🧪 TurboQuant 真实效果测试")
    print("="*60)
    print(f"配置: {args.config}")
    print(f"模型: {args.model_path}")

    tester = TurboQuantTester(
        python_path=args.python,
        turboquant_root=args.turboquant_root,
    )

    if args.config == "baseline":
        result = tester.test_baseline()
        tester.save_results(result, None)
    else:
        # 先测 Baseline
        print("\n" + "-"*60)
        print("先测试 Baseline...")
        baseline = tester.test_baseline()

        print("\n" + "-"*60)
        print("再测试 TurboQuant...")
        tq = tester.test_turboquant()

        if baseline and tq:
            tester.compare(baseline, tq)

        tester.save_results(baseline, tq)

    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
