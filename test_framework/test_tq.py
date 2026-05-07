#!/usr/bin/env python3
"""
TurboQuant 真实效果测试

方案 A: 上下文扩展能力测试 - 测量相同显存下能支持的最大上下文长度
方案 B: 并发请求测试 - 测量相同显存下能支持的最大并发数

通过 subprocess 调用预编译 vllm 二进制，避免 FlashInfer JIT 编译问题。

Usage:
    python test_tq.py context-extend    # 方案 A
    python test_tq.py concurrency        # 方案 B
    python test_tq.py all               # 运行全部测试
"""

import os
import sys
import json
import time
import signal
import socket
import argparse
import subprocess
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入测试数据集
from datasets import TEST_DATASETS, get_test_prompt

# 使用多文档QA作为主要测试（最具挑战性）
DEFAULT_DATASET = "multi_doc_qa"

def get_long_prompt(dataset_key: str = None, repeat: int = 1) -> str:
    """获取长文本测试prompt，可重复多次增加长度"""
    prompt = get_test_prompt(dataset_key or DEFAULT_DATASET)
    if repeat > 1:
        # 重复内容增加输入长度，模拟超长文档
        prompt = prompt + ("\n\n[Additional context for longer context testing]\n" + prompt) * (repeat - 1)
    return prompt


MODEL_NAME = "Qwen35_2b"
MODEL_PATH = "/mnt/data1/dmx/Models/Qwen35_2b"
UV_VLLM_BIN = "/home/deng/vllm/bin/vllm"
CONDA_SH = "/mnt/data1/dmx/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV = "vllm_qw"

SERVER_PORT = 8001
SERVER_START_TIMEOUT = 180
DEFAULT_MAX_TOKENS = 1024  # 增加到 1024 tokens，充分测试长上下文

# 长文本测试：使用真实的 TurboQuant 论文内容
LONG_CONTEXT_PROMPT = """Please read the following research paper and provide a detailed summary:

TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni

Abstract:
Vector quantization, a problem rooted in Shannon's source coding theory, aims to quantize high-dimensional Euclidean vectors while minimizing distortion in their geometric structure. We propose TurboQuant to address both mean-squared error (MSE) and inner product distortion, overcoming limitations of existing methods that fail to achieve optimal distortion rates. Our data-oblivious algorithms, suitable for online applications, achieve near-optimal distortion rates (within a small constant factor) across all bit-widths and dimensions.

TurboQuant achieves this by randomly rotating input vectors, inducing a concentrated Beta distribution on coordinates, and leveraging the near-independence property of distinct coordinates in high dimensions to simply apply optimal scalar quantizers per each coordinate. Recognizing that MSE-optimal quantizers introduce bias in inner product estimation, we propose a two-stage approach: applying an MSE quantizer followed by a 1-bit Quantized JL (QJL) transform on the residual, resulting in an unbiased inner product quantizer. We also provide a formal proof of the information-theoretic lower bounds on best achievable distortion rate by any vector quantizer, demonstrating that TurboQuant closely matches these bounds, differing only by a small constant (≈2.7) factor. Experimental results validate our theoretical findings, showing that for KV cache quantization, we achieve absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with 2.5 bits per channel. Furthermore, in nearest neighbor search tasks, our method outperforms existing product quantization techniques in recall while reducing indexing time to virtually zero.

Introduction:
Vector quantization (VQ) in Euclidean space is crucial for efficiently handling high-dimensional vectors across a spectrum of computational domains, from training and deploying large-scale AI and deep learning models to powering vector databases for search/retrieval systems. The core objective is to compress high dimensional vectors by quantizing them–converting floating-point coordinate values to low-bitwidth integers–while minimizing distortion, quantified by metrics such as mean-squared error (MSE) or inner product errors.

This problem's roots trace back to Shannon's seminal work on Source Coding theory, which established that the least distortion achievable by block source codes, now known as vector quantizers, is defined by the Shannon distortion-rate function. Today, VQ plays a critical role in fundamental computational domains, including AI, deep learning, and search systems.

A key application of VQ is in the deployment of AI models, including large language models (LLMs). As LLM capabilities depend heavily on their model size and context length, serving them requires substantial memory demands and increased inference latency. This latency is primarily attributed to communication bottlenecks between HBM and SRAM on accelerators, or across distributed clusters. By compressing or quantizing model weights and activations, we can effectively mitigate these bottlenecks, resulting in significant reductions in inference costs.

Decoder based transformer models present another compelling use case. These models must store key/value (KV) embeddings from previously generated tokens in the KV cache, the size of which scales with both model size (number of layers and attention heads) and context length. This scaling is a significant bottleneck in terms of memory usage and computational speed, especially for long context models. Therefore, reducing the KV cache size without compromising accuracy is essential.

Problem Definition:
Formally, our goal is to design a quantization map, denoted as Q : R^d → {0, 1}^B, that transforms d-dimensional vectors to a binary string of B bits. If we set B = b · d for some b ≥ 0, this quantizer will have a bit-width of b, representing the average number of bits used to encode each real-valued coordinate of R^d.

We aim to design quantizers that for any desired bit-width b minimize the following expected distortion measures:
- MSE: D_mse := E[||x - Q^(-1)(Q(x))||^2]
- Inner product error: D_prod := E[(<y,x> - <y,Q^(-1)(Q(x))>)^2]

For inner-product quantizers, we require unbiasedness: E[<y, Q^(-1)(Q(x))>] = <y, x>

Key Technical Contributions:
1. MSE Optimized TurboQuant: Our first VQ algorithm minimizes MSE distortion by applying random rotation to input vectors, inducing a Beta distribution on each coordinate. We design optimal Lloyd-Max quantizers for each coordinate by solving a continuous k-means problem.

2. Inner Product TurboQuant: We show that MSE optimized quantizers are biased for inner product estimation. Our solution is a two-stage algorithm that first applies Q_mse with bit-width one less than target, then applies QJL on the residual error.

3. Theoretical Guarantees: We prove that TurboQuant achieves near-optimal distortion bounds, differing from information-theoretic lower bounds by only a small constant factor (≈2.7).

Please provide a comprehensive summary covering:
1. Main contributions and innovations
2. Technical approach (MSE and inner product quantization)
3. Theoretical guarantees
4. Experimental results on KV cache compression
5. Implications for LLM deployment
"""


def wait_for_server(port: int, timeout: int = 120) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(3)
    return False


def get_vram() -> List[float]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        return [float(l.strip()) for l in result.stdout.strip().split("\n") if l.strip()]
    except:
        return []


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_process_on_port(port: int):
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=10)
        time.sleep(2)
    except:
        pass


def start_vllm_server(config: str, max_model_len: int, port: int) -> Optional[subprocess.Popen]:
    env = os.environ.copy()
    env["TURBOQUANT_REPO_ROOT"] = str(Path(__file__).parent.parent)

    bash_cmd = f"source {CONDA_SH} && conda activate {CONDA_ENV} && "

    if config == "turboquant":
        bash_cmd += (
            f"export TURBOQUANT_ENABLED=1 && "
            f"export TURBOQUANT_KEY_BITS=3 && "
            f"export TURBOQUANT_VALUE_BITS=2 && "
            f"export TURBOQUANT_BUFFER_SIZE=128 && "
        )

    bash_cmd += (
        f"exec {UV_VLLM_BIN} serve {MODEL_PATH} "
        f"--host 0.0.0.0 --port {port} "
        f"--served-model-name {MODEL_NAME} "
        f"--gpu-memory-utilization 0.3 "
        f"--max-model-len {max_model_len} "
        f"--tensor-parallel-size 1 "
        f"--trust-remote-code "
        f"--enforce-eager "
        f"--gdn-prefill-backend triton"
    )

    process = subprocess.Popen(
        ["bash", "-c", bash_cmd],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    return process


def stop_vllm_server(process: subprocess.Popen, port: int):
    if process:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=15)
        except:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass
    kill_process_on_port(port)
    time.sleep(3)


def generate_request(port: int, prompt: str, max_tokens: int = 128) -> Optional[Dict]:
    try:
        start = time.perf_counter()
        resp = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
            },
            stream=True,
            timeout=180
        )

        if resp.status_code != 200:
            elapsed = time.perf_counter() - start
            return {"success": False, "error": f"HTTP {resp.status_code}", "elapsed": elapsed}

        first_token_time = None
        last_token_time = None
        token_count = 0
        token_times = []

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break

            chunk_time = time.perf_counter()
            token_times.append(chunk_time)

            if first_token_time is None:
                first_token_time = chunk_time

            last_token_time = chunk_time

            try:
                chunk_data = json.loads(data_str)
                if chunk_data.get("choices", [{}])[0].get("finish_reason"):
                    break
            except json.JSONDecodeError:
                continue

        total_elapsed = time.perf_counter() - start

        if token_count == 0 and token_times:
            token_count = len(token_times)
        elif token_count == 0:
            return {"success": False, "error": "No tokens received", "elapsed": total_elapsed}

        ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0

        if len(token_times) > 1:
            tpot_ms = (last_token_time - first_token_time) * 1000 / max(token_count - 1, 1)
        else:
            tpot_ms = 0

        throughput = token_count / total_elapsed if total_elapsed > 0 else 0

        return {
            "success": True,
            "tokens": token_count,
            "elapsed": total_elapsed,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "throughput": throughput,
        }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout", "elapsed": 180}
    except Exception as e:
        return {"success": False, "error": str(e), "elapsed": 0}


def test_context_extend(config: str, port: int) -> Dict:
    print(f"\n{'='*60}")
    print(f"🧪 方案 A: 上下文扩展能力测试 ({config})")
    print(f"{'='*60}")

    results = []

    # H20 96GB 可以测试更大的上下文
    for max_len in [32768, 49152, 65536, 81920, 98304, 114688, 131072]:
        print(f"\n  测试 max_model_len={max_len}...")

        vram_before = get_vram()
        print(f"    VRAM before: {vram_before}")

        process = start_vllm_server(config, max_len, port)
        if not process:
            print(f"    ❌ 启动失败")
            results.append({"max_model_len": max_len, "success": False, "error": "启动失败"})
            continue

        server_ready = wait_for_server(port, SERVER_START_TIMEOUT)

        if not server_ready:
            print(f"    ❌ 服务启动超时")
            stop_vllm_server(process, port)
            results.append({"max_model_len": max_len, "success": False, "error": "启动超时"})
            continue

        time.sleep(5)

        vram_after_start = get_vram()
        print(f"    VRAM after start: {vram_after_start}")

        # 使用长文本测试（重复3次以增加输入长度到10k+ tokens）
        result = generate_request(port, get_long_prompt(repeat=3), DEFAULT_MAX_TOKENS)

        vram_after_req = get_vram()
        print(f"    VRAM after request: {vram_after_req}")

        stop_vllm_server(process, port)

        if result and result["success"]:
            print(f"    ✅ 成功 (tokens={result['tokens']}, elapsed={result['elapsed']:.2f}s)")
            results.append({
                "max_model_len": max_len,
                "success": True,
                "tokens": result["tokens"],
                "elapsed": result["elapsed"],
                "vram_before": vram_before,
                "vram_after_start": vram_after_start,
                "vram_after_req": vram_after_req,
            })
        else:
            error_msg = result.get("error", "未知") if result else "无响应"
            print(f"    ❌ 失败 ({error_msg})")
            results.append({
                "max_model_len": max_len,
                "success": False,
                "error": error_msg,
            })

    return results


def test_concurrency(config: str, port: int, max_model_len: int = 65536, concurrency_levels: List[int] = None) -> Dict:
    # H20 96GB 可以测试更高的并发
    if concurrency_levels is None:
        concurrency_levels = [50, 100, 200, 300, 500]

    print(f"\n{'='*60}")
    print(f"🧪 方案 B: 并发请求测试 ({config})")
    print(f"{'='*60}")

    results = []

    for concurrency in concurrency_levels:
        print(f"\n  测试并发数={concurrency}...")

        process = start_vllm_server(config, max_model_len, port)
        if not process:
            print(f"    ❌ 启动失败")
            results.append({"concurrency": concurrency, "success": False, "error": "启动失败"})
            continue

        server_ready = wait_for_server(port, SERVER_START_TIMEOUT)
        if not server_ready:
            print(f"    ❌ 服务启动超时")
            stop_vllm_server(process, port)
            results.append({"concurrency": concurrency, "success": False, "error": "启动超时"})
            continue

        time.sleep(5)

        vram_before = get_vram()
        print(f"    VRAM before: {vram_before}")

        start_time = time.time()
        success_count = 0
        failed_count = 0
        total_tokens = 0
        latencies = []

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(generate_request, port, get_long_prompt(repeat=2), DEFAULT_MAX_TOKENS)
                for _ in range(concurrency)
            ]

            for future in as_completed(futures):
                result = future.result()
                if result and result["success"]:
                    success_count += 1
                    total_tokens += result["tokens"]
                    latencies.append(result["elapsed"])
                else:
                    failed_count += 1

        total_time = time.time() - start_time

        vram_after = get_vram()
        print(f"    VRAM after: {vram_after}")

        stop_vllm_server(process, port)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        throughput = total_tokens / total_time if total_time > 0 else 0

        print(f"    成功: {success_count}/{concurrency}, "
              f"总tokens: {total_tokens}, "
              f"平均延迟: {avg_latency:.2f}s, "
              f"吞吐: {throughput:.2f} tok/s")

        results.append({
            "concurrency": concurrency,
            "success": success_count,
            "failed": failed_count,
            "total_tokens": total_tokens,
            "avg_latency": avg_latency,
            "throughput": throughput,
            "vram_before": vram_before,
            "vram_after": vram_after,
        })

    return results


def compare_results(baseline_results: Dict, tq_results: Dict):
    print(f"\n{'='*60}")
    print(f"📈 对比分析")
    print(f"{'='*60}")

    print(f"\n【方案 A: 上下文扩展能力】")
    print(f"  Baseline 最大支持: ", end="")
    baseline_max = max(
        (r for r in baseline_results["context"] if r.get("success", False)),
        key=lambda x: x["max_model_len"],
        default=None
    )
    tq_max = max(
        (r for r in tq_results["context"] if r.get("success", False)),
        key=lambda x: x["max_model_len"],
        default=None
    )

    if baseline_max:
        print(f"max_model_len={baseline_max['max_model_len']}")
    else:
        print("无法确定")

    if tq_max:
        print(f"  TurboQuant 最大支持: max_model_len={tq_max['max_model_len']}")

    if baseline_max and tq_max:
        diff = tq_max['max_model_len'] - baseline_max['max_model_len']
        pct = diff / baseline_max['max_model_len'] * 100
        print(f"  提升: +{diff} ({pct:+.1f}%)")

    print(f"\n【方案 B: 并发请求】")
    for baseline_r, tq_r in zip(baseline_results["concurrency"], tq_results["concurrency"]):
        c = baseline_r["concurrency"]
        baseline_tp = baseline_r.get("throughput", 0)
        tq_tp = tq_r.get("throughput", 0)
        if baseline_tp > 0:
            tp_diff = (tq_tp - baseline_tp) / baseline_tp * 100
            print(f"  并发={c}: Baseline {baseline_tp:.1f} tok/s → TQ {tq_tp:.1f} tok/s ({tp_diff:+.1f}%)")


def save_results(baseline_results: Dict, tq_results: Dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        "timestamp": timestamp,
        "baseline": baseline_results,
        "turboquant": tq_results,
    }

    json_path = f"results/test_results_{timestamp}.json"
    Path(json_path).parent.mkdir(exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 结果已保存: {json_path}")


def run_baseline_tests(port: int) -> Dict:
    return {
        "context": test_context_extend("baseline", port),
        "concurrency": test_concurrency("baseline", port),
    }


def run_tq_tests(port: int) -> Dict:
    return {
        "context": test_context_extend("turboquant", port),
        "concurrency": test_concurrency("turboquant", port),
    }


def main():
    parser = argparse.ArgumentParser(description="TurboQuant 真实效果测试")
    parser.add_argument("test", choices=["context-extend", "concurrency", "all"],
                       help="测试类型: context-extend(A), concurrency(B), all")
    parser.add_argument("--baseline-only", action="store_true",
                       help="仅测试 baseline")
    parser.add_argument("--tq-only", action="store_true",
                       help="仅测试 turboquant")
    parser.add_argument("--port", type=int, default=8001,
                       help="服务端口 (默认: 8001)")

    args = parser.parse_args()
    port = args.port

    print("="*60)
    print("🧪 TurboQuant 真实效果测试")
    print("="*60)
    print(f"测试类型: {args.test}")
    print(f"服务端口: {port}")

    baseline_results = None
    tq_results = None

    if args.baseline_only:
        baseline_results = run_baseline_tests(port)
    elif args.tq_only:
        tq_results = run_tq_tests(port)
    else:
        print("\n" + "-"*60)
        print("先测试 Baseline...")
        baseline_results = run_baseline_tests(port)

        print("\n" + "-"*60)
        print("再测试 TurboQuant...")
        tq_results = run_tq_tests(port)

        if baseline_results and tq_results:
            compare_results(baseline_results, tq_results)

    if baseline_results:
        save_results(baseline_results, tq_results or {})

    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()