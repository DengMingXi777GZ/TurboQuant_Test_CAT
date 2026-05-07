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


MODEL_NAME = "Qwen35_2b"
MODEL_PATH = "/mnt/data1/dmx/Models/Qwen35_2b"
UV_VLLM_BIN = "/home/deng/vllm/bin/vllm"
CONDA_SH = "/mnt/data1/dmx/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV = "vllm_qw"

SERVER_PORT = 8001
SERVER_START_TIMEOUT = 180
DEFAULT_MAX_TOKENS = 128

TEST_PROMPT = "Explain machine learning in detail. " * 20


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

    for max_len in [8192, 10240, 12288, 14336, 16384, 18432, 20480]:
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

        test_prompt = TEST_PROMPT * 4
        result = generate_request(port, test_prompt, DEFAULT_MAX_TOKENS)

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


def test_concurrency(config: str, port: int, max_model_len: int = 8192, concurrency_levels: List[int] = None) -> Dict:
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16]

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
                executor.submit(generate_request, port, TEST_PROMPT, DEFAULT_MAX_TOKENS)
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