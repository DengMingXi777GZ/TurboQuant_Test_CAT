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

# 加载真实的《战争与Peace》长文本
WAR_AND_PEACE_FILE = Path(__file__).parent / "war_and_peace.txt"
if WAR_AND_PEACE_FILE.exists():
    with open(WAR_AND_PEACE_FILE, "r", encoding="utf-8", errors="ignore") as f:
        WAR_AND_PEACE_TEXT = f.read()
    # 找到正文开始位置（跳过 Gutenberg header）
    start_idx = WAR_AND_PEACE_TEXT.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    if start_idx != -1:
        end_marker = WAR_AND_PEACE_TEXT.find("***", start_idx + 50)
        if end_marker != -1:
            WAR_AND_PEACE_TEXT = WAR_AND_PEACE_TEXT[end_marker + 3:]
    print(f"    [DATA] 加载 war_and_peace.txt: {len(WAR_AND_PEACE_TEXT)} chars")
else:
    WAR_AND_PEACE_TEXT = ""
    print(f"    [DATA] war_and_peace.txt not found, using fallback")

def get_long_prompt(chars: int = 10000) -> str:
    """获取指定长度的真实文本"""
    if WAR_AND_PEACE_TEXT:
        # 从文本开头截取指定长度
        return WAR_AND_PEACE_TEXT[:chars]
    # Fallback: 使用数据集
    return get_test_prompt()


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


def check_turboquant_enabled(port: int) -> Dict:
    """检查 TurboQuant 是否启用"""
    result = {"enabled": False, "method": "unknown", "details": ""}

    # 方法1: 检查 vLLM 模型信息
    try:
        resp = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            model_info = data.get("data", [{}])[0]
            result["model_info"] = model_info.get("id", "unknown")
    except Exception as e:
        result["details"] += f"API check failed: {e}; "

    # 方法2: 发送测试请求，检查处理时间特征
    # TQ 压缩后首次 prefill 可能稍慢，但 decode 会更快
    test_prompt = "Hello, this is a test." * 50
    try:
        start = time.perf_counter()
        resp = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={"model": MODEL_NAME, "prompt": test_prompt, "max_tokens": 10},
            timeout=60
        )
        elapsed = time.perf_counter() - start
        result["test_latency_ms"] = round(elapsed * 1000, 1)

        if resp.status_code == 200:
            result["api_works"] = True
        else:
            result["api_error"] = resp.status_code
    except Exception as e:
        result["test_error"] = str(e)

    return result


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
    turboquant_root = str(Path(__file__).parent.parent)
    env["TURBOQUANT_REPO_ROOT"] = turboquant_root
    env["PYTHONPATH"] = f"{turboquant_root}:{env.get('PYTHONPATH', '')}"

    bash_cmd = f"source {CONDA_SH} && conda activate {CONDA_ENV} && "

    if config == "turboquant":
        print(f"    [TQ] 启用 TurboQuant: key_bits=3, value_bits=2, buffer_size=128")
        bash_cmd += (
            f"export TURBOQUANT_ENABLED=1 && "
            f"export TURBOQUANT_KEY_BITS=3 && "
            f"export TURBOQUANT_VALUE_BITS=2 && "
            f"export TURBOQUANT_BUFFER_SIZE=128 && "
            f"export PYTHONPATH={turboquant_root}:$PYTHONPATH && "
        )
    else:
        print(f"    [TQ] TurboQuant 关闭 (baseline)")
        bash_cmd += "export TURBOQUANT_ENABLED=0 && "

    # 与 start_vllm_qwen35_2b.sh 保持一致
    bash_cmd += (
        f"exec {UV_VLLM_BIN} serve {MODEL_PATH} "
        f"--host 0.0.0.0 --port {port} "
        f"--served-model-name {MODEL_NAME} "
        f"--gpu-memory-utilization 0.1 "
        f"--max-model-len {max_model_len} "
        f"--gdn-prefill-backend triton "
        f"--trust-remote-code"
    )

    print(f"    [CMD] {bash_cmd[:200]}...")

    process = subprocess.Popen(
        ["bash", "-c", bash_cmd],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
        bufsize=1
    )

    time.sleep(5)
    if process.poll() is not None:
        output = process.stdout.read() if process.stdout else ""
        print(f"    [ERROR] 进程立即退出: {output[:1000]}")
        return None

    return process


def get_server_output(process: subprocess.Popen, timeout: int = 30) -> str:
    """获取服务器启动日志，用于诊断问题"""
    output = []
    try:
        import select
        import fcntl
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        start = time.time()
        while time.time() - start < timeout:
            if select.select([process.stdout], [], [], 1)[0]:
                line = process.stdout.readline()
                if line:
                    output.append(line)
                    print(f"    [vllm] {line.rstrip()}")
                elif process.poll() is not None:
                    break
    except:
        pass
    return "".join(output)


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

        full_text = ""
        first_token_time = None
        last_token_time = None
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

            try:
                chunk_data = json.loads(data_str)
                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                text = delta.get("text", "")
                if text:
                    full_text += text

                if first_token_time is None:
                    first_token_time = chunk_time
                last_token_time = chunk_time
                token_times.append(chunk_time)

                if chunk_data.get("choices", [{}])[0].get("finish_reason"):
                    break
            except json.JSONDecodeError:
                continue

        total_elapsed = time.perf_counter() - start
        token_count = len(full_text) // 4  # 粗略估算：约4个字符一个token

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

    # 使用与 shell 脚本一致的配置，测试更小的 max_model_len
    for max_len in [8192, 10240, 12288, 14336, 16384, 20480, 32768]:
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
            print(f"    ❌ 服务启动超时，打印日志:")
            get_server_output(process, timeout=10)
            stop_vllm_server(process, port)
            results.append({"max_model_len": max_len, "success": False, "error": "启动超时"})
            continue

        time.sleep(5)

        # 检查 TQ 是否启用
        print(f"    🔍 检查 TurboQuant 状态...")
        tq_status = check_turboquant_enabled(port)
        if tq_status.get("api_works"):
            print(f"    ✅ 服务正常, 测试延迟: {tq_status.get('test_latency_ms')}ms")
        else:
            print(f"    ⚠️ 服务异常: {tq_status}")

        vram_after_start = get_vram()
        print(f"    VRAM after start: {vram_after_start}")

        # 使用真实的 war_and_peace 文本 ~10k tokens
        result = generate_request(port, get_long_prompt(15000), DEFAULT_MAX_TOKENS)

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


def test_concurrency(config: str, port: int, max_model_len: int = 32768) -> Dict:
    """
    并发测试核心思路：
    - 固定 max_model_len=32k（确保 KV cache 需求足够大）
    - 逐步增加并发数，直到服务无法处理
    - 对比 Baseline vs TurboQuant 的最大并发容量
    """
    # 并发数搜索：从低到高，找到 OOM 前能处理的最大值
    base_levels = [5, 10, 20, 30, 50, 80, 100, 150, 200]

    print(f"\n{'='*60}")
    print(f"🧪 方案 B: 并发请求容量测试 ({config})")
    print(f"    max_model_len={max_model_len}, 寻找最大并发容量...")
    print(f"{'='*60}")

    results = []
    max_success = 0

    for concurrency in base_levels:
        print(f"\n  测试并发数={concurrency}...")

        process = start_vllm_server(config, max_model_len, port)
        if not process:
            print(f"    ❌ 启动失败")
            results.append({"concurrency": concurrency, "success": 0, "failed": 0, "error": "启动失败"})
            continue

        server_ready = wait_for_server(port, SERVER_START_TIMEOUT)
        if not server_ready:
            print(f"    ❌ 服务启动超时，打印日志:")
            get_server_output(process, timeout=10)
            stop_vllm_server(process, port)
            results.append({"concurrency": concurrency, "success": 0, "failed": 0, "error": "启动超时"})
            break

        time.sleep(5)

        # 检查 TQ 是否启用
        print(f"    🔍 检查 TurboQuant 状态...")
        tq_status = check_turboquant_enabled(port)
        if tq_status.get("api_works"):
            print(f"    ✅ 服务正常, 测试延迟: {tq_status.get('test_latency_ms')}ms")
        else:
            print(f"    ⚠️ 服务异常: {tq_status}")

        vram_before = get_vram()
        print(f"    VRAM before: {vram_before}")

        # 使用真实的 war_and_peace 文本 ~20k tokens，真正压榨 KV cache
        long_prompt = get_long_prompt(25000)  # ~20k tokens 输入

        start_time = time.time()
        success_count = 0
        failed_count = 0
        total_tokens = 0
        latencies = []
        errors = []

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(generate_request, port, long_prompt, 512)  # 512 tokens 输出
                for _ in range(concurrency)
            ]

            for future in as_completed(futures):
                result = future.result()
                if result and result.get("success"):
                    success_count += 1
                    total_tokens += result.get("tokens", 0)
                    latencies.append(result.get("elapsed", 0))
                else:
                    failed_count += 1
                    errors.append(result.get("error", "unknown") if result else "no result")

        total_time = time.time() - start_time

        vram_after = get_vram()
        print(f"    VRAM after: {vram_after}")

        stop_vllm_server(process, port)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        throughput = total_tokens / total_time if total_time > 0 else 0

        print(f"    成功: {success_count}/{concurrency}, "
              f"总tokens: {total_tokens}, "
              f"平均延迟: {avg_latency:.2f}s, "
              f"吞吐: {throughput:.1f} tok/s")

        if errors and len(errors) <= 3:
            print(f"    错误示例: {errors[:3]}")

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

        # 如果失败率超过 50%，说明接近容量上限，停止测试
        if failed_count > concurrency * 0.5:
            print(f"    ⚠️ 失败率 {failed_count}/{concurrency} > 50%，停止增长测试")
            max_success = success_count
            break

        max_success = max(max_success, success_count)

    print(f"\n  📊 {config} 最大并发容量: ~{max_success} 个请求")

    return results


def compare_results(baseline_results: Dict, tq_results: Dict):
    print(f"\n{'='*60}")
    print(f"📈 对比分析 - TurboQuant 真实效果")
    print(f"{'='*60}")

    # 方案 B: 并发容量对比 (最重要)
    baseline_conc = baseline_results.get("concurrency", [])
    tq_conc = tq_results.get("concurrency", [])

    # 找到 Baseline 最大成功并发数
    baseline_max = 0
    for r in baseline_conc:
        if r.get("success", 0) > 0:
            baseline_max = max(baseline_max, r["success"])

    # 找到 TQ 最大成功并发数
    tq_max = 0
    for r in tq_conc:
        if r.get("success", 0) > 0:
            tq_max = max(tq_max, r["success"])

    print(f"\n🎯 【核心指标】最大并发容量")
    print(f"  Baseline:     {baseline_max} 个并发请求")
    print(f"  TurboQuant:   {tq_max} 个并发请求")
    if baseline_max > 0 and tq_max > 0:
        improvement = (tq_max - baseline_max) / baseline_max * 100
        print(f"  提升:        +{tq_max - baseline_max} ({improvement:+.1f}%)")

    # 显示详细的并发测试结果
    print(f"\n📊 【并发测试详情】")
    print(f"{'并发':>6} | {'Baseline成功':>12} | {'TQ成功':>8} | {'提升':>8}")
    print("-" * 50)
    for i, (br, tr) in enumerate(zip(baseline_conc, tq_conc)):
        c = br.get("concurrency", 0)
        bs = br.get("success", 0)
        ts = tr.get("success", 0)
        diff = ts - bs
        diff_pct = f"{diff:+.0f}" if diff != 0 else "-"
        print(f"{c:>6} | {bs:>12} | {ts:>8} | {diff_pct:>8}")

    # 方案 A: 上下文扩展 (如果有结果)
    baseline_ctx = baseline_results.get("context", [])
    tq_ctx = tq_results.get("context", [])

    if baseline_ctx and tq_ctx:
        print(f"\n📊 【上下文扩展详情】")
        print(f"{'max_model_len':>15} | {'Baseline':>8} | {'TQ':>8}")
        print("-" * 40)
        for br, tr in zip(baseline_ctx, tq_ctx):
            ml = br.get("max_model_len", 0)
            bs = "✅" if br.get("success") else "❌"
            ts = "✅" if tr.get("success") else "❌"
            print(f"{ml:>15} | {bs:>8} | {ts:>8}")


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
        print("先测试 TurboQuant...")
        tq_results = run_tq_tests(port)

        print("\n" + "-"*60)
        print("再测试 Baseline...")
        baseline_results = run_baseline_tests(port)

        if baseline_results and tq_results:
            compare_results(baseline_results, tq_results)

    if baseline_results:
        save_results(baseline_results, tq_results or {})

    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()