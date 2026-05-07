#!/usr/bin/env python3
"""
TurboQuant HTTP API 性能测试工具

通过 vLLM HTTP 服务进行严谨的性能测试，测量真实指标：
- TTFT (Time To First Token)
- TPOT (Time Per Output Token) 
- Throughput
- 显存使用（通过 metrics 接口）

Usage:
    # 1. 先启动 vLLM 服务（Baseline）
    bash start_vllm_qwen35_2b_tq.sh
    
    # 2. 运行测试
    python test_tq_api.py --url http://localhost:8000 --test all
    
    # 3. 停止服务，启动 TQ 版本
    TQ_ENABLED=1 bash start_vllm_qwen35_2b_tq.sh
    
    # 4. 再次运行测试
    python test_tq_api.py --url http://localhost:8000 --test all

Features:
    - 精确测量 TTFT 和 TPOT（使用流式输出）
    - 获取真实显存数据（通过 /metrics 端点）
    - 生成详细 log 文件
    - 支持多次采样取平均
"""

import os
import sys
import json
import time
import argparse
import logging
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import requests


# ============== 配置 ==============

DEFAULT_URL = "http://localhost:8000"
MODEL_NAME = "Qwen35_2b"

# 测试参数
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_RUNS = 5  # 采样次数，取平均

# 测试用的长文本（内置，无需外部文件）
TEST_PROMPT_SHORT = "Explain the concept of machine learning in simple terms."

TEST_PROMPT_LONG = """Artificial intelligence has transformed numerous industries in recent decades. 
Machine learning algorithms can now process vast amounts of data to identify patterns 
and make predictions with remarkable accuracy. Deep learning, a subset of machine learning, 
uses neural networks with multiple layers to model complex patterns in data. 
The applications of AI range from computer vision and natural language processing 
to autonomous vehicles and medical diagnosis. As the field continues to evolve, 
researchers are pushing the boundaries of what machines can learn and accomplish."""

# QA 测试问题
QA_QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("What is 17 * 23?", "391"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
]


# ============== 数据结构 ==============

@dataclass
class PerformanceMetrics:
    """性能指标"""
    ttft_ms: float  # Time To First Token (ms)
    tpot_ms: float  # Time Per Output Token (ms)
    total_time_ms: float  # 总耗时
    throughput_tok_s: float  # 吞吐量 (tokens/s)
    prompt_tokens: int
    output_tokens: int
    samples: int  # 采样次数
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MemoryMetrics:
    """显存指标"""
    gpu_memory_used_mb: List[float]  # 各 GPU 显存使用 (MB)
    gpu_memory_total_mb: List[float]  # 各 GPU 显存总量 (MB)
    timestamp: str
    
    def avg_used_mb(self) -> float:
        return statistics.mean(self.gpu_memory_used_mb) if self.gpu_memory_used_mb else 0
    
    def max_used_mb(self) -> float:
        return max(self.gpu_memory_used_mb) if self.gpu_memory_used_mb else 0
    
    def to_dict(self) -> Dict:
        return {
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "avg_used_mb": self.avg_used_mb(),
            "max_used_mb": self.max_used_mb(),
            "timestamp": self.timestamp
        }


@dataclass
class TestResult:
    """单次测试结果"""
    config_name: str  # "baseline" 或 "turboquant"
    test_type: str  # "performance", "memory", "quality"
    timestamp: str
    
    # 性能指标
    performance: Optional[PerformanceMetrics] = None
    
    # 显存指标
    memory: Optional[MemoryMetrics] = None
    
    # 质量指标
    qa_correct: int = 0
    qa_total: int = 0
    qa_accuracy: float = 0.0
    
    # 原始输出样本
    sample_output: str = ""
    
    def to_dict(self) -> Dict:
        result = {
            "config_name": self.config_name,
            "test_type": self.test_type,
            "timestamp": self.timestamp,
            "qa_correct": self.qa_correct,
            "qa_total": self.qa_total,
            "qa_accuracy": self.qa_accuracy,
            "sample_output": self.sample_output
        }
        if self.performance:
            result["performance"] = self.performance.to_dict()
        if self.memory:
            result["memory"] = self.memory.to_dict()
        return result


# ============== 日志配置 ==============

def setup_logging(log_dir: str = "./logs") -> Tuple[logging.Logger, str]:
    """配置日志输出"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"test_tq_{timestamp}.log"
    
    # 创建 logger
    logger = logging.getLogger("TurboQuantTest")
    logger.setLevel(logging.DEBUG)
    
    # 文件 handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, str(log_file)


# ============== 核心测试类 ==============

class TurboQuantAPITester:
    """TurboQuant HTTP API 测试器"""
    
    def __init__(self, base_url: str, model_name: str = MODEL_NAME, logger: logging.Logger = None):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.logger = logger or logging.getLogger("TurboQuantTest")
        
        self.completions_url = f"{self.base_url}/v1/completions"
        self.metrics_url = f"{self.base_url}/metrics"
        self.health_url = f"{self.base_url}/health"
        
        self.results: List[TestResult] = []
        
    def check_health(self) -> bool:
        """检查服务是否健康"""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Optional[MemoryMetrics]:
        """获取 vLLM metrics（包含显存信息）"""
        try:
            response = requests.get(self.metrics_url, timeout=10)
            if response.status_code != 200:
                self.logger.warning(f"Failed to get metrics: {response.status_code}")
                return None
            
            metrics_text = response.text
            
            # 解析 Prometheus 格式的 metrics
            memory_used = []
            memory_total = []
            
            for line in metrics_text.split('\n'):
                # 查找 GPU 显存使用（vLLM 的 metrics）
                if 'vllm:gpu_cache_usage_perc' in line:
                    # 这个指标表示 KV cache 使用比例
                    pass
                
                # 或者查找 nvidia_gpu 相关指标
                if 'nvidia_gpu_memory_used_bytes' in line and not line.startswith('#'):
                    try:
                        value = float(line.split()[-1])
                        memory_used.append(value / 1024 / 1024)  # bytes -> MB
                    except:
                        pass
                
                if 'nvidia_gpu_memory_total_bytes' in line and not line.startswith('#'):
                    try:
                        value = float(line.split()[-1])
                        memory_total.append(value / 1024 / 1024)
                    except:
                        pass
            
            return MemoryMetrics(
                gpu_memory_used_mb=memory_used,
                gpu_memory_total_mb=memory_total,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return None
    
    def generate_streaming(self, prompt: str, max_tokens: int = 256, 
                          temperature: float = 0.0) -> Tuple[str, List[float]]:
        """
        流式生成，精确测量 TTFT 和每个 token 的时间
        
        Returns:
            (generated_text, token_timestamps)
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True  # 关键：启用流式输出
        }
        
        token_timestamps = []
        generated_text = ""
        
        try:
            start_time = time.perf_counter()
            first_token_time = None
            
            response = requests.post(
                self.completions_url,
                headers=headers,
                json=data,
                stream=True,
                timeout=120
            )
            
            if response.status_code != 200:
                self.logger.error(f"Generation failed: {response.status_code} - {response.text}")
                return "", []
            
            # 解析流式响应
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        json_str = line[6:]  # 去掉 "data: " 前缀
                        if json_str == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(json_str)
                            token_text = chunk['choices'][0].get('text', '')
                            generated_text += token_text
                            
                            current_time = time.perf_counter()
                            
                            # 记录第一个 token 的时间（TTFT）
                            if first_token_time is None:
                                first_token_time = current_time
                                ttft = (first_token_time - start_time) * 1000  # ms
                                token_timestamps.append(ttft)
                            else:
                                # 后续 token 的时间
                                token_timestamps.append((current_time - first_token_time) * 1000)
                        
                        except json.JSONDecodeError:
                            continue
            
            return generated_text, token_timestamps
            
        except Exception as e:
            self.logger.error(f"Streaming generation error: {e}")
            return "", []
    
    def test_performance(self, prompt: str = TEST_PROMPT_LONG, 
                        max_tokens: int = DEFAULT_MAX_TOKENS,
                        num_runs: int = DEFAULT_NUM_RUNS,
                        config_name: str = "unknown") -> TestResult:
        """
        测试性能指标（多次采样取平均）
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🚀 Performance Test: {num_runs} runs, {max_tokens} max_tokens")
        self.logger.info(f"{'='*70}")
        
        # 存储多次运行的结果
        ttft_list = []
        tpot_list = []
        throughput_list = []
        total_time_list = []
        prompt_tokens_list = []
        output_tokens_list = []
        
        for i in range(num_runs):
            self.logger.info(f"\nRun {i+1}/{num_runs}...")
            
            # 流式生成
            text, timestamps = self.generate_streaming(prompt, max_tokens)
            
            if not timestamps:
                self.logger.error(f"Run {i+1} failed")
                continue
            
            # 计算指标
            ttft = timestamps[0]  # 第一个 token 的时间
            total_time = timestamps[-1] if len(timestamps) > 1 else ttft
            num_tokens = len(timestamps)
            
            # TPOT: (总时间 - TTFT) / (token数 - 1)
            if num_tokens > 1:
                tpot = (total_time - ttft) / (num_tokens - 1)
            else:
                tpot = 0
            
            # Throughput: tokens / 总时间(秒)
            throughput = num_tokens / (total_time / 1000) if total_time > 0 else 0
            
            # 估算 prompt tokens（简单估算：1 token ≈ 4 chars）
            prompt_tokens = len(prompt) // 4
            
            # 记录
            ttft_list.append(ttft)
            tpot_list.append(tpot)
            throughput_list.append(throughput)
            total_time_list.append(total_time)
            prompt_tokens_list.append(prompt_tokens)
            output_tokens_list.append(num_tokens)
            
            self.logger.info(f"  TTFT: {ttft:.2f}ms, TPOT: {tpot:.2f}ms, "
                           f"Throughput: {throughput:.2f} tok/s")
            
            # 短暂停顿，避免过热
            if i < num_runs - 1:
                time.sleep(1)
        
        if not ttft_list:
            self.logger.error("All runs failed!")
            return TestResult(
                config_name=config_name,
                test_type="performance",
                timestamp=datetime.now().isoformat()
            )
        
        # 计算平均值
        perf = PerformanceMetrics(
            ttft_ms=statistics.mean(ttft_list),
            tpot_ms=statistics.mean(tpot_list),
            total_time_ms=statistics.mean(total_time_list),
            throughput_tok_s=statistics.mean(throughput_list),
            prompt_tokens=int(statistics.mean(prompt_tokens_list)),
            output_tokens=int(statistics.mean(output_tokens_list)),
            samples=len(ttft_list)
        )
        
        # 记录统计信息
        self.logger.info(f"\n{'='*70}")
        self.logger.info("📊 Performance Summary (Average of {} runs)".format(len(ttft_list)))
        self.logger.info(f"{'='*70}")
        self.logger.info(f"  TTFT:           {perf.ttft_ms:.2f} ms")
        self.logger.info(f"  TPOT:           {perf.tpot_ms:.2f} ms")
        self.logger.info(f"  Total Time:     {perf.total_time_ms:.2f} ms")
        self.logger.info(f"  Throughput:     {perf.throughput_tok_s:.2f} tokens/s")
        self.logger.info(f"  Prompt Tokens:  {perf.prompt_tokens}")
        self.logger.info(f"  Output Tokens:  {perf.output_tokens}")
        
        # 标准差（稳定性指标）
        if len(ttft_list) > 1:
            self.logger.info(f"\n  Std Dev:")
            self.logger.info(f"    TTFT:       {statistics.stdev(ttft_list):.2f} ms")
            self.logger.info(f"    TPOT:       {statistics.stdev(tpot_list):.2f} ms")
        
        return TestResult(
            config_name=config_name,
            test_type="performance",
            timestamp=datetime.now().isoformat(),
            performance=perf,
            sample_output=text[:200] if 'text' in locals() else ""
        )
    
    def test_memory(self, config_name: str = "unknown") -> TestResult:
        """测试显存占用"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("💾 Memory Test")
        self.logger.info(f"{'='*70}")
        
        # 先进行一轮生成，让显存稳定
        self.logger.info("Warming up...")
        self.generate_streaming(TEST_PROMPT_SHORT, max_tokens=100)
        time.sleep(2)
        
        # 获取显存指标（多次采样）
        memory_readings = []
        for i in range(3):
            mem = self.get_metrics()
            if mem:
                memory_readings.append(mem)
                self.logger.info(f"Sample {i+1}: GPU memory used: {mem.avg_used_mb():.0f} MB")
            time.sleep(1)
        
        if not memory_readings:
            self.logger.warning("Failed to get memory metrics")
            return TestResult(
                config_name=config_name,
                test_type="memory",
                timestamp=datetime.now().isoformat()
            )
        
        # 取平均值
        avg_used = statistics.mean([m.avg_used_mb() for m in memory_readings])
        self.logger.info(f"\nAverage GPU Memory Used: {avg_used:.0f} MB")
        
        return TestResult(
            config_name=config_name,
            test_type="memory",
            timestamp=datetime.now().isoformat(),
            memory=memory_readings[-1]  # 使用最后一次的完整数据
        )
    
    def test_quality(self, config_name: str = "unknown") -> TestResult:
        """简单 QA 质量测试"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("✅ Quality Test (Simple QA)")
        self.logger.info(f"{'='*70}")
        
        correct = 0
        results = []
        
        for question, expected in QA_QUESTIONS:
            self.logger.info(f"\nQ: {question}")
            
            text, _ = self.generate_streaming(
                question, 
                max_tokens=50,
                temperature=0.0
            )
            
            is_correct = expected.lower() in text.lower()
            if is_correct:
                correct += 1
                self.logger.info(f"  ✅ A: {text[:100]}")
            else:
                self.logger.info(f"  ❌ A: {text[:100]} (Expected: {expected})")
            
            results.append({
                "question": question,
                "expected": expected,
                "generated": text,
                "correct": is_correct
            })
        
        accuracy = correct / len(QA_QUESTIONS) if QA_QUESTIONS else 0
        
        self.logger.info(f"\nAccuracy: {accuracy:.1%} ({correct}/{len(QA_QUESTIONS)})")
        
        return TestResult(
            config_name=config_name,
            test_type="quality",
            timestamp=datetime.now().isoformat(),
            qa_correct=correct,
            qa_total=len(QA_QUESTIONS),
            qa_accuracy=accuracy
        )
    
    def run_all_tests(self, config_name: str = "unknown") -> List[TestResult]:
        """运行所有测试"""
        results = []
        
        # 性能测试
        perf_result = self.test_performance(config_name=config_name)
        results.append(perf_result)
        
        # 显存测试
        memory_result = self.test_memory(config_name=config_name)
        results.append(memory_result)
        
        # 质量测试
        quality_result = self.test_quality(config_name=config_name)
        results.append(quality_result)
        
        self.results.extend(results)
        return results
    
    def save_results(self, output_dir: str = "./results") -> str:
        """保存测试结果到 JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_path / f"test_results_{timestamp}.json"
        
        data = {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n💾 Results saved to: {json_file}")
        return str(json_file)
    
    def generate_report(self, output_dir: str = "./results") -> str:
        """生成 Markdown 报告"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"test_report_{timestamp}.md"
        
        lines = [
            "# TurboQuant HTTP API 测试报告",
            f"\n**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**服务地址**: {self.base_url}",
            f"**模型名称**: {self.model_name}",
            "\n---\n"
        ]
        
        # 按测试类型分组
        perf_results = [r for r in self.results if r.test_type == "performance"]
        memory_results = [r for r in self.results if r.test_type == "memory"]
        quality_results = [r for r in self.results if r.test_type == "quality"]
        
        # 性能结果
        if perf_results:
            lines.append("## 性能测试结果\n")
            lines.append("| 配置 | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Tokens |")
            lines.append("|------|-----------|-----------|-------------------|--------|")
            
            for r in perf_results:
                if r.performance:
                    p = r.performance
                    lines.append(f"| {r.config_name} | {p.ttft_ms:.2f} | {p.tpot_ms:.2f} | "
                               f"{p.throughput_tok_s:.2f} | {p.output_tokens} |")
            
            lines.append("")
        
        # 显存结果
        if memory_results:
            lines.append("## 显存测试结果\n")
            lines.append("| 配置 | 平均显存使用 (MB) |")
            lines.append("|------|------------------|")
            
            for r in memory_results:
                if r.memory:
                    lines.append(f"| {r.config_name} | {r.memory.avg_used_mb():.0f} |")
            
            lines.append("")
        
        # 质量结果
        if quality_results:
            lines.append("## 质量测试结果\n")
            lines.append("| 配置 | 准确率 | 正确/总数 |")
            lines.append("|------|--------|----------|")
            
            for r in quality_results:
                lines.append(f"| {r.config_name} | {r.qa_accuracy:.1%} | {r.qa_correct}/{r.qa_total} |")
            
            lines.append("")
        
        # 结论
        lines.extend([
            "## 测试结论\n",
            "- **TTFT**: Time To First Token，首 token 延迟",
            "- **TPOT**: Time Per Output Token，每个输出 token 的平均生成时间",
            "- **Throughput**: 吞吐量，每秒生成的 token 数\n"
        ])
        
        report_text = '\n'.join(lines)
        report_file.write_text(report_text, encoding='utf-8')
        
        self.logger.info(f"📄 Report saved to: {report_file}")
        return str(report_file)


# ============== 主入口 ==============

def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant HTTP API 性能测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
    # 完整测试
    python test_tq_api.py --url http://localhost:8000 --config baseline --test all
    
    # 只测性能（5次采样）
    python test_tq_api.py --url http://localhost:8000 --config baseline --test performance --runs 5
    
    # 对比测试流程：
    # 1. 启动 Baseline: bash start_vllm_qwen35_2b_tq.sh
    # 2. 测试 Baseline: python test_tq_api.py --config baseline --test all
    # 3. 停止服务，启动 TQ: TQ_ENABLED=1 bash start_vllm_qwen35_2b_tq.sh
    # 4. 测试 TQ: python test_tq_api.py --config turboquant --test all
        """
    )
    
    parser.add_argument("--url", default=DEFAULT_URL,
                       help=f"vLLM 服务地址 (默认: {DEFAULT_URL})")
    parser.add_argument("--config", default="unknown",
                       help="配置名称 (baseline/turboquant)，用于报告区分")
    parser.add_argument("--test", default="all",
                       choices=["all", "performance", "memory", "quality"],
                       help="测试类型")
    parser.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS,
                       help=f"性能测试采样次数 (默认: {DEFAULT_NUM_RUNS})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                       help=f"最大生成 token 数 (默认: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--output-dir", default="./results",
                       help="结果输出目录 (默认: ./results)")
    
    args = parser.parse_args()
    
    # 设置日志
    logger, log_file = setup_logging()
    
    logger.info("="*70)
    logger.info("🧪 TurboQuant HTTP API 性能测试")
    logger.info("="*70)
    logger.info(f"服务地址: {args.url}")
    logger.info(f"配置名称: {args.config}")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*70)
    
    # 创建测试器
    tester = TurboQuantAPITester(args.url, MODEL_NAME, logger)
    
    # 检查服务健康
    logger.info("\n检查服务健康...")
    if not tester.check_health():
        logger.error("❌ 服务未就绪，请检查 vLLM 是否已启动")
        logger.error(f"   尝试连接: {args.url}/health")
        sys.exit(1)
    
    logger.info("✅ 服务健康")
    
    # 运行测试
    if args.test == "all":
        tester.run_all_tests(config_name=args.config)
    elif args.test == "performance":
        tester.test_performance(
            num_runs=args.runs,
            max_tokens=args.max_tokens,
            config_name=args.config
        )
    elif args.test == "memory":
        tester.test_memory(config_name=args.config)
    elif args.test == "quality":
        tester.test_quality(config_name=args.config)
    
    # 保存结果
    logger.info("\n" + "="*70)
    tester.save_results(args.output_dir)
    tester.generate_report(args.output_dir)
    
    logger.info("\n✅ 测试完成!")
    logger.info(f"📊 查看日志: {log_file}")


if __name__ == "__main__":
    main()
