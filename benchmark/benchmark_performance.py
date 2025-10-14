import os
import warnings

warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
import pandas as pd
import numpy as np
import time
import psutil
import gc
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
import warnings
from scipy import stats
import ctypes
import threading
import sys

warnings.filterwarnings('ignore')
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
app_dir = os.path.join(parent_dir, 'app')
sys.path.append(app_dir)
from minhash_clustering.cluster_in_mem import MemMinhashLSHClustering
from minhash_clustering.cluster_streaming import StreamingMinHashLSHClustering


@dataclass
class ScalabilityResult:
    dataset_size: int
    method: str
    processing_time: float
    peak_memory_mb: float
    num_clusters: int
    memory_per_doc_mb: float
    throughput_docs_per_sec: float
    memory_growth_rate: float = None
    time_growth_rate: float = None
    success: bool = True
    failure_reason: str = None
    peak_rss_mb: float = 0
    memory_before_mb: float = 0


class EnhancedScalabilityProofGenerator:
    def __init__(self, csv_file: str, memory_limit_percent: float = 0.60, log_file: str = None):
        self.csv_file = csv_file
        self.system_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.memory_limit_percent = memory_limit_percent
        self.safe_memory_limit_gb = self.system_memory_gb * memory_limit_percent
        self.safe_memory_limit_mb = self.safe_memory_limit_gb * 1024
        self.results = []
        self.real_documents = []
        self.optimized_failed = False
        self.optimized_failure_size = None
        self.log_file = log_file
        self._log(f"System Memory: {self.system_memory_gb:.1f} GB")
        self._log(f"Memory Limit ({memory_limit_percent * 100:.0f}%): {self.safe_memory_limit_gb:.1f} GB")
        self.load_real_data()

    def _log(self, message: str):
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')

    def load_real_data(self):
        try:
            self._log(f"Loading data from {self.csv_file}...")
            df = pd.read_csv(self.csv_file)
            text_columns = [col for col in df.columns if 'text' in col.lower()]
            if not text_columns:
                raise ValueError("No 'text' column found in CSV")
            text_col = text_columns[0]
            self._log(f"Using text column: '{text_col}'")
            texts = df[text_col].dropna().astype(str).tolist()
            self.real_documents = [text.strip() for text in texts if text.strip()]
            self._log(f"Loaded {len(self.real_documents):,} valid documents")
            lengths = [len(text) for text in self.real_documents[:1000]]
            self._log(f"  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {np.mean(lengths):.1f}")
        except Exception as e:
            raise Exception(f"Failed to load {self.csv_file}: {e}")

    def force_memory_cleanup(self):
        gc.collect()
        gc.collect()
        try:
            if hasattr(ctypes, 'CDLL'):
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except:
                    pass
        except:
            pass
        time.sleep(0.5)

    def get_memory_usage(self) -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)

    def generate_test_sizes(self, start=1000, growth_factor=2, max_size=None):
        sizes = []
        current = start
        if max_size is None:
            max_size = len(self.real_documents)
        while current < max_size:
            sizes.append(int(current))
            current *= growth_factor
        if sizes[-1] != max_size:
            sizes.append(max_size)  # ensure full dataset is tested
        return sizes

    def create_dataset(self, size: int) -> List[str]:
        if size <= len(self.real_documents):
            return random.sample(self.real_documents, size)
        else:
            result = []
            base_docs = self.real_documents.copy()
            while len(result) < size:
                if len(result) + len(base_docs) <= size:
                    result.extend(base_docs)
                else:
                    remaining = size - len(result)
                    result.extend(random.sample(base_docs, remaining))
            return result

    def estimate_memory_needs(self, size: int, method: str) -> float:
        if not self.results:
            return size * 0.002 if method == "optimized" else max(50, size * 0.0005)
        method_results = [r for r in self.results if r.method == method and r.success]
        if not method_results:
            return size * 0.002 if method == "optimized" else size * 0.0005
        largest = max(method_results, key=lambda x: x.dataset_size)
        size_ratio = size / largest.dataset_size
        if method == "optimized":
            return largest.peak_memory_mb * (size_ratio ** 1.6)
        else:
            return largest.peak_memory_mb * (size_ratio ** 1.1)

    def run_memory_monitored_test(self, size: int, method: str) -> ScalabilityResult:
        self._log(f"\n{'=' * 60}")
        self._log(f"Testing {method.upper()} method with {size:,} documents")
        self._log(f"{'=' * 60}")
        estimated_memory = self.estimate_memory_needs(size, method)
        self._log(f"Estimated memory needed: {estimated_memory:.1f} MB")
        self._log(f"Memory limit: {self.safe_memory_limit_mb:.1f} MB")
        if estimated_memory > self.safe_memory_limit_mb:
            self._log(f"SKIPPING: Estimated memory exceeds safe limit")
            return ScalabilityResult(
                dataset_size=size, method=method, processing_time=0,
                peak_memory_mb=0, num_clusters=0, memory_per_doc_mb=0,
                throughput_docs_per_sec=0, success=False,
                failure_reason=f"Estimated memory ({estimated_memory:.0f}MB) exceeds limit ({self.safe_memory_limit_mb:.0f}MB)"
            )
        self.force_memory_cleanup()
        memory_before = self.get_memory_usage()
        self._log(f"Memory before test: {memory_before:.1f} MB")
        try:
            self._log("Generating dataset...")
            documents = self.create_dataset(size)
            self._log(f"Dataset created: {len(documents):,} documents")
        except Exception as e:
            return ScalabilityResult(
                dataset_size=size, method=method, processing_time=0,
                peak_memory_mb=0, num_clusters=0, memory_per_doc_mb=0,
                throughput_docs_per_sec=0, success=False,
                failure_reason=f"Dataset creation failed: {str(e)}"
            )
        peak_memory = memory_before
        memory_exceeded = False

        def memory_monitor():
            nonlocal peak_memory, memory_exceeded
            while True:
                try:
                    current_memory = self.get_memory_usage()
                    peak_memory = max(peak_memory, current_memory)
                    if current_memory > self.safe_memory_limit_mb:
                        memory_exceeded = True
                        self._log(f"WARNING: Memory limit exceeded! Current: {current_memory:.1f}MB")
                        return
                    time.sleep(0.5)
                except:
                    return

        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
        try:
            start_time = time.time()
            self._log("Starting clustering...")
            if method == "optimized":
                batch_size = min(10000, max(1000, size // 10))
                self._log(f"Using batch size: {batch_size:,}")
                clustering_service = MemMinhashLSHClustering(
                    threshold=0.3, shingle_size=6
                )
                clustering_service.batch_size = batch_size
                clustered_docs = clustering_service.cluster_documents(documents)
                num_clusters = len(set(doc.cluster_id for doc in clustered_docs))
            elif method == "streaming":
                chunk_size = min(20000, max(2000, size // 8))
                self._log(f"Using chunk size: {chunk_size:,}")
                clustering_service = StreamingMinHashLSHClustering(
                    threshold=0.3, shingle_size=6, chunk_size=chunk_size
                )
                cluster_results = clustering_service.cluster_streaming(documents)
                num_clusters = len(set(cluster_results.values()))
                clustering_service.cleanup()
            processing_time = time.time() - start_time
            if memory_exceeded:
                return ScalabilityResult(
                    dataset_size=size, method=method, processing_time=processing_time,
                    peak_memory_mb=peak_memory, num_clusters=0, memory_per_doc_mb=0,
                    throughput_docs_per_sec=0, success=False,
                    failure_reason=f"Memory limit exceeded during processing (peak: {peak_memory:.1f}MB)"
                )
            memory_used = max(10, peak_memory - memory_before)
            throughput = size / processing_time if processing_time > 0 else 0
            result = ScalabilityResult(
                dataset_size=size, method=method, processing_time=processing_time,
                peak_memory_mb=memory_used, num_clusters=num_clusters,
                memory_per_doc_mb=memory_used / size,
                throughput_docs_per_sec=throughput,
                success=True, peak_rss_mb=peak_memory,
                memory_before_mb=memory_before
            )
            self._log(f"SUCCESS: {processing_time:.2f}s, {memory_used:.1f}MB used, {num_clusters:,} clusters")
            self._log(f"Throughput: {throughput:.1f} docs/sec")
            self._log(f"Memory efficiency: {memory_used / size:.4f} MB/doc")
            return result
        except MemoryError as e:
            processing_time = time.time() - start_time
            return ScalabilityResult(
                dataset_size=size, method=method, processing_time=processing_time,
                peak_memory_mb=peak_memory, num_clusters=0, memory_per_doc_mb=0,
                throughput_docs_per_sec=0, success=False,
                failure_reason=f"MemoryError: {str(e)}"
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return ScalabilityResult(
                dataset_size=size, method=method, processing_time=processing_time,
                peak_memory_mb=peak_memory, num_clusters=0, memory_per_doc_mb=0,
                throughput_docs_per_sec=0, success=False,
                failure_reason=f"Error: {str(e)[:200]}"
            )
        finally:
            try:
                del documents
                if 'clustering_service' in locals():
                    del clustering_service
                if 'clustered_docs' in locals():
                    del clustered_docs
                if 'cluster_results' in locals():
                    del cluster_results
            except:
                pass
            self.force_memory_cleanup()

    def run_breaking_point_analysis(self) -> Dict:
        self._log("\n" + "=" * 80)
        self._log("BREAKING POINT ANALYSIS - FINDING REAL LIMITS")
        self._log("=" * 80)
        test_sizes = self.generate_test_sizes(start=1000, growth_factor=2, max_size=len(self.real_documents))
        self._log(f"Generated test sizes: {test_sizes}")
        optimized_results = []
        streaming_results = []
        for size in test_sizes:
            self._log(f"\n--- TESTING SIZE: {size:,} DOCUMENTS ---")
            if not self.optimized_failed:
                opt_result = self.run_memory_monitored_test(size, "optimized")
                optimized_results.append(opt_result)
                self.results.append(opt_result)
                if not opt_result.success:
                    self.optimized_failed = True
                    self.optimized_failure_size = size
                    self._log(f"\nOptimized method failed at {size:,} documents")
                    self._log(f"Failure reason: {opt_result.failure_reason}")
                    break
                else:
                    self._log(f"Optimized method succeeded at {size:,}")
            stream_result = self.run_memory_monitored_test(size, "streaming")
            streaming_results.append(stream_result)
            self.results.append(stream_result)
            if not stream_result.success:
                self._log(f"Streaming method also failed at {size:,}")
                self._log(f"Failure reason: {stream_result.failure_reason}")
                break
            else:
                self._log(f"Streaming method succeeded at {size:,}")
        return {
            "optimized_results": optimized_results,
            "streaming_results": streaming_results,
            "optimized_breaking_point": self.optimized_failure_size,
            "max_successful_optimized": max([r.dataset_size for r in optimized_results if r.success], default=0),
            "max_successful_streaming": max([r.dataset_size for r in streaming_results if r.success], default=0)
        }

    def calculate_complexity_metrics(self) -> Dict:
        if len(self.results) < 4:
            return {"error": "Insufficient data for complexity analysis"}
        opt_successful = [r for r in self.results if r.method == "optimized" and r.success]
        stream_successful = [r for r in self.results if r.method == "streaming" and r.success]
        if len(opt_successful) < 3 or len(stream_successful) < 3:
            return {"error": "Need at least 3 successful results per method"}

        def fit_power_law(sizes, values):
            try:
                log_sizes = np.log10(sizes)
                log_values = np.log10(values)
                coeffs = np.polyfit(log_sizes, log_values, 1)
                return coeffs[0]
            except:
                return None

        opt_sizes = np.array([r.dataset_size for r in opt_successful])
        opt_times = np.array([r.processing_time for r in opt_successful])
        opt_memories = np.array([r.peak_memory_mb for r in opt_successful])
        stream_sizes = np.array([r.dataset_size for r in stream_successful])
        stream_times = np.array([r.processing_time for r in stream_successful])
        stream_memories = np.array([r.peak_memory_mb for r in stream_successful])
        return {
            "optimized_time_complexity": fit_power_law(opt_sizes, opt_times),
            "optimized_memory_complexity": fit_power_law(opt_sizes, opt_memories),
            "streaming_time_complexity": fit_power_law(stream_sizes, stream_times),
            "streaming_memory_complexity": fit_power_law(stream_sizes, stream_memories)
        }

    def statistical_analysis(self) -> Dict:
        opt_successful = [r for r in self.results if r.method == "optimized" and r.success]
        stream_successful = [r for r in self.results if r.method == "streaming" and r.success]
        if len(opt_successful) < 3 or len(stream_successful) < 3:
            return {"error": "Insufficient successful results for statistical analysis"}
        opt_memory_per_doc = [r.memory_per_doc_mb for r in opt_successful]
        stream_memory_per_doc = [r.memory_per_doc_mb for r in stream_successful]
        opt_throughput = [r.throughput_docs_per_sec for r in opt_successful]
        stream_throughput = [r.throughput_docs_per_sec for r in stream_successful]
        try:
            memory_stat, memory_p = stats.mannwhitneyu(
                stream_memory_per_doc, opt_memory_per_doc, alternative='less'
            )
            throughput_stat, throughput_p = stats.mannwhitneyu(
                stream_throughput, opt_throughput, alternative='greater'
            )
        except:
            memory_p = 1.0
            throughput_p = 1.0
        return {
            "memory_efficiency": {
                "streaming_mean_mb_per_doc": np.mean(stream_memory_per_doc),
                "optimized_mean_mb_per_doc": np.mean(opt_memory_per_doc),
                "streaming_std": np.std(stream_memory_per_doc),
                "optimized_std": np.std(opt_memory_per_doc),
                "p_value": memory_p,
                "streaming_more_efficient": np.mean(stream_memory_per_doc) < np.mean(opt_memory_per_doc),
                "efficiency_improvement": (np.mean(opt_memory_per_doc) - np.mean(stream_memory_per_doc)) / np.mean(
                    opt_memory_per_doc) * 100
            },
            "throughput": {
                "streaming_mean_docs_per_sec": np.mean(stream_throughput),
                "optimized_mean_docs_per_sec": np.mean(opt_throughput),
                "streaming_std": np.std(stream_throughput),
                "optimized_std": np.std(opt_throughput),
                "p_value": throughput_p
            }
        }

    def generate_comprehensive_report(self, breaking_point_data: Dict,
                                      complexity_data: Dict,
                                      statistical_data: Dict) -> str:
        report = []
        report.append("COMPREHENSIVE SCALABILITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append("SYSTEM CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Total System Memory: {self.system_memory_gb:.1f} GB")
        report.append(f"Memory Limit (60%): {self.safe_memory_limit_gb:.1f} GB")
        report.append(f"CPU Cores: {psutil.cpu_count()}")
        report.append(f"Dataset: {self.csv_file}")
        report.append(f"Available Documents: {len(self.real_documents):,}")
        report.append("")
        report.append("BREAKING POINT RESULTS")
        report.append("-" * 40)
        max_opt = breaking_point_data.get("max_successful_optimized", 0)
        max_stream = breaking_point_data.get("max_successful_streaming", 0)
        opt_break = breaking_point_data.get("optimized_breaking_point")
        if max_opt > 0:
            opt_results = [r for r in self.results if r.method == "optimized" and r.success]
            largest_opt = max(opt_results, key=lambda x: x.dataset_size)
            report.append(f"Optimized Method:")
            report.append(f"  - Maximum successful dataset: {max_opt:,} documents")
            report.append(f"  - Peak memory at max size: {largest_opt.peak_memory_mb:.1f} MB")
            report.append(f"  - Memory efficiency: {largest_opt.memory_per_doc_mb:.6f} MB/doc")
            report.append(f"  - Processing time: {largest_opt.processing_time:.2f} seconds")
        if opt_break:
            report.append(f"  - FAILED at: {opt_break:,} documents")
        else:
            report.append(f"  - No failure point found within tested range")
        report.append("")
        if max_stream > 0:
            stream_results = [r for r in self.results if r.method == "streaming" and r.success]
            largest_stream = max(stream_results, key=lambda x: x.dataset_size)
            report.append(f"Streaming Method:")
            report.append(f"  - Maximum successful dataset: {max_stream:,} documents")
            report.append(f"  - Peak memory at max size: {largest_stream.peak_memory_mb:.1f} MB")
            report.append(f"  - Memory efficiency: {largest_stream.memory_per_doc_mb:.6f} MB/doc")
            report.append(f"  - Processing time: {largest_stream.processing_time:.2f} seconds")
        report.append("")
        if max_opt > 0 and max_stream > 0:
            scalability_factor = max_stream / max_opt if max_opt > 0 else float('inf')
            report.append("SCALABILITY COMPARISON")
            report.append("-" * 40)
            if scalability_factor > 1:
                report.append(f"Streaming method processed {scalability_factor:.1f}x larger datasets")
            else:
                report.append(f"Both methods reached similar maximum sizes")
        if "optimized_time_complexity" in complexity_data:
            report.append("")
            report.append("EMPIRICAL COMPLEXITY ANALYSIS")
            report.append("-" * 40)
            report.append(f"Time Complexity Exponents:")
            report.append(f"  - Optimized: O(n^{complexity_data['optimized_time_complexity']:.2f})")
            report.append(f"  - Streaming: O(n^{complexity_data['streaming_time_complexity']:.2f})")
            report.append(f"Memory Complexity Exponents:")
            report.append(f"  - Optimized: O(n^{complexity_data['optimized_memory_complexity']:.2f})")
            report.append(f"  - Streaming: O(n^{complexity_data['streaming_memory_complexity']:.2f})")
        if "memory_efficiency" in statistical_data:
            report.append("")
            report.append("STATISTICAL ANALYSIS")
            report.append("-" * 40)
            mem_data = statistical_data["memory_efficiency"]
            report.append(f"Memory Efficiency:")
            report.append(
                f"  - Optimized: {mem_data['optimized_mean_mb_per_doc']:.6f} ± {mem_data['optimized_std']:.6f} MB/doc")
            report.append(
                f"  - Streaming: {mem_data['streaming_mean_mb_per_doc']:.6f} ± {mem_data['streaming_std']:.6f} MB/doc")
            if mem_data['streaming_more_efficient']:
                report.append(f"  - Streaming is {mem_data['efficiency_improvement']:.1f}% more memory efficient")
            report.append(f"  - Statistical significance (p-value): {mem_data['p_value']:.6f}")
        report.append("")
        report.append("CONCLUSIONS")
        report.append("-" * 40)
        if opt_break and max_stream > opt_break:
            report.append("1. SCALABILITY: Streaming method handles larger datasets")
            report.append("2. ROBUSTNESS: Streaming method more resistant to memory issues")
            report.append("3. PRODUCTION READY: Streaming recommended for large-scale deployments")
        if "memory_efficiency" in statistical_data and statistical_data["memory_efficiency"][
            "streaming_more_efficient"]:
            report.append("4. MEMORY EFFICIENCY: Streaming uses memory more efficiently")
        if not opt_break:
            report.append("1. INCONCLUSIVE: No clear breaking point found for optimized method")
            report.append("2. RECOMMENDATION: Extend testing to larger dataset sizes")
        return "\n".join(report)

    def create_visualizations(self, output_folder: str):
        if not self.results:
            self._log("No results to visualize")
            return
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            self._log("No successful results to visualize")
            return
        df = pd.DataFrame([{
            'Dataset Size': r.dataset_size,
            'Method': r.method,
            'Processing Time (s)': r.processing_time,
            'Peak Memory (MB)': r.peak_memory_mb,
            'Memory per Doc (MB)': r.memory_per_doc_mb,
            'Throughput (docs/s)': r.throughput_docs_per_sec,
        } for r in successful_results])

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scalability Analysis: Real Breaking Point Testing', fontsize=16, fontweight='bold')

        colorblind_colors = {'optimized': '#0173B2', 'streaming': '#DE8F05'}

        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]
            color = colorblind_colors[method]
            marker = 'o' if method == 'optimized' else 's'
            linestyle = '-' if method == 'optimized' else '--'
            axes[0, 0].loglog(method_data['Dataset Size'], method_data['Peak Memory (MB)'],
                              marker=marker, linestyle=linestyle, label=f'{method.title()}',
                              linewidth=2, markersize=8, color=color)
        axes[0, 0].axhline(y=self.safe_memory_limit_mb, color='red', linestyle=':',
                           label=f'Memory Limit ({self.memory_limit_percent * 100:.0f}%)', linewidth=2)
        axes[0, 0].set_xlabel('Dataset Size (documents)')
        axes[0, 0].set_ylabel('Peak Memory Usage (MB)')
        axes[0, 0].set_title('Memory Scaling Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]
            color = colorblind_colors[method]
            marker = 'o' if method == 'optimized' else 's'
            linestyle = '-' if method == 'optimized' else '--'
            axes[0, 1].loglog(method_data['Dataset Size'], method_data['Processing Time (s)'],
                              marker=marker, linestyle=linestyle, label=f'{method.title()}',
                              linewidth=2, markersize=8, color=color)
        axes[0, 1].set_xlabel('Dataset Size (documents)')
        axes[0, 1].set_ylabel('Processing Time (seconds)')
        axes[0, 1].set_title('Time Scaling Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]
            color = colorblind_colors[method]
            marker = 'o' if method == 'optimized' else 's'
            linestyle = '-' if method == 'optimized' else '--'
            axes[1, 0].semilogx(method_data['Dataset Size'], method_data['Memory per Doc (MB)'],
                                marker=marker, linestyle=linestyle, label=f'{method.title()}',
                                linewidth=2, markersize=8, color=color)
        axes[1, 0].set_xlabel('Dataset Size (documents)')
        axes[1, 0].set_ylabel('Memory per Document (MB)')
        axes[1, 0].set_title('Memory Efficiency Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'scalability_analysis.png'), dpi=300)
        plt.close()

        fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=False)
        metrics = ["Processing Time (s)", "Throughput (docs/s)", "Peak Memory (MB)", "Memory per Doc (MB)"]

        for ax, metric in zip(axes, metrics):
            for method in df['Method'].unique():
                method_data = df[df['Method'] == method]
                color = colorblind_colors[method]
                marker = 'o' if method == 'optimized' else 's'
                linestyle = '-' if method == 'optimized' else '--'
                ax.plot(method_data['Dataset Size'], method_data[metric],
                        marker=marker, linestyle=linestyle, label=method.title(),
                        color=color, linewidth=2, markersize=8)
            ax.set_ylabel(metric)
            ax.set_xlabel("Dataset Size (documents)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)

        plt.suptitle("Performance Comparison: Optimized vs Streaming", fontsize=14, weight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(output_folder, 'performance_comparison.png'), dpi=300)
        plt.close()


if __name__ == "__main__":
    # csv_path = "../test.csv"

    csv_path = "../papyri_sentences_no_gaps_clean.csv"
    output_folder = os.path.join(script_dir, "benchmark_performance_results")
    os.makedirs(output_folder, exist_ok=True)
    log_file = os.path.join(output_folder, "scalability_log.txt")

    with open(log_file, 'w') as f:
        f.write("Scalability Benchmark Log\n" + "=" * 50 + "\n")

    benchmark = EnhancedScalabilityProofGenerator(csv_path, log_file=log_file)
    breaking_points = benchmark.run_breaking_point_analysis()
    complexity = benchmark.calculate_complexity_metrics()
    stats = benchmark.statistical_analysis()
    report = benchmark.generate_comprehensive_report(breaking_points, complexity, stats)

    report_path = os.path.join(output_folder, "scalability_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    benchmark._log(f"Report saved to: {report_path}")

    benchmark.create_visualizations(output_folder)
    benchmark._log(f"Visualizations saved to: {output_folder}")