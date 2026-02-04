"""
Computational Complexity Benchmark
==================================

Benchmarks for measuring execution time and memory usage of GAPoT
compared to traditional methods across varying numbers of harmonics.

This script generates the results referenced in the paper's discussion
of computational complexity (Point 1.3).

Usage:
    python computational_complexity.py [--max-harmonics N] [--output FILE]
"""

import numpy as np
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gapot import GeometricPower
from gapot.traditional import IEEE1459Power, PQTheory
from gapot.utils import generate_distorted_signal


def generate_test_signals(n_harmonics: int, f1: float = 50.0, 
                          fs: float = 50000.0, cycles: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Generate test signals with specified number of harmonics."""
    T = cycles / f1
    t = np.linspace(0, T, int(T * fs), endpoint=False)
    
    # Fundamental
    U1, I1 = 230.0, 10.0
    phi1 = np.pi / 6
    
    u = U1 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t)
    i = I1 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t - phi1)
    
    # Add harmonics (typical distribution: magnitude decreases with order)
    for h in range(2, n_harmonics + 1):
        # Skip triplen harmonics in balanced three-phase assumption
        if h % 3 == 0:
            continue
            
        U_h = U1 * 0.1 / h  # Voltage decreases as 1/h
        I_h = I1 * 0.3 / h  # Current decreases as 1/h
        phi_h = np.random.uniform(0, 2*np.pi)  # Random phase
        
        u += U_h * np.sqrt(2) * np.sin(2 * np.pi * h * f1 * t)
        i += I_h * np.sqrt(2) * np.sin(2 * np.pi * h * f1 * t - phi_h)
    
    return u, i


def benchmark_method(method_fn, u: np.ndarray, i: np.ndarray, 
                     n_runs: int = 10) -> Dict[str, float]:
    """Benchmark a single method over multiple runs."""
    times = []
    
    for _ in range(n_runs):
        start = time.perf_counter()
        result = method_fn(u, i)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }


def run_benchmarks(max_harmonics: int = 50, 
                   harmonic_steps: List[int] = None) -> Dict:
    """Run complete benchmark suite."""
    
    if harmonic_steps is None:
        harmonic_steps = [1, 5, 10, 15, 20, 25, 30, 40, 50]
    
    harmonic_steps = [h for h in harmonic_steps if h <= max_harmonics]
    
    results = {
        'harmonic_counts': harmonic_steps,
        'gapot': {'times': [], 'std': []},
        'ieee1459': {'times': [], 'std': []},
        'pq_theory': {'times': [], 'std': []},
        'parameters': {
            'f1': 50.0,
            'fs': 50000.0,
            'cycles': 4,
            'n_runs': 10
        }
    }
    
    print("=" * 70)
    print("COMPUTATIONAL COMPLEXITY BENCHMARK")
    print("=" * 70)
    print(f"{'Harmonics':>10} {'GAPoT (ms)':>15} {'IEEE 1459 (ms)':>15} {'p-q (ms)':>15}")
    print("-" * 70)
    
    for n_harm in harmonic_steps:
        # Generate test signals
        u, i = generate_test_signals(n_harm)
        
        # Benchmark GAPoT
        gapot_result = benchmark_method(
            lambda u, i: GeometricPower(u, i, f1=50.0, fs=50000.0),
            u, i
        )
        
        # Benchmark IEEE 1459
        ieee_result = benchmark_method(
            lambda u, i: IEEE1459Power(u, i, f1=50.0, fs=50000.0),
            u, i
        )
        
        # Benchmark p-q Theory
        pq_result = benchmark_method(
            lambda u, i: PQTheory(u, i, fs=50000.0),
            u, i
        )
        
        # Store results
        results['gapot']['times'].append(gapot_result['mean_time'] * 1000)
        results['gapot']['std'].append(gapot_result['std_time'] * 1000)
        
        results['ieee1459']['times'].append(ieee_result['mean_time'] * 1000)
        results['ieee1459']['std'].append(ieee_result['std_time'] * 1000)
        
        results['pq_theory']['times'].append(pq_result['mean_time'] * 1000)
        results['pq_theory']['std'].append(pq_result['std_time'] * 1000)
        
        print(f"{n_harm:>10} {gapot_result['mean_time']*1000:>15.3f} "
              f"{ieee_result['mean_time']*1000:>15.3f} "
              f"{pq_result['mean_time']*1000:>15.3f}")
    
    print("=" * 70)
    
    # Calculate complexity ratios
    if len(harmonic_steps) >= 2:
        # Approximate scaling: time = a * n^b
        # log(time) = log(a) + b * log(n)
        log_n = np.log(harmonic_steps)
        
        for method in ['gapot', 'ieee1459', 'pq_theory']:
            log_t = np.log(np.array(results[method]['times']) + 1e-10)
            coeffs = np.polyfit(log_n, log_t, 1)
            results[method]['scaling_exponent'] = coeffs[0]
    
    return results


def memory_benchmark(max_harmonics: int = 50) -> Dict:
    """Estimate memory usage for different harmonic counts."""
    try:
        from memory_profiler import memory_usage
        has_memory_profiler = True
    except ImportError:
        has_memory_profiler = False
        print("Warning: memory_profiler not installed. Skipping memory benchmark.")
        return {}
    
    harmonic_steps = [1, 10, 25, 50]
    harmonic_steps = [h for h in harmonic_steps if h <= max_harmonics]
    
    results = {'harmonic_counts': harmonic_steps, 'memory_mb': []}
    
    print("\nMEMORY USAGE BENCHMARK")
    print("-" * 40)
    
    for n_harm in harmonic_steps:
        u, i = generate_test_signals(n_harm)
        
        def create_gapot():
            gp = GeometricPower(u, i, f1=50.0, fs=50000.0)
            return gp.S
        
        mem = memory_usage((create_gapot,), max_usage=True)
        results['memory_mb'].append(mem)
        
        print(f"Harmonics: {n_harm:>3}, Memory: {mem:.2f} MB")
    
    return results


def save_results(results: Dict, output_path: str):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def plot_results(results: Dict, output_dir: str = None):
    """Generate benchmark plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plots.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_harm = results['harmonic_counts']
    
    # Execution time plot
    ax1 = axes[0]
    ax1.errorbar(n_harm, results['gapot']['times'], 
                 yerr=results['gapot']['std'], 
                 marker='o', label='GAPoT', capsize=3)
    ax1.errorbar(n_harm, results['ieee1459']['times'], 
                 yerr=results['ieee1459']['std'],
                 marker='s', label='IEEE 1459', capsize=3)
    ax1.errorbar(n_harm, results['pq_theory']['times'], 
                 yerr=results['pq_theory']['std'],
                 marker='^', label='p-q Theory', capsize=3)
    
    ax1.set_xlabel('Number of Harmonics')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Computational Time vs. Number of Harmonics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Scaling analysis
    ax2 = axes[1]
    for method, label in [('gapot', 'GAPoT'), 
                           ('ieee1459', 'IEEE 1459'), 
                           ('pq_theory', 'p-q Theory')]:
        if 'scaling_exponent' in results[method]:
            exp = results[method]['scaling_exponent']
            ax2.bar(label, exp)
    
    ax2.set_ylabel('Scaling Exponent')
    ax2.set_title('Computational Complexity Scaling (time ∝ n^exponent)')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='O(n)')
    ax2.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='O(n²)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'benchmark_results.png', dpi=150)
        print(f"Plot saved to: {Path(output_dir) / 'benchmark_results.png'}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='GAPoT Computational Complexity Benchmark')
    parser.add_argument('--max-harmonics', type=int, default=50,
                        help='Maximum number of harmonics to test')
    parser.add_argument('--output', type=str, default='results/benchmark_results.json',
                        help='Output file path')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    parser.add_argument('--memory', action='store_true',
                        help='Include memory benchmarks')
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmarks(max_harmonics=args.max_harmonics)
    
    # Optional memory benchmark
    if args.memory:
        mem_results = memory_benchmark(max_harmonics=args.max_harmonics)
        results['memory'] = mem_results
    
    # Save results
    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path))
    
    # Generate plots
    if args.plot:
        plot_results(results, str(output_path.parent))
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for method in ['gapot', 'ieee1459', 'pq_theory']:
        if 'scaling_exponent' in results[method]:
            exp = results[method]['scaling_exponent']
            print(f"{method:>12}: scaling exponent = {exp:.2f} (O(n^{exp:.1f}))")
    
    return results


if __name__ == "__main__":
    main()
