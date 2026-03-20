"""
Test harness for factoring algorithms.
Generates semiprimes of increasing bit-size, runs candidate algorithms,
verifies correctness, and measures timing to detect polynomial vs exponential scaling.
"""

import time
import random
import math
from sympy import isprime, nextprime
from typing import Callable, Optional


def generate_prime(bits: int) -> int:
    """Generate a random prime of approximately the given bit-size."""
    low = 1 << (bits - 1)
    high = (1 << bits) - 1
    candidate = random.randint(low, high)
    return nextprime(candidate)


def generate_semiprime(bits: int) -> tuple[int, int, int]:
    """Generate a semiprime n = p * q where n is approximately `bits` bits."""
    half = bits // 2
    p = generate_prime(half)
    q = generate_prime(bits - half)
    while p == q:
        q = generate_prime(bits - half)
    return p * q, min(p, q), max(p, q)


def verify_factors(n: int, factors: list[int]) -> bool:
    """Verify that factors multiply to n and are all prime."""
    if not factors:
        return False
    product = 1
    for f in factors:
        if f < 2:
            return False
        if not isprime(f):
            return False
        product *= f
    return product == n


def time_algorithm(algo: Callable, n: int, timeout: float = 60.0) -> tuple[Optional[list[int]], float]:
    """Run a factoring algorithm with a real timeout using multiprocessing."""
    import multiprocessing as mp
    import queue

    result_q = mp.Queue()

    def _worker(algo, n, q):
        try:
            r = algo(n)
            q.put(r)
        except Exception:
            q.put(None)

    proc = mp.Process(target=_worker, args=(algo, n, result_q))
    start = time.time()
    proc.start()
    proc.join(timeout=timeout)
    elapsed = time.time() - start

    if proc.is_alive():
        proc.kill()
        proc.join()
        return None, elapsed

    try:
        result = result_q.get_nowait()
    except Exception:
        result = None

    return result, elapsed


def run_benchmark(algorithms: dict[str, Callable], bit_sizes: list[int],
                  samples_per_size: int = 5, timeout: float = 60.0):
    """
    Benchmark factoring algorithms across bit sizes.
    Returns dict of {algo_name: {bits: [times]}}
    """
    results = {}
    for name in algorithms:
        results[name] = {}

    for bits in bit_sizes:
        print(f"\n{'='*60}")
        print(f"  Bit size: {bits}")
        print(f"{'='*60}")

        test_cases = []
        for _ in range(samples_per_size):
            n, p, q = generate_semiprime(bits)
            test_cases.append((n, p, q))

        for name, algo in algorithms.items():
            times = []
            correct = 0
            for n, p, q in test_cases:
                factors, elapsed = time_algorithm(algo, n, timeout)
                if factors is not None and verify_factors(n, factors):
                    times.append(elapsed)
                    correct += 1
                else:
                    times.append(None)

            results[name][bits] = times
            valid_times = [t for t in times if t is not None]
            if valid_times:
                avg = sum(valid_times) / len(valid_times)
                print(f"  {name:30s} | {correct}/{samples_per_size} correct | avg {avg:.6f}s")
            else:
                print(f"  {name:30s} | {correct}/{samples_per_size} correct | all failed/timeout")

    return results


def analyze_scaling(results: dict, algo_name: str):
    """
    Analyze whether an algorithm's timing suggests polynomial or exponential scaling.
    Fits log(time) vs bits and log(time) vs log(bits).
    """
    bit_sizes = sorted(results[algo_name].keys())
    avg_times = []
    valid_bits = []

    for bits in bit_sizes:
        times = [t for t in results[algo_name][bits] if t is not None]
        if times:
            avg_times.append(sum(times) / len(times))
            valid_bits.append(bits)

    if len(valid_bits) < 3:
        print(f"\n{algo_name}: Not enough data points for scaling analysis")
        return

    print(f"\n{'='*60}")
    print(f"  Scaling analysis: {algo_name}")
    print(f"{'='*60}")

    # Print raw data
    for b, t in zip(valid_bits, avg_times):
        print(f"  {b:4d} bits: {t:.6f}s")

    # Check ratios between consecutive sizes
    print(f"\n  Growth ratios (time[i+1] / time[i]):")
    for i in range(1, len(avg_times)):
        if avg_times[i-1] > 0:
            ratio = avg_times[i] / avg_times[i-1]
            bit_ratio = valid_bits[i] / valid_bits[i-1]
            print(f"  {valid_bits[i-1]:4d} -> {valid_bits[i]:4d} bits: "
                  f"time ratio = {ratio:.2f}x, bit ratio = {bit_ratio:.2f}x")

    # Polynomial fit: log(time) vs log(bits) -> time ~ bits^k
    if all(t > 0 for t in avg_times):
        log_bits = [math.log(b) for b in valid_bits]
        log_times = [math.log(t) for t in avg_times]

        # Simple linear regression
        n = len(log_bits)
        sum_x = sum(log_bits)
        sum_y = sum(log_times)
        sum_xy = sum(x*y for x, y in zip(log_bits, log_times))
        sum_xx = sum(x*x for x in log_bits)

        denom = n * sum_xx - sum_x * sum_x
        if denom != 0:
            poly_slope = (n * sum_xy - sum_x * sum_y) / denom
            print(f"\n  Polynomial fit: time ~ bits^{poly_slope:.2f}")
            if poly_slope < 10:
                print(f"  -> POLYNOMIAL scaling (degree ~{poly_slope:.1f})")
            else:
                print(f"  -> Likely EXPONENTIAL (poly degree too high)")

        # Exponential fit: log(time) vs bits -> time ~ 2^(k*bits)
        sum_x2 = sum(valid_bits)
        sum_xy2 = sum(b*lt for b, lt in zip(valid_bits, log_times))
        sum_xx2 = sum(b*b for b in valid_bits)
        denom2 = n * sum_xx2 - sum_x2 * sum_x2
        if denom2 != 0:
            exp_slope = (n * sum_xy2 - sum_x2 * sum_y) / denom2
            print(f"  Exponential fit: time ~ e^({exp_slope:.4f} * bits)")
            doubling_bits = math.log(2) / exp_slope if exp_slope > 0 else float('inf')
            print(f"  -> Time doubles every {doubling_bits:.1f} bits")
