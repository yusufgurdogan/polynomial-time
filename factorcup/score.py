#!/usr/bin/env python3
"""
FactorCup Scorer — Measures scaling behavior.

Run: python score.py              # score your entry
     python score.py --baselines  # also score baseline algorithms

Scoring:
    Primary:   max_bits     (highest bit size where 3/5 instances solved)
    Secondary: exponent     (best-fit k in t(n) = c * n^k)
    Tertiary:  time_at_max  (median time at max_bits)

Do not modify this file.
"""

import sys
import time
import math
import signal
import argparse
import multiprocessing
from generate import generate_test_suite, SCORING_BIT_SIZES, CASES_PER_SIZE, TIMEOUT_SECONDS, ALIVE_THRESHOLD


def run_with_timeout(func, args, timeout):
    """Run func(args) with a hard timeout. Returns (result, elapsed) or (None, elapsed)."""
    result_queue = multiprocessing.Queue()

    def worker(q, fn, a):
        try:
            t0 = time.time()
            r = fn(*a)
            elapsed = time.time() - t0
            q.put((r, elapsed))
        except Exception as e:
            q.put((None, time.time() - t0))

    t_start = time.time()
    proc = multiprocessing.Process(target=worker, args=(result_queue, func, args))
    proc.start()
    proc.join(timeout=timeout + 1)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return None, timeout

    if not result_queue.empty():
        return result_queue.get()
    return None, time.time() - t_start


def score_algorithm(name, factor_fn, bit_sizes=None, quiet=False):
    """
    Score a factoring algorithm across bit sizes.

    Returns dict with: max_bits, exponent, time_at_max, details
    """
    if bit_sizes is None:
        bit_sizes = SCORING_BIT_SIZES

    from interface import validate_factors

    if not quiet:
        print(f"\n  Scoring: {name}")
        print(f"  {'bits':>6} {'solved':>8} {'median_t':>10} {'status':>10}")
        print(f"  {'-' * 40}")

    details = {}
    max_bits = 0
    alive = True

    for bits in bit_sizes:
        if not alive:
            break

        cases = generate_test_suite(bits, CASES_PER_SIZE)
        times = []
        solved = 0

        for N, p_exp, q_exp in cases:
            result, elapsed = run_with_timeout(factor_fn, (N,), TIMEOUT_SECONDS)

            if result is not None:
                ok, _ = validate_factors(N, result)
                if ok and elapsed <= TIMEOUT_SECONDS:
                    solved += 1
                    times.append(elapsed)

        median_t = sorted(times)[len(times) // 2] if times else float('inf')
        details[bits] = {
            'solved': solved,
            'total': CASES_PER_SIZE,
            'median_time': median_t,
            'times': times,
        }

        status = "alive" if solved >= ALIVE_THRESHOLD else "DEAD"
        if solved >= ALIVE_THRESHOLD:
            max_bits = bits
        else:
            alive = False

        if not quiet:
            t_str = f"{median_t:.4f}s" if median_t < float('inf') else "timeout"
            print(f"  {bits:>6} {solved:>5}/{CASES_PER_SIZE:>2} {t_str:>10} {status:>10}")

    # Compute scaling exponent via log-log regression
    exponent = _compute_exponent(details, max_bits)

    # Time at max bits
    time_at_max = details.get(max_bits, {}).get('median_time', float('inf'))

    return {
        'name': name,
        'max_bits': max_bits,
        'exponent': exponent,
        'time_at_max': time_at_max,
        'details': details,
    }


def _compute_exponent(details, max_bits):
    """
    Fit t(n) = c * n^k using log-log regression on bit sizes up to max_bits.
    Returns k (the scaling exponent).
    """
    log_n = []
    log_t = []

    for bits, d in sorted(details.items()):
        if bits > max_bits:
            break
        if d['median_time'] > 0 and d['median_time'] < float('inf') and d['solved'] >= ALIVE_THRESHOLD:
            log_n.append(math.log(bits))
            log_t.append(math.log(d['median_time']))

    if len(log_n) < 3:
        return float('inf')

    # Linear regression on log-log data: log(t) = k * log(n) + log(c)
    n = len(log_n)
    sum_x = sum(log_n)
    sum_y = sum(log_t)
    sum_xy = sum(x * y for x, y in zip(log_n, log_t))
    sum_x2 = sum(x * x for x in log_n)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-15:
        return float('inf')

    k = (n * sum_xy - sum_x * sum_y) / denom
    return k


def print_scoreboard(results):
    """Print the final scoreboard."""
    # Sort: max_bits desc, exponent asc, time_at_max asc
    ranked = sorted(results, key=lambda r: (-r['max_bits'], r['exponent'], r['time_at_max']))

    print(f"\n{'=' * 70}")
    print(f"  SCOREBOARD")
    print(f"{'=' * 70}")
    print(f"  {'Rank':>4} {'Algorithm':<25} {'max_bits':>9} {'exponent':>9} {'time@max':>10}")
    print(f"  {'-' * 62}")

    for rank, r in enumerate(ranked, 1):
        exp_str = f"{r['exponent']:.2f}" if r['exponent'] < float('inf') else "n/a"
        t_str = f"{r['time_at_max']:.4f}s" if r['time_at_max'] < float('inf') else "n/a"
        marker = " <-- YOUR ENTRY" if r['name'] == "entry" else ""
        print(f"  {rank:>4} {r['name']:<25} {r['max_bits']:>9} {exp_str:>9} {t_str:>10}{marker}")

    print(f"\n  Scoring: max_bits (higher=better), exponent (lower=better), time@max (lower=better)")
    print(f"  A polynomial-time algorithm shows constant exponent at 1024 bits.")

    # Check for polynomial-time claim
    for r in ranked:
        if r['max_bits'] >= 512 and r['exponent'] < 20:
            print(f"\n  *** {r['name']} may be polynomial-time! Exponent = {r['exponent']:.2f} ***")
            print(f"  *** Verify by extending to 2048+ bits. ***")


def main():
    parser = argparse.ArgumentParser(description="FactorCup Scorer")
    parser.add_argument('--baselines', action='store_true',
                        help='Also score baseline algorithms')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer bit sizes, smaller timeout')
    args = parser.parse_args()

    print("=" * 70)
    print("  FactorCup Scorer")
    print("=" * 70)

    if args.quick:
        bit_sizes = [32, 48, 64, 80, 96]
    else:
        bit_sizes = SCORING_BIT_SIZES

    results = []

    # Score the entry
    try:
        import entry
        result = score_algorithm("entry", entry.factor, bit_sizes)
        results.append(result)
    except ImportError:
        print("\nERROR: No entry.py found. Run test.py first.")
        sys.exit(1)

    # Score baselines if requested
    if args.baselines:
        import baselines

        for name, fn in [
            ("trial_division", baselines.trial_division),
            ("pollard_rho", baselines.pollard_rho),
            ("quadratic_sieve", baselines.quadratic_sieve),
        ]:
            result = score_algorithm(name, fn, bit_sizes)
            results.append(result)

    print_scoreboard(results)


if __name__ == "__main__":
    main()
