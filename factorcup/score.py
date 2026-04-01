#!/usr/bin/env python3
"""
FactorCup Scorer — Measures scaling behavior.

Run: python score.py              # score your entry
     python score.py --baselines  # also score baseline algorithms
     python score.py --quick      # fewer bit sizes

ANTI-CHEAT:
  1. Test cases generated with os.urandom (unpredictable)
  2. Entry runs in 'spawn' subprocess (no forked memory — gc attack blocked)
  3. Timing measured in PARENT process (time monkey-patch blocked)
  4. Suspicion flag for impossibly fast results

Do not modify this file.
"""

import sys
import os
import time
import math
import argparse
import multiprocessing
from sympy import nextprime, isprime
import random as _random


# ============================================================================
# Force 'spawn' — child gets NO parent memory
# ============================================================================

_MP_CONTEXT = multiprocessing.get_context('spawn')


# ============================================================================
# Constants
# ============================================================================

SCORING_BIT_SIZES = [
    32, 48, 64, 80, 96, 112, 128,
    160, 192, 224, 256,
    320, 384, 448, 512,
    640, 768, 896, 1024,
]

CASES_PER_SIZE = 5
TIMEOUT_SECONDS = 60
ALIVE_THRESHOLD = 3


# ============================================================================
# Secure test case generation
# ============================================================================

def _secure_semiprime(bits, rng):
    half = bits // 2
    lo = 1 << (half - 1)
    hi = (1 << half) - 1
    for _ in range(1000):
        p = nextprime(rng.randint(lo, hi))
        if p > hi:
            continue
        q = nextprime(rng.randint(lo, hi))
        if q > hi or q == p:
            continue
        N = p * q
        if N.bit_length() >= bits - 1 and N.bit_length() <= bits + 1:
            return N, min(p, q), max(p, q)
    raise RuntimeError(f"Failed to generate {bits}-bit semiprime")


def _generate_scoring_cases(bits, count):
    seed = int.from_bytes(os.urandom(16), 'big')
    rng = _random.Random(seed)
    return [_secure_semiprime(bits, rng) for _ in range(count)]


# ============================================================================
# Validation
# ============================================================================

def _validate(N, result):
    if result is None:
        return False
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        return False
    p, q = result
    if not isinstance(p, int) or not isinstance(q, int):
        return False
    if p <= 1 or q <= 1:
        return False
    if p * q != N:
        return False
    if not isprime(p) or not isprime(q):
        return False
    return True


# ============================================================================
# Sandboxed execution — spawn + parent-side timing
# ============================================================================

def _worker(entry_module_name, N, result_queue):
    """
    Runs in a SPAWNED subprocess. Fresh Python interpreter.
    No access to parent heap. Only receives entry module name and N.
    """
    try:
        mod = __import__(entry_module_name)
        result = mod.factor(N)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(None)


def _run_sandboxed(entry_module_name, N, timeout):
    """
    Run factor(N) in a spawn'd subprocess.
    Timing is measured HERE in the parent — child can't fake it.
    """
    result_queue = _MP_CONTEXT.Queue()

    proc = _MP_CONTEXT.Process(
        target=_worker,
        args=(entry_module_name, N, result_queue),
    )

    t0 = time.monotonic()  # monotonic — immune to system clock changes
    proc.start()
    proc.join(timeout=timeout + 2)
    elapsed = time.monotonic() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return None, timeout

    if not result_queue.empty():
        result = result_queue.get_nowait()
        return result, elapsed

    return None, elapsed


# ============================================================================
# Model fitting
# ============================================================================

def _linreg(xs, ys):
    n = len(xs)
    if n < 2:
        return 0, 0, 0
    sx = sum(xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2 = sum(x * x for x in xs)
    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-15:
        return 0, 0, 0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - sy / n) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return slope, intercept, max(0, r2)


def _fit_polynomial(data):
    if len(data) < 3:
        return {'k': float('inf'), 'r2': 0}
    xs = [math.log(b) for b, _ in data]
    ys = [math.log(t) for _, t in data]
    k, log_c, r2 = _linreg(xs, ys)
    return {'k': k, 'c': math.exp(log_c), 'r2': r2}


def _fit_exponential(data):
    if len(data) < 3:
        return {'a': float('inf'), 'r2': 0}
    xs = [b for b, _ in data]
    ys = [math.log(t) for _, t in data]
    a, log_c, r2 = _linreg(xs, ys)
    return {'a': a, 'c': math.exp(log_c), 'r2': r2}


def _fit_subexponential(data):
    if len(data) < 3:
        return {'a': float('inf'), 'r2': 0}
    xs = [b ** (1 / 3) * math.log(b) ** (2 / 3) for b, _ in data]
    ys = [math.log(t) for _, t in data]
    a, log_c, r2 = _linreg(xs, ys)
    return {'a': a, 'c': math.exp(log_c), 'r2': r2}


def _best_model(poly, exp, subexp):
    models = [("POLY", poly['r2']), ("EXP", exp['r2']), ("SUBEXP", subexp['r2'])]
    return max(models, key=lambda x: x[1])[0]


# ============================================================================
# Scoring
# ============================================================================

def score_algorithm(name, entry_module_name, bit_sizes=None, quiet=False):
    if bit_sizes is None:
        bit_sizes = SCORING_BIT_SIZES

    # Warmup (absorb spawn overhead on first call)
    _run_sandboxed(entry_module_name, 15, timeout=10)

    if not quiet:
        print(f"\n  Scoring: {name}")
        print(f"  {'bits':>6} {'solved':>8} {'median_t':>10} {'status':>10}")
        print(f"  {'-' * 42}")

    details = {}
    max_bits = 0
    alive = True
    suspiciously_fast = 0

    for bits in bit_sizes:
        if not alive:
            break

        cases = _generate_scoring_cases(bits, CASES_PER_SIZE)
        times = []
        solved = 0

        for N, p_true, q_true in cases:
            result, elapsed = _run_sandboxed(entry_module_name, N, TIMEOUT_SECONDS)

            if result is not None and _validate(N, result) and elapsed <= TIMEOUT_SECONDS:
                solved += 1
                times.append(elapsed)
                if bits >= 64 and elapsed < 0.05:
                    suspiciously_fast += 1

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

    # Fit models
    data_points = [(b, d['median_time']) for b, d in sorted(details.items())
                   if b <= max_bits and d['median_time'] > 0
                   and d['median_time'] < float('inf') and d['solved'] >= ALIVE_THRESHOLD]
    poly_fit = _fit_polynomial(data_points)
    exp_fit = _fit_exponential(data_points)
    subexp_fit = _fit_subexponential(data_points)
    best_model = _best_model(poly_fit, exp_fit, subexp_fit)
    time_at_max = details.get(max_bits, {}).get('median_time', float('inf'))

    flagged = suspiciously_fast > CASES_PER_SIZE
    if not quiet and flagged:
        print(f"\n  !! WARNING: {suspiciously_fast} solves in <50ms at 64+ bits — flagged")

    if not quiet and data_points:
        print(f"\n  Model fits (R² closer to 1.0 = better fit):")
        print(f"    POLY   t(n) = c*n^k:               k={poly_fit['k']:.2f}  R²={poly_fit['r2']:.4f}")
        print(f"    EXP    t(n) = c*exp(a*n):           a={exp_fit['a']:.4f}  R²={exp_fit['r2']:.4f}")
        print(f"    SUBEXP t(n) = c*exp(a*n^{{1/3}}*...): a={subexp_fit['a']:.4f}  R²={subexp_fit['r2']:.4f}")
        print(f"    Best fit: {best_model}")
        if best_model == "POLY" and poly_fit['k'] < 20 and not flagged:
            print(f"    *** POLYNOMIAL SCALING DETECTED: t(n) ~ n^{poly_fit['k']:.1f} ***")

    return {
        'name': name,
        'max_bits': max_bits,
        'poly_k': poly_fit['k'],
        'poly_r2': poly_fit['r2'],
        'exp_r2': exp_fit['r2'],
        'subexp_r2': subexp_fit['r2'],
        'best_model': best_model,
        'time_at_max': time_at_max,
        'flagged': flagged,
        'details': details,
    }


def print_scoreboard(results):
    ranked = sorted(results, key=lambda r: (
        -r['max_bits'],
        0 if r['best_model'] == 'POLY' else 1,
        r['poly_k'] if r['poly_k'] < float('inf') else 9999,
        r['time_at_max'],
    ))

    print(f"\n{'=' * 78}")
    print(f"  SCOREBOARD")
    print(f"{'=' * 78}")
    print(f"  {'#':>2} {'Algorithm':<22} {'max_bits':>8} {'model':>7} {'poly_k':>7} "
          f"{'R²_poly':>7} {'R²_sub':>7} {'time@max':>9} {'flag':>5}")
    print(f"  {'-' * 76}")

    for rank, r in enumerate(ranked, 1):
        k_str = f"{r['poly_k']:.1f}" if r['poly_k'] < 500 else "n/a"
        rp = f"{r['poly_r2']:.3f}" if r['poly_r2'] > 0 else "n/a"
        rs = f"{r['subexp_r2']:.3f}" if r['subexp_r2'] > 0 else "n/a"
        t_str = f"{r['time_at_max']:.3f}s" if r['time_at_max'] < float('inf') else "n/a"
        flag = "!!" if r.get('flagged') else ""
        marker = " ***" if r['best_model'] == 'POLY' and r['max_bits'] >= 256 and not r.get('flagged') else ""
        print(f"  {rank:>2} {r['name']:<22} {r['max_bits']:>8} {r['best_model']:>7} "
              f"{k_str:>7} {rp:>7} {rs:>7} {t_str:>9} {flag:>5}{marker}")

    print(f"\n  Models: POLY=polynomial, EXP=exponential, SUBEXP=L[1/3] sub-exponential")
    print(f"  !!  = flagged suspicious")
    print(f"  *** = polynomial-time candidate")
    print(f"\n  Anti-cheat: spawn isolation + parent timing + os.urandom seeds")


def main():
    parser = argparse.ArgumentParser(description="FactorCup Scorer")
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    print("=" * 78)
    print("  FactorCup Scorer")
    print("  Anti-cheat: spawn isolation + parent timing + os.urandom seeds")
    print("=" * 78)

    bit_sizes = [32, 48, 64, 80, 96] if args.quick else SCORING_BIT_SIZES

    results = []

    # Score entry
    try:
        result = score_algorithm("entry", "entry", bit_sizes)
        results.append(result)
    except Exception as e:
        print(f"\nERROR loading entry: {e}")
        sys.exit(1)

    if args.baselines:
        # Score baselines by creating temporary wrapper modules
        _score_baseline(results, "trial_division", bit_sizes)
        _score_baseline(results, "pollard_rho", bit_sizes)
        _score_baseline(results, "quadratic_sieve", bit_sizes)

    print_scoreboard(results)


def _score_baseline(results, name, bit_sizes):
    """Score a baseline by dynamically creating a wrapper entry module."""
    wrapper_code = f"""
from baselines import {name}
def factor(N):
    return {name}(N)
"""
    wrapper_path = f"_baseline_{name}.py"
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_code)
    try:
        result = score_algorithm(name, f"_baseline_{name}", bit_sizes)
        results.append(result)
    finally:
        try:
            os.remove(wrapper_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
