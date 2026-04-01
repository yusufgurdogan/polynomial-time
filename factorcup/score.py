#!/usr/bin/env python3
"""
FactorCup Scorer — Measures scaling behavior.

Run: python score.py              # score your entry
     python score.py --baselines  # also score baseline algorithms
     python score.py --quick      # fewer bit sizes

The KEY diagnostic: fits three models to your timing data and reports
which one your algorithm matches:
    POLY:   t(n) = c * n^k         (polynomial — YOU WIN if k is constant)
    EXP:    t(n) = c * exp(a * n)  (exponential — trial division class)
    SUBEXP: t(n) = c * exp(a * n^{1/3} * (ln n)^{2/3})  (L[1/3] — NFS class)

Do not modify this file.
"""

import sys
import time
import math
import argparse
import multiprocessing
from generate import (generate_test_suite, SCORING_BIT_SIZES,
                      CASES_PER_SIZE, TIMEOUT_SECONDS, ALIVE_THRESHOLD)


def run_with_timeout(func, args, timeout):
    """Run func(args) with a hard timeout. Returns (result, elapsed) or (None, elapsed)."""
    result_queue = multiprocessing.Queue()

    def worker(q, fn, a):
        try:
            t0 = time.time()
            r = fn(*a)
            elapsed = time.time() - t0
            q.put((r, elapsed))
        except Exception:
            q.put((None, time.time()))

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
    return None, timeout


def score_algorithm(name, factor_fn, bit_sizes=None, quiet=False):
    """
    Score a factoring algorithm across bit sizes.

    Returns dict with scoring info and model fits.
    """
    if bit_sizes is None:
        bit_sizes = SCORING_BIT_SIZES

    from interface import validate_factors

    # Warmup call (absorb import/JIT overhead)
    try:
        factor_fn(15)
    except Exception:
        pass

    if not quiet:
        print(f"\n  Scoring: {name}")
        print(f"  {'bits':>6} {'solved':>8} {'median_t':>10} {'status':>10}")
        print(f"  {'-' * 42}")

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

    # Fit models
    data_points = _extract_data(details, max_bits)
    poly_fit = _fit_polynomial(data_points)
    exp_fit = _fit_exponential(data_points)
    subexp_fit = _fit_subexponential(data_points)
    best_model = _best_model(poly_fit, exp_fit, subexp_fit)

    time_at_max = details.get(max_bits, {}).get('median_time', float('inf'))

    if not quiet and data_points:
        print(f"\n  Model fits (R² closer to 1.0 = better fit):")
        print(f"    POLY   t(n) = c·n^k:            k={poly_fit['k']:.2f}  R²={poly_fit['r2']:.4f}")
        print(f"    EXP    t(n) = c·exp(a·n):        a={exp_fit['a']:.4f}  R²={exp_fit['r2']:.4f}")
        print(f"    SUBEXP t(n) = c·exp(a·n^⅓·ln²/³): a={subexp_fit['a']:.4f}  R²={subexp_fit['r2']:.4f}")
        print(f"    Best fit: {best_model}")
        if best_model == "POLY" and poly_fit['k'] < 20:
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
        'details': details,
    }


def _extract_data(details, max_bits):
    """Extract (bits, median_time) pairs for fitting."""
    pts = []
    for bits, d in sorted(details.items()):
        if bits > max_bits:
            break
        t = d['median_time']
        if t > 0 and t < float('inf') and d['solved'] >= ALIVE_THRESHOLD:
            pts.append((bits, t))
    return pts


def _linreg(xs, ys):
    """Simple linear regression. Returns (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 2:
        return 0, 0, 0
    sx = sum(xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2 = sum(x * x for x in xs)
    sy2 = sum(y * y for y in ys)

    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-15:
        return 0, 0, 0

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    # R²
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - sy / n) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0

    return slope, intercept, max(0, r2)


def _fit_polynomial(data):
    """Fit t(n) = c * n^k via log-log regression."""
    if len(data) < 3:
        return {'k': float('inf'), 'r2': 0}
    xs = [math.log(b) for b, _ in data]
    ys = [math.log(t) for _, t in data]
    k, log_c, r2 = _linreg(xs, ys)
    return {'k': k, 'c': math.exp(log_c), 'r2': r2}


def _fit_exponential(data):
    """Fit t(n) = c * exp(a * n) via semi-log regression."""
    if len(data) < 3:
        return {'a': float('inf'), 'r2': 0}
    xs = [b for b, _ in data]
    ys = [math.log(t) for _, t in data]
    a, log_c, r2 = _linreg(xs, ys)
    return {'a': a, 'c': math.exp(log_c), 'r2': r2}


def _fit_subexponential(data):
    """Fit t(n) = c * exp(a * n^{1/3} * (ln n)^{2/3}) — the L[1/3] model."""
    if len(data) < 3:
        return {'a': float('inf'), 'r2': 0}
    # Transform: ln(t) = a * n^{1/3} * (ln n)^{2/3} + ln(c)
    # Let u = n^{1/3} * (ln n)^{2/3}, then ln(t) = a*u + ln(c)
    xs = [b ** (1/3) * math.log(b) ** (2/3) for b, _ in data]
    ys = [math.log(t) for _, t in data]
    a, log_c, r2 = _linreg(xs, ys)
    return {'a': a, 'c': math.exp(log_c), 'r2': r2}


def _best_model(poly, exp, subexp):
    """Return the name of the best-fitting model."""
    models = [
        ("POLY", poly['r2']),
        ("EXP", exp['r2']),
        ("SUBEXP", subexp['r2']),
    ]
    return max(models, key=lambda x: x[1])[0]


def print_scoreboard(results):
    """Print the final scoreboard."""
    ranked = sorted(results, key=lambda r: (
        -r['max_bits'],
        0 if r['best_model'] == 'POLY' else 1,
        r['poly_k'] if r['poly_k'] < float('inf') else 9999,
        r['time_at_max'],
    ))

    print(f"\n{'=' * 75}")
    print(f"  SCOREBOARD")
    print(f"{'=' * 75}")
    print(f"  {'#':>2} {'Algorithm':<22} {'max_bits':>8} {'model':>7} {'poly_k':>7} "
          f"{'R²_poly':>7} {'R²_sub':>7} {'time@max':>9}")
    print(f"  {'-' * 73}")

    for rank, r in enumerate(ranked, 1):
        k_str = f"{r['poly_k']:.1f}" if r['poly_k'] < 500 else "n/a"
        rp = f"{r['poly_r2']:.3f}" if r['poly_r2'] > 0 else "n/a"
        rs = f"{r['subexp_r2']:.3f}" if r['subexp_r2'] > 0 else "n/a"
        t_str = f"{r['time_at_max']:.3f}s" if r['time_at_max'] < float('inf') else "n/a"
        marker = " ***" if r['best_model'] == 'POLY' and r['max_bits'] >= 256 else ""
        print(f"  {rank:>2} {r['name']:<22} {r['max_bits']:>8} {r['best_model']:>7} "
              f"{k_str:>7} {rp:>7} {rs:>7} {t_str:>9}{marker}")

    print(f"\n  Models: POLY=polynomial, EXP=exponential, SUBEXP=L[1/3] sub-exponential")
    print(f"  A breakthrough shows: model=POLY, poly_k=constant, max_bits≥512")

    for r in ranked:
        if r['best_model'] == 'POLY' and r['max_bits'] >= 512 and r['poly_k'] < 20:
            print(f"\n  !!! {r['name']}: POLYNOMIAL TIME CANDIDATE !!!")
            print(f"  !!! k = {r['poly_k']:.2f}, alive at {r['max_bits']} bits !!!")
            print(f"  !!! Extend to 2048+ bits to confirm. !!!")


def main():
    parser = argparse.ArgumentParser(description="FactorCup Scorer")
    parser.add_argument('--baselines', action='store_true',
                        help='Also score baseline algorithms')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer bit sizes')
    args = parser.parse_args()

    print("=" * 75)
    print("  FactorCup Scorer — Model Comparison Edition")
    print("=" * 75)

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
