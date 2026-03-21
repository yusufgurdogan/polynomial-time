#!/usr/bin/env python3
"""
B2 deep probe: sweep m values, find what makes biquadratic lift work.

For each N = pq:
  Try all primes m up to 500.
  Record which m values let LLL recover R_N.
  Measure:
    1. Fraction of m that work — does it shrink with N?
    2. Do successful m share a property computable without knowing p?
"""

import math
import random
import sys
import time
import mpmath
import numpy as np
from fpylll import IntegerMatrix, LLL

mpmath.mp.dps = 50
sys.stdout.reconfigure(line_buffering=True)


def small_primes(limit):
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            for j in range(i*i, limit + 1, i):
                s[j] = False
    return [i for i in range(limit + 1) if s[i]]


def fundamental_unit_cf(d, max_iter=500000):
    sqrt_d = math.isqrt(d)
    if sqrt_d * sqrt_d == d:
        return None
    m, dd, a0 = 0, 1, sqrt_d
    a = a0
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1
    for _ in range(1, max_iter):
        m = dd * a - m
        dd = (d - m * m) // dd
        if dd == 0: break
        a = (a0 + m) // dd
        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev
        val = p_curr * p_curr - d * q_curr * q_curr
        if val == 1 or val == -1:
            return (p_curr, q_curr, val)
    return None


def log_embed(x, y, d):
    val = mpmath.mpf(x) + mpmath.mpf(y) * mpmath.sqrt(d)
    if val > 0:
        return float(mpmath.log(val))
    return None


def log_embed_biquad(a, b, c, d_coeff, N, m_val):
    sqrt_N = mpmath.sqrt(N)
    sqrt_m = mpmath.sqrt(m_val)
    sqrt_Nm = mpmath.sqrt(mpmath.mpf(N) * m_val)
    v1 = abs(a + b * sqrt_N + c * sqrt_m + d_coeff * sqrt_Nm)
    v2 = abs(a - b * sqrt_N + c * sqrt_m - d_coeff * sqrt_Nm)
    v3 = abs(a + b * sqrt_N - c * sqrt_m - d_coeff * sqrt_Nm)
    if v1 <= 0 or v2 <= 0 or v3 <= 0:
        return None
    return (float(mpmath.log(v1)), float(mpmath.log(v2)), float(mpmath.log(v3)))


def try_biquad(N, p, q, m_val, R_N_true, unit_N, scale=1000):
    """Try biquadratic lift with given m. Returns (success, R_error, properties)."""
    Nm = N * m_val
    sqrt_Nm = math.isqrt(Nm)
    if sqrt_Nm * sqrt_Nm == Nm:
        return False, None, {}

    unit_m = fundamental_unit_cf(m_val)
    unit_Nm = fundamental_unit_cf(Nm)
    if unit_m is None or unit_Nm is None:
        return False, None, {'reason': 'no_unit'}

    x_N, y_N, _ = unit_N
    x_m, y_m, _ = unit_m
    x_Nm, y_Nm, _ = unit_Nm

    log_N = log_embed_biquad(x_N, y_N, 0, 0, N, m_val)
    log_m = log_embed_biquad(x_m, 0, y_m, 0, N, m_val)
    log_Nm = log_embed_biquad(x_Nm, 0, 0, y_Nm, N, m_val)

    if log_N is None or log_m is None or log_Nm is None:
        return False, None, {'reason': 'log_fail'}

    # LLL on 3D lattice
    vectors = [[int(round(v * scale)) for v in lv] for lv in [log_N, log_m, log_Nm]]
    B = IntegerMatrix(3, 3)
    for i in range(3):
        for j in range(3):
            B[i, j] = vectors[i][j]
    LLL.reduction(B)

    # Check reduced vectors for R_N
    for row in range(3):
        v = [int(B[row, j]) / scale for j in range(3)]
        norm = math.sqrt(sum(x * x for x in v))
        if norm < 1e-10:
            continue
        cand_R = abs(v[0] - v[1]) / 2
        if cand_R > 0 and R_N_true > 0:
            ratio = cand_R / R_N_true
            nearest = round(ratio)
            if nearest > 0:
                rel_err = abs(ratio - nearest) / nearest
                if rel_err < 0.01:
                    # Compute properties of this successful m
                    R_Nm = log_embed(x_Nm, y_Nm, Nm)
                    props = {
                        'R_Nm': R_Nm,
                        'R_Nm_over_R_N': R_Nm / R_N_true if R_Nm and R_N_true else None,
                        'cf_period_Nm': _cf_period_length(Nm),
                        'Nm_mod_4': Nm % 4,
                        'jacobi_m_p': pow(m_val, (p - 1) // 2, p) if p > 2 else 0,
                        'jacobi_m_q': pow(m_val, (q - 1) // 2, q) if q > 2 else 0,
                        'legendre_m_p': 1 if pow(m_val, (p-1)//2, p) == 1 else -1,
                        'legendre_m_q': 1 if pow(m_val, (q-1)//2, q) == 1 else -1,
                        'jacobi_m_N': _jacobi(m_val, N),
                        'gcd_m_p1': math.gcd(m_val, p - 1),
                        'gcd_m_q1': math.gcd(m_val, q - 1),
                    }
                    return True, rel_err, props

    return False, None, {'reason': 'lll_miss'}


def _cf_period_length(d):
    """Length of the CF period of √d."""
    sqrt_d = math.isqrt(d)
    if sqrt_d * sqrt_d == d:
        return 0
    m, dd, a0 = 0, 1, sqrt_d
    a = a0
    for length in range(1, 100000):
        m = dd * a - m
        dd_new = (d - m * m) // dd
        if dd_new == 0:
            return length
        dd = dd_new
        a = (a0 + m) // dd
        if a == 2 * a0:
            return length
    return -1


def _jacobi(a, n):
    if n <= 0 or n % 2 == 0: return 0
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in [3, 5]: result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3: result = -result
        a = a % n
    return result if n == 1 else 0


def sweep_experiment(bits, num_instances=30, max_m=200):
    from sympy import nextprime

    m_primes = small_primes(max_m)
    m_primes = [m for m in m_primes if m > 2]  # skip 2 for speed, or keep it

    print(f"\n{'='*75}")
    print(f"  B2 SWEEP: {bits}-bit, {num_instances} instances, m up to {max_m}")
    print(f"  Testing {len(m_primes)} prime m values per instance")
    print(f"{'='*75}")

    # Track per-instance results
    fractions = []  # fraction of m that work per instance
    all_success_props = []  # properties of successful m values
    all_fail_props = []
    m_success_counts = {m: 0 for m in m_primes}
    m_tried_counts = {m: 0 for m in m_primes}

    for inst in range(num_instances):
        half = bits // 2
        lo, hi = 1 << (half - 1), (1 << half) - 1
        p = nextprime(random.randint(lo, hi))
        q = nextprime(random.randint(lo, hi))
        while q == p:
            q = nextprime(random.randint(lo, hi))
        if p > q: p, q = q, p
        N = p * q

        unit_N = fundamental_unit_cf(N)
        if unit_N is None:
            continue
        R_N = log_embed(unit_N[0], unit_N[1], N)
        if R_N is None:
            continue

        successes = 0
        tried = 0

        for m_val in m_primes:
            if N % m_val == 0:
                continue
            Nm = N * m_val
            if math.isqrt(Nm) ** 2 == Nm:
                continue

            tried += 1
            m_tried_counts[m_val] += 1

            ok, err, props = try_biquad(N, p, q, m_val, R_N, unit_N)
            if ok:
                successes += 1
                m_success_counts[m_val] += 1
                props['m'] = m_val
                props['N'] = N
                props['bits'] = bits
                all_success_props.append(props)
            else:
                all_fail_props.append({'m': m_val, 'N': N,
                    'jacobi_m_N': _jacobi(m_val, N),
                    'legendre_m_p': 1 if pow(m_val, (p-1)//2, p) == 1 else -1,
                    'legendre_m_q': 1 if pow(m_val, (q-1)//2, q) == 1 else -1,
                    'cf_period_Nm': _cf_period_length(N * m_val),
                })

        frac = successes / tried if tried > 0 else 0
        fractions.append(frac)

        if (inst + 1) % 10 == 0:
            print(f"  [{inst+1}/{num_instances}] avg success rate: {np.mean(fractions):.1%}")

    # Analysis
    print(f"\n  === QUESTION 1: Does success fraction shrink with N? ===")
    print(f"  Mean success rate: {np.mean(fractions):.3%}")
    print(f"  Std: {np.std(fractions):.3%}")
    print(f"  Min: {min(fractions):.3%}, Max: {max(fractions):.3%}")

    # Per-m success rates
    print(f"\n  === Top m values by success rate ===")
    m_rates = [(m_success_counts[m] / max(m_tried_counts[m], 1), m, m_success_counts[m], m_tried_counts[m])
               for m in m_primes]
    m_rates.sort(reverse=True)
    for rate, m, succ, tried in m_rates[:15]:
        print(f"    m={m:4d}: {succ:3d}/{tried:3d} = {rate:.1%}")

    # Question 2: properties of successful m
    if all_success_props:
        print(f"\n  === QUESTION 2: Properties of successful m ===")
        print(f"  Total successes: {len(all_success_props)}")

        # Jacobi symbol (m/N) for successful vs failed
        succ_jacobi = [p['jacobi_m_N'] for p in all_success_props]
        fail_jacobi = [p['jacobi_m_N'] for p in all_fail_props[:len(all_success_props)*5]]
        print(f"\n  Jacobi (m/N) distribution:")
        print(f"    Success: +1={succ_jacobi.count(1)}, -1={succ_jacobi.count(-1)}, 0={succ_jacobi.count(0)}")
        if fail_jacobi:
            print(f"    Failure: +1={fail_jacobi.count(1)}, -1={fail_jacobi.count(-1)}, 0={fail_jacobi.count(0)}")

        # Legendre symbols
        succ_lp = [p.get('legendre_m_p', 0) for p in all_success_props]
        succ_lq = [p.get('legendre_m_q', 0) for p in all_success_props]
        fail_lp = [p.get('legendre_m_p', 0) for p in all_fail_props[:len(all_success_props)*5]]
        fail_lq = [p.get('legendre_m_q', 0) for p in all_fail_props[:len(all_success_props)*5]]
        print(f"\n  Legendre (m/p) for successful m:")
        print(f"    QR(+1): {succ_lp.count(1)}, QNR(-1): {succ_lp.count(-1)}")
        print(f"  Legendre (m/p) for failed m:")
        print(f"    QR(+1): {fail_lp.count(1)}, QNR(-1): {fail_lp.count(-1)}")

        # Split pattern: (m/p) vs (m/q)
        succ_split = [(p.get('legendre_m_p', 0), p.get('legendre_m_q', 0)) for p in all_success_props]
        fail_split = [(p.get('legendre_m_p', 0), p.get('legendre_m_q', 0)) for p in all_fail_props[:len(all_success_props)*5]]
        from collections import Counter
        print(f"\n  Split pattern ((m/p), (m/q)):")
        print(f"    Success: {dict(Counter(succ_split))}")
        print(f"    Failure: {dict(Counter(fail_split))}")

        # CF period of Nm
        succ_cf = [p.get('cf_period_Nm', 0) for p in all_success_props if p.get('cf_period_Nm', 0) > 0]
        fail_cf = [p.get('cf_period_Nm', 0) for p in all_fail_props if p.get('cf_period_Nm', 0) > 0]
        if succ_cf and fail_cf:
            print(f"\n  CF period of √(Nm):")
            print(f"    Success: mean={np.mean(succ_cf):.0f}, median={np.median(succ_cf):.0f}")
            print(f"    Failure: mean={np.mean(fail_cf):.0f}, median={np.median(fail_cf):.0f}")
            ratio = np.mean(succ_cf) / np.mean(fail_cf) if np.mean(fail_cf) > 0 else 0
            print(f"    Ratio: {ratio:.3f} (< 1 means success has shorter CF period)")

        # R(Nm) / R(N) ratio
        succ_R_ratio = [p.get('R_Nm_over_R_N', 0) for p in all_success_props if p.get('R_Nm_over_R_N')]
        if succ_R_ratio:
            print(f"\n  R(Q(√Nm)) / R(Q(√N)) for successful m:")
            print(f"    Mean: {np.mean(succ_R_ratio):.3f}, Median: {np.median(succ_R_ratio):.3f}")
            print(f"    Min: {min(succ_R_ratio):.3f}, Max: {max(succ_R_ratio):.3f}")

    return fractions, all_success_props


if __name__ == "__main__":
    random.seed(42)
    for bits in [16, 20, 24, 28]:
        sweep_experiment(bits, num_instances=30, max_m=200)
