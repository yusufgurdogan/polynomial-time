#!/usr/bin/env python3
"""
THE ONE CLEAN EXPERIMENT.

Question: Can you compute L(1,χ_Δ) for Δ = 4N (or N) to enough precision
to recover floor(R), using only poly(log N) terms of the Dirichlet series?

If h = 1 (25-41% of semiprimes), then R = √Δ · L(1,χ) / 2.
Computing floor(R) would give factoring via infrastructure navigation to R/2.

Three evaluation methods for L(1,χ):
  A. Partial Dirichlet series: L_M = Σ_{n=1}^{M} χ(n)/n
  B. Partial Euler product: L_B = ∏_{p ≤ B} (1 - χ(p)/p)^{-1}
  C. Smoothed partial sum (exponential damping): L_M^{sm} = Σ χ(n)/n · exp(-n/M)
     with correction factor

For each method, measure: as M increases from poly(log N) to √N,
at what point does |R_approx - R_true| < 1?

That M determines whether this is poly-time, sub-exponential, or exponential.
"""

import math
import random
import sys
import time
import mpmath
from sympy import nextprime

sys.stdout.reconfigure(line_buffering=True)
mpmath.mp.dps = 50


# =========================================================================
# Kronecker symbol (Δ/n) — the character χ_Δ
# =========================================================================

def kronecker(a, n):
    """Kronecker symbol (a/n) for general a, n."""
    if n == 0:
        return 1 if abs(a) == 1 else 0
    if n == 1:
        return 1

    # Handle n < 0
    if n < 0:
        return kronecker(a, -1) * kronecker(a, -n)

    # (a/-1) = -1 if a < 0, else 1
    if n == -1:
        return -1 if a < 0 else 1

    # Factor out 2s from n
    v = 0
    n_abs = abs(n)
    while n_abs % 2 == 0:
        v += 1
        n_abs //= 2

    # (a/2) for the 2-part
    result = 1
    if v > 0:
        a_mod8 = a % 8
        if a % 2 == 0:
            k2 = 0
        elif a_mod8 in (1, 7):
            k2 = 1
        else:
            k2 = -1
        result *= k2 ** v

    # Now compute (a/n_abs) where n_abs is odd
    if n_abs > 1:
        result *= jacobi_symbol(a, n_abs)

    return result


def jacobi_symbol(a, n):
    """Jacobi symbol (a/n) for odd n > 0."""
    if n <= 0 or n % 2 == 0:
        return 0
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0


# =========================================================================
# Fundamental discriminant and character
# =========================================================================

def fundamental_disc(N):
    """Return the fundamental discriminant Δ for Q(√N)."""
    # Remove square factors
    D = N
    d = 2
    while d * d <= D:
        while D % (d * d) == 0:
            D //= (d * d)
        d += 1
    # D is now squarefree
    if D % 4 == 1:
        return D
    else:
        return 4 * D


# =========================================================================
# Regulator computation via CF (the exact value)
# =========================================================================

def compute_regulator(N, max_iter=2000000):
    """Compute the regulator R of Q(√N) via continued fractions.
    Returns (R, x, y, norm) where x + y√N is the fundamental unit."""
    D = N
    # Remove square part
    d = 2
    while d * d <= D:
        while D % (d * d) == 0:
            D //= (d * d)
        d += 1

    sqrt_D = math.isqrt(D)
    if sqrt_D * sqrt_D == D:
        return None  # perfect square

    # For D ≡ 1 mod 4: ring of integers is Z[(1+√D)/2]
    # Fundamental unit solves x² - D·y² = ±4 or x² - D·y² = ±1
    # For D ≡ 2,3 mod 4: ring of integers is Z[√D]
    # Fundamental unit solves x² - D·y² = ±1

    m, dd, a0 = 0, 1, sqrt_D
    a = a0
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    for i in range(1, max_iter):
        m = dd * a - m
        dd = (D - m * m) // dd
        if dd == 0:
            break
        a = (a0 + m) // dd
        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev

        val = p_curr * p_curr - D * q_curr * q_curr
        if val == 1 or val == -1:
            R = float(mpmath.log(mpmath.mpf(p_curr) + mpmath.mpf(q_curr) * mpmath.sqrt(D)))
            return R, p_curr, q_curr, val

        # Also check for x² - D·y² = ±4 (for D ≡ 1 mod 4)
        if D % 4 == 1 and abs(val) == 4:
            # (p_curr + q_curr·√D)/2 is a unit in the maximal order
            R = float(mpmath.log((mpmath.mpf(p_curr) + mpmath.mpf(q_curr) * mpmath.sqrt(D)) / 2))
            return R, p_curr, q_curr, val

    return None


# =========================================================================
# L(1, χ_Δ) computation methods
# =========================================================================

def L1_dirichlet(Delta, M):
    """Partial Dirichlet series: L_M = Σ_{n=1}^{M} χ(n)/n."""
    total = mpmath.mpf(0)
    for n in range(1, M + 1):
        chi_n = kronecker(Delta, n)
        if chi_n != 0:
            total += mpmath.mpf(chi_n) / n
    return float(total)


def L1_euler(Delta, B):
    """Partial Euler product: ∏_{p ≤ B} (1 - χ(p)/p)^{-1}."""
    # Sieve primes up to B
    if B < 2:
        return 1.0
    sieve = [True] * (B + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(B**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, B + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, B + 1) if sieve[i]]

    product = mpmath.mpf(1)
    for p in primes:
        chi_p = kronecker(Delta, p)
        if chi_p != 0:
            product *= 1 / (1 - mpmath.mpf(chi_p) / p)
    return float(product)


def L1_smoothed(Delta, M):
    """Smoothed partial sum with exponential damping.
    L^sm_M = Σ χ(n)/n · exp(-n/M), then divide by correction factor."""
    # The correction: Σ_{n=1}^∞ 1/n · exp(-n/M) ≈ -log(1 - exp(-1/M)) ≈ log(M) + γ for large M
    # But for the character sum, the smoothing helps with cancellation.
    # Actually, the smoothed sum approximates L(1,χ) · (something).
    # More precisely: Σ χ(n)/n · exp(-n/M) = ∫₀^∞ L(1+t, χ) · M/(M·t+1)² dt ... not quite.
    #
    # Simpler: just compute the smoothed sum and compare to exact L.
    # The point is that exponential smoothing kills the oscillations in the tail.
    total = mpmath.mpf(0)
    for n in range(1, 10 * M + 1):
        chi_n = kronecker(Delta, n)
        if chi_n != 0:
            total += mpmath.mpf(chi_n) / n * mpmath.exp(-mpmath.mpf(n) / M)
    return float(total)


# =========================================================================
# The class number formula: h * R = √Δ * L(1,χ_Δ) / 2
# (for real quadratic fields with Δ > 0)
# =========================================================================

def class_number_formula(Delta, L1):
    """Given Δ and L(1,χ_Δ), compute h * R."""
    return math.sqrt(Delta) * L1 / 2


# =========================================================================
# Main experiment
# =========================================================================

def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, hi))
    while q == p:
        q = nextprime(random.randint(lo, hi))
    if p > q: p, q = q, p
    return p * q, p, q


def experiment_precision(bits_list=None, num_instances=30):
    if bits_list is None:
        bits_list = [16, 20, 24, 28, 32]

    print(f"{'='*75}")
    print(f"  L-FUNCTION PRECISION EXPERIMENT")
    print(f"  Can L(1,χ) computed with M terms recover floor(R)?")
    print(f"{'='*75}")
    print(f"  If h=1: R = √Δ · L(1,χ) / 2")
    print(f"  Need |R_approx - R_true| < 1 to get floor(R).")
    print(f"  Question: how does the required M scale with N?")

    # M values to test: from poly(log N) to √N
    # For n-bit N: log(N) = n·log(2) ≈ 0.7n
    # poly(log N) ≈ n², n³
    # √N = 2^{n/2}

    for bits in bits_list:
        print(f"\n{'='*75}")
        print(f"  {bits}-BIT SEMIPRIMES")
        print(f"{'='*75}")

        n = bits
        sqrt_N_approx = 2 ** (bits // 2)
        N_approx = 2 ** bits

        # M values: logarithmic spacing from n² to √N
        M_values = []
        # Polynomial in log(N)
        for k in [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50]:
            M = k * n
            if M < sqrt_N_approx * 2:
                M_values.append(M)
        # Add some larger values up to √N
        for frac in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
            M = max(1, int(frac * sqrt_N_approx))
            if M not in M_values and M <= sqrt_N_approx * 2:
                M_values.append(M)
        M_values = sorted(set(M_values))

        # Limit computation time
        max_M = min(M_values[-1], 500000)
        M_values = [m for m in M_values if m <= max_M]

        results_by_M = {M: [] for M in M_values}
        h1_count = 0
        total = 0
        skipped = 0

        t0 = time.time()
        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)
            total += 1

            # Compute true R
            D = N
            d = 2
            while d * d <= D:
                while D % (d * d) == 0:
                    D //= (d * d)
                d += 1

            reg_result = compute_regulator(D, max_iter=min(1000000, int(10 * math.sqrt(N))))
            if reg_result is None:
                skipped += 1
                continue
            R_true, _, _, norm = reg_result
            if R_true <= 0:
                skipped += 1
                continue

            Delta = fundamental_disc(N)

            # Compute true L(1,χ) from the class number formula
            # h * R = √Δ * L(1,χ) / 2
            # For now, compute h = round(√Δ * L_exact / (2R))
            # where L_exact uses many terms
            L_exact = L1_dirichlet(Delta, min(100000, max(10000, int(10 * math.sqrt(abs(Delta))))))
            hR = math.sqrt(abs(Delta)) * L_exact / 2
            h = max(1, round(hR / R_true))

            if h != 1:
                continue
            h1_count += 1

            # For each M, compute L_M and the resulting R_approx
            for M in M_values:
                L_M = L1_dirichlet(Delta, M)
                R_approx = math.sqrt(abs(Delta)) * L_M / 2  # Since h=1
                error = abs(R_approx - R_true)
                results_by_M[M].append({
                    'N': N, 'R_true': R_true, 'R_approx': R_approx,
                    'error': error, 'floor_match': int(R_approx) == int(R_true),
                    'relative_error': error / R_true if R_true > 0 else float('inf'),
                })

            if (inst + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{inst+1}/{num_instances}] h=1: {h1_count}, skipped: {skipped} ({elapsed:.1f}s)")

        if h1_count == 0:
            print(f"  No h=1 instances found at {bits} bits!")
            continue

        # Report results
        print(f"\n  Found {h1_count} instances with h=1 out of {total}")
        print(f"  √N ≈ {sqrt_N_approx}, log²(N) ≈ {n*n}")
        print()
        print(f"  {'M':>10} {'M/√N':>8} {'M/n²':>6} | {'mean_err':>10} {'med_err':>10} "
              f"{'floor_ok':>8} {'rel_err':>10} | {'verdict'}")
        print(f"  {'-'*10} {'-'*8} {'-'*6}-+-{'-'*10} {'-'*10} {'-'*8} {'-'*10}-+-{'-'*10}")

        threshold_M = None
        for M in M_values:
            data = results_by_M[M]
            if not data:
                continue
            errors = [d['error'] for d in data]
            rel_errors = [d['relative_error'] for d in data]
            floor_ok = sum(d['floor_match'] for d in data)
            mean_err = sum(errors) / len(errors)
            med_err = sorted(errors)[len(errors) // 2]
            mean_rel = sum(rel_errors) / len(rel_errors)
            floor_pct = floor_ok / len(data)

            M_over_sqrtN = M / sqrt_N_approx
            M_over_n2 = M / (n * n)

            if floor_pct >= 0.5:
                verdict = "✓ WORKS" if floor_pct >= 0.9 else "~ partial"
                if threshold_M is None:
                    threshold_M = M
            else:
                verdict = "✗ fails"

            print(f"  {M:10d} {M_over_sqrtN:8.4f} {M_over_n2:6.1f} | "
                  f"{mean_err:10.3f} {med_err:10.3f} {floor_pct:7.0%} {mean_rel:10.6f} | {verdict}")

        if threshold_M:
            M_over_sqrtN = threshold_M / sqrt_N_approx
            M_over_n2 = threshold_M / (n * n)
            print(f"\n  >>> THRESHOLD: M = {threshold_M} "
                  f"(= {M_over_sqrtN:.4f}·√N = {M_over_n2:.1f}·n²)")
            if M_over_sqrtN < 0.01:
                print(f"  >>> This is SUB-√N — potentially sub-exponential!")
            elif M_over_n2 < 10:
                print(f"  >>> This is POLYNOMIAL in log(N) — would give poly-time factoring!")
            else:
                print(f"  >>> This is between poly(log N) and √N — sub-exponential?")
        else:
            print(f"\n  >>> No threshold found within tested M range")
            print(f"  >>> floor(R) NOT recoverable with M ≤ {M_values[-1]}")

    # Final scaling analysis
    print(f"\n{'='*75}")
    print(f"  SCALING ANALYSIS")
    print(f"{'='*75}")
    print(f"  The key question: how does threshold_M scale with N?")
    print(f"  - If M ~ poly(log N): POLYNOMIAL TIME factoring for h=1 semiprimes!")
    print(f"  - If M ~ N^ε for ε < 1/4: beats SQUFOF")
    print(f"  - If M ~ √N: same as CF/infrastructure, no advantage")
    print(f"  - If M ~ N: worse than brute force")
    print()
    print(f"  Theory predicts (Polya-Vinogradov):")
    print(f"    |L(1,χ) - L_M(1,χ)| ≤ C·√Δ·log(Δ) / M  (unconditional)")
    print(f"    For floor(R) accuracy: need error < 1/√Δ")
    print(f"    → M > C·Δ·log(Δ) ~ N·log(N) (EXPONENTIAL in bits)")
    print()
    print(f"  Under GRH:")
    print(f"    |L(1,χ) - L_M(1,χ)| ≤ C·log(Δ) / √M  (conditional)")
    print(f"    For floor(R): need error < 1/√Δ")
    print(f"    → M > C·Δ·log²(Δ) ~ N·log²(N) (still EXPONENTIAL)")
    print()
    print(f"  Bach's result: Euler product with B = O(log²Δ) primes gives")
    print(f"    L(1,χ) to MULTIPLICATIVE O(1) precision (constant factor).")
    print(f"    But floor(R) needs ABSOLUTE precision 1/√Δ.")
    print(f"    Multiplicative → absolute: need relative error < 1/(R) ~ 1/√N.")
    print(f"    Bach's relative error at B = log²Δ: O(log Δ / log Δ) = O(1). Not enough.")
    print()
    print(f"  CONCLUSION: Even under GRH, recovering floor(R) from L(1,χ)")
    print(f"  requires M ~ Δ ~ N terms. This is exponential in bit-length.")
    print(f"  The h·R route does NOT give polynomial-time factoring.")


if __name__ == "__main__":
    random.seed(42)
    experiment_precision(bits_list=[16, 20, 24, 28], num_instances=40)
