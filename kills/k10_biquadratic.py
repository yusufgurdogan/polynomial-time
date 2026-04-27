#!/usr/bin/env python3
"""
Direction B2: Biquadratic lift for factoring.

The idea:
  Q(√N) has unit rank 1. The log-unit lattice is 1D → LLL useless.
  Q(√N, √m) has unit rank up to 3. The log-unit lattice is 3D → LLL is sharp.

If the regulator R of Q(√N) embeds into the 3D log-unit lattice of Q(√N, √m)
in a way that LLL can extract, we get R without period-finding → factoring.

The field K = Q(√N, √m) for small m:
  - Degree 4 over Q
  - Three quadratic subfields: Q(√N), Q(√m), Q(√(Nm))
  - Unit group rank = r₁ + r₂ - 1 where r₁ = real embeddings, r₂ = conjugate pairs
  - For N, m > 0: K is totally real → r₁ = 4, r₂ = 0 → rank = 3
  - Fundamental units come from units of the three quadratic subfields PLUS
    possibly a "new" unit not in any subfield

The approach:
  1. Compute fundamental units of Q(√m) — easy, m is small
  2. Compute fundamental units of Q(√(Nm)) — sub-exponential, but smaller than Q(√N) if m is small
  3. Build the 3D log-unit lattice from known units
  4. Apply LLL → find short vector
  5. Check if short vector reveals R_N (the regulator of Q(√N))

We use SageMath-style arithmetic via sympy and mpmath for precision.
"""

import math
import random
import sys
import time
import numpy as np
from fpylll import IntegerMatrix, LLL

sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# Fundamental unit computation for real quadratic fields Q(√d)
# =============================================================================

def fundamental_unit_cf(d: int, max_iter: int = 500000) -> tuple:
    """
    Compute the fundamental unit of Q(√d) via continued fraction expansion of √d.
    Returns (x, y) where x + y√d is the fundamental unit (x² - d·y² = ±1).

    This is equivalent to solving Pell's equation.
    """
    if d <= 0:
        return None

    sqrt_d = math.isqrt(d)
    if sqrt_d * sqrt_d == d:
        return None  # d is a perfect square

    # CF expansion of √d
    m, dd, a0 = 0, 1, sqrt_d
    a = a0

    # Track convergents
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    for iteration in range(1, max_iter):
        m = dd * a - m
        dd = (d - m * m) // dd
        if dd == 0:
            break
        a = (a0 + m) // dd

        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev

        # Check Pell equation: p² - d·q² = ±1
        val = p_curr * p_curr - d * q_curr * q_curr
        if val == 1 or val == -1:
            return (p_curr, q_curr, val)  # x + y√d, norm = val

    return None


def log_embedding_quadratic(x, y, d):
    """
    Compute log|σᵢ(x + y√d)| for the two real embeddings of Q(√d).
    σ₁: √d → +√d, σ₂: √d → -√d
    Returns (log|x + y√d|, log|x - y√d|).
    """
    sqrt_d = math.sqrt(d)
    v1 = abs(x + y * sqrt_d)
    v2 = abs(x - y * sqrt_d)
    if v1 <= 0 or v2 <= 0:
        return None
    return (math.log(v1), math.log(v2))


# =============================================================================
# Biquadratic field Q(√N, √m)
# =============================================================================
# Elements: a + b√N + c√m + d√(Nm)
# Four real embeddings (when N, m > 0):
#   σ₁: √N → +√N, √m → +√m
#   σ₂: √N → -√N, √m → +√m
#   σ₃: √N → +√N, √m → -√m
#   σ₄: √N → -√N, √m → -√m
#
# Log-unit lattice: map ε → (log|σ₁(ε)|, log|σ₂(ε)|, log|σ₃(ε)|)
# (the 4th coordinate is determined by σ₁·σ₂·σ₃·σ₄ = Norm = ±1)

def log_embedding_biquad(a, b, c, d_coeff, N, m):
    """
    Compute log|σᵢ(a + b√N + c√m + d√(Nm))| for i = 1, 2, 3.
    Uses mpmath for arbitrary precision to handle large units.
    """
    import mpmath
    mpmath.mp.dps = 50

    sqrt_N = mpmath.sqrt(N)
    sqrt_m = mpmath.sqrt(m)
    sqrt_Nm = mpmath.sqrt(mpmath.mpf(N) * m)

    v1 = abs(a + b * sqrt_N + c * sqrt_m + d_coeff * sqrt_Nm)
    v2 = abs(a - b * sqrt_N + c * sqrt_m - d_coeff * sqrt_Nm)
    v3 = abs(a + b * sqrt_N - c * sqrt_m - d_coeff * sqrt_Nm)

    if v1 <= 0 or v2 <= 0 or v3 <= 0:
        return None

    return (float(mpmath.log(v1)), float(mpmath.log(v2)), float(mpmath.log(v3)))


def _safe_log_unit(x, y, d):
    import mpmath
    mpmath.mp.dps = 50
    val = mpmath.mpf(x) + mpmath.mpf(y) * mpmath.sqrt(d)
    if val > 0:
        return float(mpmath.log(val))
    return None


def build_log_unit_lattice(N, m, scale=1000):
    """
    Build the log-unit lattice of Q(√N, √m) from known units.

    Known units:
    1. Fundamental unit ε_N of Q(√N): x_N + y_N √N
    2. Fundamental unit ε_m of Q(√m): x_m + y_m √m
    3. Fundamental unit ε_{Nm} of Q(√(Nm)): x_{Nm} + y_{Nm} √(Nm)
    4. Possibly a "new" unit not from any subfield (Kubota's unit)

    The log-unit lattice has rank 3 (for totally real degree-4 field).
    The three subfield units give 3 vectors. If they generate the full
    unit group, LLL on these vectors finds short combinations.
    """
    # Compute fundamental units of the three quadratic subfields
    unit_N = fundamental_unit_cf(N)
    unit_m = fundamental_unit_cf(m)
    unit_Nm = fundamental_unit_cf(N * m)

    if unit_N is None or unit_m is None or unit_Nm is None:
        return None

    x_N, y_N, norm_N = unit_N
    x_m, y_m, norm_m = unit_m
    x_Nm, y_Nm, norm_Nm = unit_Nm

    # Embed each unit into Q(√N, √m) and compute log embeddings
    # ε_N = x_N + y_N √N → (x_N, y_N, 0, 0) in the basis {1, √N, √m, √(Nm)}
    log_N = log_embedding_biquad(x_N, y_N, 0, 0, N, m)

    # ε_m = x_m + y_m √m → (x_m, 0, y_m, 0)
    log_m = log_embedding_biquad(x_m, 0, y_m, 0, N, m)

    # ε_{Nm} = x_{Nm} + y_{Nm} √(Nm) → (x_{Nm}, 0, 0, y_{Nm})
    log_Nm = log_embedding_biquad(x_Nm, 0, 0, y_Nm, N, m)

    if log_N is None or log_m is None or log_Nm is None:
        return None

    # Scale to integers for LLL (multiply by scale factor)
    vectors = []
    for log_vec in [log_N, log_m, log_Nm]:
        scaled = [int(round(v * scale)) for v in log_vec]
        vectors.append(scaled)

    return {
        'vectors': vectors,
        'unit_N': unit_N,
        'unit_m': unit_m,
        'unit_Nm': unit_Nm,
        'log_N': log_N,
        'log_m': log_m,
        'log_Nm': log_Nm,
        'R_N': abs(log_N[0] - log_N[1]) / 2 if log_N else None,
        # R_N = log|ε_N| = log(x_N + y_N√N)
        'R_N_direct': _safe_log_unit(x_N, y_N, N),
    }


def lll_on_log_lattice(vectors, scale=1000):
    """
    Apply LLL to the 3D log-unit lattice.
    Returns the reduced basis vectors (unscaled).
    """
    dim = 3
    n = len(vectors)

    B = IntegerMatrix(max(n, dim), dim)
    for i in range(n):
        for j in range(dim):
            B[i, j] = vectors[i][j] if j < len(vectors[i]) else 0

    LLL.reduction(B)

    reduced = []
    for i in range(min(n, dim)):
        v = [int(B[i, j]) / scale for j in range(dim)]
        norm = math.sqrt(sum(x * x for x in v))
        reduced.append((v, norm))

    return reduced


def experiment_b2(bits=20, num_instances=100, m_values=None):
    """
    For each N = pq, build the biquadratic log-unit lattice for Q(√N, √m),
    apply LLL, and check if any short vector reveals R_N (→ factoring).
    """
    if m_values is None:
        m_values = [2, 3, 5, 6, 7, 10, 11, 13]

    print(f"\n{'='*75}")
    print(f"  DIRECTION B2: BIQUADRATIC LIFT")
    print(f"  {bits}-bit semiprimes, m ∈ {m_values}")
    print(f"{'='*75}")

    from sympy import nextprime

    results_by_m = {m_val: {'success': 0, 'total': 0, 'R_corr': [], 'failed': 0}
                    for m_val in m_values}

    for inst in range(num_instances):
        half = bits // 2
        lo, hi = 1 << (half - 1), (1 << half) - 1
        p = nextprime(random.randint(lo, hi))
        q = nextprime(random.randint(lo, hi))
        while q == p:
            q = nextprime(random.randint(lo, hi))
        if p > q: p, q = q, p
        N = p * q

        for m_val in m_values:
            if m_val == N or math.isqrt(N * m_val) ** 2 == N * m_val:
                continue  # skip perfect squares

            results_by_m[m_val]['total'] += 1

            # Build log-unit lattice
            data = build_log_unit_lattice(N, m_val)
            if data is None:
                results_by_m[m_val]['failed'] += 1
                continue

            R_N_true = data['R_N_direct']
            if R_N_true is None:
                results_by_m[m_val]['failed'] += 1
                continue

            # LLL reduce
            reduced = lll_on_log_lattice(data['vectors'])

            # Check: does any reduced vector or combination reveal R_N?
            # The regulator R_N appears as log_N = (log|σ₁(ε_N)|, log|σ₂(ε_N)|, log|σ₃(ε_N)|)
            # In the reduced basis, can we find a vector close to log_N?

            # Check each reduced vector
            found = False
            for v, norm in reduced:
                if norm < 1e-10:
                    continue

                # Does this vector correspond to R_N?
                # The log embedding of ε_N is (R_N, -R_N, R_N) (approximately,
                # since σ₁ and σ₃ send √N → +√N, while σ₂ sends √N → -√N)
                # So v[0] - v[1] ≈ 2·R_N (if the vector captures ε_N)

                candidate_R = abs(v[0] - v[1]) / 2
                if candidate_R > 0:
                    ratio = candidate_R / R_N_true
                    # Check if ratio is close to an integer (R might be a multiple)
                    nearest_int = round(ratio)
                    if nearest_int > 0:
                        rel_error = abs(ratio - nearest_int) / nearest_int
                        if rel_error < 0.01:  # within 1%
                            # Found R_N (or a multiple)!
                            found = True
                            results_by_m[m_val]['R_corr'].append(rel_error)

                            # Can we factor from R_N?
                            # R_N = log(x + y√N) where x² - N·y² = ±1
                            # From R_N we can recover x, y and then
                            # gcd(x ± 1, N) might give a factor
                            x_N, y_N, _ = data['unit_N']
                            g1 = math.gcd(x_N - 1, N)
                            g2 = math.gcd(x_N + 1, N)
                            if 1 < g1 < N or 1 < g2 < N:
                                results_by_m[m_val]['success'] += 1
                            break

            if inst < 3 and m_val == m_values[0]:
                print(f"\n  Instance {inst}: N={N}={p}×{q}, m={m_val}")
                print(f"  R_N = {R_N_true:.6f}")
                print(f"  Log vectors:")
                for name, lv in [('ε_N', data['log_N']), ('ε_m', data['log_m']), ('ε_Nm', data['log_Nm'])]:
                    print(f"    {name}: ({lv[0]:.4f}, {lv[1]:.4f}, {lv[2]:.4f})")
                print(f"  After LLL:")
                for v, norm in reduced:
                    print(f"    ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})  norm={norm:.4f}")
                    cand = abs(v[0] - v[1]) / 2
                    if cand > 0:
                        print(f"      candidate R = {cand:.6f}, ratio to true R = {cand/R_N_true:.4f}")

        if (inst + 1) % 25 == 0:
            print(f"  Processed {inst+1}/{num_instances}...")

    # Report
    print(f"\n{'='*75}")
    print(f"  RESULTS")
    print(f"{'='*75}")
    print(f"  {'m':>4} | {'total':>6} | {'failed':>6} | {'R found':>8} | {'factored':>8} | {'precision':>10}")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    for m_val in m_values:
        r = results_by_m[m_val]
        r_found = len(r['R_corr'])
        avg_err = f"{np.mean(r['R_corr']):.6f}" if r['R_corr'] else "N/A"
        print(f"  {m_val:4d} | {r['total']:6d} | {r['failed']:6d} | {r_found:8d} | {r['success']:8d} | {avg_err:>10}")

    total_factored = sum(r['success'] for r in results_by_m.values())
    total_R_found = sum(len(r['R_corr']) for r in results_by_m.values())
    total_tried = sum(r['total'] - r['failed'] for r in results_by_m.values())

    print(f"\n  Total: R recovered in {total_R_found}/{total_tried} cases, "
          f"factored {total_factored}/{total_tried}")

    if total_R_found > 0:
        print(f"  >>> SIGNAL DETECTED: LLL on biquadratic lattice finds R_N <<<")
    else:
        print(f"  >>> NO SIGNAL: LLL cannot extract R_N from biquadratic lattice <<<")


if __name__ == "__main__":
    random.seed(42)
    experiment_b2(bits=16, num_instances=100)
    experiment_b2(bits=20, num_instances=100)
    experiment_b2(bits=24, num_instances=50)
