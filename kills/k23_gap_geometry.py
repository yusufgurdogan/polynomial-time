#!/usr/bin/env python3
"""
GAP GEOMETRY: The three-distance theorem applied to modular exponential orbits.

The sequence {b^k mod N}/N for k = 0, 1, ..., T lives on the circle [0,1).
These T+1 points partition the circle into gaps.
The three-distance theorem: at most 3 distinct gap lengths.

Gap lengths are REAL NUMBERS computed by sorting and subtraction —
operations that BREAK CRT. They encode period information about
ord_N(b) = lcm(ord_p(b), ord_q(b)) in a non-algebraic way.

Key question: do the gap lengths, their ratios, or their continued
fractions encode (p-1) or (q-1) in a recoverable way?

This is the one place where CRT doesn't apply — we're operating on
the real-number geometry of the orbit, not on Z/NZ algebra.
"""

import math
import random
import sys
import time
import numpy as np
from sympy import nextprime

sys.stdout.reconfigure(line_buffering=True)


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, hi))
    while q == p:
        q = nextprime(random.randint(lo, hi))
    if p > q: p, q = q, p
    return p * q, p, q


def modexp_orbit(b, N, T):
    """Generate the orbit {b^k mod N}/N for k = 0, ..., T-1."""
    orbit = []
    val = 1
    for k in range(T):
        orbit.append(val / N)
        val = (val * b) % N
    return orbit


def compute_gaps(points):
    """
    Given points on [0,1), compute the gap lengths after sorting.
    Returns sorted gaps and the distinct gap lengths.
    """
    pts = sorted(set(points))
    n = len(pts)
    if n < 2:
        return [], []

    gaps = []
    for i in range(n - 1):
        gaps.append(pts[i + 1] - pts[i])
    # Wraparound gap
    gaps.append(1.0 - pts[-1] + pts[0])

    # Find distinct gap lengths (with tolerance)
    gap_arr = np.array(sorted(gaps))
    distinct = [gap_arr[0]]
    for g in gap_arr[1:]:
        if abs(g - distinct[-1]) > 1e-10:
            distinct.append(g)

    return gaps, distinct


def gap_analysis(N, p, q, base=2, T_values=None):
    """
    Full gap analysis for a single semiprime.
    """
    if T_values is None:
        n = N.bit_length()
        T_values = [n, 2*n, 5*n, 10*n, 20*n, 50*n, 100*n]
        T_values = [t for t in T_values if t <= 50000]

    # True multiplicative orders
    ord_p = multiplicative_order(base, p)
    ord_q = multiplicative_order(base, q)
    ord_N = lcm(ord_p, ord_q)

    results = {}

    for T in T_values:
        orbit = modexp_orbit(base, N, T)
        gaps, distinct = compute_gaps(orbit)

        if not gaps:
            continue

        gap_arr = np.array(gaps)
        n_distinct = len(distinct)

        # Gap statistics
        gap_min = min(gaps)
        gap_max = max(gaps)
        gap_ratio = gap_max / gap_min if gap_min > 1e-15 else float('inf')

        # Check: do gap lengths relate to p-1, q-1, ord_p, ord_q?
        # The three-distance theorem for irrational rotation α:
        # gaps are related to the CF partial quotients of α.
        # For our orbit: the "rotation" is multiplication by b mod N,
        # not addition. So the three-distance theorem doesn't directly apply.
        # But the gap structure still encodes period information.

        # Test: CF of gap ratio
        cf_gap_ratio = continued_fraction(gap_ratio, max_terms=20) if gap_ratio < 1e10 else []

        # Test: do any gap lengths equal k/N, k/p, k/q?
        p_related = []
        q_related = []
        for g in distinct:
            # g ≈ k/p for some integer k?
            k_p = round(g * p)
            if abs(g - k_p / p) < 1e-8:
                p_related.append((g, k_p))
            k_q = round(g * q)
            if abs(g - k_q / q) < 1e-8:
                q_related.append((g, k_q))

        # Test: do gap lengths × N give integers related to p or q?
        gN_values = [g * N for g in distinct]
        gN_gcds = [math.gcd(round(gN), N) for gN in gN_values if abs(round(gN) - gN) < 0.01]

        # Test: gaps × T — the three-distance theorem says these should
        # be related to the CF of T/ord_N(b)
        gT_values = [g * T for g in distinct]

        results[T] = {
            'n_points': len(set(orbit)),
            'n_gaps': len(gaps),
            'n_distinct_gaps': n_distinct,
            'gap_min': gap_min,
            'gap_max': gap_max,
            'gap_ratio': gap_ratio,
            'distinct_gaps': distinct,
            'cf_gap_ratio': cf_gap_ratio,
            'p_related': p_related,
            'q_related': q_related,
            'gN_gcds': gN_gcds,
            'gT_values': gT_values,
        }

    return {
        'N': N, 'p': p, 'q': q, 'base': base,
        'ord_p': ord_p, 'ord_q': ord_q, 'ord_N': ord_N,
        'T_results': results,
    }


def continued_fraction(x, max_terms=20):
    """Compute the continued fraction expansion of x."""
    cf = []
    for _ in range(max_terms):
        a = int(math.floor(x))
        cf.append(a)
        frac = x - a
        if abs(frac) < 1e-12:
            break
        x = 1.0 / frac
        if abs(x) > 1e15:
            break
    return cf


def multiplicative_order(a, p):
    if math.gcd(a, p) > 1: return 0
    order = 1
    val = a % p
    while val != 1:
        val = (val * a) % p
        order += 1
        if order > p: return 0
    return order


def lcm(a, b):
    return a * b // math.gcd(a, b)


# =========================================================================
# Main experiment
# =========================================================================

def experiment(bits_list=None, num_instances=20):
    if bits_list is None:
        bits_list = [16, 20, 24, 28, 32]

    print(f"{'='*75}")
    print(f"  GAP GEOMETRY EXPERIMENT")
    print(f"  Three-distance theorem on modular exponential orbits")
    print(f"{'='*75}")
    print(f"  Points: {{b^k mod N}}/N on [0,1) for k = 0,...,T-1")
    print(f"  Gaps: sorted differences between consecutive points")
    print(f"  Question: do gap lengths encode p-1 or q-1?")

    for bits in bits_list:
        print(f"\n{'='*75}")
        print(f"  {bits}-BIT SEMIPRIMES")
        print(f"{'='*75}")

        n = bits

        # Aggregated statistics
        p_found_count = 0
        q_found_count = 0
        gcd_nontrivial_count = 0
        factor_from_gaps = 0
        total = 0

        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)
            total += 1

            result = gap_analysis(N, p, q, base=2)

            if inst < 3:  # Detailed output for first 3 instances
                print(f"\n  Instance {inst+1}: N = {N} = {p} × {q}")
                print(f"    ord_p(2) = {result['ord_p']}, ord_q(2) = {result['ord_q']}, "
                      f"ord_N(2) = {result['ord_N']}")

                for T, tr in result['T_results'].items():
                    print(f"\n    T = {T}:")
                    print(f"      Distinct points: {tr['n_points']}, "
                          f"Distinct gap lengths: {tr['n_distinct_gaps']}")
                    print(f"      Gap range: [{tr['gap_min']:.8f}, {tr['gap_max']:.8f}]")
                    print(f"      Gap ratio: {tr['gap_ratio']:.4f}")

                    if tr['n_distinct_gaps'] <= 10:
                        for i, g in enumerate(tr['distinct_gaps'][:10]):
                            gN = g * N
                            gcd_val = math.gcd(round(gN), N) if abs(round(gN) - gN) < 0.1 else 1
                            p_match = any(abs(g - k/p) < 1e-8 for k in range(1, 20))
                            q_match = any(abs(g - k/q) < 1e-8 for k in range(1, 20))
                            flags = []
                            if p_match: flags.append("p-related!")
                            if q_match: flags.append("q-related!")
                            if gcd_val > 1 and gcd_val < N: flags.append(f"gcd={gcd_val}!")
                            flag_str = " ".join(flags)
                            print(f"        gap[{i}] = {g:.10f}  "
                                  f"×N = {gN:.4f}  "
                                  f"×T = {g*T:.4f}  {flag_str}")

                    if tr['cf_gap_ratio']:
                        print(f"      CF(gap_ratio) = {tr['cf_gap_ratio'][:10]}")

            # Check ALL T values for factoring clues
            for T, tr in result['T_results'].items():
                if tr['p_related']:
                    p_found_count += 1
                if tr['q_related']:
                    q_found_count += 1
                for gcd_val in tr['gN_gcds']:
                    if gcd_val > 1 and gcd_val < N:
                        gcd_nontrivial_count += 1
                        if gcd_val == p or gcd_val == q:
                            factor_from_gaps += 1

        # Summary
        print(f"\n  --- Summary for {bits}-bit ({total} instances) ---")
        print(f"  Gap lengths related to p: {p_found_count} instances")
        print(f"  Gap lengths related to q: {q_found_count} instances")
        print(f"  Gap × N gives nontrivial gcd: {gcd_nontrivial_count}")
        print(f"  FACTORED from gaps: {factor_from_gaps}/{total}")

    # =====================================================================
    # DEEP DIVE: What do gaps encode?
    # =====================================================================
    print(f"\n{'='*75}")
    print(f"  DEEP DIVE: What structure do gaps have?")
    print(f"{'='*75}")

    p, q = 137, 157
    N = p * q
    base = 2
    ord_p = multiplicative_order(base, p)
    ord_q = multiplicative_order(base, q)
    ord_N_val = lcm(ord_p, ord_q)

    print(f"  N = {N} = {p} × {q}")
    print(f"  ord_p(2) = {ord_p}, ord_q(2) = {ord_q}, ord_N(2) = {ord_N_val}")

    # The orbit b^k mod N is a sequence of INTEGERS in [0, N-1].
    # By CRT: b^k mod N ↔ (b^k mod p, b^k mod q).
    # (b^k mod p) cycles with period ord_p.
    # (b^k mod q) cycles with period ord_q.
    # The combined sequence cycles with period ord_N = lcm(ord_p, ord_q).

    # When we divide by N to get points on [0,1), we compute:
    # x_k = (b^k mod N) / N
    # This is NOT the same as (b^k mod p)/p or (b^k mod q)/q.
    # The division by N is a NON-CRT operation!

    # Let's see: b^k mod N = CRT(a_k, b_k) where a_k = b^k mod p, b_k = b^k mod q.
    # CRT: b^k mod N = a_k * q * (q^{-1} mod p) + b_k * p * (p^{-1} mod q)  mod N
    # So x_k = (a_k * q * q_inv_p + b_k * p * p_inv_q) / N  mod 1
    #        = a_k * (q * q_inv_p / N) + b_k * (p * p_inv_q / N)  mod 1
    #        = a_k / p * (q_inv_p * q / N * p) ... let me simplify.
    # q * q_inv_p / N = q * q_inv_p / (pq) = q_inv_p / p
    # p * p_inv_q / N = p * p_inv_q / (pq) = p_inv_q / q
    # So x_k = a_k * q_inv_p / p + b_k * p_inv_q / q  mod 1

    q_inv_p = pow(q, -1, p)  # q^{-1} mod p
    p_inv_q = pow(p, -1, q)  # p^{-1} mod q

    print(f"\n  CRT decomposition of x_k = (b^k mod N) / N:")
    print(f"  x_k = a_k · {q_inv_p}/{p} + b_k · {p_inv_q}/{q}  mod 1")
    print(f"  where a_k = 2^k mod {p}, b_k = 2^k mod {q}")
    print(f"  {q_inv_p}/{p} = {q_inv_p/p:.10f}")
    print(f"  {p_inv_q}/{q} = {p_inv_q/q:.10f}")

    # So x_k is a LINEAR COMBINATION of a_k/p and b_k/q
    # (up to the CRT coefficients q_inv_p and p_inv_q).
    # a_k/p ∈ {0, 1/p, 2/p, ..., (p-1)/p} cycles with period ord_p
    # b_k/q ∈ {0, 1/q, 2/q, ..., (q-1)/q} cycles with period ord_q

    # The gaps between consecutive x_k values (after sorting) depend on
    # the INTERLEAVING of these two component sequences.

    # THIS IS THE SAME STRUCTURE AS THE QUANTUM SAMPLES!
    # x_k = f(a_k, b_k) where f is linear in a_k and b_k.
    # The CRT coefficients (q_inv_p/p and p_inv_q/q) play the role of
    # dlp_i/(p-1) and dlq_i/(q-1) in the quantum formulation.

    print(f"\n  THIS IS THE SAME STRUCTURE AS QUANTUM SAMPLES:")
    print(f"  x_k = (CRT coeff_p) · a_k + (CRT coeff_q) · b_k  mod 1")
    print(f"  The difference: quantum CHOOSES (α, β) freely,")
    print(f"  while classical must follow the orbit (a_k, b_k) = (2^k mod p, 2^k mod q).")
    print(f"  The orbit traces a 1D PATH through the 2D parameter space.")

    # The gaps between SORTED orbit points:
    # When T < ord_N, not all points are visited. The gaps depend on
    # which (a_k, b_k) pairs we've seen.
    # When T = ord_N, all points are visited exactly once.

    T = min(ord_N_val, 5000)
    orbit = modexp_orbit(base, N, T)
    gaps, distinct = compute_gaps(orbit[:T])

    print(f"\n  With T = {T} (= ord_N = {ord_N_val}):")
    print(f"  Distinct points: {len(set(orbit[:T]))}")
    print(f"  Distinct gap lengths: {len(distinct)}")

    # The crucial question: do the gap lengths, when multiplied by N,
    # give integers that share a factor with N?
    print(f"\n  Gap lengths × N:")
    factoring_clue = False
    for i, g in enumerate(distinct[:20]):
        gN = g * N
        gN_round = round(gN)
        gN_err = abs(gN - gN_round)

        # Also try g * p and g * q
        gp = g * p
        gq = g * q
        gp_round = round(gp)
        gq_round = round(gq)
        gp_err = abs(gp - gp_round)
        gq_err = abs(gq - gq_round)

        flags = []
        if gN_err < 0.01:
            gc = math.gcd(gN_round, N)
            if gc > 1 and gc < N:
                flags.append(f"gcd(gap×N, N)={gc} ← FACTOR!")
                factoring_clue = True
            elif gc == N:
                flags.append("gap×N divisible by N")
        if gp_err < 0.01:
            flags.append(f"gap×p ≈ {gp_round}")
        if gq_err < 0.01:
            flags.append(f"gap×q ≈ {gq_round}")

        print(f"    gap[{i:2d}] = {g:.10f}  ×N={gN:.4f} (err={gN_err:.6f})  "
              f"×p={gp:.4f} (err={gp_err:.6f})  "
              f"×q={gq:.4f} (err={gq_err:.6f})  "
              f"{'  '.join(flags)}")

    if not factoring_clue:
        print(f"\n  No factoring clue from gap × N.")

    # Alternative: look at gap RATIOS
    print(f"\n  Gap RATIOS (distinct pairs):")
    for i in range(min(len(distinct), 5)):
        for j in range(i+1, min(len(distinct), 5)):
            ratio = distinct[i] / distinct[j] if distinct[j] > 1e-15 else float('inf')
            cf = continued_fraction(ratio, 15) if ratio < 1e10 else []
            # Does the ratio relate to p/q or (p-1)/(q-1)?
            ratio_pq = p / q
            ratio_p1q1 = (p-1) / (q-1)
            flags = []
            if abs(ratio - ratio_pq) < 0.01: flags.append("≈ p/q!")
            if abs(ratio - ratio_p1q1) < 0.01: flags.append("≈ (p-1)/(q-1)!")
            if abs(ratio - 1/ratio_pq) < 0.01: flags.append("≈ q/p!")
            print(f"    gap[{i}]/gap[{j}] = {ratio:.8f}  CF={cf[:8]}  {'  '.join(flags)}")

    print(f"\n  Reference: p/q = {p/q:.8f}, (p-1)/(q-1) = {(p-1)/(q-1):.8f}")
    print(f"  CF(p/q) = {continued_fraction(p/q, 10)}")
    print(f"  CF((p-1)/(q-1)) = {continued_fraction((p-1)/(q-1), 10)}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    experiment(bits_list=[16, 20, 24, 28], num_instances=15)
