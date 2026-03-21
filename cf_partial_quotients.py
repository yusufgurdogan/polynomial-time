#!/usr/bin/env python3
"""
CF PARTIAL QUOTIENTS: The one piece of infrastructure we haven't looked at.

The continued fraction of √N: [a₀; a₁, a₂, ..., a_{T}, 2a₀, a₁, ...]
has period T (the CF period). For N = pq, does the SEQUENCE of partial
quotients a₁, a₂, ..., a_T have any statistical signature distinguishing
semiprimes from primes?

The CF expansion is a walk in SL(2,Z) — noncommutative structure that
CRT doesn't directly capture. Each partial quotient a_i is an integer
determined by the floor function applied to an irrational, which is a
non-algebraic operation.

Tests:
1. Distribution of partial quotients: do semiprimes have different
   distributions than primes of similar size?
2. Autocorrelation: are there periodic patterns in the partial quotients?
3. Specific positions: do partial quotients at positions related to
   √(p-1) or √(q-1) differ from others?
4. The CF period T itself: for N = pq, T = R/log(ε) where ε is the
   fundamental unit. Does T have structure related to p and q?
5. Partial quotient sums and products: do these encode p or q?
6. GCD tests: gcd(a_i, N), gcd(a_i * a_{i+1} - 1, N), etc.
"""

import math
import random
import sys
import time
import numpy as np
from collections import Counter
from sympy import nextprime, isprime

sys.stdout.reconfigure(line_buffering=True)


def cf_expansion(N, max_terms=100000):
    """
    Compute the CF expansion of √N: [a₀; a₁, a₂, ...]
    Returns (partial_quotients, period, convergents).
    The CF is periodic: after the initial a₀, the sequence
    a₁, ..., a_T repeats, where a_T = 2a₀.
    """
    sqrt_N = math.isqrt(N)
    if sqrt_N * sqrt_N == N:
        return None, 0, []  # Perfect square

    a0 = sqrt_N
    pqs = [a0]

    # Track convergents h_k/k_k
    h_prev, h_curr = 1, a0
    k_prev, k_curr = 0, 1
    convergents = [(h_curr, k_curr)]

    m, d, a = 0, 1, a0
    period = 0

    for i in range(1, max_terms):
        m = d * a - m
        d = (N - m * m) // d
        if d == 0:
            break
        a = (a0 + m) // d
        pqs.append(a)

        h_prev, h_curr = h_curr, a * h_curr + h_prev
        k_prev, k_curr = k_curr, a * k_curr + k_prev
        convergents.append((h_curr, k_curr))

        if a == 2 * a0:
            period = i
            break

    return pqs, period, convergents


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, hi))
    while q == p:
        q = nextprime(random.randint(lo, hi))
    if p > q: p, q = q, p
    return p * q, p, q


def generate_prime(bits):
    lo, hi = 1 << (bits - 1), (1 << bits) - 1
    return nextprime(random.randint(lo, hi))


# =========================================================================
# Analysis functions
# =========================================================================

def pq_stats(pqs):
    """Compute statistics of partial quotients."""
    if len(pqs) < 2:
        return {}
    # Skip a0 (it's just floor(√N))
    seq = pqs[1:]
    if not seq:
        return {}
    arr = np.array(seq, dtype=np.float64)
    return {
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr),
        'max': np.max(arr),
        'min': np.min(arr),
        'geometric_mean': np.exp(np.mean(np.log(arr + 1))) - 1,
        'entropy': -np.sum(np.array(list(Counter(seq).values())) / len(seq) *
                          np.log2(np.array(list(Counter(seq).values())) / len(seq))),
        'n_distinct': len(set(seq)),
        'n_ones': seq.count(1),
        'frac_ones': seq.count(1) / len(seq),
        'frac_large': sum(1 for x in seq if x > 2 * np.mean(arr)) / len(seq),
    }


def gcd_scan(pqs, N):
    """Check if any partial quotient or combination gives a factor."""
    factors_found = []
    for i, a in enumerate(pqs):
        # Direct gcd
        g = math.gcd(a, N)
        if 1 < g < N:
            factors_found.append(('direct', i, a, g))

        # Product of consecutive
        if i > 0:
            g = math.gcd(pqs[i] * pqs[i-1] - 1, N)
            if 1 < g < N:
                factors_found.append(('product-1', i, pqs[i] * pqs[i-1] - 1, g))
            g = math.gcd(pqs[i] * pqs[i-1] + 1, N)
            if 1 < g < N:
                factors_found.append(('product+1', i, pqs[i] * pqs[i-1] + 1, g))

        # Convergent h_k
        # h_k^2 - N * k_k^2 = (-1)^k * something
        # So gcd(h_k^2 - N * k_k^2, N) might give factors at specific positions

    return factors_found


def convergent_scan(convergents, N):
    """Check if convergent-based quantities give factors."""
    factors_found = []
    for i, (h, k) in enumerate(convergents):
        val = h * h - N * k * k
        if val != 0:
            g = math.gcd(abs(val), N)
            if 1 < g < N:
                factors_found.append(('h²-Nk²', i, val, g))

        # Also try gcd(h, N) and gcd(k, N)
        for name, v in [('h', h), ('k', k)]:
            g = math.gcd(v % N, N)
            if 1 < g < N:
                factors_found.append((name, i, v, g))

    return factors_found


# =========================================================================
# Main experiment
# =========================================================================

def experiment(bits_list=None, num_instances=50):
    if bits_list is None:
        bits_list = [16, 20, 24, 28, 32]

    print(f"{'='*75}")
    print(f"  CF PARTIAL QUOTIENT ANALYSIS")
    print(f"  Do semiprimes have different CF statistics than primes?")
    print(f"{'='*75}")

    for bits in bits_list:
        print(f"\n{'='*75}")
        print(f"  {bits}-BIT NUMBERS")
        print(f"{'='*75}")

        # Collect stats for semiprimes and primes
        semi_stats = []
        prime_stats = []
        semi_periods = []
        prime_periods = []
        semi_gcd_hits = 0
        semi_conv_hits = 0

        t0 = time.time()

        for inst in range(num_instances):
            # Semiprime
            N, p, q = generate_semiprime(bits)
            pqs, period, convs = cf_expansion(N, max_terms=200000)
            if pqs is None or period == 0:
                continue
            stats = pq_stats(pqs)
            if stats:
                semi_stats.append(stats)
                semi_periods.append(period)

                # GCD scan
                gcd_hits = gcd_scan(pqs, N)
                if gcd_hits:
                    semi_gcd_hits += 1
                conv_hits = convergent_scan(convs[:min(len(convs), period + 10)], N)
                if conv_hits:
                    semi_conv_hits += 1

            # Prime of similar size
            P = generate_prime(bits)
            pqs_p, period_p, convs_p = cf_expansion(P, max_terms=200000)
            if pqs_p is None or period_p == 0:
                continue
            stats_p = pq_stats(pqs_p)
            if stats_p:
                prime_stats.append(stats_p)
                prime_periods.append(period_p)

        elapsed = time.time() - t0

        if not semi_stats or not prime_stats:
            print(f"  Not enough data (elapsed: {elapsed:.1f}s)")
            continue

        # Compare statistics
        print(f"  Computed in {elapsed:.1f}s")
        print(f"  Semiprimes: {len(semi_stats)} instances, Primes: {len(prime_stats)} instances")

        print(f"\n  --- CF Period ---")
        print(f"  Semiprimes: mean={np.mean(semi_periods):.1f}, "
              f"median={np.median(semi_periods):.0f}, "
              f"std={np.std(semi_periods):.1f}")
        print(f"  Primes:     mean={np.mean(prime_periods):.1f}, "
              f"median={np.median(prime_periods):.0f}, "
              f"std={np.std(prime_periods):.1f}")
        ratio = np.mean(semi_periods) / np.mean(prime_periods) if np.mean(prime_periods) > 0 else 0
        print(f"  Ratio (semi/prime): {ratio:.3f}")

        # Compare each statistic
        print(f"\n  --- Partial Quotient Statistics ---")
        stat_keys = ['mean', 'median', 'std', 'max', 'geometric_mean', 'entropy',
                     'n_distinct', 'frac_ones', 'frac_large']
        print(f"  {'statistic':>20} | {'semiprime':>12} | {'prime':>12} | {'ratio':>8} | {'signal?'}")
        print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}")

        for key in stat_keys:
            semi_vals = [s.get(key, 0) for s in semi_stats if key in s]
            prime_vals = [s.get(key, 0) for s in prime_stats if key in s]
            if not semi_vals or not prime_vals:
                continue
            sm = np.mean(semi_vals)
            pm = np.mean(prime_vals)
            r = sm / pm if pm > 0 else float('inf')
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(semi_vals) + np.var(prime_vals)) / 2)
            d = abs(sm - pm) / pooled_std if pooled_std > 0 else 0
            signal = "SIGNAL" if d > 0.5 else ("weak" if d > 0.2 else "none")
            print(f"  {key:>20} | {sm:12.4f} | {pm:12.4f} | {r:8.3f} | {signal} (d={d:.2f})")

        # GCD and convergent hits
        print(f"\n  --- Factoring from CF ---")
        print(f"  GCD hits (a_i shares factor with N): {semi_gcd_hits}/{len(semi_stats)}")
        print(f"  Convergent hits (h²-Nk² shares factor): {semi_conv_hits}/{len(semi_stats)}")

    # =====================================================================
    # DEEP DIVE: Autocorrelation of partial quotients
    # =====================================================================
    print(f"\n{'='*75}")
    print(f"  DEEP DIVE: Autocorrelation of partial quotients")
    print(f"{'='*75}")

    for bits in [20, 24, 28]:
        print(f"\n  {bits}-bit semiprimes:")
        for trial in range(3):
            N, p, q = generate_semiprime(bits)
            pqs, period, _ = cf_expansion(N, max_terms=200000)
            if pqs is None or period < 10:
                continue
            seq = np.array(pqs[1:period+1], dtype=np.float64)
            if len(seq) < 20:
                continue
            seq_c = seq - np.mean(seq)
            var = np.var(seq_c)
            if var < 1e-10:
                continue

            print(f"    N = {N} = {p} × {q}, period = {period}")

            # Autocorrelation
            max_lag = min(period // 2, 500)
            peaks = []
            for lag in range(1, max_lag):
                ac = np.mean(seq_c[:len(seq_c)-lag] * seq_c[lag:]) / var
                if lag > 1 and lag < max_lag - 1:
                    ac_prev = np.mean(seq_c[:len(seq_c)-lag+1] * seq_c[lag-1:]) / var
                    ac_next = np.mean(seq_c[:len(seq_c)-lag-1] * seq_c[lag+1:]) / var
                    if ac > ac_prev and ac > ac_next and ac > 0.1:
                        peaks.append((lag, ac))
                        # Check if lag relates to p, q, p-1, q-1
                        flags = []
                        if period > 0 and lag > 0 and period % lag == 0:
                            flags.append(f"divides period")
                        for target, name in [(p-1, 'p-1'), (q-1, 'q-1'),
                                            (p, 'p'), (q, 'q'),
                                            (math.isqrt(p-1), '√(p-1)'),
                                            (math.isqrt(q-1), '√(q-1)')]:
                            if target > 0 and abs(lag - target) < 3:
                                flags.append(f"≈ {name}={target}")
                            if target > 0 and lag > 0 and target % lag == 0:
                                flags.append(f"divides {name}={target}")

                        if flags:
                            print(f"      AC peak at lag {lag}: {ac:.4f} — {', '.join(flags)}")

            if not peaks:
                print(f"      No significant AC peaks")

    # =====================================================================
    # DEEP DIVE: Does the CF period T encode p and q?
    # =====================================================================
    print(f"\n{'='*75}")
    print(f"  DEEP DIVE: CF period T vs factors")
    print(f"{'='*75}")
    print(f"  T = period of CF(√N). For N = pq, T ≈ R/ln(ε) where R is the regulator.")
    print(f"  Does T relate to p, q, p-1, q-1?")

    for bits in [16, 20, 24]:
        print(f"\n  {bits}-bit semiprimes:")
        for trial in range(5):
            N, p, q = generate_semiprime(bits)
            pqs, period, _ = cf_expansion(N, max_terms=500000)
            if pqs is None or period == 0:
                continue

            # Check relationships
            flags = []
            for target, name in [(p-1, 'p-1'), (q-1, 'q-1'), (p, 'p'), (q, 'q'),
                                  (p+q, 'p+q'), (p*q, 'N'), ((p-1)*(q-1), 'φ(N)'),
                                  (math.isqrt(N), '√N'), (math.gcd(p-1, q-1), 'gcd(p-1,q-1)')]:
                if target > 0:
                    g = math.gcd(period, target)
                    if g > 1:
                        flags.append(f"gcd(T,{name})={g}")
                    ratio = period / target
                    if abs(ratio - round(ratio)) < 0.01 and round(ratio) > 0:
                        flags.append(f"T/{name}≈{round(ratio)}")

            print(f"    N={N}={p}×{q}: T={period}  {', '.join(flags[:5])}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    experiment(bits_list=[16, 20, 24, 28], num_instances=40)
