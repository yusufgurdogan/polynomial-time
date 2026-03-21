#!/usr/bin/env python3
"""
Spectral scout: Cayley graph eigenvalues on (Z/NZ)*.

Build the Cayley graph with generators = small primes.
Compute full eigendecomposition. Compare spectral statistics
between N = pq (composite) and N = prime.

Signal: ANY spectral statistic that separates composites from primes.
- Spectral gap (λ₁ - λ₂)
- Spectral radius
- Trace of A^k for various k
- Eigenvalue distribution moments
- Number of distinct eigenvalues
- Spectral entropy

Prediction: dies. Eigenvalues factor via CRT for N=pq, but the
products are statistically indistinguishable from cyclic case.

Keep it small: 16-20 bits (full eigendecomposition is O(φ(N)³)).
"""

import math
import random
import sys
import time
import numpy as np
from scipy import stats as sp_stats
from sympy import nextprime, isprime, primitive_root as sym_proot

sys.stdout.reconfigure(line_buffering=True)


def euler_phi(n):
    """Compute Euler's totient."""
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def group_elements(N):
    """Return list of elements of (Z/NZ)*."""
    return [a for a in range(1, N) if math.gcd(a, N) == 1]


def build_cayley_adjacency(N, generators):
    """
    Build adjacency matrix of Cayley graph Cay((Z/NZ)*, S)
    where S = generators ∪ generators^{-1}.
    """
    elems = group_elements(N)
    n = len(elems)
    elem_to_idx = {a: i for i, a in enumerate(elems)}

    # Symmetric generator set
    gen_set = set()
    for g in generators:
        if math.gcd(g, N) == 1:
            gen_set.add(g % N)
            gen_set.add(pow(g, -1, N))

    A = np.zeros((n, n), dtype=float)
    for i, a in enumerate(elems):
        for g in gen_set:
            b = (a * g) % N
            if b in elem_to_idx:
                j = elem_to_idx[b]
                A[i, j] = 1.0

    return A


def spectral_stats(A):
    """Compute spectral statistics of adjacency matrix."""
    eigenvalues = np.linalg.eigvalsh(A)  # real symmetric
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending

    n = len(eigenvalues)
    stats = {}

    # Basic
    stats['lambda_1'] = eigenvalues[0]  # largest (= degree for regular graph)
    stats['lambda_2'] = eigenvalues[1] if n > 1 else 0
    stats['lambda_min'] = eigenvalues[-1]
    stats['spectral_gap'] = eigenvalues[0] - eigenvalues[1] if n > 1 else 0
    stats['spectral_gap_normalized'] = stats['spectral_gap'] / eigenvalues[0] if eigenvalues[0] > 0 else 0

    # Spectral radius (second largest absolute value)
    abs_eigs = np.abs(eigenvalues)
    abs_sorted = np.sort(abs_eigs)[::-1]
    stats['spectral_radius_2'] = abs_sorted[1] if n > 1 else 0
    stats['ramanujan_ratio'] = abs_sorted[1] / (2 * math.sqrt(eigenvalues[0] - 1)) if eigenvalues[0] > 1 else 0

    # Distribution
    stats['mean_eigenvalue'] = np.mean(eigenvalues)
    stats['std_eigenvalue'] = np.std(eigenvalues)
    stats['skew_eigenvalue'] = float(sp_stats.skew(eigenvalues))
    stats['kurtosis_eigenvalue'] = float(sp_stats.kurtosis(eigenvalues))

    # Trace of powers (= number of closed walks)
    stats['trace_A2'] = np.sum(eigenvalues ** 2)  # = number of edges × 2
    stats['trace_A3'] = np.sum(eigenvalues ** 3)  # = 6 × triangles
    stats['trace_A4'] = np.sum(eigenvalues ** 4)

    # Normalized traces
    if eigenvalues[0] > 0:
        normed = eigenvalues / eigenvalues[0]
        stats['trace_normed_2'] = np.sum(normed ** 2)
        stats['trace_normed_3'] = np.sum(normed ** 3)
        stats['trace_normed_4'] = np.sum(normed ** 4)
        stats['trace_normed_6'] = np.sum(normed ** 6)
    else:
        stats['trace_normed_2'] = 0
        stats['trace_normed_3'] = 0
        stats['trace_normed_4'] = 0
        stats['trace_normed_6'] = 0

    # Number of distinct eigenvalues (up to tolerance)
    unique = len(np.unique(np.round(eigenvalues, 6)))
    stats['num_distinct_eigenvalues'] = unique
    stats['distinct_ratio'] = unique / n if n > 0 else 0

    # Spectral entropy
    pos_eigs = eigenvalues[eigenvalues > 1e-10]
    if len(pos_eigs) > 0:
        probs = pos_eigs / np.sum(pos_eigs)
        stats['spectral_entropy'] = -np.sum(probs * np.log(probs + 1e-30))
    else:
        stats['spectral_entropy'] = 0

    # Eigenvalue gaps (sorted)
    gaps = np.diff(eigenvalues[::-1])  # ascending order gaps
    stats['mean_gap'] = np.mean(gaps) if len(gaps) > 0 else 0
    stats['std_gap'] = np.std(gaps) if len(gaps) > 0 else 0
    stats['max_gap'] = np.max(gaps) if len(gaps) > 0 else 0

    return stats


def run_experiment(bits=16, num_instances=200, num_generators=6):
    """Compare spectral stats: N=pq vs N=prime."""
    print(f"\n{'='*75}")
    print(f"  SPECTRAL SCOUT: {bits}-bit, {num_instances} instances, {num_generators} generators")
    print(f"{'='*75}")

    generators = [2, 3, 5, 7, 11, 13][:num_generators]

    composite_stats = []
    prime_stats = []

    for i in range(num_instances):
        # Generate composite N = pq
        half = bits // 2
        lo, hi = 1 << (half - 1), (1 << half) - 1
        p = nextprime(random.randint(lo, hi))
        q = nextprime(random.randint(lo, hi))
        while q == p:
            q = nextprime(random.randint(lo, hi))
        N_comp = p * q

        # Generate prime N of similar size
        N_prime = nextprime(N_comp)

        # Skip if too large (eigendecomp is O(φ(N)³))
        if euler_phi(N_comp) > 15000 or euler_phi(N_prime) > 15000:
            continue

        # Build and analyze composite
        A_comp = build_cayley_adjacency(N_comp, generators)
        s_comp = spectral_stats(A_comp)
        s_comp['N'] = N_comp
        s_comp['type'] = 'composite'
        s_comp['phi_N'] = euler_phi(N_comp)
        composite_stats.append(s_comp)

        # Build and analyze prime
        A_prime = build_cayley_adjacency(N_prime, generators)
        s_prime = spectral_stats(A_prime)
        s_prime['N'] = N_prime
        s_prime['type'] = 'prime'
        s_prime['phi_N'] = euler_phi(N_prime)
        prime_stats.append(s_prime)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_instances}...")

    if not composite_stats or not prime_stats:
        print("  No valid instances!")
        return

    # Compare all statistics
    stat_names = [k for k in composite_stats[0].keys()
                  if k not in ['N', 'type', 'phi_N']]

    print(f"\n  {'statistic':<30} | {'comp_mean':>10} | {'prime_mean':>10} | {'effect_d':>8} | {'p_value':>10} | signal")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-------")

    results = []
    for sname in stat_names:
        comp_vals = [s[sname] for s in composite_stats if not np.isnan(s[sname]) and not np.isinf(s[sname])]
        prime_vals = [s[sname] for s in prime_stats if not np.isnan(s[sname]) and not np.isinf(s[sname])]

        if len(comp_vals) < 10 or len(prime_vals) < 10:
            continue

        comp_mean = np.mean(comp_vals)
        prime_mean = np.mean(prime_vals)
        pooled_std = np.sqrt((np.var(comp_vals) + np.var(prime_vals)) / 2)

        if pooled_std > 1e-10:
            effect_d = abs(comp_mean - prime_mean) / pooled_std
        else:
            effect_d = 0

        try:
            _, p_val = sp_stats.mannwhitneyu(comp_vals, prime_vals, alternative='two-sided')
        except Exception:
            p_val = 1.0

        flag = " ← YES" if effect_d > 0.5 and p_val < 0.01 else ""
        results.append((effect_d, sname, comp_mean, prime_mean, p_val, flag))

    results.sort(reverse=True)
    for effect_d, sname, cm, pm, pv, flag in results[:25]:
        print(f"  {sname:<30} | {cm:>10.4f} | {pm:>10.4f} | {effect_d:>8.4f} | {pv:>10.6f} |{flag}")

    signals = [(s, d, p) for d, s, _, _, p, f in results if f]
    if signals:
        print(f"\n  SIGNALS DETECTED:")
        for sname, eff, pv in signals:
            print(f"    {sname}: effect_d={eff:.4f}, p={pv:.6f}")
    else:
        print(f"\n  NO SIGNALS (effect_d > 0.5 AND p < 0.01)")

    return results


if __name__ == "__main__":
    random.seed(42)
    run_experiment(bits=14, num_instances=200, num_generators=6)
    run_experiment(bits=16, num_instances=200, num_generators=6)
