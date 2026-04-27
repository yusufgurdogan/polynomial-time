#!/usr/bin/env python3
"""
Direction B1: Infrastructure correlation experiment.

For each reduced form in the principal cycle at known distance d,
compute every polynomial-time function of the form coefficients (a, b, c)
and test whether ANY of them correlates with distance-to-ambiguous-form.

The ambiguous form (which factors N) sits at distance R/2.
If some computable function f(a, b, c) predicts how far we are from R/2,
we can "teleport" — binary search on f instead of walking the cycle.

Functions to test:
  - |a|, |c|, b, a/c, a+c, a*c
  - gcd(a, various things), a mod small primes
  - log|a|/log(√N), the "relative size" of a
  - The infrastructure distance d itself (as ground truth)
  - b/√Δ (normalized b-value)
  - |a² - N| (how close a² is to N)
  - Whether a is a QR mod small primes (character values)
  - The continued fraction partial quotient at this step

We measure: Pearson correlation, Spearman rank correlation, and
mutual information between each function and the "signed distance
to R/2" (i.e., d - R/2).
"""

import math
import random
import sys
import time
import numpy as np
from scipy import stats as sp_stats

sys.stdout.reconfigure(line_buffering=True)


def generate_semiprime(bits):
    from sympy import nextprime
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, (1 << (bits - half)) - 1))
    while q == p:
        q = nextprime(random.randint(lo, (1 << (bits - half)) - 1))
    if p > q: p, q = q, p
    return p * q, p, q


def enumerate_cycle_with_features(N):
    """Walk the principal cycle, compute features at each step."""
    D = 4 * N
    sqrt_D = math.isqrt(D)
    if (sqrt_D + 1) ** 2 <= D: sqrt_D += 1
    sqrt_N = math.isqrt(N)

    # Start with principal form and reduce
    a, b, c = 1, 0, -N

    # Reduce
    for _ in range(10000):
        if c == 0: break
        abs_c = abs(c)
        r = (-b) % (2 * abs_c)
        while r <= sqrt_D - 2 * abs_c: r += 2 * abs_c
        if r >= sqrt_D: r -= 2 * abs_c
        if r <= 0: r += 2 * abs_c
        c_new = (r * r - D) // (4 * c)
        a, b, c = c, r, c_new
        if 0 < b < sqrt_D and sqrt_D - b < 2 * abs(a) < sqrt_D + b:
            break

    # Walk the cycle
    start = (a, b, c)
    forms = []
    distance = 0.0
    ambig_dist = None

    for step in range(200000):
        abs_a = abs(a)

        # Check ambiguous
        is_ambig = (a != 0 and b % abs_a == 0)
        gcd_a_N = math.gcd(abs_a, N)
        is_factor_form = (1 < gcd_a_N < N)

        if is_factor_form and ambig_dist is None:
            ambig_dist = distance

        # Compute features
        features = {}
        features['a'] = a
        features['abs_a'] = abs_a
        features['b'] = b
        features['c'] = c
        features['abs_c'] = abs(c)
        features['step'] = step
        features['distance'] = distance

        # Ratios and products
        if abs_a > 0:
            features['b_over_2a'] = b / (2 * abs_a)
            features['c_over_a'] = c / a if a != 0 else 0
        else:
            features['b_over_2a'] = 0
            features['c_over_a'] = 0

        features['a_plus_c'] = abs_a + abs(c)
        features['a_times_c'] = abs_a * abs(c)
        features['a_minus_c'] = abs_a - abs(c)

        # Normalized values
        features['a_over_sqrtN'] = abs_a / sqrt_N if sqrt_N > 0 else 0
        features['log_a'] = math.log(abs_a) if abs_a > 0 else 0
        features['b_over_sqrtD'] = b / sqrt_D if sqrt_D > 0 else 0

        # Distance from a² to N
        features['a_sq_minus_N'] = abs(a * a - N)
        features['a_sq_minus_N_rel'] = abs(a * a - N) / N

        # GCD features
        features['gcd_a_N'] = gcd_a_N
        features['gcd_c_N'] = math.gcd(abs(c), N)
        features['gcd_ac_N'] = math.gcd(abs_a * abs(c), N)

        # Modular features
        for m in [3, 4, 5, 7, 8]:
            features[f'a_mod_{m}'] = abs_a % m
            features[f'b_mod_{m}'] = b % m

        # Partial quotient (the CF coefficient at this step)
        if abs_a > 0:
            features['partial_quot'] = (sqrt_D + b) // (2 * abs_a)
        else:
            features['partial_quot'] = 0

        # Ambiguous-form related
        features['is_ambiguous'] = int(is_ambig)
        features['is_factor'] = int(is_factor_form)

        forms.append(features)

        # Rho step
        if c == 0: break
        abs_c_step = abs(c)
        r = (-b) % (2 * abs_c_step)
        while r <= sqrt_D - 2 * abs_c_step: r += 2 * abs_c_step
        if r >= sqrt_D: r -= 2 * abs_c_step
        if r <= 0: r += 2 * abs_c_step
        c_new = (r * r - D) // (4 * c)

        # Distance increment
        if a != 0:
            val = (sqrt_D + b) / (2.0 * abs(a))
            if val > 0:
                distance += math.log(val)

        a, b, c = c, r, c_new

        if (a, b, c) == start and step > 0:
            break

    R = distance  # regulator
    return forms, R, ambig_dist


def correlation_analysis(bits=20, num_instances=100):
    """
    For many N=pq, enumerate cycles, compute all features,
    measure correlation between each feature and distance-to-ambiguous.
    """
    print(f"{'='*75}")
    print(f"  INFRASTRUCTURE CORRELATION ANALYSIS")
    print(f"  {bits}-bit semiprimes, {num_instances} instances")
    print(f"{'='*75}")

    # Collect all feature-vs-distance data across instances
    feature_names = None
    all_correlations = {}

    for inst in range(num_instances):
        N, p, q = generate_semiprime(bits)
        forms, R, ambig_dist = enumerate_cycle_with_features(N)

        if R <= 0 or ambig_dist is None or len(forms) < 10:
            continue

        # Target: signed distance to R/2 (normalized)
        for f in forms:
            f['dist_to_half'] = (f['distance'] - R / 2) / R  # in [-0.5, 0.5]
            f['abs_dist_to_half'] = abs(f['dist_to_half'])
            f['frac_distance'] = f['distance'] / R  # in [0, 1]

        if feature_names is None:
            feature_names = [k for k in forms[0].keys()
                           if k not in ['is_ambiguous', 'is_factor',
                                       'dist_to_half', 'abs_dist_to_half',
                                       'frac_distance', 'distance', 'step']]

        # For each feature, compute correlation with frac_distance
        for fname in feature_names:
            vals = np.array([f[fname] for f in forms], dtype=float)
            target = np.array([f['frac_distance'] for f in forms], dtype=float)

            # Skip constant features
            if np.std(vals) < 1e-10:
                continue

            try:
                pearson_r, pearson_p = sp_stats.pearsonr(vals, target)
                spearman_r, spearman_p = sp_stats.spearmanr(vals, target)
            except Exception:
                continue

            if fname not in all_correlations:
                all_correlations[fname] = {'pearson': [], 'spearman': []}
            all_correlations[fname]['pearson'].append(pearson_r)
            all_correlations[fname]['spearman'].append(spearman_r)

        if (inst + 1) % 20 == 0:
            print(f"  Processed {inst+1}/{num_instances} instances...")

    # Aggregate correlations
    print(f"\n{'='*75}")
    print(f"  RESULTS: Feature correlation with fractional cycle position")
    print(f"  (sorted by |mean Spearman correlation|)")
    print(f"{'='*75}")
    print(f"  {'feature':<25} | {'mean_pearson':>12} | {'mean_spearman':>13} | {'consistency':>11}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*13}-+-{'-'*11}")

    ranked = []
    for fname in sorted(all_correlations.keys()):
        data = all_correlations[fname]
        if len(data['spearman']) < num_instances * 0.3:
            continue
        mp = np.mean(data['pearson'])
        ms = np.mean(data['spearman'])
        # Consistency: fraction of instances where sign matches mean
        sign_consistency = np.mean([1 if s * ms > 0 else 0 for s in data['spearman']])
        ranked.append((abs(ms), fname, mp, ms, sign_consistency))

    ranked.sort(reverse=True)

    for _, fname, mp, ms, consistency in ranked[:30]:
        flag = " ← SIGNAL" if abs(ms) > 0.3 and consistency > 0.7 else ""
        print(f"  {fname:<25} | {mp:>+12.4f} | {ms:>+13.4f} | {consistency:>10.1%}{flag}")

    # Highlight anything that looks promising
    signals = [(f, ms, cons) for _, f, _, ms, cons in ranked if abs(ms) > 0.3 and cons > 0.7]
    if signals:
        print(f"\n  POTENTIAL SIGNALS FOUND:")
        for fname, ms, cons in signals:
            print(f"    {fname}: mean_spearman={ms:+.4f}, consistency={cons:.1%}")
    else:
        print(f"\n  NO SIGNALS above threshold (|spearman| > 0.3, consistency > 70%)")

    return ranked


if __name__ == "__main__":
    random.seed(42)

    # Install scipy if needed
    try:
        from scipy import stats as sp_stats
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy', '-q'])
        from scipy import stats as sp_stats

    # Run at multiple bit sizes
    for bits in [20, 24, 28]:
        ranked = correlation_analysis(bits=bits, num_instances=100)
        print()
