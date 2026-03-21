#!/usr/bin/env python3
"""
CARRY CLASS GEOMETRY: What does the solution space look like INSIDE each carry class?

The tensor measurement showed: at position k, there are ~5 carry values,
each compatible with ~2^(k-3) different (p_prefix, q_prefix) pairs.
The exponential is INSIDE each carry class.

Question: do the (p,q) prefixes within a carry class have STRUCTURE?
- Uniform scatter → meet-in-the-middle optimal, nothing more to find
- Algebraic structure → faster search possible, potential breakthrough

For each carry value c at the midpoint:
1. Enumerate all (p_prefix, q_prefix) pairs
2. Look at p_prefix values: clustered? coset of subgroup? structured gaps?
3. Look at (p_prefix, q_prefix) as lattice points: lines? curves? sublattices?
4. p_prefix mod small primes: uniform or biased?
"""

import math
import random
import sys
import numpy as np
from collections import defaultdict, Counter
from sympy import nextprime

sys.stdout.reconfigure(line_buffering=True)


def get_bit(x, i):
    return (x >> i) & 1


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, hi))
    while q == p:
        q = nextprime(random.randint(lo, hi))
    if p > q: p, q = q, p
    return p * q, p, q


def enumerate_carry_classes(N, k):
    """
    Enumerate ALL valid (p_prefix, q_prefix, carry) states at position k.
    Returns: dict mapping carry_value → list of (p_prefix, q_prefix) pairs.
    """
    N_bits = [(N >> i) & 1 for i in range(2 * k + 2)]

    # DP from position 0 to k-1
    # State: (p_int, q_int) → carry
    states = {}

    # Position 0
    for p0 in range(2):
        for q0 in range(2):
            total = p0 * q0
            if total % 2 == N_bits[0]:
                carry = total // 2
                states[(p0, q0)] = carry

    # Propagate positions 1 to k-1
    for pos in range(1, k):
        new_states = {}
        for (p_int, q_int), carry_in in states.items():
            for p_bit in range(2):
                for q_bit in range(2):
                    new_p = p_int | (p_bit << pos)
                    new_q = q_int | (q_bit << pos)

                    # Convolution at this position
                    conv = 0
                    for j in range(pos + 1):
                        l = pos - j
                        if 0 <= l <= pos:
                            conv += ((new_p >> j) & 1) * ((new_q >> l) & 1)

                    total = conv + carry_in
                    if total % 2 == N_bits[pos]:
                        new_carry = total // 2
                        key = (new_p, new_q)
                        new_states[key] = new_carry

        states = new_states

    # Group by carry value
    carry_classes = defaultdict(list)
    for (p_int, q_int), carry in states.items():
        carry_classes[carry].append((p_int, q_int))

    return carry_classes


def analyze_carry_class(pairs, carry_val, k, N, p_true, q_true):
    """Analyze the geometry of (p_prefix, q_prefix) pairs within one carry class."""
    n_pairs = len(pairs)
    if n_pairs < 2:
        return {'n_pairs': n_pairs, 'carry': carry_val}

    p_vals = sorted(set(pv for pv, qv in pairs))
    q_vals = sorted(set(qv for pv, qv in pairs))

    result = {
        'carry': carry_val,
        'n_pairs': n_pairs,
        'n_distinct_p': len(p_vals),
        'n_distinct_q': len(q_vals),
    }

    # --- Test 1: Gap structure of p_prefix values ---
    if len(p_vals) > 1:
        gaps = [p_vals[i+1] - p_vals[i] for i in range(len(p_vals)-1)]
        gap_counter = Counter(gaps)
        most_common_gap = gap_counter.most_common(1)[0]
        n_distinct_gaps = len(gap_counter)

        result['p_gaps'] = {
            'n_distinct_gaps': n_distinct_gaps,
            'most_common_gap': most_common_gap,
            'min_gap': min(gaps),
            'max_gap': max(gaps),
            'mean_gap': np.mean(gaps),
            'std_gap': np.std(gaps),
            'is_arithmetic': n_distinct_gaps == 1,  # All gaps equal = AP
        }

        # Check if p_vals form an arithmetic progression
        if n_distinct_gaps <= 3:
            result['p_gaps']['gap_distribution'] = dict(gap_counter.most_common(5))

    # --- Test 2: p_prefix mod small primes ---
    result['p_mod'] = {}
    for m in [2, 3, 4, 5, 7, 8, 16]:
        residues = [pv % m for pv in p_vals]
        counts = Counter(residues)
        # Chi-squared test for uniformity
        expected = len(p_vals) / m
        if expected > 0:
            chi2 = sum((c - expected)**2 / expected for c in counts.values())
            # Significant non-uniformity if chi2 > 2*m
            result['p_mod'][m] = {
                'distribution': dict(counts),
                'chi2': chi2,
                'significant': chi2 > 2 * m,
            }

    # --- Test 3: (p, q) as 2D lattice points ---
    pairs_arr = np.array(pairs)
    if n_pairs > 2:
        # Check if points lie on lines
        # Compute the convex hull width / bounding box ratio
        p_range = max(p_vals) - min(p_vals) if p_vals else 0
        q_range = max(q_vals) - min(q_vals) if q_vals else 0

        # Check collinearity: do all points lie on a single line?
        if n_pairs >= 3:
            # Use first two points to define a line, check if others lie on it
            p0, q0 = pairs[0]
            p1, q1 = pairs[1]
            dp, dq = p1 - p0, q1 - q0
            on_line = 0
            for pi, qi in pairs[2:]:
                cross = (pi - p0) * dq - (qi - q0) * dp
                if cross == 0:
                    on_line += 1
            collinear_frac = (on_line + 2) / n_pairs
            result['collinear_frac'] = collinear_frac

        # Check if points form a sublattice
        # Take differences between consecutive pairs (sorted by p)
        sorted_pairs = sorted(pairs)
        if len(sorted_pairs) > 1:
            diffs = [(sorted_pairs[i+1][0] - sorted_pairs[i][0],
                      sorted_pairs[i+1][1] - sorted_pairs[i][1])
                     for i in range(len(sorted_pairs)-1)]
            diff_counter = Counter(diffs)
            most_common_diff = diff_counter.most_common(1)[0]
            result['lattice'] = {
                'n_distinct_diffs': len(diff_counter),
                'most_common_diff': most_common_diff,
                'is_sublattice': len(diff_counter) <= 3,
            }

    # --- Test 4: Relationship between p_prefix and q_prefix ---
    # For each pair, compute p + q, p - q, p * q, p XOR q
    sums = [pv + qv for pv, qv in pairs]
    diffs_pq = [abs(pv - qv) for pv, qv in pairs]
    prods = [(pv * qv) % (1 << (2*k)) for pv, qv in pairs]  # truncated product
    xors = [pv ^ qv for pv, qv in pairs]

    for name, vals in [('p+q', sums), ('|p-q|', diffs_pq), ('p*q mod 2^2k', prods), ('p XOR q', xors)]:
        unique = len(set(vals))
        result[f'{name}_distinct'] = unique
        result[f'{name}_entropy'] = -sum(c/len(vals) * math.log2(c/len(vals))
                                         for c in Counter(vals).values()) if vals else 0

    # --- Test 5: Does the true (p, q) have a special position? ---
    p_prefix_true = p_true % (1 << k)
    q_prefix_true = q_true % (1 << k)
    true_in_class = (p_prefix_true, q_prefix_true) in set(pairs)
    if true_in_class:
        # Rank of true p among all p_vals
        p_rank = sorted(p_vals).index(p_prefix_true) if p_prefix_true in p_vals else -1
        result['true_p_rank'] = p_rank
        result['true_p_percentile'] = p_rank / len(p_vals) if p_vals else -1

    return result


# =========================================================================
# Main experiment
# =========================================================================

def experiment(bits_list=None, num_instances=10):
    if bits_list is None:
        bits_list = [12, 14, 16, 18, 20]

    print(f"{'='*75}")
    print(f"  CARRY CLASS GEOMETRY")
    print(f"  What does the solution space look like inside each carry class?")
    print(f"{'='*75}")

    for bits in bits_list:
        k = bits // 2  # bits per factor
        midpoint = k  # cut at the midpoint

        if k > 12:
            print(f"\n  Skipping {bits}-bit: k={k} too large for exact enumeration")
            continue

        print(f"\n{'='*75}")
        print(f"  {bits}-BIT SEMIPRIMES (k={k}, cut at position {midpoint})")
        print(f"{'='*75}")

        all_results = []

        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)

            carry_classes = enumerate_carry_classes(N, midpoint)

            if inst < 3:  # Detailed output for first 3
                print(f"\n  Instance {inst+1}: N = {N} = {p} × {q}")
                print(f"  {len(carry_classes)} carry classes at position {midpoint}")

                for carry_val in sorted(carry_classes.keys()):
                    pairs = carry_classes[carry_val]
                    analysis = analyze_carry_class(pairs, carry_val, midpoint, N, p, q)

                    n = analysis['n_pairs']
                    print(f"\n    Carry = {carry_val}: {n} valid (p_prefix, q_prefix) pairs")

                    if n <= 20:
                        for pv, qv in sorted(pairs):
                            marker = " ← TRUE" if pv == p % (1 << midpoint) and qv == q % (1 << midpoint) else ""
                            print(f"      p={pv:>{midpoint+1}b} ({pv:>5d})  "
                                  f"q={qv:>{midpoint+1}b} ({qv:>5d})  "
                                  f"p+q={pv+qv:>5d}  p*q mod {1<<(2*midpoint)}="
                                  f"{(pv*qv)%(1<<(2*midpoint)):>5d}{marker}")

                    if 'p_gaps' in analysis:
                        pg = analysis['p_gaps']
                        print(f"      p_prefix gaps: {pg['n_distinct_gaps']} distinct, "
                              f"most common: {pg['most_common_gap']}")
                        if pg['is_arithmetic']:
                            print(f"      *** ARITHMETIC PROGRESSION with gap {pg['most_common_gap'][0]} ***")
                        if 'gap_distribution' in pg:
                            print(f"      Gap distribution: {pg['gap_distribution']}")

                    if 'lattice' in analysis:
                        lat = analysis['lattice']
                        print(f"      (p,q) diffs: {lat['n_distinct_diffs']} distinct, "
                              f"most common: {lat['most_common_diff']}")
                        if lat['is_sublattice']:
                            print(f"      *** SUBLATTICE STRUCTURE with ≤3 generators ***")

                    if 'collinear_frac' in analysis:
                        print(f"      Collinearity: {analysis['collinear_frac']:.1%} on a line")

                    # Mod structure
                    for m in [2, 3, 4, 5]:
                        if m in analysis.get('p_mod', {}):
                            mod_info = analysis['p_mod'][m]
                            sig = " *** NON-UNIFORM ***" if mod_info['significant'] else ""
                            print(f"      p mod {m}: {mod_info['distribution']} "
                                  f"chi2={mod_info['chi2']:.1f}{sig}")

                    # p+q and p*q structure
                    for key in ['p+q_distinct', 'p+q_entropy', 'p XOR q_distinct']:
                        if key in analysis:
                            print(f"      {key}: {analysis[key]}")

            # Aggregate statistics
            for carry_val, pairs in carry_classes.items():
                analysis = analyze_carry_class(pairs, carry_val, midpoint, N, p, q)
                all_results.append(analysis)

        # Summary statistics
        if all_results:
            print(f"\n  --- Aggregate statistics across {len(all_results)} carry classes ---")

            # How often do we see arithmetic progressions?
            n_ap = sum(1 for r in all_results if r.get('p_gaps', {}).get('is_arithmetic', False)
                       and r['n_pairs'] > 2)
            n_total = sum(1 for r in all_results if r['n_pairs'] > 2)
            print(f"  Arithmetic progressions: {n_ap}/{n_total} ({n_ap/n_total:.0%})" if n_total > 0 else "")

            # How often sublattice structure?
            n_lat = sum(1 for r in all_results if r.get('lattice', {}).get('is_sublattice', False)
                        and r['n_pairs'] > 2)
            print(f"  Sublattice structure (≤3 generators): {n_lat}/{n_total} ({n_lat/n_total:.0%})" if n_total > 0 else "")

            # Collinearity
            col_fracs = [r['collinear_frac'] for r in all_results if 'collinear_frac' in r]
            if col_fracs:
                print(f"  Mean collinearity: {np.mean(col_fracs):.1%}")
                print(f"  Carry classes with >50% collinear: "
                      f"{sum(1 for f in col_fracs if f > 0.5)}/{len(col_fracs)}")

            # Mod-m non-uniformity
            for m in [2, 3, 4, 5]:
                n_sig = sum(1 for r in all_results
                           if r.get('p_mod', {}).get(m, {}).get('significant', False))
                n_tested = sum(1 for r in all_results if m in r.get('p_mod', {}))
                if n_tested > 0:
                    print(f"  p mod {m} non-uniform: {n_sig}/{n_tested} ({n_sig/n_tested:.0%})")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    experiment(bits_list=[12, 14, 16, 18, 20], num_instances=10)
