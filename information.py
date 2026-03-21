#!/usr/bin/env python3
"""
Information-theoretic approach to factoring.

Factoring as a decoding problem:
  N = p × q is a deterministic "channel" encoding (p, q) into N.
  We observe N (2k bits) and want to recover p (k bits).

Questions:
1. How much information does each bit of N carry about each bit of p?
2. Is there a bit-position pattern in the mutual information?
3. Can belief propagation on the carry structure recover bits of p?
4. Does the carry structure make this easier than generic decoding?

The carry chain of multiplication creates correlations between bits of N
and bits of p. If these correlations are strong enough at specific
positions, iterative decoding might work.
"""

import math
import random
import sys
import numpy as np
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)


def generate_balanced_semiprimes(k: int, num_samples: int) -> list:
    """Generate num_samples semiprimes where p, q are k-bit primes."""
    from sympy import isprime, nextprime
    samples = []
    for _ in range(num_samples):
        lo = 1 << (k - 1)
        hi = (1 << k) - 1
        p = nextprime(random.randint(lo, hi))
        while p > hi:
            p = nextprime(random.randint(lo, hi))
        q = nextprime(random.randint(lo, hi))
        while q > hi or q == p:
            q = nextprime(random.randint(lo, hi))
        if p > q:
            p, q = q, p
        samples.append((p, q, p * q))
    return samples


def bit(x: int, i: int) -> int:
    """Get i-th bit of x."""
    return (x >> i) & 1


def mutual_information_bits(samples: list, k: int):
    """
    Compute mutual information I(p_i ; N_j) for each pair of bit positions.
    p has k bits, N has up to 2k bits.

    MI(X;Y) = H(X) + H(Y) - H(X,Y)
    For binary variables: compute from joint distribution.
    """
    n_bits = 2 * k
    mi_matrix = np.zeros((k, n_bits))

    for pi in range(k):
        for nj in range(n_bits):
            # Count joint occurrences
            counts = defaultdict(int)
            for p, q, N in samples:
                pb = bit(p, pi)
                nb = bit(N, nj)
                counts[(pb, nb)] += 1

            total = len(samples)
            if total == 0:
                continue

            # Compute MI
            p_marg = defaultdict(float)
            n_marg = defaultdict(float)
            for (pb, nb), c in counts.items():
                p_marg[pb] += c / total
                n_marg[nb] += c / total

            mi = 0.0
            for (pb, nb), c in counts.items():
                pxy = c / total
                px = p_marg[pb]
                py = n_marg[nb]
                if pxy > 0 and px > 0 and py > 0:
                    mi += pxy * math.log2(pxy / (px * py))

            mi_matrix[pi, nj] = mi

    return mi_matrix


def conditional_entropy(samples: list, k: int):
    """
    Compute H(p_i | N) — how uncertain is each bit of p given ALL of N?

    If H(p_i | N) = 0 for all i, then N uniquely determines p (which it does
    for semiprimes where p < q). The question is how much each bit of N
    contributes to reducing uncertainty about each bit of p.

    We approximate by computing H(p_i | N_0, ..., N_j) for increasing j.
    """
    n_bits = 2 * k

    # For each bit of p, track how uncertainty decreases as we reveal more bits of N
    # Start from LSB and from MSB of N

    print(f"\n  --- Conditional entropy H(p_i | N_0..N_j) from LSB ---")

    for pi in [0, 1, k//4, k//2, 3*k//4, k-2, k-1]:
        if pi >= k:
            continue
        uncertainties_lsb = []
        uncertainties_msb = []

        # From LSB: reveal N_0, N_1, ..., N_j
        for j in range(min(n_bits, 2*k)):
            # Group samples by the revealed N bits
            groups = defaultdict(lambda: [0, 0])
            for p, q, N in samples:
                key = N & ((1 << (j + 1)) - 1)  # N mod 2^(j+1)
                pb = bit(p, pi)
                groups[key][pb] += 1

            # H(p_i | N_0..N_j) = Σ_n P(n) H(p_i | N=n)
            total = len(samples)
            cond_h = 0.0
            for key, counts in groups.items():
                group_total = counts[0] + counts[1]
                if group_total == 0:
                    continue
                p_n = group_total / total
                for c in counts:
                    if c > 0:
                        p_bit = c / group_total
                        cond_h -= p_n * p_bit * math.log2(p_bit)
            uncertainties_lsb.append(cond_h)

        # From MSB: reveal N_{2k-1}, N_{2k-2}, ..., N_{2k-1-j}
        for j in range(min(n_bits, 2*k)):
            groups = defaultdict(lambda: [0, 0])
            for p, q, N in samples:
                key = N >> (n_bits - j - 1)  # top j+1 bits
                pb = bit(p, pi)
                groups[key][pb] += 1

            total = len(samples)
            cond_h = 0.0
            for key, counts in groups.items():
                group_total = counts[0] + counts[1]
                if group_total == 0:
                    continue
                p_n = group_total / total
                for c in counts:
                    if c > 0:
                        p_bit = c / group_total
                        cond_h -= p_n * p_bit * math.log2(p_bit)
            uncertainties_msb.append(cond_h)

        # Report
        print(f"    p[{pi:2d}]: LSB entropy path: {' '.join(f'{u:.3f}' for u in uncertainties_lsb[:8])} ...")
        # Find how many N bits needed to determine p[pi]
        for j, u in enumerate(uncertainties_lsb):
            if u < 0.01:
                print(f"           → determined after {j+1} LSB bits of N")
                break
        else:
            print(f"           → NOT determined after {len(uncertainties_lsb)} LSB bits (min H={min(uncertainties_lsb):.4f})")


def bit_correlation_map(samples: list, k: int):
    """
    Compute correlation between bit i of p and bit j of N.
    This is a simpler measure than MI — just Pearson correlation.
    """
    n_bits = 2 * k
    p_bits = np.zeros((len(samples), k))
    n_bits_arr = np.zeros((len(samples), n_bits))

    for idx, (p, q, N) in enumerate(samples):
        for i in range(k):
            p_bits[idx, i] = bit(p, i)
        for j in range(n_bits):
            n_bits_arr[idx, j] = bit(N, j)

    # Correlation matrix
    corr = np.zeros((k, n_bits))
    for i in range(k):
        for j in range(n_bits):
            # Pearson correlation
            px = p_bits[:, i]
            ny = n_bits_arr[:, j]
            if np.std(px) > 0 and np.std(ny) > 0:
                corr[i, j] = np.corrcoef(px, ny)[0, 1]

    return corr


def carry_propagation_analysis(k: int, num_samples: int = 10000):
    """
    Analyze the carry structure of p × q multiplication.

    In binary multiplication, bit j of N depends on:
    N_j = (Σ_{i+l=j} p_i × q_l + carry_in_j) mod 2
    carry_{j+1} = (Σ_{i+l=j} p_i × q_l + carry_in_j) >> 1

    The carry chain creates long-range dependencies.
    How far does a single bit of p influence bits of N?
    """
    print(f"\n  --- Carry propagation analysis (k={k}) ---")

    samples = generate_balanced_semiprimes(k, num_samples)

    # For each bit of p, flip it and see which bits of N change
    influence = np.zeros((k, 2 * k))

    for p, q, N in samples[:1000]:
        for i in range(k):
            # Flip bit i of p
            p_flipped = p ^ (1 << i)
            N_flipped = p_flipped * q
            # Which bits of N changed?
            diff = N ^ N_flipped
            for j in range(2 * k):
                if bit(diff, j):
                    influence[i, j] += 1

    influence /= min(len(samples), 1000)

    print(f"  Influence map (P(N_j changes | p_i flipped)):")
    print(f"  {'p\\N':>4}", end="")
    for j in range(min(2 * k, 20)):
        print(f"  N{j:<2d}", end="")
    print()
    for i in [0, 1, k//4, k//2, 3*k//4, k-2, k-1]:
        if i >= k:
            continue
        print(f"  p{i:<2d}:", end="")
        for j in range(min(2 * k, 20)):
            val = influence[i, j]
            if val > 0.8:
                marker = "████"
            elif val > 0.4:
                marker = "▓▓▓▓"
            elif val > 0.1:
                marker = "░░░░"
            else:
                marker = "    "
            print(f" {marker}", end="")
        print(f"  (nonzero: {np.sum(influence[i] > 0.01):.0f} bits)")

    # Average influence range
    for i in range(k):
        nonzero = np.where(influence[i] > 0.01)[0]
        if len(nonzero) > 0:
            span = nonzero[-1] - nonzero[0] + 1
        else:
            span = 0
        if i in [0, 1, k//4, k//2, 3*k//4, k-2, k-1]:
            print(f"  p[{i}] influences N bits {nonzero[0]}..{nonzero[-1]} (span={span})" if len(nonzero) > 0 else f"  p[{i}] no influence")

    return influence


def belief_propagation_test(k: int, num_trials: int = 100):
    """
    Can we recover p from N using iterative belief propagation
    on the binary multiplication factor graph?

    The factor graph has:
    - Variable nodes: bits of p, bits of q, carry bits
    - Factor nodes: the multiplication constraints at each bit position

    N_j = (Σ_{i+l=j} p_i × q_l + c_j) mod 2
    c_{j+1} = (Σ_{i+l=j} p_i × q_l + c_j) >> 1

    We observe N and try to infer p (and q) via message passing.
    """
    print(f"\n  --- Belief propagation test (k={k}) ---")
    from sympy import nextprime

    successes = 0
    partial_successes = 0

    for trial in range(num_trials):
        # Generate instance
        lo, hi = 1 << (k - 1), (1 << k) - 1
        p = nextprime(random.randint(lo, hi))
        while p > hi:
            p = nextprime(random.randint(lo, hi))
        q = nextprime(random.randint(lo, hi))
        while q > hi or q == p:
            q = nextprime(random.randint(lo, hi))
        if p > q:
            p, q = q, p
        N = p * q

        # Simple iterative decoding:
        # Start with uniform priors on p bits.
        # Use the known low bits of N to constrain low bits of p.
        # N mod 2 = 1 → p_0 = 1, q_0 = 1 (both odd)
        # N mod 4 → constrains p_1 + q_1
        # etc.

        p_known = [None] * k
        q_known = [None] * k
        p_known[0] = 1  # p is odd
        q_known[0] = 1  # q is odd

        # Iterative constraint propagation from LSB
        carry = 0
        bits_determined = 1
        for j in range(1, k):
            # At position j: N_j = (Σ_{i+l=j} p_i*q_l + carry) mod 2
            # With known bits, compute the known part of the sum
            known_sum = carry
            unknowns = []
            for i in range(j + 1):
                l = j - i
                if i < k and l < k:
                    if p_known[i] is not None and q_known[l] is not None:
                        known_sum += p_known[i] * q_known[l]
                    elif p_known[i] is not None:
                        unknowns.append(('q', l, p_known[i]))
                    elif q_known[l] is not None:
                        unknowns.append(('p', i, q_known[l]))
                    else:
                        unknowns.append(('pq', i, l))

            target = bit(N, j)

            # If all terms known, verify and compute carry
            if not unknowns:
                # Check consistency
                actual = known_sum
                if actual % 2 != target:
                    break  # inconsistency (shouldn't happen with correct propagation)
                carry = actual >> 1
                continue

            # If exactly one unknown product p_i*q_l:
            if len(unknowns) == 1:
                kind, *args = unknowns[0]
                if kind == 'q':
                    l, p_val = args
                    if p_val == 1:
                        q_known[l] = (target - known_sum % 2) % 2
                        carry = (known_sum + q_known[l]) >> 1
                        bits_determined += 1
                        continue
                elif kind == 'p':
                    i, q_val = args
                    if q_val == 1:
                        p_known[i] = (target - known_sum % 2) % 2
                        carry = (known_sum + p_known[i]) >> 1
                        bits_determined += 1
                        continue

            # Multiple unknowns — can't determine uniquely
            # Try: use the N bit to constrain parity
            carry = 0  # reset (approximation)

        # Check how many bits of p we determined correctly
        correct = 0
        for i in range(k):
            if p_known[i] is not None and p_known[i] == bit(p, i):
                correct += 1

        if correct == k:
            successes += 1
        if correct > k * 0.7:
            partial_successes += 1

    print(f"  Full recovery: {successes}/{num_trials}")
    print(f"  70%+ recovery: {partial_successes}/{num_trials}")
    return successes, partial_successes


if __name__ == "__main__":
    random.seed(42)

    for k in [8, 12, 16]:
        print(f"\n{'='*70}")
        print(f"  INFORMATION THEORY ANALYSIS: k={k} bit primes, N={2*k} bits")
        print(f"{'='*70}")

        num_samples = 5000
        samples = generate_balanced_semiprimes(k, num_samples)

        # 1. Mutual information map
        print(f"\n  --- Mutual Information I(p_i; N_j) ---")
        mi = mutual_information_bits(samples, k)
        print(f"  Max MI per p-bit:")
        for i in [0, 1, k//4, k//2, 3*k//4, k-2, k-1]:
            if i >= k: continue
            best_j = np.argmax(mi[i])
            print(f"    p[{i:2d}]: max MI = {mi[i, best_j]:.4f} at N[{best_j}], "
                  f"total MI = {np.sum(mi[i]):.4f}")

        # 2. Correlation map
        corr = bit_correlation_map(samples, k)
        print(f"\n  --- Bit correlations ---")
        for i in [0, 1, k//4, k//2, 3*k//4, k-2, k-1]:
            if i >= k: continue
            best_j = np.argmax(np.abs(corr[i]))
            print(f"    p[{i:2d}]: max |corr| = {abs(corr[i, best_j]):.4f} at N[{best_j}]")

        # 3. Carry propagation
        carry_propagation_analysis(k, num_samples)

        # 4. Conditional entropy
        conditional_entropy(samples[:2000], k)

        # 5. Belief propagation
        belief_propagation_test(k, num_trials=200)
