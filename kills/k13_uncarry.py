#!/usr/bin/env python3
"""
Carry-free convolution recovery ("uncarrying").

N = p * q where p, q are k-bit numbers with binary digits p_i, q_j.
The carry-free convolution: c_i = Σ_{j+l=i} p_j * q_l  (no mod, no carry)
Each c_i ∈ [0, min(i+1, k, 2k-1-i)] since p_j, q_l ∈ {0,1}.

The integer N relates to the convolution via:
  N_i = (c_i + carry_i) mod 2
  carry_{i+1} = (c_i + carry_i) // 2
  carry_0 = 0

This is a CHAIN constraint. Propagate from both ends:
  Low end: carry_0 = 0, c_0 = 1 (both p,q odd)
  High end: carry after position 2k must be 0

Question: do these constraints uniquely determine the c_i sequence?
If yes → factor the polynomial p(x)*q(x) in polynomial time → done.
"""

import math
import random
import sys
import numpy as np
from sympy import nextprime, factor as sym_factor, Poly, Symbol, ZZ

sys.stdout.reconfigure(line_buffering=True)


def get_bits(n, num_bits):
    return [(n >> i) & 1 for i in range(num_bits)]


def true_convolution(p, q, k):
    """Compute the carry-free convolution of p and q's bit representations."""
    p_bits = get_bits(p, k)
    q_bits = get_bits(q, k)
    conv = [0] * (2 * k)
    for i in range(k):
        for j in range(k):
            conv[i + j] += p_bits[i] * q_bits[j]
    return conv


def n_to_bits(N, num_bits):
    return [(N >> i) & 1 for i in range(num_bits)]


def forward_propagate(N_bits, max_c, num_positions):
    """
    Forward pass: propagate carry constraints from LSB to MSB.

    At each position i, maintain the set of possible (c_i, carry_i) pairs.
    Returns list of sets of possible (c_i, carry_out) at each position.
    """
    # Position 0: carry_in = 0
    possible = [set() for _ in range(num_positions)]

    # Start: carry_in = 0 at position 0
    carry_states = {0}  # possible carry_in values at position 0

    for i in range(num_positions):
        max_ci = max_c[i]
        new_carry_states = set()

        for carry_in in carry_states:
            for ci in range(max_ci + 1):
                total = ci + carry_in
                if total % 2 == N_bits[i]:
                    carry_out = total // 2
                    possible[i].add((ci, carry_in, carry_out))
                    new_carry_states.add(carry_out)

        carry_states = new_carry_states

        if not carry_states:
            break

    return possible, carry_states


def backward_propagate(N_bits, max_c, num_positions, final_carry=0):
    """
    Backward pass: propagate from MSB to LSB.
    The carry AFTER the last position must be 0 (or small).
    """
    possible = [set() for _ in range(num_positions)]

    # Start from the end: carry_out at last position must be final_carry
    carry_out_states = {final_carry}

    for i in range(num_positions - 1, -1, -1):
        max_ci = max_c[i]
        new_carry_in_states = set()

        for carry_out in carry_out_states:
            for ci in range(max_ci + 1):
                # total = ci + carry_in, carry_out = total // 2, N_i = total % 2
                # So total = 2 * carry_out + N_bits[i]
                total = 2 * carry_out + N_bits[i]
                carry_in = total - ci
                if carry_in >= 0:
                    possible[i].add((ci, carry_in, carry_out))
                    new_carry_in_states.add(carry_in)

        carry_out_states = new_carry_in_states

    return possible


def intersect_propagation(fwd, bwd, num_positions):
    """Intersect forward and backward possible sets."""
    result = []
    for i in range(num_positions):
        intersection = fwd[i] & bwd[i]
        result.append(intersection)
    return result


def recover_convolution(N, k):
    """
    Try to recover the carry-free convolution from N.
    Returns (convolution, num_candidates) or (None, num_candidates).
    """
    num_pos = 2 * k
    N_bits = n_to_bits(N, num_pos + 1)

    # Max possible c_i at each position
    max_c = [0] * num_pos
    for i in range(num_pos):
        # Number of (j, l) pairs with j+l = i, 0 ≤ j < k, 0 ≤ l < k
        max_c[i] = min(i + 1, k, 2 * k - 1 - i)

    # Forward propagation
    fwd, final_carries = forward_propagate(N_bits, max_c, num_pos)

    # The carry after the last position should match remaining high bits of N
    # N might have bits beyond position 2k-1
    remaining = N >> num_pos

    # Backward propagation
    bwd = backward_propagate(N_bits, max_c, num_pos, final_carry=remaining)

    # Intersect
    combined = intersect_propagation(fwd, bwd, num_pos)

    # Count unique c_i values at each position
    c_options = []
    for i in range(num_pos):
        ci_vals = set(ci for ci, _, _ in combined[i])
        c_options.append(ci_vals)

    # Check if uniquely determined
    unique = all(len(opts) == 1 for opts in c_options)
    total_candidates = 1
    for opts in c_options:
        total_candidates *= max(len(opts), 1)

    if unique:
        conv = [list(opts)[0] for opts in c_options]
        return conv, 1

    # Try to extract by following the unique chain
    # Greedy: pick the unique path if it exists
    conv = []
    carry = 0
    num_ambiguous = 0
    for i in range(num_pos):
        valid_ci = set()
        for ci, c_in, c_out in combined[i]:
            if c_in == carry:
                valid_ci.add((ci, c_out))

        if len(valid_ci) == 1:
            ci, carry = list(valid_ci)[0]
            conv.append(ci)
        elif len(valid_ci) > 1:
            num_ambiguous += 1
            # Pick smallest (arbitrary)
            ci, carry = min(valid_ci)
            conv.append(ci)
        else:
            conv.append(-1)  # impossible
            break

    return conv if len(conv) == num_pos else None, total_candidates


def factor_from_convolution(conv, k):
    """
    Given carry-free convolution c_i = Σ p_j q_l (j+l=i),
    this is the polynomial product p(x)*q(x) evaluated coefficient-wise.
    Factor it over Z[x] to recover p(x) and q(x), then evaluate at x=2.
    """
    x = Symbol('x')

    # Build polynomial from convolution
    poly_expr = sum(c * x**i for i, c in enumerate(conv) if c != 0)

    if poly_expr == 0:
        return None

    try:
        p = Poly(poly_expr, x, domain=ZZ)
        factors = p.factor_list()

        # factors = (content, [(factor1, mult1), (factor2, mult2), ...])
        content, factor_pairs = factors

        # Evaluate each factor at x = 2 to get integer factors
        int_factors = []
        for f, mult in factor_pairs:
            val = int(f.eval(2))
            for _ in range(mult):
                int_factors.append(abs(val))

        return int_factors
    except Exception as e:
        return None


def experiment(bits_list=None, num_instances=100):
    """Run the uncarrying experiment."""
    if bits_list is None:
        bits_list = [8, 10, 12, 14, 16, 20, 24, 28, 32]

    print(f"{'='*75}")
    print(f"  CARRY-FREE CONVOLUTION RECOVERY")
    print(f"{'='*75}")

    for k in bits_list:
        print(f"\n  --- k = {k} ({2*k}-bit N) ---")

        unique_count = 0
        factored_count = 0
        total = 0
        ambiguity_sizes = []

        for _ in range(num_instances):
            lo, hi = 1 << (k - 1), (1 << k) - 1
            p = nextprime(random.randint(lo, hi))
            if p > hi: continue
            q = nextprime(random.randint(lo, hi))
            if q > hi or q == p: continue
            if p > q: p, q = q, p
            N = p * q

            total += 1
            true_conv = true_convolution(p, q, k)

            # Try to recover
            recovered, num_candidates = recover_convolution(N, k)

            if recovered is not None and num_candidates == 1:
                unique_count += 1
                # Verify
                if recovered == true_conv[:len(recovered)]:
                    # Try polynomial factoring
                    int_factors = factor_from_convolution(recovered, k)
                    if int_factors and (p in int_factors or q in int_factors):
                        factored_count += 1

            ambiguity_sizes.append(num_candidates)

        if total == 0:
            continue

        avg_ambiguity = np.mean(np.log2(np.array(ambiguity_sizes, dtype=float) + 1))
        median_ambiguity = np.median(ambiguity_sizes)

        print(f"  Unique recovery: {unique_count}/{total} ({unique_count/total:.1%})")
        print(f"  Factored:        {factored_count}/{total} ({factored_count/total:.1%})")
        print(f"  Median candidates: {median_ambiguity:.0f}")
        print(f"  Avg log2(candidates): {avg_ambiguity:.1f}")

        if unique_count == 0 and median_ambiguity > 1000:
            print(f"  Ambiguity is exponential — stopping larger sizes")
            break


if __name__ == "__main__":
    random.seed(42)
    experiment(bits_list=[6, 8, 10, 12, 14, 16, 20], num_instances=100)
