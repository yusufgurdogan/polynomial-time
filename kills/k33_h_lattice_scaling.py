#!/usr/bin/env python3
"""
Does the H-lattice beat birthday?

If the H-lattice is a real structural factoring mechanism, its success rate
should NOT decay as 2*d^2 / sqrt(N). If it's birthday-class luck amplified
by LLL, it should match that decay exactly.

Null hypothesis: H-factoring rate = 1 - (1 - 2/sqrt(N))^(O(d^2))
If observed rate >> null prediction: real signal. If ≈ null: birthday artifact.
"""
import math, random, sys
from sympy import nextprime
sys.stdout.reconfigure(line_buffering=True)

# Import from v2
from k33_h_lattice_v2 import (generate_semiprime, compute_H, try_factor_from_vector,
                              primitive_root, discrete_log)

from fpylll import IntegerMatrix, LLL


def h_lattice_factor_attempt(N, bases, H_vals):
    """Run H-lattice + exhaustive factor tests. Return factor or None."""
    d = len(bases)
    dim_h = d + 1
    B_h = IntegerMatrix(dim_h, dim_h)
    for i in range(d):
        B_h[i, i] = 1
        B_h[i, d] = H_vals[i]
    B_h[d, d] = N
    LLL.reduction(B_h)

    h_vectors = []
    for i in range(dim_h):
        v = [int(B_h[i, j]) for j in range(d)]
        if any(x != 0 for x in v):
            h_vectors.append(v)

    for v in h_vectors:
        f, _ = try_factor_from_vector(v, bases, N)
        if f:
            return f

    # pairs
    for i in range(len(h_vectors)):
        for j in range(i+1, len(h_vectors)):
            for sign in (1, -1):
                combined = [h_vectors[i][k] + sign * h_vectors[j][k] for k in range(d)]
                if all(x == 0 for x in combined): continue
                f, _ = try_factor_from_vector(combined, bases, N)
                if f: return f
    return None


def birthday_random_attempt(N, bases, n_tries):
    """Baseline: random small exponent vectors, same factor tests."""
    d = len(bases)
    for _ in range(n_tries):
        v = [random.randint(-3, 3) for _ in range(d)]
        if all(x == 0 for x in v): continue
        f, _ = try_factor_from_vector(v, bases, N)
        if f: return f
    return None


def run(bits, n_instances):
    print(f"\n{'='*70}\n  bits={bits}, n={n_instances}\n{'='*70}")

    h_factors = 0
    birthday_factors = 0

    # Count attempts inside h_lattice for fair comparison
    # h_lattice tries: d+1 single vectors + C(d+1,2)*2 pairs ≈ d^2 attempts
    n_attempts = 12*12 + 12 + 12  # matches dimension=12+1, pairs both signs

    for inst in range(n_instances):
        N, p, q = generate_semiprime(bits)
        primes = [pr for pr in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47] if N % pr != 0]
        d = min(len(primes), 12)
        bases = primes[:d]

        H_vals = [compute_H(b, N)[0] for b in bases]

        # H-lattice attempt
        f = h_lattice_factor_attempt(N, bases, H_vals)
        if f: h_factors += 1

        # Birthday baseline: same number of random attempts
        f = birthday_random_attempt(N, bases, n_attempts)
        if f: birthday_factors += 1

    print(f"  H-lattice:     {h_factors}/{n_instances}")
    print(f"  Random (same #attempts): {birthday_factors}/{n_instances}")
    return h_factors, birthday_factors, n_instances


if __name__ == "__main__":
    random.seed(123)
    print("H-lattice vs random-exponent birthday baseline")
    print("If H-lattice matches random, it's birthday-class (dead by scaling).\n")

    rows = []
    for bits in [14, 20, 26, 32, 40]:
        h, b, n = run(bits, 100)
        rows.append((bits, h, b, n))

    print(f"\n\n{'='*70}\n  VERDICT\n{'='*70}")
    print(f"  {'bits':>5} {'H-lattice':>12} {'random':>12}  delta")
    for bits, h, b, n in rows:
        print(f"  {bits:>5} {h:>6}/{n:<4} {b:>6}/{n:<4}  {h-b:+d}")

    print()
    print("  If H-lattice ≈ random at every bit size: birthday artifact (kill #33).")
    print("  If H-lattice >> random at large bits: real structural signal.")
