"""
First-principles experiment: Can multi-base residue patterns reveal factors?

Key idea: N mod b for small bases b gives partial information about p.
Specifically, N ≡ p*q (mod b), so p*q ≡ N (mod b).
For each b, there are only O(b) possible (p mod b, q mod b) pairs.

If we combine constraints from many bases via CRT, can we narrow down p?

For primes b1, b2, ..., bk with product B = ∏bi:
- We know p*q mod bi for each i
- For each bi, there are ≤ bi candidate pairs (p mod bi, q mod bi)
- The total candidate count is ∏(number of solutions mod bi)
- CRT lifts each combination to a candidate p mod B

If B > √N, then we've found p. But the number of candidates
grows as ∏bi ≈ B, so we'd need to check B candidates — same as trial division!

UNLESS there's a way to prune candidates across bases faster than brute force.

This experiment tests whether lattice reduction or meet-in-the-middle
can reduce the candidate explosion.
"""

import math
import random
import time
from sympy import nextprime, isprime, factorint
from itertools import product as cartprod


def analyze_multi_base(N, p_true, q_true, max_base=100):
    """Analyze what multi-base residues reveal about factors."""
    print(f"N = {N} ({N.bit_length()} bits)")
    print(f"p = {p_true}, q = {q_true}")

    primes = []
    p = 2
    while p < max_base:
        if N % p != 0:  # skip primes dividing N
            primes.append(p)
        p = nextprime(p)

    print(f"\nUsing {len(primes)} prime bases: {primes[:20]}...")

    # For each prime base, find valid (p mod b, q mod b) pairs
    total_candidates = 1
    constraints = []
    for b in primes:
        n_mod_b = N % b
        valid_pairs = []
        for r in range(b):
            s = n_mod_b * pow(r, -1, b) % b if math.gcd(r, b) == 1 else -1
            if s >= 0 and r * s % b == n_mod_b:
                valid_pairs.append((r, s))
        constraints.append((b, valid_pairs))
        total_candidates *= len(valid_pairs)

        # Check that true p,q is among valid pairs
        p_mod = p_true % b
        q_mod = q_true % b
        found = any((p_mod == r and q_mod == s) or (p_mod == s and q_mod == r)
                     for r, s in valid_pairs)

    print(f"Total candidate combinations: {total_candidates:.2e}")
    print(f"vs trial division: {math.isqrt(N):.2e}")

    # Key question: is total_candidates << √N?
    log_candidates = sum(math.log2(len(pairs)) for _, pairs in constraints)
    log_sqrt_N = N.bit_length() / 2
    print(f"\nlog2(candidates) = {log_candidates:.1f}")
    print(f"log2(√N) = {log_sqrt_N:.1f}")
    print(f"Improvement ratio: {2**(log_sqrt_N - log_candidates):.2e}x")

    return log_candidates < log_sqrt_N


def experiment_lattice_pruning(N, p_true, q_true):
    """
    Can lattice reduction prune the candidate space?

    Idea: For each prime base b, p ≡ r (mod b) gives a linear constraint.
    If we have k constraints, p lies on a k-dimensional lattice coset.
    The shortest vector in this lattice might be close to p.
    """
    print(f"\n{'='*60}")
    print(f"Lattice pruning experiment")
    print(f"N = {N} ({N.bit_length()} bits)")

    # Collect residues
    bases = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    residues = []
    for b in bases:
        n_mod = N % b
        # p*q ≡ n_mod (mod b), so for each candidate p mod b,
        # q mod b = n_mod * p^(-1) mod b
        for r in range(1, b):  # p is not divisible by small primes (probably)
            if math.gcd(r, b) == 1:
                s = n_mod * pow(r, -1, b) % b
                if s > 0 and r <= s:  # p <= q convention
                    residues.append((b, r, s))

    # Check: how many valid (p mod B) values are there for B = ∏bases?
    # This is the CRT combination count
    B = 1
    for b in bases:
        B *= b

    print(f"Product of bases B = {B} ({B.bit_length()} bits)")
    print(f"√N = {math.isqrt(N)} ({(N.bit_length()+1)//2} bits)")
    print(f"B {'>' if B > math.isqrt(N) else '<'} √N")

    # The real question: for each base, how many solutions exist?
    for b in bases[:8]:
        n_mod = N % b
        count = sum(1 for r in range(1, b)
                    if math.gcd(r, b) == 1 and (n_mod * pow(r, -1, b) % b) > 0)
        print(f"  base {b:2d}: {count} solutions (out of {b-1}), ratio={count/(b-1):.2f}")


def experiment_digit_correlation(N, p_true, q_true):
    """
    Do the digits of N in different bases correlate with digits of p?

    If there's ANY polynomial-time computable function f(N) that predicts
    even ONE bit of p with probability > 1/2 + 1/poly(n), then repeated
    application could recover all of p.
    """
    print(f"\n{'='*60}")
    print(f"Digit correlation experiment")

    bits = N.bit_length()
    p_bits = [(p_true >> i) & 1 for i in range(bits)]

    # Test various functions of N as predictors of p's bits
    predictors = {
        'N mod 3': N % 3,
        'N mod 5': N % 5,
        'N mod 7': N % 7,
        'isqrt(N) parity': math.isqrt(N) % 2,
        'isqrt(N) mod 3': math.isqrt(N) % 3,
        'bit_count(N) parity': bin(N).count('1') % 2,
    }

    print(f"p bit 0 (LSB) = {p_bits[0]}")
    print(f"Predictors: {predictors}")

    # The bit_length//2 bit of p (most uncertain bit)
    mid = bits // 4
    print(f"p bit {mid} = {p_bits[mid]}")

    # Statistical test over many semiprimes
    rng = random.Random(42)
    n_trials = 1000
    correct = {name: 0 for name in ['sqrt_mod2', 'sqrt_mod3', 'bitcount', 'fermat']}

    for _ in range(n_trials):
        pp = nextprime(rng.randint(1 << 15, 1 << 16))
        qq = nextprime(rng.randint(1 << 15, 1 << 16))
        nn = pp * qq

        # Predict p's middle bit
        target = (pp >> 8) & 1  # bit 8 of p

        correct['sqrt_mod2'] += (math.isqrt(nn) % 2 == target)
        correct['sqrt_mod3'] += (math.isqrt(nn) % 3 == target)
        correct['bitcount'] += (bin(nn).count('1') % 2 == target)
        # Fermat's method starting point
        s = math.isqrt(nn)
        correct['fermat'] += ((s * s - nn) % 2 == target)

    print(f"\nPredicting bit 8 of p over {n_trials} random 32-bit semiprimes:")
    for name, c in correct.items():
        pct = c / n_trials * 100
        bias = abs(pct - 50)
        print(f"  {name:15s}: {pct:.1f}% correct (bias: {bias:.1f}%)")


def experiment_continued_fraction_structure(N, p_true, q_true):
    """
    The continued fraction of √N has period related to the fundamental
    solution of Pell's equation. Does the CF structure encode factoring info?

    Key insight: For N = pq, √N ≈ √p · √q. The CF expansion of √N
    interleaves information about √p and √q in a specific way.
    """
    print(f"\n{'='*60}")
    print(f"Continued fraction structure experiment")
    print(f"N = {N} ({N.bit_length()} bits)")

    # Compute CF expansion of √N
    cf = []
    a0 = math.isqrt(N)
    if a0 * a0 == N:
        print("Perfect square, skip")
        return

    m, d, a = 0, 1, a0
    seen = {}
    for i in range(200):
        m = d * a - m
        d = (N - m * m) // d
        if d == 0:
            break
        a = (a0 + m) // d
        cf.append(a)

        state = (m, d)
        if state in seen:
            period = i - seen[state]
            print(f"CF period: {period} (detected at step {i})")
            break
        seen[state] = i

    print(f"First 20 CF coefficients: {cf[:20]}")
    print(f"Max CF coefficient: {max(cf[:100]) if len(cf) >= 100 else max(cf)}")

    # Check convergents — do any give factors?
    h_prev, h_curr = 1, a0
    k_prev, k_curr = 0, 1

    factors_found = 0
    for i, ai in enumerate(cf[:100]):
        h_prev, h_curr = h_curr, ai * h_curr + h_prev
        k_prev, k_curr = k_curr, ai * k_curr + k_prev

        # Check if convergent reveals factor
        g = math.gcd(h_curr, N)
        if 1 < g < N:
            print(f"  Convergent {i}: h={h_curr}, gcd(h,N) = {g} = {'p' if g==p_true else 'q' if g==q_true else '??'}")
            factors_found += 1

        # Check h^2 mod N
        r = (h_curr * h_curr - N * k_curr * k_curr)
        if abs(r) < 1000:
            print(f"  Convergent {i}: h²-Nk² = {r}")

    if factors_found == 0:
        print("No factors found in convergents")


if __name__ == "__main__":
    rng = random.Random(42)

    # Test at different sizes
    for bits in [32, 48, 64]:
        half = bits // 2
        p = nextprime(rng.randint(1 << (half - 1), (1 << half) - 1))
        q = nextprime(rng.randint(1 << (half - 1), (1 << half) - 1))
        N = p * q

        print(f"\n{'#'*60}")
        print(f"# {bits}-bit semiprime")
        print(f"{'#'*60}")

        analyze_multi_base(N, p, q)
        experiment_digit_correlation(N, p, q)
        experiment_continued_fraction_structure(N, p, q)

    experiment_lattice_pruning(
        nextprime(rng.randint(1<<31, 1<<32)) * nextprime(rng.randint(1<<31, 1<<32)),
        0, 0  # we'll skip true factor checks
    )
