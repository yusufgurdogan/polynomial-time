#!/usr/bin/env python3
"""
Miller-Rabin non-trivial square-root-of-1 factoring.

For N = pq composite, pick random a coprime to N. Write N - 1 = 2^s * d, d odd.
Compute a^d, a^{2d}, a^{4d}, ..., a^{2^s d} mod N. If some a^{2^r d} = 1 mod N
but a^{2^{r-1} d} is NOT ±1 mod N, we have a non-trivial sqrt of 1 → factor
via gcd(a^{2^{r-1} d} - 1, N).

Per-attempt cost: O(log N) modular multiplications = polynomial.
If per-attempt SUCCESS probability is >= constant (not 1/sqrt(N)), this is
a polynomial-time factoring algorithm.

Test the rate vs bit size carefully.
"""
import math, random, sys, time
from sympy import nextprime
sys.stdout.reconfigure(line_buffering=True)


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    while True:
        p = nextprime(random.randint(lo, hi))
        if p > hi: continue
        q = nextprime(random.randint(lo, hi))
        if q > hi or q == p: continue
        N = p * q
        if N.bit_length() >= bits - 1:
            return N, min(p, q), max(p, q)


def mr_factor_attempt(N, a):
    """One Miller-Rabin-style factor attempt.
    Returns (factor, None) if found, (None, reason) otherwise.
    """
    if math.gcd(a, N) > 1:
        g = math.gcd(a, N)
        if g < N:
            return g, "gcd_direct"

    # N - 1 = 2^s * d
    d = N - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # Compute sequence
    prev = pow(a, d, N)
    if prev == 1 or prev == N - 1:
        return None, "no_witness"  # a is a non-witness

    for _ in range(s):
        cur = (prev * prev) % N
        if cur == 1:
            # prev is a square root of 1; if non-trivial, factor!
            if prev != 1 and prev != N - 1:
                g = math.gcd(prev - 1, N)
                if 1 < g < N:
                    return g, "nontrivial_sqrt"
            return None, "trivial_sqrt"
        if cur == N - 1:
            return None, "hit_minus1"
        prev = cur
    # Sequence ended without hitting 1: Fermat witness but no factor
    return None, "fermat_witness"


def run(bits, n_instances, n_tries):
    print(f"\n{'='*70}\n  bits={bits}, n={n_instances}, tries={n_tries}\n{'='*70}")
    factors = 0
    total_trials_to_factor = 0
    reasons = {}
    outcome_counts = {}
    t0 = time.time()

    for inst in range(n_instances):
        N, p, q = generate_semiprime(bits)
        hit = None
        tries_this_instance = 0
        for t in range(n_tries):
            tries_this_instance += 1
            a = random.randint(2, N - 2)
            res, reason = mr_factor_attempt(N, a)
            outcome_counts[reason or "factor"] = outcome_counts.get(reason or "factor", 0) + 1
            if res:
                hit = (res, reason); break
        if hit:
            factors += 1
            total_trials_to_factor += tries_this_instance
            reasons[hit[1]] = reasons.get(hit[1], 0) + 1

    dt = time.time() - t0
    rate = factors / n_instances
    avg_tries = total_trials_to_factor / max(factors, 1)
    log_inv = -math.log2(rate) if rate > 0 else float('inf')
    print(f"  MR factors: {factors}/{n_instances}  log2(1/rate)={log_inv:.2f}  avg_tries_if_factored={avg_tries:.1f}")
    print(f"  Reasons: {reasons}")
    print(f"  Per-attempt outcomes: {outcome_counts}")
    return bits, rate


if __name__ == "__main__":
    random.seed(4242)
    print("Miller-Rabin non-trivial sqrt-of-1 factoring\n")

    data = []
    for bits in [16, 24, 32, 48, 64, 96, 128, 192, 256]:
        data.append(run(bits, 100, 30))

    print(f"\n{'='*70}\n  SCALING\n{'='*70}")
    for i in range(len(data) - 1):
        b1, r1 = data[i]
        b2, r2 = data[i + 1]
        if r1 > 0 and r2 > 0:
            alpha = -(math.log2(r2) - math.log2(r1)) / (b2 - b1)
            print(f"  bits {b1}->{b2}: alpha = {alpha:.3f}")

    print()
    print("  If alpha ≈ 0: POLYNOMIAL TIME (constant success rate per attempt)")
    print("  If alpha > 0: sub-exponential or worse")
