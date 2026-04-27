#!/usr/bin/env python3
"""
Binomial AKS-discrepancy test.

Theory: C(N, k) mod N is a q-multiple iff p|k, a p-multiple iff q|k, else 0.
Reduction mod (X^r - 1) sums over residue classes; classes mod poly-r
contain both p-multiples and q-multiples, so S_j is typically uniform noise.

Test: for various r, compute S_j = sum_{k ≡ j mod r} C(N, k) mod N via
polynomial exponentiation (X+1)^N mod (N, X^r - 1). Check gcd(S_j - c, N)
for c ∈ {0, ±1, 2^N, (1+any_corrections)}. Compare to random-gcd baseline.

Hypothesis: H-lattice-like death — signal = birthday rate, 2d/√N.
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


def poly_mul_mod(A, B, r, N):
    """Multiply two polynomials A, B of degree < r in Z/NZ[X]/(X^r - 1)."""
    C = [0] * r
    for i, ai in enumerate(A):
        if ai == 0: continue
        for j, bj in enumerate(B):
            if bj == 0: continue
            C[(i + j) % r] = (C[(i + j) % r] + ai * bj) % N
    return C


def poly_pow_mod(A, e, r, N):
    """A^e in Z/NZ[X]/(X^r - 1)."""
    result = [1] + [0] * (r - 1)
    base = list(A)
    while e > 0:
        if e & 1:
            result = poly_mul_mod(result, base, r, N)
        base = poly_mul_mod(base, base, r, N)
        e >>= 1
    return result


def try_factor_from_poly(poly, N, r_mod, a):
    """poly = (X+a)^N mod (N, X^r - 1). Expected if prime: X^(N mod r) + a^N mod N.
    Actually a^N mod N for N composite is not necessarily a. Use Fermat-liar value."""
    r = len(poly)
    # "Ideal" polynomial if N were prime: X^(N mod r) + a^N mod N
    # But a^N mod N != a for composite N; use actual expected value by Fermat
    N_mod_r = (1 << (N.bit_length() - 1)) % r  # dummy
    # Actually compute N mod r directly
    N_mod_r = None
    # pass in as parameter would be cleaner; compute here
    return None  # placeholder, rewritten below


def run_instance(N, p, q, r, a):
    """For this N, r, a: compute (X+a)^N mod (N, X^r - 1) and check all coefficients."""
    # Build polynomial X + a
    base = [a % N, 1] + [0] * (r - 2) if r >= 2 else [a % N]
    if r < 2:
        return None

    poly = poly_pow_mod(base, N, r, N)
    # What "should" poly be if we fabricate a target? We test gcd(coef, N) for each coef.
    # Also gcd(coef - 1, N), gcd(coef ± a^N mod N, N), etc.

    aN_mod_N = pow(a, N, N)  # poly-time compute

    for j, c in enumerate(poly):
        for delta in (0, 1, -1, aN_mod_N, (aN_mod_N - a) % N, (1 + aN_mod_N) % N):
            val = (c - delta) % N
            if val == 0: continue
            g = math.gcd(val, N)
            if 1 < g < N:
                return (g, j, delta)
    return None


def run(bits, n_instances=50, r_list=(3, 5, 7, 11, 13, 17), a_list=(2, 3, 5)):
    print(f"\n{'='*70}\n  bits={bits}, n={n_instances}, r_list={r_list}\n{'='*70}")

    factors = 0
    random_factors = 0  # baseline: random small numbers, check gcd with N

    # Number of (r, a, j, delta) tuples probed per instance
    n_probes_per_instance = sum(r_list) * len(a_list) * 6  # 6 delta options

    for inst in range(n_instances):
        N, p, q = generate_semiprime(bits)
        got_factor = None
        for r in r_list:
            for a in a_list:
                res = run_instance(N, p, q, r, a)
                if res:
                    got_factor = res
                    break
            if got_factor: break

        if got_factor:
            factors += 1

        # Birthday baseline with same number of probes
        rbird = False
        for _ in range(n_probes_per_instance):
            x = random.randint(2, N - 2)
            g = math.gcd(x, N)
            if 1 < g < N:
                rbird = True; break
        if rbird: random_factors += 1

    print(f"  Binomial-AKS: {factors}/{n_instances}")
    print(f"  Random birthday (~{n_probes_per_instance} probes/inst): {random_factors}/{n_instances}")
    return factors, random_factors, n_instances


if __name__ == "__main__":
    random.seed(42)
    print("Binomial AKS-discrepancy factoring test")
    print("Does (X+a)^N mod (N, X^r - 1) give factor-gcds beyond birthday?\n")

    rows = []
    for bits in [14, 20, 26, 32, 40, 48]:
        t0 = time.time()
        b, r, n = run(bits, n_instances=50)
        dt = time.time() - t0
        rows.append((bits, b, r, n, dt))

    print(f"\n\n{'='*70}\n  VERDICT\n{'='*70}")
    print(f"  {'bits':>5} {'Binomial':>12} {'Random':>12}  delta  time")
    for bits, b, r, n, dt in rows:
        print(f"  {bits:>5} {b:>5}/{n:<4} {r:>5}/{n:<4}  {b-r:+d}    {dt:.1f}s")

    any_delta = any(b > r for bits, b, r, n, dt in rows)
    if any_delta:
        print("\n  Binomial beats random somewhere. Check which (r, a, j, delta) triggered.")
    else:
        print("\n  Binomial ≈ random everywhere. Confirmed: mod-(X^r-1) reduction")
        print("  destroys factor info by mixing p-multiples with q-multiples.")
