#!/usr/bin/env python3
"""
H-LATTICE v2: Corrected factor tests.

v1's factor test was broken: it computed s = sum e_i * H(b_i) mod N, then
checked gcd(s, N) for 1 < g < N. But LLL produces exact kernel vectors
(s = 0), forcing g = N — the test could never succeed.

v2 tries every sensible factor test on each lattice vector e:
  (A) gcd(prod b_i^{e_i} ± 1, N)     # multiplicative relation test
  (B) gcd(prod b_i^{e_i}, N)          # direct
  (C) mod-N^2 lift: check N-component for factor
  (D) even-exponent square-root trick (genus-even vectors in L_R)
  (E) pairwise: combine two vectors, repeat (A-D)

If ANY of these factor N, we report it. If none do across 50 instances per
bit size through 28 bits, the H-lattice direction dies and we know why.
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


def compute_H(a, N):
    """H(a) = floor(a^N mod N^2 / N) — high-N part of the lift."""
    N2 = N * N
    aN = pow(a, N, N2)
    return aN // N, aN % N  # (H, residue mod N)


def primitive_root(p):
    if p == 2: return 1
    phi = p - 1
    factors = set()
    n = phi; d = 2
    while d * d <= n:
        while n % d == 0: factors.add(d); n //= d
        d += 1
    if n > 1: factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    return None


def discrete_log(a, g, p):
    a = a % p
    if a == 0: return 0
    cur = 1
    for k in range(p):
        if cur == a: return k
        cur = (cur * g) % p
    return 0


def multiplicative_product(e, bases, N):
    """Compute prod b_i^{e_i} mod N. Returns (P, early_factor_or_None).

    If any inversion fails because gcd(b_i, N) > 1, returns that factor."""
    P = 1
    for i, ei in enumerate(e):
        if ei == 0: continue
        b = bases[i]
        if ei > 0:
            P = P * pow(b, ei, N) % N
        else:
            g = math.gcd(b, N)
            if 1 < g < N:
                return None, g
            try:
                inv = pow(b, -1, N)
            except ValueError:
                return None, None
            P = P * pow(inv, -ei, N) % N
    return P, None


def try_factor_from_vector(e, bases, N):
    """All factor tests for lattice vector e."""
    # (A/B) multiplicative product mod N, check gcd(P ± 1, N), gcd(P, N)
    P, early = multiplicative_product(e, bases, N)
    if early is not None:
        return early, "early_gcd"
    if P is None:
        return None, None

    for delta in (0, 1, -1):
        val = (P - delta) % N
        if val == 0: continue
        g = math.gcd(val, N)
        if 1 < g < N:
            return g, f"P{delta:+d}" if delta else "P"

    # (C) lift to mod N^2, check N-component
    N2 = N * N
    P2, early2 = multiplicative_product(e, bases, N2)
    if early2 is not None and 1 < early2 < N:
        return early2, "lift_gcd"
    if P2 is not None:
        # (P2 - P) is always divisible by N if P is the reduction of P2 mod N.
        # Its high part (P2 - P) / N mod N might share a factor with N.
        # But typically P2 mod N == P (trivially), so (P2 - P) / N is well-defined.
        if P2 % N == P:
            k = (P2 - P) // N
            g = math.gcd(k % N, N)
            if 1 < g < N:
                return g, "lift_k"
        # Also check P2 ± 1 mod N^2
        for delta in (1, -1):
            val = (P2 - delta) % N2
            if val == 0: continue
            g = math.gcd(val, N)
            if 1 < g < N:
                return g, f"P2{delta:+d}"

    # (D) square-root trick for even-exponent vectors
    if all(ei % 2 == 0 for ei in e) and any(ei != 0 for ei in e):
        half_e = [ei // 2 for ei in e]
        Y, earlyY = multiplicative_product(half_e, bases, N)
        if earlyY is not None and 1 < earlyY < N:
            return earlyY, "sqrt_gcd"
        if Y is not None:
            # If e is a multiplicative relation (P ≡ 1 mod N), then Y^2 ≡ 1 mod N.
            # Non-trivial square roots of 1 give factoring.
            for delta in (1, -1):
                val = (Y - delta) % N
                if val == 0: continue
                g = math.gcd(val, N)
                if 1 < g < N:
                    return g, f"sqrt{delta:+d}"
    return None, None


def run_experiment(bits, n_instances=50):
    print(f"\n{'='*70}")
    print(f"  H-LATTICE v2: {bits}-bit semiprimes, {n_instances} instances")
    print(f"{'='*70}")

    from fpylll import IntegerMatrix, LLL

    h_factors = 0
    r_factors = 0
    h_genus_odd_total = 0
    factor_reasons = {}

    for inst in range(n_instances):
        N, p, q = generate_semiprime(bits)
        primes = [pr for pr in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47] if N % pr != 0]
        d = min(len(primes), 12)
        bases = primes[:d]

        # H values
        H_vals, R_vals = [], []
        for b in bases:
            h, r = compute_H(b, N)
            H_vals.append(h); R_vals.append(r)

        # Discrete logs for genus check (requires p,q; this is only diagnostic)
        g_p = primitive_root(p)
        g_q = primitive_root(q)
        if g_p is None or g_q is None:
            continue
        dlogs_p = [discrete_log(b % p, g_p, p) for b in bases]
        dlogs_q = [discrete_log(b % q, g_q, q) for b in bases]
        genus = [dl % 2 for dl in dlogs_p]

        # --- H-lattice ---
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
            aux = int(B_h[i, d])
            # Keep BOTH exact kernel (aux=0) AND near-kernel (small aux)
            if any(x != 0 for x in v):
                h_vectors.append((v, aux))

        h_factored = None
        h_reason = None
        h_genus_odd_here = 0

        # (A-D) Per-vector tests
        for v, aux in h_vectors:
            genus_parity = sum(v[i] * genus[i] for i in range(d)) % 2
            if genus_parity == 1:
                h_genus_odd_here += 1
            f, reason = try_factor_from_vector(v, bases, N)
            if f:
                h_factored = f; h_reason = reason; break

        # (E) Pairwise combinations
        if not h_factored:
            for i in range(len(h_vectors)):
                if h_factored: break
                for j in range(i+1, len(h_vectors)):
                    for sign in (1, -1):
                        combined = [h_vectors[i][0][k] + sign * h_vectors[j][0][k] for k in range(d)]
                        if all(x == 0 for x in combined): continue
                        f, reason = try_factor_from_vector(combined, bases, N)
                        if f:
                            h_factored = f; h_reason = f"pair:{reason}"; break
                    if h_factored: break

        if h_genus_odd_here > 0:
            h_genus_odd_total += 1

        # --- Relation lattice (baseline, uses p/q knowledge) ---
        dim_r = d + 2
        B_r = IntegerMatrix(dim_r, dim_r)
        for i in range(d):
            B_r[i, i] = 1
            B_r[i, d] = dlogs_p[i]
            B_r[i, d + 1] = dlogs_q[i]
        B_r[d, d] = p - 1
        B_r[d + 1, d + 1] = q - 1
        LLL.reduction(B_r)

        r_vectors = []
        for i in range(dim_r):
            v = [int(B_r[i, j]) for j in range(d)]
            aux = [int(B_r[i, j]) for j in range(d, dim_r)]
            if all(a == 0 for a in aux) and any(x != 0 for x in v):
                r_vectors.append(v)

        r_factored = None
        for v in r_vectors:
            f, reason = try_factor_from_vector(v, bases, N)
            if f:
                r_factored = f; break

        if h_factored:
            h_factors += 1
            factor_reasons[h_reason] = factor_reasons.get(h_reason, 0) + 1
            print(f"  [{inst:3d}] N={N}={p}×{q}  H-FACTORED via {h_reason}: {h_factored}")
        if r_factored:
            r_factors += 1

    print(f"\n  SUMMARY ({bits} bits, {n_instances} instances):")
    print(f"    H-lattice factored: {h_factors}/{n_instances}")
    print(f"    R-lattice factored: {r_factors}/{n_instances}")
    print(f"    H-lattice genus-odd vectors present: {h_genus_odd_total}/{n_instances}")
    if factor_reasons:
        print(f"    Factor-reason histogram: {factor_reasons}")

    return {
        'bits': bits, 'n': n_instances,
        'h_factors': h_factors, 'r_factors': r_factors,
        'h_genus_odd': h_genus_odd_total,
        'reasons': factor_reasons,
    }


if __name__ == "__main__":
    random.seed(42)
    print("H-LATTICE v2 — corrected factor tests")
    print("Does the Fermat-quotient lattice actually yield factorizations?\n")

    results = []
    for bits in [14, 18, 22, 26, 30]:
        r = run_experiment(bits, n_instances=50)
        results.append(r)

    print(f"\n\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'bits':>5} {'H-factor':>10} {'R-factor':>10} {'H-genus-odd':>12}  reasons")
    for r in results:
        print(f"  {r['bits']:>5} {r['h_factors']:>5}/{r['n']:<4} {r['r_factors']:>5}/{r['n']:<4} "
              f"{r['h_genus_odd']:>7}/{r['n']:<4}  {dict(r['reasons'])}")

    any_factored = any(r['h_factors'] > 0 for r in results)
    if any_factored:
        print(f"\n  *** H-lattice factored some instances. Scrutinize the reasons carefully. ***")
    else:
        print(f"\n  H-lattice did not factor any instance, despite ~100% genus-odd rate.")
        print(f"  Conclusion: escaping genus ≠ factoring. The H-kernel vectors")
        print(f"  are not multiplicative relations, and no tested projection extracts a factor.")
