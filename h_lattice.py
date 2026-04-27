#!/usr/bin/env python3
"""
H-LATTICE EXPERIMENT: Does the N²-lift lattice avoid the genus obstruction?

The standard relation lattice L_R uses discrete logs. LLL finds only
genus-even vectors (killed by genus hyperplane).

The H-lattice uses Fermat quotients (p-adic logarithms) instead.
These are NOT proportional to discrete logs (the relationship involves
the Teichmüller lift). So the genus obstruction MIGHT NOT APPLY.

Test: build the H-lattice from H(b_i) values, run LLL, check if
short vectors give non-trivial gcd with N.
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
    """H(a) = floor(a^N mod N^2 / N)"""
    N2 = N * N
    aN = pow(a, N, N2)
    return aN // N

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

def run_experiment(bits, n_instances=30):
    print(f"\n{'='*70}")
    print(f"  H-LATTICE vs RELATION LATTICE: {bits}-bit semiprimes")
    print(f"{'='*70}")

    from fpylll import IntegerMatrix, LLL

    h_lattice_factors = 0
    r_lattice_factors = 0
    h_lattice_genus_odd = 0
    r_lattice_genus_odd = 0
    total = 0

    for inst in range(n_instances):
        N, p, q = generate_semiprime(bits)
        primes = [pr for pr in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47] if N % pr != 0]
        d = min(len(primes), 12)
        bases = primes[:d]

        # Compute H-values
        H_vals = [compute_H(b, N) for b in bases]

        # Compute discrete logs (for comparison, requires knowing p,q)
        g_p = primitive_root(p)
        g_q = primitive_root(q)
        if g_p is None or g_q is None:
            continue

        dlogs_p = [discrete_log(b % p, g_p, p) for b in bases]
        dlogs_q = [discrete_log(b % q, g_q, q) for b in bases]

        # Genus character: epsilon_i = dlog_p[i] % 2
        genus = [dl % 2 for dl in dlogs_p]

        # ============================================================
        # BUILD H-LATTICE: {e : sum e_i * H(b_i) ≡ 0 mod N}
        # Augmented lattice: [I_d | H_col] with last row [0...0 | N]
        # ============================================================
        dim_h = d + 1
        B_h = IntegerMatrix(dim_h, dim_h)
        for i in range(d):
            B_h[i, i] = 1
            B_h[i, d] = H_vals[i]
        B_h[d, d] = N

        LLL.reduction(B_h)

        # Extract kernel vectors (last column = 0)
        h_vectors = []
        for i in range(dim_h):
            v = [int(B_h[i, j]) for j in range(d)]
            aux = int(B_h[i, d])
            if aux == 0 and any(x != 0 for x in v):
                h_vectors.append(v)

        # ============================================================
        # BUILD RELATION LATTICE: {e : sum e_i * dlog_p(b_i) ≡ 0 mod (p-1)
        #                          AND sum e_i * dlog_q(b_i) ≡ 0 mod (q-1)}
        # ============================================================
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

        # ============================================================
        # ANALYSIS
        # ============================================================
        total += 1

        # Check H-lattice vectors
        h_factored = False
        h_genus_odd_count = 0
        for v in h_vectors:
            # Compute sum e_i * H(b_i) mod N
            s = sum(v[i] * H_vals[i] for i in range(d)) % N
            g = math.gcd(s, N) if s != 0 else N
            if 1 < g < N:
                h_factored = True

            # Check genus parity
            genus_val = sum(v[i] * genus[i] for i in range(d)) % 2
            if genus_val == 1:
                h_genus_odd_count += 1

        # Check relation lattice vectors
        r_factored = False
        r_genus_odd_count = 0
        for v in r_vectors:
            prod = 1
            for i in range(d):
                if v[i] >= 0:
                    prod = prod * pow(bases[i], v[i], N) % N
                else:
                    prod = prod * pow(bases[i], -v[i], N) % N
                    # Need modular inverse
            # Check genus
            genus_val = sum(v[i] * genus[i] for i in range(d)) % 2
            if genus_val == 1:
                r_genus_odd_count += 1

            # Direct factor test
            val = sum(v[i] * dlogs_p[i] for i in range(d)) % (p - 1)
            val_q = sum(v[i] * dlogs_q[i] for i in range(d)) % (q - 1)
            if val == 0 and val_q != 0:
                r_factored = True

        if h_factored:
            h_lattice_factors += 1
        if r_factored:
            r_lattice_factors += 1
        if h_genus_odd_count > 0:
            h_lattice_genus_odd += 1
        if r_genus_odd_count > 0:
            r_lattice_genus_odd += 1

        if inst < 5 or h_factored:
            print(f"  N={N}={p}×{q}  H-vecs:{len(h_vectors)} R-vecs:{len(r_vectors)}  "
                  f"H-genus-odd:{h_genus_odd_count} R-genus-odd:{r_genus_odd_count}  "
                  f"H-factor:{'YES!' if h_factored else 'no'} R-factor:{'YES!' if r_factored else 'no'}")

    print(f"\n  SUMMARY ({bits} bits, {total} instances):")
    print(f"    H-lattice: {h_lattice_factors}/{total} factored, "
          f"{h_lattice_genus_odd}/{total} had genus-odd vectors")
    print(f"    R-lattice: {r_lattice_factors}/{total} factored, "
          f"{r_lattice_genus_odd}/{total} had genus-odd vectors")

    if h_lattice_genus_odd > r_lattice_genus_odd:
        print(f"    *** H-LATTICE HAS MORE GENUS-ODD VECTORS! ***")
    elif h_lattice_genus_odd == r_lattice_genus_odd == 0:
        print(f"    Both lattices: zero genus-odd vectors (genus obstruction holds for both)")
    else:
        print(f"    R-lattice has same or more genus-odd vectors")

    return {
        'bits': bits, 'total': total,
        'h_factors': h_lattice_factors, 'r_factors': r_lattice_factors,
        'h_genus_odd': h_lattice_genus_odd, 'r_genus_odd': r_lattice_genus_odd,
    }

random.seed(42)
print("H-LATTICE EXPERIMENT")
print("Does the Fermat quotient lattice avoid the genus obstruction?\n")

results = []
for bits in [14, 16, 18, 20, 24]:
    r = run_experiment(bits, n_instances=30)
    results.append(r)

print(f"\n\n{'='*70}")
print(f"  FINAL COMPARISON")
print(f"{'='*70}")
print(f"  {'bits':>5} {'H-factor':>10} {'R-factor':>10} {'H-genus-odd':>12} {'R-genus-odd':>12}")
for r in results:
    print(f"  {r['bits']:>5} {r['h_factors']:>10}/{r['total']} {r['r_factors']:>10}/{r['total']} "
          f"{r['h_genus_odd']:>12}/{r['total']} {r['r_genus_odd']:>12}/{r['total']}")

print(f"\n  KEY QUESTION: Does H-lattice produce genus-odd vectors where R-lattice doesn't?")
any_better = any(r['h_genus_odd'] > r['r_genus_odd'] for r in results)
if any_better:
    print(f"  *** YES — the H-lattice BYPASSES the genus obstruction! ***")
    print(f"  This is potentially a breakthrough direction.")
else:
    print(f"  No. Both lattices are equally trapped by genus.")
    print(f"  The Fermat quotient does NOT escape the genus obstruction.")
