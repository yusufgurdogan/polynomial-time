#!/usr/bin/env python3
"""
Direction A: Structured bases for Regev's lattice.

Replace generic small primes with bases derived from N itself.
Measure whether genus-odd vectors become shorter.

Basis types:
  A. Control: first d small primes (Schnorr/Regev standard)
  B. CF convergents: numerators/denominators of √N continued fraction
  C. Infrastructure elements: a-values from reduced forms of disc 4N
  D. Small-norm algebraic: elements of Z[√N] with small norm, reduced mod N
  E. Hybrid: mix of small primes + N-derived elements

For each basis type, build L_R, LLL-reduce, and measure:
  1. Shortest vector norm
  2. Genus parity of all short vectors (even vs odd)
  3. Whether any genus-odd vector appears in the top-k shortest
  4. If genus-odd found: does it factor N?

Kill criterion: genus-odd vectors shortest in <1% of 500 instances → dead.
"""

import math
import random
import sys
import time
import numpy as np
from collections import Counter
from fpylll import IntegerMatrix, LLL

sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# Helpers
# =============================================================================

def small_primes(limit):
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            for j in range(i*i, limit + 1, i):
                s[j] = False
    return [i for i in range(limit + 1) if s[i]]


def primitive_root(p):
    if p == 2: return 1
    phi = p - 1
    factors = set()
    n = phi
    d = 2
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


# =============================================================================
# Basis generators
# =============================================================================

def basis_small_primes(N, d):
    """Control: first d small primes not dividing N."""
    return [p for p in small_primes(200) if N % p != 0][:d]


def basis_cf_convergents(N, d):
    """Continued fraction convergents of √N, reduced mod N."""
    sqrt_N = math.isqrt(N)
    bases = []
    seen = set()

    m, dd, a0 = 0, 1, sqrt_N
    a = a0
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    for _ in range(d * 10):
        m = dd * a - m
        dd_new = (N - m * m) // dd
        if dd_new == 0: break
        dd = dd_new
        a = (a0 + m) // dd
        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev

        # Use both numerator and denominator as candidate bases
        for val in [p_curr % N, q_curr % N]:
            if val > 1 and val < N and math.gcd(val, N) == 1 and val not in seen:
                seen.add(val)
                bases.append(val)
                if len(bases) >= d:
                    return bases

    # Pad with small primes if not enough
    for p in small_primes(200):
        if len(bases) >= d: break
        if N % p != 0 and p not in seen:
            bases.append(p)
            seen.add(p)

    return bases[:d]


def basis_infrastructure(N, d):
    """a-values from reduced forms in the principal cycle of disc 4N."""
    D = 4 * N
    sqrt_D = math.isqrt(D)
    if (sqrt_D + 1) ** 2 <= D: sqrt_D += 1

    bases = []
    seen = set()

    # Walk the principal cycle
    a, b, c = 1, 0, -N  # principal form

    # Reduce first
    for _ in range(1000):
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

    # Now walk and collect a-values
    start = (a, b, c)
    for step in range(d * 20):
        val = abs(a)
        if val > 1 and val < N and math.gcd(val, N) == 1 and val not in seen:
            seen.add(val)
            bases.append(val)
            if len(bases) >= d:
                return bases

        # rho step
        if c == 0: break
        abs_c = abs(c)
        r = (-b) % (2 * abs_c)
        while r <= sqrt_D - 2 * abs_c: r += 2 * abs_c
        if r >= sqrt_D: r -= 2 * abs_c
        if r <= 0: r += 2 * abs_c
        c_new = (r * r - D) // (4 * c)
        a, b, c = c, r, c_new

        if (a, b, c) == start and step > 0:
            break

    # Pad
    for p in small_primes(200):
        if len(bases) >= d: break
        if N % p != 0 and p not in seen:
            bases.append(p)
            seen.add(p)

    return bases[:d]


def basis_small_norm(N, d):
    """Elements of Z[√N] with small norm, reduced to (Z/NZ)*."""
    sqrt_N = math.isqrt(N)
    bases = []
    seen = set()

    # Elements a + b√N have norm a² - N·b²
    # We want |a² - N·b²| small, and (a + b·√N) mod N = a + b·isqrt(N) mod N
    for b in range(1, d * 5):
        for a in range(max(1, int(math.sqrt(N) * b) - d), int(math.sqrt(N) * b) + d + 1):
            norm = abs(a * a - N * b * b)
            if norm == 0: continue
            val = (a + b * sqrt_N) % N
            if val > 1 and val < N and math.gcd(val, N) == 1 and val not in seen:
                seen.add(val)
                bases.append(val)
                if len(bases) >= d:
                    return bases

    # Pad
    for p in small_primes(200):
        if len(bases) >= d: break
        if N % p != 0 and p not in seen:
            bases.append(p)
            seen.add(p)

    return bases[:d]


def basis_hybrid(N, d):
    """Mix: half small primes, half CF convergents."""
    half = d // 2
    sp = basis_small_primes(N, half)
    cf = basis_cf_convergents(N, d - half)
    # Deduplicate
    seen = set(sp)
    result = list(sp)
    for b in cf:
        if b not in seen:
            result.append(b)
            seen.add(b)
    while len(result) < d:
        for p in small_primes(500):
            if p not in seen and N % p != 0:
                result.append(p)
                seen.add(p)
                if len(result) >= d: break
    return result[:d]


BASIS_TYPES = {
    'small_primes': basis_small_primes,
    'cf_convergents': basis_cf_convergents,
    'infrastructure': basis_infrastructure,
    'small_norm': basis_small_norm,
    'hybrid': basis_hybrid,
}


# =============================================================================
# Lattice construction + genus measurement
# =============================================================================

def build_and_analyze(N, p, q, bases, top_k=20):
    """
    Build L_R from bases, LLL-reduce, extract short vectors,
    label genus parity, return stats.
    """
    d = len(bases)

    # Compute discrete logs and genus characters
    g_p = primitive_root(p)
    g_q = primitive_root(q)
    if g_p is None or g_q is None:
        return None

    dlp = [discrete_log(b % p, g_p, p) for b in bases]
    dlq = [discrete_log(b % q, g_q, q) for b in bases]
    genus = [dl % 2 for dl in dlp]  # εᵢ = dlog_p(bᵢ) mod 2

    # Check if genus character is non-trivial on the bases
    if all(g == 0 for g in genus):
        return {'trivial_genus': True, 'genus': genus}

    # Build augmented lattice
    dim = d + 2
    B = IntegerMatrix(dim, dim)
    for i in range(d):
        B[i, i] = 1
        B[i, d] = dlp[i]
        B[i, d + 1] = dlq[i]
    B[d, d] = p - 1
    B[d + 1, d + 1] = q - 1

    LLL.reduction(B)

    # Extract kernel vectors (aux coords = 0) and near-kernel
    vectors = []
    for i in range(dim):
        v = [int(B[i, j]) for j in range(d)]
        aux = [int(B[i, j]) for j in range(d, dim)]
        if all(a == 0 for a in aux) and any(x != 0 for x in v):
            norm = math.sqrt(sum(x * x for x in v))
            gp = sum(v[j] * genus[j] for j in range(d)) % 2
            vectors.append({
                'vec': v, 'norm': norm, 'genus_parity': gp,
            })

    vectors.sort(key=lambda x: x['norm'])

    # Stats
    even = [v for v in vectors if v['genus_parity'] == 0]
    odd = [v for v in vectors if v['genus_parity'] == 1]

    result = {
        'trivial_genus': False,
        'genus': genus,
        'num_kernel_vecs': len(vectors),
        'num_even': len(even),
        'num_odd': len(odd),
        'shortest_norm': vectors[0]['norm'] if vectors else None,
        'shortest_genus': vectors[0]['genus_parity'] if vectors else None,
        'shortest_even_norm': even[0]['norm'] if even else None,
        'shortest_odd_norm': odd[0]['norm'] if odd else None,
        'odd_in_top_k': sum(1 for v in vectors[:top_k] if v['genus_parity'] == 1),
        'factored': False,
    }

    # Check if any genus-odd vector factors N
    for v in odd[:5]:
        vec = v['vec']
        if all(e % 2 == 0 for e in vec):
            half = 1
            for i, e in enumerate(vec):
                if e != 0:
                    half = (half * pow(bases[i], abs(e) // 2, N)) % N
            g1 = math.gcd(half - 1, N)
            g2 = math.gcd(half + 1, N)
            if 1 < g1 < N or 1 < g2 < N:
                result['factored'] = True
                break

    return result


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark(bits=20, d=8, num_instances=500):
    """Run the full benchmark across all basis types."""
    print(f"{'='*75}")
    print(f"  STRUCTURED BASIS BENCHMARK")
    print(f"  {bits}-bit semiprimes, d={d} bases, {num_instances} instances")
    print(f"{'='*75}")

    summary = {}

    for btype, bfunc in BASIS_TYPES.items():
        print(f"\n  --- {btype} ---")
        t0 = time.time()

        stats = {
            'total': 0, 'trivial_genus': 0, 'no_kernel': 0,
            'all_even': 0, 'has_odd': 0, 'odd_shortest': 0,
            'factored': 0,
            'even_norms': [], 'odd_norms': [],
        }

        for i in range(num_instances):
            N, p, q = generate_semiprime(bits)
            bases = bfunc(N, d)

            if len(bases) < d:
                continue

            result = build_and_analyze(N, p, q, bases)
            if result is None:
                continue

            stats['total'] += 1

            if result.get('trivial_genus'):
                stats['trivial_genus'] += 1
                continue

            if result['num_kernel_vecs'] == 0:
                stats['no_kernel'] += 1
                continue

            if result['num_odd'] == 0:
                stats['all_even'] += 1
            else:
                stats['has_odd'] += 1
                if result['shortest_genus'] == 1:
                    stats['odd_shortest'] += 1
                if result['factored']:
                    stats['factored'] += 1

            if result['shortest_even_norm'] is not None:
                stats['even_norms'].append(result['shortest_even_norm'])
            if result['shortest_odd_norm'] is not None:
                stats['odd_norms'].append(result['shortest_odd_norm'])

        elapsed = time.time() - t0

        # Report
        total = stats['total']
        non_triv = total - stats['trivial_genus']
        with_kernel = non_triv - stats['no_kernel']

        print(f"  Total instances: {total}")
        print(f"  Trivial genus (all bases QR): {stats['trivial_genus']}")
        print(f"  No kernel vectors: {stats['no_kernel']}")
        print(f"  All genus-even: {stats['all_even']}")
        print(f"  Has genus-odd: {stats['has_odd']}")
        print(f"  Genus-odd SHORTEST: {stats['odd_shortest']}")
        print(f"  Factored via odd vector: {stats['factored']}")
        if stats['even_norms']:
            print(f"  Avg shortest even norm: {np.mean(stats['even_norms']):.3f}")
        if stats['odd_norms']:
            print(f"  Avg shortest odd norm: {np.mean(stats['odd_norms']):.3f}")
            ratio = np.mean(stats['odd_norms']) / np.mean(stats['even_norms']) if stats['even_norms'] else 0
            print(f"  Odd/Even norm ratio: {ratio:.3f}")
        print(f"  Time: {elapsed:.1f}s")

        odd_rate = stats['odd_shortest'] / max(with_kernel, 1)
        print(f"  >>> GENUS-ODD SHORTEST RATE: {odd_rate:.1%} <<<")

        summary[btype] = {
            'odd_rate': odd_rate,
            'has_odd_rate': stats['has_odd'] / max(with_kernel, 1),
            'factored': stats['factored'],
            'avg_even': np.mean(stats['even_norms']) if stats['even_norms'] else None,
            'avg_odd': np.mean(stats['odd_norms']) if stats['odd_norms'] else None,
        }

    # Final comparison
    print(f"\n{'='*75}")
    print(f"  COMPARISON ACROSS BASIS TYPES")
    print(f"{'='*75}")
    print(f"  {'basis_type':<20} | {'odd_shortest':>12} | {'has_odd':>8} | {'factored':>8} | {'odd/even':>8}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for btype, s in summary.items():
        ratio_str = f"{s['avg_odd']/s['avg_even']:.3f}" if s['avg_odd'] and s['avg_even'] else "N/A"
        print(f"  {btype:<20} | {s['odd_rate']:>11.1%} | {s['has_odd_rate']:>7.1%} | {s['factored']:>8d} | {ratio_str:>8}")


if __name__ == "__main__":
    random.seed(42)

    # Start with 20-bit semiprimes (small enough for full analysis)
    run_benchmark(bits=20, d=8, num_instances=200)

    # If promising, scale up
    print("\n\n")
    run_benchmark(bits=24, d=8, num_instances=200)
