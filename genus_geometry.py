#!/usr/bin/env python3
"""
Do genus-odd vectors in L_R have detectable geometric properties?

For small N = pq where we KNOW p, label every short vector in L_R
as genus-even (in L_C, doesn't factor) or genus-odd (factors N).

Then test: is there ANY geometric signal distinguishing them?
- Length distribution
- Angle to LLL basis vectors
- Component statistics (mean, variance, skew)
- Parity patterns
- Inner products with specific directions

If genus-odd vectors are geometrically distinguishable without knowing
the genus character, that's a path to polynomial-time factoring.
If distributions are identical, that's a strong negative result.
"""

import math
import random
import sys
import numpy as np
from collections import defaultdict
from fpylll import IntegerMatrix, LLL, BKZ

sys.stdout.reconfigure(line_buffering=True)


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
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1: factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    return None


def discrete_log(a, g, p):
    a = a % p
    if a == 0: return 0
    current = 1
    for k in range(p):
        if current == a: return k
        current = (current * g) % p
    return 0


def build_regev_lattice_full(N, p, q, bases):
    """
    Build L_R as an explicit lattice basis using the modular kernel construction.
    Returns the LLL-reduced basis as a numpy array, plus metadata.
    """
    d = len(bases)
    g_p = primitive_root(p)
    g_q = primitive_root(q)

    dlogs_p = [discrete_log(b % p, g_p, p) for b in bases]
    dlogs_q = [discrete_log(b % q, g_q, q) for b in bases]

    # Genus character: (b/p) = 1 if dlog_p is even, -1 if odd
    # (since g_p is a primitive root, b is QR iff its dlog is even)
    genus = [dl % 2 for dl in dlogs_p]

    # Build the lattice using the standard modular kernel construction
    # L_R = {e ∈ Z^d : Σ eᵢ dlp[i] ≡ 0 mod (p-1) AND Σ eᵢ dlq[i] ≡ 0 mod (q-1)}
    #
    # Construct via augmented lattice:
    # B = [I_d | M^T · S] where M = [dlp; dlq], S = scaling
    # Then LLL and extract rows with zero auxiliary coords.
    #
    # Better: use the HNF kernel approach
    # The lattice L_R has a basis computable from:
    # B_R = rows of (d+2)×(d+2) matrix:
    #   [I_d      | dlp_0 dlq_0 ]
    #   [         | dlp_1 dlq_1 ]
    #   [  ...    |  ...   ...  ]
    #   [         | dlp_{d-1} dlq_{d-1} ]
    #   [0...0    | p-1    0    ]
    #   [0...0    | 0      q-1  ]

    dim = d + 2
    B = IntegerMatrix(dim, dim)
    for i in range(d):
        B[i, i] = 1
        B[i, d] = dlogs_p[i]
        B[i, d+1] = dlogs_q[i]
    B[d, d] = p - 1
    B[d+1, d+1] = q - 1

    LLL.reduction(B)

    # Extract ALL short vectors, including those with nonzero auxiliary coords
    # But focus on those in the kernel (aux = 0)
    kernel_basis = []
    other_vecs = []
    for i in range(dim):
        v = [int(B[i, j]) for j in range(d)]
        aux = [int(B[i, j]) for j in range(d, dim)]
        if all(a == 0 for a in aux):
            if any(x != 0 for x in v):
                kernel_basis.append(np.array(v))
        else:
            other_vecs.append((np.array(v), np.array(aux)))

    return kernel_basis, genus, dlogs_p, dlogs_q, other_vecs


def enumerate_short_vectors(kernel_basis, max_coeff=5):
    """
    Enumerate short vectors in L_R as integer combinations of basis vectors.
    """
    if not kernel_basis:
        return []

    d = len(kernel_basis[0])
    rank = len(kernel_basis)

    vectors = set()
    # Enumerate over integer combinations with small coefficients
    if rank == 0:
        return []

    def _recurse(idx, current):
        if idx == rank:
            v = tuple(current)
            if any(x != 0 for x in v):
                vectors.add(v)
            return
        for c in range(-max_coeff, max_coeff + 1):
            _recurse(idx + 1, current + c * kernel_basis[idx])

    _recurse(0, np.zeros(d, dtype=int))
    return [np.array(v) for v in vectors]


def genus_parity(v, genus):
    """Compute the genus parity of a vector: Σ vᵢ · genus[i] mod 2."""
    return sum(int(v[i]) * genus[i] for i in range(len(genus))) % 2


def analyze_geometry(N, p, q, num_bases=8, max_coeff=6):
    """
    The main experiment: enumerate short vectors in L_R,
    label them by genus, and compare geometric properties.
    """
    bases = [pr for pr in small_primes(60) if N % pr != 0][:num_bases]
    d = len(bases)

    print(f"\n{'='*70}")
    print(f"  N = {N} = {p} × {q} ({N.bit_length()} bits), d = {d} bases")
    print(f"  Bases: {bases}")
    print(f"{'='*70}")

    kernel_basis, genus, dlp, dlq, _ = build_regev_lattice_full(N, p, q, bases)

    print(f"  Genus character (dlog mod 2): {genus}")
    print(f"  Kernel rank: {len(kernel_basis)}")

    if len(kernel_basis) == 0:
        print(f"  No kernel vectors found — N too large for this basis size")
        return None

    for i, v in enumerate(kernel_basis):
        print(f"  Basis {i}: {v}  genus={'ODD' if genus_parity(v, genus) else 'even'}")

    # Enumerate short vectors
    vectors = enumerate_short_vectors(kernel_basis, max_coeff=max_coeff)
    print(f"\n  Enumerated {len(vectors)} short vectors (|coeff| ≤ {max_coeff})")

    if not vectors:
        return None

    # Label each vector
    even_vecs = []
    odd_vecs = []
    for v in vectors:
        gp = genus_parity(v, genus)
        if gp == 0:
            even_vecs.append(v)
        else:
            odd_vecs.append(v)

    print(f"  Genus-even: {len(even_vecs)}, Genus-odd: {len(odd_vecs)}")
    print(f"  Ratio odd/total: {len(odd_vecs)/len(vectors):.3f}")

    if not odd_vecs or not even_vecs:
        print(f"  Cannot compare — need both classes")
        return None

    # =================================================================
    # GEOMETRIC COMPARISON
    # =================================================================

    # 1. Length distribution
    even_norms = [float(np.linalg.norm(v)) for v in even_vecs]
    odd_norms = [float(np.linalg.norm(v)) for v in odd_vecs]

    print(f"\n  --- Length (L2 norm) ---")
    print(f"  Even: mean={np.mean(even_norms):.3f} std={np.std(even_norms):.3f} "
          f"min={min(even_norms):.3f} max={max(even_norms):.3f}")
    print(f"  Odd:  mean={np.mean(odd_norms):.3f} std={np.std(odd_norms):.3f} "
          f"min={min(odd_norms):.3f} max={max(odd_norms):.3f}")

    # Statistical test: are means significantly different?
    if len(even_norms) > 1 and len(odd_norms) > 1:
        diff = abs(np.mean(even_norms) - np.mean(odd_norms))
        pooled_std = np.sqrt((np.var(even_norms) + np.var(odd_norms)) / 2)
        if pooled_std > 0:
            effect_size = diff / pooled_std
            print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        else:
            effect_size = 0

    # 2. L1 norm
    even_l1 = [float(np.sum(np.abs(v))) for v in even_vecs]
    odd_l1 = [float(np.sum(np.abs(v))) for v in odd_vecs]
    print(f"\n  --- L1 norm ---")
    print(f"  Even: mean={np.mean(even_l1):.3f}")
    print(f"  Odd:  mean={np.mean(odd_l1):.3f}")

    # 3. Linf norm (max component)
    even_linf = [float(np.max(np.abs(v))) for v in even_vecs]
    odd_linf = [float(np.max(np.abs(v))) for v in odd_vecs]
    print(f"\n  --- Linf norm (max component) ---")
    print(f"  Even: mean={np.mean(even_linf):.3f}")
    print(f"  Odd:  mean={np.mean(odd_linf):.3f}")

    # 4. Number of nonzero components
    even_nnz = [int(np.count_nonzero(v)) for v in even_vecs]
    odd_nnz = [int(np.count_nonzero(v)) for v in odd_vecs]
    print(f"\n  --- Sparsity (nonzero components) ---")
    print(f"  Even: mean={np.mean(even_nnz):.3f}")
    print(f"  Odd:  mean={np.mean(odd_nnz):.3f}")

    # 5. Component-wise parity: are even/odd components distributed differently?
    even_parity_sums = [sum(int(abs(v[i])) % 2 for i in range(d)) for v in even_vecs]
    odd_parity_sums = [sum(int(abs(v[i])) % 2 for i in range(d)) for v in odd_vecs]
    print(f"\n  --- Component parity (# odd components) ---")
    print(f"  Even vectors: mean={np.mean(even_parity_sums):.3f}")
    print(f"  Odd vectors:  mean={np.mean(odd_parity_sums):.3f}")

    # 6. Per-coordinate analysis
    print(f"\n  --- Per-coordinate mean ---")
    even_arr = np.array(even_vecs, dtype=float)
    odd_arr = np.array(odd_vecs, dtype=float)
    for i in range(d):
        em = np.mean(even_arr[:, i])
        om = np.mean(odd_arr[:, i])
        diff_flag = " ← DIFFERS" if abs(em - om) > 0.1 else ""
        print(f"  coord {i} ({bases[i]:>2}): even={em:+.3f}  odd={om:+.3f}{diff_flag}")

    # 7. Per-coordinate variance
    print(f"\n  --- Per-coordinate variance ---")
    for i in range(d):
        ev = np.var(even_arr[:, i])
        ov = np.var(odd_arr[:, i])
        diff_flag = " ← DIFFERS" if abs(ev - ov) / max(ev, ov, 0.01) > 0.1 else ""
        print(f"  coord {i} ({bases[i]:>2}): even={ev:.3f}  odd={ov:.3f}{diff_flag}")

    # 8. Inner product with the genus direction itself
    genus_dir = np.array(genus, dtype=float)
    if np.linalg.norm(genus_dir) > 0:
        genus_dir = genus_dir / np.linalg.norm(genus_dir)
        even_proj = [float(np.dot(v, genus_dir)) for v in even_vecs]
        odd_proj = [float(np.dot(v, genus_dir)) for v in odd_vecs]
        print(f"\n  --- Projection onto genus direction ---")
        print(f"  Even: mean={np.mean(even_proj):.3f} std={np.std(even_proj):.3f}")
        print(f"  Odd:  mean={np.mean(odd_proj):.3f} std={np.std(odd_proj):.3f}")
        # The genus constraint IS the parity of this projection.
        # Even vectors have even integer projection, odd have odd.
        # But the CONTINUOUS projection might still differ.

    # 9. Verify factor revelation
    print(f"\n  --- Factor revelation ---")
    factors_from_odd = 0
    factors_from_even = 0
    for v in odd_vecs[:20]:
        if all(int(v[i]) % 2 == 0 for i in range(d)):
            half = 1
            for i in range(d):
                e = int(v[i]) // 2
                if e >= 0:
                    half = (half * pow(bases[i], e, N)) % N
                else:
                    half = (half * pow(pow(bases[i], -e, N), -1, N)) % N
            g1 = math.gcd(half - 1, N)
            g2 = math.gcd(half + 1, N)
            if 1 < g1 < N or 1 < g2 < N:
                factors_from_odd += 1
    for v in even_vecs[:20]:
        if all(int(v[i]) % 2 == 0 for i in range(d)):
            half = 1
            for i in range(d):
                e = int(v[i]) // 2
                if e >= 0:
                    half = (half * pow(bases[i], e, N)) % N
                else:
                    half = (half * pow(pow(bases[i], -e, N), -1, N)) % N
            g1 = math.gcd(half - 1, N)
            g2 = math.gcd(half + 1, N)
            if 1 < g1 < N or 1 < g2 < N:
                factors_from_even += 1

    print(f"  Factors from genus-odd (first 20 even-exponent): {factors_from_odd}")
    print(f"  Factors from genus-even (first 20 even-exponent): {factors_from_even}")

    return {
        'N': N, 'p': p, 'q': q,
        'num_even': len(even_vecs), 'num_odd': len(odd_vecs),
        'even_norm_mean': np.mean(even_norms), 'odd_norm_mean': np.mean(odd_norms),
        'even_l1_mean': np.mean(even_l1), 'odd_l1_mean': np.mean(odd_l1),
        'even_nnz_mean': np.mean(even_nnz), 'odd_nnz_mean': np.mean(odd_nnz),
    }


if __name__ == "__main__":
    random.seed(42)

    results = []
    # Small N where we can enumerate enough vectors
    test_cases = [
        (7, 11),      # N = 77
        (13, 17),     # N = 221
        (23, 29),     # N = 667
        (31, 37),     # N = 1147
        (43, 47),     # N = 2021
        (59, 61),     # N = 3599
        (71, 73),     # N = 5183
        (83, 89),     # N = 7387
        (97, 101),    # N = 9797
    ]

    for p, q in test_cases:
        r = analyze_geometry(p * q, p, q, num_bases=8, max_coeff=5)
        if r:
            results.append(r)

    if results:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: Geometric signal detection")
        print(f"{'='*70}")
        print(f"  {'N':>8} | {'even':>5} | {'odd':>5} | {'norm_e':>8} | {'norm_o':>8} | {'Δnorm':>8} | {'signal?'}")
        print(f"  {'-'*8}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
        for r in results:
            delta = abs(r['even_norm_mean'] - r['odd_norm_mean'])
            avg = (r['even_norm_mean'] + r['odd_norm_mean']) / 2
            rel = delta / avg if avg > 0 else 0
            signal = "YES" if rel > 0.05 else "no"
            print(f"  {r['N']:8d} | {r['num_even']:5d} | {r['num_odd']:5d} | "
                  f"{r['even_norm_mean']:8.3f} | {r['odd_norm_mean']:8.3f} | "
                  f"{delta:8.3f} | {signal}")
