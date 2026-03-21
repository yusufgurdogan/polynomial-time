#!/usr/bin/env python3
"""
Compare Regev's lattice vs Class Group relation lattice for small N = pq.

For each N = pq, build two lattices from the same small prime factor base:

L_R (Regev): {(e₁,...,e_d) ∈ Z^d : ∏ bᵢ^eᵢ ≡ 1 (mod N)}
             Multiplicative relations in (Z/NZ)*

L_C (Class): {(e₁,...,e_d) ∈ Z^d : ∏ 𝔭ᵢ^eᵢ ~ (1) in Cl(Q(√N))}
             Relations in the ideal class group

Questions:
- Are they the same lattice?
- Is one a sublattice of the other?
- After LLL, which has shorter vectors?
- Does either shortest vector directly reveal p or q?
"""

import math
import random
import sys
from typing import List, Tuple, Optional
from fpylll import IntegerMatrix, LLL

sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# Small primes and discrete log (for small N)
# =============================================================================

def small_primes(limit: int) -> List[int]:
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            for j in range(i*i, limit + 1, i):
                s[j] = False
    return [i for i in range(limit + 1) if s[i]]


def multiplicative_order(a: int, n: int) -> int:
    """Compute ord(a, n) — the multiplicative order of a mod n."""
    if math.gcd(a, n) != 1:
        return 0
    order = 1
    current = a % n
    while current != 1:
        current = (current * a) % n
        order += 1
        if order > n:
            return 0
    return order


def discrete_log_brute(g: int, target: int, n: int, max_order: int = None) -> Optional[int]:
    """Brute force discrete log: find k such that g^k ≡ target (mod n)."""
    if max_order is None:
        max_order = n
    current = 1
    for k in range(max_order):
        if current == target % n:
            return k
        current = (current * g) % n
    return None


# =============================================================================
# Build Regev's lattice L_R
# =============================================================================

def build_regev_lattice(N: int, bases: List[int], search_bound: int = None) -> List[List[int]]:
    """
    Build L_R = {e ∈ Z^d : ∏ bases[i]^e[i] ≡ 1 (mod N)}.

    For small N, find relations by brute-force searching over exponent vectors.
    Returns list of basis vectors for L_R.
    """
    d = len(bases)
    if search_bound is None:
        search_bound = max(200, N)

    relations = []

    # Method 1: For each pair of bases, find relations via discrete log
    # If b_j = b_0^k mod N, then (k, 0, ..., -1, ..., 0) is a relation
    # where -1 is at position j

    # Method 2: Exhaustive search over small exponent vectors
    # For d bases, search vectors with |e_i| ≤ B

    # For small N, use factored group structure
    # (Z/NZ)* for N = pq has order (p-1)(q-1)
    # Find order of each base
    orders = [multiplicative_order(b, N) for b in bases]
    group_order = orders[0] if orders[0] > 0 else None

    # Find relations by checking products
    # Start with 2D relations: b_i^a * b_j^b ≡ 1
    for i in range(d):
        for j in range(i, d):
            if orders[i] == 0 or orders[j] == 0:
                continue
            # Find a, b such that bases[i]^a * bases[j]^b ≡ 1 mod N
            # Equivalently: bases[i]^a ≡ bases[j]^(-b) mod N
            max_search = min(search_bound, max(orders[i], orders[j]) + 1)
            # Precompute powers of bases[j]
            inv_powers = {}
            current = 1
            for b in range(max_search):
                inv_j = pow(current, -1, N) if math.gcd(current, N) == 1 else None
                if inv_j is not None:
                    inv_powers[inv_j] = b
                current = (current * bases[j]) % N

            # Check powers of bases[i]
            current = 1
            for a in range(max_search):
                if current in inv_powers:
                    b_val = inv_powers[current]
                    if a == 0 and b_val == 0:
                        continue
                    rel = [0] * d
                    rel[i] = a
                    rel[j] = -b_val
                    # Verify
                    product = 1
                    for k in range(d):
                        if rel[k] > 0:
                            product = (product * pow(bases[k], rel[k], N)) % N
                        elif rel[k] < 0:
                            product = (product * pow(bases[k], -rel[k], N)) % N
                            product = pow(product, -1, N) if math.gcd(product, N) == 1 else 0
                    # Simpler verification
                    prod = 1
                    for k in range(d):
                        prod = (prod * pow(bases[k], rel[k] % (N * N), N)) % N
                    if prod == 1 and any(r != 0 for r in rel):
                        relations.append(rel)
                current = (current * bases[i]) % N

    # Also add single-base order relations: (order_i, 0, ...) at position i
    for i in range(d):
        if orders[i] > 0:
            rel = [0] * d
            rel[i] = orders[i]
            relations.append(rel)

    # Deduplicate and return
    unique = []
    seen = set()
    for rel in relations:
        key = tuple(rel)
        if key not in seen and tuple(-r for r in rel) not in seen:
            seen.add(key)
            unique.append(rel)

    return unique


# =============================================================================
# Build Class Group relation lattice L_C
# =============================================================================

def legendre_symbol(a: int, p: int) -> int:
    """Compute (a/p) for prime p."""
    if a % p == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def splits_in_Q_sqrt_N(prime: int, N: int, p_factor: int, q_factor: int) -> Tuple[bool, bool]:
    """
    Check if a small prime ℓ splits in Q(√N) for N = p*q.
    A prime ℓ splits iff (N/ℓ) = 1 (Legendre symbol), i.e., N is a QR mod ℓ.
    Returns (splits, ramifies).
    """
    if N % prime == 0:
        return False, True  # ramifies
    ls = legendre_symbol(N % prime, prime)
    return ls == 1, False


def build_class_lattice(N: int, p: int, q: int, bases: List[int]) -> List[List[int]]:
    """
    Build L_C: relations among prime ideals above the bases in the class group of Q(√N).

    For a prime ℓ that splits in Q(√N):
      ℓO = 𝔭·𝔭' where 𝔭 = (ℓ, √N - r) for some root r of N mod ℓ.
      The class [𝔭] in Cl(Q(√N)) might be trivial or non-trivial.

    A relation ∏ 𝔭ᵢ^eᵢ = (α) (principal) gives exponent vector (e₁,...,e_d).

    For the real quadratic field Q(√N), we use the correspondence between
    binary quadratic forms and ideal classes. A form (a, b, c) with disc = 4N
    corresponds to the ideal (a, (b + √(4N))/2).

    For our small primes: the form (ℓ, b, c) with ℓ | disc represents
    the ideal above ℓ. A relation in forms = a relation in ideals.
    """
    d = len(bases)
    D = 4 * N  # discriminant

    relations = []

    # For each base prime ℓ, find the corresponding quadratic form
    # The prime ℓ splits iff (D/ℓ) = 0 or 1.
    # The form representing 𝔭 above ℓ is (ℓ, b, c) where
    # b² ≡ D (mod 4ℓ) and c = (b² - D)/(4ℓ).
    forms = {}
    for idx, ell in enumerate(bases):
        if D % ell == 0:
            # ℓ divides discriminant — ramified
            # Form: (ℓ, 0, -N/ℓ) or similar
            b = 0
            while (b * b - D) % (4 * ell) != 0:
                b += 1
                if b > 2 * ell:
                    break
            if (b * b - D) % (4 * ell) == 0:
                c = (b * b - D) // (4 * ell)
                forms[idx] = (ell, b, c)
            continue

        # Find b such that b² ≡ D (mod 4ℓ)
        found_b = None
        for b in range(2 * ell):
            if (b * b - D) % (4 * ell) == 0:
                found_b = b
                break
        if found_b is None:
            continue  # ℓ is inert, skip

        c = (found_b * found_b - D) // (4 * ell)
        forms[idx] = (ell, found_b, c)

    # Now find relations: compositions of forms that give the principal form
    # The principal form is (1, b₀, c₀) where b₀² ≡ D mod 4.
    # For D = 4N: principal form is (1, 0, -N).

    # For each pair of forms, compose and check if result is principal
    # Composition of forms (a₁,b₁,c₁) and (a₂,b₂,c₂):
    # Use Gauss/Dirichlet composition

    # For the class group of disc 4N with N=pq:
    # The class group has order h (class number).
    # The 2-part has rank 1 (genus theory: 2 genera for 2 prime factors).
    # So there's exactly one non-trivial element of order dividing 2 — the ambiguous class.

    # The ambiguous class contains forms (p, 0, -q), (q, 0, -p), etc.
    # (up to equivalence and signs)

    # A relation in the class group: ∏ [𝔭ᵢ]^eᵢ = [1]
    # This means the exponent vector maps to 0 in Cl.

    # For our purposes: compute the class of each form and find the kernel.
    # The class of form (ℓ, b, c) is determined by its genus and position
    # in the class group.

    # For small N, compute classes by reduction
    from infrastructure import reduce_form, QF, rho_step

    form_classes = {}  # idx -> reduced form (representative of the class)
    for idx, (a, b, c) in forms.items():
        f = QF(a, b, c)
        rf = reduce_form(f)
        form_classes[idx] = (rf.a, rf.b, rf.c)

    # Principal class: reduce (1, 0, -N)
    principal = reduce_form(QF(1, 0, -N))
    principal_key = (principal.a, principal.b, principal.c)

    # For each form, determine if it's in the principal class or the non-principal genus
    # Two forms are in the same class iff they reduce to the same form
    # (for our purposes, check if reduced forms match)

    # Find relations: subsets of forms whose composition is principal
    # For small d, we can brute-force compositions

    # Simple approach: for each form, find its order in the class group
    # by repeated composition (squaring)
    form_orders = {}
    for idx in forms:
        a, b, c = forms[idx]
        # Compose with itself repeatedly until we get principal
        current = QF(a, b, c)
        order = 1
        for _ in range(1000):
            # Compose current with the original form
            composed = compose_forms(current.a, current.b, current.c,
                                     a, b, c, D)
            if composed is None:
                break
            current = reduce_form(QF(*composed))
            order += 1
            if (current.a, current.b, current.c) == principal_key:
                break
            # Also check if current ≈ principal (same absolute values)
            if abs(current.a) == abs(principal.a) and abs(current.b) == abs(principal.b):
                break
        form_orders[idx] = order

        # The relation: ℓ^order ≡ principal
        rel = [0] * d
        rel[idx] = order
        relations.append(rel)

    # Pairwise relations: compose form_i with form_j, check if principal
    form_indices = list(forms.keys())
    for i_pos, i in enumerate(form_indices):
        for j in form_indices[i_pos:]:
            ai, bi, ci = forms[i]
            aj, bj, cj = forms[j]
            composed = compose_forms(ai, bi, ci, aj, bj, cj, D)
            if composed is None:
                continue
            reduced = reduce_form(QF(*composed))
            if (reduced.a, reduced.b, reduced.c) == principal_key or \
               abs(reduced.a) == 1:
                # form_i * form_j = principal → relation e_i = 1, e_j = 1
                rel = [0] * d
                rel[i] = 1
                rel[j] = 1
                relations.append(rel)

            # Also try form_i * form_j^(-1)
            # Inverse of (a, b, c) is (a, -b, c)
            composed_inv = compose_forms(ai, bi, ci, aj, -bj, cj, D)
            if composed_inv is not None:
                reduced_inv = reduce_form(QF(*composed_inv))
                if abs(reduced_inv.a) == 1:
                    rel = [0] * d
                    rel[i] = 1
                    rel[j] = -1
                    relations.append(rel)

    # Deduplicate
    unique = []
    seen = set()
    for rel in relations:
        key = tuple(rel)
        neg_key = tuple(-r for r in rel)
        if key not in seen and neg_key not in seen:
            seen.add(key)
            unique.append(rel)

    return unique


def compose_forms(a1, b1, c1, a2, b2, c2, D):
    """
    Gauss/Dirichlet composition of two forms of discriminant D.
    Returns (a3, b3, c3) or None if computation fails.
    """
    # Shanks' NUCOMP or simplified Gauss composition
    # For (a1,b1,c1) * (a2,b2,c2):

    g = math.gcd(a1, a2)
    g = math.gcd(g, (b1 + b2) // 2) if (b1 + b2) % 2 == 0 else math.gcd(g, b1 + b2)

    # Simplified: assume gcd(a1, a2) is small
    # Use the formula for composition
    # a3 = a1*a2 / g^2
    # b3 = b2 + 2*a2*(... ) / g
    # This is getting complicated. Use the direct method.

    # Direct method for discriminant D:
    # The composed form has a = a1*a2/d^2, where d = gcd(a1, a2, (b1+b2)/2)
    # and b satisfies b ≡ b1 (mod 2a1/d), b ≡ b2 (mod 2a2/d), b² ≡ D (mod 4a)

    try:
        d = math.gcd(math.gcd(a1, a2), (b1 + b2) // 2)
        a3 = (a1 * a2) // (d * d)

        # Find b3 using CRT
        # b3 ≡ b1 (mod 2*a1//d)
        # b3 ≡ b2 (mod 2*a2//d)
        m1 = 2 * a1 // d
        m2 = 2 * a2 // d

        # CRT
        if m1 == 0 or m2 == 0:
            return None

        g12 = math.gcd(m1, m2)
        if (b2 - b1) % g12 != 0:
            return None

        lcm_m = m1 * m2 // g12

        try:
            inv = pow(m1 // g12, -1, m2 // g12)
        except ValueError:
            return None

        b3 = (b1 + m1 * ((b2 - b1) // g12 * inv % (m2 // g12))) % lcm_m

        # Ensure b3² ≡ D (mod 4*a3)
        # Adjust b3 to be in the right range
        while b3 < 0:
            b3 += 2 * a3
        b3 = b3 % (2 * a3)

        # Compute c3
        c3 = (b3 * b3 - D) // (4 * a3)

        return (a3, b3, c3)
    except (ZeroDivisionError, ValueError):
        return None


# =============================================================================
# Lattice comparison
# =============================================================================

def vectors_to_matrix(vectors: List[List[int]], d: int) -> Optional[IntegerMatrix]:
    """Convert list of integer vectors to fpylll matrix."""
    if not vectors:
        return None
    n = len(vectors)
    M = IntegerMatrix(n, d)
    for i, vec in enumerate(vectors):
        for j in range(d):
            M[i, j] = int(vec[j]) if j < len(vec) else 0
    return M


def lattice_stats(name: str, vectors: List[List[int]], d: int, N: int, p: int, q: int):
    """Compute and print statistics about a lattice."""
    if not vectors:
        print(f"  {name}: NO VECTORS FOUND")
        return

    print(f"\n  {name}: {len(vectors)} generating vectors in Z^{d}")

    # Print vectors
    for i, v in enumerate(vectors[:15]):
        norm = math.sqrt(sum(x*x for x in v))
        print(f"    v{i}: {v}  (norm={norm:.2f})")

    if len(vectors) > 15:
        print(f"    ... ({len(vectors) - 15} more)")

    # Build matrix and reduce with LLL
    if len(vectors) >= d:
        M = vectors_to_matrix(vectors[:d+5], d)
        if M is not None:
            try:
                LLL.reduction(M)
                print(f"  After LLL:")
                for i in range(min(d, M.nrows)):
                    v = [int(M[i, j]) for j in range(d)]
                    norm = math.sqrt(sum(x*x for x in v))
                    # Check if this vector reveals a factor
                    # A vector (e1,...,ed) where ∏ bi^ei = p (or q) reveals the factorization
                    print(f"    v{i}: {v}  (norm={norm:.2f})")
            except Exception as e:
                print(f"  LLL failed: {e}")

    # Check if any vector directly reveals factors
    print(f"  Factor detection:")
    for v in vectors:
        # Check: does this relation produce a value related to p or q?
        # For Regev lattice: ∏ bi^ei mod N should be 1, but
        # gcd(∏ bi^(ei/2) - 1, N) might give a factor
        pass


def compare_lattices(N: int, p: int, q: int, num_primes: int = 6):
    """Main comparison: build both lattices and compare."""
    print(f"\n{'='*70}")
    print(f"  LATTICE COMPARISON: N = {N} = {p} × {q} ({N.bit_length()} bits)")
    print(f"{'='*70}")

    # Choose factor base: first num_primes primes that don't divide N
    all_primes = small_primes(50)
    bases = [pr for pr in all_primes if N % pr != 0][:num_primes]
    d = len(bases)
    print(f"  Factor base: {bases} (d={d})")

    # Build Regev's lattice
    print(f"\n  --- Building Regev lattice L_R ---")
    L_R = build_regev_lattice(N, bases, search_bound=min(N, 5000))
    lattice_stats("L_R (Regev)", L_R, d, N, p, q)

    # Build Class group lattice
    print(f"\n  --- Building Class group lattice L_C ---")
    L_C = build_class_lattice(N, p, q, bases)
    lattice_stats("L_C (Class)", L_C, d, N, p, q)

    # Compare
    print(f"\n  --- Comparison ---")
    print(f"  L_R vectors: {len(L_R)}, L_C vectors: {len(L_C)}")

    # Check if L_R ⊆ L_C or L_C ⊆ L_R
    # A vector in L_R (∏ bi^ei ≡ 1 mod N) is also in L_C iff
    # the corresponding ideal product is principal
    r_in_c = 0
    c_in_r = 0

    for v in L_R:
        # Check if v is in L_C by verifying ∏ 𝔭i^ei is principal
        # For small cases: the relation holds in (Z/NZ)* iff it holds
        # mod p AND mod q. The class group relation holds iff the ideal is principal.
        # Since (Z/NZ)* ≅ (Z/pZ)* × (Z/qZ)*, a relation mod N means relations mod both.
        # In the class group, a relation mod p corresponds to the ideal 𝔭 above ℓ being
        # principal in Q(√N), which is a STRONGER condition.
        pass

    # The key structural difference:
    # L_R captures relations in (Z/NZ)* = (Z/pZ)* × (Z/qZ)*
    # L_C captures relations in Cl(Q(√N))
    # The class group map factors through the unit group:
    # (Z/NZ)* → Cl(Q(√N)) → 0 is NOT exact in general

    # For N = pq: elements of (Z/NZ)* that lift to principal ideals form a subgroup.
    # The quotient detects the class group structure.

    # Check: do L_R and L_C have the same determinant?
    if len(L_R) >= d and len(L_C) >= d:
        # Take d vectors from each, form matrix, compute det
        M_R = vectors_to_matrix(L_R[:d], d)
        M_C = vectors_to_matrix(L_C[:d], d)

        if M_R is not None and M_C is not None:
            try:
                # Compute determinants by converting to numpy
                import numpy as np
                arr_R = np.array([[int(M_R[i, j]) for j in range(d)] for i in range(min(d, M_R.nrows))])
                arr_C = np.array([[int(M_C[i, j]) for j in range(d)] for i in range(min(d, M_C.nrows))])

                if arr_R.shape[0] == arr_R.shape[1]:
                    det_R = abs(int(round(np.linalg.det(arr_R.astype(float)))))
                    print(f"  det(L_R basis): {det_R}")
                if arr_C.shape[0] == arr_C.shape[1]:
                    det_C = abs(int(round(np.linalg.det(arr_C.astype(float)))))
                    print(f"  det(L_C basis): {det_C}")

                if arr_R.shape == arr_C.shape:
                    if det_R == det_C:
                        print(f"  DETERMINANTS MATCH → possibly same lattice")
                    else:
                        ratio = max(det_R, det_C) / max(min(det_R, det_C), 1)
                        print(f"  Determinant ratio: {ratio:.2f}")
                        if det_R > 0 and det_C > 0 and det_R % det_C == 0:
                            print(f"  L_C might be a sublattice of L_R (index {det_R // det_C})")
                        elif det_C > 0 and det_R > 0 and det_C % det_R == 0:
                            print(f"  L_R might be a sublattice of L_C (index {det_C // det_R})")
            except Exception as e:
                print(f"  Determinant computation failed: {e}")

    # Factor revelation test
    print(f"\n  --- Factor revelation test ---")
    for name, lattice in [("L_R", L_R), ("L_C", L_C)]:
        for v in lattice:
            # For a relation vector v, compute ∏ bases[i]^(v[i]/2) mod N
            # if all v[i] are even
            if all(e % 2 == 0 for e in v) and any(e != 0 for e in v):
                half_product = 1
                for i, e in enumerate(v):
                    if e != 0:
                        half_product = (half_product * pow(bases[i], e // 2, N)) % N
                g1 = math.gcd(half_product - 1, N)
                g2 = math.gcd(half_product + 1, N)
                if 1 < g1 < N:
                    print(f"  {name} vector {v}: gcd(half_prod - 1, N) = {g1} ← FACTOR!")
                if 1 < g2 < N:
                    print(f"  {name} vector {v}: gcd(half_prod + 1, N) = {g2} ← FACTOR!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    # Small test cases — inspect everything by hand
    test_cases = [
        (3, 5),      # N = 15
        (7, 11),     # N = 77
        (13, 17),    # N = 221
        (23, 29),    # N = 667
        (101, 103),  # N = 10403
        (211, 223),  # N = 47053
        (541, 631),  # N = 341371  (19 bits)
    ]

    for p, q in test_cases:
        N = p * q
        d = 6 if N < 1000 else 8
        compare_lattices(N, p, q, num_primes=d)
