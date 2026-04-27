#!/usr/bin/env python3
"""
Novel factoring approaches based on deep mathematical reasoning.

The key insight driving all these approaches:
For n = pq, the group (Z/nZ)* ≅ (Z/pZ)* × (Z/qZ)* via CRT.
Every computation mod n secretly happens "in parallel" mod p and mod q.
The factorization is ENCODED in how these two parallel computations interact.

Our goal: find a polynomial-time way to DETECT the seam between the two groups.
"""

import math
import random
import numpy as np
from typing import Optional
from sympy import isprime


# =============================================================================
# IDEA 1: Square root collision
# =============================================================================
# For n = pq, the equation x^2 ≡ a (mod n) has 4 solutions (if a is a QR mod both p,q)
# Two solutions are "boring" (±x) and two are "interesting" — they reveal factors.
#
# If we find x, y with x^2 ≡ y^2 (mod n) but x ≢ ±y (mod n), then
# gcd(x-y, n) is a factor.
#
# The question is: can we find colliding square roots in poly time?
#
# Novel twist: instead of searching for x^2 ≡ y^2 (mod n), search for
# x^k ≡ y^k (mod n) for HIGHER powers k. The collision space is richer.
# For each k, the k-th power map mod n decomposes differently mod p and mod q.
# The "kernel" of this map (elements with a^k ≡ 1) has size gcd(k, p-1)*gcd(k, q-1).
# If gcd(k, p-1) ≠ gcd(k, q-1), computing a^((n-1)/k) can split n.

def power_residue_attack(n: int) -> Optional[list[int]]:
    """
    For each small prime k, compute a^((n-1)/k) mod n for random bases.
    If gcd(k, p-1) ≠ gcd(k, q-1), this yields non-trivial residues
    that factor n via gcd.

    This generalizes Miller-Rabin (k=2) to all small primes.
    The probability of success per (a, k) pair depends on the arithmetic
    of p-1 and q-1, but trying many k values covers more cases.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    small_primes = _sieve(min(bits * 10, 500))

    for _ in range(bits * 3):
        a = random.randint(2, n - 1)
        g = math.gcd(a, n)
        if 1 < g < n:
            return _finish(n, g)

        for k in small_primes:
            if (n - 1) % k != 0:
                continue

            # Compute a^((n-1)/k) mod n
            exp = (n - 1) // k
            val = pow(a, exp, n)

            if val == 1 or val == n - 1:
                continue

            # val^k ≡ a^(n-1) ≡ 1 (mod n) [if n is a Fermat pseudoprime to base a]
            # But val ≢ 1 (mod n), so val is a non-trivial k-th root of unity mod n.
            # gcd(val - 1, n) might give a factor.
            g = math.gcd(val - 1, n)
            if 1 < g < n:
                return _finish(n, g)

            # Also try: val is a k-th root of 1 mod n.
            # So val^j for j = 1, ..., k-1 are all k-th roots.
            # Check gcd(val^j - 1, n) for each j.
            current = val
            for j in range(1, min(k, 50)):
                g = math.gcd(current - 1, n)
                if 1 < g < n:
                    return _finish(n, g)
                current = (current * val) % n

            # Squaring chain from val
            v = val
            for _ in range(int(math.log2(k)) + 2):
                v = (v * v) % n
                if v == 1:
                    break
                g = math.gcd(v - 1, n)
                if 1 < g < n:
                    return _finish(n, g)

    return None


# =============================================================================
# IDEA 2: Lattice-based Coppersmith (proper implementation)
# =============================================================================
# Coppersmith's theorem: given f(x) mod N where f has a small root x₀,
# we can find x₀ in polynomial time using LLL lattice reduction.
#
# For factoring: if we know p ≈ p₀ (approximate value), let f(x) = p₀ + x.
# Then f(x₀) ≡ 0 (mod p) where x₀ = p - p₀ is small.
# Coppersmith finds x₀ if |x₀| < N^(1/4) (for degree 1).
#
# The strategy: generate candidate approximations p₀ from various sources,
# then use Coppersmith to check if any is close enough.

def coppersmith_attack(n: int) -> Optional[list[int]]:
    """
    Generate polynomial-many approximations to p, then use
    lattice-based methods to check if any is within N^(1/4).

    Approximation sources:
    1. Continued fraction convergents of sqrt(n)
    2. n / small_integers (if q is small, p ≈ n/q)
    3. Rational approximations from n mod small primes via CRT
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    sqrt_n = math.isqrt(n)

    # Source 1: CF convergents of sqrt(n)
    # These naturally produce good approximations related to the factors
    approximations = []

    # CF expansion of sqrt(n)
    m0, d0, a0 = 0, 1, sqrt_n
    m, d, a = m0, d0, a0
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    for _ in range(min(bits * 4, 200)):
        m = d * a - m
        d_new = (n - m * m) // d
        if d_new == 0:
            break
        d = d_new
        a = (a0 + m) // d
        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev

        # The convergent p_curr/q_curr ≈ sqrt(n)
        # So p_curr^2 ≈ n * q_curr^2
        # If n = p*q, then p_curr^2 - n*q_curr^2 is small
        # and its factorization might relate to p and q

        residue = p_curr * p_curr - n * q_curr * q_curr
        if residue != 0:
            g = math.gcd(abs(residue), n)
            if 1 < g < n:
                return _finish(n, g)

        # Also: p_curr/q_curr ≈ sqrt(n), so n/p_curr * q_curr ≈ sqrt(n) ≈ p (if p ≈ q)
        # More useful: the CF convergents of n/p₀ for various guesses p₀
        # approximate q = n/p

        # Use p_curr as a candidate near sqrt(n)
        approximations.append(p_curr % n)
        if q_curr > 0:
            approximations.append((n * q_curr // p_curr) if p_curr > 0 else 0)

    # Source 2: small multiples/fractions of sqrt(n)
    for k in range(1, min(bits, 30)):
        for num, den in [(k, 1), (1, k), (k, k+1), (k+1, k)]:
            approx = sqrt_n * num // den
            if approx > 1:
                approximations.append(approx)

    # Source 3: for each approximation, try Coppersmith-style small root finding
    # For f(x) = (p₀ + x), we need |x| < n^(1/4) ≈ 2^(bits/4)
    bound = 1 << (bits // 4)

    for p0 in approximations:
        if p0 < 2 or p0 > n:
            continue

        # Quick check: is p0 itself a factor?
        g = math.gcd(p0, n)
        if 1 < g < n:
            return _finish(n, g)

        # Check p0 ± small values (cheaper than full LLL)
        # This is the "trial" version of Coppersmith
        for delta in range(min(bound, bits * bits)):
            for sign in [1, -1]:
                candidate = p0 + sign * delta
                if candidate > 1 and candidate < n and n % candidate == 0:
                    return _finish(n, candidate)

        # For proper Coppersmith, we'd use LLL here.
        # Simplified version: Howgrave-Graham's formulation
        factor = _coppersmith_small_root(p0, n, bound)
        if factor is not None:
            return _finish(n, factor)

    return None


def _coppersmith_small_root(p0: int, n: int, bound: int) -> Optional[int]:
    """
    Simplified Coppersmith: find x such that (p0 + x) divides n, |x| < bound.

    Uses a 2D lattice: the polynomial f(x) = p0 + x has a root mod p (unknown factor).
    We construct a lattice from f(x) and n, reduce it, and look for short vectors
    that correspond to the root.

    Full Coppersmith uses higher-dimensional lattices for tighter bounds.
    """
    # Construct lattice basis for the polynomial f(x) = p0 + x
    # modular equation: p0 + x ≡ 0 (mod p) for some factor p | n
    #
    # Lattice approach (Howgrave-Graham simplified):
    # We want to find small (a, b) such that a*n + b*(p0) is small
    # This corresponds to a*(n) + b*(p0 + x) = a*n + b*p0 + b*x being ≡ 0 mod p
    # with |b*x| < bound
    #
    # 2D lattice with basis:
    # [n,    0]
    # [p0,   X]  where X = bound
    #
    # Short vectors in this lattice correspond to linear combinations
    # a*[n, 0] + b*[p0, X] = [a*n + b*p0, b*X]
    # that are small. If |b*X| is small enough and a*n + b*p0 ≡ 0 (mod p),
    # then (a*n + b*p0) / gcd is a multiple of p.

    X = bound
    if X < 2:
        return None

    # Use numpy for the lattice reduction (Gram-Schmidt approximation of LLL)
    try:
        basis = np.array([[n, 0], [p0, X]], dtype=np.int64)

        # Simple 2D lattice reduction (exact for 2D)
        reduced = _lll_reduce_2d(n, 0, p0, X)
        if reduced is None:
            return None

        # Check each reduced basis vector
        for (v0, v1) in reduced:
            if v1 == 0:
                continue
            # v1 = b * X, so b = v1 / X
            if v1 % X != 0:
                continue
            b = v1 // X
            if b == 0:
                continue
            # v0 = a*n + b*p0
            # The root is: x = -p0 - v0/b... no wait
            # We want p0 + x ≡ 0 (mod p), and v0 = a*n + b*p0 ≡ b*p0 (mod n)
            # If this is a multiple of p: gcd(v0, n)
            g = math.gcd(abs(v0), n)
            if 1 < g < n:
                return g

            # Also: the polynomial value at -v0/b (if b divides v0)
            if b != 0:
                candidate = p0 - v0 // b if v0 % b == 0 else None
                if candidate and candidate > 1 and candidate < n and n % candidate == 0:
                    return candidate
    except (OverflowError, ValueError):
        pass

    return None


def _lll_reduce_2d(a11, a12, a21, a22):
    """LLL reduction for a 2×2 integer lattice. Returns reduced basis."""
    # For 2D, this is just iterated size-reduction + swap
    b1 = [a11, a12]
    b2 = [a21, a22]

    for _ in range(100):  # shouldn't need many iterations
        # Size-reduce b2 by b1
        dot_b1 = b1[0]*b1[0] + b1[1]*b1[1]
        if dot_b1 == 0:
            break
        mu = (b2[0]*b1[0] + b2[1]*b1[1]) / dot_b1
        mu_round = round(mu)
        b2 = [b2[0] - mu_round*b1[0], b2[1] - mu_round*b1[1]]

        # Check Lovász condition
        norm_b1 = b1[0]**2 + b1[1]**2
        norm_b2 = b2[0]**2 + b2[1]**2

        if norm_b2 < 0.75 * norm_b1:
            b1, b2 = b2, b1
        else:
            break

    return [b1, b2]


# =============================================================================
# IDEA 3: Multiplicative order modular probing
# =============================================================================
# Novel idea: for random a, we can't find ord(a, n) directly.
# But we CAN test whether specific values divide the order.
#
# For a small prime ℓ: ℓ | ord_p(a) iff a^((p-1)/ℓ) ≢ 1 (mod p).
# We don't know p, but we can compute a^((n-1)/ℓ) mod n.
# If ℓ | (p-1) but ℓ ∤ (q-1), or vice versa, this might split n.
#
# More precisely: define χ_ℓ(a, n) = a^((n-1)/ℓ) mod n.
# This is a "character-like" function that behaves differently mod p vs mod q
# when ℓ divides exactly one of p-1, q-1.
#
# Testing many (a, ℓ) pairs gives us a matrix of character values.
# The RANK of this matrix (viewed over appropriate rings) encodes
# information about the factorization.

def character_matrix_attack(n: int) -> Optional[list[int]]:
    """
    Build a matrix of generalized character values and analyze its structure.

    M[i][j] = a_i^((n-1)/ℓ_j) mod n

    The rank structure of M over various quotient rings reveals
    how p-1 and q-1 relate to the primes ℓ_j.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    num_bases = min(bits * 2, 60)
    primes = _sieve(min(bits * 5, 200))

    # Only use primes that divide n-1
    valid_primes = [ℓ for ℓ in primes if (n - 1) % ℓ == 0]
    if not valid_primes:
        return None

    bases = [random.randint(2, n - 1) for _ in range(num_bases)]

    # Quick GCD check
    for a in bases:
        g = math.gcd(a, n)
        if 1 < g < n:
            return _finish(n, g)

    # Build character matrix and check each entry
    for a in bases:
        for ℓ in valid_primes:
            exp = (n - 1) // ℓ
            val = pow(a, exp, n)

            if val == 1 or val == n - 1:
                continue

            # Non-trivial ℓ-th root of unity mod n!
            # This means the ℓ-th root structure differs mod p vs mod q.
            g = math.gcd(val - 1, n)
            if 1 < g < n:
                return _finish(n, g)

            # Check all powers of val
            current = val
            for _ in range(min(ℓ, 100)):
                g = math.gcd(current - 1, n)
                if 1 < g < n:
                    return _finish(n, g)
                current = (current * val) % n

            # Also: combine with other character values
            # If val1 and val2 are from different (a, ℓ) pairs,
            # val1 * val2^(-1) might reveal structure
            pass

    # Phase 2: pairwise combination of character values
    # For each prime ℓ, collect all a^((n-1)/ℓ) values
    for ℓ in valid_primes[:20]:
        exp = (n - 1) // ℓ
        values = []
        for a in bases:
            val = pow(a, exp, n)
            values.append(val)

        # Look for distinct non-trivial values that multiply to 1
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                product = (values[i] * values[j]) % n
                for target in [1, n - 1]:
                    if product == target:
                        # values[i] * values[j] ≡ ±1 but neither is ±1
                        if values[i] not in [1, n-1] and values[j] not in [1, n-1]:
                            diff = (values[i] - values[j]) % n
                            g = math.gcd(diff, n)
                            if 1 < g < n:
                                return _finish(n, g)
                quotient_minus_1 = (values[i] * pow(values[j], -1, n) - 1) % n
                if quotient_minus_1 != 0:
                    g = math.gcd(quotient_minus_1, n)
                    if 1 < g < n:
                        return _finish(n, g)

    return None


# =============================================================================
# IDEA 4: Algebraic norm attack via Z[sqrt(n)]
# =============================================================================
# In the ring Z[√n], every element a + b√n has norm a² - nb².
# If we can find elements whose norms factor nicely, we get
# relations that help factor n.
#
# Novel twist: instead of Z[√n], work in Z[√(-n)] (Gaussian-like integers).
# The norm is a² + nb². For factoring, we want:
# a² + nb² = p * (something), which means a² ≡ -nb² (mod p).
# If -n is a QR mod p, such elements exist.
#
# Even more novel: work in Z[ω] where ω = e^(2πi/k) for small k.
# The norm form is the PRODUCT of (a + b*ω^j) for all conjugates j.
# Different k values give different norm forms, each of which
# interacts differently with the factorization of n.

def algebraic_norm_attack(n: int) -> Optional[list[int]]:
    """
    Search for elements in various algebraic rings whose norms
    reveal the factorization of n.

    For Z[√d] with various d, the norm is a² - d*b².
    We look for (a, b) where gcd(a² - d*b², n) is non-trivial.

    Key insight: if d ≡ perfect_square (mod p) but d ≢ perfect_square (mod q),
    then elements of Z[√d] split differently mod p vs mod q.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    sqrt_n = math.isqrt(n)

    # Try various discriminants d
    for d in range(-min(bits * bits, 200), min(bits * bits, 200)):
        if d == 0:
            continue

        # Compute Jacobi symbol (d/n) to classify
        if d > 0:
            j = _jacobi(d, n)
        else:
            j = _jacobi((-d) % n, n)
            # Adjust for sign based on n mod 4
            if n % 4 == 3:
                j = -j

        if j == 0:
            g = math.gcd(abs(d), n)
            if 1 < g < n:
                return _finish(n, g)
            continue

        # For Jacobi symbol = -1: d is a QNR mod n
        # This means (d/p)(d/q) = -1, so one is QR and one is QNR
        # -> the ring Z[√d] splits completely differently mod p vs mod q!
        if j == -1:
            # In this case, √d exists mod one of p, q but not the other.
            # Elements of specific norm in Z[√d] can reveal this split.

            # Compute x^2 ≡ d (mod n) — this will fail since (d/n) = -1
            # But trying to compute it reveals structure:
            # Tonelli-Shanks will fail mod one factor and succeed mod the other

            # Alternative: compute d^((n-1)/2) mod n
            val = pow(abs(d) % n, (n - 1) // 2, n)
            if d < 0 and n % 4 == 3:
                val = (n - val) % n

            for v in [val - 1, val + 1, val, (n - val) % n]:
                g = math.gcd(v % n, n)
                if 1 < g < n:
                    return _finish(n, g)

            # Power of d: d^((n-1)/4), d^((n-1)/8), etc.
            exp = (n - 1) // 2
            while exp > 0 and exp % 2 == 0:
                exp //= 2
                val = pow(abs(d) % n, exp, n)
                for v in [val - 1, val + 1]:
                    g = math.gcd(v % n, n)
                    if 1 < g < n:
                        return _finish(n, g)

    # Phase 2: combine information from multiple discriminants
    # For each d where (d/n) = -1, we know (d/p) ≠ (d/q).
    # If we find d1, d2 both with (d/n) = -1, then either:
    # - (d1/p) = 1, (d1/q) = -1, (d2/p) = 1, (d2/q) = -1 → (d1*d2/p) = 1, (d1*d2/q) = 1
    # - (d1/p) = 1, (d1/q) = -1, (d2/p) = -1, (d2/q) = 1 → (d1*d2/p) = -1, (d1*d2/q) = -1
    # etc.
    # The product d1*d2 has (d1*d2/n) = 1, but we can check whether
    # d1*d2 is actually a QR mod n by trying to compute sqrt(d1*d2) mod n.

    qnr_discriminants = []
    for d in range(2, min(bits * bits, 200)):
        if _jacobi(d, n) == -1:
            qnr_discriminants.append(d)

    # For pairs of QNR discriminants, their product is a QR mod n (Jacobi = +1)
    # Try to compute sqrt of the product mod n — if it splits, we factor
    for i in range(len(qnr_discriminants)):
        for j in range(i + 1, min(i + 20, len(qnr_discriminants))):
            d1, d2 = qnr_discriminants[i], qnr_discriminants[j]
            product = (d1 * d2) % n

            # Try to compute sqrt(product) mod n using Tonelli-Shanks
            # If it exists, the two square roots might reveal factors
            roots = _sqrt_mod_composite(product, n)
            if roots:
                for r in roots:
                    g = math.gcd(r, n)
                    if 1 < g < n:
                        return _finish(n, g)
                    # Also check: r^2 ≡ product (mod n), so
                    # r^2 - product ≡ 0 (mod n)
                    # (r - sqrt(product mod p))(r + sqrt(product mod p)) ≡ 0 (mod p)
                    # This doesn't help directly, but...
                    # Compare two different square roots of the same value
                    if len(roots) >= 2:
                        for r1 in roots:
                            for r2 in roots:
                                if r1 != r2 and r1 != (n - r2):
                                    g = math.gcd(r1 - r2, n)
                                    if 1 < g < n:
                                        return _finish(n, g)
                break  # only need to find one successful pair

    return None


def _sqrt_mod_composite(a: int, n: int) -> list[int]:
    """Try to find square roots of a mod n (composite). Returns list of roots found."""
    a = a % n
    if a == 0:
        return [0]

    roots = []

    # Tonelli-Shanks-like for composite n
    # This actually reveals factors if we're lucky!
    if _jacobi(a, n) != 1:
        return []

    # Try random approach: pick random r, compute (a + r^2)^((n+1)/4) type expressions
    for _ in range(20):
        r = random.randint(0, n - 1)
        # Cipolla-like: work in the extension ring Z/nZ[√(r²-a)]
        d = (r * r - a) % n
        if d == 0:
            if (r * r) % n == a:
                roots.append(r)
            continue

        if _jacobi(d, n) != -1:
            continue

        # Compute (r + √d)^((n+1)/2) in Z/nZ[√d]
        # Result should be (x, 0) if successful, where x² ≡ a (mod n)
        result = _pow_extension(r, 1, (n + 1) // 2, d, n)
        if result is not None:
            x, y = result
            if y == 0 and (x * x) % n == a:
                roots.append(x)
                roots.append((n - x) % n)

    return list(set(roots))


def _pow_extension(a_real: int, a_imag: int, exp: int, d: int, n: int):
    """Compute (a_real + a_imag * √d)^exp in Z/nZ[√d]."""
    # Binary exponentiation
    rx, ry = 1, 0  # result = 1
    bx, by = a_real % n, a_imag % n  # base

    while exp > 0:
        if exp & 1:
            # result *= base
            new_rx = (rx * bx + ry * by % n * d) % n
            new_ry = (rx * by + ry * bx) % n
            rx, ry = new_rx, new_ry
        # base *= base
        new_bx = (bx * bx + by * by % n * d) % n
        new_by = (2 * bx * by) % n
        bx, by = new_bx, new_by
        exp >>= 1

    return (rx, ry)


# =============================================================================
# IDEA 5: Trace-based factoring
# =============================================================================
# For an element a ∈ (Z/nZ)*, the "trace" tr(a) = a + a^(-1) mod n
# has special properties: tr(a) mod p = a_p + a_p^(-1) where a_p = a mod p.
#
# If a has order r mod p, then {tr(a^k) : k = 0, ..., r-1} takes at most
# (r+1)/2 distinct values (since tr(a^k) = tr(a^{-k})).
#
# The key insight: tr(a^k) can be computed using a LINEAR RECURRENCE:
# tr(a^{k+1}) = tr(a) * tr(a^k) - tr(a^{k-1})
# This is just the Chebyshev recurrence! (Lucas sequences are traces.)
#
# Novel idea: the minimal polynomial of the sequence {tr(a^k)} mod p
# has degree related to the structure of the group. If we can detect
# when this minimal polynomial has degree < expected, we get information.

def trace_sequence_attack(n: int) -> Optional[list[int]]:
    """
    Analyze the trace sequence tr(a^k) = a^k + a^(-k) mod n.
    This sequence satisfies tr(k+1) = tr(1)*tr(k) - tr(k-1).

    We look for:
    1. Values where the sequence mod p "wraps" but mod q doesn't
    2. GCD of sequential differences that reveal factor structure
    3. Polynomial relations in the trace values
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()

    for _ in range(bits * 3):
        a = random.randint(2, n - 1)
        g = math.gcd(a, n)
        if 1 < g < n:
            return _finish(n, g)

        # Compute a^(-1) mod n
        try:
            a_inv = pow(a, -1, n)
        except ValueError:
            g = math.gcd(a, n)
            if 1 < g < n:
                return _finish(n, g)
            continue

        # tr(a) = a + a^(-1) mod n
        tr_1 = (a + a_inv) % n

        # Generate trace sequence using recurrence
        tr_prev = 2  # tr(a^0) = 2
        tr_curr = tr_1

        # Accumulate product of (tr(k) - 2) — this is 0 when a^k = 1
        accumulated = 1
        max_k = bits * bits * 2

        for k in range(2, min(max_k, 20000)):
            tr_next = (tr_1 * tr_curr - tr_prev) % n
            tr_prev, tr_curr = tr_curr, tr_next

            # tr(k) = 2 means a^k + a^(-k) = 2, i.e., a^k = 1 (mod something)
            diff = (tr_curr - 2) % n
            if diff == 0:
                # a has order dividing k mod n (both p and q)
                # Try factoring via order
                factor = _factor_from_order_composite(a, k, n)
                if factor:
                    return factor
                continue

            # Accumulate
            accumulated = (accumulated * diff) % n

            if k % 50 == 0:
                g = math.gcd(accumulated, n)
                if 1 < g < n:
                    return _finish(n, g)
                if g == n:
                    # All diffs were 0 mod n — reset
                    accumulated = 1

        g = math.gcd(accumulated, n)
        if 1 < g < n:
            return _finish(n, g)

    return None


def _factor_from_order_composite(a: int, order: int, n: int) -> Optional[list[int]]:
    """Given a potential order of a mod n, try to extract factors."""
    # If order is the order mod both p and q, it's the lcm
    # Factor it and try divisors
    if order <= 1:
        return None

    # Try: a^(order/ℓ) for prime ℓ dividing order
    temp = order
    for ℓ in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        while temp % ℓ == 0:
            half = temp // ℓ
            val = pow(a, half, n)
            if val != 1:
                g = math.gcd(val - 1, n)
                if 1 < g < n:
                    return _finish(n, g)
            temp = half
            if temp == 0:
                break

    return None


# =============================================================================
# IDEA 6: Resultant-based polynomial attack
# =============================================================================
# For n = pq, consider two polynomials:
# f(x) = x^2 - s*x + n  (where s = p + q, roots are p, q)
# g(x) = x^k mod (x^2 - s*x + n)  for various k
#
# We don't know s, but we can compute in the quotient ring Z[x]/(x^2 + n)
# (setting s = 0 as approximation) and look for structure.
#
# More concretely: in the ring R = Z[x]/(x^2 - n), we have
# x^2 = n, so x acts like √n. Elements are a + b*√n.
# The norm is a² - n*b². This is just Z[√n] again.
#
# Novel angle: work in R = (Z/nZ)[x]/(x^2 - c) for RANDOM c.
# For each c, R mod p ≅ F_p² or F_p × F_p depending on whether
# c is a QR mod p. If c is QR mod p but QNR mod q (or vice versa),
# then R has fundamentally different structure mod p vs mod q.
# This difference can be detected!

def polynomial_ring_attack(n: int) -> Optional[list[int]]:
    """
    Work in the ring (Z/nZ)[x]/(x^2 - c) for various c.
    Compute high powers of (1 + x) in this ring.
    The behavior differs mod p vs mod q when c has different
    quadratic residuosity.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()

    for c in range(2, min(bits * bits, 300)):
        j = _jacobi(c, n)
        if j == 0:
            g = math.gcd(c, n)
            if 1 < g < n:
                return _finish(n, g)
            continue

        # Work in (Z/nZ)[x]/(x² - c)
        # Element: (a, b) represents a + b*x, where x² = c
        # Multiplication: (a1+b1*x)(a2+b2*x) = (a1*a2 + b1*b2*c) + (a1*b2+a2*b1)*x
        # Norm: a² - c*b²

        # Compute (1 + x)^(n-1) in this ring
        # By Frobenius, if the ring is F_p × F_p (c is QR mod p):
        #   (1+x)^(p-1) = ((1+√c)(1-√c))^...  complicated
        # If the ring is F_p² (c is QNR mod p):
        #   (1+x)^(p²-1) = 1 in this ring

        # Compute (1 + x)^((n-1)/2) in (Z/nZ)[x]/(x²-c)
        result = _pow_in_ring(1, 1, (n - 1) // 2, c, n)
        if result is None:
            continue
        a_res, b_res = result

        # Check if this reveals a factor
        # The norm is a_res² - c * b_res²
        norm = (a_res * a_res - c * b_res * b_res) % n
        for val in [a_res - 1, a_res + 1, b_res, norm - 1, norm + 1, a_res, norm]:
            g = math.gcd(val % n, n)
            if 1 < g < n:
                return _finish(n, g)

        # Compute (1 + x)^n in (Z/nZ)[x]/(x²-c)
        result_n = _pow_in_ring(1, 1, n, c, n)
        if result_n is None:
            continue
        an, bn = result_n

        # By Frobenius endomorphism:
        # In F_p: (1+x)^p = 1 + x^p = 1 + x^p
        # If c is QR mod p: x^p = x (since x ∈ F_p), so (1+x)^p = 1 + x ← only if p=2
        # Actually: x^p in F_p[x]/(x²-c) depends on whether x² - c splits
        # If splits: x^((p-1)/2) = 1, so x^p = x
        # If doesn't split: x^((p-1)/2) = -1 (mod p), so x^p = -x (mod p)
        # Therefore: (1+x)^p = 1 + x if (c/p) = 1
        #            (1+x)^p = 1 - x if (c/p) = -1
        # So (1+x)^n = (1+x)^p * (1+x)^q (in the CRT sense)

        # If (c/p) ≠ (c/q), then:
        # (1+x)^n mod p = (1+x)^(p*q) mod p = ((1+x)^p)^q mod p
        # = (1+x)^q or (1-x)^q depending on (c/p)
        # This gets complicated, but the KEY test:
        # If (c/p)=1 and (c/q)=-1:
        #   (1+x)^n mod p ≡ ((1+x))^q ≡ (1 + εx) for some ε
        #   (1+x)^n mod q ≡ ((1-x))^p ≡ (1 + ε'x) for some ε'
        #   These are different!

        # Simple test: check if bn = 0 or if an = 1
        for val in [an - 1, an + 1, bn, bn - 1, bn + 1, an - bn, an + bn]:
            g = math.gcd(val % n, n)
            if 1 < g < n:
                return _finish(n, g)

    return None


def _pow_in_ring(a: int, b: int, exp: int, c: int, n: int):
    """Compute (a + b*x)^exp in (Z/nZ)[x]/(x²-c). Returns (real, imag) or None."""
    if exp < 0:
        return None

    # Binary exponentiation
    rx, ry = 1, 0  # result = 1
    bx, by = a % n, b % n

    while exp > 0:
        if exp & 1:
            new_rx = (rx * bx + ry * by * c) % n
            new_ry = (rx * by + ry * bx) % n
            rx, ry = new_rx, new_ry
        new_bx = (bx * bx + by * by * c) % n
        new_by = (2 * bx * by) % n
        bx, by = new_bx, new_by
        exp >>= 1

    return (rx, ry)


# =============================================================================
# IDEA 7: Multi-exponent GCD accumulation
# =============================================================================
# Instead of one exponent (like in p-1), use MULTIPLE carefully chosen
# exponents simultaneously. The Chinese Remainder structure means that
# different exponents "hit" different parts of the group structure.
#
# Key insight: let e₁, e₂, ..., eₖ be exponents. Compute:
# product = Π_i (a^eᵢ - 1) mod n
# Then gcd(product, n) reveals p if ANY eᵢ is a multiple of ord_p(a)
# but not all eᵢ are multiples of ord_q(a) (or vice versa).
#
# The art is choosing the eᵢ to maximize coverage.
# Novel choice: use eᵢ = n^i mod (small primes product) for various i.
# This probes the group structure at different "frequencies".

def multi_exponent_attack(n: int) -> Optional[list[int]]:
    """
    Compute product of (a^eᵢ - 1) mod n for carefully chosen exponents eᵢ.
    The exponents are derived from n itself to maximize information extraction.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()

    # Generate exponent families
    exponent_families = []

    # Family 1: powers of small primes up to B
    B = bits * 3
    exp1 = 1
    for p in _sieve(B):
        pk = p
        while pk <= B:
            exp1 *= p
            pk *= p
    exponent_families.append(("smooth_B", exp1))

    # Family 2: (n-1)/2, (n+1)/2
    exponent_families.append(("n_minus_1_half", (n - 1) // 2))
    exponent_families.append(("n_plus_1_half", (n + 1) // 2))

    # Family 3: isqrt(n), isqrt(n) ± 1
    sq = math.isqrt(n)
    exponent_families.append(("isqrt", sq))
    exponent_families.append(("isqrt_m1", sq - 1))
    exponent_families.append(("isqrt_p1", sq + 1))

    # Family 4: n mod m for various m (probes the residue structure)
    for m in [3, 5, 7, 8, 11, 13, 16, 17, 19, 23, 24, 32]:
        exponent_families.append((f"n_mod_{m}", n % m if n % m > 0 else m))

    # Family 5: Fibonacci and Lucas numbers near bits
    fib_a, fib_b = 1, 1
    for i in range(bits * 2):
        fib_a, fib_b = fib_b, fib_a + fib_b
    exponent_families.append(("fibonacci", fib_b))

    # Family 6: factorial-like (lcm of range)
    lcm_val = 1
    for i in range(2, bits * 2):
        lcm_val = lcm_val * i // math.gcd(lcm_val, i)
    exponent_families.append(("lcm_range", lcm_val))

    # Try multiple bases
    for _ in range(bits * 2):
        a = random.randint(2, n - 1)
        g = math.gcd(a, n)
        if 1 < g < n:
            return _finish(n, g)

        # Accumulate product of (a^eᵢ - 1)
        accumulated = 1
        for name, exp in exponent_families:
            if exp <= 0 or exp > 10**18:  # skip absurdly large exponents
                # For very large exponents, compute modularly
                val = pow(a, exp % (n * n), n) if exp > n * n else pow(a, exp, n)
            else:
                val = pow(a, exp, n)

            diff = (val - 1) % n
            if diff == 0:
                continue
            accumulated = (accumulated * diff) % n

            # Also check individually
            g = math.gcd(diff, n)
            if 1 < g < n:
                return _finish(n, g)

        g = math.gcd(accumulated, n)
        if 1 < g < n:
            return _finish(n, g)

    return None


# =============================================================================
# MASTER: try all novel approaches in order of speed
# =============================================================================

def novel_combined(n: int) -> Optional[list[int]]:
    """Run all novel approaches."""
    if n < 4:
        return None
    if n % 2 == 0:
        return _finish(n, 2)

    for algo in [
        multi_exponent_attack,       # fast, broad coverage
        power_residue_attack,        # generalized Miller-Rabin
        character_matrix_attack,     # character theory
        polynomial_ring_attack,      # Frobenius in extension rings
        trace_sequence_attack,       # Lucas/Chebyshev sequences
        algebraic_norm_attack,       # number field norms
        coppersmith_attack,          # lattice-based
    ]:
        try:
            result = algo(n)
            if result is not None:
                return result
        except Exception:
            continue

    return None


# =============================================================================
# Helpers
# =============================================================================

def _finish(n, f):
    factors = []
    remaining = n
    for part in [f, n // f]:
        if part <= 1:
            continue
        if isprime(part):
            factors.append(part)
        else:
            from baselines import pollard_rho
            sub = pollard_rho(part)
            if sub:
                factors.extend(sub)
            else:
                factors.append(part)
    result = []
    temp = n
    for ff in sorted(set(factors)):
        while temp % ff == 0:
            result.append(ff)
            temp //= ff
    if temp > 1:
        result.append(temp)
    return sorted(result)


def _sieve(limit):
    if limit < 2:
        return []
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            for j in range(i*i, limit + 1, i):
                s[j] = False
    return [i for i in range(limit + 1) if s[i]]


def _jacobi(a, n):
    if n <= 0 or n % 2 == 0:
        return 0
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in [3, 5]:
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0
