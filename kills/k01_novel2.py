#!/usr/bin/env python3
"""
Novel approaches v2 — refined based on Round 1 results.

The best performers from v1 were:
- trace_sequence (40 bits): Lucas/Chebyshev sequences with GCD accumulation
- polynomial_ring (32 bits): Frobenius in extension rings

Key lesson: approaches that check GCDs of ACCUMULATED products work better
than those checking individual GCDs, because accumulation amplifies signal.

V2 focuses on maximizing the probability of GCD accumulation hitting a factor.
"""

import math
import random
from typing import Optional
from sympy import isprime


# =============================================================================
# APPROACH 1: Frobenius ring walk with GCD accumulation
# =============================================================================
# This combines the best ideas from polynomial_ring and trace_sequence.
#
# For each small c, work in (Z/nZ)[x]/(x² - c).
# Compute (1+x)^k for k = 2, 3, 5, 7, 11, ... (primes) using repeated squaring.
# After each prime power, accumulate the "imaginary part" b_k into a product.
#
# Why this works: in F_p[x]/(x²-c), the Frobenius acts as x ↦ x^p.
# If (c/p) = 1: (1+x)^p ≡ 1+x (mod p), so b_p ≡ 1 (mod p)
# If (c/p) = -1: (1+x)^p ≡ 1-x (mod p), so b_p ≡ -1 (mod p)
#
# For n = pq, (1+x)^p mod n has b-component that encodes (c/p) vs (c/q).
# We can't compute (1+x)^p directly (don't know p), but we can compute
# (1+x)^k for many k values and accumulate b-components.
#
# If ANY k happens to equal 0 mod ord_p but not mod ord_q (in the ring),
# the accumulated product reveals the factor.

def frobenius_walk(n: int) -> Optional[list[int]]:
    """
    Frobenius-based factoring via extension ring walks.
    Try multiple ring extensions (Z/nZ)[x]/(x² - c) for various c.
    For each, compute high powers via prime-by-prime exponentiation
    and accumulate imaginary parts.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    primes = _sieve(max(bits * bits, 500))

    for c in range(2, min(bits * 3, 100)):
        j = _jacobi(c, n)
        if j == 0:
            g = math.gcd(c, n)
            if 1 < g < n:
                return _finish(n, g)
            continue

        # Compute (1 + x)^M where M = lcm(1..B) via sequential prime powers
        # In ring (Z/nZ)[x]/(x² - c): multiply (a + bx)
        a, b = 1, 1  # start with 1 + x
        accumulated = 1

        for p in primes:
            # Raise to p-th power
            a, b = _pow_ring(a, b, p, c, n)

            # Accumulate b (imaginary part)
            if b != 0:
                accumulated = (accumulated * b) % n

            # Periodic check
            g = math.gcd(accumulated, n)
            if 1 < g < n:
                return _finish(n, g)
            if g == n:
                # Reset — all b's were 0 mod n
                accumulated = 1

            # Also check a - 1 and b directly
            g = math.gcd((a - 1) % n, n)
            if 1 < g < n:
                return _finish(n, g)

        # Final check
        g = math.gcd(accumulated, n)
        if 1 < g < n:
            return _finish(n, g)

    return None


def _pow_ring(a, b, exp, c, n):
    """Compute (a + bx)^exp in (Z/nZ)[x]/(x²-c)."""
    ra, rb = 1, 0  # result = 1
    ba, bb = a % n, b % n  # base

    while exp > 0:
        if exp & 1:
            new_ra = (ra * ba + rb * bb % n * c) % n
            new_rb = (ra * bb + rb * ba) % n
            ra, rb = new_ra, new_rb
        new_ba = (ba * ba + bb * bb % n * c) % n
        new_bb = (2 * ba * bb) % n
        ba, bb = new_ba, new_bb
        exp >>= 1

    return ra, rb


# =============================================================================
# APPROACH 2: Multi-ring Frobenius with systematic c selection
# =============================================================================
# Instead of trying random c values, choose c systematically:
# Use the first k primes as c values. For each, compute (1+x)^((n²-1)/2)
# in (Z/nZ)[x]/(x²-c).
#
# Key insight: (1+x)^(n²-1) ≡ 1 in (Z/nZ)[x]/(x²-c) when x²-c is
# irreducible mod both p and q. But if it's irreducible mod one and
# reducible mod the other, the behavior differs.
#
# Computing (1+x)^((n²-1)/2) takes O(log(n²)) = O(bits) ring multiplications,
# each costing O(bits²). Total: O(bits³) per c value. Polynomial!

def frobenius_systematic(n: int) -> Optional[list[int]]:
    """
    For each small prime c, compute (1+x)^((n²-1)/2) in (Z/nZ)[x]/(x²-c).
    This is a Frobenius-based compositeness/factoring test.

    The exponent (n²-1)/2 is chosen because:
    - The multiplicative group of F_{p²} has order p²-1
    - Elements of (Z/nZ)[x]/(x²-c) "live" in this group
    - (n²-1)/2 is the analogue of (n-1)/2 in Euler's criterion
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()

    # Precompute n² - 1 and its useful divisors
    n_sq = n * n
    exp_half = (n_sq - 1) // 2

    for c in range(2, min(bits * 5, 200)):
        j = _jacobi(c, n)
        if j == 0:
            g = math.gcd(c, n)
            if 1 < g < n:
                return _finish(n, g)
            continue

        # Compute (1 + x)^((n²-1)/2) in (Z/nZ)[x]/(x²-c)
        a, b = _pow_ring(1, 1, exp_half, c, n)

        # Check: in the "nice" case, this should be ±1 (norm 1 element)
        # If it's not ±1, we might factor
        for val in [a - 1, a + 1, b, a - b, a + b]:
            g = math.gcd(val % n, n)
            if 1 < g < n:
                return _finish(n, g)

        # Also try (n-1)/2 exponent (Euler criterion in the ring)
        if (n - 1) % 2 == 0:
            a2, b2 = _pow_ring(1, 1, (n - 1) // 2, c, n)
            for val in [a2 - 1, a2 + 1, b2, b2 - 1, b2 + 1]:
                g = math.gcd(val % n, n)
                if 1 < g < n:
                    return _finish(n, g)

        # Try n+1 exponent (related to p+1 method)
        a3, b3 = _pow_ring(1, 1, n + 1, c, n)
        for val in [a3 - 1, a3 + 1, b3, b3 - 1]:
            g = math.gcd(val % n, n)
            if 1 < g < n:
                return _finish(n, g)

        # Try (n-1)(n+1)/4 = (n²-1)/4 if divisible
        if (n_sq - 1) % 4 == 0:
            a4, b4 = _pow_ring(1, 1, (n_sq - 1) // 4, c, n)
            for val in [a4 - 1, a4 + 1, b4]:
                g = math.gcd(val % n, n)
                if 1 < g < n:
                    return _finish(n, g)

    return None


# =============================================================================
# APPROACH 3: Iterated Frobenius with smooth exponent
# =============================================================================
# Combine Frobenius ring with p-1 style smooth exponent.
# Compute (1+x)^(B!) in (Z/nZ)[x]/(x²-c).
# This captures BOTH p-1 smoothness AND p+1 smoothness simultaneously!
#
# Why: the order of (1+x) in the multiplicative group of (Z/nZ)[x]/(x²-c)
# divides lcm of:
#   - ord in (F_p[x]/(x²-c))* : divides p²-1 if irreducible, (p-1)² if reducible
#   - ord in (F_q[x]/(x²-c))* : divides q²-1 if irreducible, (q-1)² if reducible
#
# If (c/p)=-1 (irreducible mod p): order divides p²-1 = (p-1)(p+1)
# If (c/p)=1 (reducible mod p): order divides (p-1)²... actually lcm(p-1, p-1) = p-1
#
# So: if (c/p)=-1, order divides (p-1)(p+1). The smooth part of (p-1)(p+1)
# is typically larger than just (p-1) because we get TWO numbers' smooth parts.
# This is Williams' p+1 method embedded in a more general framework!
#
# The KEY advantage: by trying multiple c values, we effectively try
# p-1 method, p+1 method, AND hybrid methods simultaneously.

def iterated_frobenius(n: int) -> Optional[list[int]]:
    """
    The most promising novel approach: p-1/p+1 hybrid via Frobenius rings.
    For each c, compute (1+x)^(M) where M = lcm(1..B) in (Z/nZ)[x]/(x²-c).
    Check both real and imaginary parts for factor extraction.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    B = bits * bits  # smoothness bound — polynomial in bits

    primes = _sieve(B)

    for c in range(2, min(bits * 3, 80)):
        j = _jacobi(c, n)
        if j == 0:
            g = math.gcd(c, n)
            if 1 < g < n:
                return _finish(n, g)
            continue

        # Compute (1+x)^(lcm(1..B)) in the ring
        a, b = 1, 1  # 1 + x

        for p in primes:
            pk = p
            while pk <= B:
                a, b = _pow_ring(a, b, p, c, n)
                pk *= p

            # Check after each prime
            for val in [b, a - 1, a + 1]:
                v = val % n
                if v == 0:
                    continue
                g = math.gcd(v, n)
                if 1 < g < n:
                    return _finish(n, g)

    return None


# =============================================================================
# APPROACH 4: Multi-dimensional Frobenius
# =============================================================================
# Go beyond degree 2: use (Z/nZ)[x]/(f(x)) for cubic and quartic polynomials.
# The group structure is richer, giving more chances to split.
#
# For f(x) = x³ - c: the Galois group matters.
# If f splits as (x-r)(x²+rx+r²) mod p, the ring structure is F_p × F_{p²}.
# The order of elements divides lcm(p-1, p²-1) = (p-1)(p²-1)/(gcd stuff).
# This gives even more "smooth chances" than degree 2.

def cubic_frobenius(n: int) -> Optional[list[int]]:
    """
    Work in (Z/nZ)[x]/(x³ - c) for various c.
    Compute (1+x)^(lcm(1..B)) and check for factor extraction.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    B = bits * bits
    primes = _sieve(B)

    for c in range(2, min(bits * 2, 50)):
        g = math.gcd(c, n)
        if 1 < g < n:
            return _finish(n, g)

        # In (Z/nZ)[x]/(x³-c): elements are (a, b, d) = a + bx + dx²
        # x³ = c, so x² * x = c, used in multiplication

        # Start with 1 + x = (1, 1, 0)
        elem = (1, 1, 0)

        for p in primes:
            pk = p
            while pk <= B:
                elem = _pow_cubic_ring(elem, p, c, n)
                pk *= p

            # Check all components
            a, b, d = elem
            for val in [a - 1, a + 1, b, b - 1, d, d - 1, a - b, a + b, b - d]:
                v = val % n
                if v == 0:
                    continue
                g = math.gcd(v, n)
                if 1 < g < n:
                    return _finish(n, g)

    return None


def _mul_cubic(e1, e2, c, n):
    """Multiply two elements in (Z/nZ)[x]/(x³-c)."""
    a1, b1, d1 = e1
    a2, b2, d2 = e2
    # (a1 + b1*x + d1*x²)(a2 + b2*x + d2*x²)
    # = a1*a2 + (a1*b2 + b1*a2)*x + (a1*d2 + b1*b2 + d1*a2)*x²
    #   + (b1*d2 + d1*b2)*x³ + d1*d2*x⁴
    # x³ = c, x⁴ = c*x
    ra = (a1*a2 + c * (b1*d2 + d1*b2)) % n
    rb = (a1*b2 + b1*a2 + c * d1*d2) % n
    rd = (a1*d2 + b1*b2 + d1*a2) % n
    return (ra, rb, rd)


def _pow_cubic_ring(elem, exp, c, n):
    """Compute elem^exp in (Z/nZ)[x]/(x³-c)."""
    result = (1, 0, 0)  # identity
    base = (elem[0] % n, elem[1] % n, elem[2] % n)

    while exp > 0:
        if exp & 1:
            result = _mul_cubic(result, base, c, n)
        base = _mul_cubic(base, base, c, n)
        exp >>= 1

    return result


# =============================================================================
# APPROACH 5: Combined multi-ring attack
# =============================================================================
# Run Frobenius in degree 2 AND degree 3 rings simultaneously,
# accumulating GCDs across all of them.

def multi_ring_attack(n: int) -> Optional[list[int]]:
    """
    Combined attack using multiple ring extensions simultaneously.
    For each c: try both (Z/nZ)[x]/(x²-c) and (Z/nZ)[x]/(x³-c).
    Accumulate GCD products across all rings for maximum coverage.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    B = bits * bits
    primes = _sieve(B)

    accumulated = 1

    for c in range(2, min(bits * 2, 50)):
        g = math.gcd(c, n)
        if 1 < g < n:
            return _finish(n, g)

        # Degree 2 ring
        a2, b2 = 1, 1
        for p in primes:
            pk = p
            while pk <= B:
                a2, b2 = _pow_ring(a2, b2, p, c, n)
                pk *= p

        for val in [b2, a2 - 1, a2 + 1]:
            v = val % n
            if v != 0:
                accumulated = (accumulated * v) % n
                g = math.gcd(accumulated, n)
                if 1 < g < n:
                    return _finish(n, g)
                if g == n:
                    accumulated = 1

        # Degree 3 ring
        e3 = (1, 1, 0)
        for p in primes:
            pk = p
            while pk <= B:
                e3 = _pow_cubic_ring(e3, p, c, n)
                pk *= p

        a3, b3, d3 = e3
        for val in [b3, d3, a3 - 1, a3 + 1]:
            v = val % n
            if v != 0:
                accumulated = (accumulated * v) % n
                g = math.gcd(accumulated, n)
                if 1 < g < n:
                    return _finish(n, g)
                if g == n:
                    accumulated = 1

    return None


# =============================================================================
# MASTER COMBINATION
# =============================================================================

def novel2_combined(n: int) -> Optional[list[int]]:
    if n < 4:
        return None
    if n % 2 == 0:
        return _finish(n, 2)

    for algo in [
        frobenius_systematic,   # O(bits³) per c value, pure Frobenius test
        iterated_frobenius,     # p-1/p+1 hybrid via smooth exponent in ring
        cubic_frobenius,        # degree-3 extension rings
        multi_ring_attack,      # combined multi-ring
        frobenius_walk,         # prime-by-prime ring walk
    ]:
        try:
            result = algo(n)
            if result:
                return result
        except Exception:
            continue
    return None


# =============================================================================
# Helpers
# =============================================================================

def _finish(n, f):
    factors = []
    for part in [f, n // f]:
        if part <= 1:
            continue
        if isprime(part):
            factors.append(part)
        else:
            from baselines import pollard_rho
            sub = pollard_rho(part)
            factors.extend(sub if sub else [part])
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
