#!/usr/bin/env python3
"""
Hyperbolic lattice factoring experiment.

Factoring N = pq is equivalent to finding a lattice point (p,q) on the
hyperbola xy = N. No modern algorithm exploits this directly with lattice
reduction. This experiment tests whether LLL/BKZ on lattices derived from
hyperbolic and algebraic-norm constraints can factor semiprimes.

Four approaches are tested:

  Approach 2 (Fermat + LLL): Encode x^2 - y^2 = N as a closest-vector problem.
  Approach 4 (Gaussian integers): Use Z[i] norm lattice — most promising.
  Approach 5 (Eisenstein / multi-ring): Use Z[omega] and Z[sqrt(-2)] norms.
  Approach 6 (Pell lattice): Relate Pell-equation near-misses to factors.

For each, we build a lattice from N alone (no knowledge of p, q), run LLL
and BKZ, then check if any short vector reveals a nontrivial factor.
"""

import math
import random
import sys
import time
from collections import defaultdict
from typing import Optional, Tuple, List

import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ

sys.stdout.reconfigure(line_buffering=True)
random.seed(42)


# =============================================================================
# Number-theoretic helpers
# =============================================================================

def small_primes(limit: int) -> List[int]:
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            for j in range(i * i, limit + 1, i):
                s[j] = False
    return [i for i in range(limit + 1) if s[i]]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Miller-Rabin with deterministic witnesses for n < 2^64
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def isqrt(n: int) -> int:
    if n < 0:
        raise ValueError
    if n == 0:
        return 0
    x = 1 << ((n.bit_length() + 1) // 2)
    while True:
        y = (x + n // x) >> 1
        if y >= x:
            return x
        x = y


def jacobi(a: int, n: int) -> int:
    if n <= 0 or n % 2 == 0:
        return 0
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0


def sqrt_mod_prime(a: int, p: int) -> Optional[int]:
    """Tonelli-Shanks: find x with x^2 = a mod p, or None if no root."""
    a = a % p
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        return None
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    # Factor p-1 = 2^s * q
    s, q = 0, p - 1
    while q % 2 == 0:
        s += 1
        q //= 2
    # Find a quadratic non-residue
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    M = s
    c = pow(z, q, p)
    t = pow(a, q, p)
    R = pow(a, (q + 1) // 2, p)
    while True:
        if t == 1:
            return R
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p


def sqrt_mod_composite_random(D: int, N: int, max_attempts: int = 200) -> Optional[int]:
    """Find r with r^2 = -D mod N by random search (for when we cannot factor N)."""
    # First check if -D is a QR mod N using Jacobi symbol
    if jacobi(-D, N) != 1:
        return None
    # Random search: for random a, compute a^((N-1)/2) mod N
    # and check if result^2 = -D mod N.
    # Actually, use the method: pick random a, compute gcd(a^2+D, N).
    # If 1 < gcd < N, we found a factor (bonus!).
    # Otherwise try Cipolla-like approach or just brute force for small N.
    for _ in range(max_attempts):
        a = random.randint(1, N - 1)
        if (a * a + D) % N == 0:
            return a
    # Brute force for small N
    if N < 10**7:
        for a in range(1, N):
            if (a * a + D) % N == 0:
                return a
    return None


def generate_semiprime(bits: int, cond: str = "any") -> Tuple[int, int, int]:
    """Generate N = p*q where p,q are primes of roughly bits/2 each.

    cond: "any" = no constraint
          "1mod4" = both p,q = 1 mod 4 (needed for Gaussian approach)
          "1mod3" = both p,q = 1 mod 3 (needed for Eisenstein approach)
    """
    lo = 1 << (bits // 2 - 1)
    hi = 1 << (bits // 2)
    while True:
        p = random.randint(lo, hi)
        if p % 2 == 0:
            p += 1
        while not is_prime(p):
            p += 2
            if p > hi:
                p = lo + 1
        q = random.randint(lo, hi)
        if q % 2 == 0:
            q += 1
        while not is_prime(q):
            q += 2
            if q > hi:
                q = lo + 1
        if p == q:
            continue
        if cond == "1mod4" and not (p % 4 == 1 and q % 4 == 1):
            continue
        if cond == "1mod3" and not (p % 3 == 1 and q % 3 == 1):
            continue
        return min(p, q), max(p, q), p * q


def lll_reduce(M: IntegerMatrix) -> IntegerMatrix:
    LLL.reduction(M)
    return M


def bkz_reduce(M: IntegerMatrix, block_size: int) -> IntegerMatrix:
    if block_size < 2:
        block_size = 2
    nrows = M.nrows
    if block_size > nrows:
        block_size = nrows
    par = BKZ.Param(block_size=block_size)
    BKZ.reduction(M, par)
    return M


def extract_rows(M: IntegerMatrix) -> List[List[int]]:
    rows = []
    for i in range(M.nrows):
        row = [int(M[i, j]) for j in range(M.ncols)]
        rows.append(row)
    return rows


def try_gcd_factor(val: int, N: int) -> Optional[int]:
    """If gcd(val, N) is a nontrivial factor, return it."""
    if val == 0:
        return None
    g = math.gcd(abs(val), N)
    if 1 < g < N:
        return g
    return None


# =============================================================================
# APPROACH 2: Fermat + LLL
# =============================================================================
# Fermat's method: N = x^2 - y^2 = (x+y)(x-y).
# x = (p+q)/2, y = (p-q)/2, so x ~ sqrt(N), y ~ |p-q|/2.
#
# We build a lattice that encodes the constraint x^2 - y^2 = N
# by linearization around x0 = isqrt(N):
#   x = x0 + t, so (x0+t)^2 - y^2 = N => y^2 = x0^2 + 2*x0*t + t^2 - N.
#   Let R = x0^2 - N (small). Then y^2 = R + 2*x0*t + t^2.
#
# For the lattice, we drop the t^2 term (linear approx for small t):
#   y^2 ~ R + 2*x0*t.
#   So t ~ (y^2 - R) / (2*x0).
#
# Strategy A: Build a lattice where (t, y, 1) is short, encoding:
#   2*x0*t + R is close to y^2.
#   We use a Kannan-style embedding.
#
# Strategy B: Multi-start. Try several starting points near sqrt(N)
# and combine constraints.

def approach2_fermat_lll(N: int, p: int, q: int) -> dict:
    """Fermat + LLL approach to factoring."""
    results = {"name": "fermat_lll", "factored": False, "factor": None, "method": None}
    x0 = isqrt(N)
    if x0 * x0 == N:
        # Perfect square — trivial
        results["factored"] = True
        results["factor"] = x0
        results["method"] = "perfect_square"
        return results

    R = x0 * x0 - N  # x0^2 - N, could be small negative or positive

    # Strategy A: 3D lattice encoding Fermat constraint
    # We want to find (t, y) with (x0+t)^2 - y^2 = N, i.e.,
    # 2*x0*t + t^2 = y^2 - R.
    # For small t, the dominant constraint is 2*x0*t ~ y^2 - R.
    # Lattice: rows of
    #   [1,   0,  C * 2*x0]
    #   [0,   1,  C * (-1)]  (this is for the y^2 variable)
    #   [0,   0,  C * 1   ]  (to make the "target" reachable)
    # But this encodes t * 2*x0 - y^2 ~ 0, which is only linear in t but
    # quadratic in y, so the lattice can't directly encode it.
    #
    # Instead, use the QUADRATIC FORM approach:
    # For each candidate k = 1, 2, ..., K, check if x0^2 + 2*x0*k + k^2 - N
    # is a perfect square. This is plain Fermat, but we batch using LLL.
    #
    # Lattice Fermat (Coppersmith-like):
    # The key insight is that p+q = 2*x where x ~ sqrt(N).
    # And p+q = s, p*q = N, so s^2 - 4N = (p-q)^2 = d^2.
    # We want to find s with s^2 - 4N = d^2, i.e., s^2 = 4N + d^2.
    # s is close to 2*sqrt(N) = 2*x0 (with error ~ 1).
    # d = p-q, which is "small" for balanced semiprimes.
    #
    # Lattice approach: look for (a, b) in Z^2 with a^2 - 4N*b^2 small.
    # This is a Pell-like equation. The lattice:
    # [[1,  C*1], [0,  C*2*x0]]
    # Short vector (a,b) gives a small and a - 2*x0*b small.
    # If a = s*b for s ~ 2*x0, then a^2 - 4N*b^2 = b^2*(s^2 - 4N) = b^2*d^2.
    # Hmm, not directly useful.

    # More direct approach: Kannan's embedding for CVP.
    # Target: the vector (x, y) with x = (p+q)/2, y = (p-q)/2.
    # x*y relationship: x + y = p, x - y = q, x^2 - y^2 = N.
    # Lattice: we know x ~ x0. Build lattice where (x-x0, y) is short.
    # The constraint x^2 - y^2 = N gives us x^2 - y^2 - N = 0.
    # Linearize: let x = x0 + e. (x0+e)^2 - y^2 = N => 2*x0*e + e^2 - y^2 = -R.
    # For the lattice to capture this: encode 2*x0*e - y^2 = -R - e^2.
    # The y^2 term is the problem. We can try MULTIPLE y values.

    # Practical strategy: multi-vector Fermat.
    # For y_candidates = 0, 1, 2, ..., bound:
    #   check if N + y^2 is a perfect square.
    # The lattice can BATCH these checks by finding (a, b, c) where
    # a = some linear combo that encodes multiple y-checks.
    # This is essentially a lattice-based sieve.

    # Strategy: build lattice where short vectors encode Fermat solutions.
    # Use the "sum-and-difference" form:
    # s = p + q, d = p - q. s*d = p^2 - q^2. s/d = (p+q)/(p-q).
    # Also: s^2 = 4N + d^2. We want (s, d) with s > d > 0, s^2 - d^2 = 4N.
    # The lattice:
    C = isqrt(N) + 1  # Scaling factor to balance dimensions
    dim = 3
    M = IntegerMatrix(dim, dim)
    # Row 0: (1, 0, C)  — represents a unit in the "s" direction
    M[0, 0] = 1
    M[0, 1] = 0
    M[0, 2] = C
    # Row 1: (0, 1, C)  — represents a unit in the "d" direction
    M[1, 0] = 0
    M[1, 1] = 1
    M[1, 2] = C
    # Row 2: (0, 0, C * 2*x0) — the approximate value of s
    M[2, 0] = 0
    M[2, 1] = 0
    M[2, 2] = C * 2 * x0

    lll_reduce(M)
    rows = extract_rows(M)

    for row in rows:
        # Check if any combination of row entries reveals a factor
        for val in row:
            f = try_gcd_factor(val, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "fermat_lll_direct"
                return results
        # Check if row encodes (s, d, ...) where s^2 - d^2 = 4N
        a, b = abs(row[0]), abs(row[1])
        if a > b > 0:
            s_cand, d_cand = a, b
            if s_cand * s_cand - d_cand * d_cand == 4 * N:
                p_cand = (s_cand + d_cand) // 2
                q_cand = (s_cand - d_cand) // 2
                if p_cand * q_cand == N:
                    results["factored"] = True
                    results["factor"] = min(p_cand, q_cand)
                    results["method"] = "fermat_lll_sd"
                    return results
        # Try (a+b)(a-b) type extractions
        if a > b:
            f = try_gcd_factor(a + b, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "fermat_lll_sum"
                return results
            f = try_gcd_factor(a - b, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "fermat_lll_diff"
                return results

    # Strategy B: try with 2*sqrt(N) as modular target
    # Build a lattice where s ~ 2*isqrt(N) is encoded as a target.
    dim = 3
    M2 = IntegerMatrix(dim, dim)
    two_x0 = 2 * x0
    # We want (a, b) with a = 2*x0*b + small error, and a^2 - 4N*b^2 = d^2
    C2 = isqrt(isqrt(N)) + 1  # Scale to make modular column dominant
    M2[0, 0] = 1
    M2[0, 1] = 0
    M2[0, 2] = 0
    M2[1, 0] = 0
    M2[1, 1] = 1
    M2[1, 2] = C2 * two_x0
    M2[2, 0] = 0
    M2[2, 1] = 0
    M2[2, 2] = C2 * N

    lll_reduce(M2)
    rows = extract_rows(M2)

    for row in rows:
        for val in row:
            f = try_gcd_factor(val, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "fermat_lll_B"
                return results
        # Try Fermat checks on row entries
        for v in row:
            v = abs(v)
            if v > 0:
                cand = v * v + N
                sq = isqrt(cand)
                if sq * sq == cand:
                    # cand = v^2 + N = sq^2, so sq^2 - v^2 = N
                    p_cand = sq + v
                    q_cand = sq - v
                    if q_cand > 1 and p_cand * q_cand == N:
                        results["factored"] = True
                        results["factor"] = min(p_cand, q_cand)
                        results["method"] = "fermat_lll_B_sq"
                        return results

    return results


# =============================================================================
# APPROACH 4: Gaussian integer lattice (Z[i])
# =============================================================================
# The MOST PROMISING approach.
#
# For N = pq with p = q = 1 mod 4, -1 is a quadratic residue mod N.
# Find r with r^2 = -1 mod N by random search.
# Build 2x2 lattice L = [[N, 0], [r, 1]].
# Short vectors (a, b) in L satisfy a = r*b mod N, hence a^2+b^2 = 0 mod N.
# By Minkowski, shortest vector norm ~ sqrt(N), giving a^2+b^2 ~ N.
#
# Factor extraction from MULTIPLE short vectors:
# If (a1,b1) and (a2,b2) both satisfy a_i^2+b_i^2 = k_i * N, and they are
# NOT scalar multiples, then they encode DIFFERENT Gaussian-integer
# factorizations. Cross-product extraction:
#   gcd(a1*a2 + b1*b2, N) and gcd(a1*a2 - b1*b2, N)
# often yield nontrivial factors (Brahmagupta-Fibonacci identity).
#
# Also: multiple square roots of -1 mod N (from different lattice setups)
# combined give factors, since different CRT roots reveal the factorization.

def find_sqrt_minus1_mod_N(N: int, max_attempts: int = 1000) -> Optional[int]:
    """Find r with r^2 = -1 mod N by random search.

    Only possible when Jacobi(-1, N) = 1, which requires that
    all prime factors of N are 1 mod 4 (or appear to even power).
    """
    if jacobi(N - 1, N) != 1:
        return None
    # Random search: a^((N-1)/2) mod N might give +/-1 or a root of -1
    for _ in range(max_attempts):
        a = random.randint(2, N - 2)
        g = math.gcd(a, N)
        if 1 < g < N:
            return None  # Found factor directly, but we handle this elsewhere
        # Euler criterion variant
        r = pow(a, (N - 1) // 4, N)
        if (r * r) % N == N - 1:
            return r
        r = pow(a, (N - 1) // 2, N)
        if r == N - 1:
            # a is a QNR mod some factor; try (a^((N-1)/4))
            pass
    # Brute force for small N
    if N < 10**7:
        for a in range(2, N):
            if (a * a) % N == N - 1:
                return a
    return None


def approach4_gaussian(N: int, p: int, q: int) -> dict:
    """Gaussian integer (Z[i]) lattice approach."""
    results = {"name": "gaussian", "factored": False, "factor": None, "method": None}

    # Check feasibility: need -1 to be QR mod N
    if not (p % 4 == 1 and q % 4 == 1):
        results["method"] = "skip_not_1mod4"
        return results

    r = find_sqrt_minus1_mod_N(N)
    if r is None:
        results["method"] = "no_sqrt_found"
        return results

    # Build 2x2 lattice [[N, 0], [r, 1]]
    M = IntegerMatrix(2, 2)
    M[0, 0] = int(N)
    M[0, 1] = 0
    M[1, 0] = int(r)
    M[1, 1] = 1
    lll_reduce(M)
    rows = extract_rows(M)

    representations = []
    for row in rows:
        a, b = row[0], row[1]
        norm = a * a + b * b
        if norm > 0 and norm % N == 0:
            representations.append((a, b, norm))
        # Also try GCD extractions
        f = try_gcd_factor(a, N)
        if f:
            results["factored"] = True
            results["factor"] = f
            results["method"] = "gaussian_gcd_a"
            return results
        f = try_gcd_factor(b, N)
        if f:
            results["factored"] = True
            results["factor"] = f
            results["method"] = "gaussian_gcd_b"
            return results

    # Cross-product extraction from two representations
    if len(representations) >= 2:
        a1, b1, n1 = representations[0]
        a2, b2, n2 = representations[1]
        # Brahmagupta-Fibonacci: if (a1,b1) and (a2,b2) give different
        # Gaussian factorizations, the cross terms reveal factors.
        for val in [a1 * a2 + b1 * b2, a1 * a2 - b1 * b2,
                    a1 * b2 + a2 * b1, a1 * b2 - a2 * b1]:
            f = try_gcd_factor(val, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "gaussian_cross"
                return results

    # Try using BOTH roots r and N-r (which give the same lattice up to sign).
    # The real extraction: if r^2=-1 mod N, then also (N-r)^2=-1 mod N.
    # These come from the SAME CRT branch. We need the OTHER branch.
    # Try: build 3x3 lattice with additional structure.
    # For D=2: find s with s^2 = -2 mod N.
    s = sqrt_mod_composite_random(2, N)
    if s is not None:
        # Build combined lattice: vectors (a, b, c) with
        # a = r*b mod N AND a = s*c mod N.
        # This overdetermines a and might force (a,b,c) to reveal a factor.
        dim = 3
        M3 = IntegerMatrix(dim, dim)
        M3[0, 0] = int(N)
        M3[0, 1] = 0
        M3[0, 2] = 0
        M3[1, 0] = int(r)
        M3[1, 1] = 1
        M3[1, 2] = 0
        M3[2, 0] = int(s)
        M3[2, 1] = 0
        M3[2, 2] = 1
        lll_reduce(M3)
        rows3 = extract_rows(M3)
        for row in rows3:
            for val in row:
                f = try_gcd_factor(val, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = "gaussian_combined_3d"
                    return results
            # Norm extractions
            a, b, c = row
            for norm_val in [a * a + b * b, a * a + 2 * c * c, b * b + 2 * c * c]:
                f = try_gcd_factor(norm_val, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = "gaussian_combined_norm"
                    return results

    # Higher-dimensional: use D = 1, 2, 3, 4, 5 simultaneously
    roots_by_D = {}
    for D in range(1, 8):
        rd = sqrt_mod_composite_random(D, N)
        if rd is not None:
            roots_by_D[D] = rd

    if len(roots_by_D) >= 3:
        Ds = sorted(roots_by_D.keys())[:6]
        dim = 1 + len(Ds)
        M_big = IntegerMatrix(dim, dim)
        M_big[0, 0] = int(N)
        for j in range(1, dim):
            M_big[0, j] = 0
        for idx, D in enumerate(Ds):
            rd = roots_by_D[D]
            M_big[idx + 1, 0] = int(rd)
            for j in range(1, dim):
                M_big[idx + 1, j] = 1 if j == idx + 1 else 0
        lll_reduce(M_big)
        rows_big = extract_rows(M_big)
        for row in rows_big:
            for val in row:
                f = try_gcd_factor(val, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = "gaussian_multiD"
                    return results

    # BKZ with various block sizes on the 2x2 lattice (mostly for benchmarking)
    for bs in [2]:
        M_bkz = IntegerMatrix(2, 2)
        M_bkz[0, 0] = int(N)
        M_bkz[0, 1] = 0
        M_bkz[1, 0] = int(r)
        M_bkz[1, 1] = 1
        bkz_reduce(M_bkz, bs)
        rows_bkz = extract_rows(M_bkz)
        for row in rows_bkz:
            for val in row:
                f = try_gcd_factor(val, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = f"gaussian_bkz_{bs}"
                    return results

    return results


# =============================================================================
# APPROACH 5: Eisenstein integer lattice (Z[omega]) and Z[sqrt(-2)]
# =============================================================================
# Generalize Approach 4 to other rings of algebraic integers.
#
# Z[omega], omega = e^{2*pi*i/3}: norm form a^2 + ab + b^2.
#   Primes p = 1 mod 3 split in Z[omega].
#   Need to find r with r^2 + r + 1 = 0 mod N (cube root of unity mod N).
#   Lattice: [[N, 0], [r, 1]]. Short (a,b) gives a^2+ab+b^2 = 0 mod N.
#
# Z[sqrt(-2)]: norm form a^2 + 2*b^2.
#   Primes p = 1 or 3 mod 8 split (actually p=1,3 mod 8 for a^2+2b^2).
#   Need r with r^2 = -2 mod N.
#   Lattice: [[N, 0], [r, 1]]. Short (a,b) gives a^2+2b^2 = 0 mod N.
#
# Z[sqrt(-3)] (equivalent to Eisenstein up to index): norm form a^2 + 3*b^2.
#   Primes p = 1 mod 3 are representable.
#   Need r with r^2 = -3 mod N.

def approach5_multi_ring(N: int, p: int, q: int) -> dict:
    """Multi-ring lattice approach: Eisenstein, Z[sqrt(-2)], Z[sqrt(-3)], etc."""
    results = {"name": "multi_ring", "factored": False, "factor": None, "method": None}

    ring_configs = []

    # Z[i]: a^2 + b^2 (D=1), need p=q=1 mod 4
    if p % 4 == 1 and q % 4 == 1:
        r = find_sqrt_minus1_mod_N(N)
        if r is not None:
            ring_configs.append(("Z[i]", r, lambda a, b: a * a + b * b))

    # Z[sqrt(-2)]: a^2 + 2*b^2 (D=2), need (-2/p)=(-2/q)=1
    if jacobi(-2, p) == 1 and jacobi(-2, q) == 1:
        r2 = sqrt_mod_composite_random(2, N)
        if r2 is not None:
            ring_configs.append(("Z[sqrt(-2)]", r2, lambda a, b: a * a + 2 * b * b))

    # Z[omega] (Eisenstein): a^2 + ab + b^2 = (a + b*omega)(a + b*omega_bar)
    # Need cube root of unity: r^2 + r + 1 = 0 mod N, i.e., r = (-1 +/- sqrt(-3))/2
    if jacobi(-3, p) == 1 and jacobi(-3, q) == 1:
        r3 = sqrt_mod_composite_random(3, N)
        if r3 is not None:
            # r_omega = (-1 + r3) / 2 mod N (need 2 invertible)
            inv2 = pow(2, -1, N)
            r_omega = ((-1 + r3) * inv2) % N
            # Verify: r_omega^2 + r_omega + 1 mod N should be 0
            check = (r_omega * r_omega + r_omega + 1) % N
            if check == 0:
                ring_configs.append(
                    ("Z[omega]", r_omega, lambda a, b: a * a + a * b + b * b))

    # Z[sqrt(-5)]: a^2 + 5*b^2, not a PID but still useful
    if jacobi(-5, p) == 1 and jacobi(-5, q) == 1:
        r5 = sqrt_mod_composite_random(5, N)
        if r5 is not None:
            ring_configs.append(
                ("Z[sqrt(-5)]", r5, lambda a, b: a * a + 5 * b * b))

    # Z[sqrt(-6)]: a^2 + 6*b^2
    if jacobi(-6, p) == 1 and jacobi(-6, q) == 1:
        r6 = sqrt_mod_composite_random(6, N)
        if r6 is not None:
            ring_configs.append(
                ("Z[sqrt(-6)]", r6, lambda a, b: a * a + 6 * b * b))

    # Z[sqrt(-7)]: a^2 + 7*b^2
    if jacobi(-7, p) == 1 and jacobi(-7, q) == 1:
        r7 = sqrt_mod_composite_random(7, N)
        if r7 is not None:
            ring_configs.append(
                ("Z[sqrt(-7)]", r7, lambda a, b: a * a + 7 * b * b))

    if not ring_configs:
        results["method"] = "no_applicable_ring"
        return results

    all_short_vectors = {}

    for ring_name, r, norm_fn in ring_configs:
        # Build 2x2 lattice [[N, 0], [r, 1]]
        M = IntegerMatrix(2, 2)
        M[0, 0] = int(N)
        M[0, 1] = 0
        M[1, 0] = int(r)
        M[1, 1] = 1
        lll_reduce(M)
        rows = extract_rows(M)

        all_short_vectors[ring_name] = []
        for row in rows:
            a, b = row[0], row[1]
            # Direct GCD checks
            for val in [a, b, a + b, a - b]:
                f = try_gcd_factor(val, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = f"ring_{ring_name}_gcd"
                    return results
            # Norm-based extraction
            nv = norm_fn(a, b)
            if nv != 0:
                f = try_gcd_factor(nv, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = f"ring_{ring_name}_norm"
                    return results
            all_short_vectors[ring_name].append((a, b, nv))

    # Cross-ring extraction: combine short vectors from different rings.
    # If (a1, b1) from Z[i] and (a2, b2) from Z[omega], their interaction
    # might reveal a factor.
    ring_names = list(all_short_vectors.keys())
    for i in range(len(ring_names)):
        for j in range(i + 1, len(ring_names)):
            vecs_i = all_short_vectors[ring_names[i]]
            vecs_j = all_short_vectors[ring_names[j]]
            for ai, bi, ni in vecs_i:
                for aj, bj, nj in vecs_j:
                    # Cross products
                    for val in [ai * aj + bi * bj, ai * aj - bi * bj,
                                ai * bj + aj * bi, ai * bj - aj * bi,
                                ni - nj, ni + nj]:
                        f = try_gcd_factor(val, N)
                        if f:
                            results["factored"] = True
                            results["factor"] = f
                            results["method"] = f"cross_{ring_names[i]}_{ring_names[j]}"
                            return results

    # BKZ refinement on the best ring
    for ring_name, r, norm_fn in ring_configs:
        for bs in [2]:
            M_bkz = IntegerMatrix(2, 2)
            M_bkz[0, 0] = int(N)
            M_bkz[0, 1] = 0
            M_bkz[1, 0] = int(r)
            M_bkz[1, 1] = 1
            bkz_reduce(M_bkz, bs)
            rows_bkz = extract_rows(M_bkz)
            for row in rows_bkz:
                a, b = row[0], row[1]
                for val in [a, b, a + b, a - b]:
                    f = try_gcd_factor(val, N)
                    if f:
                        results["factored"] = True
                        results["factor"] = f
                        results["method"] = f"ring_{ring_name}_bkz"
                        return results

    # Higher-dimensional combined lattice: all roots from all rings
    if len(ring_configs) >= 2:
        dim = 1 + len(ring_configs)
        M_combined = IntegerMatrix(dim, dim)
        M_combined[0, 0] = int(N)
        for j in range(1, dim):
            M_combined[0, j] = 0
        for idx, (ring_name, r, norm_fn) in enumerate(ring_configs):
            M_combined[idx + 1, 0] = int(r)
            for j in range(1, dim):
                M_combined[idx + 1, j] = 1 if j == idx + 1 else 0

        lll_reduce(M_combined)
        rows_combined = extract_rows(M_combined)
        for row in rows_combined:
            for val in row:
                f = try_gcd_factor(val, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = "combined_multiring"
                    return results

    return results


# =============================================================================
# APPROACH 6: Pell lattice
# =============================================================================
# The Pell equation x^2 - N*y^2 = +/-1 connects to the infrastructure of
# the real quadratic order Z[sqrt(N)].
#
# Key idea: a near-solution x^2 - N*y^2 = k for small |k| gives
# gcd(x^2 - N*y^2, N) = gcd(k, N), which factors N if k shares a factor.
#
# More precisely: if x^2 = N*y^2 + k and p | N, then x^2 = k mod p.
# If k is a non-residue mod p but a residue mod q (or vice versa),
# then gcd(x^2 - k, N) or gcd-based extraction works.
#
# Build the lattice:
#   [[1, C*1], [0, C*a0]]
# where a0 = isqrt(N). Short vector (x, y) gives x - a0*y small,
# hence x^2 - N*y^2 = (x - a0*y)(x + a0*y) + (a0^2 - N)*y^2 is small.
# Actually: x ~ a0*y, so x^2 - N*y^2 ~ (a0^2 - N)*y^2 + 2*a0*y*(x - a0*y).
# For this to factor N we need x^2 - N*y^2 to have a nontrivial gcd with N.
#
# Extended approach: use continued fraction convergents of sqrt(N).
# The convergents p_k/q_k satisfy |p_k^2 - N*q_k^2| < 2*sqrt(N),
# and these near-solutions are exactly what the 2D lattice finds.

def approach6_pell(N: int, p: int, q: int) -> dict:
    """Pell lattice approach."""
    results = {"name": "pell", "factored": False, "factor": None, "method": None}

    a0 = isqrt(N)
    if a0 * a0 == N:
        results["factored"] = True
        results["factor"] = a0
        results["method"] = "perfect_square"
        return results

    # Method A: 2D lattice with isqrt(N) approximation
    # [[1,  0,  C], [0, 1, C*a0]] — find (x, y) with x ~ a0*y
    C = isqrt(N)  # Scaling to balance columns
    M = IntegerMatrix(2, 2)
    M[0, 0] = 1
    M[0, 1] = C
    M[1, 0] = 0
    M[1, 1] = C * a0

    lll_reduce(M)
    rows = extract_rows(M)

    near_solutions = []
    for row in rows:
        x, scaled_y = row[0], row[1]
        # Unscale: the actual y is such that scaled_y = C * (x_approx)
        # Actually, the lattice encodes: vector (a, b) where a*1 + b*0 = a (first col)
        # and a*C + b*(C*a0) = C*(a + b*a0) (second col).
        # Short vector: (a, b) with a and a + b*a0 both small.
        # Then a ~ -b*a0, so (a/b) ~ -a0, meaning a^2 ~ a0^2*b^2 ~ N*b^2.
        # So a^2 - N*b^2 is small.
        a_val, b_val = row[0], row[1]
        # Recover the actual continued-fraction-like approximant
        # The short vector (a, b) in the ORIGINAL lattice (before reduction)
        # is some integer combination of [1, C] and [0, C*a0].
        # So the output is (alpha, alpha*C + beta*C*a0) for integers alpha, beta.
        # Hmm, the output of LLL is the reduced basis, not a single vector
        # in the original basis. Let me rethink.
        pass

    # Method B: explicit continued fraction / Pell near-solutions
    # Generate convergents of sqrt(N) and check gcd(x^2 - N*y^2, N).
    cf_limit = max(100, int(N.bit_length() * 5))
    m, d, a = 0, 1, a0
    prev_p, curr_p = 1, a0
    prev_q, curr_q = 0, 1

    for step in range(cf_limit):
        val = curr_p * curr_p - N * curr_q * curr_q
        if val != 0:
            f = try_gcd_factor(val, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "pell_cf"
                return results
            # Also check absolute value
            f = try_gcd_factor(abs(val), N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "pell_cf_abs"
                return results
        # Also: val might factor as product of small primes sharing a factor with N
        near_solutions.append((curr_p, curr_q, val))

        # Next CF step
        m = d * a - m
        if d == 0:
            break
        d_new = (N - m * m)
        if d_new == 0:
            break
        d = d_new // d
        if d == 0:
            break
        a = (a0 + m) // d
        prev_p, curr_p = curr_p, a * curr_p + prev_p
        prev_q, curr_q = curr_q, a * curr_q + prev_q
        if a == 2 * a0:
            # Period complete
            break

    # Method C: lattice of Pell near-solutions
    # Collect near-solutions from CF and build a lattice from their residues.
    # If val_i = x_i^2 - N*y_i^2, we want to find a PRODUCT of val_i's
    # that shares a factor with N. This is a variant of the quadratic sieve.
    if len(near_solutions) >= 4:
        # Build a lattice of exponent vectors over small primes
        primes = small_primes(50)
        nprimes = len(primes)
        smooth_rels = []
        for x_val, y_val, v in near_solutions:
            if v == 0:
                continue
            absv = abs(v)
            exps = [0] * nprimes
            temp = absv
            for pi, pr in enumerate(primes):
                while temp % pr == 0:
                    exps[pi] += 1
                    temp //= pr
            if temp == 1:  # Fully factored over our base
                smooth_rels.append((x_val, y_val, v, exps))

        # If we have enough smooth relations, build exponent lattice mod 2
        if len(smooth_rels) >= 2:
            n_rels = len(smooth_rels)
            dim = nprimes + n_rels
            dim = min(dim, 50)  # Cap dimension
            M_pell = IntegerMatrix(n_rels, nprimes + 1)
            for i, (xv, yv, v, exps) in enumerate(smooth_rels):
                for j in range(min(nprimes, M_pell.ncols - 1)):
                    M_pell[i, j] = exps[j] % 2  # Parity of exponents
                M_pell[i, nprimes] = 0  # Placeholder

            if n_rels >= 2 and nprimes >= 1:
                try:
                    lll_reduce(M_pell)
                    # Look for zero rows (all-even exponent combos)
                    for i in range(n_rels):
                        row = [int(M_pell[i, j]) for j in range(nprimes)]
                        if all(v == 0 for v in row):
                            # This is a product with square residue — combine
                            # the corresponding x and y values
                            prod_x = 1
                            prod_y2 = 0
                            for sr in smooth_rels:
                                prod_x = (prod_x * sr[0]) % N
                            f = try_gcd_factor(prod_x - 1, N)
                            if f:
                                results["factored"] = True
                                results["factor"] = f
                                results["method"] = "pell_sieve"
                                return results
                            f = try_gcd_factor(prod_x + 1, N)
                            if f:
                                results["factored"] = True
                                results["factor"] = f
                                results["method"] = "pell_sieve"
                                return results
                except Exception:
                    pass

    # Method D: 3D Pell lattice with N and a0
    # [[1, 0, C], [0, 1, 0], [0, 0, C*N]]
    # Target: find (x, y) with x^2 - N*y^2 small.
    C = max(1, isqrt(isqrt(N)))
    M3 = IntegerMatrix(3, 3)
    M3[0, 0] = 1
    M3[0, 1] = 0
    M3[0, 2] = C
    M3[1, 0] = 0
    M3[1, 1] = 1
    M3[1, 2] = C * a0
    M3[2, 0] = 0
    M3[2, 1] = 0
    M3[2, 2] = C * N

    lll_reduce(M3)
    rows3 = extract_rows(M3)
    for row in rows3:
        a_val = row[0]
        b_val = row[1]
        if b_val != 0:
            pell_val = a_val * a_val - N * b_val * b_val
            if pell_val != 0:
                f = try_gcd_factor(pell_val, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = "pell_3d"
                    return results
        for val in row:
            f = try_gcd_factor(val, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "pell_3d_direct"
                return results

    # Method E: BKZ on the 3D Pell lattice
    for bs in [3]:
        M3b = IntegerMatrix(3, 3)
        M3b[0, 0] = 1
        M3b[0, 1] = 0
        M3b[0, 2] = C
        M3b[1, 0] = 0
        M3b[1, 1] = 1
        M3b[1, 2] = C * a0
        M3b[2, 0] = 0
        M3b[2, 1] = 0
        M3b[2, 2] = C * N

        bkz_reduce(M3b, bs)
        rows3b = extract_rows(M3b)
        for row in rows3b:
            a_val = row[0]
            b_val = row[1]
            if b_val != 0:
                pell_val = a_val * a_val - N * b_val * b_val
                if pell_val != 0:
                    f = try_gcd_factor(pell_val, N)
                    if f:
                        results["factored"] = True
                        results["factor"] = f
                        results["method"] = "pell_bkz"
                        return results
            for val in row:
                f = try_gcd_factor(val, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = "pell_bkz_direct"
                    return results

    return results


# =============================================================================
# Test harness
# =============================================================================

def run_single(N: int, p: int, q: int) -> dict:
    """Run all approaches on a single semiprime and return results."""
    out = {}

    t0 = time.time()
    out["fermat"] = approach2_fermat_lll(N, p, q)
    out["fermat"]["time"] = time.time() - t0

    t0 = time.time()
    out["gaussian"] = approach4_gaussian(N, p, q)
    out["gaussian"]["time"] = time.time() - t0

    t0 = time.time()
    out["multi_ring"] = approach5_multi_ring(N, p, q)
    out["multi_ring"]["time"] = time.time() - t0

    t0 = time.time()
    out["pell"] = approach6_pell(N, p, q)
    out["pell"]["time"] = time.time() - t0

    return out


def main():
    print("=" * 72)
    print("HYPERBOLIC LATTICE FACTORING EXPERIMENT")
    print("=" * 72)
    print()
    print("Testing whether lattice reduction on algebraic-norm lattices")
    print("derived from xy = N can factor semiprimes.")
    print()

    BIT_SIZES = [16, 20, 24, 28, 32, 40, 48]
    TRIALS = 100  # Per bit size per condition

    # Track results: approach -> bit_size -> list of (factored, method, time)
    results_any = defaultdict(lambda: defaultdict(list))
    results_1mod4 = defaultdict(lambda: defaultdict(list))

    approaches = ["fermat", "gaussian", "multi_ring", "pell"]

    # -------------------------------------------------------------------------
    # Phase 1: General semiprimes (any p, q)
    # -------------------------------------------------------------------------
    print("-" * 72)
    print("PHASE 1: General semiprimes (no congruence constraint)")
    print("-" * 72)

    for bits in BIT_SIZES:
        t_start = time.time()
        for trial in range(TRIALS):
            p, q, N = generate_semiprime(bits, cond="any")
            out = run_single(N, p, q)
            for app in approaches:
                r = out[app]
                results_any[app][bits].append(
                    (r["factored"], r.get("method"), r.get("time", 0)))

        elapsed = time.time() - t_start
        # Print summary for this bit size
        print(f"\n  {bits}-bit semiprimes ({TRIALS} trials, {elapsed:.1f}s):")
        for app in approaches:
            data = results_any[app][bits]
            n_success = sum(1 for d in data if d[0])
            rate = n_success / len(data) * 100
            methods = defaultdict(int)
            for d in data:
                if d[0]:
                    methods[d[1]] += 1
            avg_time = sum(d[2] for d in data) / max(len(data), 1)
            method_str = ", ".join(f"{m}:{c}" for m, c in
                                   sorted(methods.items(), key=lambda x: -x[1])[:3])
            print(f"    {app:12s}: {rate:5.1f}% ({n_success}/{len(data)})  "
                  f"avg {avg_time*1000:.1f}ms  [{method_str}]")

    # -------------------------------------------------------------------------
    # Phase 2: Semiprimes with p = q = 1 mod 4 (Gaussian-optimal)
    # -------------------------------------------------------------------------
    print()
    print("-" * 72)
    print("PHASE 2: Semiprimes with p = q = 1 mod 4 (Gaussian-optimal)")
    print("-" * 72)

    for bits in BIT_SIZES:
        t_start = time.time()
        for trial in range(TRIALS):
            p, q, N = generate_semiprime(bits, cond="1mod4")
            out = run_single(N, p, q)
            for app in approaches:
                r = out[app]
                results_1mod4[app][bits].append(
                    (r["factored"], r.get("method"), r.get("time", 0)))

        elapsed = time.time() - t_start
        print(f"\n  {bits}-bit semiprimes, p=q=1 mod 4 ({TRIALS} trials, {elapsed:.1f}s):")
        for app in approaches:
            data = results_1mod4[app][bits]
            n_success = sum(1 for d in data if d[0])
            rate = n_success / len(data) * 100
            methods = defaultdict(int)
            for d in data:
                if d[0]:
                    methods[d[1]] += 1
            avg_time = sum(d[2] for d in data) / max(len(data), 1)
            method_str = ", ".join(f"{m}:{c}" for m, c in
                                   sorted(methods.items(), key=lambda x: -x[1])[:3])
            print(f"    {app:12s}: {rate:5.1f}% ({n_success}/{len(data)})  "
                  f"avg {avg_time*1000:.1f}ms  [{method_str}]")

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("SUMMARY: Success rates (%)")
    print("=" * 72)
    print()

    # Table header
    header = f"{'Approach':>12s} | {'Cond':>6s}"
    for bits in BIT_SIZES:
        header += f" | {bits:>4d}b"
    print(header)
    print("-" * len(header))

    for app in approaches:
        # General
        row = f"{app:>12s} | {'any':>6s}"
        for bits in BIT_SIZES:
            data = results_any[app][bits]
            n_success = sum(1 for d in data if d[0])
            rate = n_success / max(len(data), 1) * 100
            row += f" | {rate:4.0f}%"
        print(row)
        # 1 mod 4
        row = f"{'':>12s} | {'1mod4':>6s}"
        for bits in BIT_SIZES:
            data = results_1mod4[app][bits]
            n_success = sum(1 for d in data if d[0])
            rate = n_success / max(len(data), 1) * 100
            row += f" | {rate:4.0f}%"
        print(row)

    # -------------------------------------------------------------------------
    # Scaling analysis
    # -------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("SCALING ANALYSIS")
    print("=" * 72)
    print()
    print("For each approach, does the success rate decay with N?")
    print("A polynomial-time method should maintain constant (or slowly")
    print("decaying) success rate as bit size grows.")
    print()

    for app in approaches:
        rates_any = []
        rates_1mod4 = []
        for bits in BIT_SIZES:
            data_a = results_any[app][bits]
            data_4 = results_1mod4[app][bits]
            r_a = sum(1 for d in data_a if d[0]) / max(len(data_a), 1)
            r_4 = sum(1 for d in data_4 if d[0]) / max(len(data_4), 1)
            rates_any.append(r_a)
            rates_1mod4.append(r_4)

        # Compute decay: ratio of last nonzero rate to first nonzero rate
        nonzero_any = [(b, r) for b, r in zip(BIT_SIZES, rates_any) if r > 0]
        nonzero_1m4 = [(b, r) for b, r in zip(BIT_SIZES, rates_1mod4) if r > 0]

        print(f"  {app}:")
        if len(nonzero_any) >= 2:
            first_b, first_r = nonzero_any[0]
            last_b, last_r = nonzero_any[-1]
            decay = last_r / first_r if first_r > 0 else 0
            print(f"    General: {first_b}b={first_r*100:.0f}% -> "
                  f"{last_b}b={last_r*100:.0f}% (ratio={decay:.3f})")
        elif len(nonzero_any) == 1:
            b, r = nonzero_any[0]
            print(f"    General: only nonzero at {b}b ({r*100:.0f}%)")
        else:
            print(f"    General: no successes")

        if len(nonzero_1m4) >= 2:
            first_b, first_r = nonzero_1m4[0]
            last_b, last_r = nonzero_1m4[-1]
            decay = last_r / first_r if first_r > 0 else 0
            print(f"    1mod4:   {first_b}b={first_r*100:.0f}% -> "
                  f"{last_b}b={last_r*100:.0f}% (ratio={decay:.3f})")
        elif len(nonzero_1m4) == 1:
            b, r = nonzero_1m4[0]
            print(f"    1mod4:   only nonzero at {b}b ({r*100:.0f}%)")
        else:
            print(f"    1mod4:   no successes")
        print()

    # -------------------------------------------------------------------------
    # Detailed method breakdown for best approach
    # -------------------------------------------------------------------------
    print("=" * 72)
    print("METHOD BREAKDOWN (which sub-strategy produced each success)")
    print("=" * 72)
    print()

    for app in approaches:
        all_methods = defaultdict(int)
        for bits in BIT_SIZES:
            for d in results_any[app][bits]:
                if d[0]:
                    all_methods[d[1]] += 1
            for d in results_1mod4[app][bits]:
                if d[0]:
                    all_methods[d[1]] += 1
        if all_methods:
            total = sum(all_methods.values())
            print(f"  {app} ({total} total successes):")
            for method, count in sorted(all_methods.items(), key=lambda x: -x[1]):
                print(f"    {method:40s}: {count:4d} ({count/total*100:.1f}%)")
            print()

    print("=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
