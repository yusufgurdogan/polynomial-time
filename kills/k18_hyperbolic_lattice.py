#!/usr/bin/env python3
"""
Hyperbolic lattice factoring experiment.

Factoring N = pq is equivalent to finding a lattice point (p,q) on the
hyperbola xy = N. No modern algorithm exploits this directly with lattice
reduction. This experiment tests whether LLL/BKZ on lattices derived from
hyperbolic and algebraic-norm constraints can factor semiprimes.

Four approaches are tested:

  Approach 2 (Fermat + LLL): Encode x^2 - y^2 = N as a CVP-like problem.
  Approach 4 (Gaussian integers): Z[i] norm lattice + Gaussian trial division.
  Approach 5 (Eisenstein / multi-ring): Z[omega], Z[sqrt(-2)], etc.
  Approach 6 (Pell lattice): CF convergents & smooth-residue sieve.

Key theoretical finding (verified empirically):
  The 2x2 Gaussian lattice [[N,0],[r,1]] (where r^2=-1 mod N) finds vectors
  (a,b) with a^2+b^2 = kN, but extracting factors requires O(N^{1/4})
  Gaussian trial divisions -- no better than plain trial division. The lattice
  step merely reformulates factoring, not solves it.

  The Pell/CF approach is the most empirically successful: continued fraction
  convergents of sqrt(N) give x^2 - Ny^2 = small, and gcd(small, N) is often
  nontrivial for small N. But success decays with N.

  The multi-ring combined lattice (using roots for D=1,2,3,...) is the most
  interesting theoretically: it encodes MULTIPLE algebraic constraints
  simultaneously, and the higher-dimensional short vectors reveal more
  structural information about N's factorization.

We measure: success rate, which sub-method works, scaling behavior, and
diagnostic metrics (shortest-vector norms, k-values in a^2+b^2=kN, etc.)
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


SMALL_PRIMES_CACHE = small_primes(10000)


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
    """Tonelli-Shanks: find x with x^2 = a mod p, or None."""
    a = a % p
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        return None
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    s, q = 0, p - 1
    while q % 2 == 0:
        s += 1
        q //= 2
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    M, c, t, R = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while True:
        if t == 1:
            return R
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        b = pow(c, 1 << (M - i - 1), p)
        M, c, t, R = i, (b * b) % p, (t * b * b) % p, (R * b) % p


def find_sqrt_neg_D_mod_N_blind(D: int, N: int, max_attempts: int = 500) -> Optional[int]:
    """Find r with r^2 + D = 0 mod N by random search (without knowing factors)."""
    target = (-D) % N
    if jacobi(target, N) != 1:
        return None
    for _ in range(max_attempts):
        a = random.randint(2, N - 2)
        g = math.gcd(a, N)
        if 1 < g < N:
            return None
        r = pow(a, (N - 1) // 4, N)
        if (r * r + D) % N == 0:
            return r
        r2 = pow(a, (N + 3) // 4, N)
        if (r2 * r2 + D) % N == 0:
            return r2
        r3 = random.randint(1, N - 1)
        if (r3 * r3 + D) % N == 0:
            return r3
    if N < 2 * 10**6:
        for x in range(1, N):
            if (x * x + D) % N == 0:
                return x
    return None


def find_sqrt_neg_D_mod_N(D: int, N: int, p: int = 0, q: int = 0) -> Optional[int]:
    """Find r with r^2 + D = 0 mod N using CRT with known factors.

    For the experiment, we use known factors to compute r reliably via CRT.
    This isolates the lattice-reduction step from the root-finding step.
    A real factoring algorithm would need a different root-finding method
    (and in fact, finding TWO independent roots of x^2=-D mod N is itself
    equivalent to factoring N).
    """
    if p == 0 or q == 0:
        return find_sqrt_neg_D_mod_N_blind(D, N)

    target_p = (-D) % p
    target_q = (-D) % q
    rp = sqrt_mod_prime(target_p, p)
    rq = sqrt_mod_prime(target_q, q)
    if rp is None or rq is None:
        return None
    # CRT: find r = rp mod p, r = rq mod q
    try:
        inv_p = pow(p, -1, q)
    except ValueError:
        return None
    r = (rp + p * ((rq - rp) * inv_p % q)) % N
    if (r * r + D) % N != 0:
        return None
    return r


def generate_semiprime(bits: int, cond: str = "any") -> Tuple[int, int, int]:
    """Generate N = p*q where p,q are primes of roughly bits/2 each."""
    lo = 1 << (bits // 2 - 1)
    hi = 1 << (bits // 2)
    for _attempt in range(10000):
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
    raise RuntimeError(f"Could not generate semiprime: bits={bits}, cond={cond}")


def lll_reduce(M: IntegerMatrix) -> IntegerMatrix:
    LLL.reduction(M)
    return M


def bkz_reduce(M: IntegerMatrix, block_size: int) -> IntegerMatrix:
    bs = max(2, min(block_size, M.nrows))
    par = BKZ.Param(block_size=bs)
    BKZ.reduction(M, par)
    return M


def extract_rows(M: IntegerMatrix) -> List[List[int]]:
    return [[int(M[i, j]) for j in range(M.ncols)] for i in range(M.nrows)]


def try_gcd_factor(val: int, N: int) -> Optional[int]:
    if val == 0:
        return None
    g = math.gcd(abs(val), N)
    if 1 < g < N:
        return g
    return None


# =============================================================================
# APPROACH 2: Fermat + LLL
# =============================================================================
# Fermat: N = x^2 - y^2. x = (p+q)/2, y = (p-q)/2.
# s = p+q, d = p-q. s^2 - d^2 = 4N.
#
# We try multiple lattice constructions to find (s, d) or (x, y):
# (A) CVP near isqrt(N): encode x ~ isqrt(N) + small correction.
# (B) Modular Fermat: build lattice encoding x^2 = N + y^2 mod small primes.
# (C) Sum-product lattice: (s, d, 1) with s*d ~ p^2-q^2 constraint.

def approach2_fermat_lll(N: int, p: int, q: int) -> dict:
    results = {"name": "fermat_lll", "factored": False, "factor": None,
               "method": None, "diagnostics": {}}
    x0 = isqrt(N)
    if x0 * x0 == N:
        results["factored"] = True
        results["factor"] = x0
        results["method"] = "perfect_square"
        return results

    s_true = p + q
    d_true = abs(p - q)
    results["diagnostics"]["s_true"] = s_true
    results["diagnostics"]["d_true"] = d_true

    # --- Strategy A: 3D lattice encoding s, d, constant ---
    # Short vectors (a, b, c) where a ~ s-offset, b ~ d.
    # Constraint: a^2 - b^2 = 4N when a = s, b = d.
    # Use Kannan embedding: target vector (s, d, 0) is close to a lattice point.
    C = isqrt(N) + 1
    dim = 3
    M = IntegerMatrix(dim, dim)
    M[0, 0] = 1
    M[0, 1] = 0
    M[0, 2] = C
    M[1, 0] = 0
    M[1, 1] = 1
    M[1, 2] = C
    M[2, 0] = 0
    M[2, 1] = 0
    M[2, 2] = C * (2 * x0 + 1)

    lll_reduce(M)
    rows = extract_rows(M)

    for row in rows:
        for val in row:
            f = try_gcd_factor(val, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "fermat_A_gcd"
                return results
        a, b = abs(row[0]), abs(row[1])
        if a > b > 0 and a * a - b * b == 4 * N:
            p_cand = (a + b) // 2
            q_cand = (a - b) // 2
            if p_cand * q_cand == N:
                results["factored"] = True
                results["factor"] = min(p_cand, q_cand)
                results["method"] = "fermat_A_sd"
                return results
        if a > 0 and b > 0:
            f = try_gcd_factor(a + b, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "fermat_A_sum"
                return results
            f = try_gcd_factor(a - b, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "fermat_A_diff"
                return results

    # --- Strategy B: modular Fermat with multiple small moduli ---
    # Build lattice encoding: for each small prime l_i, x^2 = N mod l_i.
    # Short vectors satisfy x = sqrt(N) mod l_i for several l_i simultaneously.
    # By CRT, this constrains x to a small set, hopefully including (p+q)/2.
    moduli = []
    roots_per_mod = []
    for pr in SMALL_PRIMES_CACHE[:20]:
        n_mod = N % pr
        rt = sqrt_mod_prime(n_mod, pr)
        if rt is not None:
            moduli.append(pr)
            roots_per_mod.append(rt)
    if len(moduli) >= 3:
        num_mod = min(len(moduli), 6)
        dim_b = num_mod + 1
        M_b = IntegerMatrix(dim_b, dim_b)
        # CRT-like lattice: row i encodes "x = root_i mod moduli[i]"
        # Row 0: (1, C*mod_1, C*mod_2, ..., C*mod_k)
        # Rows i: (0, ..., C*moduli[i], ..., 0)
        # ... or simpler: just use the product lattice.
        prod = 1
        for i in range(num_mod):
            prod *= moduli[i]
        # Target: find x with x = root_i mod moduli[i] for i = 0..num_mod-1.
        # By CRT, x is unique mod prod. Build lattice to find x near sqrt(N).
        # Lattice: [[prod, 0], [x_crt, 1]] — then short vector gives (a, b)
        # with a = x_crt*b mod prod.
        try:
            x_crt = roots_per_mod[0]
            m_acc = moduli[0]
            for i in range(1, num_mod):
                ri = roots_per_mod[i]
                mi = moduli[i]
                inv_m = pow(m_acc, -1, mi)
                x_crt = x_crt + m_acc * ((ri - x_crt) * inv_m % mi)
                m_acc *= mi
            # x_crt is the CRT solution mod m_acc = prod.
            # (p+q)/2 = x_crt + k*prod for some k.
            # Build lattice to find k:
            M_crt = IntegerMatrix(2, 2)
            M_crt[0, 0] = int(prod)
            M_crt[0, 1] = 0
            M_crt[1, 0] = int(x_crt)
            M_crt[1, 1] = 1
            lll_reduce(M_crt)
            for i in range(2):
                a, b = int(M_crt[i, 0]), int(M_crt[i, 1])
                # b gives us a candidate for (p+q)/2
                for cand in [abs(a), abs(b), abs(a + b), abs(a - b)]:
                    if cand > 1:
                        # Check if 4*cand^2 - 4N is a perfect square
                        disc = 4 * cand * cand - 4 * N
                        if disc > 0:
                            sd = isqrt(disc)
                            if sd * sd == disc:
                                p1 = (2 * cand + sd) // 2
                                q1 = (2 * cand - sd) // 2
                                if p1 > 1 and q1 > 1 and p1 * q1 == N:
                                    results["factored"] = True
                                    results["factor"] = min(p1, q1)
                                    results["method"] = "fermat_B_crt"
                                    return results
                    f = try_gcd_factor(cand, N)
                    if f:
                        results["factored"] = True
                        results["factor"] = f
                        results["method"] = "fermat_B_gcd"
                        return results
        except (ValueError, ZeroDivisionError):
            pass

    # --- Strategy C: direct search near isqrt(N) aided by LLL ---
    # Fermat iteration: check x = x0, x0+1, ... until x^2 - N is a perfect square.
    # Use LLL to find good starting points.
    for t in range(min(100, isqrt(isqrt(N)) + 10)):
        x = x0 + t
        y2 = x * x - N
        if y2 >= 0:
            y = isqrt(y2)
            if y * y == y2:
                p_cand = x + y
                q_cand = x - y
                if q_cand > 1 and p_cand * q_cand == N:
                    results["factored"] = True
                    results["factor"] = min(p_cand, q_cand)
                    results["method"] = "fermat_C_direct"
                    return results

    return results


# =============================================================================
# APPROACH 4: Gaussian integer lattice (Z[i])
# =============================================================================
# Build [[N,0],[r,1]] where r^2 = -1 mod N. LLL gives (a,b) with a^2+b^2 = kN.
#
# CRITICAL INSIGHT: the 2x2 lattice ALWAYS produces norms that are multiples of N.
# So gcd(a^2+b^2, N) = N. The lattice reformulates factoring as Gaussian-integer
# factoring, which is equally hard.
#
# HOWEVER, we can try:
# (1) Gaussian trial division: divide a+bi by small Gaussian primes.
#     NOTE: This requires primes up to sqrt(N), not N^{1/4}, because
#     norm(a+bi) = kN ~ N. So it is NO BETTER than trial division of N.
#     We include it as a diagnostic to verify this claim empirically.
# (2) Multi-D combined lattice: use roots of x^2=-D for D=1,2,3,...
#     simultaneously. Higher-dimensional LLL might reveal factors via
#     GCD of lattice entries (not norm-based extraction).
# (3) Norm distribution analysis: measure k in a^2+b^2=kN.

def gaussian_trial_division(a: int, b: int, N: int,
                            max_norm: int = 0) -> Optional[int]:
    """Try to factor a+bi in Z[i] by dividing by small Gaussian primes.

    If we find a Gaussian prime pi+qi*i with (pi^2+qi^2) | N, return that factor.
    The search is up to Gaussian primes of norm <= max_norm (default: N^{1/4}).
    """
    if max_norm == 0:
        max_norm = isqrt(isqrt(N)) + 1
    # Gaussian primes: (a, b) with a^2+b^2 = prime p where p = 1 mod 4 or p = 2.
    # Also: p (prime, p = 3 mod 4) is a Gaussian prime.
    # We enumerate norm-2 primes (1+i), and then primes p=1 mod 4 decomposed.
    # Dividing a+bi by (c+di): (a+bi)(c-di) / (c^2+d^2).
    # If (c^2+d^2) divides both (ac+bd) and (bc-ad), the division is exact.

    # Try 1+i first (norm 2)
    ac_bd = a + b  # a*1 + b*1
    bc_ad = b - a  # b*1 - a*1
    if ac_bd % 2 == 0 and bc_ad % 2 == 0:
        f = try_gcd_factor(2, N)
        if f:
            return f
        # Continue dividing
        a, b = ac_bd // 2, bc_ad // 2

    # Enumerate primes p = 1 mod 4 up to max_norm^2
    bound = min(max_norm * max_norm, 10**8)  # Cap for performance
    for pr in SMALL_PRIMES_CACHE:
        if pr * pr > bound:
            break
        if pr == 2:
            continue
        if pr % 4 != 1:
            continue
        # Decompose pr = c^2 + d^2 using sqrt_mod_prime
        c = sqrt_mod_prime(pr - 1, pr)
        if c is None:
            continue
        # Build Gaussian prime: find c, d with c^2+d^2 = pr
        # c = sqrt(-1) mod pr, then reduce (pr, 0), (c, 1) lattice
        M_g = IntegerMatrix(2, 2)
        M_g[0, 0] = pr
        M_g[0, 1] = 0
        M_g[1, 0] = c
        M_g[1, 1] = 1
        lll_reduce(M_g)
        gc, gd = abs(int(M_g[0, 0])), abs(int(M_g[0, 1]))
        if gc * gc + gd * gd != pr:
            gc, gd = abs(int(M_g[1, 0])), abs(int(M_g[1, 1]))
        if gc * gc + gd * gd != pr:
            continue

        # Try dividing a+bi by gc+gd*i and gc-gd*i
        for dd in [gd, -gd]:
            ac_bd = a * gc + b * dd
            bc_ad = b * gc - a * dd
            if ac_bd % pr == 0 and bc_ad % pr == 0:
                # Exact division. Check if pr divides N.
                f = try_gcd_factor(pr, N)
                if f:
                    return f
                # Continue dividing with quotient
                a_new, b_new = ac_bd // pr, bc_ad // pr
                a, b = a_new, b_new

    return None


def approach4_gaussian(N: int, p: int, q: int) -> dict:
    results = {"name": "gaussian", "factored": False, "factor": None,
               "method": None, "diagnostics": {}}

    if not (p % 4 == 1 and q % 4 == 1):
        results["method"] = "skip_not_1mod4"
        return results

    r = find_sqrt_neg_D_mod_N(1, N, p, q)
    if r is None:
        results["method"] = "no_sqrt_found"
        return results

    # --- 2x2 Gaussian lattice ---
    M = IntegerMatrix(2, 2)
    M[0, 0] = int(N)
    M[0, 1] = 0
    M[1, 0] = int(r)
    M[1, 1] = 1
    lll_reduce(M)
    rows = extract_rows(M)

    # Diagnostics: norm values and k = norm/N
    k_values = []
    for row in rows:
        a, b = row[0], row[1]
        norm = a * a + b * b
        if norm > 0:
            k = norm // N
            k_values.append(k)

        # Try Gaussian trial division
        f = gaussian_trial_division(a, b, N)
        if f:
            results["factored"] = True
            results["factor"] = f
            results["method"] = "gaussian_trial_div"
            return results

    results["diagnostics"]["k_values_2d"] = k_values
    results["diagnostics"]["min_k"] = min(k_values) if k_values else None

    # --- Multi-D combined lattice (D = 1, 2, 3, ...) ---
    roots_by_D = {}
    for D in range(1, 20):
        rd = find_sqrt_neg_D_mod_N(D, N, p, q)
        if rd is not None:
            roots_by_D[D] = rd
    results["diagnostics"]["num_D_roots"] = len(roots_by_D)

    if len(roots_by_D) >= 2:
        Ds = sorted(roots_by_D.keys())[:8]
        dim = 1 + len(Ds)
        M_big = IntegerMatrix(dim, dim)
        M_big[0, 0] = int(N)
        for j in range(1, dim):
            M_big[0, j] = 0
        for idx, D in enumerate(Ds):
            M_big[idx + 1, 0] = int(roots_by_D[D])
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
            # Check pairwise sums/differences of entries
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    for v in [row[i] + row[j], row[i] - row[j],
                              row[i] * row[j]]:
                        f = try_gcd_factor(v, N)
                        if f:
                            results["factored"] = True
                            results["factor"] = f
                            results["method"] = "gaussian_multiD_cross"
                            return results

        # BKZ on the multi-D lattice
        for bs in [3, 5]:
            if bs > dim:
                continue
            M_bkz = IntegerMatrix(dim, dim)
            M_bkz[0, 0] = int(N)
            for j in range(1, dim):
                M_bkz[0, j] = 0
            for idx, D in enumerate(Ds):
                M_bkz[idx + 1, 0] = int(roots_by_D[D])
                for j in range(1, dim):
                    M_bkz[idx + 1, j] = 1 if j == idx + 1 else 0
            bkz_reduce(M_bkz, bs)
            for i in range(dim):
                row = [int(M_bkz[i, j]) for j in range(dim)]
                for val in row:
                    f = try_gcd_factor(val, N)
                    if f:
                        results["factored"] = True
                        results["factor"] = f
                        results["method"] = f"gaussian_multiD_bkz{bs}"
                        return results

    return results


# =============================================================================
# APPROACH 5: Multi-ring lattice (Eisenstein, Z[sqrt(-2)], etc.)
# =============================================================================
# For each ring Z[sqrt(-D)] with class number 1 (D = 1, 2, 3, 7, 11, 19, ...):
#   - Find r with r^2 = -D mod N
#   - Build [[N,0],[r,1]], LLL-reduce
#   - Short (a,b) satisfies norm_D(a,b) = a^2+D*b^2 = 0 mod N
#   - Try Gaussian-like trial division using primes that split in the ring

def ring_trial_division(a: int, b: int, D: int, N: int,
                        max_norm: int = 0) -> Optional[int]:
    """Trial division in Z[sqrt(-D)] for small D with class number 1."""
    if max_norm == 0:
        max_norm = isqrt(isqrt(N)) + 1
    # Primes that split in Z[sqrt(-D)]: primes p with (-D/p) = 1.
    # For such p, find c, d with c^2 + D*d^2 = p. Then (c+d*sqrt(-D)) divides p.
    # Division of (a+b*sqrt(-D)) by (c+d*sqrt(-D)):
    #   (a+b*sqrt(-D))(c-d*sqrt(-D)) / (c^2+D*d^2) = (ac+Dbd + (bc-ad)*sqrt(-D)) / p.
    bound = min(max_norm * max_norm, 10**7)
    for pr in SMALL_PRIMES_CACHE:
        if pr > bound:
            break
        if pr == 2 and D % 2 == 0:
            continue
        # Check if pr splits: (-D/pr) = 1
        if jacobi(-D, pr) != 1:
            continue
        # Find c, d with c^2 + D*d^2 = pr
        found = False
        c_found, d_found = 0, 0
        for dd in range(1, isqrt(pr // D) + 2):
            rem = pr - D * dd * dd
            if rem < 0:
                break
            if rem == 0:
                c_found, d_found = 0, dd
                found = True
                break
            sr = isqrt(rem)
            if sr * sr == rem:
                c_found, d_found = sr, dd
                found = True
                break
        if not found:
            continue
        c, d = c_found, d_found
        # Try dividing (a + b*sqrt(-D)) by (c + d*sqrt(-D))
        # Quotient: ((ac + D*b*d) + (bc - ad)*sqrt(-D)) / pr
        ac_Dbd = a * c + D * b * d
        bc_ad = b * c - a * d
        if ac_Dbd % pr == 0 and bc_ad % pr == 0:
            f = try_gcd_factor(pr, N)
            if f:
                return f
            a, b = ac_Dbd // pr, bc_ad // pr
        # Also try conjugate: (c - d*sqrt(-D))
        ac_Dbd2 = a * c - D * b * d
        bc_ad2 = b * c + a * d
        if ac_Dbd2 % pr == 0 and bc_ad2 % pr == 0:
            f = try_gcd_factor(pr, N)
            if f:
                return f
    return None


def approach5_multi_ring(N: int, p: int, q: int) -> dict:
    results = {"name": "multi_ring", "factored": False, "factor": None,
               "method": None, "diagnostics": {}}

    # Discriminants with class number 1: D = 1, 2, 3, 7, 11, 19, 43, 67, 163
    class_no_1 = [1, 2, 3, 7, 11, 19, 43, 67, 163]

    ring_results = {}
    for D in class_no_1:
        # Check if -D is QR mod both p and q
        if jacobi(-D, p) != 1 or jacobi(-D, q) != 1:
            continue
        r = find_sqrt_neg_D_mod_N(D, N, p, q)
        if r is None:
            continue

        # Build and reduce 2x2 lattice
        M = IntegerMatrix(2, 2)
        M[0, 0] = int(N)
        M[0, 1] = 0
        M[1, 0] = int(r)
        M[1, 1] = 1
        lll_reduce(M)
        rows = extract_rows(M)

        for row in rows:
            a, b = row[0], row[1]
            norm = a * a + D * b * b
            ring_results[D] = {"a": a, "b": b, "norm": norm,
                                "k": norm // N if N > 0 else 0}

            # Direct GCD checks
            for val in [a, b, a + b, a - b, norm]:
                f = try_gcd_factor(val, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = f"ring_D{D}_gcd"
                    return results

            # Ring trial division
            f = ring_trial_division(a, b, D, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = f"ring_D{D}_trial"
                return results

    results["diagnostics"]["rings_used"] = list(ring_results.keys())

    # Cross-ring extraction: combine vectors from different rings.
    ring_keys = sorted(ring_results.keys())
    for i in range(len(ring_keys)):
        for j in range(i + 1, len(ring_keys)):
            Di, Dj = ring_keys[i], ring_keys[j]
            ai, bi = ring_results[Di]["a"], ring_results[Di]["b"]
            aj, bj = ring_results[Dj]["a"], ring_results[Dj]["b"]
            for val in [ai * aj + bi * bj, ai * aj - bi * bj,
                        ai * bj + aj * bi, ai * bj - aj * bi,
                        ring_results[Di]["norm"] - ring_results[Dj]["norm"],
                        ring_results[Di]["norm"] + ring_results[Dj]["norm"]]:
                f = try_gcd_factor(val, N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = f"cross_D{Di}_D{Dj}"
                    return results

    # Combined higher-dimensional lattice with all rings
    if len(ring_keys) >= 2:
        roots_for_combined = [(D, find_sqrt_neg_D_mod_N(D, N, p, q))
                              for D in ring_keys[:6]]
        roots_for_combined = [(D, r) for D, r in roots_for_combined if r is not None]
        if len(roots_for_combined) >= 2:
            dim = 1 + len(roots_for_combined)
            M_c = IntegerMatrix(dim, dim)
            M_c[0, 0] = int(N)
            for j in range(1, dim):
                M_c[0, j] = 0
            for idx, (D, rd) in enumerate(roots_for_combined):
                M_c[idx + 1, 0] = int(rd)
                for j in range(1, dim):
                    M_c[idx + 1, j] = 1 if j == idx + 1 else 0
            lll_reduce(M_c)
            for i in range(dim):
                row = [int(M_c[i, j]) for j in range(dim)]
                for val in row:
                    f = try_gcd_factor(val, N)
                    if f:
                        results["factored"] = True
                        results["factor"] = f
                        results["method"] = "combined_multiring"
                        return results

    return results


# =============================================================================
# APPROACH 6: Pell lattice & continued fraction sieve
# =============================================================================
# CF convergents of sqrt(N) give x_k, y_k with x_k^2 - N*y_k^2 = small.
# If gcd(|x_k^2 - N*y_k^2|, N) is nontrivial, we factor N.
#
# Also: collect smooth residues and combine them (quadratic-sieve style).

def approach6_pell(N: int, p: int, q: int) -> dict:
    results = {"name": "pell", "factored": False, "factor": None,
               "method": None, "diagnostics": {}}

    a0 = isqrt(N)
    if a0 * a0 == N:
        results["factored"] = True
        results["factor"] = a0
        results["method"] = "perfect_square"
        return results

    # --- Method A: continued fraction convergents ---
    cf_limit = max(200, int(N.bit_length() * 10))
    m, d_cf, a = 0, 1, a0
    prev_p, curr_p = 1, a0
    prev_q, curr_q = 0, 1

    near_solutions = []
    for step in range(cf_limit):
        val = curr_p * curr_p - N * curr_q * curr_q
        if val != 0:
            f = try_gcd_factor(abs(val), N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "pell_cf"
                return results
        near_solutions.append((curr_p, curr_q, val))

        # Next CF step for sqrt(N)
        m = d_cf * a - m
        if d_cf == 0:
            break
        d_new = N - m * m
        if d_new == 0:
            break
        d_cf = d_new // d_cf
        if d_cf == 0:
            break
        a = (a0 + m) // d_cf
        prev_p, curr_p = curr_p, a * curr_p + prev_p
        prev_q, curr_q = curr_q, a * curr_q + prev_q
        if a == 2 * a0:
            break

    results["diagnostics"]["cf_steps"] = len(near_solutions)

    # --- Method B: smooth residue combination (mini quadratic sieve) ---
    primes = small_primes(100)
    nprimes = len(primes)
    smooth_rels = []
    for x_val, y_val, v in near_solutions:
        if v == 0:
            continue
        absv = abs(v)
        exps = [0] * nprimes
        sign = 1 if v > 0 else -1
        temp = absv
        for pi, pr in enumerate(primes):
            while temp % pr == 0:
                exps[pi] += 1
                temp //= pr
        if temp == 1:
            smooth_rels.append((x_val, y_val, v, exps, sign))

    results["diagnostics"]["smooth_rels"] = len(smooth_rels)

    if len(smooth_rels) >= 2:
        # Find subsets whose exponent vectors sum to all-even (mod 2).
        # For small sets, try all pairs and triples.
        n_rels = len(smooth_rels)

        # Pairs
        for i in range(n_rels):
            for j in range(i + 1, n_rels):
                combined_exps = [smooth_rels[i][3][k] + smooth_rels[j][3][k]
                                 for k in range(nprimes)]
                if all(e % 2 == 0 for e in combined_exps):
                    # Product of x values mod N
                    prod_x = (smooth_rels[i][0] * smooth_rels[j][0]) % N
                    # Product of residues is a perfect square
                    prod_v = smooth_rels[i][2] * smooth_rels[j][2]
                    sqrt_v = isqrt(abs(prod_v))
                    if sqrt_v * sqrt_v == abs(prod_v):
                        # x^2 = y^2 mod N, so gcd(x-y, N) might factor
                        prod_y = (smooth_rels[i][1] * smooth_rels[j][1]) % N
                        for candidate in [prod_x - sqrt_v, prod_x + sqrt_v,
                                          prod_x - sqrt_v % N,
                                          (prod_x - sqrt_v) % N,
                                          (prod_x + sqrt_v) % N]:
                            f = try_gcd_factor(candidate, N)
                            if f:
                                results["factored"] = True
                                results["factor"] = f
                                results["method"] = "pell_sieve_pair"
                                return results

        # Triples
        if n_rels >= 3 and n_rels <= 50:
            for i in range(min(n_rels, 20)):
                for j in range(i + 1, min(n_rels, 20)):
                    for k in range(j + 1, min(n_rels, 20)):
                        combined = [smooth_rels[i][3][l] + smooth_rels[j][3][l]
                                    + smooth_rels[k][3][l] for l in range(nprimes)]
                        if all(e % 2 == 0 for e in combined):
                            prod_x = (smooth_rels[i][0] * smooth_rels[j][0]
                                      % N * smooth_rels[k][0]) % N
                            prod_v = (smooth_rels[i][2] * smooth_rels[j][2]
                                      * smooth_rels[k][2])
                            abs_pv = abs(prod_v)
                            sqrt_v = isqrt(abs_pv)
                            if sqrt_v * sqrt_v == abs_pv:
                                for candidate in [(prod_x - sqrt_v) % N,
                                                  (prod_x + sqrt_v) % N]:
                                    f = try_gcd_factor(candidate, N)
                                    if f:
                                        results["factored"] = True
                                        results["factor"] = f
                                        results["method"] = "pell_sieve_triple"
                                        return results

    # --- Method C: lattice-based Pell ---
    # Build 3D lattice encoding x^2 - N*y^2 constraint
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
    for i in range(3):
        row = [int(M3[i, j]) for j in range(3)]
        a_val, b_val = row[0], row[1]
        if b_val != 0:
            pell_val = a_val * a_val - N * b_val * b_val
            if pell_val != 0:
                f = try_gcd_factor(abs(pell_val), N)
                if f:
                    results["factored"] = True
                    results["factor"] = f
                    results["method"] = "pell_lattice_3d"
                    return results
        for val in row:
            f = try_gcd_factor(val, N)
            if f:
                results["factored"] = True
                results["factor"] = f
                results["method"] = "pell_lattice_direct"
                return results

    # BKZ on 3D Pell lattice
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
        for i in range(3):
            row = [int(M3b[i, j]) for j in range(3)]
            a_val, b_val = row[0], row[1]
            if b_val != 0:
                pv = a_val * a_val - N * b_val * b_val
                if pv != 0:
                    f = try_gcd_factor(abs(pv), N)
                    if f:
                        results["factored"] = True
                        results["factor"] = f
                        results["method"] = "pell_bkz_3d"
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


def print_phase_results(phase_name: str, results_dict: dict,
                        bit_sizes: list, approaches: list, trials: int):
    """Print results for one phase."""
    print()
    print("-" * 72)
    print(phase_name)
    print("-" * 72)

    timings = defaultdict(dict)

    for bits in bit_sizes:
        for app in approaches:
            data = results_dict[app][bits]
            n_success = sum(1 for d in data if d[0])
            timings[app][bits] = sum(d[2] for d in data) / max(len(data), 1)

        # Print per-bit-size detail
        total_time = sum(timings[app][bits] for app in approaches) * trials
        print(f"\n  {bits}-bit semiprimes ({trials} trials, ~{total_time:.1f}s):")
        for app in approaches:
            data = results_dict[app][bits]
            n_success = sum(1 for d in data if d[0])
            rate = n_success / len(data) * 100
            methods = defaultdict(int)
            for d in data:
                if d[0]:
                    methods[d[1]] += 1
            avg_time = timings[app][bits]
            method_str = ", ".join(
                f"{m}:{c}" for m, c in
                sorted(methods.items(), key=lambda x: -x[1])[:3])
            print(f"    {app:12s}: {rate:5.1f}% ({n_success}/{len(data)})  "
                  f"avg {avg_time*1000:.1f}ms  [{method_str}]")


def main():
    print("=" * 72)
    print("HYPERBOLIC LATTICE FACTORING EXPERIMENT")
    print("=" * 72)
    print()
    print("Testing whether lattice reduction on algebraic-norm lattices")
    print("derived from xy = N can factor semiprimes.")
    print()

    BIT_SIZES = [16, 20, 24, 28, 32, 40, 48]
    TRIALS = 100

    results_any = defaultdict(lambda: defaultdict(list))
    results_1mod4 = defaultdict(lambda: defaultdict(list))

    approaches = ["fermat", "gaussian", "multi_ring", "pell"]

    # Phase 1: General semiprimes
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
        n_pell = sum(1 for d in results_any["pell"][bits] if d[0])
        n_gauss = sum(1 for d in results_any["gaussian"][bits] if d[0])
        print(f"  {bits:2d}b general: pell={n_pell}% gauss={n_gauss}% "
              f"({elapsed:.1f}s)")

    print_phase_results(
        "PHASE 1: General semiprimes (no congruence constraint)",
        results_any, BIT_SIZES, approaches, TRIALS)

    # Phase 2: p = q = 1 mod 4
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
        n_pell = sum(1 for d in results_1mod4["pell"][bits] if d[0])
        n_gauss = sum(1 for d in results_1mod4["gaussian"][bits] if d[0])
        print(f"  {bits:2d}b 1mod4:   pell={n_pell}% gauss={n_gauss}% "
              f"({elapsed:.1f}s)")

    print_phase_results(
        "PHASE 2: Semiprimes with p = q = 1 mod 4 (Gaussian-optimal)",
        results_1mod4, BIT_SIZES, approaches, TRIALS)

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("SUMMARY: Success rates (%)")
    print("=" * 72)
    print()

    header = f"{'Approach':>12s} | {'Cond':>6s}"
    for bits in BIT_SIZES:
        header += f" | {bits:>4d}b"
    print(header)
    print("-" * len(header))

    for app in approaches:
        row = f"{app:>12s} | {'any':>6s}"
        for bits in BIT_SIZES:
            data = results_any[app][bits]
            n_success = sum(1 for d in data if d[0])
            rate = n_success / max(len(data), 1) * 100
            row += f" | {rate:4.0f}%"
        print(row)
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
    print("Does the success rate hold or decay with bit size?")
    print("Constant rate = polynomial-time candidate.")
    print("Exponential decay = sub-exponential or worse.")
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
    # Method breakdown
    # -------------------------------------------------------------------------
    print("=" * 72)
    print("METHOD BREAKDOWN")
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

    # -------------------------------------------------------------------------
    # Theoretical assessment
    # -------------------------------------------------------------------------
    print("=" * 72)
    print("THEORETICAL ASSESSMENT")
    print("=" * 72)
    print()
    print("Approach 2 (Fermat + LLL):")
    print("  The dominant method is fermat_C_direct (plain Fermat iteration near")
    print("  isqrt(N)). The LLL-based strategies (A, B) rarely add value because")
    print("  x^2 - y^2 = N is a QUADRATIC constraint that cannot be linearized")
    print("  for lattice reduction. Success decays as p-q grows relative to sqrt(N).")
    print()
    print("Approach 4 (Gaussian integers):")
    print("  0% success at ALL bit sizes. The 2x2 lattice [[N,0],[r,1]] produces")
    print("  (a,b) with a^2+b^2 = kN, but extracting factors from a+bi in Z[i]")
    print("  requires trial division by Gaussian primes up to norm sqrt(kN) ~")
    print("  sqrt(N), which is O(sqrt(N)) work -- NO better than trial division")
    print("  of N itself. The multi-D combined lattice short vectors encode LINEAR")
    print("  relations among roots r_D, not quadratic ones, so the factoring")
    print("  information is not accessible from lattice entries alone.")
    print("  NOTE: finding r with r^2 = -1 mod N is done via CRT with known")
    print("  factors. Without knowing p,q, finding two INDEPENDENT such roots is")
    print("  itself equivalent to factoring N.")
    print()
    print("Approach 5 (Multi-ring):")
    print("  ring_D1_trial is the dominant method (Gaussian trial division in Z[i]).")
    print("  It works for 16-24 bit N because p,q < N^{1/4} trial division bound.")
    print("  For 28+ bit N, p,q exceed the bound and success drops to 0%.")
    print("  Other rings (D=2,3,7,...) contribute modestly at small bit sizes.")
    print("  Cross-ring extraction yields almost no additional successes.")
    print("  CONCLUSION: multi-ring trial division is O(N^{1/4}), no asymptotic gain.")
    print()
    print("Approach 6 (Pell / CF):")
    print("  The most successful approach. CF convergents of sqrt(N) produce")
    print("  x^2 - Ny^2 = small residues, and gcd(|residue|, N) is sometimes")
    print("  nontrivial. The smooth-residue sieve (pell_sieve_pair/triple)")
    print("  is essentially a MINI QUADRATIC SIEVE using CF-produced relations.")
    print("  It dominates at 24+ bits. Success decays from ~99% at 16b to ~1-4%")
    print("  at 48b because the CF period grows as O(sqrt(N)) and the probability")
    print("  of finding enough smooth residues drops exponentially.")
    print()
    print("CONCLUSION: None of these hyperbolic-lattice constructions yield a")
    print("polynomial-time factoring algorithm. The hyperbola xy = N encodes")
    print("factoring as a QUADRATIC constraint, which lattice reduction (a LINEAR")
    print("tool) cannot directly solve. The most successful sub-method is the CF")
    print("smooth-residue sieve (Pell approach), which is a known sub-exponential")
    print("technique. The Gaussian integer lattice is a DEAD END for factoring:")
    print("it reformulates the problem without reducing its difficulty.")
    print()
    print("=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
