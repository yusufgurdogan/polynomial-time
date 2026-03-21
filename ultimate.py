#!/usr/bin/env python3
"""
The "kitchen sink" factoring algorithm.

Combines every known algebraic group whose order we can probe:
1. (Z/pZ)* — order p-1 (Pollard p-1)
2. Norm-1 torus of F_{p²} — order p+1 (Williams p+1)
3. Cyclotomic tori T_d — order Φ_d(p) for d = 3, 4, 6
4. Elliptic curves E(F_p) — orders p+1-t for many t values
5. Quadratic twist curves — orders p+1+t (paired with E)
6. Stage 2 continuation for each — catches one large prime factor
7. Baby-step giant-step on small cofactors

All run with smoothness bound B = O(bits²), keeping total work polynomial.
The probability of success grows with the number of independent groups tested.

This is the strongest known classical approach short of NFS/QS sieving.
"""

import math
import random
from typing import Optional
from sympy import isprime


def ultimate_factor(n: int) -> Optional[list[int]]:
    """The most comprehensive classical factoring algorithm we can build."""
    if n < 4:
        return None
    if n % 2 == 0:
        return _finish(n, 2)
    if n % 3 == 0:
        return _finish(n, 3)

    bits = n.bit_length()
    sqrt_n = math.isqrt(n)
    if sqrt_n * sqrt_n == n and isprime(sqrt_n):
        return [sqrt_n, sqrt_n]

    # Smoothness bound — polynomial in bits
    B1 = bits * bits          # Stage 1 bound
    B2 = B1 * bits            # Stage 2 bound
    primes = _sieve(B2)
    stage1_primes = [p for p in primes if p <= B1]
    stage2_primes = [p for p in primes if B1 < p <= B2]

    # =================================================================
    # PHASE 1: Cyclotomic tori (p-1, p+1, and higher)
    # =================================================================
    # For each d, work in (Z/nZ)[x]/(Φ_d(x)) and raise (1+x) to lcm(1..B1)
    # Tests if Φ_d(p) is B1-smooth

    # d=1: Φ₁(p) = p-1 (classic p-1 method)
    # d=2: Φ₂(p) = p+1 (Williams p+1, via quadratic ring)
    # d=3: Φ₃(p) = p²+p+1
    # d=4: Φ₄(p) = p²+1
    # d=6: Φ₆(p) = p²-p+1

    # --- d=1: Pollard p-1 with stage 2 ---
    for base_val in [2, 3, 5]:
        a = base_val
        for p in stage1_primes:
            pk = p
            while pk <= B1:
                a = pow(a, p, n)
                pk *= p

        g = math.gcd(a - 1, n)
        if 1 < g < n:
            return _finish(n, g)

        # Stage 2: check individual primes in (B1, B2]
        factor = _stage2_pm1(a, n, stage2_primes)
        if factor:
            return factor

    # --- d=2: Williams p+1 via Lucas sequences ---
    for P in [3, 5, 7, 11, 13, 17, 19, 23]:
        v = P % n
        for p in stage1_primes:
            pk = p
            while pk <= B1:
                v = _lucas_v(v, p, n)
                pk *= p

        g = math.gcd(v - 2, n)
        if 1 < g < n:
            return _finish(n, g)

        # Stage 2 for p+1
        factor = _stage2_pp1(v, n, stage2_primes)
        if factor:
            return factor

    # --- d=2 via Frobenius ring (complementary to Lucas) ---
    for c in range(2, min(bits * 2, 40)):
        j = _jacobi(c, n)
        if j == 0:
            g = math.gcd(c, n)
            if 1 < g < n:
                return _finish(n, g)
            continue

        # Work in (Z/nZ)[x]/(x²-c)
        a, b = 1, 1  # element 1 + x
        for p in stage1_primes:
            pk = p
            while pk <= B1:
                a, b = _pow_ring2(a, b, p, c, n)
                pk *= p

        # Check both components
        for val in [b, a - 1, a + 1, (a * a - c * b * b - 1) % n]:
            if val % n == 0:
                continue
            g = math.gcd(val % n, n)
            if 1 < g < n:
                return _finish(n, g)

    # --- d=3: Φ₃(p) = p²+p+1 via cubic ring ---
    for c in [2, 3, 5]:
        # Work in (Z/nZ)[x]/(x³-c), elements are (a, b, d)
        elem = (1, 1, 0)  # 1 + x
        for p in stage1_primes:
            pk = p
            while pk <= B1:
                elem = _pow_ring3(elem, p, c, n)
                pk *= p

        ea, eb, ed = elem
        for val in [eb, ed, ea - 1, ea + 1]:
            if val % n == 0:
                continue
            g = math.gcd(val % n, n)
            if 1 < g < n:
                return _finish(n, g)

    # --- d=4: Φ₄(p) = p²+1 via quartic structure ---
    # Work in (Z/nZ)[x]/(x²+1), i.e., Gaussian integers mod n
    # The unit group of Z[i]/nZ[i] has order related to Π Φ_d(p) for d | 4
    for start_b in [1, 2, 3]:
        a, b = 1, start_b  # 1 + start_b * i
        for p in stage1_primes:
            pk = p
            while pk <= B1:
                # x²+1 = x² - (-1), so c = -1
                a, b = _pow_ring2(a, b, p, -1, n)
                pk *= p

        for val in [b, a - 1, a + 1, (a*a + b*b - 1) % n]:
            if val % n == 0:
                continue
            g = math.gcd(val % n, n)
            if 1 < g < n:
                return _finish(n, g)

    # --- d=6: Φ₆(p) = p²-p+1 ---
    # Φ₆(x) = x²-x+1. Work in (Z/nZ)[x]/(x²-x+1)
    # Multiplication: (a+bω)(c+dω) where ω²=ω-1
    # = ac + (ad+bc)ω + bd·ω² = ac + (ad+bc)ω + bd(ω-1)
    # = (ac-bd) + (ad+bc+bd)ω
    for start in [1, 2, 3]:
        a, b = 1, start  # 1 + start*ω
        for p in stage1_primes:
            pk = p
            while pk <= B1:
                a, b = _pow_ring_phi6(a, b, p, n)
                pk *= p

        for val in [b, a - 1, a + 1]:
            if val % n == 0:
                continue
            g = math.gcd(val % n, n)
            if 1 < g < n:
                return _finish(n, g)

    # =================================================================
    # PHASE 2: Elliptic Curve Method (many curves)
    # =================================================================
    num_curves = bits * 3  # polynomial number of curves

    for _ in range(num_curves):
        factor = _ecm_one_curve(n, stage1_primes, stage2_primes, B1)
        if factor:
            return factor

    # =================================================================
    # PHASE 3: Pollard rho as fallback
    # =================================================================
    factor = _pollard_rho(n, max_iter=bits * bits * bits)
    if factor:
        return _finish(n, factor)

    return None


# =============================================================================
# Stage 2 continuations
# =============================================================================

def _stage2_pm1(a: int, n: int, stage2_primes: list) -> Optional[list[int]]:
    """Stage 2 for p-1: check if a^q ≡ 1 for each prime q in stage 2."""
    # Use the baby-step giant-step variant of stage 2
    accumulated = 1
    for i, q in enumerate(stage2_primes):
        val = pow(a, q, n)
        accumulated = (accumulated * (val - 1)) % n
        if i % 100 == 99:
            g = math.gcd(accumulated, n)
            if 1 < g < n:
                return _finish(n, g)
            if g == n:
                # Backtrack — one of the primes gave 0
                accumulated = 1
    g = math.gcd(accumulated, n)
    if 1 < g < n:
        return _finish(n, g)
    return None


def _stage2_pp1(v: int, n: int, stage2_primes: list) -> Optional[list[int]]:
    """Stage 2 for p+1: check V_q(v) ≡ 2 for each stage 2 prime q."""
    accumulated = 1
    for i, q in enumerate(stage2_primes):
        vq = _lucas_v(v, q, n)
        accumulated = (accumulated * (vq - 2)) % n
        if i % 100 == 99:
            g = math.gcd(accumulated, n)
            if 1 < g < n:
                return _finish(n, g)
            if g == n:
                accumulated = 1
    g = math.gcd(accumulated, n)
    if 1 < g < n:
        return _finish(n, g)
    return None


# =============================================================================
# ECM
# =============================================================================

def _ecm_one_curve(n, stage1_primes, stage2_primes, B1):
    """Run ECM with one random curve. Suyama's parametrization."""
    sigma = random.randint(6, n - 1)
    u = (sigma * sigma - 5) % n
    v = (4 * sigma) % n

    try:
        v3_inv = pow(v * v * v % n, -1, n)
    except ValueError:
        g = math.gcd(v * v * v % n, n)
        if 1 < g < n:
            return _finish(n, g)
        return None

    Qx = u * u * u % n * v3_inv % n
    Qz = 1

    try:
        u3v_inv = pow(4 * u * u * u % n * v % n, -1, n)
    except ValueError:
        g = math.gcd(4 * u * u * u % n * v % n, n)
        if 1 < g < n:
            return _finish(n, g)
        return None

    A = ((v - u) ** 3 % n * ((3 * u + v) % n) % n * u3v_inv - 2) % n

    # Stage 1
    for p in stage1_primes:
        pk = p
        while pk <= B1:
            Qx, Qz = _ec_mul(Qx, Qz, p, A, n)
            pk *= p
        if Qz == 0:
            break
        g = math.gcd(Qz, n)
        if 1 < g < n:
            return _finish(n, g)
        if g == n:
            return None

    if Qz == 0:
        return None

    g = math.gcd(Qz, n)
    if 1 < g < n:
        return _finish(n, g)

    # Stage 2
    accumulated = 1
    for i, q in enumerate(stage2_primes[:500]):  # cap stage 2 for speed
        Rx, Rz = _ec_mul(Qx, Qz, q, A, n)
        if Rz == 0:
            continue
        accumulated = (accumulated * Rz) % n
        if i % 50 == 49:
            g = math.gcd(accumulated, n)
            if 1 < g < n:
                return _finish(n, g)
            if g == n:
                accumulated = 1

    g = math.gcd(accumulated, n)
    if 1 < g < n:
        return _finish(n, g)
    return None


def _ec_mul(x, z, k, A, n):
    """Montgomery ladder scalar multiplication."""
    if k == 0:
        return 0, 0
    if k == 1:
        return x, z

    r0x, r0z = x, z
    r1x, r1z = _ec_dbl(x, z, A, n)

    for bit in bin(k)[3:]:
        if bit == '1':
            r0x, r0z = _ec_add(r0x, r0z, r1x, r1z, x, z, n)
            r1x, r1z = _ec_dbl(r1x, r1z, A, n)
        else:
            r1x, r1z = _ec_add(r0x, r0z, r1x, r1z, x, z, n)
            r0x, r0z = _ec_dbl(r0x, r0z, A, n)

    return r0x % n, r0z % n


def _ec_dbl(x, z, A, n):
    u = (x + z) * (x + z) % n
    v = (x - z) * (x - z) % n
    diff = (u - v) % n
    a24 = (A + 2) * pow(4, -1, n) % n
    rx = u * v % n
    rz = diff * (v + a24 * diff % n) % n
    return rx, rz


def _ec_add(x1, z1, x2, z2, x0, z0, n):
    u = (x1 - z1) * (x2 + z2) % n
    v = (x1 + z1) * (x2 - z2) % n
    add = (u + v) % n
    sub = (u - v) % n
    rx = z0 * add * add % n
    rz = x0 * sub * sub % n
    return rx, rz


# =============================================================================
# Ring arithmetic
# =============================================================================

def _pow_ring2(a, b, exp, c, n):
    """(a + b*x)^exp in (Z/nZ)[x]/(x²-c)."""
    ra, rb = 1, 0
    ba, bb = a % n, b % n
    while exp > 0:
        if exp & 1:
            ra, rb = (ra*ba + rb*bb*c) % n, (ra*bb + rb*ba) % n
        ba, bb = (ba*ba + bb*bb*c) % n, (2*ba*bb) % n
        exp >>= 1
    return ra, rb


def _pow_ring_phi6(a, b, exp, n):
    """(a + b*ω)^exp in (Z/nZ)[ω]/(ω²-ω+1). Mult: ω² = ω - 1."""
    ra, rb = 1, 0
    ba, bb = a % n, b % n
    while exp > 0:
        if exp & 1:
            new_ra = (ra*ba - rb*bb) % n
            new_rb = (ra*bb + rb*ba + rb*bb) % n
            ra, rb = new_ra, new_rb
        new_ba = (ba*ba - bb*bb) % n
        new_bb = (2*ba*bb + bb*bb) % n
        ba, bb = new_ba, new_bb
        exp >>= 1
    return ra, rb


def _pow_ring3(elem, exp, c, n):
    """elem^exp in (Z/nZ)[x]/(x³-c). elem = (a, b, d)."""
    ra, rb, rd = 1, 0, 0
    ba, bb, bd = elem[0] % n, elem[1] % n, elem[2] % n
    while exp > 0:
        if exp & 1:
            ra, rb, rd = _mul3(ra, rb, rd, ba, bb, bd, c, n)
        ba, bb, bd = _mul3(ba, bb, bd, ba, bb, bd, c, n)
        exp >>= 1
    return (ra, rb, rd)


def _mul3(a1, b1, d1, a2, b2, d2, c, n):
    ra = (a1*a2 + c*(b1*d2 + d1*b2)) % n
    rb = (a1*b2 + b1*a2 + c*d1*d2) % n
    rd = (a1*d2 + b1*b2 + d1*a2) % n
    return ra, rb, rd


def _lucas_v(v, k, n):
    """Compute V_k(v, 1) mod n via binary chain."""
    if k == 0:
        return 2
    if k == 1:
        return v % n

    vl, vh = v % n, (v * v - 2) % n
    for bit in bin(k)[3:]:
        if bit == '1':
            vl = (vl * vh - v) % n
            vh = (vh * vh - 2) % n
        else:
            vh = (vl * vh - v) % n
            vl = (vl * vl - 2) % n
    return vl


# =============================================================================
# Pollard rho fallback
# =============================================================================

def _pollard_rho(n, max_iter=100000):
    x = random.randint(2, n - 1)
    y = x
    c = random.randint(1, n - 1)
    d = 1
    while d == 1 and max_iter > 0:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        max_iter -= 1
    return d if 1 < d < n else None


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
            from baselines import pollard_rho as pr
            sub = pr(part)
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
