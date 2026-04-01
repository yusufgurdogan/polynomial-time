"""
FactorCup Entry — Pollard rho + ECM escalation.

  ≤80b:  Pollard rho (Brent + batch GCD)
  >80b:  sympy.factorint (rho + p-1 + escalating ECM)
         This internally does: trial div → p-1 → rho → ECM(B1=10K) → ECM(B1=50K) → ...
"""

import math
import random
from sympy.ntheory import factorint as _sympy_factorint


def factor(N: int) -> tuple[int, int]:
    if N % 2 == 0:
        return (2, N // 2)
    s = math.isqrt(N)
    if s * s == N:
        return (s, s)

    for d in range(3, min(10000, s + 1), 2):
        if N % d == 0:
            return _ret(d, N)

    bits = N.bit_length()

    # Pollard rho for small numbers
    if bits <= 80:
        result = _pollard_rho(N, max_r=500000)
        if result:
            return result

    # sympy.factorint handles everything else
    # Internally it escalates: trial div → p-1 → rho → ECM(B1=10K,50c) → ECM(B1=50K,200c)
    return _sympy_factor(N)


def _ret(d, N):
    q = N // d
    return (min(d, q), max(d, q))


def _sympy_factor(N):
    factors = _sympy_factorint(N)
    ps = list(factors.keys())
    if len(ps) >= 2:
        p = int(ps[0])  # convert from mpz if gmpy2 is present
        return _ret(p, N)
    raise ValueError(f"Failed to factor {N}")


def _pollard_rho(N, max_r=500000):
    for c in range(1, 50):
        x = y = (c * 7 + 13) % N + 2
        d, r, q = 1, 1, 1
        while d == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % N
            k = 0
            while k < r and d == 1:
                ys = y
                m = min(128, r - k)
                for _ in range(m):
                    y = (y * y + c) % N
                    q = q * (x - y) % N
                d = math.gcd(q, N)
                k += m
            r <<= 1
            if r > max_r:
                break
        if d == N:
            d = 1
            while d == 1:
                ys = (ys * ys + c) % N
                d = math.gcd(x - ys, N)
        if 1 < d < N:
            return _ret(d, N)
    return None
