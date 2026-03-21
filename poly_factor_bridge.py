#!/usr/bin/env python3
"""
Experiment 6: Polynomial factoring ↔ integer factoring bridge.

Umans-Wang (2025) showed a connection between polynomial factoring over
finite fields and deterministic integer factoring. The idea:

For N = pq, consider the polynomial ring F_2[x] / (x^n - 1) or similar.
The Chinese Remainder Theorem applies to polynomials too:
  Z/NZ ≅ Z/pZ × Z/qZ
  F_p[x]/(f) ≅ product of extensions

Can we exploit polynomial factoring over Z/NZ to factor N?

Key ideas to test:
1. Factor x^k - 1 mod N for various k. Over Z/pZ, x^k - 1 factors into
   cyclotomic polynomials. The factorization differs mod p vs mod q.
   Inconsistent factorizations → factor of N.

2. Berlekamp's algorithm for polynomial factoring uses the Frobenius
   endomorphism x → x^p mod f(x). Over Z/NZ, the "Frobenius" is
   x → x^N mod f(x). But x^N mod f ≠ x^p · x^q in any useful sense...
   Or does it? If f splits differently mod p and mod q, the GCD of
   (x^N - x) mod f(x) mod N with f(x) mod N might reveal structure.

3. Hensel lifting: factor a polynomial mod p, then lift to mod p^k.
   If we could factor mod N = pq, the factorization would reveal
   information about p and q separately.

4. The resultant approach: for two polynomials f, g in Z/NZ[x],
   Res(f, g) mod N encodes information about common roots mod p vs mod q.
"""

import math
import random
import sys
import numpy as np
from sympy import nextprime, Poly, Symbol, ZZ, GF, factorint, gcd as sym_gcd

sys.stdout.reconfigure(line_buffering=True)


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, hi))
    while q == p:
        q = nextprime(random.randint(lo, hi))
    if p > q:
        p, q = q, p
    return p * q, p, q


def poly_mul_mod(a, b, f, N):
    """Multiply polynomials a, b modulo f(x) and N."""
    # a, b, f are lists of coefficients [a0, a1, ..., ad]
    deg_f = len(f) - 1
    # Multiply
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] = (result[i + j] + ai * bj) % N
    # Reduce mod f
    while len(result) > deg_f:
        if result[-1] != 0:
            coeff = result[-1]
            # f is monic (leading coeff = 1), so subtract coeff * f * x^(len-deg_f-1)
            shift = len(result) - deg_f - 1
            for k in range(deg_f + 1):
                result[shift + k] = (result[shift + k] - coeff * f[k]) % N
        result.pop()
    return [c % N for c in result]


def poly_pow_mod(base, exp, f, N):
    """Compute base^exp mod (f(x), N) using repeated squaring."""
    if exp == 0:
        result = [1] + [0] * (len(f) - 2)
        return result
    result = [1] + [0] * (len(f) - 2)
    b = list(base)
    while exp > 0:
        if exp % 2 == 1:
            result = poly_mul_mod(result, b, f, N)
        b = poly_mul_mod(b, b, f, N)
        exp //= 2
    return result


def poly_gcd_mod(a, b, N):
    """GCD of polynomials mod N. May fail if leading coeff not invertible."""
    a = list(a)
    b = list(b)
    # Remove trailing zeros
    while a and a[-1] % N == 0:
        a.pop()
    while b and b[-1] % N == 0:
        b.pop()
    if not a:
        return b
    if not b:
        return a
    while b:
        while b and b[-1] % N == 0:
            b.pop()
        if not b:
            break
        if len(a) < len(b):
            a, b = b, a
        # Try to invert leading coeff of b
        lc = b[-1] % N
        g = math.gcd(lc, N)
        if g > 1 and g < N:
            return [g]  # Found a factor!
        if g == N:
            b.pop()
            continue
        lc_inv = pow(lc, -1, N)
        # Make b monic
        b = [(c * lc_inv) % N for c in b]
        # Reduce a by b
        while len(a) >= len(b):
            coeff = a[-1] % N
            shift = len(a) - len(b)
            for k in range(len(b)):
                a[shift + k] = (a[shift + k] - coeff * b[k]) % N
            while a and a[-1] % N == 0:
                a.pop()
        a, b = b, a
    return a


# ========================================================================
# Approach 1: Frobenius endomorphism inconsistency
# ========================================================================

def frobenius_attack(N, p, q, max_deg=20):
    """
    For a random polynomial f(x) of degree d over Z/NZ:
    - Compute g(x) = x^N - x mod f(x) mod N
    - Over Z/pZ: g(x) = x^p - x mod f has roots = elements of F_p in the splitting field
    - Over Z/qZ: g(x) = x^q - x mod f has roots = elements of F_q
    - gcd(g, f) mod N should split differently mod p and mod q
    - If the gcd computation encounters a non-invertible element → factor!
    """
    results = []
    for deg in range(2, max_deg + 1):
        # Random monic polynomial of degree deg
        f = [random.randint(0, N - 1) for _ in range(deg)] + [1]

        # x = [0, 1, 0, ..., 0]
        x_poly = [0, 1] + [0] * (deg - 2)

        # Compute x^N mod f mod N
        try:
            xN = poly_pow_mod(x_poly, N, f, N)
        except Exception:
            continue

        # g = x^N - x mod f mod N
        g = list(xN)
        while len(g) < len(x_poly):
            g.append(0)
        g[1] = (g[1] - 1) % N

        # Remove trailing zeros
        while g and g[-1] % N == 0:
            g.pop()
        if not g:
            continue

        # gcd(g, f) mod N
        try:
            h = poly_gcd_mod(g, f, N)
        except Exception:
            continue

        # Check if gcd computation found a factor
        if h and len(h) == 1 and h[0] > 1 and h[0] < N:
            factor = h[0]
            if N % factor == 0:
                results.append(('frobenius_gcd', deg, factor))
                return results

        # Also check coefficients of h for gcd with N
        if h:
            for coeff in h:
                g_val = math.gcd(int(coeff) % N, N)
                if 1 < g_val < N:
                    results.append(('frobenius_coeff', deg, g_val))
                    return results

    return results


# ========================================================================
# Approach 2: Berlekamp-style splitting
# ========================================================================

def berlekamp_attack(N, p, q, max_deg=15):
    """
    Berlekamp's algorithm factors f(x) over F_p by:
    1. Computing the Frobenius matrix: columns are x^{ip} mod f for i = 0,...,d-1
    2. Finding the null space of (Frob - I)
    3. Null space vectors give splitting info

    Over Z/NZ, the "Frobenius" at N doesn't correspond to either p or q.
    But we can try: compute x^{(N-1)/2} mod f(x) mod N.
    If f is irreducible mod p but splits mod q (or vice versa),
    then gcd(x^{(N-1)/2} - 1, f) mod N might factor.
    """
    results = []
    for deg in range(2, max_deg + 1):
        f = [random.randint(0, N - 1) for _ in range(deg)] + [1]

        x_poly = [0, 1] + [0] * (deg - 2)

        # Compute x^{(N-1)/2} mod f mod N  (Euler criterion analogue)
        exp = (N - 1) // 2
        try:
            xE = poly_pow_mod(x_poly, exp, f, N)
        except Exception:
            continue

        # g = x^{(N-1)/2} - 1 mod f
        g1 = list(xE)
        g1[0] = (g1[0] - 1) % N

        # g = x^{(N-1)/2} + 1 mod f
        g2 = list(xE)
        g2[0] = (g2[0] + 1) % N

        for g in [g1, g2]:
            while g and g[-1] % N == 0:
                g.pop()
            if not g:
                continue
            try:
                h = poly_gcd_mod(g, f, N)
            except Exception:
                continue
            if h and len(h) == 1 and h[0] > 1 and h[0] < N:
                if N % h[0] == 0:
                    results.append(('berlekamp', deg, h[0]))
                    return results
            if h:
                for coeff in h:
                    gv = math.gcd(int(coeff) % N, N)
                    if 1 < gv < N:
                        results.append(('berlekamp_coeff', deg, gv))
                        return results

    return results


# ========================================================================
# Approach 3: Cyclotomic polynomial splitting
# ========================================================================

def cyclotomic_attack(N, p, q, max_k=50):
    """
    x^k - 1 = ∏ Φ_d(x) for d | k.
    Over F_p: Φ_d(x) splits into φ(d)/ord_d(p) irreducible factors.
    Over F_q: Φ_d(x) splits into φ(d)/ord_d(q) irreducible factors.

    If ord_d(p) ≠ ord_d(q), the splitting of Φ_d mod N is inconsistent.
    GCD of intermediate results might reveal factors.
    """
    results = []
    for k in range(2, max_k + 1):
        # Build x^k - 1 mod N
        f = [-1] + [0] * (k - 1) + [1]

        x_poly = [0, 1] + [0] * (k - 2) if k > 2 else [0, 1]

        # Compute gcd(x^{N} - x, x^k - 1) mod N
        # This should give ∏ Φ_d where d | gcd(k, p-1) over F_p
        # and ∏ Φ_d where d | gcd(k, q-1) over F_q
        try:
            xN = poly_pow_mod(x_poly, N, f, N)
        except Exception:
            continue

        g = list(xN)
        while len(g) < 2:
            g.append(0)
        g[1] = (g[1] - 1) % N

        while g and g[-1] % N == 0:
            g.pop()
        if not g:
            continue

        try:
            h = poly_gcd_mod(g, f, N)
        except Exception:
            continue

        if h and len(h) == 1 and h[0] > 1 and h[0] < N:
            if N % h[0] == 0:
                results.append(('cyclotomic', k, h[0]))
                return results
        if h:
            for coeff in h:
                gv = math.gcd(int(coeff) % N, N)
                if 1 < gv < N:
                    results.append(('cyclotomic_coeff', k, gv))
                    return results

    return results


# ========================================================================
# Approach 4: Random polynomial splitting
# ========================================================================

def random_poly_split(N, p, q, num_trials=100, max_deg=10):
    """
    Pick random polynomial f(x), compute gcd(f(x)^{(N-1)/2} - 1, x^k - 1) mod N
    for various k. The Euler criterion applied to polynomials can reveal
    splitting behavior differences mod p vs mod q.
    """
    results = []
    exp = (N - 1) // 2

    for trial in range(num_trials):
        deg = random.randint(2, max_deg)
        f_mod = [random.randint(0, N - 1) for _ in range(deg)] + [1]

        # Use x^deg - a for random a (simpler)
        a = random.randint(2, N - 1)
        f_mod = [(-a) % N] + [0] * (deg - 1) + [1]

        x_poly = [0, 1] + [0] * (deg - 2)

        try:
            xE = poly_pow_mod(x_poly, exp, f_mod, N)
        except Exception:
            continue

        # Check x^{(N-1)/2} mod f — if f is irred mod p but split mod q,
        # this equals different things mod p and mod q
        # Check gcd(x^{(N-1)/2} - 1, f) and gcd(x^{(N-1)/2} + 1, f)
        for delta in [1, -1]:
            g = list(xE)
            g[0] = (g[0] - delta) % N
            while g and g[-1] % N == 0:
                g.pop()
            if not g:
                continue
            try:
                h = poly_gcd_mod(g, f_mod, N)
            except Exception:
                continue
            if h:
                for coeff in h:
                    gv = math.gcd(int(coeff) % N, N)
                    if 1 < gv < N:
                        results.append(('random_split', deg, gv))
                        return results

    return results


# ========================================================================
# Main experiment
# ========================================================================

def run_experiment(bits_list=None, num_instances=50):
    if bits_list is None:
        bits_list = [16, 20, 24, 28, 32, 40]

    print(f"{'='*75}")
    print(f"  POLYNOMIAL FACTORING ↔ INTEGER FACTORING BRIDGE")
    print(f"  Testing 4 approaches across {bits_list} bit sizes")
    print(f"{'='*75}")

    approaches = [
        ('Frobenius', frobenius_attack),
        ('Berlekamp', berlekamp_attack),
        ('Cyclotomic', cyclotomic_attack),
        ('RandomSplit', random_poly_split),
    ]

    summary = {}

    for bits in bits_list:
        print(f"\n  --- {bits}-bit semiprimes ({num_instances} instances) ---")
        counts = {name: 0 for name, _ in approaches}

        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)

            for name, func in approaches:
                try:
                    results = func(N, p, q)
                    if results:
                        counts[name] += 1
                except Exception:
                    pass

            if (inst + 1) % 25 == 0:
                rates = "  ".join(f"{name}={counts[name]}/{inst+1}" for name, _ in approaches)
                print(f"    [{inst+1}/{num_instances}] {rates}")

        print(f"\n  Results at {bits} bits:")
        for name, _ in approaches:
            rate = counts[name] / num_instances
            print(f"    {name:15s}: {counts[name]:3d}/{num_instances} = {rate:.1%}")
        summary[bits] = dict(counts)

    # Summary table
    print(f"\n{'='*75}")
    print(f"  SUMMARY: Success rates by approach and bit size")
    print(f"{'='*75}")
    header = f"  {'bits':>4}"
    for name, _ in approaches:
        header += f"  {name:>12}"
    print(header)
    print(f"  {'-'*4}" + f"  {'-'*12}" * len(approaches))
    for bits in bits_list:
        if bits not in summary:
            continue
        row = f"  {bits:4d}"
        for name, _ in approaches:
            rate = summary[bits][name] / num_instances
            row += f"  {rate:11.1%}"
        print(row)

    # Analysis
    print(f"\n  ANALYSIS:")
    print(f"  The Frobenius/Berlekamp/cyclotomic approaches work by computing")
    print(f"  polynomial GCDs mod N. When the GCD computation encounters a")
    print(f"  non-invertible element mod N, it reveals a factor.")
    print(f"  This is essentially the same as Pollard's p-1 / ECM: it works")
    print(f"  when the multiplicative orders of elements differ mod p vs mod q.")
    print(f"  The polynomial layer doesn't add fundamentally new structure.")


if __name__ == "__main__":
    random.seed(42)
    run_experiment(bits_list=[16, 20, 24, 28, 32], num_instances=30)
