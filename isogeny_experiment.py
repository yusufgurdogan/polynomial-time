#!/usr/bin/env python3
"""
Isogeny-asymmetry factoring.

Key idea (not on the kill list AFAIK):
For random E/Z/NZ, the j-invariant j(E) satisfies Phi_ell(j, Y) = 0 for ell-isogenous Y.
Mod p and mod q, the roots of Phi_ell(j, Y) form potentially DIFFERENT sets:
  - An elliptic curve is ordinary mod p (|roots|=ell+1) or supersingular mod p (|roots|=1,2,0).
  - Same curve can be ordinary mod p but supersingular mod q, giving ASYMMETRIC root counts.

When we try to find a root of a poly over Z/NZ[Y] whose mod-p and mod-q factorizations
have different degrees, Euclidean GCD between the poly and (Y^p - Y) hits zero divisors
→ factor of N.

Test: pick random E (random j-invariant mod N), compute Phi_2(j, Y), then
compute gcd(Phi_2(j, Y), Y^N - Y) over Z/NZ[Y]. If p-factorization ≠ q-factorization,
a leading coefficient should be a zero divisor → gcd reveals factor.
"""
import math, random, sys, time
from sympy import nextprime
sys.stdout.reconfigure(line_buffering=True)


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    while True:
        p = nextprime(random.randint(lo, hi))
        if p > hi: continue
        q = nextprime(random.randint(lo, hi))
        if q > hi or q == p: continue
        N = p * q
        if N.bit_length() >= bits - 1:
            return N, min(p, q), max(p, q)


def phi2_coeffs():
    """Classical modular polynomial Phi_2(X, Y). Symmetric in X, Y.
    Phi_2(X, Y) = X^3 + Y^3 - X^2*Y^2 + 1488*X^2*Y + 1488*X*Y^2
                  - 162000*X^2 - 162000*Y^2 + 40773375*X*Y
                  + 8748000000*X + 8748000000*Y - 157464000000000
    Return as dict {(i,j): coeff} for X^i Y^j.
    """
    return {
        (3, 0): 1, (0, 3): 1, (2, 2): -1,
        (2, 1): 1488, (1, 2): 1488,
        (2, 0): -162000, (0, 2): -162000,
        (1, 1): 40773375,
        (1, 0): 8748000000, (0, 1): 8748000000,
        (0, 0): -157464000000000,
    }


def phi2_Y_polynomial(j, N):
    """Given j-invariant j in Z/NZ, return Phi_2(j, Y) as polynomial in Y.
    Returns coefficients [c0, c1, c2, c3] meaning c0 + c1*Y + c2*Y^2 + c3*Y^3."""
    coeffs = [0, 0, 0, 0]
    for (i, k), c in phi2_coeffs().items():
        c_mod = c % N
        ji = pow(j, i, N)
        coeffs[k] = (coeffs[k] + c_mod * ji) % N
    return coeffs  # in Z/NZ


def poly_mul(A, B, N):
    if not A or not B: return []
    C = [0] * (len(A) + len(B) - 1)
    for i, a in enumerate(A):
        if a == 0: continue
        for j, b in enumerate(B):
            if b == 0: continue
            C[i + j] = (C[i + j] + a * b) % N
    while C and C[-1] == 0: C.pop()
    return C


def poly_divmod(A, B, N):
    """Divide A by B in Z/NZ[Y]. B leading coeff must be invertible mod N.
    Returns (Q, R) or raises with zero-divisor info."""
    A = list(A)
    if not B: raise ValueError("div by zero")
    lead_b = B[-1]
    g = math.gcd(lead_b, N)
    if 1 < g < N:
        # Found factor!
        raise FactorFound(g)
    try:
        inv_lead = pow(lead_b, -1, N)
    except ValueError:
        raise FactorFound(math.gcd(lead_b, N))

    Q = [0] * max(0, len(A) - len(B) + 1)
    while len(A) >= len(B) and A:
        deg_diff = len(A) - len(B)
        factor = (A[-1] * inv_lead) % N
        Q[deg_diff] = factor
        for i in range(len(B)):
            A[deg_diff + i] = (A[deg_diff + i] - factor * B[i]) % N
        while A and A[-1] == 0: A.pop()
    return Q, A


class FactorFound(Exception):
    def __init__(self, f): self.f = f


def poly_gcd(A, B, N):
    """GCD of A, B in Z/NZ[Y]. May hit zero divisor → factor."""
    while B:
        _, R = poly_divmod(A, B, N)
        A, B = B, R
    return A


def poly_pow_mod(base, exp, modulus, N):
    """Compute base^exp mod modulus in Z/NZ[Y]. modulus is a polynomial."""
    result = [1]
    while exp > 0:
        if exp & 1:
            result = poly_mul(result, base, N)
            _, result = poly_divmod(result, modulus, N)
        base = poly_mul(base, base, N)
        _, base = poly_divmod(base, modulus, N)
        exp >>= 1
    return result


def try_isogeny_factor(N, j):
    """Compute Phi_2(j, Y) over Z/NZ, then gcd(Phi_2, Y^N - Y)."""
    try:
        P = phi2_Y_polynomial(j, N)  # degree 3 in Y
        # Check leading coeffs for direct factor
        for c in P:
            g = math.gcd(c, N)
            if 1 < g < N:
                return g, "direct_coef"

        # Compute Y^N mod P (= Y^N reduced by Phi_2(j, Y))
        Y = [0, 1]  # Y
        YN = poly_pow_mod(Y, N, P, N)
        # Y^N - Y mod P
        if len(YN) < 2:
            YN = YN + [0] * (2 - len(YN))
        YN[1] = (YN[1] - 1) % N
        while YN and YN[-1] == 0: YN.pop()

        if not YN:
            return None, "empty"

        G = poly_gcd(P, YN, N)
        # If G nontrivial polynomial AND its coefficients share factor with N, done
        for c in G:
            g = math.gcd(c, N)
            if 1 < g < N:
                return g, "gcd_coef"
        return None, "no_signal"
    except FactorFound as fe:
        return fe.f, "zero_divisor"


def run(bits, n_instances):
    print(f"\n{'='*70}\n  bits={bits}, n={n_instances}\n{'='*70}")

    factors = 0
    reasons = {}
    random_factors = 0  # baseline
    t0 = time.time()

    for inst in range(n_instances):
        N, p, q = generate_semiprime(bits)
        # Try multiple random j-invariants per N (budget ~20)
        got = None
        for _ in range(20):
            j = random.randint(2, N - 2)
            # Skip obvious: j = 0, 1728 are supersingular on specific primes
            res, reason = try_isogeny_factor(N, j)
            if res:
                got = (res, reason); break
        if got:
            factors += 1
            reasons[got[1]] = reasons.get(got[1], 0) + 1

        # Birthday baseline: 20 random gcds (matching attempt count)
        rbird = False
        for _ in range(20):
            x = random.randint(2, N - 2)
            if 1 < math.gcd(x, N) < N:
                rbird = True; break
        if rbird: random_factors += 1

    dt = time.time() - t0
    print(f"  Isogeny factor: {factors}/{n_instances}  reasons={reasons}  ({dt:.1f}s)")
    print(f"  Random (20 gcds): {random_factors}/{n_instances}")
    return factors, random_factors, n_instances


if __name__ == "__main__":
    random.seed(17)
    print("Isogeny j-root-finding factoring test")
    print("Phi_2(j, Y) root-finding via gcd with Y^N - Y over Z/NZ[Y]\n")

    rows = []
    for bits in [16, 24, 32, 40, 48, 56]:
        f, r, n = run(bits, 50)
        rows.append((bits, f, r, n))

    print(f"\n\n{'='*70}\n  VERDICT\n{'='*70}")
    print(f"  {'bits':>5} {'Isogeny':>12} {'Random':>12}  delta")
    for bits, f, r, n in rows:
        print(f"  {bits:>5} {f:>5}/{n:<4} {r:>5}/{n:<4}  {f-r:+d}")
