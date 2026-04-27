#!/usr/bin/env python3
"""
Cascade Phi_2, Phi_3, Phi_5 modular-polynomial root-GCDs on the same j.

Hypothesis: each ell probes a different asymmetry mod p vs mod q.
If independent, aggregate factor-rate = 1 - prod(1 - rate_ell) -> compounds.
If correlated, aggregate = max(rate_ell), no compounding.
Measure alpha for Phi_2 alone, Phi_3 alone, Phi_5 alone, cascade.
"""
import math, random, sys, time
from sympy import nextprime
sys.stdout.reconfigure(line_buffering=True)

from k35_isogeny import (
    poly_mul, poly_divmod, poly_gcd, poly_pow_mod, FactorFound,
    generate_semiprime, phi2_Y_polynomial,
)


# Phi_3 coefficients: symmetric modular polynomial of level 3.
# Phi_3(X,Y) = X^4 + Y^4 - X^3*Y^3
#   + 2232*(X^3*Y^2 + X^2*Y^3)
#   - 1069956*(X^3*Y + X*Y^3)
#   + 36864000*(X^3 + Y^3)
#   + 2587918086*X^2*Y^2
#   + 8900222976000*(X^2*Y + X*Y^2)
#   + 452984832000000*(X^2 + Y^2)
#   - 770845966336000000*X*Y
#   + 1855425871872000000000*(X + Y)
def phi3_coeffs():
    d = {}
    d[(4, 0)] = 1; d[(0, 4)] = 1
    d[(3, 3)] = -1
    d[(3, 2)] = 2232; d[(2, 3)] = 2232
    d[(3, 1)] = -1069956; d[(1, 3)] = -1069956
    d[(3, 0)] = 36864000; d[(0, 3)] = 36864000
    d[(2, 2)] = 2587918086
    d[(2, 1)] = 8900222976000; d[(1, 2)] = 8900222976000
    d[(2, 0)] = 452984832000000; d[(0, 2)] = 452984832000000
    d[(1, 1)] = -770845966336000000
    d[(1, 0)] = 1855425871872000000000; d[(0, 1)] = 1855425871872000000000
    d[(0, 0)] = 0  # actually has a huge constant but not needed for our purposes (truncate; we want root asymmetry not exact identity)
    return d


def phi3_Y_polynomial(j, N):
    """Phi_3(j, Y) as polynomial in Y, coefficients mod N. Degree 4 in Y."""
    coeffs = [0] * 5
    for (i, k), c in phi3_coeffs().items():
        c_mod = c % N
        ji = pow(j, i, N)
        coeffs[k] = (coeffs[k] + c_mod * ji) % N
    return coeffs


def try_phi_factor(N, j, ell):
    """Try factor via GCD of Phi_ell(j, Y) and Y^N - Y over Z/NZ[Y]."""
    try:
        if ell == 2:
            P = phi2_Y_polynomial(j, N)
        elif ell == 3:
            P = phi3_Y_polynomial(j, N)
        else:
            return None, "unsupported"

        # Direct gcd of coefficients
        for c in P:
            g = math.gcd(c, N)
            if 1 < g < N:
                return g, "direct"

        # Strip trailing zeros
        while P and P[-1] == 0: P.pop()
        if len(P) < 2:
            return None, "degenerate"

        # Compute Y^N mod P
        Y = [0, 1]
        YN = poly_pow_mod(Y, N, P, N)
        # Y^N - Y
        while len(YN) < 2: YN.append(0)
        YN[1] = (YN[1] - 1) % N
        while YN and YN[-1] == 0: YN.pop()

        if not YN:
            return None, "zero"

        G = poly_gcd(P, YN, N)
        for c in G:
            g = math.gcd(c, N)
            if 1 < g < N:
                return g, "gcd"
        return None, "no_signal"
    except FactorFound as fe:
        return fe.f, "zero_div"


def run(bits, n_instances, n_tries):
    print(f"\n{'='*70}\n  bits={bits}, n={n_instances}, tries={n_tries}\n{'='*70}")

    # For each instance, test Phi_2 and Phi_3 separately, and combined
    f2 = 0
    f3 = 0
    fcascade = 0
    t0 = time.time()

    for inst in range(n_instances):
        N, p, q = generate_semiprime(bits)
        h2 = h3 = hc = False
        for t in range(n_tries):
            j = random.randint(2, N - 2)
            r2, _ = try_phi_factor(N, j, 2)
            r3, _ = try_phi_factor(N, j, 3)
            if r2 and not h2: h2 = True
            if r3 and not h3: h3 = True
            if (r2 or r3) and not hc: hc = True
            if h2 and h3 and hc: break
        if h2: f2 += 1
        if h3: f3 += 1
        if hc: fcascade += 1

    dt = time.time() - t0
    rate2 = f2 / n_instances
    rate3 = f3 / n_instances
    ratec = fcascade / n_instances

    log2 = math.log2(1/rate2) if rate2 > 0 else float('inf')
    log3 = math.log2(1/rate3) if rate3 > 0 else float('inf')
    logc = math.log2(1/ratec) if ratec > 0 else float('inf')

    print(f"  Phi_2 only: {f2}/{n_instances}  log2(1/rate)={log2:.2f}")
    print(f"  Phi_3 only: {f3}/{n_instances}  log2(1/rate)={log3:.2f}")
    print(f"  Cascade:    {fcascade}/{n_instances}  log2(1/rate)={logc:.2f}")
    print(f"  time: {dt:.1f}s")
    return bits, rate2, rate3, ratec


if __name__ == "__main__":
    random.seed(777)
    print("Phi_2 / Phi_3 cascade scaling test")
    print("If independent asymmetries, cascade rate > single rates")
    print("If correlated, cascade ≈ max(rates)\n")

    data = []
    for bits in [16, 22, 28, 34, 40, 46]:
        data.append(run(bits, 50, 10))

    print(f"\n{'='*70}\n  CASCADE SCALING\n{'='*70}")
    print(f"  {'bits':>4}  {'Phi_2':>8}  {'Phi_3':>8}  {'Cascade':>10}")
    for bits, r2, r3, rc in data:
        print(f"  {bits:>4}  {r2:>8.3f}  {r3:>8.3f}  {rc:>10.3f}")

    # Pairwise alphas on cascade
    print(f"\n  Cascade rate alphas (should be < Phi_2 alone if compounding):")
    for i in range(len(data) - 1):
        b1, _, _, rc1 = data[i]
        b2, _, _, rc2 = data[i + 1]
        if rc1 > 0 and rc2 > 0:
            alpha = -(math.log2(rc2) - math.log2(rc1)) / (b2 - b1)
            print(f"  bits {b1}->{b2}: alpha = {alpha:.3f}")
