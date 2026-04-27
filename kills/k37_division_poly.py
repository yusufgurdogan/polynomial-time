#!/usr/bin/env python3
"""
Division polynomial psi_l Frobenius-action factoring.

For E/Z/NZ: y^2 = x^3 + A*x + B, the l-th division polynomial psi_l(x) has
roots = x-coordinates of l-torsion points. Degree (l^2-1)/2 (odd l).

Over F_p, Frobenius acts on E[l] with characteristic polynomial X^2 - a_p X + p.
So in Z/NZ[x]/(psi_l(x)), x^N = x^(pq) satisfies specific algebraic relation
involving a_p mod l. Over Z/NZ, "Schoof-like" computation tries to find single
trace value; mod p and mod q it would be different.

Test: for random E, compute (x^N mod psi_l) - c*x for various c in [0, l-1],
check if any of these polynomial values share factor with N via coefficient GCD.
"""
import math, random, sys, time
from sympy import nextprime
sys.stdout.reconfigure(line_buffering=True)

from k35_isogeny import (
    poly_mul, poly_divmod, poly_gcd, poly_pow_mod,
    FactorFound, generate_semiprime
)


def psi_3(A, B, N):
    """psi_3(x) = 3x^4 + 6Ax^2 + 12Bx - A^2. Return as list [c0, c1, c2, c3, c4]."""
    return [(-A * A) % N, (12 * B) % N, (6 * A) % N, 0, 3 % N]


def psi_5(A, B, N):
    """psi_5(x) is degree 12. Using standard formula.
    psi_5 = psi_2 * psi_3^3 - psi_1 * psi_4^2 ... getting complex.
    Use recursion via psi_{2m+1} = psi_{m+2} psi_m^3 - psi_{m-1} psi_{m+1}^3.

    Return psi_5(x)^2 / y if psi_5 involves y. For our purposes, use psi_5^{odd}
    meaning the polynomial that is psi_5(x) when x is the x-coord of an l-torsion point.
    Actually psi_5 is pure polynomial in x for odd l. Let me compute directly.
    psi_5 = 5x^12 + 62Ax^10 + 380Bx^9 - 105A^2 x^8 + 240ABx^7
          - (300A^3 + 240B^2) x^6 - 696A^2 B x^5 - (125A^4 + 1920AB^2) x^4
          - (80A^3 B + 256B^3) x^3 - (50A^5 + 240A^2 B^2) x^2
          + (100A^4 B - 192AB^3)... this is getting complicated.
    I'll just use psi_3 for the test.
    """
    raise NotImplementedError("psi_5 not needed; use psi_3")


def try_division_poly_factor(N, A, B, l=3):
    """Compute x^N mod psi_l over Z/NZ[x], check for factor via coefficient gcds."""
    try:
        if l == 3:
            P = psi_3(A, B, N)
        else:
            raise ValueError(f"l={l} not supported")

        # Direct coefficient gcd
        for c in P:
            g = math.gcd(c, N)
            if 1 < g < N:
                return g, "direct"

        # Strip
        while P and P[-1] == 0: P.pop()
        if len(P) < 2:
            return None, "degenerate"

        # Compute x^N mod psi_l
        X = [0, 1]  # x
        XN = poly_pow_mod(X, N, P, N)

        # (x^N - c*x) for various c, check if any coef has common factor with N
        for c in range(l):
            candidate = list(XN)
            while len(candidate) < 2: candidate.append(0)
            candidate[1] = (candidate[1] - c) % N
            while candidate and candidate[-1] == 0: candidate.pop()
            if not candidate: continue

            # GCD with psi_l
            try:
                G = poly_gcd(P, candidate, N)
                for cf in G:
                    g = math.gcd(cf, N)
                    if 1 < g < N:
                        return g, f"c={c}_gcd"
            except FactorFound as fe:
                return fe.f, f"c={c}_zerodiv"
        return None, "no_signal"
    except FactorFound as fe:
        return fe.f, "zero_div"


def generate_random_curve(N):
    """Random (A, B) such that 4A^3 + 27B^2 != 0 mod N (non-singular)."""
    for _ in range(20):
        A = random.randint(1, N - 1)
        B = random.randint(1, N - 1)
        disc = (4 * pow(A, 3, N) + 27 * pow(B, 2, N)) % N
        if disc != 0:
            g = math.gcd(disc, N)
            if 1 < g < N:
                # Singular curve, but disc shares factor -- lucky!
                return A, B, g
            return A, B, None
    return None, None, None


def run(bits, n_instances, n_tries):
    print(f"\n{'='*70}\n  bits={bits}, n={n_instances}\n{'='*70}")
    factors = 0
    reasons = {}
    t0 = time.time()

    for inst in range(n_instances):
        N, p, q = generate_semiprime(bits)
        hit = None
        for _ in range(n_tries):
            A, B, lucky = generate_random_curve(N)
            if A is None: continue
            if lucky:
                hit = (lucky, "disc_lucky"); break
            res, reason = try_division_poly_factor(N, A, B, l=3)
            if res:
                hit = (res, reason); break
        if hit:
            factors += 1
            reasons[hit[1]] = reasons.get(hit[1], 0) + 1

    dt = time.time() - t0
    rate = factors / n_instances
    log_inv = -math.log2(rate) if rate > 0 else float('inf')
    print(f"  Div poly psi_3: {factors}/{n_instances}  log2(1/rate)={log_inv:.2f}  reasons={reasons}  ({dt:.1f}s)")
    return bits, rate


if __name__ == "__main__":
    random.seed(2024)
    print("Division polynomial psi_3 Frobenius-action factoring\n")

    data = []
    for bits in [16, 20, 24, 28, 32, 36, 40]:
        data.append(run(bits, 30, 15))

    print(f"\n  Scaling:")
    for i in range(len(data) - 1):
        b1, r1 = data[i]
        b2, r2 = data[i + 1]
        if r1 > 0 and r2 > 0:
            alpha = -(math.log2(r2) - math.log2(r1)) / (b2 - b1)
            print(f"  bits {b1}->{b2}: alpha = {alpha:.3f}")
