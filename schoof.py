#!/usr/bin/env python3
"""
Schoof's algorithm mod p vs mod N — studying where factoring hides.

The plan:
1. Implement elliptic curve arithmetic over Z/mZ for arbitrary m
2. Run Schoof-like computations mod p (prime) — should work
3. Run the same computations mod N = pq (composite) — will break
4. Study EXACTLY where it breaks: the inversion failures ARE factors

The key insight (Dieulefait-Urroz): factoring N is equivalent to
counting points on elliptic curves mod N. Schoof counts points mod p
in polynomial time. The gap between "mod p" and "mod pq" is where
the factoring hardness lives.

When we compute in E[ℓ] (ℓ-torsion) over Z/NZ, we need to:
- Compute division polynomials mod N
- Invert elements mod N (to do point arithmetic)
- If an inversion fails, gcd with N gives a factor!

The question: can we engineer curves E and torsion primes ℓ such that
inversion failure happens quickly (polynomial many attempts)?
"""

import math
import random
import time
import sys
from typing import Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# Elliptic curve arithmetic over Z/mZ
# =============================================================================
# Curve: y² = x³ + ax + b over Z/mZ
# Points: (x, y) or None (point at infinity)
# When m is composite, inversion can fail — this reveals factors!

class InversionFailure(Exception):
    """Raised when modular inversion fails, revealing a factor."""
    def __init__(self, factor):
        self.factor = factor
        super().__init__(f"Found factor: {factor}")


def mod_inv(a: int, m: int) -> int:
    """Compute a^(-1) mod m. Raises InversionFailure if gcd(a,m) != 1."""
    g = math.gcd(a % m, m)
    if g == 1:
        return pow(a, -1, m)
    elif 1 < g < m:
        raise InversionFailure(g)
    else:
        raise InversionFailure(m)  # a ≡ 0


def ec_add(P, Q, a: int, m: int) -> Optional[Tuple[int, int]]:
    """Add points P, Q on y² = x³ + ax + b over Z/mZ."""
    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    if x1 % m == x2 % m:
        if (y1 + y2) % m == 0:
            return None  # P + (-P) = O
        if y1 % m == y2 % m:
            # Point doubling: λ = (3x₁² + a) / (2y₁)
            num = (3 * x1 * x1 + a) % m
            den = (2 * y1) % m
            lam = (num * mod_inv(den, m)) % m
        else:
            # Shouldn't happen if y1 ≡ y2 but y1 ≢ -y2
            num = (y2 - y1) % m
            den = (x2 - x1) % m
            lam = (num * mod_inv(den, m)) % m
    else:
        # Standard addition: λ = (y₂ - y₁) / (x₂ - x₁)
        num = (y2 - y1) % m
        den = (x2 - x1) % m
        lam = (num * mod_inv(den, m)) % m

    x3 = (lam * lam - x1 - x2) % m
    y3 = (lam * (x1 - x3) - y1) % m
    return (x3, y3)


def ec_mul(k: int, P, a: int, m: int):
    """Scalar multiplication [k]P on y² = x³ + ax + b over Z/mZ."""
    if k == 0 or P is None:
        return None
    if k < 0:
        k = -k
        P = (P[0], (-P[1]) % m)

    result = None
    base = P
    while k > 0:
        if k & 1:
            result = ec_add(result, base, a, m)
        base = ec_add(base, base, a, m)
        k >>= 1
    return result


# =============================================================================
# Division polynomials (for computing ℓ-torsion)
# =============================================================================
# ψ_ℓ(x) are the division polynomials. P ∈ E[ℓ] iff ψ_ℓ(P) = 0.
# We work with them as polynomials mod (y² - x³ - ax - b) over Z/mZ.

def division_poly(ell: int, x, a: int, b: int, m: int) -> int:
    """
    Evaluate the ℓ-th division polynomial at x over Z/mZ.
    Returns ψ_ℓ(x) mod m (up to factors of y which we handle separately).

    For Schoof, we need ψ_ℓ mod the curve equation.
    This simplified version computes ψ_ℓ(x) for the x-coordinate part.
    """
    # Base cases (ψ values, omitting y factors for odd ℓ)
    # ψ₁ = 1
    # ψ₂ = 2y (we track the "y-free" part)
    # ψ₃ = 3x⁴ + 6ax² + 12bx - a²
    # ψ₄ = 4y(x⁶ + 5ax⁴ + 20bx³ - 5a²x² - 4abx - 8b² - a³)

    if ell == 0:
        return 0
    if ell == 1:
        return 1
    if ell == 2:
        return 1  # the 2y factor handled separately

    x2 = (x * x) % m
    x3 = (x2 * x) % m
    x4 = (x3 * x) % m
    f = (x3 + a * x + b) % m  # y² = f(x)

    if ell == 3:
        return (3 * x4 + 6 * a * x2 + 12 * b * x - a * a) % m
    if ell == 4:
        x6 = (x3 * x3) % m
        return (2 * (x6 + 5*a*x4 + 20*b*x3 - 5*a*a*x2 - 4*a*b*x - 8*b*b - a*a*a)) % m

    # For larger ℓ, use recurrence relations:
    # For odd n = 2k+1:
    #   ψ_{2k+1} = ψ_{k+2}·ψ_k³ - ψ_{k-1}·ψ_{k+1}³
    # For even n = 2k:
    #   ψ_{2k} = ψ_k·(ψ_{k+2}·ψ_{k-1}² - ψ_{k-2}·ψ_{k+1}²) / (2y)

    # Build up using a cache
    cache = {}
    cache[0] = 0
    cache[1] = 1
    cache[2] = 1  # 2y factor tracked separately
    cache[3] = (3 * x4 + 6 * a * x2 + 12 * b * x - a * a) % m
    cache[4] = (2 * ((x * x % m) * (x * x % m) % m * (x * x % m) % m
                     + 5*a*x4 + 20*b*x3 - 5*a*a*x2 - 4*a*b*x - 8*b*b - a*a*a)) % m

    def psi(n):
        if n in cache:
            return cache[n]

        k = n // 2
        if n % 2 == 1:
            # ψ_{2k+1} = ψ_{k+2}·ψ_k³ - ψ_{k-1}·ψ_{k+1}³
            # But we need to account for y factors properly
            pk2 = psi(k + 2)
            pk = psi(k)
            pkm1 = psi(k - 1)
            pk1 = psi(k + 1)

            pk3 = (pk * pk % m * pk) % m
            pk13 = (pk1 * pk1 % m * pk1) % m

            if k % 2 == 1:
                # k odd: ψ_{k+2} and ψ_{k-1} have y factors, ψ_k and ψ_{k+1} don't
                # ψ_{k+2}·ψ_k³ has f factor, ψ_{k-1}·ψ_{k+1}³ has f factor
                result = (pk2 * pk3 * f - pkm1 * pk13 * f) % m
            else:
                # k even: ψ_{k+2} and ψ_{k-1} don't have y factors, ψ_k and ψ_{k+1} have them
                result = (pk2 * pk3 * f * f % m * f % m - pkm1 * pk13) % m

            # Simplified: just do the standard recurrence treating everything mod m
            # This loses the y-factor tracking but still gives useful results
            result = (pk2 * pk3 - pkm1 * pk13) % m
        else:
            # ψ_{2k} involves division by 2y — just compute numerator
            pk2 = psi(k + 2)
            pkm1 = psi(k - 1)
            pkm2 = psi(k - 2)
            pk1 = psi(k + 1)
            pk = psi(k)

            pkm12 = (pkm1 * pkm1) % m
            pk12 = (pk1 * pk1) % m

            result = (pk * (pk2 * pkm12 - pkm2 * pk12)) % m

        cache[n] = result % m
        return result % m

    return psi(ell)


# =============================================================================
# Schoof-like computation: trace of Frobenius mod ℓ
# =============================================================================

def frobenius_trace_mod_ell(a_curve: int, b_curve: int, m: int, ell: int) -> Optional[int]:
    """
    Compute the trace of Frobenius t mod ℓ for the curve y² = x³ + ax + b
    over Z/mZ.

    For m = prime p: this gives t mod ℓ where |E(F_p)| = p + 1 - t.
    For m = N = pq: this SHOULD fail (revealing factors) because the
    Frobenius isn't well-defined over Z/NZ.

    Returns t mod ℓ, or None if computation fails.
    Raises InversionFailure if a factor is found!
    """
    # For each candidate t in {0, 1, ..., ℓ-1}, check:
    # (x^m², y^m²) + [t](x^m, y^m) = [m mod ℓ](x, y)  on E[ℓ]
    #
    # Simplified approach for small ℓ:
    # Pick a random point P on E, compute [m+1]P and check if it
    # matches [t] applied to Frobenius.
    #
    # Even simpler for our purposes: compute [m+1-t]P for each candidate t
    # and see which gives O (point at infinity).

    # Find a random point on the curve over Z/mZ
    for _ in range(100):
        x = random.randint(0, m - 1)
        rhs = (x * x * x + a_curve * x + b_curve) % m

        # Check if rhs is a QR mod m (for composite m, this is tricky)
        # Just try to compute sqrt via Tonelli-Shanks-like approach
        y = _sqrt_mod(rhs, m)
        if y is not None and (y * y) % m == rhs:
            P = (x, y)
            break
    else:
        return None  # couldn't find a point

    # For each candidate t, check if [m + 1 - t]P = O
    for t_candidate in range(ell):
        k = (m + 1 - t_candidate) % ell
        if k == 0:
            # Check if [ell]P = O (P is in ℓ-torsion)
            try:
                Q = ec_mul(ell, P, a_curve, m)
                if Q is None:
                    return t_candidate
            except InversionFailure:
                raise  # propagate factor!
            continue

        try:
            # Compute [m + 1 - t_candidate]P
            exp = m + 1 - t_candidate
            Q = ec_mul(exp, P, a_curve, m)
            if Q is None:
                return t_candidate
        except InversionFailure:
            raise  # THIS IS THE INTERESTING CASE — factor found!

    return None


# =============================================================================
# Main experiment: Schoof mod p vs mod N
# =============================================================================

def experiment_schoof_mod_p(p: int, num_curves: int = 5):
    """Run Schoof on curves mod a prime p. Should always work."""
    print(f"\n  Schoof mod p={p} ({p.bit_length()} bits)")
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23]

    for _ in range(num_curves):
        # Random curve
        a = random.randint(1, p - 1)
        b = random.randint(1, p - 1)
        disc = (4 * a * a * a + 27 * b * b) % p
        if disc == 0:
            continue

        traces = {}
        for ell in small_primes:
            if ell == p:
                continue
            try:
                t = frobenius_trace_mod_ell(a, b, p, ell)
                if t is not None:
                    traces[ell] = t
            except InversionFailure as e:
                print(f"    UNEXPECTED: inversion failure mod prime! factor={e.factor}")
            except Exception:
                pass

        if traces:
            print(f"    Curve a={a}, b={b}: traces mod ℓ = {traces}")


def experiment_schoof_mod_n(n: int, p: int, q: int, num_curves: int = 20,
                            max_ell: int = 30):
    """
    Run Schoof on curves mod N = pq. Study where it breaks.
    Every InversionFailure reveals a factor!
    """
    print(f"\n  Schoof mod N={n} ({n.bit_length()} bits) = {p} × {q}")
    small_primes = [ell for ell in [3, 5, 7, 11, 13, 17, 19, 23, 29]
                    if ell <= max_ell and ell != p and ell != q]

    factors_found = 0
    total_attempts = 0
    inversion_points = []  # track WHERE inversions fail

    for curve_idx in range(num_curves):
        a = random.randint(1, n - 1)
        b = random.randint(1, n - 1)
        disc = (4 * a * a * a + 27 * b * b) % n
        if disc == 0:
            continue

        for ell in small_primes:
            total_attempts += 1
            try:
                t = frobenius_trace_mod_ell(a, b, n, ell)
                # If this SUCCEEDS mod N, that's actually interesting too
                # It means the computation didn't hit a non-invertible element
            except InversionFailure as e:
                factor = e.factor
                if 1 < factor < n and n % factor == 0:
                    factors_found += 1
                    inversion_points.append({
                        'curve': (a, b),
                        'ell': ell,
                        'factor': factor,
                        'attempt': total_attempts,
                    })
                    if factors_found <= 5:
                        print(f"    FACTOR at attempt {total_attempts}: "
                              f"curve=({a},{b}), ℓ={ell}, factor={factor}")
            except Exception:
                pass

    print(f"\n    Results: {factors_found} factors found in {total_attempts} attempts")
    if total_attempts > 0:
        print(f"    Success rate: {factors_found/total_attempts:.1%}")
    if inversion_points:
        ells = [ip['ell'] for ip in inversion_points]
        from collections import Counter
        ell_counts = Counter(ells)
        print(f"    Factors by ℓ: {dict(ell_counts)}")

    return factors_found, total_attempts


def experiment_scaling(bit_sizes=None):
    """
    The key experiment: how does the success rate of finding factors
    via Schoof-mod-N scale with the bit size of N?

    If it stays constant (or grows) → potential polynomial-time algorithm!
    If it drops exponentially → this approach is sub-exponential at best.
    """
    from harness import generate_semiprime

    if bit_sizes is None:
        bit_sizes = [16, 20, 24, 28, 32, 40, 48, 56, 64]

    print("=" * 70)
    print("  SCHOOF MOD N — SCALING EXPERIMENT")
    print("  Question: does factoring success rate scale polynomially?")
    print("=" * 70)

    results = []

    for bits in bit_sizes:
        n, p, q = generate_semiprime(bits)
        num_curves = max(20, bits * 2)
        max_ell = min(30, p - 1)  # ℓ must be smaller than p

        t0 = time.time()
        found, attempts = experiment_schoof_mod_n(n, p, q,
                                                  num_curves=num_curves,
                                                  max_ell=max_ell)
        elapsed = time.time() - t0

        rate = found / attempts if attempts > 0 else 0
        results.append((bits, found, attempts, rate, elapsed))

        if elapsed > 60:
            print(f"\n  Stopping — too slow at {bits} bits")
            break

    print(f"\n{'='*70}")
    print(f"  SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'bits':>5} | {'found':>6} | {'attempts':>8} | {'rate':>8} | {'time':>8}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for bits, found, attempts, rate, elapsed in results:
        print(f"  {bits:5d} | {found:6d} | {attempts:8d} | {rate:7.1%} | {elapsed:7.2f}s")


def _sqrt_mod(a: int, m: int) -> Optional[int]:
    """Compute sqrt(a) mod m. Returns None if not a QR."""
    a = a % m
    if a == 0:
        return 0

    # For prime m, use Tonelli-Shanks
    if m % 4 == 3:
        r = pow(a, (m + 1) // 4, m)
        if (r * r) % m == a:
            return r
        return None

    # For composite m or m ≡ 1 mod 4, try random approach
    # (This is intentionally simple — we're studying Schoof, not optimizing sqrt)
    for _ in range(50):
        r = random.randint(1, m - 1)
        if (r * r) % m == a:
            return r

    # Tonelli-Shanks for general case
    if pow(a, (m - 1) // 2, m) != 1:
        return None

    Q, S = m - 1, 0
    while Q % 2 == 0:
        Q //= 2
        S += 1

    z = 2
    while pow(z, (m - 1) // 2, m) != m - 1:
        z += 1
        if z > 100:
            return None

    M = S
    c = pow(z, Q, m)
    t = pow(a, Q, m)
    R = pow(a, (Q + 1) // 2, m)

    while True:
        if t == 1:
            return R
        i = 1
        temp = (t * t) % m
        while temp != 1:
            temp = (temp * temp) % m
            i += 1
            if i >= M:
                return None
        b = pow(c, 1 << (M - i - 1), m)
        M = i
        c = (b * b) % m
        t = (t * c) % m
        R = (R * b) % m


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    # Test 1: Schoof mod prime (sanity check)
    print("=" * 70)
    print("  TEST 1: Schoof mod prime (should work)")
    print("=" * 70)
    from sympy import nextprime
    experiment_schoof_mod_p(nextprime(1000), num_curves=3)

    # Test 2: Schoof mod small composite (study breakage)
    print("\n" + "=" * 70)
    print("  TEST 2: Schoof mod N = pq (study where it breaks)")
    print("=" * 70)
    experiment_schoof_mod_n(15, 3, 5, num_curves=10)
    experiment_schoof_mod_n(77, 7, 11, num_curves=10)
    experiment_schoof_mod_n(1003, 17, 59, num_curves=20)

    # Test 3: Scaling experiment
    print()
    experiment_scaling([16, 20, 24, 28, 32, 40, 48])
