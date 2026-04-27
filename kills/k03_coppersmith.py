#!/usr/bin/env python3
"""
Coppersmith's method — proper implementation following Howgrave-Graham.

The goal: given f(x) monic of degree d with a small root x₀ modulo
an unknown divisor b of N (where b ≥ N^β), find x₀.

For factoring: f(x) = x + a where a ≈ p, b = p, N = n.
Recovers p if |p - a| < N^(β²/d).

For balanced semiprimes (β = 0.5, d = 1): bound is N^0.25.
"""

import math
from fpylll import IntegerMatrix, LLL, BKZ
from typing import Optional


def coppersmith_smallroot(f_coeffs: list[int], N: int, X: int,
                          beta: float = 1.0, m: int = None) -> list[int]:
    """
    Find all small integer roots of f(x) modulo some factor b of N,
    where b >= N^beta and |root| <= X.

    f_coeffs: polynomial coefficients [a₀, a₁, ..., aₐ] for f(x) = a₀ + a₁x + ... + aₐx^d
              MUST be monic (leading coefficient = 1)
    N: modulus (composite we want to factor)
    X: bound on absolute value of root
    beta: b >= N^beta for the unknown divisor b
    m: lattice parameter (auto-chosen if None)

    Returns list of integer roots found.
    """
    d = len(f_coeffs) - 1
    if d <= 0:
        return []

    # Ensure monic
    assert f_coeffs[-1] == 1, "f must be monic"

    # Auto-choose m
    if m is None:
        # Heuristic: m = ceil(beta^2 / (d * epsilon)) where epsilon is small
        # Larger m = better bound but bigger lattice
        # For practical purposes, m ~ 4-7 works well
        m = max(1, int(math.ceil(7 * beta / d)))
        m = min(m, 12)

    t = max(1, int(d * m * (1.0 / beta - 1)))
    t = min(t, m)

    # Construct shifted polynomials
    # g_{i,j}(x) = x^j * N^(m-i) * f(x)^i  for i = 0..m, j = 0..d-1
    # h_l(x)     = x^l * f(x)^m             for l = 0..t-1
    #
    # These all vanish mod b^m at the root x₀.
    # We evaluate at xX (scale variable) so the lattice geometry works.

    polys = []

    # g polynomials
    for i in range(m + 1):
        for j in range(d):
            # g_{i,j}(x) = x^j * f(x)^i * N^(m-i)
            p = _poly_mul_scalar([0] * j + [1], 1)  # x^j
            fi = _poly_power(f_coeffs, i)             # f(x)^i
            p = _poly_mul(p, fi)
            p = _poly_mul_scalar(p, N ** (m - i))
            polys.append(p)

    # h polynomials
    fm = _poly_power(f_coeffs, m)  # f(x)^m
    for l in range(t):
        p = _poly_mul([0] * l + [1], fm)  # x^l * f(x)^m
        polys.append(p)

    # Build lattice matrix
    # Each polynomial evaluated at x*X gives a row
    # Columns correspond to powers of x: x^0, x^1, ..., x^(dim-1)
    dim = len(polys)
    max_deg = max(len(p) for p in polys)
    dim = max(dim, max_deg)

    B = IntegerMatrix(dim, dim)

    for row_idx, poly in enumerate(polys):
        for j, coeff in enumerate(poly):
            if j < dim:
                # Scale x^j by X^j
                B[row_idx, j] = int(coeff * (X ** j))

    # LLL reduce
    LLL.reduction(B)

    # Extract roots from short vectors (Howgrave-Graham's theorem)
    # A short vector corresponds to a polynomial h(x) that's small at x₀.
    # If ||h(xX)|| < b^m / sqrt(dim), then h(x₀) = 0 over Z.
    roots = []

    for row_idx in range(dim):
        # Unscale: column j was multiplied by X^j
        poly = []
        valid = True
        for j in range(dim):
            val = int(B[row_idx, j])
            if j == 0:
                poly.append(val)
            elif X ** j == 0:
                valid = False
                break
            elif val % (X ** j) == 0:
                poly.append(val // (X ** j))
            else:
                # Not exactly divisible — this row doesn't give a clean polynomial
                valid = False
                break

        if not valid or all(c == 0 for c in poly):
            continue

        # Find integer roots of this polynomial in [-X, X]
        for root in _integer_roots(poly, X):
            if root not in roots:
                roots.append(root)

    return roots


def factor_coppersmith(n: int, p_approx: int, bound: int) -> Optional[int]:
    """
    Factor n given an approximation p_approx to a factor p.
    |p - p_approx| must be < bound.

    Returns the factor p, or None.
    """
    if bound < 1:
        return None

    g = math.gcd(p_approx, n)
    if 1 < g < n:
        return g

    # f(x) = x + p_approx, monic degree 1
    # Has root (p - p_approx) mod p
    f = [p_approx, 1]  # p_approx + 1*x

    # Try various m values
    for m_val in [4, 6, 8, 12, 16, 20]:
        roots = coppersmith_smallroot(f, n, bound, beta=0.49, m=m_val)
        for root in roots:
            candidate = p_approx + root
            if 1 < candidate < n and n % candidate == 0:
                return candidate

    return None


def factor_coppersmith_with_modular_info(n: int, p_mod: int, modulus: int) -> Optional[int]:
    """
    Factor n given that p ≡ p_mod (mod modulus) for a factor p.

    Constructs f(x) = p_mod + modulus*x, which has root (p - p_mod)/modulus mod p.
    The bound on the root is sqrt(n) / modulus.
    """
    sqrt_n = math.isqrt(n)
    root_bound = sqrt_n // modulus + 1

    # If small enough, just search directly
    if root_bound < 10 ** 7:
        candidate = p_mod
        while candidate <= sqrt_n + modulus:
            if candidate > 1 and n % candidate == 0:
                return candidate
            candidate += modulus
        return None

    # f(x) = (p_mod + modulus * x), root mod p is (p - p_mod) / modulus
    # We need to make this monic: divide by modulus
    # g(x) = x + p_mod/modulus ... but p_mod/modulus isn't integer in general
    # Instead: f(x) = p_mod + modulus*x has root mod p, but isn't monic
    # Trick: set f(x) = x + (p_mod * modulus_inv mod n) where modulus_inv = modulus^(-1) mod n
    # Then f has root (p - p_mod) * modulus_inv mod p ... no, this changes the root.

    # Better approach: use the lattice directly for non-monic case
    # f(x) = modulus * x + p_mod, root x₀ = (p - p_mod)/modulus mod p
    # The root x₀ < sqrt(n)/modulus

    # Build lattice manually
    X = root_bound

    for m_val in [4, 6, 8]:
        dim = m_val + 1
        B = IntegerMatrix(dim, dim)

        # Row 0: [n^m, 0, 0, ...]
        # Row 1: [p_mod * n^(m-1), modulus * X * n^(m-1), 0, ...]
        # Row i: coefficients of n^(m-i) * f(xX)^i
        for i in range(dim):
            # n^(m-i) * (modulus * xX + p_mod)^i
            n_pow = n ** max(0, m_val - i) if i <= m_val else 1
            for k in range(i + 1):
                if k >= dim:
                    break
                coeff = math.comb(i, k) * (modulus ** k) * (X ** k) * (p_mod ** (i - k)) * n_pow
                B[i, k] = int(coeff)

        try:
            LLL.reduction(B)
        except Exception:
            continue

        for row in range(dim):
            for col in range(dim):
                val = abs(int(B[row, col]))
                if val > 1:
                    g = math.gcd(val, n)
                    if 1 < g < n:
                        return g

    return None


# =============================================================================
# Partial information extraction + Coppersmith pipeline
# =============================================================================

def extract_partial_info(n: int) -> dict:
    """
    Extract all polynomial-time-computable partial info about a factor p.
    Returns {prime_modulus: set of possible residues for p}.
    """
    info = {}
    bits = n.bit_length()

    for m in _small_primes(min(bits * 5, 300)):
        if n % m == 0:
            info[m] = {n % m}  # degenerate
            continue
        n_mod = n % m
        possible = set()
        for r in range(1, m):
            try:
                q_mod = (n_mod * pow(r, -1, m)) % m
            except ValueError:
                continue
            if q_mod > 0:
                possible.add(r)
        if len(possible) < m - 1:  # only keep if it actually constrains
            info[m] = possible

    return info


def try_coppersmith_pipeline(n: int, verbose: bool = False) -> Optional[list[int]]:
    """
    Full pipeline:
    1. Extract modular constraints on p
    2. CRT-combine to narrow the search
    3. Use Coppersmith to close the gap
    """
    from sympy import isprime

    if n % 2 == 0:
        return _factorize(n, 2)

    bits = n.bit_length()
    sqrt_n = math.isqrt(n)

    # Step 1: gather constraints
    info = extract_partial_info(n)

    if verbose:
        useful = {k: v for k, v in info.items() if len(v) < k - 1}
        print(f"  {len(info)} modular constraints")

    # Step 2: CRT combination with pruning
    # Process primes from smallest to largest
    sorted_primes = sorted(info.keys())

    candidates = [(r, sorted_primes[0]) for r in info[sorted_primes[0]]] if sorted_primes else []

    for prime in sorted_primes[1:]:
        residues = info[prime]
        new_candidates = []

        for r_crt, m_crt in candidates:
            for r_new in residues:
                g = math.gcd(m_crt, prime)
                if (r_new - r_crt) % g != 0:
                    continue
                lcm = m_crt * prime // g
                try:
                    inv = pow(m_crt // g, -1, prime // g)
                except ValueError:
                    continue
                combined = (r_crt + m_crt * ((r_new - r_crt) // g * inv % (prime // g))) % lcm
                new_candidates.append((combined, lcm))

        candidates = new_candidates

        # Prune: candidates where p > sqrt(n) are for q, not p
        candidates = [(r, m) for r, m in candidates if r <= sqrt_n + m]

        if len(candidates) > 50000:
            candidates.sort(key=lambda x: -x[1])
            candidates = candidates[:50000]

        # Check: can we directly search?
        best_mod = max(m for _, m in candidates) if candidates else 0
        search_per = (sqrt_n // best_mod + 1) if best_mod > 0 else float('inf')

        if verbose and prime <= 13:
            print(f"  After p={prime}: {len(candidates)} candidates, best modulus {best_mod}, search/candidate ~{search_per}")

        # If search is small enough, go for it
        if search_per < bits ** 3 and len(candidates) < bits ** 2:
            for r, m in candidates:
                cand = r
                while cand <= sqrt_n + m:
                    if cand > 1 and n % cand == 0:
                        return _factorize(n, cand)
                    cand += m
            return None

        # If modulus > n^0.25, try Coppersmith
        n_quarter = int(n ** 0.25)
        if best_mod > n_quarter:
            if verbose:
                print(f"  Modulus {best_mod} > N^0.25 = {n_quarter}, trying Coppersmith...")
            for r, m in candidates[:100]:
                factor = factor_coppersmith_with_modular_info(n, r, m)
                if factor:
                    return _factorize(n, factor)

    # Step 3: Fallback — try Coppersmith from sqrt(n) approximation
    factor = factor_coppersmith(n, sqrt_n, int(n ** 0.25) + 1)
    if factor:
        return _factorize(n, factor)

    return None


# =============================================================================
# Polynomial arithmetic helpers
# =============================================================================

def _poly_mul(a, b):
    """Multiply two polynomials (coefficient lists)."""
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def _poly_mul_scalar(p, s):
    """Multiply polynomial by scalar."""
    return [c * s for c in p]


def _poly_power(p, k):
    """Raise polynomial to k-th power."""
    if k == 0:
        return [1]
    if k == 1:
        return p[:]
    result = [1]
    for _ in range(k):
        result = _poly_mul(result, p)
    return result


def _integer_roots(coeffs, bound):
    """Find integer roots of polynomial in [-bound, bound]."""
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]

    if not coeffs:
        return []
    if len(coeffs) == 1:
        return [0] if coeffs[0] == 0 else []

    roots = []

    # Linear case: a + b*x = 0 → x = -a/b
    if len(coeffs) == 2:
        a, b = coeffs
        if b != 0 and a % b == 0:
            r = -a // b
            if abs(r) <= bound:
                roots.append(r)
        return roots

    # Quadratic case: a + b*x + c*x² = 0
    if len(coeffs) == 3:
        a, b, c = coeffs
        if c != 0:
            disc = b * b - 4 * a * c
            if disc >= 0:
                sqrt_disc = math.isqrt(disc)
                if sqrt_disc * sqrt_disc == disc:
                    for sign in [1, -1]:
                        num = -b + sign * sqrt_disc
                        if num % (2 * c) == 0:
                            r = num // (2 * c)
                            if abs(r) <= bound and r not in roots:
                                roots.append(r)
        return roots

    # General case: use numerical root finding then verify
    # Convert to float polynomial, find approximate roots, round and check
    try:
        import numpy as np
        # numpy can handle moderate-degree polynomials
        # coeffs are [a0, a1, ..., an], numpy wants [an, ..., a1, a0]
        np_coeffs = [float(c) for c in reversed(coeffs)]
        approx_roots = np.roots(np_coeffs)
        for r in approx_roots:
            if abs(r.imag) < 0.5:  # real root
                candidate = int(round(r.real))
                if abs(candidate) <= bound:
                    # Verify exactly
                    val = 0
                    xk = 1
                    for c in coeffs:
                        val += c * xk
                        xk *= candidate
                    if val == 0 and candidate not in roots:
                        roots.append(candidate)
                    # Also check neighbors (rounding error)
                    for delta in [-1, 1]:
                        cand2 = candidate + delta
                        if abs(cand2) <= bound:
                            val = 0
                            xk = 1
                            for c in coeffs:
                                val += c * xk
                                xk *= cand2
                            if val == 0 and cand2 not in roots:
                                roots.append(cand2)
    except (np.linalg.LinAlgError, ValueError, OverflowError):
        pass

    # Fallback: check x = 0
    if coeffs[0] == 0 and 0 not in roots:
        roots.append(0)

    # Fallback: rational root theorem for small constant terms
    if coeffs[0] != 0 and abs(coeffs[0]) < 10**15:
        c0 = abs(coeffs[0])
        for d in _divisors(c0, min(bound + 1, 10000)):
            for sign in [1, -1]:
                x = sign * d
                if abs(x) > bound or x in roots:
                    continue
                val = 0
                xk = 1
                for c in coeffs:
                    val += c * xk
                    xk *= x
                if val == 0:
                    roots.append(x)

    return roots


def _divisors(n, limit):
    """Small divisors of n up to limit."""
    divs = set()
    isqrt = math.isqrt(n)
    for d in range(1, min(isqrt + 2, limit + 1)):
        if n % d == 0:
            divs.add(d)
            if n // d <= limit:
                divs.add(n // d)
    return sorted(divs)


def _small_primes(limit):
    if limit < 2:
        return []
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            for j in range(i * i, limit + 1, i):
                s[j] = False
    return [i for i in range(limit + 1) if s[i]]


def _factorize(n, f):
    from sympy import isprime
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


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys, random, time
    sys.stdout.reconfigure(line_buffering=True)
    random.seed(42)

    from harness import generate_semiprime, verify_factors

    print("=" * 70)
    print("  COPPERSMITH PIPELINE — PROPER IMPLEMENTATION")
    print("=" * 70)

    # Test 1: Coppersmith with known good approximation
    print("\n--- Coppersmith with known partial info ---")
    for bits in [32, 48, 64, 80, 96, 128, 160, 192, 256]:
        n, p, q = generate_semiprime(bits)
        bound = int(n ** 0.24) + 1  # slightly under N^0.25
        noise = random.randint(0, min(bound - 1, p - 2))
        p_approx = p - noise

        t0 = time.time()
        result = factor_coppersmith(n, p_approx, bound)
        elapsed = time.time() - t0

        if result and n % result == 0:
            print(f"  {bits:3d} bits: OK  ({elapsed:.3f}s) [noise={noise}, bound={bound}]")
        else:
            print(f"  {bits:3d} bits: FAIL ({elapsed:.3f}s) [noise={noise}, bound={bound}]")

    # Test 2: Full pipeline (no prior info)
    print("\n--- Full pipeline (extract info from n) ---")
    for bits in [16, 20, 24, 28, 32, 36, 40, 48]:
        n, p, q = generate_semiprime(bits)
        t0 = time.time()
        result = try_coppersmith_pipeline(n, verbose=(bits <= 20))
        elapsed = time.time() - t0

        if result and verify_factors(n, result):
            print(f"  {bits:3d} bits: SUCCESS in {elapsed:.3f}s")
        else:
            print(f"  {bits:3d} bits: FAILED in {elapsed:.3f}s")
            if elapsed > 60:
                print("  (stopping — too slow)")
                break
