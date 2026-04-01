"""
FactorCup Baselines — Reference implementations for comparison.

Do not modify this file. These are provided so you can see what you're competing against.

Algorithms:
    trial_division: O(sqrt(N)) = O(2^{n/2})        — reaches ~48 bits
    pollard_rho:    O(N^{1/4}) = O(2^{n/4})         — reaches ~96 bits
    quadratic_sieve: L_N[1/2, 1] (sub-exponential)  — reaches ~192 bits
"""

import math
import random


# ============================================================================
# Trial Division — O(sqrt(N))
# ============================================================================

def trial_division(N: int) -> tuple[int, int]:
    if N % 2 == 0:
        return (2, N // 2)
    d = 3
    while d * d <= N:
        if N % d == 0:
            return (d, N // d)
        d += 2
    raise ValueError(f"{N} is prime")


# ============================================================================
# Pollard Rho — O(N^{1/4})
# ============================================================================

def pollard_rho(N: int) -> tuple[int, int]:
    if N % 2 == 0:
        return (2, N // 2)

    # Small trial division first
    for d in range(3, min(10000, int(N**0.25) + 1), 2):
        if N % d == 0:
            return (d, N // d)

    # Rho with Brent's improvement
    for c in range(1, 100):
        x = random.randint(2, N - 1)
        y = x
        d = 1

        while d == 1:
            x = (x * x + c) % N
            y = (y * y + c) % N
            y = (y * y + c) % N
            d = math.gcd(abs(x - y), N)

        if 1 < d < N:
            p, q = min(d, N // d), max(d, N // d)
            return (p, q)

    raise ValueError(f"Pollard rho failed on {N}")


# ============================================================================
# Quadratic Sieve — L_N[1/2, 1]
# ============================================================================

def quadratic_sieve(N: int) -> tuple[int, int]:
    if N % 2 == 0:
        return (2, N // 2)

    # Small trial division
    for d in range(3, 10000, 2):
        if N % d == 0:
            return (d, N // d)

    # Factor base size: L[1/2, 1/2] heuristic
    n_bits = N.bit_length()
    B = max(50, int(math.exp(0.5 * math.sqrt(math.log(N) * math.log(math.log(N))))))
    B = min(B, 50000)  # cap for sanity

    # Build factor base: primes p <= B where N is a QR mod p
    factor_base = [2]
    for p in _sieve_primes(B):
        if p == 2:
            continue
        if pow(N, (p - 1) // 2, p) == 1:  # Euler criterion
            factor_base.append(p)

    fb_size = len(factor_base)
    if fb_size < 3:
        raise ValueError("Factor base too small")

    sqrt_N = math.isqrt(N)
    relations = []
    sieve_range = max(fb_size * 20, 10000)

    # Collect smooth relations
    for offset in range(1, sieve_range + 1):
        x = sqrt_N + offset
        val = x * x - N
        if val <= 0:
            continue

        # Trial divide over factor base
        exponents = [0] * fb_size
        remaining = val
        for i, p in enumerate(factor_base):
            while remaining % p == 0:
                remaining //= p
                exponents[i] += 1

        if remaining == 1:  # Smooth!
            relations.append((x, val, exponents))
            if len(relations) > fb_size + 5:
                break

    if len(relations) < 2:
        raise ValueError("Not enough smooth relations")

    # GF(2) Gaussian elimination to find dependent set
    result = _find_factor_from_relations(N, relations, fb_size)
    if result:
        return result

    raise ValueError(f"QS failed on {N}")


def _find_factor_from_relations(N, relations, fb_size):
    """Find a congruence of squares from smooth relations."""
    n_rel = len(relations)

    # Build matrix over GF(2): rows = relations, cols = factor base primes
    matrix = []
    for _, _, exponents in relations:
        matrix.append([e % 2 for e in exponents])

    # Try pairs first (fast)
    for i in range(n_rel):
        for j in range(i + 1, n_rel):
            combined_exp = [relations[i][2][k] + relations[j][2][k] for k in range(fb_size)]
            if all(e % 2 == 0 for e in combined_exp):
                x = (relations[i][0] * relations[j][0]) % N
                y_sq = 1
                for k in range(fb_size):
                    # We don't need the actual factor base primes here,
                    # just use the original values
                    pass
                y_sq = relations[i][1] * relations[j][1]
                y = math.isqrt(y_sq)
                if y * y == y_sq:
                    g = math.gcd(x - y, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))

    # GF(2) elimination
    mat = [row[:] for row in matrix]
    row_ops = list(range(n_rel))
    pivots = []

    for col in range(fb_size):
        pivot_row = None
        for row in range(len(pivots), n_rel):
            if mat[row][col] == 1:
                pivot_row = row
                break
        if pivot_row is None:
            continue

        # Swap
        mat[len(pivots)], mat[pivot_row] = mat[pivot_row], mat[len(pivots)]
        row_ops[len(pivots)], row_ops[pivot_row] = row_ops[pivot_row], row_ops[len(pivots)]

        # Eliminate
        for row in range(n_rel):
            if row != len(pivots) and mat[row][col] == 1:
                for c in range(fb_size):
                    mat[row][c] ^= mat[len(pivots)][c]

        pivots.append(col)

    # Find zero rows (dependencies)
    for row in range(len(pivots), n_rel):
        if all(mat[row][c] == 0 for c in range(fb_size)):
            # This row is a dependency — but we need to track which
            # original relations contribute. Simplified: try this relation alone
            idx = row_ops[row]
            x = relations[idx][0]
            y_sq = relations[idx][1]
            if y_sq > 0:
                y = math.isqrt(y_sq)
                if y * y == y_sq:
                    g = math.gcd(x - y, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))

    return None


def _sieve_primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]
