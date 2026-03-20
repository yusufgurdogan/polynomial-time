"""
Baseline factoring algorithms for comparison.
These are known algorithms — our experimental approaches need to beat their scaling.
"""

import math
import random
from typing import Optional


def trial_division(n: int) -> Optional[list[int]]:
    """O(sqrt(n)) — exponential in bit-length. The simplest possible baseline."""
    if n < 2:
        return None
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors if factors else None


def pollard_rho(n: int) -> Optional[list[int]]:
    """
    Pollard's rho algorithm. Expected O(n^(1/4)) per factor.
    Sub-exponential but not polynomial in bit-length.
    """
    if n < 2:
        return None
    if n % 2 == 0:
        return _factor_with(n, 2)

    factors = []
    _pollard_rho_recursive(n, factors)
    factors.sort()
    return factors if factors else None


def _pollard_rho_recursive(n: int, factors: list[int]):
    from sympy import isprime
    if n == 1:
        return
    if isprime(n):
        factors.append(n)
        return

    d = _pollard_rho_find_factor(n)
    if d is None:
        # Fallback: try harder
        d = _pollard_rho_find_factor(n, max_iterations=1_000_000)
    if d is None:
        factors.append(n)  # give up, return n as-is
        return

    _pollard_rho_recursive(d, factors)
    _pollard_rho_recursive(n // d, factors)


def _pollard_rho_find_factor(n: int, max_iterations: int = 100_000) -> Optional[int]:
    """Find a single non-trivial factor of n using Brent's improvement."""
    if n % 2 == 0:
        return 2

    for _ in range(10):  # try different random starts
        c = random.randint(1, n - 1)
        x = random.randint(2, n - 1)
        y = x
        d = 1

        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)

            max_iterations -= 1
            if max_iterations <= 0:
                break

        if 1 < d < n:
            return d

    return None


def _factor_with(n: int, small_factor: int) -> list[int]:
    """Factor n knowing one small factor."""
    from sympy import isprime
    factors = []
    while n % small_factor == 0:
        factors.append(small_factor)
        n //= small_factor
    if n > 1:
        if isprime(n):
            factors.append(n)
        else:
            more = pollard_rho(n)
            if more:
                factors.extend(more)
    factors.sort()
    return factors


def quadratic_sieve(n: int) -> Optional[list[int]]:
    """
    Simplified quadratic sieve implementation.
    Sub-exponential: L(1/2, 1) = exp(sqrt(log n * log log n))
    """
    from sympy import isprime
    if n < 2:
        return None
    if isprime(n):
        return [n]
    if n % 2 == 0:
        return _factor_with(n, 2)

    # For small n, fall back to Pollard's rho
    if n.bit_length() < 40:
        return pollard_rho(n)

    factor = _qs_core(n)
    if factor is None or factor == n or factor == 1:
        return pollard_rho(n)  # fallback

    factors = []
    for part in [factor, n // factor]:
        if isprime(part):
            factors.append(part)
        else:
            sub = quadratic_sieve(part)
            if sub:
                factors.extend(sub)
    factors.sort()
    return factors


def _qs_core(n: int) -> Optional[int]:
    """Core quadratic sieve: find a factor of n."""
    sqrt_n = math.isqrt(n)
    if sqrt_n * sqrt_n == n:
        return sqrt_n

    # Determine factor base size
    bits = n.bit_length()
    if bits < 60:
        B = 50
    elif bits < 80:
        B = 100
    elif bits < 100:
        B = 200
    else:
        B = int(math.exp(0.5 * math.sqrt(math.log(n) * math.log(math.log(n)))))
        B = min(B, 5000)

    # Build factor base: small primes where n is a quadratic residue
    factor_base = []
    primes = _small_primes(B)
    for p in primes:
        if p == 2 or pow(n % p, (p - 1) // 2, p) == 1:
            factor_base.append(p)
        if len(factor_base) >= B:
            break

    if not factor_base:
        return None

    fb_size = len(factor_base)
    needed = fb_size + 10  # need more relations than factor base size

    # Sieve: collect (x, Q(x) = (x+sqrt_n)^2 - n) pairs where Q(x) is smooth
    relations = []  # (x_val, exponent_vector)
    x_values = []

    sieve_range = max(fb_size * 20, 10000)

    for x in range(1, sieve_range):
        val = (sqrt_n + x) ** 2 - n
        if val <= 0:
            continue

        # Try to factor val over the factor base
        exponents = []
        remaining = val
        for p in factor_base:
            exp = 0
            while remaining % p == 0:
                remaining //= p
                exp += 1
            exponents.append(exp % 2)  # only care about parity

        if remaining == 1:  # smooth!
            relations.append(((sqrt_n + x), exponents))
            if len(relations) >= needed:
                break

    if len(relations) < 2:
        return None

    # Gaussian elimination over GF(2) to find a subset with all-even exponents
    factor = _solve_and_factor(n, relations, factor_base, sqrt_n)
    return factor


def _solve_and_factor(n, relations, factor_base, sqrt_n):
    """Use gathered relations to try to find a factor."""
    # Try random subsets of relations (simplified vs full Gaussian elimination)
    num_relations = len(relations)

    # First try pairs
    for i in range(num_relations):
        for j in range(i + 1, min(i + 50, num_relations)):
            x_val_i, exp_i = relations[i]
            x_val_j, exp_j = relations[j]

            # Check if combined exponents are all even
            combined = [(exp_i[k] + exp_j[k]) % 2 for k in range(len(factor_base))]
            if all(c == 0 for c in combined):
                # We have x^2 ≡ y^2 (mod n)
                x = (x_val_i * x_val_j) % n

                # Compute y from the combined factorization
                y_sq_exponents = []
                for k in range(len(factor_base)):
                    # Need actual exponents, not just parity
                    pass

                # Simpler: just compute directly
                qi = x_val_i * x_val_i - n
                qj = x_val_j * x_val_j - n
                y_sq = qi * qj
                if y_sq < 0:
                    continue
                y = math.isqrt(abs(y_sq))
                if y * y == abs(y_sq):
                    g = math.gcd(x - y, n)
                    if 1 < g < n:
                        return g
                    g = math.gcd(x + y, n)
                    if 1 < g < n:
                        return g

    # Full Gaussian elimination over GF(2)
    return _gaussian_elimination_factor(n, relations, factor_base)


def _gaussian_elimination_factor(n, relations, factor_base):
    """Full GF(2) Gaussian elimination to find dependencies."""
    num_rels = len(relations)
    fb_size = len(factor_base)

    # Build matrix
    matrix = []
    for _, exps in relations:
        row = exps[:] + [0] * (fb_size - len(exps))
        matrix.append(row)

    # Track which relations are combined
    identity = [[1 if i == j else 0 for j in range(num_rels)] for i in range(num_rels)]

    # Row reduce
    pivot_col = 0
    for col in range(fb_size):
        # Find pivot row
        pivot_row = None
        for row in range(pivot_col, num_rels):
            if matrix[row][col] == 1:
                pivot_row = row
                break
        if pivot_row is None:
            continue

        # Swap
        matrix[pivot_col], matrix[pivot_row] = matrix[pivot_row], matrix[pivot_col]
        identity[pivot_col], identity[pivot_row] = identity[pivot_row], identity[pivot_col]

        # Eliminate
        for row in range(num_rels):
            if row != pivot_col and matrix[row][col] == 1:
                for c in range(fb_size):
                    matrix[row][c] ^= matrix[pivot_col][c]
                for c in range(num_rels):
                    identity[row][c] ^= identity[pivot_col][c]

        pivot_col += 1

    # Find zero rows (dependencies)
    for row in range(num_rels):
        if all(matrix[row][c] == 0 for c in range(fb_size)):
            # This row gives us a dependency
            subset_indices = [j for j in range(num_rels) if identity[row][j] == 1]
            if len(subset_indices) < 2:
                continue

            # Compute x = product of (sqrt_n + x_i) mod n
            x = 1
            for idx in subset_indices:
                x = (x * relations[idx][0]) % n

            # Compute y^2 = product of Q(x_i), then y = sqrt
            y_sq = 1
            for idx in subset_indices:
                qi = relations[idx][0] * relations[idx][0] - n
                y_sq *= qi

            if y_sq < 0:
                continue

            y = math.isqrt(y_sq)
            if y * y != y_sq:
                continue

            y = y % n
            g = math.gcd((x - y) % n, n)
            if 1 < g < n:
                return g
            g = math.gcd((x + y) % n, n)
            if 1 < g < n:
                return g

    return None


def _small_primes(limit: int) -> list[int]:
    """Sieve of Eratosthenes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]
