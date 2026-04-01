"""
FactorCup Interface — Type definitions and validation.

Do not modify this file.
"""

from sympy import isprime


def validate_factors(N: int, result) -> tuple[bool, str]:
    """
    Validate that result is a correct factorization of N.

    Returns (True, "") on success, (False, reason) on failure.
    """
    if result is None:
        return False, "returned None"

    if not isinstance(result, (tuple, list)) or len(result) != 2:
        return False, f"expected (p, q) tuple, got {type(result)}"

    p, q = result

    if not isinstance(p, int) or not isinstance(q, int):
        return False, f"factors must be int, got ({type(p)}, {type(q)})"

    if p <= 1 or q <= 1:
        return False, f"factors must be > 1, got ({p}, {q})"

    if p * q != N:
        return False, f"p * q = {p * q} != {N}"

    if not isprime(p):
        return False, f"p = {p} is not prime"

    if not isprime(q):
        return False, f"q = {q} is not prime"

    return True, ""


def normalize_factors(p: int, q: int) -> tuple[int, int]:
    """Return (smaller, larger) factor."""
    return (min(p, q), max(p, q))
