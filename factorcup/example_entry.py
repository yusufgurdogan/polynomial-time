"""
FactorCup Example Entry — Trial Division.

This is the simplest possible entry. It checks every odd number up to sqrt(N).
Complexity: O(sqrt(N)) = O(2^{n/2}) — exponential in bit size.

Copy this file to entry.py and replace with your algorithm:
    cp example_entry.py entry.py
"""


def factor(N: int) -> tuple[int, int]:
    """Factor N = p * q by trial division."""
    if N % 2 == 0:
        return (2, N // 2)

    d = 3
    while d * d <= N:
        if N % d == 0:
            return (d, N // d)
        d += 2

    raise ValueError(f"N = {N} is prime, not a semiprime")
