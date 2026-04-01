"""
FactorCup Test Case Generator — Deterministic semiprime generation.

Do not modify this file.
"""

import random
from sympy import nextprime


def generate_semiprime(bits: int, seed: int) -> tuple[int, int, int]:
    """
    Generate a semiprime N = p * q with approximately `bits` bits.

    Uses deterministic seeding for reproducibility.

    Returns: (N, p, q) with p <= q.
    """
    rng = random.Random(seed)
    half = bits // 2

    lo = 1 << (half - 1)
    hi = (1 << half) - 1

    for _ in range(1000):
        p = nextprime(rng.randint(lo, hi))
        if p > hi:
            continue
        q = nextprime(rng.randint(lo, hi))
        if q > hi or q == p:
            continue
        N = p * q
        if N.bit_length() >= bits - 1 and N.bit_length() <= bits + 1:
            p, q = min(p, q), max(p, q)
            return N, p, q

    raise RuntimeError(f"Failed to generate {bits}-bit semiprime with seed {seed}")


def generate_test_suite(bits: int, count: int = 5, base_seed: int = 42) -> list:
    """
    Generate `count` semiprimes at a given bit size.

    Returns list of (N, p, q) tuples. Deterministic given base_seed.
    """
    cases = []
    for i in range(count):
        seed = base_seed * 1000000 + bits * 1000 + i
        cases.append(generate_semiprime(bits, seed))
    return cases


# Standard bit sizes for scoring
SCORING_BIT_SIZES = [
    32, 48, 64, 80, 96, 112, 128,
    160, 192, 224, 256,
    320, 384, 448, 512,
    640, 768, 896, 1024,
]

CASES_PER_SIZE = 5
TIMEOUT_SECONDS = 60
ALIVE_THRESHOLD = 3  # must solve at least 3/5 to be "alive"
