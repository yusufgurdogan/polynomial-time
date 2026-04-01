# FactorCup

**The fastest classical integer factoring algorithm wins.**

A competition to find a polynomial-time classical factoring algorithm — or prove the current barriers are fundamental.

## Summary

A classical integer factoring algorithm takes a semiprime N = p × q and returns its prime factors. The best known classical algorithm (General Number Field Sieve) runs in sub-exponential time L[1/3]. No polynomial-time classical algorithm is known. Quantum computers can factor in polynomial time (Shor's algorithm), but no classical algorithm achieves this.

Your task: implement `factor(N)` in `entry.py`. The entry with the best **scaling behavior** across increasing bit sizes wins.

## Quick Start

```bash
pip install -r requirements.txt
cp example_entry.py entry.py    # start from the example
python test.py                   # check correctness
python score.py                  # measure performance
```

Edit `entry.py` with your algorithm. The only requirement: implement the `factor(N)` function.

## Interface

```python
# entry.py

def factor(N: int) -> tuple[int, int]:
    """
    Factor a semiprime N = p * q into its two prime factors.
    
    Args:
        N: A semiprime (product of exactly two distinct primes).
           Both primes are of similar size (balanced).
           N fits in standard Python int (arbitrary precision).
    
    Returns:
        (p, q) where p <= q, p * q == N, and both p and q are prime.
    
    Constraints:
        - N is always a product of exactly two distinct odd primes.
        - The primes are balanced: both have approximately n/2 bits where n = N.bit_length().
        - Your function will be tested on semiprimes from 32 bits to 1024 bits.
        - Timeout: 60 seconds per factorization.
        - No network access. No reading from disk (except importing your own modules).
        - No hardcoded lookup tables larger than 1MB.
        - Standard library + numpy + sympy + fpylll are available.
    """
    pass
```

## Scoring

Your entry is scored on **scaling behavior**, not raw speed.

### Procedure

1. Generate semiprimes at bit sizes: 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024.
2. At each bit size: 5 random semiprimes, 60-second timeout each.
3. Record: success rate and median time at each bit size.
4. An algorithm is **alive** at bit size n if it solves at least 3/5 instances.

### Score Formula

```
Primary:   max_bits    = largest bit size where algorithm is alive
Secondary: exponent    = best-fit k in t(n) = c * n^k (log-log regression on median times)
Tertiary:  time_at_max = median time at max_bits
```

**Lower is better for exponent and time_at_max. Higher is better for max_bits.**

Entries are ranked by: (1) max_bits descending, (2) exponent ascending, (3) time_at_max ascending.

### What the scores mean

| max_bits | exponent | What it is |
|----------|----------|------------|
| 64 | - | Worse than trial division |
| 128 | ~0.25 (of N) | Pollard rho class: O(N^{1/4}) |
| 256 | growing | Quadratic sieve class: L[1/2] |
| 512+ | growing faster | NFS class: L[1/3] |
| 1024 | constant k | **Polynomial time — you win** |

A polynomial-time algorithm would reach 1024 bits with a **constant** exponent k (e.g., t(n) = c * n^5 gives k = 5 regardless of n). All known algorithms show k increasing with n — that's the sub-exponential signature.

## Files

| File | Purpose | Modify? |
|------|---------|---------|
| `entry.py` | **Your submission** | YES |
| `example_entry.py` | Naive example (trial division) | No |
| `baselines.py` | Reference implementations for comparison | No |
| `test.py` | Correctness autotester | No |
| `score.py` | Performance scorer | No |
| `generate.py` | Deterministic test case generator | No |
| `interface.py` | Type hints and validation | No |
| `requirements.txt` | Python dependencies | No |
| `README.md` | This file | No |

## Correctness Tests (`python test.py`)

Tests are incremental. Fix errors before scoring.

1. **Basic**: Factor small semiprimes (6-16 bits). Verifies p * q == N and both prime.
2. **Medium**: Factor medium semiprimes (20-40 bits). Tests balanced primes.
3. **Timing**: Factor 48-bit semiprimes within 60 seconds.
4. **Consistency**: Same input always produces same output.
5. **Edge cases**: Primes of different sizes, small primes, etc.

## Baselines (`python score.py --baselines`)

| Algorithm | Expected max_bits | Complexity |
|-----------|-------------------|------------|
| Trial division | ~48 | O(√N) = O(2^{n/2}) |
| Pollard rho | ~96 | O(N^{1/4}) = O(2^{n/4}) |
| Quadratic sieve | ~192 | L_N[1/2, 1] |

These are provided for comparison. Your goal is to beat all of them on the scaling metric.

## Rules

1. Implement all logic in `entry.py` (may import additional `.py` files you provide).
2. Must be written in Python. No calling external binaries or C extensions you wrote (numpy/sympy/fpylll are fine).
3. Must pass all correctness tests in `test.py`.
4. No network access. No reading files except your own source.
5. No hardcoded factorization tables larger than 1MB.
6. Algorithm must be **general-purpose**: no special-casing based on bit size.
7. Must be **deterministic or probabilistic with consistent scaling** (Monte Carlo ok, but scaling must be reproducible).

## Definitions

- **Semiprime**: N = p × q where p and q are distinct primes.
- **Balanced**: p and q have approximately the same number of bits.
- **Bit size**: `N.bit_length()` — the number of bits in the binary representation of N.
- **L-notation**: L_N[α, c] = exp(c · (ln N)^α · (ln ln N)^{1-α}). Sub-exponential when 0 < α < 1.
- **Polynomial time**: t(n) = O(n^k) for some constant k, where n = log₂(N) is the input size in bits.

## Background

This challenge was born from a research project that killed 32 approaches to polynomial-time factoring and identified five fundamental barriers:

1. **CRT blindness**: Ring operations on Z/NZ can't distinguish p from q.
2. **Genus obstruction**: Lattice methods find only genus-even vectors (useless for factoring).
3. **Birthday/smoothness barrier**: Collision methods hit O(N^{1/4}), smooth methods hit L[1/3].
4. **Dimensionality barrier**: Detecting structure at scale p requires O(p) = O(√N) data.
5. **Noise barrier**: No classical process produces Regev-style dual lattice samples better than random.

Every known approach maps to at least one barrier. The winning entry bypasses all five.

## Motivation

From George Hotz: *"It's just a matter of time before AI finds a polynomial time factoring algorithm. I see no reason factoring should be hard."*

Prove him right. Or prove the barriers are fundamental. Either way — show your work.
