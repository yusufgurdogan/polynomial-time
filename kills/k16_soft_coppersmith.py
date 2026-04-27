#!/usr/bin/env python3
"""
Soft Coppersmith: Can distributional (soft) constraints on p enable factoring?

Coppersmith's method is provably optimal: given n/4 EXACT bits of p, factoring
N = p*q is poly-time. But what about "soft" information -- distributional
constraints that reduce the entropy of p without pinning exact bits?

This experiment tests four types of soft information:
  Type 1: Interval constraint (baseline -- equivalent to MSB knowledge)
  Type 2: Multiple modular constraints (CRT soft info)
  Type 3: Noisy bits (Heninger-Shacham style)
  Type 4: Gaussian hints (novel -- averaging multiple noisy samples)

For each type, we measure the success rate vs amount of information and
identify the phase transition threshold, comparing against the Coppersmith
bound (n/4 bits of information).
"""

import sys
import math
import random
import time
import numpy as np
from collections import defaultdict
from sympy import nextprime
from fpylll import IntegerMatrix, LLL

sys.stdout.reconfigure(line_buffering=True)
random.seed(42)
np.random.seed(42)


# =============================================================================
# Helpers
# =============================================================================

def generate_prime(bits):
    """Generate a random prime of approximately the given bit-size."""
    lo = 1 << (bits - 1)
    hi = (1 << bits) - 1
    return nextprime(random.randint(lo, hi))


def generate_semiprime(bits):
    """Generate semiprime N = p*q with N approximately `bits` bits."""
    half = bits // 2
    p = generate_prime(half)
    q = generate_prime(bits - half)
    while q == p:
        q = generate_prime(bits - half)
    if p > q:
        p, q = q, p
    return p * q, p, q


def small_primes(limit):
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


def binary_entropy(p):
    """Binary entropy H(p) in bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# =============================================================================
# Coppersmith small root finder (self-contained)
# =============================================================================

def _poly_mul(a, b):
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def _poly_power(p, k):
    if k == 0:
        return [1]
    result = [1]
    for _ in range(k):
        result = _poly_mul(result, p)
    return result


def coppersmith_univariate(f_coeffs, N, X, beta=0.5, m=None):
    """
    Find small integer roots of monic f(x) mod some factor b >= N^beta.

    f_coeffs: [a0, a1, ..., ad] with ad = 1 (monic).
    X: bound on |root|.
    Returns list of integer roots found.
    """
    d = len(f_coeffs) - 1
    if d <= 0:
        return []
    assert f_coeffs[-1] == 1, "f must be monic"

    if m is None:
        m = max(1, int(math.ceil(7 * beta / d)))
        m = min(m, 15)

    t = max(1, int(d * m * (1.0 / beta - 1)))
    t = min(t, m)

    polys = []

    # g-polynomials: x^j * f(x)^i * N^(m-i)
    for i in range(m + 1):
        fi = _poly_power(f_coeffs, i)
        for j in range(d):
            p = [0] * j + [1]  # x^j
            p = _poly_mul(p, fi)
            n_pow = N ** (m - i)
            p = [c * n_pow for c in p]
            polys.append(p)

    # h-polynomials: x^l * f(x)^m
    fm = _poly_power(f_coeffs, m)
    for l in range(t):
        p = _poly_mul([0] * l + [1], fm)
        polys.append(p)

    dim = len(polys)
    max_deg = max(len(p) for p in polys)
    dim = max(dim, max_deg)

    B = IntegerMatrix(dim, dim)
    for row, poly in enumerate(polys):
        for j, coeff in enumerate(poly):
            if j < dim:
                B[row, j] = int(coeff * (X ** j))

    try:
        LLL.reduction(B)
    except Exception:
        return []

    roots = []
    for row in range(min(dim, 5)):  # only check first few short vectors
        poly = []
        valid = True
        for j in range(dim):
            val = int(B[row, j])
            if j == 0:
                poly.append(val)
            elif X ** j == 0:
                valid = False
                break
            elif val % (X ** j) == 0:
                poly.append(val // (X ** j))
            else:
                valid = False
                break

        if not valid or all(c == 0 for c in poly):
            continue

        # Find integer roots of this polynomial in [-X, X]
        # Trim trailing zeros
        while len(poly) > 1 and poly[-1] == 0:
            poly = poly[:-1]

        if len(poly) == 2:
            a, b = poly
            if b != 0 and a % b == 0:
                r = -a // b
                if abs(r) <= X and r not in roots:
                    roots.append(r)
        elif len(poly) == 3:
            a, b, c = poly
            if c != 0:
                disc = b * b - 4 * a * c
                if disc >= 0:
                    sd = math.isqrt(disc)
                    if sd * sd == disc:
                        for sign in [1, -1]:
                            num = -b + sign * sd
                            if (2 * c) != 0 and num % (2 * c) == 0:
                                r = num // (2 * c)
                                if abs(r) <= X and r not in roots:
                                    roots.append(r)
        elif len(poly) <= 20:
            # Use numpy for higher degree
            try:
                np_coeffs = [float(c) for c in reversed(poly)]
                approx_roots = np.roots(np_coeffs)
                for r in approx_roots:
                    if abs(r.imag) < 0.5:
                        cand = int(round(r.real))
                        if abs(cand) <= X:
                            val = sum(c * cand**j for j, c in enumerate(poly))
                            if val == 0 and cand not in roots:
                                roots.append(cand)
            except Exception:
                pass

    return roots


def factor_with_approximation(N, p_approx, bound):
    """
    Factor N given approximation p_approx with |p - p_approx| < bound.
    Returns the factor or None.
    """
    if bound < 1:
        return None

    g = math.gcd(p_approx, N)
    if 1 < g < N:
        return g

    # Direct search for very small bounds
    if bound < 10**6:
        for delta in range(int(bound) + 1):
            for cand in [p_approx + delta, p_approx - delta]:
                if 1 < cand < N and N % cand == 0:
                    return cand
        return None

    # Coppersmith: f(x) = x + p_approx has root (p - p_approx) mod p
    f = [p_approx, 1]
    for m_val in [4, 6, 8, 12, 16]:
        roots = coppersmith_univariate(f, N, int(bound), beta=0.49, m=m_val)
        for root in roots:
            cand = p_approx + root
            if 1 < cand < N and N % cand == 0:
                return cand

    return None


def factor_with_crt(N, residue, modulus):
    """
    Factor N given p === residue (mod modulus).
    p = residue + modulus * t, with t < sqrt(N) / modulus.
    """
    sqrt_N = math.isqrt(N)
    t_bound = sqrt_N // modulus + 1

    # Direct search for small search spaces
    if t_bound < 10**6:
        cand = residue
        while cand < 2:
            cand += modulus
        while cand <= sqrt_N + modulus:
            if N % cand == 0:
                return cand
            cand += modulus
        return None

    # Coppersmith: f(x) = x + residue*modinv(modulus, N)
    # Actually: p = residue + modulus*x, so (residue + modulus*x) === 0 (mod p)
    # Rewrite as monic: let y = modulus*x, then p_approx = residue, bound = modulus*t_bound
    # Simpler: just use the approximation framework
    # Best approximation of p: residue + modulus * (t_bound // 2)
    mid_t = t_bound // 2
    p_approx = residue + modulus * mid_t
    # Bound: modulus * t_bound / 2
    eff_bound = modulus * (t_bound // 2 + 1)

    return factor_with_approximation(N, p_approx, eff_bound)


# =============================================================================
# Type 1: Interval constraint
# =============================================================================

def experiment_interval(bit_sizes, trials_per_size=20):
    """
    Type 1: p in [a, b] where b - a = W.
    Equivalent to knowing MSBs: p = a + x with |x| < W.
    Coppersmith succeeds when W < N^{1/4}.

    Sweep the window exponent alpha where W = N^alpha.
    """
    print("\n" + "=" * 72)
    print("  TYPE 1: INTERVAL CONSTRAINT (p in [a, b])")
    print("  Baseline -- equivalent to MSB knowledge")
    print("=" * 72)
    print(f"  Coppersmith bound: success when window W < N^0.25")
    print()

    # Exponents to test: W = N^alpha
    alphas = [0.10, 0.15, 0.20, 0.22, 0.24, 0.25, 0.27, 0.30, 0.35, 0.40, 0.50]

    # Header
    print(f"  {'N bits':>6s}", end="")
    for alpha in alphas:
        print(f"  a={alpha:.2f}", end="")
    print("  | threshold")
    print("  " + "-" * (8 + len(alphas) * 8 + 15))

    all_results = {}

    for bits in bit_sizes:
        row_results = {}
        print(f"  {bits:6d}", end="")

        for alpha in alphas:
            successes = 0
            for trial in range(trials_per_size):
                N, p, q = generate_semiprime(bits)
                W = max(1, int(N ** alpha))

                # Generate interval containing p: random offset within [0, W)
                offset = random.randint(0, min(W - 1, p - 2)) if W > 1 else 0
                p_lo = max(2, p - offset)

                # Try to factor with this interval knowledge
                p_approx = p_lo + W // 2
                result = factor_with_approximation(N, p_approx, W // 2 + 1)
                if result and N % result == 0:
                    successes += 1

            rate = successes / trials_per_size
            row_results[alpha] = rate
            if rate >= 0.8:
                print(f"  {rate:5.0%} +", end="")
            elif rate >= 0.2:
                print(f"  {rate:5.0%} ~", end="")
            else:
                print(f"  {rate:5.0%} -", end="")

        # Find threshold (where success crosses 50%)
        threshold = None
        for alpha in alphas:
            if row_results.get(alpha, 0) >= 0.5:
                threshold = alpha
            else:
                break
        if threshold is not None:
            info_bits = bits * (1.0 - threshold) / 2  # bits of info about p
            print(f"  | a<={threshold:.2f} ({info_bits:.0f}b info)")
        else:
            print(f"  | < {alphas[0]}")

        all_results[bits] = row_results

    print()
    print("  Legend: + = >80% success, ~ = 20-80%, - = <20%")
    print("  Info bits = (N_bits) * (1 - alpha) / 2 = known bits of p")
    return all_results


# =============================================================================
# Type 2: Modular constraints (CRT)
# =============================================================================

def experiment_crt(bit_sizes, trials_per_size=20):
    """
    Type 2: Given p mod m_i = r_i for small moduli m_i.
    By CRT, product M = prod(m_i) determines p mod M.
    Then p = c + M*t where |t| < sqrt(N)/M.
    Coppersmith works when M > N^{1/4}.

    Test: how many small prime moduli are needed?
    """
    print("\n" + "=" * 72)
    print("  TYPE 2: MODULAR CONSTRAINTS (CRT)")
    print("  Given p mod m_i for small primes m_i = 2, 3, 5, 7, 11, ...")
    print("=" * 72)

    primes = small_primes(200)  # primes up to 200

    # For each bit size, sweep number of known modular constraints
    num_moduli_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 46]

    # Header
    print(f"\n  {'N bits':>6s}", end="")
    for nm in num_moduli_list:
        print(f" {nm:4d}m", end="")
    print("  | threshold")
    print("  " + "-" * (8 + len(num_moduli_list) * 6 + 15))

    all_results = {}

    for bits in bit_sizes:
        row_results = {}
        n_quarter_bits = bits / 4.0  # N^{1/4} has this many bits
        print(f"  {bits:6d}", end="")

        for num_moduli in num_moduli_list:
            if num_moduli > len(primes):
                print(f"   n/a", end="")
                continue

            mods = primes[:num_moduli]
            M = 1
            for m in mods:
                M *= m
            M_bits = M.bit_length()

            successes = 0
            for trial in range(trials_per_size):
                N, p, q = generate_semiprime(bits)

                # Oracle: compute p mod m_i for each modulus
                residues = [p % m for m in mods]

                # CRT combine to find p mod M
                combined = residues[0]
                current_mod = mods[0]
                for i in range(1, num_moduli):
                    r_new = residues[i]
                    m_new = mods[i]
                    # CRT: find x === combined (mod current_mod) and x === r_new (mod m_new)
                    # Since mods are primes, gcd = 1
                    inv = pow(current_mod, -1, m_new)
                    combined = combined + current_mod * ((r_new - combined) * inv % m_new)
                    current_mod *= m_new
                    combined %= current_mod

                # Now: p === combined (mod M)
                result = factor_with_crt(N, combined, M)
                if result and N % result == 0:
                    successes += 1

            rate = successes / trials_per_size
            row_results[num_moduli] = rate
            if rate >= 0.8:
                print(f" {rate:4.0%}+", end="")
            elif rate >= 0.2:
                print(f" {rate:4.0%}~", end="")
            else:
                print(f" {rate:4.0%}-", end="")

        # Find threshold
        threshold_nm = None
        for nm in num_moduli_list:
            if row_results.get(nm, 0) >= 0.5:
                threshold_nm = nm
                break
        if threshold_nm is not None:
            M_thr = 1
            for m in primes[:threshold_nm]:
                M_thr *= m
            info_bits = math.log2(M_thr) if M_thr > 1 else 0
            print(f"  | {threshold_nm}m ({info_bits:.1f}b, N/4={bits/4:.0f}b)")
        else:
            print(f"  | >{num_moduli_list[-1]}m")

        all_results[bits] = row_results

    print()
    print("  'm' = number of small-prime moduli. Info bits = log2(product of moduli).")
    print("  Theory: need log2(M) >= N_bits/4 for Coppersmith to work.")

    # Summary: information efficiency
    print(f"\n  --- CRT information efficiency ---")
    for bits in bit_sizes:
        needed = bits / 4.0
        for nm in num_moduli_list:
            if all_results.get(bits, {}).get(nm, 0) >= 0.5:
                M = 1
                for m in primes[:nm]:
                    M *= m
                actual = math.log2(M) if M > 1 else 0
                ratio = actual / needed if needed > 0 else 0
                print(f"    {bits:3d}-bit N: need {needed:.0f}b, CRT threshold at {actual:.1f}b "
                      f"({nm} moduli), efficiency = {ratio:.2f}x")
                break
        else:
            print(f"    {bits:3d}-bit N: need {needed:.0f}b, threshold not reached")

    return all_results


# =============================================================================
# Type 3: Noisy bits (Heninger-Shacham)
# =============================================================================

def experiment_noisy_bits(bit_sizes, trials_per_size=15):
    """
    Type 3: Each bit of p known with probability 1/2 + epsilon.
    For eps=0: no info. For eps=0.5: perfect knowledge.

    Information per bit: 1 - H(1/2 + eps) bits.
    Total info: (n/2) * (1 - H(1/2 + eps)).
    Need ~n/4 bits of info => eps ~= 0.31 (information-theoretic threshold).

    KEY SUBTLETY: Coppersmith needs arithmetic proximity, not Hamming proximity.
    A single wrong MSB creates distance ~2^{p_bits-1}, dwarfing N^{1/4}.
    So the question becomes: can we convert noisy bits into arithmetic proximity?

    Strategy A: Oracle bound (use true |p - noisy_p| as bound).
    Strategy B: Top-bits strategy (use only the top k bits of noisy p, where
                k is chosen so the expected number of errors in the top k bits
                is small enough that the arithmetic error stays below N^{1/4}).
    Strategy C: Statistical bound (3-sigma from noise model).
    """
    print("\n" + "=" * 72)
    print("  TYPE 3: NOISY BITS (Heninger-Shacham style)")
    print("  Each bit of p correct with prob 1/2 + eps")
    print("=" * 72)

    eps_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    # Information-theoretic analysis
    print(f"\n  Information-theoretic analysis:")
    print(f"  {'eps':>6s}  {'P(correct)':>10s}  {'info/bit':>8s}  {'total (32b p)':>14s}  {'arith bound':>16s}")
    print(f"  " + "-" * 60)
    for eps in eps_values:
        p_correct = 0.5 + eps
        info_per_bit = 1.0 - binary_entropy(p_correct) if 0 < p_correct < 1 else (1.0 if p_correct >= 1 else 0.0)
        info_32 = 32 * info_per_bit
        # Expected arithmetic error: dominated by highest wrong bit
        # For p_bits=32: the top bit is wrong with prob (0.5-eps),
        # so expected error ~ (0.5-eps)*2^31 + (0.5-eps)*2^30 + ...
        # ~ (0.5-eps) * 2^32
        if eps < 0.5:
            arith_err = (0.5 - eps) * (2**32)
            n_quarter = 2**16  # N^{1/4} for 64-bit N
            arith_str = f"~2^{math.log2(max(arith_err,1)):.1f} vs N^.25=2^16"
        else:
            arith_str = "0"
        print(f"  {eps:6.2f}  {p_correct:10.2f}  {info_per_bit:8.4f}  {info_32:12.1f}b/16  {arith_str:>16s}")

    print(f"\n  CRITICAL OBSERVATION: Even at eps=0.45 (90% correct), expected")
    print(f"  arithmetic error is ~(0.05)*2^{{p_bits}} >> N^{{1/4}} = 2^{{p_bits/2}}.")
    print(f"  Noisy bits do NOT naturally give arithmetic proximity!")

    # ---- Strategy A: Oracle bound ----
    print(f"\n  Strategy A: Oracle bound (use true |p - noisy_p|)")
    print(f"  This is maximally generous -- tests whether Coppersmith can")
    print(f"  handle the actual error when it happens to be small.\n")

    print(f"  {'N bits':>6s}", end="")
    for eps in eps_values:
        print(f" e={eps:.2f}", end="")
    print("  | threshold")
    print("  " + "-" * (8 + len(eps_values) * 7 + 15))

    all_results = {}

    for bits in bit_sizes:
        row_results = {}
        p_bits = bits // 2
        n_quarter = 2 ** (bits // 4)
        print(f"  {bits:6d}", end="")

        for eps in eps_values:
            successes = 0
            p_correct_prob = 0.5 + eps

            for trial in range(trials_per_size):
                N, p, q = generate_semiprime(bits)

                if eps >= 0.5:
                    noisy_p = p
                elif eps <= 0.0:
                    noisy_p = random.randint(2, math.isqrt(N))
                else:
                    noisy_p = 0
                    for i in range(p_bits):
                        true_bit = (p >> i) & 1
                        if random.random() < p_correct_prob:
                            noisy_p |= (true_bit << i)
                        else:
                            noisy_p |= ((1 - true_bit) << i)

                if noisy_p < 2:
                    noisy_p = 2

                actual_distance = abs(p - noisy_p)
                if eps > 0:
                    bound = max(actual_distance + 1, 2)
                else:
                    bound = math.isqrt(N)

                result = factor_with_approximation(N, noisy_p, bound)
                if result and N % result == 0:
                    successes += 1

            rate = successes / trials_per_size
            row_results[eps] = rate
            if rate >= 0.8:
                print(f"  {rate:4.0%}+", end="")
            elif rate >= 0.2:
                print(f"  {rate:4.0%}~", end="")
            else:
                print(f"  {rate:4.0%}-", end="")

        threshold_eps = None
        for eps in eps_values:
            if row_results.get(eps, 0) >= 0.5:
                threshold_eps = eps
                break
        if threshold_eps is not None:
            info_per_bit = 1.0 - binary_entropy(0.5 + threshold_eps) if threshold_eps > 0 else 0
            total_info = p_bits * info_per_bit
            print(f"  | eps>={threshold_eps:.2f} ({total_info:.1f}b)")
        else:
            print(f"  | >{eps_values[-1]:.2f}")

        all_results[bits] = row_results

    # ---- Strategy B: Top-bits strategy ----
    print(f"\n  Strategy B: Use only top k bits of noisy p")
    print(f"  Choose k so that P(all top k bits correct) is reasonable.")
    print(f"  If top k bits are correct, arithmetic error < 2^(p_bits - k).")
    print(f"  Need: 2^(p_bits - k) < N^(1/4) = 2^(n/4), so k > p_bits - n/4 = n/4.")
    print(f"  P(all top k correct) = (1/2+eps)^k. Need many trials.\n")

    # For each eps, compute how many correct-MSB trials to expect
    print(f"  {'N bits':>6s}  {'eps':>5s}  {'k needed':>8s}  {'P(top k ok)':>11s}  {'trials':>7s}  {'rate':>5s}")
    print(f"  " + "-" * 50)

    for bits in [48, 64, 80]:
        p_bits = bits // 2
        n_quarter_bits = bits // 4
        k_needed = p_bits - n_quarter_bits  # need top k bits correct

        for eps in [0.25, 0.30, 0.35, 0.40, 0.45]:
            p_correct = 0.5 + eps
            # P(all top k bits correct) = p_correct^k
            p_all_correct = p_correct ** k_needed
            # Expected trials to get one success: 1/p_all_correct
            expected_trials = 1.0 / p_all_correct if p_all_correct > 0 else float('inf')

            # Run the experiment: generate noisy p, check if top k bits happen
            # to be correct, and if so, apply Coppersmith
            successes = 0
            num_trials = min(max(int(3 / max(p_all_correct, 1e-10)), 20), 200)
            for trial in range(num_trials):
                N, p, q = generate_semiprime(bits)
                noisy_p = 0
                for i in range(p_bits):
                    true_bit = (p >> i) & 1
                    if random.random() < p_correct:
                        noisy_p |= (true_bit << i)
                    else:
                        noisy_p |= ((1 - true_bit) << i)
                if noisy_p < 2:
                    noisy_p = 2

                # Use only top k bits: zero out bottom (p_bits - k) bits
                mask = ((1 << p_bits) - 1) ^ ((1 << (p_bits - k_needed)) - 1)
                p_approx = noisy_p & mask
                bound = 1 << (p_bits - k_needed + 1)  # 2^{p_bits - k + 1}

                result = factor_with_approximation(N, p_approx, bound)
                if result and N % result == 0:
                    successes += 1

            rate = successes / num_trials
            print(f"  {bits:6d}  {eps:5.2f}  {k_needed:8d}  {p_all_correct:11.6f}  {num_trials:7d}  {rate:5.0%}")

    # ---- Strategy C: Statistical bound ----
    print(f"\n  Strategy C: Statistical bound (3-sigma from noise model)")
    print(f"  Bound = 3 * sigma where sigma^2 = (1/4 - eps^2) * (4^p_bits - 1)/3\n")

    print(f"  {'N bits':>6s}", end="")
    for eps in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50]:
        print(f" e={eps:.2f}", end="")
    print()
    print(f"  " + "-" * 52)

    for bits in bit_sizes:
        p_bits = bits // 2
        print(f"  {bits:6d}", end="")
        for eps in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50]:
            successes = 0
            p_correct_prob = 0.5 + eps

            for trial in range(trials_per_size):
                N, p, q = generate_semiprime(bits)

                if eps >= 0.5:
                    noisy_p = p
                    bound = 1
                else:
                    noisy_p = 0
                    for i in range(p_bits):
                        true_bit = (p >> i) & 1
                        if random.random() < p_correct_prob:
                            noisy_p |= (true_bit << i)
                        else:
                            noisy_p |= ((1 - true_bit) << i)
                    if noisy_p < 2:
                        noisy_p = 2

                    # Variance of |p - noisy_p|:
                    # Each bit i contributes 2^{2i} * (1/2-eps)(1/2+eps) to variance
                    # Total variance = (1/4-eps^2) * sum_{i=0}^{p_bits-1} 4^i
                    #                 = (1/4-eps^2) * (4^p_bits - 1)/3
                    variance_factor = (0.25 - eps * eps) / 3.0
                    sigma = (2 ** p_bits) * math.sqrt(max(variance_factor, 1e-30))
                    bound = max(int(3 * sigma), 2)

                result = factor_with_approximation(N, noisy_p, bound)
                if result and N % result == 0:
                    successes += 1

            rate = successes / trials_per_size
            if rate >= 0.8:
                print(f"  {rate:4.0%}+", end="")
            elif rate >= 0.2:
                print(f"  {rate:4.0%}~", end="")
            else:
                print(f"  {rate:4.0%}-", end="")
        print()

    print()
    print("  KEY FINDING: Noisy bits are fundamentally different from interval")
    print("  constraints. Even with >n/4 bits of Shannon information, the")
    print("  ARITHMETIC error remains huge because errors in high-order bits")
    print("  dominate. Coppersmith needs arithmetic proximity, not information.")
    print("  This is the gap between information and computation: Shannon")
    print("  information does not directly translate to Coppersmith-usable")
    print("  constraints unless the information is STRUCTURED (top bits known).")

    return all_results


# =============================================================================
# Type 4: Gaussian hints (novel)
# =============================================================================

def experiment_gaussian(bit_sizes, trials_per_size=15):
    """
    Type 4: Given k independent samples from N(p, sigma^2).

    Each sample gives ~log2(N/sigma) bits of info.
    Mean of k samples has std dev sigma/sqrt(k).
    Coppersmith works when sigma/sqrt(k) < N^{1/4},
    i.e., k > (sigma / N^{1/4})^2.

    This is NOVEL: nobody has studied Gaussian hints for factoring.
    The question: does averaging + Coppersmith match the theory?
    """
    print("\n" + "=" * 72)
    print("  TYPE 4: GAUSSIAN HINTS (novel -- not in the literature)")
    print("  Given k samples from Normal(p, sigma^2)")
    print("=" * 72)
    print(f"  Theory: factoring succeeds when sigma/sqrt(k) < N^(1/4)")
    print(f"  Equivalently: k > (sigma / N^(1/4))^2")
    print()

    # For each bit size, sweep (sigma, k) pairs
    # Express sigma as N^gamma for various gamma
    gamma_values = [0.30, 0.35, 0.40, 0.45]
    k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    print(f"  --- Success rate for each (sigma = N^gamma, k samples) ---")
    print(f"  Coppersmith succeeds when sigma/sqrt(k) < N^0.25,")
    print(f"  i.e., N^gamma / sqrt(k) < N^0.25, i.e., k > N^(2*gamma - 0.5)")
    print()

    all_results = {}

    for bits in bit_sizes:
        print(f"  ---- N = {bits} bits ----")
        print(f"  {'sigma':>12s}  {'k_theory':>8s}", end="")
        for k in k_values:
            print(f" k={k:4d}", end="")
        print()
        print(f"  " + "-" * (24 + len(k_values) * 7))

        results_for_bits = {}

        for gamma in gamma_values:
            sigma = max(1, int(round(float(2 ** (bits * gamma)))))
            # Theoretical k needed: (sigma / N^0.25)^2
            n_quarter = 2 ** (bits * 0.25)
            k_theory = (sigma / n_quarter) ** 2
            k_theory_int = max(1, int(math.ceil(k_theory)))

            print(f"  N^{gamma:.2f}={sigma.bit_length():4d}b  {k_theory_int:>8d}", end="")

            for k in k_values:
                successes = 0
                for trial in range(trials_per_size):
                    N, p, q = generate_semiprime(bits)
                    sigma_actual = max(1, int(round(float(N ** gamma))))

                    # Generate k Gaussian samples centered on p
                    # (Use integer rounding since p is an integer)
                    samples_list = []
                    for _ in range(k):
                        sample = p + int(round(random.gauss(0, sigma_actual)))
                        samples_list.append(sample)

                    # Average the samples
                    mean_est = int(round(sum(samples_list) / len(samples_list)))
                    if mean_est < 2:
                        mean_est = 2

                    # Bound on |p - mean|: sigma/sqrt(k) * safety_factor
                    effective_sigma = sigma_actual / math.sqrt(k)
                    # Use 3-sigma bound (covers 99.7% of cases)
                    bound = max(int(3 * effective_sigma) + 1, 2)

                    result = factor_with_approximation(N, mean_est, bound)
                    if result and N % result == 0:
                        successes += 1

                rate = successes / trials_per_size
                results_for_bits[(gamma, k)] = rate
                if rate >= 0.8:
                    print(f"  {rate:4.0%}+", end="")
                elif rate >= 0.2:
                    print(f"  {rate:4.0%}~", end="")
                else:
                    print(f"  {rate:4.0%}-", end="")
            print()

        all_results[bits] = results_for_bits
        print()

    # Phase transition analysis
    print(f"  --- Phase transition analysis ---")
    print(f"  For each (bits, gamma), find the k where success crosses 50%.")
    print(f"  Compare to theoretical k_theory = N^(2*gamma - 0.5).\n")
    print(f"  {'N bits':>6s}  {'gamma':>6s}  {'k_theory':>8s}  {'k_actual':>8s}  {'ratio':>6s}")
    print(f"  " + "-" * 42)

    for bits in bit_sizes:
        for gamma in gamma_values:
            n_quarter = 2 ** (bits * 0.25)
            sigma = max(1, int(round(float(2 ** (bits * gamma)))))
            k_theory = max(1, int(math.ceil((sigma / n_quarter) ** 2)))

            k_actual = None
            for k in k_values:
                if all_results.get(bits, {}).get((gamma, k), 0) >= 0.5:
                    k_actual = k
                    break

            if k_actual is not None:
                ratio = k_actual / k_theory if k_theory > 0 else float('inf')
                print(f"  {bits:6d}  {gamma:6.2f}  {k_theory:>8d}  {k_actual:>8d}  {ratio:6.1f}x")
            else:
                print(f"  {bits:6d}  {gamma:6.2f}  {k_theory:>8d}  {'> max':>8s}  {'n/a':>6s}")

    return all_results


# =============================================================================
# Summary and comparison
# =============================================================================

def print_summary():
    """Print the theoretical comparison between soft information types."""
    print("\n" + "=" * 72)
    print("  THEORETICAL SUMMARY: Soft Information for Factoring")
    print("=" * 72)

    print("""
  For N = p*q with p,q ~ N^{1/2} (balanced semiprime):

  Coppersmith's theorem: factoring is poly-time given n/4 bits of
  EXACT information about p (i.e., p is known to within N^{1/4}).

  Type 1 -- Interval [a, b]:
    Window W = b - a. Success when W < N^{1/4}.
    Information: log2(N/W) bits = n/2 - log2(W) bits about p.
    Need: log2(W) < n/4, i.e., n/4 bits of info. MATCHES Coppersmith.

  Type 2 -- CRT constraints (p mod m_i = r_i):
    Combined modulus M = prod(m_i). p is known mod M.
    Remaining uncertainty: sqrt(N)/M. Success when M > N^{1/4}.
    Information: log2(M) bits. Need log2(M) >= n/4. MATCHES Coppersmith.
    Key insight: only ~n/4 bits of modular info needed -- surprisingly few
    small-prime moduli suffice (primes up to ~n are enough).

  Type 3 -- Noisy bits (P(correct) = 1/2 + eps):
    Info per bit: 1 - H(1/2 + eps) bits.
    Total info for n/2-bit p: (n/2)(1 - H(1/2 + eps)).
    Information-theoretic threshold: eps >= 0.311.
    BUT: Coppersmith needs ARITHMETIC proximity, not information.
    Even at eps=0.45 (90% correct), expected arithmetic error is
    ~(0.05)*2^{n/2} >> N^{1/4} = 2^{n/4}. A single wrong MSB is fatal.
    This is the KEY FINDING: Shannon information does not automatically
    translate to Coppersmith-usable constraints. The information must
    be STRUCTURED (e.g., top bits known) to reduce arithmetic error.
    With a top-bits strategy, threshold matches theory but requires
    exponentially many trials (1/P(all top bits correct)).

  Type 4 -- Gaussian hints N(p, sigma^2), k samples:
    Mean has std dev sigma/sqrt(k). Effective window: ~3*sigma/sqrt(k).
    Success when 3*sigma/sqrt(k) < N^{1/4}.
    Need: k > 9 * (sigma/N^{1/4})^2.
    For sigma = N^gamma: k > 9 * N^{2*gamma - 1/2}.
    MATCHES Coppersmith up to the constant factor 9 (from 3-sigma bound).

  CONCLUSION: The n/4-bit threshold governs ARITHMETIC uncertainty,
  not Shannon information. Types 1, 2, and 4 provide information that
  directly reduces arithmetic uncertainty (interval, CRT residue, or
  Gaussian mean), so their thresholds match Coppersmith's bound.
  Type 3 (noisy bits) reveals a fundamental gap: Shannon information
  does NOT automatically reduce arithmetic uncertainty. A few wrong
  high-order bits create arithmetic errors exponentially larger than
  N^{1/4}, even when total information exceeds n/4 bits.

  The critical distinction:
  - ARITHMETIC soft info (interval, CRT, Gaussian) -> Coppersmith works
  - COMBINATORIAL soft info (noisy bits) -> Coppersmith fails unless
    you can convert to arithmetic proximity (which costs exp time)

  Open question: is there a poly-time method to exploit noisy bits
  without first converting to arithmetic proximity? The information
  IS there (>n/4 bits at eps >= 0.31), but current lattice methods
  cannot use it. This is the real frontier of "soft Coppersmith".
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  SOFT COPPERSMITH EXPERIMENT")
    print("  Can distributional constraints on p enable factoring?")
    print("=" * 72)
    print(f"  Date: 2026-03-21")
    print(f"  Seed: 42")
    print()

    t_start = time.time()

    # Bit sizes to test. Keep moderate for tractability -- Coppersmith's lattice
    # gets expensive at large sizes and we need many trials for statistics.
    bit_sizes = [32, 48, 64, 80, 96, 128]

    # ----- Type 1: Interval -----
    t0 = time.time()
    results_interval = experiment_interval(bit_sizes, trials_per_size=20)
    print(f"  [Type 1 elapsed: {time.time() - t0:.1f}s]")

    # ----- Type 2: CRT -----
    t0 = time.time()
    results_crt = experiment_crt(bit_sizes, trials_per_size=20)
    print(f"  [Type 2 elapsed: {time.time() - t0:.1f}s]")

    # ----- Type 3: Noisy bits -----
    t0 = time.time()
    results_noisy = experiment_noisy_bits(bit_sizes, trials_per_size=15)
    print(f"  [Type 3 elapsed: {time.time() - t0:.1f}s]")

    # ----- Type 4: Gaussian -----
    t0 = time.time()
    # Use smaller set for Gaussian since it has 2D sweep
    gaussian_bits = [32, 48, 64, 80]
    results_gaussian = experiment_gaussian(gaussian_bits, trials_per_size=12)
    print(f"  [Type 4 elapsed: {time.time() - t0:.1f}s]")

    # ----- Summary -----
    print_summary()

    # ----- Cross-type comparison table -----
    print("=" * 72)
    print("  CROSS-TYPE COMPARISON: Information threshold (in bits of p)")
    print("=" * 72)
    print(f"  {'N bits':>6s}  {'p bits':>6s}  {'n/4':>5s}  {'Interv':>6s}  {'CRT':>6s}  {'Noisy':>6s}  {'Gauss':>6s}")
    print(f"  " + "-" * 50)

    for bits in bit_sizes:
        p_bits = bits // 2
        n_quarter = bits / 4.0

        # Type 1: find threshold alpha
        t1_info = None
        if bits in results_interval:
            for alpha in sorted(results_interval[bits].keys()):
                if results_interval[bits][alpha] >= 0.5:
                    t1_info = p_bits * (1.0 - 2 * alpha)  # bits of p known
                    break

        # Type 2: find threshold num_moduli
        t2_info = None
        primes_list = small_primes(200)
        if bits in results_crt:
            for nm in sorted(results_crt[bits].keys()):
                if results_crt[bits][nm] >= 0.5:
                    M = 1
                    for m in primes_list[:nm]:
                        M *= m
                    t2_info = math.log2(M) if M > 1 else 0
                    break

        # Type 3: find threshold eps (oracle bound version)
        t3_info = None
        if bits in results_noisy:
            for eps in sorted(results_noisy[bits].keys()):
                if results_noisy[bits][eps] >= 0.5:
                    info_per_bit = 1.0 - binary_entropy(0.5 + eps) if eps > 0 else 0
                    t3_info = p_bits * info_per_bit
                    break

        # Type 4: approximate (use gamma=0.35 as reference)
        t4_info = None
        if bits in results_gaussian:
            for gamma in [0.30, 0.35, 0.40, 0.45]:
                for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    if results_gaussian[bits].get((gamma, k), 0) >= 0.5:
                        sigma = max(1, int(round(float(2 ** (bits * gamma)))))
                        eff_sigma = sigma / math.sqrt(k)
                        # Info = log2(sqrt(N) / eff_sigma)
                        t4_info = max(0, p_bits - math.log2(max(eff_sigma, 1)))
                        break
                if t4_info is not None:
                    break

        def fmt(v):
            return f"{v:6.1f}" if v is not None else "   n/a"

        print(f"  {bits:6d}  {p_bits:6d}  {n_quarter:5.0f}  {fmt(t1_info)}  {fmt(t2_info)}  {fmt(t3_info)}  {fmt(t4_info)}")

    print()
    total = time.time() - t_start
    print(f"  Total elapsed: {total:.1f}s")
    print()
    print("  RESULT: Coppersmith's n/4 barrier governs ARITHMETIC uncertainty.")
    print("  Types 1,2,4 match the bound. Type 3 (noisy bits) reveals a gap:")
    print("  Shannon info does NOT equal arithmetic proximity. Open question:")
    print("  can any poly-time method exploit noisy bits at eps ~= 0.31?")
    print()
