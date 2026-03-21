#!/usr/bin/env python3
"""
Dual lattice sampling experiment.

Can any classical process produce vectors correlated with L*/Z^d
(the dual of the multiplicative relation lattice)?

Regev's quantum factoring algorithm produces ~sqrt(n) samples from L*/Z^d,
each within distance 2^{-O(sqrt(n))} of a uniform dual lattice element.
The classical post-processing (LLL) then recovers factors.

This experiment tests whether ANY classical process can produce samples
that are even slightly better than random (i.e., closer to L*/Z^d than
a uniform random vector would be).

For each semiprime N = pq, we:
  1. Build L_R explicitly using discrete logs and LLL
  2. Compute L* (the dual lattice) from L_R's basis
  3. Generate classical "sample" vectors via six strategies
  4. Measure how close each sample is to L*/Z^d
  5. Report noise ratios (< 1 means correlated with the dual)
"""

import math
import sys
import time
import random
import numpy as np
from fpylll import IntegerMatrix, LLL
from sympy import nextprime, isprime

sys.stdout.reconfigure(line_buffering=True)

RNG = np.random.RandomState(42)
random.seed(42)

NUM_SAMPLES = 1000


# =============================================================================
# Utilities
# =============================================================================

def small_primes(limit):
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            for j in range(i * i, limit + 1, i):
                s[j] = False
    return [i for i in range(limit + 1) if s[i]]


def primitive_root(p):
    """Find a primitive root mod prime p."""
    if p == 2:
        return 1
    phi = p - 1
    factors = factorize_small(phi)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    return None


def discrete_log(a, g, p):
    """Compute log_g(a) mod (p-1) by brute force. Only feasible for small p."""
    a = a % p
    if a == 0:
        return 0
    current = 1
    for k in range(p):
        if current == a:
            return k
        current = (current * g) % p
    return 0


def factorize_small(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def jacobi_symbol(a, n):
    """Compute the Jacobi symbol (a/n) for odd n > 0."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be odd and positive")
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0


def generate_semiprime(bits):
    """Generate a semiprime N = p*q where N has approximately `bits` bits."""
    half = bits // 2
    lo = 1 << (half - 1)
    hi = (1 << half) - 1
    while True:
        p = nextprime(random.randint(lo, hi))
        if p > hi:
            continue
        q = nextprime(random.randint(lo, hi))
        if q > hi or q == p:
            continue
        N = p * q
        if N.bit_length() >= bits - 1 and N.bit_length() <= bits + 1:
            return N, min(p, q), max(p, q)


# =============================================================================
# Step 1: Build L_R
# =============================================================================

def build_LR(N, p, q, bases):
    """
    Build the relation lattice L_R = {e in Z^d : prod b_i^{e_i} = 1 mod N}.

    Uses discrete logs mod p and mod q, then the augmented-lattice + LLL approach
    from lattice_compare2.py.

    Returns:
        basis: numpy array of shape (rank, d), rows are basis vectors of L_R
        dlogs_p, dlogs_q: discrete log vectors
    """
    d = len(bases)
    g_p = primitive_root(p)
    g_q = primitive_root(q)

    dlogs_p = [discrete_log(b % p, g_p, p) for b in bases]
    dlogs_q = [discrete_log(b % q, g_q, q) for b in bases]

    # Augmented lattice: (d+2) x (d+2) matrix
    # First d rows: [e_i | dlogs_p[i] | dlogs_q[i]]
    # Last 2 rows:  [0...0 | p-1 | 0], [0...0 | 0 | q-1]
    dim = d + 2
    B = IntegerMatrix(dim, dim)
    for i in range(d):
        B[i, i] = 1
        B[i, d] = dlogs_p[i]
        B[i, d + 1] = dlogs_q[i]
    B[d, d] = p - 1
    B[d + 1, d + 1] = q - 1

    LLL.reduction(B)

    # Extract kernel vectors: rows where auxiliary coords (last 2) are zero
    kernel_vecs = []
    for i in range(dim):
        v = [int(B[i, j]) for j in range(d)]
        aux = [int(B[i, j]) for j in range(d, dim)]
        if all(a == 0 for a in aux) and any(x != 0 for x in v):
            kernel_vecs.append(v)

    if len(kernel_vecs) == 0:
        # Fallback: use larger scaling to force more kernel vectors
        scale = max(p - 1, q - 1) * 10
        B2 = IntegerMatrix(dim, dim)
        for i in range(d):
            B2[i, i] = scale
            B2[i, d] = dlogs_p[i]
            B2[i, d + 1] = dlogs_q[i]
        B2[d, d] = p - 1
        B2[d + 1, d + 1] = q - 1
        LLL.reduction(B2)
        for i in range(dim):
            v = [int(B2[i, j]) // scale for j in range(d)]
            aux = [int(B2[i, j]) for j in range(d, dim)]
            if all(a == 0 for a in aux) and any(x != 0 for x in v):
                kernel_vecs.append(v)

    basis = np.array(kernel_vecs, dtype=np.float64) if kernel_vecs else None
    return basis, dlogs_p, dlogs_q


def ensure_full_rank_LR(N, p, q, max_primes=30):
    """
    Build L_R with enough factor-base primes to get a full-rank lattice.
    The lattice L_R in Z^d has rank d-2 generically (two constraints from p-1, q-1).
    We want at least rank >= 2 for a meaningful dual.

    Returns: basis (rows = basis vectors), bases (the factor base primes), d
    """
    all_primes = [pr for pr in small_primes(200) if N % pr != 0]

    # Start with enough primes so rank = d - 2 >= 2, i.e., d >= 4
    for d in range(4, min(max_primes + 1, len(all_primes) + 1)):
        bases = all_primes[:d]
        basis, dlogs_p, dlogs_q = build_LR(N, p, q, bases)
        if basis is not None and len(basis) >= 2:
            return basis, bases, d, dlogs_p, dlogs_q

    # Last resort: use all available primes
    bases = all_primes[:max_primes]
    d = len(bases)
    basis, dlogs_p, dlogs_q = build_LR(N, p, q, bases)
    return basis, bases, d, dlogs_p, dlogs_q


# =============================================================================
# Step 2: Distance measurement
# =============================================================================

def dist_to_dual_mod_Zd(w, LR_basis):
    """
    Measure dist(w, L*/Z^d).

    For w to lie in L*, we need <w, x> in Z for all x in L_R.
    Given L_R basis B (rows = basis vectors), we compute B @ w.
    If B @ w is an integer vector, then w is in L*.
    Modulo Z^d, the distance is measured by how close B @ w is to integers.

    dist(w, L*/Z^d) = || frac(B @ w) ||
    where frac(x) = x - round(x) maps each component to [-0.5, 0.5).
    """
    Bw = LR_basis @ w  # shape (rank,)
    frac = Bw - np.round(Bw)
    return np.linalg.norm(frac)


def batch_distances(samples, LR_basis):
    """Compute distances for a batch of sample vectors."""
    # samples: (n_samples, d)
    # LR_basis: (rank, d)
    Bw = samples @ LR_basis.T  # (n_samples, rank)
    frac = Bw - np.round(Bw)
    return np.linalg.norm(frac, axis=1)


# =============================================================================
# Step 3: Classical sampling strategies
# =============================================================================

def strategy_A_uniform(d, n_samples):
    """Baseline: uniform random vectors in [0, 1)^d."""
    return RNG.random((n_samples, d))


def strategy_B_random_walk(N, bases, d, n_samples):
    """
    Random walks on (Z/NZ)*.

    Pick random g, compute g^k mod N for k = 1, 2, ..., extract "phase"
    information by looking at fractional position of (g^k mod N) / N
    and correlating with each base prime.
    """
    samples = np.zeros((n_samples, d))
    for s in range(n_samples):
        g = random.randint(2, N - 1)
        while math.gcd(g, N) != 1:
            g = random.randint(2, N - 1)
        # Walk length: proportional to sqrt(N)
        walk_len = max(10, int(math.sqrt(N)))
        walk_len = min(walk_len, 5000)  # cap for speed
        acc = np.zeros(d)
        gk = g
        for k in range(1, walk_len + 1):
            gk = (gk * g) % N
            phase = gk / N  # fractional position in [0, 1)
            for i, b in enumerate(bases):
                # Correlate: does g^k mod b_i reveal phase structure?
                acc[i] += (gk % b) / b
        samples[s] = (acc / walk_len) % 1.0
    return samples


def strategy_C_jacobi(N, bases, d, n_samples):
    """
    Jacobi symbol patterns.

    For each small prime b_i, compute Jacobi(b_i, N).
    Then for random offsets m, compute Jacobi(m*b_i, N) and use
    the pattern of +1/-1 values to construct a phase vector.
    """
    samples = np.zeros((n_samples, d))
    base_jacobi = [jacobi_symbol(b, N) for b in bases]

    for s in range(n_samples):
        m = random.randint(1, N - 1)
        while math.gcd(m, N) != 1:
            m = random.randint(1, N - 1)
        for i, b in enumerate(bases):
            # Jacobi(m * b, N) = Jacobi(m, N) * Jacobi(b, N)
            j_mb = jacobi_symbol(m * b % N, N) if (m * b % N) != 0 else 0
            j_m = jacobi_symbol(m, N)
            # Use the Jacobi info to bias the coordinate
            # Map: +1 -> 0.25, -1 -> 0.75, 0 -> 0.5
            if j_mb == 1:
                samples[s, i] = 0.25 + 0.1 * RNG.randn()
            elif j_mb == -1:
                samples[s, i] = 0.75 + 0.1 * RNG.randn()
            else:
                samples[s, i] = RNG.random()
        samples[s] = samples[s] % 1.0
    return samples


def strategy_D_smooth_scan(N, bases, d, n_samples):
    """
    Smooth number scanning.

    Scan integers near sqrt(N) for B-smooth values.
    For each smooth value, extract the exponent vector over the factor base.
    Project the exponent vector into [0,1)^d to get a "sample".
    """
    sqrt_N = math.isqrt(N)
    B_bound = max(bases)
    samples = np.zeros((n_samples, d))
    base_set = set(bases)

    found = 0
    scan_offset = 0
    max_scan = max(50000, n_samples * 100)

    while found < n_samples and scan_offset < max_scan:
        val = (sqrt_N + scan_offset) ** 2 - N
        scan_offset += 1
        if val <= 0:
            continue

        # Try to factor val over the factor base
        remaining = abs(val)
        exponents = np.zeros(d)
        for i, b in enumerate(bases):
            while remaining % b == 0:
                remaining //= b
                exponents[i] += 1
        if remaining == 1:
            # Smooth! Use fractional parts of exponents / (some scale)
            # The exponent vector is a relation candidate; project to [0,1)
            scale = max(np.max(np.abs(exponents)), 1)
            samples[found] = (exponents / (2.0 * scale)) % 1.0
            found += 1

    # Fill remaining with uniform (if not enough smooth found)
    if found < n_samples:
        samples[found:] = RNG.random((n_samples - found, d))

    return samples


def strategy_E_character_sum(N, bases, d, n_samples):
    """
    Character sum estimation.

    Estimate Dirichlet character sums mod N using partial Euler products.
    For each sample, pick a random "frequency" and estimate the character
    sum contribution from each base prime.

    The idea: Dirichlet characters mod N decompose as chi_p x chi_q (CRT).
    If we could estimate chi(b_i) for the right character, we'd get dual info.
    """
    samples = np.zeros((n_samples, d))

    for s in range(n_samples):
        # Pick a random "frequency" k (analogy to Regev's phase)
        k = random.randint(1, N - 1)
        for i, b in enumerate(bases):
            # Partial character sum: sum_{m=1}^{M} e^{2 pi i k m / N} * indicator(gcd(m,N)=1)
            # Truncated to a small range for efficiency
            M = min(200, N)
            char_sum_real = 0.0
            char_sum_imag = 0.0
            for m in range(1, M + 1):
                if math.gcd(m, N) == 1:
                    phase = 2.0 * math.pi * k * m / N
                    # Weight by whether m relates to b_i
                    weight = 1.0 / (1.0 + abs(m % b - b // 2))
                    char_sum_real += math.cos(phase) * weight
                    char_sum_imag += math.sin(phase) * weight

            # Extract a coordinate from the character sum
            angle = math.atan2(char_sum_imag, char_sum_real)
            samples[s, i] = (angle / (2.0 * math.pi)) % 1.0

    return samples


def strategy_F_power_residue(N, bases, d, n_samples):
    """
    Power residue probing.

    Compute b_i^{(N-1)/M} mod N for various small M.
    This probes the M-th roots of unity in (Z/NZ)*.
    The result encodes information about gcd(M, p-1) vs gcd(M, q-1).
    """
    samples = np.zeros((n_samples, d))
    small_M = [m for m in range(2, 50) if (N - 1) % m == 0]
    if not small_M:
        # If no exact divisors, use small primes anyway (result will be 1 or nontrivial)
        small_M = small_primes(30)

    for s in range(n_samples):
        # Pick a random M divisor or small prime
        M = small_M[s % len(small_M)] if small_M else 2
        # Also pick a random base multiplier for variety
        a = random.randint(1, N - 1)
        while math.gcd(a, N) != 1:
            a = random.randint(1, N - 1)

        exp = (N - 1) // M if (N - 1) % M == 0 else (N - 1) // max(math.gcd(N - 1, M), 1)

        for i, b in enumerate(bases):
            # Compute (a * b_i)^exp mod N
            val = pow((a * b) % N, exp, N)
            # Map the residue to [0, 1): use val / N as the coordinate
            samples[s, i] = val / N

    return samples


# =============================================================================
# Step 4: Run the experiment
# =============================================================================

def run_one_size(bits, n_samples=NUM_SAMPLES):
    """Run all strategies for a single semiprime size."""
    print(f"\n{'=' * 74}")
    print(f"  N size: {bits} bits, {n_samples} samples per strategy")
    print(f"{'=' * 74}")

    N, p, q = generate_semiprime(bits)
    print(f"  N = {N} = {p} * {q}  (actual: {N.bit_length()} bits)")

    # Build L_R
    t0 = time.time()
    basis, bases, d, dlogs_p, dlogs_q = ensure_full_rank_LR(N, p, q)
    t_lr = time.time() - t0

    if basis is None or len(basis) < 1:
        print(f"  SKIP: could not build L_R with sufficient rank")
        return None

    rank = len(basis)
    print(f"  Factor base: {bases}  (d = {d})")
    print(f"  L_R rank: {rank}  (built in {t_lr:.2f}s)")

    # Verify a few basis vectors
    for idx, v in enumerate(basis[:3]):
        chk_p = sum(int(v[i]) * dlogs_p[i] for i in range(d)) % (p - 1)
        chk_q = sum(int(v[i]) * dlogs_q[i] for i in range(d)) % (q - 1)
        ok = "ok" if (chk_p == 0 and chk_q == 0) else "FAIL"
        nrm = np.linalg.norm(v)
        print(f"    basis[{idx}] = {v.astype(int).tolist()[:8]}{'...' if d > 8 else ''}  "
              f"norm={nrm:.2f}  verify={ok}")

    # Compute dual lattice info
    # L* = {y in R^d : <x, y> in Z for all x in L_R}
    # If B is (rank x d), then B^T @ B is (d x d).
    # Dual basis: B_star = B^T (B B^T)^{-1} if rank < d.
    # For measuring dist(w, L*/Z^d): use || frac(B @ w) ||
    # This works because w in L* iff B @ w in Z^rank.
    print(f"\n  Distance metric: || frac(B_LR @ w) || where frac maps to [-0.5, 0.5)")
    print(f"  Perfect dual sample: distance = 0")
    print(f"  Expected distance for uniform random in [0,1)^d:")
    # For rank independent uniform [-0.5, 0.5) components, E[||x||^2] = rank/12
    # so E[||x||] ~ sqrt(rank/12)
    expected_uniform = math.sqrt(rank / 12.0)
    print(f"    theoretical ~ sqrt(rank/12) = {expected_uniform:.4f}")

    # Generate samples from each strategy
    strategies = {}

    print(f"\n  Generating samples...")

    t0 = time.time()
    strategies["A: Uniform random"] = strategy_A_uniform(d, n_samples)
    print(f"    A: Uniform random          ({time.time() - t0:.2f}s)")

    t0 = time.time()
    strategies["B: Random walk"] = strategy_B_random_walk(N, bases, d, n_samples)
    print(f"    B: Random walk             ({time.time() - t0:.2f}s)")

    t0 = time.time()
    strategies["C: Jacobi patterns"] = strategy_C_jacobi(N, bases, d, n_samples)
    print(f"    C: Jacobi patterns         ({time.time() - t0:.2f}s)")

    t0 = time.time()
    strategies["D: Smooth scanning"] = strategy_D_smooth_scan(N, bases, d, n_samples)
    print(f"    D: Smooth scanning         ({time.time() - t0:.2f}s)")

    t0 = time.time()
    strategies["E: Character sums"] = strategy_E_character_sum(N, bases, d, n_samples)
    print(f"    E: Character sums          ({time.time() - t0:.2f}s)")

    t0 = time.time()
    strategies["F: Power residue"] = strategy_F_power_residue(N, bases, d, n_samples)
    print(f"    F: Power residue           ({time.time() - t0:.2f}s)")

    # Measure distances
    print(f"\n  {'Strategy':<28s} {'mean':>8s} {'median':>8s} {'std':>8s} "
          f"{'<0.1':>6s} {'<0.3':>6s} {'<0.5':>6s} {'noise_ratio':>12s}")
    print(f"  {'-' * 94}")

    results = {}
    baseline_mean = None

    for name, samples in strategies.items():
        dists = batch_distances(samples, basis)
        mean_d = np.mean(dists)
        median_d = np.median(dists)
        std_d = np.std(dists)
        frac_01 = np.mean(dists < 0.1)
        frac_03 = np.mean(dists < 0.3)
        frac_05 = np.mean(dists < 0.5)

        if "Uniform" in name:
            baseline_mean = mean_d

        noise_ratio = mean_d / baseline_mean if baseline_mean and baseline_mean > 0 else float('nan')

        flag = ""
        if noise_ratio < 0.95 and "Uniform" not in name:
            flag = " <-- CORRELATED?"
        elif noise_ratio > 1.05 and "Uniform" not in name:
            flag = " (anti-corr)"

        print(f"  {name:<28s} {mean_d:8.4f} {median_d:8.4f} {std_d:8.4f} "
              f"{frac_01:6.3f} {frac_03:6.3f} {frac_05:6.3f} {noise_ratio:12.4f}{flag}")

        results[name] = {
            "mean": mean_d, "median": median_d, "std": std_d,
            "frac_01": frac_01, "frac_03": frac_03, "frac_05": frac_05,
            "noise_ratio": noise_ratio,
        }

    return results


def main():
    print("=" * 74)
    print("  DUAL LATTICE SAMPLING EXPERIMENT")
    print("  Can classical processes produce vectors correlated with L*/Z^d?")
    print("=" * 74)
    print()
    print("  Regev's quantum algorithm samples from L*/Z^d with distance")
    print("  2^{-O(sqrt(n))} from a true dual element.")
    print("  We test 6 classical strategies and measure their noise ratio:")
    print("    noise_ratio = (mean dist from strategy) / (mean dist from uniform)")
    print("    < 1 means the strategy produces vectors closer to L* than random.")
    print()

    bit_sizes = [10, 14, 18, 22, 26]
    all_results = {}

    for bits in bit_sizes:
        results = run_one_size(bits)
        if results is not None:
            all_results[bits] = results

    # =================================================================
    # Summary table across all sizes
    # =================================================================
    print(f"\n\n{'=' * 74}")
    print(f"  SUMMARY: Noise ratios across bit sizes")
    print(f"  (noise_ratio < 1 = correlated with dual; 1.0 = random baseline)")
    print(f"{'=' * 74}")

    strategy_names = [
        "A: Uniform random",
        "B: Random walk",
        "C: Jacobi patterns",
        "D: Smooth scanning",
        "E: Character sums",
        "F: Power residue",
    ]

    header = f"  {'Strategy':<28s}"
    for bits in bit_sizes:
        if bits in all_results:
            header += f" {bits:>6d}b"
    header += "     trend"
    print(header)
    print(f"  {'-' * (28 + 8 * len(all_results) + 12)}")

    for name in strategy_names:
        row = f"  {name:<28s}"
        ratios = []
        for bits in bit_sizes:
            if bits in all_results and name in all_results[bits]:
                r = all_results[bits][name]["noise_ratio"]
                row += f" {r:7.4f}"
                ratios.append(r)
            else:
                row += f"     n/a"

        # Trend: is the noise ratio decreasing (improving) with size?
        if len(ratios) >= 2 and "Uniform" not in name:
            diffs = [ratios[i + 1] - ratios[i] for i in range(len(ratios) - 1)]
            avg_diff = sum(diffs) / len(diffs)
            if avg_diff < -0.02:
                trend = "improving"
            elif avg_diff > 0.02:
                trend = "worsening"
            else:
                trend = "flat"
            row += f"  {trend}"
        else:
            row += "  (baseline)" if "Uniform" in name else "  n/a"

        print(row)

    # Final verdict
    print(f"\n  {'=' * 70}")
    print(f"  VERDICT")
    print(f"  {'=' * 70}")

    any_correlated = False
    for name in strategy_names:
        if "Uniform" in name:
            continue
        sustained = True
        for bits in bit_sizes:
            if bits in all_results and name in all_results[bits]:
                if all_results[bits][name]["noise_ratio"] >= 0.97:
                    sustained = False
                    break
        if sustained and any(
            bits in all_results and name in all_results[bits]
            and all_results[bits][name]["noise_ratio"] < 0.97
            for bits in bit_sizes
        ):
            any_correlated = True
            print(f"  {name}: SUSTAINED correlation detected across sizes!")

    if not any_correlated:
        # Check for any single-size correlations
        best_name = None
        best_ratio = 1.0
        best_bits = None
        for bits in bit_sizes:
            if bits not in all_results:
                continue
            for name in strategy_names:
                if "Uniform" in name:
                    continue
                if name in all_results[bits]:
                    r = all_results[bits][name]["noise_ratio"]
                    if r < best_ratio:
                        best_ratio = r
                        best_name = name
                        best_bits = bits

        if best_ratio < 0.95:
            print(f"  Weak signal: {best_name} at {best_bits} bits (ratio={best_ratio:.4f})")
            print(f"  But no strategy shows sustained improvement across all sizes.")
        else:
            print(f"  No classical strategy produced vectors correlated with L*/Z^d.")
            print(f"  Best noise ratio: {best_ratio:.4f} ({best_name} at {best_bits}b)")

    print(f"\n  This is consistent with the expectation that sampling from L*/Z^d")
    print(f"  requires quantum interference (Regev's algorithm), and no polynomial-")
    print(f"  time classical process can produce even weakly correlated samples.")
    print()


if __name__ == "__main__":
    main()
