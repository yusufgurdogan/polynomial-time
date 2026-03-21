#!/usr/bin/env python3
"""
Partial Regev experiment: bridging smooth-number sieving with Regev's
quantum factoring framework.

KEY IDEA
--------
Regev's quantum algorithm produces samples from L*/Z^d (dual lattice mod
integers).  His classical post-processing (LLL on a recovery lattice)
recovers factors from these samples.  The quantum part is the bottleneck.

The NFS/QS classical approach finds SMOOTH NUMBERS that give vectors in
L_R (the primal relation lattice).  A full set of d-2 independent smooth
relations determines L_R completely, and from there L* = (B_R)^{-T} is
trivially computed.

THE BRIDGE: what if you have only r < rank(L_R) smooth relations?  You
get a sublattice S_r of L_R, whose dual S_r* is a SUPERLATTICE of L*.
Sampling from S_r*/Z^d gives NOISY samples from L*/Z^d because
S_r* contains L* plus extra "junk" cosets.

This experiment measures:
  (A) Can Regev-style LLL recovery extract L_R vectors from noisy dual
      samples obtained from partial smooth relations?
  (B) Can those L_R vectors be used to factor N?
  (C) How does the number of needed relations r scale with N?

For each semiprime N = pq we:
  1. Build the true L_R via discrete logs (cheating with known p, q).
  2. For r = 1, ..., rank(L_R): take r random L_R basis vectors as
     the "found" smooth relations.
  3. Build the partial dual from those r vectors.
  4. Sample from the partial dual mod Z^d.
  5. Feed samples into Regev's LLL recovery lattice.
  6. Check if LLL produces vectors in L_R (lattice recovery success).
  7. Check if those vectors factor N (factoring success).

TWO METRICS measured separately:
  - Lattice recovery rate: how often does LLL find ANY L_R vector?
  - Factoring rate: how often does the found vector actually factor N?
"""

import math
import sys
import time
import random
import numpy as np
from fpylll import IntegerMatrix, LLL

sys.stdout.reconfigure(line_buffering=True)

random.seed(42)
RNG = np.random.default_rng(42)


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
    if p == 2:
        return 1
    phi = p - 1
    factors = factorize_small(phi)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    return None


def discrete_log(a, g, p):
    a = a % p
    if a == 0:
        return 0
    cur = 1
    for k in range(p):
        if cur == a:
            return k
        cur = (cur * g) % p
    return 0


def factorize_small(n):
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


def generate_semiprime(bits):
    from sympy import nextprime
    half = bits // 2
    lo = 1 << (half - 1)
    hi = (1 << half) - 1
    for _ in range(10000):
        p = nextprime(random.randint(lo, hi))
        if p > hi:
            continue
        q = nextprime(random.randint(lo, hi))
        if q > hi or q == p:
            continue
        N = p * q
        if bits - 1 <= N.bit_length() <= bits + 1:
            return N, min(p, q), max(p, q)
    raise RuntimeError(f"failed to generate {bits}-bit semiprime")


# =============================================================================
# Build L_R
# =============================================================================

def build_LR(N, p, q, bases):
    """
    Build L_R = {e in Z^d : prod b_i^{e_i} = 1 mod N} using the
    augmented-lattice + LLL approach.
    """
    d = len(bases)
    g_p = primitive_root(p)
    g_q = primitive_root(q)

    dlogs_p = [discrete_log(b % p, g_p, p) for b in bases]
    dlogs_q = [discrete_log(b % q, g_q, q) for b in bases]

    dim = d + 2
    B = IntegerMatrix(dim, dim)
    for i in range(d):
        B[i, i] = 1
        B[i, d] = dlogs_p[i]
        B[i, d + 1] = dlogs_q[i]
    B[d, d] = p - 1
    B[d + 1, d + 1] = q - 1

    LLL.reduction(B)

    kernel_vecs = []
    for i in range(dim):
        v = [int(B[i, j]) for j in range(d)]
        aux = [int(B[i, j]) for j in range(d, dim)]
        if all(a == 0 for a in aux) and any(x != 0 for x in v):
            kernel_vecs.append(v)

    if not kernel_vecs:
        return None, dlogs_p, dlogs_q

    return np.array(kernel_vecs, dtype=np.int64), dlogs_p, dlogs_q


def build_full_LR(N, p, q, max_primes=30):
    """Build L_R with enough primes to get rank >= 3."""
    all_primes = [pr for pr in small_primes(500) if N % pr != 0]

    for d in range(5, min(max_primes + 1, len(all_primes) + 1)):
        bases = all_primes[:d]
        basis, dlogs_p, dlogs_q = build_LR(N, p, q, bases)
        if basis is not None and len(basis) >= 3:
            return basis, bases, d, dlogs_p, dlogs_q

    bases = all_primes[:max_primes]
    d = len(bases)
    basis, dlogs_p, dlogs_q = build_LR(N, p, q, bases)
    return basis, bases, d, dlogs_p, dlogs_q


def verify_LR(v, dlogs_p, dlogs_q, p, q):
    """Check v lies in L_R."""
    d = len(v)
    return (sum(int(v[i]) * dlogs_p[i] for i in range(d)) % (p - 1) == 0 and
            sum(int(v[i]) * dlogs_q[i] for i in range(d)) % (q - 1) == 0)


# =============================================================================
# Partial dual sampling
# =============================================================================

def partial_dual_samples(LR_basis, r, d, num_samples):
    """
    Given r rows of L_R, sample from the dual of the r-dim sublattice,
    reduced mod Z^d.
    """
    A = LR_basis[:r].astype(np.float64)

    try:
        AAt = A @ A.T
        AAt_inv = np.linalg.inv(AAt)
        A_pinv = A.T @ AAt_inv
    except np.linalg.LinAlgError:
        return None

    samples = np.zeros((num_samples, d))
    for s in range(num_samples):
        z = RNG.integers(-50, 51, size=r).astype(np.float64)
        y = A_pinv @ z
        samples[s] = y - np.floor(y)

    return samples


# =============================================================================
# Distance measurement
# =============================================================================

def dist_to_true_dual(samples, LR_basis):
    """Measure distance of each sample to L*/Z^d."""
    B = LR_basis.astype(np.float64)
    By = samples @ B.T
    frac = By - np.round(By)
    return np.linalg.norm(frac, axis=1)


# =============================================================================
# Factor extraction
# =============================================================================

def try_factor(v, bases, N, p, q):
    """
    Given an L_R vector, try to extract a factor.
    Uses the genus-based approach from lattice_compare2.py:
    a vector reveals a factor iff it is "genus-odd", meaning
    the product is a QR mod one prime and QNR mod the other.

    We directly test: compute prod bases[i]^v[i] mod p and mod q
    via their discrete log properties, then use gcd tricks.
    """
    d = len(v)
    if all(x == 0 for x in v):
        return None

    # Method 1: half-product (all even exponents)
    if all(e % 2 == 0 for e in v):
        half = 1
        for i in range(d):
            e = v[i] // 2
            if e > 0:
                half = (half * pow(bases[i], e, N)) % N
            elif e < 0:
                inv = pow(bases[i], -e, N)
                g = math.gcd(inv, N)
                if 1 < g < N:
                    return g
                half = (half * pow(inv, N - 2, N)) % N
        for delta in [1, -1]:
            g = math.gcd((half + delta) % N, N)
            if 1 < g < N:
                return g

    # Method 2: positive vs negative partial products
    pos = 1
    neg = 1
    for i in range(d):
        if v[i] > 0:
            pos = (pos * pow(bases[i], v[i], N)) % N
        elif v[i] < 0:
            neg = (neg * pow(bases[i], -v[i], N)) % N
    for a in [pos, neg]:
        for delta in [1, -1]:
            g = math.gcd((a + delta) % N, N)
            if 1 < g < N:
                return g
    for val in [pos - neg, pos + neg, pos * neg - 1, pos * neg + 1]:
        g = math.gcd(val % N, N)
        if 1 < g < N:
            return g

    # Method 3: Euler criterion on sub-products
    # Compute x = prod bases[i]^|v[i]| mod N
    x = pos * pow(neg, N - 2, N) % N  # = prod bases[i]^v[i] mod N (should be 1)
    # But the interesting thing is the half-power of sub-products
    for a in [pos, neg]:
        if a <= 1:
            continue
        y = pow(a, (N - 1) // 2, N)
        for delta in [1, -1]:
            g = math.gcd((y + delta) % N, N)
            if 1 < g < N:
                return g

    return None


def try_factor_from_set(vectors, bases, N, p, q):
    """
    Try factoring from individual vectors and their combinations.
    """
    d = len(vectors[0]) if vectors else 0

    # Individual
    for v in vectors:
        f = try_factor(v, bases, N, p, q)
        if f is not None:
            return f

    # Pairwise sums/differences
    n = len(vectors)
    for i in range(min(n, 15)):
        for j in range(i + 1, min(n, 15)):
            for sign in [1, -1]:
                combo = [vectors[i][k] + sign * vectors[j][k] for k in range(d)]
                if any(c != 0 for c in combo):
                    f = try_factor(combo, bases, N, p, q)
                    if f is not None:
                        return f

    # GF(2) kernel: find subsets summing to all-even
    if n >= 2 and d > 0:
        mat = np.array(vectors, dtype=np.int64) % 2
        for combo_idx in _gf2_kernel(mat, max_combos=10):
            combined = [0] * d
            for idx in combo_idx:
                for k in range(d):
                    combined[k] += vectors[idx][k]
            f = try_factor(combined, bases, N, p, q)
            if f is not None:
                return f

    return None


def _gf2_kernel(mat, max_combos=10):
    """Find subsets of rows of mat (over GF(2)) that sum to zero."""
    n, d = mat.shape
    aug = np.zeros((n, d + n), dtype=np.int64)
    aug[:, :d] = mat % 2
    for i in range(n):
        aug[i, d + i] = 1

    pivot_row = 0
    for col in range(d):
        found = -1
        for row in range(pivot_row, n):
            if aug[row, col] == 1:
                found = row
                break
        if found == -1:
            continue
        aug[[pivot_row, found]] = aug[[found, pivot_row]]
        for row in range(n):
            if row != pivot_row and aug[row, col] == 1:
                aug[row] = (aug[row] + aug[pivot_row]) % 2
        pivot_row += 1

    results = []
    for row in range(n):
        if all(aug[row, j] == 0 for j in range(d)):
            indices = [i for i in range(n) if aug[row, d + i] == 1]
            if len(indices) >= 2:
                results.append(indices)
                if len(results) >= max_combos:
                    break
    return results


# =============================================================================
# Regev's LLL recovery
# =============================================================================

def regev_recover(samples, d, N, bases, p, q, dlogs_p, dlogs_q):
    """
    Build Regev's recovery lattice from dual samples and run LLL.

    The lattice is (d+m) x (d+m):
        [  S*I_d   |    0   ]
        [ round(S*W^T) | I_m ]

    where W is d x m (columns = samples), S is a scaling parameter.

    Short vectors from LLL with last m coords small and first d coords
    nonzero are candidate L_R vectors.

    Returns (found_LR_vectors, factored, factor).
    """
    m = len(samples)
    found_LR = []

    # Try multiple scaling parameters
    for S in _scaling_schedule(N, d):
        dim = d + m
        B = IntegerMatrix(dim, dim)

        for i in range(d):
            B[i, i] = S

        for j in range(m):
            for i in range(d):
                B[d + j, i] = int(round(S * samples[j, i]))
            B[d + j, d + j] = 1

        try:
            LLL.reduction(B)
        except Exception:
            continue

        for row_idx in range(dim):
            v = [int(B[row_idx, j]) for j in range(d)]
            if all(x == 0 for x in v):
                continue
            norm_v = math.sqrt(sum(x * x for x in v))
            if norm_v > 80 * d:
                continue
            if verify_LR(v, dlogs_p, dlogs_q, p, q):
                found_LR.append(list(v))

    # Deduplicate
    unique = []
    seen = set()
    for v in found_LR:
        key = tuple(v)
        neg = tuple(-x for x in v)
        if key not in seen and neg not in seen:
            seen.add(key)
            unique.append(v)

    if not unique:
        return [], False, None

    factor = try_factor_from_set(unique, bases, N, p, q)
    return unique, factor is not None, factor


def _scaling_schedule(N, d):
    """Generate a sequence of scaling parameters to try."""
    vals = set()
    for mult in [0.5, 1, 2, 4, 8, 16, 32, 64]:
        vals.add(max(3, int(mult * N ** 0.25)))
    vals.add(max(3, int(N ** 0.5)))
    vals.add(max(3, int(N ** 0.125 * d)))
    # Also try small values (important for small N)
    for s in [3, 5, 10, 20, 50, 100]:
        vals.add(s)
    return sorted(vals)


# =============================================================================
# Direct L_R recovery (bypass Regev, use dual directly)
# =============================================================================

def direct_LR_recovery(samples, d, N, bases, p, q, dlogs_p, dlogs_q):
    """
    Alternative recovery: instead of Regev's lattice, use the dual
    samples to build an integer-relation lattice directly.

    If w is a dual sample (close to L*/Z^d), then for any x in L_R,
    <w, x> is close to an integer.  So round(<w, x>) = <w, x> + noise.

    Build the lattice of integer vectors x such that <w_i, x> ~= integer
    for all samples w_i.  This is the lattice with basis:
        rows of [round(S * W^T)] where W has columns = samples,
    intersected with Z^d.

    We do this by building an augmented system and running LLL.
    """
    m = len(samples)
    found_LR = []

    for S in [10, 50, 100, 500, 1000]:
        # Build (m+d) x d matrix: W^T scaled and rounded, stacked with S*I_d
        # The short vectors in the row lattice of this system correspond
        # to approximate integer relations.
        dim = d + m
        B = IntegerMatrix(dim, d + m)

        # Top block: S * I_d | 0
        for i in range(d):
            B[i, i] = S

        # Bottom block: round(S * w_j) | I_m (one row per sample)
        for j in range(m):
            for i in range(d):
                B[d + j, i] = int(round(S * samples[j, i]))
            B[d + j, d + j] = 1

        try:
            LLL.reduction(B)
        except Exception:
            continue

        for row_idx in range(dim):
            v = [int(B[row_idx, j]) for j in range(d)]
            if all(x == 0 for x in v):
                continue
            norm_v = math.sqrt(sum(x * x for x in v))
            if norm_v > 80 * d:
                continue
            if verify_LR(v, dlogs_p, dlogs_q, p, q):
                found_LR.append(list(v))

    # Deduplicate
    unique = []
    seen = set()
    for v in found_LR:
        key = tuple(v)
        neg = tuple(-x for x in v)
        if key not in seen and neg not in seen:
            seen.add(key)
            unique.append(v)

    if not unique:
        return [], False, None

    factor = try_factor_from_set(unique, bases, N, p, q)
    return unique, factor is not None, factor


# =============================================================================
# Main experiment
# =============================================================================

def run_one_instance(N, p, q, num_trials=5):
    """
    Sweep r = 1..rank and measure recovery and factoring success.
    """
    result = build_full_LR(N, p, q)
    basis, bases, d, dlogs_p, dlogs_q = result

    if basis is None or len(basis) < 3:
        return None

    rank = len(basis)
    for idx in range(rank):
        if not verify_LR(basis[idx], dlogs_p, dlogs_q, p, q):
            return None

    m = d + 4
    results_by_r = {}

    for r in range(1, rank + 1):
        lr_found = 0
        factored = 0

        for trial in range(num_trials):
            perm = RNG.permutation(rank)
            partial = basis[perm[:r]]

            samples = partial_dual_samples(partial, r, d, m)
            if samples is None:
                continue

            # Method 1: Regev recovery
            vecs1, ok1, f1 = regev_recover(
                samples, d, N, bases, p, q, dlogs_p, dlogs_q
            )

            # Method 2: direct recovery
            vecs2, ok2, f2 = direct_LR_recovery(
                samples, d, N, bases, p, q, dlogs_p, dlogs_q
            )

            # Merge results
            all_vecs = vecs1 + vecs2
            if all_vecs:
                lr_found += 1
            if ok1 or ok2:
                factored += 1
            elif all_vecs:
                # Try combined factor extraction
                f3 = try_factor_from_set(all_vecs, bases, N, p, q)
                if f3 is not None:
                    factored += 1

        results_by_r[r] = {
            "lr_rate": lr_found / num_trials,
            "factor_rate": factored / num_trials,
        }

    return {
        "N": N, "p": p, "q": q, "bits": N.bit_length(),
        "d": d, "rank": rank, "m": m,
        "results_by_r": results_by_r,
    }


def run_bit_size(bits, num_instances=4, num_trials=5):
    print(f"\n{'=' * 74}")
    print(f"  {bits}-BIT SEMIPRIMES")
    print(f"{'=' * 74}")

    lr_crossovers = []
    fac_crossovers = []

    for inst in range(num_instances):
        N, p, q = generate_semiprime(bits)
        print(f"\n  Instance {inst + 1}: N = {N} = {p} * {q}  "
              f"({N.bit_length()} bits)")

        # Noise profile
        result = build_full_LR(N, p, q)
        basis, bases, d, dlogs_p, dlogs_q = result
        if basis is None or len(basis) < 3:
            print(f"    SKIP (rank < 3)")
            continue

        rank = len(basis)
        m = d + 4

        print(f"    d={d}, rank={rank}, m={m}")
        print(f"    Noise vs r:")
        for r in range(1, rank + 1):
            perm = RNG.permutation(rank)
            partial = basis[perm[:r]]
            samples = partial_dual_samples(partial, r, d, m)
            if samples is not None:
                dists = dist_to_true_dual(samples, basis)
                print(f"      r={r}: mean_dist={np.mean(dists):.4f}  "
                      f"max_dist={np.max(dists):.4f}")

        # Main sweep
        t0 = time.time()
        res = run_one_instance(N, p, q, num_trials=num_trials)
        elapsed = time.time() - t0

        if res is None:
            print(f"    SKIP")
            continue

        rbr = res["results_by_r"]

        print(f"\n    {'r':>4s}  {'r/rank':>7s}  {'LR_found':>9s}  "
              f"{'factored':>9s}  {'bar (LR)':>15s}  {'bar (factor)'}")
        print(f"    {'----':>4s}  {'------':>7s}  {'--------':>9s}  "
              f"{'--------':>9s}  {'--------':>15s}  {'------------'}")

        lr_cross_r = None
        fac_cross_r = None

        for r in range(1, rank + 1):
            info = rbr[r]
            lr = info["lr_rate"]
            fc = info["factor_rate"]
            frac = r / rank
            bar_lr = "#" * int(lr * 30)
            bar_fc = "+" * int(fc * 30)

            markers = ""
            if lr_cross_r is None and lr >= 0.5:
                lr_cross_r = r
                markers += " <-LR50%"
            if fac_cross_r is None and fc >= 0.5:
                fac_cross_r = r
                markers += " <-FAC50%"

            print(f"    {r:4d}  {frac:7.3f}  {lr:9.1%}  {fc:9.1%}  "
                  f"{bar_lr:>15s}  {bar_fc}{markers}")

        if lr_cross_r is not None:
            lr_crossovers.append(lr_cross_r / rank)
            print(f"\n    LR recovery crossover: r={lr_cross_r}/{rank} "
                  f"= {lr_cross_r / rank:.2%}")
        else:
            print(f"\n    LR recovery: never reached 50%")

        if fac_cross_r is not None:
            fac_crossovers.append(fac_cross_r / rank)
            print(f"    Factoring crossover:   r={fac_cross_r}/{rank} "
                  f"= {fac_cross_r / rank:.2%}")
        else:
            print(f"    Factoring: never reached 50%")

        print(f"    Time: {elapsed:.1f}s")

    return {
        "bits": bits,
        "lr_fracs": lr_crossovers,
        "fac_fracs": fac_crossovers,
    }


def main():
    print("=" * 74)
    print("  PARTIAL REGEV EXPERIMENT")
    print("  Bridging smooth-number sieving with Regev's quantum framework")
    print("=" * 74)
    print()
    print("  Two metrics tracked separately:")
    print("    LR_found  = did LLL recover ANY valid L_R vector?")
    print("    factored  = did we extract a nontrivial factor of N?")
    print()

    bit_sizes = [16, 20, 24, 28, 32]
    all_results = {}

    for bits in bit_sizes:
        if bits <= 20:
            n_inst, n_trial = 4, 8
        elif bits <= 28:
            n_inst, n_trial = 3, 6
        else:
            n_inst, n_trial = 3, 4

        t0 = time.time()
        res = run_bit_size(bits, num_instances=n_inst, num_trials=n_trial)
        elapsed = time.time() - t0
        all_results[bits] = res
        print(f"\n  [{bits}-bit total: {elapsed:.1f}s]")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n\n{'=' * 74}")
    print(f"  SUMMARY")
    print(f"{'=' * 74}")

    print(f"\n  Crossover fractions (50% success threshold):")
    print(f"  {'bits':>5s}  {'LR recovery':>25s}  {'Factoring':>25s}")
    print(f"  {'-----':>5s}  {'-------------------------':>25s}  "
          f"{'-------------------------':>25s}")

    for bits in bit_sizes:
        res = all_results.get(bits, {})
        lr = res.get("lr_fracs", [])
        fc = res.get("fac_fracs", [])
        lr_str = (f"mean={np.mean(lr):.3f} ({len(lr)} instances)"
                  if lr else "no crossover")
        fc_str = (f"mean={np.mean(fc):.3f} ({len(fc)} instances)"
                  if fc else "no crossover")
        print(f"  {bits:5d}  {lr_str:>25s}  {fc_str:>25s}")

    # Verdict
    print(f"\n  {'=' * 68}")
    print(f"  VERDICT")
    print(f"  {'=' * 68}")

    any_lr = any(all_results.get(b, {}).get("lr_fracs", []) for b in bit_sizes)
    any_fac = any(all_results.get(b, {}).get("fac_fracs", []) for b in bit_sizes)

    if not any_lr:
        print(f"  LLL never recovered L_R vectors from partial-dual samples")
        print(f"  at any bit size.  The noise from missing smooth relations")
        print(f"  overwhelms the lattice signal that Regev's post-processing")
        print(f"  needs.")
        print()
        print(f"  KEY INSIGHT: Regev's quantum algorithm works because it")
        print(f"  produces samples at distance 2^{{-O(sqrt(n))}} from L*,")
        print(f"  which is exponentially small.  Partial smooth relations")
        print(f"  yield samples at distance ~O(1) from L* (as seen in the")
        print(f"  noise profiles above).  This O(1) vs 2^{{-O(sqrt(n))}} gap")
        print(f"  cannot be bridged by LLL -- it is a fundamental barrier.")
        print()
        print(f"  The smooth-to-Regev bridge requires either:")
        print(f"    (a) All smooth relations (= full NFS), OR")
        print(f"    (b) A quantum device to produce the missing signal.")
        print(f"  There is no useful middle ground at any scale tested.")
    elif not any_fac:
        print(f"  LLL recovered L_R vectors but none factored N.")
        print(f"  The recovered vectors are all in the 'genus-even'")
        print(f"  subspace (L_C) and do not split N.  Only 'genus-odd'")
        print(f"  vectors reveal factors, and those require finer control")
        print(f"  of the dual sampling than partial relations provide.")
    else:
        # We have factoring crossovers -- analyze scaling
        all_fac = []
        for bits in bit_sizes:
            all_fac.extend(all_results.get(bits, {}).get("fac_fracs", []))
        mean_f = np.mean(all_fac)
        print(f"  Factoring crossover found at mean {mean_f:.1%} of rank.")

        means_by_bits = []
        for bits in bit_sizes:
            fc = all_results.get(bits, {}).get("fac_fracs", [])
            if fc:
                means_by_bits.append((bits, np.mean(fc)))
        if len(means_by_bits) >= 2:
            first = means_by_bits[0][1]
            last = means_by_bits[-1][1]
            if last < first - 0.1:
                print(f"  Fraction DECREASES with N: potential speedup!")
            elif last > first + 0.1:
                print(f"  Fraction INCREASES with N: no asymptotic advantage.")
            else:
                print(f"  Fraction roughly CONSTANT: constant-factor savings.")

    print()


if __name__ == "__main__":
    main()
