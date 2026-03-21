#!/usr/bin/env python3
"""
LATTICE RECOVERY FROM CLASSICAL ORBIT

The visualization showed: classical orbit points lie on a 2D lattice.
The lattice vectors v_α and v_β encode the factorization.

Algorithm:
1. Compute classical orbit {(b1^k mod N, b2^k mod N)/N} for k = 0..T
2. Compute 2D pairwise differences → autocorrelation on the torus
3. Peaks in the autocorrelation = lattice vectors
4. From lattice vectors → v_α, v_β → p-1, q-1 → factor N

Key questions:
- How does orbit coverage scale with N?
- Do autocorrelation peaks survive as coverage drops?
- Can we extract v_α and v_β from the peaks?
- Does this give a polynomial-time factoring algorithm?
"""

import math
import random
import sys
import time
import numpy as np
from collections import Counter
from sympy import nextprime

sys.stdout.reconfigure(line_buffering=True)


def primitive_root(p):
    if p == 2: return 1
    phi = p - 1
    factors = set()
    n = phi
    d = 2
    while d * d <= n:
        while n % d == 0: factors.add(d); n //= d
        d += 1
    if n > 1: factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    return None


def discrete_log(a, g, p):
    a = a % p
    if a == 0: return 0
    cur = 1
    for k in range(p):
        if cur == a: return k
        cur = (cur * g) % p
    return 0


def multiplicative_order(a, n):
    if math.gcd(a, n) > 1: return 0
    order = 1
    val = a % n
    while val != 1:
        val = (val * a) % n
        order += 1
        if order > n: return 0
    return order


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, hi))
    while q == p:
        q = nextprime(random.randint(lo, hi))
    if p > q: p, q = q, p
    return p * q, p, q


def classical_orbit(b1, b2, N, T):
    """Generate T points of the classical orbit on [0,1)^2."""
    points = np.zeros((T, 2))
    v1, v2 = 1, 1
    for k in range(T):
        points[k, 0] = v1 / N
        points[k, 1] = v2 / N
        v1 = (v1 * b1) % N
        v2 = (v2 * b2) % N
    return points


def autocorrelation_2d(points, resolution=500):
    """
    Compute 2D autocorrelation of a point set on the torus [0,1)^2.
    For each pair of points, compute the difference mod 1.
    Bin the differences into a 2D histogram.
    Peaks = lattice vectors.
    """
    n = len(points)

    # For large n, subsample pairwise differences
    max_pairs = min(n * (n - 1) // 2, 2000000)
    if n > 3000:
        # Random subsample
        indices = random.sample(range(n), min(n, 3000))
        pts = points[indices]
    else:
        pts = points

    n_pts = len(pts)

    # Compute all pairwise differences on the torus
    # diff[i,j] = (pts[i] - pts[j]) mod 1, shifted to [-0.5, 0.5)
    diffs = []
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            d = pts[i] - pts[j]
            d = d - np.round(d)  # wrap to [-0.5, 0.5)
            diffs.append(d)

    if not diffs:
        return None, None

    diffs = np.array(diffs)

    # 2D histogram
    R = resolution
    hist, xedges, yedges = np.histogram2d(
        diffs[:, 0], diffs[:, 1],
        bins=R, range=[[-0.5, 0.5], [-0.5, 0.5]]
    )

    # Normalize
    hist = hist / hist.sum()

    # Find peaks (exclude the origin peak)
    # The origin (0,0) is always a peak (self-differences)
    center = R // 2

    # Find local maxima
    peaks = []
    for i in range(1, R - 1):
        for j in range(1, R - 1):
            if (abs(i - center) < 2 and abs(j - center) < 2):
                continue  # skip origin
            val = hist[i, j]
            if (val > hist[i-1, j] and val > hist[i+1, j] and
                val > hist[i, j-1] and val > hist[i, j+1] and
                val > 0):
                x = (xedges[i] + xedges[i+1]) / 2
                y = (yedges[j] + yedges[j+1]) / 2
                peaks.append((val, x, y))

    peaks.sort(reverse=True)
    return hist, peaks[:20]


def try_factor_from_peaks(peaks, N, bases):
    """
    Given autocorrelation peaks (candidate lattice vectors),
    try to extract p and q.

    If v_α = (dlp1/(p-1), dlp2/(p-1)), then:
    - v_α[0] / v_α[1] = dlp1 / dlp2 (ratio of discrete logs, independent of p)
    - |v_α| = sqrt(dlp1^2 + dlp2^2) / (p-1)
    - So p-1 = sqrt(dlp1^2 + dlp2^2) / |v_α|

    But we don't know dlp1 and dlp2. However:
    - (p-1) * v_α[0] = dlp1 (an integer!)
    - (p-1) * v_α[1] = dlp2 (an integer!)

    So for each candidate peak (x, y):
    - Try p-1 = k for k = 1, 2, ..., sqrt(N)
    - Check if k*x and k*y are both close to integers
    - If yes, and k+1 is prime, and N mod (k+1) == 0 → factor found!

    More efficiently: the peak coordinates are approximately a/m for small integers a
    and m = p-1 or q-1. Use continued fractions to find m.
    """
    factors_found = []

    for val, px, py in peaks:
        # Try each coordinate separately
        for coord in [px, py]:
            if abs(coord) < 1e-10:
                continue
            # CF of |coord| to find denominator
            cf = []
            x = abs(coord)
            for _ in range(20):
                a = int(math.floor(x))
                cf.append(a)
                frac = x - a
                if abs(frac) < 1e-12:
                    break
                x = 1.0 / frac
                if x > 1e10:
                    break

            # Build convergents
            h_prev, h_curr = 0, 1
            k_prev, k_curr = 1, 0
            for a in cf:
                h_prev, h_curr = h_curr, a * h_curr + h_prev
                k_prev, k_curr = k_curr, a * k_curr + k_prev

                if k_curr < 2 or k_curr > int(math.sqrt(N)) * 2:
                    continue

                # k_curr is a candidate for (p-1) or (q-1)
                p_cand = k_curr + 1
                if N % p_cand == 0 and p_cand > 1 and p_cand < N:
                    factors_found.append(('cf_denom', p_cand, coord, k_curr))

        # Try the 2D approach: find m such that m*px ≈ integer AND m*py ≈ integer
        for m in range(2, min(int(math.sqrt(N)) + 10, 100000)):
            r1 = m * abs(px)
            r2 = m * abs(py)
            err1 = abs(r1 - round(r1))
            err2 = abs(r2 - round(r2))
            if err1 < 0.02 and err2 < 0.02:
                p_cand = m + 1
                if N % p_cand == 0 and p_cand > 1 and p_cand < N:
                    factors_found.append(('2d_period', p_cand, (px, py), m))
                    break
                # Also try m itself as a factor
                if N % m == 0 and m > 1 and m < N:
                    factors_found.append(('2d_period_m', m, (px, py), m))
                    break

    return factors_found


# =========================================================================
# Main experiment
# =========================================================================

def experiment(bits_list=None, num_instances=15):
    if bits_list is None:
        bits_list = [12, 16, 20, 24, 28, 32]

    bases = [2, 3]

    print(f"{'='*75}")
    print(f"  LATTICE RECOVERY FROM CLASSICAL ORBIT")
    print(f"  Can 2D autocorrelation extract v_α and v_β from classical points?")
    print(f"{'='*75}")

    for bits in bits_list:
        print(f"\n{'='*75}")
        print(f"  {bits}-BIT SEMIPRIMES")
        print(f"{'='*75}")

        n_factored = 0
        coverages = []
        total = 0
        t0 = time.time()

        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)
            total += 1

            # True lattice vectors
            g_p = primitive_root(p)
            g_q = primitive_root(q)
            dlp = [discrete_log(b % p, g_p, p) for b in bases]
            dlq = [discrete_log(b % q, g_q, q) for b in bases]

            v_alpha_true = (dlp[0] / (p-1), dlp[1] / (p-1))
            v_beta_true = (dlq[0] / (q-1), dlq[1] / (q-1))

            # Classical orbit coverage
            ord_p1 = multiplicative_order(bases[0], p)
            ord_q1 = multiplicative_order(bases[0], q)
            orbit_period = ord_p1 * ord_q1 // math.gcd(ord_p1, ord_q1)
            full_group = (p - 1) * (q - 1)
            coverage = orbit_period / full_group
            coverages.append(coverage)

            # Generate classical orbit
            T = min(orbit_period, 5000)  # Cap at 5000 for speed
            points = classical_orbit(bases[0], bases[1], N, T)

            # 2D autocorrelation
            res = min(300, max(100, int(math.sqrt(T))))
            hist, peaks = autocorrelation_2d(points, resolution=res)
            if peaks is None:
                continue

            # Try to factor from peaks
            factors = try_factor_from_peaks(peaks, N, bases)
            factored = len(factors) > 0

            if factored:
                n_factored += 1

            if inst < 3:
                print(f"\n  Instance {inst+1}: N = {N} = {p} × {q}")
                print(f"    Coverage: {coverage:.4f} ({orbit_period}/{full_group})")
                print(f"    T = {T} orbit points, resolution = {res}")
                print(f"    True v_α = ({v_alpha_true[0]:.6f}, {v_alpha_true[1]:.6f})")
                print(f"    True v_β = ({v_beta_true[0]:.6f}, {v_beta_true[1]:.6f})")
                print(f"    Top 5 autocorrelation peaks:")
                for i, (val, px, py) in enumerate(peaks[:5]):
                    # Check if peak matches a true vector
                    match_a = (abs(abs(px) - abs(v_alpha_true[0])) < 0.02 and
                               abs(abs(py) - abs(v_alpha_true[1])) < 0.02)
                    match_b = (abs(abs(px) - abs(v_beta_true[0])) < 0.02 and
                               abs(abs(py) - abs(v_beta_true[1])) < 0.02)
                    tag = ""
                    if match_a: tag = " ← MATCHES v_α!"
                    if match_b: tag = " ← MATCHES v_β!"
                    print(f"      peak {i}: ({px:+.6f}, {py:+.6f}) strength={val:.6f}{tag}")

                if factors:
                    print(f"    FACTORED! via {factors[0]}")
                else:
                    print(f"    Not factored from peaks")

        elapsed = time.time() - t0
        mean_cov = np.mean(coverages)
        print(f"\n  Summary ({total} instances, {elapsed:.1f}s):")
        print(f"    Mean coverage: {mean_cov:.4f}")
        print(f"    Min coverage: {min(coverages):.4f}")
        print(f"    Factored via autocorrelation: {n_factored}/{total} ({n_factored/total:.0%})")

    print(f"\n{'='*75}")
    print(f"  COVERAGE SCALING")
    print(f"{'='*75}")
    print(f"  coverage = orbit_period / (p-1)(q-1)")
    print(f"  = lcm(ord_p(b), ord_q(b)) / (p-1)(q-1)")
    print(f"  When ord_p(b) | p-1 and ord_q(b) | q-1:")
    print(f"    coverage = lcm(ord_p, ord_q) / ((p-1)(q-1))")
    print(f"  For b=2: ord_p(2) divides p-1. Typical ord_p(2) ≈ (p-1)/small.")
    print(f"  coverage ≈ 1/gcd(p-1, q-1). For random p,q: gcd ≈ O(log N).")
    print(f"  So coverage ≈ 1/O(log N) — slowly decreasing.")
    print(f"  But the ABSOLUTE number of orbit points = orbit_period ≈ (p-1)(q-1)/gcd")
    print(f"  which is Θ(N/log N) — exponentially many points.")
    print(f"  Autocorrelation with poly(log N) points might not see the lattice.")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    experiment(bits_list=[12, 16, 20, 24, 28], num_instances=12)
