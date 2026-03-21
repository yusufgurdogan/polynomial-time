#!/usr/bin/env python3
"""
ADDITIVE JACOBI MATRIX: A genuinely new approach to factoring.

For N = pq, construct the d×d matrix J where:
    J[i,j] = Jacobi_symbol(a_i + a_j, N)

Under CRT, this decomposes as a Hadamard (element-wise) product:
    J = L_p ⊙ L_q

where L_p[i,j] = Legendre((a_i + a_j) mod p, p).

Key insight: multiplicative Jacobi symbols (a/N) give ONE BIT per element
(genus information, known to be insufficient). But ADDITIVE Jacobi symbols
(a_i + a_j / N) depend on the FULL residues a_i mod p, creating O(d²)
constraints on O(d) unknowns.

Question: Can spectral analysis (SVD, correlation, tensor methods) of J
recover the hidden Hadamard factorization and thus factor N?
"""

import math
import random
import sys
import time
import numpy as np
from sympy import nextprime

sys.stdout.reconfigure(line_buffering=True)


def jacobi_symbol(a, n):
    """Compute the Jacobi symbol (a/n) for odd n > 0."""
    if n <= 0 or n % 2 == 0:
        return 0
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


def legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p) for prime p."""
    a = a % p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def generate_semiprime(bits):
    """Generate N = p*q with approximately `bits` bits."""
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
        if N.bit_length() >= bits - 1:
            return N, min(p, q), max(p, q)


# =============================================================================
# Core: Build the Additive Jacobi Matrix
# =============================================================================

def build_additive_jacobi_matrix(N, elements):
    """
    Build J[i,j] = Jacobi(a_i + a_j, N) for the given elements.
    Returns the matrix J (d x d) with entries in {-1, 0, 1}.
    """
    d = len(elements)
    J = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        for j in range(i, d):
            s = (elements[i] + elements[j]) % N
            if s == 0 or math.gcd(s, N) > 1:
                J[i, j] = 0
                J[j, i] = 0
            else:
                js = jacobi_symbol(s, N)
                J[i, j] = js
                J[j, i] = js
    return J


def build_legendre_matrix(elements, p):
    """
    Build L[i,j] = Legendre((a_i + a_j) mod p, p).
    This is one factor in the Hadamard decomposition J = L_p ⊙ L_q.
    """
    d = len(elements)
    L = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        for j in range(i, d):
            s = (elements[i] + elements[j]) % p
            if s == 0:
                L[i, j] = 0
                L[j, i] = 0
            else:
                L[i, j] = legendre_symbol(s, p)
                L[j, i] = L[i, j]

    return L


# =============================================================================
# Analysis: Spectral decomposition
# =============================================================================

def analyze_spectral(J, L_p, L_q, N, p, q, d):
    """
    Analyze the spectral structure of J and its Hadamard factors.
    """
    print(f"\n  --- Spectral Analysis ---")

    # Convert to float for SVD
    Jf = J.astype(np.float64)
    Lpf = L_p.astype(np.float64)
    Lqf = L_q.astype(np.float64)

    # SVD of J
    U_J, s_J, Vt_J = np.linalg.svd(Jf)
    # SVD of L_p
    U_p, s_p, Vt_p = np.linalg.svd(Lpf)
    # SVD of L_q
    U_q, s_q, Vt_q = np.linalg.svd(Lqf)

    # Ranks (effective)
    tol = 1e-6
    rank_J = np.sum(s_J > tol * s_J[0]) if s_J[0] > 0 else 0
    rank_p = np.sum(s_p > tol * s_p[0]) if s_p[0] > 0 else 0
    rank_q = np.sum(s_q > tol * s_q[0]) if s_q[0] > 0 else 0

    print(f"  Effective ranks: J={rank_J}, L_p={rank_p}, L_q={rank_q}")
    print(f"  Top 10 singular values of J:  {np.round(s_J[:10], 2)}")
    print(f"  Top 10 singular values of Lp: {np.round(s_p[:10], 2)}")
    print(f"  Top 10 singular values of Lq: {np.round(s_q[:10], 2)}")

    # Key question: does J's spectrum reveal the Hadamard factorization?
    # If L_p has rank r_p and L_q has rank r_q, then J = L_p ⊙ L_q
    # has rank at most r_p * r_q (by properties of Hadamard products).
    print(f"\n  Rank bound: rank(J) ≤ rank(Lp) × rank(Lq) = {rank_p * rank_q}")
    print(f"  Actual rank(J) = {rank_J}")
    print(f"  Ratio: {rank_J / max(1, rank_p * rank_q):.3f}")

    # Singular value decay
    if len(s_J) > 1 and s_J[0] > 0:
        ratio_12 = s_J[1] / s_J[0] if s_J[0] > 0 else 0
        ratio_23 = s_J[2] / s_J[1] if len(s_J) > 2 and s_J[1] > 0 else 0
        print(f"\n  Singular value ratios: σ₂/σ₁={ratio_12:.4f}, σ₃/σ₂={ratio_23:.4f}")

    return s_J, s_p, s_q


def analyze_correlation(J, L_p, L_q, N, p, q, d):
    """
    Test whether correlations in J reveal factoring information.
    """
    print(f"\n  --- Correlation Analysis ---")

    Jf = J.astype(np.float64)

    # Correlation matrix of rows of J
    # If rows with a_i ≡ a_j mod p are more correlated, this reveals p
    # (because L_p[i,:] = L_p[j,:] when a_i ≡ a_j mod p)

    corr = np.corrcoef(Jf)

    # Check: are rows with same residue mod p more correlated?
    same_p = []
    diff_p = []
    same_q = []
    diff_q = []

    elements = list(range(d))  # placeholder, actual elements stored elsewhere
    return corr


def analyze_row_clustering(J, elements, N, p, q, d):
    """
    Key test: rows of J corresponding to elements with the same
    residue mod p should be IDENTICAL in the L_p factor, hence
    more correlated in J.
    """
    print(f"\n  --- Row Clustering Analysis ---")

    Jf = J.astype(np.float64)

    # Group elements by residue mod p
    residues_p = {}
    for i, a in enumerate(elements):
        r = a % p
        if r not in residues_p:
            residues_p[r] = []
        residues_p[r].append(i)

    # Group by residue mod q
    residues_q = {}
    for i, a in enumerate(elements):
        r = a % q
        if r not in residues_q:
            residues_q[r] = []
        residues_q[r].append(i)

    # For pairs of rows: compute correlation
    # Same residue mod p → L_p rows identical → correlation from L_q agreement
    # Different residue mod p → L_p rows different → correlation from random

    same_p_corrs = []
    diff_p_corrs = []

    for i in range(d):
        for j in range(i + 1, d):
            # Correlation between row i and row j of J
            ri = Jf[i, :]
            rj = Jf[j, :]
            # Remove zero entries for fair comparison
            mask = (J[i, :] != 0) & (J[j, :] != 0)
            if np.sum(mask) < 3:
                continue
            ri_m = ri[mask]
            rj_m = rj[mask]
            if np.std(ri_m) < 1e-10 or np.std(rj_m) < 1e-10:
                continue
            corr = np.corrcoef(ri_m, rj_m)[0, 1]

            if elements[i] % p == elements[j] % p:
                same_p_corrs.append(corr)
            else:
                diff_p_corrs.append(corr)

    if same_p_corrs:
        print(f"  Same residue mod p: n={len(same_p_corrs)}, "
              f"mean_corr={np.mean(same_p_corrs):.4f}, "
              f"std={np.std(same_p_corrs):.4f}")
    else:
        print(f"  Same residue mod p: no pairs found (d too small relative to p)")

    if diff_p_corrs:
        print(f"  Diff residue mod p: n={len(diff_p_corrs)}, "
              f"mean_corr={np.mean(diff_p_corrs):.4f}, "
              f"std={np.std(diff_p_corrs):.4f}")

    # Statistical test: are same-p correlations significantly higher?
    if same_p_corrs and diff_p_corrs:
        from scipy import stats
        try:
            t_stat, p_val = stats.ttest_ind(same_p_corrs, diff_p_corrs, alternative='greater')
            print(f"  t-test (same > diff): t={t_stat:.3f}, p={p_val:.6f}")
            if p_val < 0.01:
                print(f"  *** SIGNIFICANT: same-residue pairs are more correlated! ***")
        except Exception:
            pass

    # Now test the KEY idea: can we recover the clustering from J alone?
    # Use spectral clustering on J's rows
    print(f"\n  --- Spectral Clustering (blind, from J only) ---")

    # Compute pairwise row correlations
    n_valid = min(d, 100)  # limit for speed
    corr_matrix = np.zeros((n_valid, n_valid))
    for i in range(n_valid):
        for j in range(n_valid):
            mask = (J[i, :n_valid] != 0) & (J[j, :n_valid] != 0)
            if np.sum(mask) < 3:
                corr_matrix[i, j] = 0
                continue
            ri = Jf[i, :n_valid][mask]
            rj = Jf[j, :n_valid][mask]
            if np.std(ri) < 1e-10 or np.std(rj) < 1e-10:
                corr_matrix[i, j] = 0
                continue
            corr_matrix[i, j] = np.corrcoef(ri, rj)[0, 1]

    # Spectral analysis of correlation matrix
    eigvals = np.linalg.eigvalsh(corr_matrix)
    eigvals = eigvals[::-1]  # descending
    print(f"  Top 10 eigenvalues of row-correlation matrix: {np.round(eigvals[:10], 3)}")

    # The number of large eigenvalues should correspond to the number of
    # distinct residue classes mod p (which is p itself, for d >> p)
    n_large = np.sum(eigvals > 1.0)  # eigenvalue > 1 is "significant"
    print(f"  Eigenvalues > 1.0: {n_large}")
    print(f"  Actual p = {p} (number of residue classes mod p)")
    print(f"  Actual q = {q} (number of residue classes mod q)")

    return same_p_corrs, diff_p_corrs


def analyze_gcd_from_svd(J, elements, N, p, q, d):
    """
    Attempt to extract factors from the SVD of J.

    If J = L_p ⊙ L_q, and L_p has rank ~ p/2, L_q has rank ~ q/2,
    then the leading singular vectors of J might align with the
    block structure induced by residues mod p or mod q.
    """
    print(f"\n  --- Factor Extraction from SVD ---")

    Jf = J.astype(np.float64)
    U, s, Vt = np.linalg.svd(Jf, full_matrices=False)

    # Take the leading singular vector
    u1 = U[:, 0]

    # Try to cluster elements based on the leading singular vector
    # Elements with same residue mod p should cluster

    # Sort elements by their u1 coordinate
    order = np.argsort(u1)

    # Look at differences between consecutive elements in sorted order
    # Large gaps might correspond to residue class boundaries
    sorted_u1 = u1[order]
    gaps = np.diff(sorted_u1)

    if len(gaps) > 0:
        # Find the largest gap
        max_gap_idx = np.argmax(gaps)
        max_gap = gaps[max_gap_idx]
        mean_gap = np.mean(gaps)
        print(f"  Leading SV: max_gap={max_gap:.4f} at idx {max_gap_idx}, "
              f"mean_gap={mean_gap:.4f}, ratio={max_gap/mean_gap:.2f}")

        # Split at the largest gap
        group_A = set(order[:max_gap_idx + 1])
        group_B = set(order[max_gap_idx + 1:])

        # Check: does this split correspond to a residue mod p or q?
        # For each group, check gcd of pairwise differences
        if len(group_A) >= 2 and len(group_B) >= 2:
            # Try: gcd of all (a_i - a_j) for i, j in same group
            diffs_A = []
            for i in group_A:
                for j in group_A:
                    if i < j:
                        diffs_A.append(abs(elements[i] - elements[j]) % N)

            # If all elements in group_A have same residue mod p,
            # then all diffs are divisible by p
            if diffs_A:
                g = diffs_A[0]
                for d_val in diffs_A[1:]:
                    g = math.gcd(g, d_val)
                g = math.gcd(g, N)
                print(f"  Group A ({len(group_A)} elements): gcd of diffs with N = {g}")
                if 1 < g < N:
                    print(f"  *** FACTOR FOUND: {g} ***")
                    return g

    # Try with more singular vectors
    for k in range(1, min(5, len(s))):
        if s[k] < 0.01 * s[0]:
            break
        uk = U[:, k]

        # k-means-style split
        median_uk = np.median(uk)
        group_A = [i for i in range(len(uk)) if uk[i] > median_uk]
        group_B = [i for i in range(len(uk)) if uk[i] <= median_uk]

        if len(group_A) >= 2:
            diffs = []
            for i in range(min(len(group_A), 50)):
                for j in range(i + 1, min(len(group_A), 50)):
                    diffs.append(abs(elements[group_A[i]] - elements[group_A[j]]) % N)
            if diffs:
                g = diffs[0]
                for d_val in diffs[1:]:
                    g = math.gcd(g, d_val)
                g = math.gcd(g, N)
                if 1 < g < N:
                    print(f"  SV[{k}]: *** FACTOR FOUND: {g} ***")
                    return g

    print(f"  No factor extracted from SVD.")
    return None


def analyze_hadamard_recovery(J, elements, N, p, q, d):
    """
    Can we recover L_p (or L_q) from J using the constraint that
    J = L_p ⊙ L_q and both L_p, L_q are Legendre sum matrices?

    A Legendre sum matrix has the property:
    L[i,j] depends only on (a_i + a_j) mod p.
    So rows i and j are identical iff a_i ≡ a_j mod p.

    We can test this WITHOUT knowing p:
    - Compute row similarity matrix S[i,j] = fraction of columns where J[i,:] * J[j,:] = 1
    - If a_i ≡ a_j mod p: L_p rows are identical → S[i,j] measures L_q agreement
    - If a_i ≢ a_j mod p: S[i,j] measures mixed agreement
    """
    print(f"\n  --- Hadamard Recovery ---")

    n = min(d, 80)  # limit for O(n³)

    # Compute agreement matrix: A[i,j] = fraction of positions where J[i,k] == J[j,k]
    # (only counting positions where both are nonzero)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            mask = (J[i, :n] != 0) & (J[j, :n] != 0)
            count = np.sum(mask)
            if count == 0:
                A[i, j] = A[j, i] = 0.5
                continue
            agree = np.sum(J[i, :n][mask] == J[j, :n][mask])
            A[i, j] = A[j, i] = agree / count

    # The agreement matrix should have BLOCK STRUCTURE:
    # - Pairs with same residue mod p: agreement determined by L_q correlation
    # - Pairs with different residue mod p: agreement ~ 0.5 (random)

    # Check: what's the distribution of agreement values?
    upper_tri = A[np.triu_indices(n, k=1)]

    same_p_agree = []
    diff_p_agree = []
    for i in range(n):
        for j in range(i + 1, n):
            if elements[i] % p == elements[j] % p:
                same_p_agree.append(A[i, j])
            else:
                diff_p_agree.append(A[i, j])

    print(f"  Agreement matrix statistics (d={n}):")
    print(f"    Overall: mean={np.mean(upper_tri):.4f}, std={np.std(upper_tri):.4f}")
    if same_p_agree:
        print(f"    Same mod p: n={len(same_p_agree)}, mean={np.mean(same_p_agree):.4f}, "
              f"std={np.std(same_p_agree):.4f}")
    if diff_p_agree:
        print(f"    Diff mod p: n={len(diff_p_agree)}, mean={np.mean(diff_p_agree):.4f}, "
              f"std={np.std(diff_p_agree):.4f}")

    # Gap between same and diff?
    if same_p_agree and diff_p_agree:
        gap = np.mean(same_p_agree) - np.mean(diff_p_agree)
        print(f"    GAP (same - diff): {gap:.4f}")
        if gap > 0.05:
            print(f"    *** DETECTABLE SIGNAL in agreement matrix! ***")

    # Spectral clustering of agreement matrix
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_A = eigvals_A[::-1]
    eigvecs_A = eigvecs_A[:, ::-1]
    print(f"\n  Top 10 eigenvalues of agreement matrix: {np.round(eigvals_A[:10], 3)}")

    # The number of "large" eigenvalues should relate to min(p, q)
    threshold = eigvals_A[0] * 0.1  # 10% of largest
    n_significant = np.sum(eigvals_A > threshold)
    print(f"  Significant eigenvalues (>10% of λ₁): {n_significant}")
    print(f"  Expected ≈ p = {p} (if d > p)")

    return A


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(bits, d_factor=3):
    """
    Run the additive Jacobi matrix experiment for a given bit size.
    d_factor: d = d_factor * sqrt(N) ... too big. Use d = d_factor * p instead.
    """
    N, p, q = generate_semiprime(bits)

    # Choose d proportional to p (so we get multiple elements per residue class)
    # For small N, d = 2*p to 3*p gives good coverage
    d = min(d_factor * p, 500)  # cap for speed

    print(f"\n{'='*75}")
    print(f"  N = {N} = {p} × {q}  ({bits} bits)")
    print(f"  d = {d} elements  (d/p = {d/p:.1f}, d/q = {d/q:.1f})")
    print(f"{'='*75}")

    # Choose random elements
    elements = []
    while len(elements) < d:
        a = random.randint(1, N - 1)
        if math.gcd(a, N) == 1:
            elements.append(a)

    # Build the additive Jacobi matrix
    t0 = time.time()
    J = build_additive_jacobi_matrix(N, elements)
    t_build = time.time() - t0
    print(f"\n  Built J ({d}×{d}) in {t_build:.2f}s")

    # Count entries
    n_plus = np.sum(J == 1)
    n_minus = np.sum(J == -1)
    n_zero = np.sum(J == 0)
    total = d * d
    print(f"  Entries: +1: {n_plus} ({100*n_plus/total:.1f}%), "
          f"-1: {n_minus} ({100*n_minus/total:.1f}%), "
          f"0: {n_zero} ({100*n_zero/total:.1f}%)")

    # Build the ground truth: L_p and L_q
    L_p = build_legendre_matrix(elements, p)
    L_q = build_legendre_matrix(elements, q)

    # Verify Hadamard decomposition
    # J should equal L_p * L_q (element-wise) wherever both are nonzero
    mask = (L_p != 0) & (L_q != 0) & (J != 0)
    if np.sum(mask) > 0:
        match = np.sum(J[mask] == (L_p[mask] * L_q[mask]))
        total_check = np.sum(mask)
        print(f"  Hadamard verification: {match}/{total_check} entries match "
              f"({100*match/total_check:.1f}%)")

    # Analysis 1: Spectral
    s_J, s_p, s_q = analyze_spectral(J, L_p, L_q, N, p, q, d)

    # Analysis 2: Row clustering
    same_corr, diff_corr = analyze_row_clustering(J, elements, N, p, q, d)

    # Analysis 3: Hadamard recovery
    A = analyze_hadamard_recovery(J, elements, N, p, q, d)

    # Analysis 4: Factor extraction
    factor = analyze_gcd_from_svd(J, elements, N, p, q, d)

    return {
        'bits': bits, 'N': N, 'p': p, 'q': q, 'd': d,
        'factor_found': factor is not None,
        'same_corr': np.mean(same_corr) if same_corr else None,
        'diff_corr': np.mean(diff_corr) if diff_corr else None,
    }


def run_scaling_test():
    """
    Test how the signal scales with N.
    The key question: does the gap between same-residue and
    different-residue correlations persist as N grows?
    """
    print(f"\n{'='*75}")
    print(f"  SCALING TEST: How does the signal scale with N?")
    print(f"{'='*75}")

    results = []
    for bits in [10, 12, 14, 16, 18, 20, 24]:
        try:
            result = run_experiment(bits, d_factor=3)
            results.append(result)
        except Exception as e:
            print(f"  Error at {bits} bits: {e}")

    # Summary
    print(f"\n\n{'='*75}")
    print(f"  SCALING SUMMARY")
    print(f"{'='*75}")
    print(f"  {'bits':>5s} {'p':>8s} {'q':>8s} {'d':>6s} {'d/p':>6s} "
          f"{'same_corr':>10s} {'diff_corr':>10s} {'gap':>8s} {'factor?':>8s}")
    print(f"  {'-'*72}")

    for r in results:
        gap = "n/a"
        if r['same_corr'] is not None and r['diff_corr'] is not None:
            gap = f"{r['same_corr'] - r['diff_corr']:.4f}"
        print(f"  {r['bits']:>5d} {r['p']:>8d} {r['q']:>8d} {r['d']:>6d} "
              f"{r['d']/r['p']:>6.1f} "
              f"{r['same_corr'] or 0:>10.4f} {r['diff_corr'] or 0:>10.4f} "
              f"{gap:>8s} {'YES' if r['factor_found'] else 'no':>8s}")

    print(f"\n  Key question: does the 'gap' persist or vanish as bits increase?")
    print(f"  If gap → 0: approach is dead (random matrices dominate)")
    print(f"  If gap → const > 0: PROMISING (detectable signal at all sizes)")
    print(f"  If gap → const and factor extraction works: BREAKTHROUGH")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print(f"{'='*75}")
    print(f"  ADDITIVE JACOBI MATRIX EXPERIMENT")
    print(f"  J[i,j] = Jacobi(a_i + a_j, N)")
    print(f"  Testing: can spectral methods separate L_p from L_q?")
    print(f"{'='*75}")

    run_scaling_test()
