#!/usr/bin/env python3
"""
FREQUENCY EXTRACTION FROM MODULAR EXPONENTIAL SIGNALS

The quantum microscope revealed: factoring = separating two periodicities.
Quantum samples are w_i = dlp_i·α/(p-1) + dlq_i·β/(q-1) mod 1.

Classical observation: b^k mod N for k = 0,1,2,... traces a 1D path
through the 2D periodic structure. The path has quasi-period lcm(p-1,q-1)
but the LOCAL behavior encodes two frequencies related to (p-1) and (q-1).

This experiment applies classical frequency estimation methods to
modular exponential sequences, asking: can we extract (p-1) or (q-1)
from the signal b^k mod N?

Methods:
  1. Prony's method: recover frequencies from sum of exponentials
  2. ESPRIT: subspace-based frequency estimation
  3. FFT peak detection: classical spectral analysis
  4. Autocorrelation period detection
"""

import math
import random
import sys
import time
import numpy as np
from numpy.linalg import svd, eig, lstsq
from sympy import nextprime

sys.stdout.reconfigure(line_buffering=True)


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, hi))
    while q == p:
        q = nextprime(random.randint(lo, hi))
    if p > q:
        p, q = q, p
    return p * q, p, q


# =========================================================================
# Signal generation: b^k mod N normalized to [0, 1)
# =========================================================================

def modexp_signal(b, N, T):
    """Generate the signal x[k] = b^k mod N / N for k = 0, ..., T-1."""
    signal = np.zeros(T)
    val = 1
    for k in range(T):
        signal[k] = val / N
        val = (val * b) % N
    return signal


def multi_base_signal(bases, N, T):
    """Generate d-dimensional signal: x_i[k] = b_i^k mod N / N."""
    d = len(bases)
    signals = np.zeros((T, d))
    for i, b in enumerate(bases):
        signals[:, i] = modexp_signal(b, N, T)
    return signals


# =========================================================================
# Method 1: Prony's method
# =========================================================================

def prony_method(signal, num_freqs):
    """
    Prony's method: recover frequencies from a signal assumed to be
    x[k] = Σ_{j=1}^{M} a_j · z_j^k
    where z_j = exp(2πi·f_j) are complex exponentials.

    Given 2M samples, recovers the M frequencies f_j.

    For modular exponential signals, this is approximate since
    b^k mod N ≠ clean sum of exponentials.
    """
    T = len(signal)
    M = num_freqs

    if T < 2 * M + 1:
        return None

    # Build the Hankel matrix
    # H[i,j] = signal[i+j] for i=0..T-M-1, j=0..M
    rows = T - M
    H = np.zeros((rows, M + 1))
    for i in range(rows):
        for j in range(M + 1):
            if i + j < T:
                H[i, j] = signal[i + j]

    # Solve the linear prediction: find coefficients c such that
    # signal[k+M] = c[0]*signal[k] + ... + c[M-1]*signal[k+M-1]
    A = H[:, :M]
    b_vec = H[:, M]

    try:
        c, residuals, rank, sv = lstsq(A, b_vec, rcond=None)
    except Exception:
        return None

    # The roots of z^M - c[M-1]*z^{M-1} - ... - c[0] = 0
    # give the complex exponentials z_j
    poly_coeffs = np.zeros(M + 1)
    poly_coeffs[M] = 1.0
    for i in range(M):
        poly_coeffs[i] = -c[i]

    try:
        roots = np.roots(poly_coeffs[::-1])  # numpy wants highest degree first
    except Exception:
        return None

    # Extract frequencies from roots
    # z_j = exp(2πi·f_j), so f_j = angle(z_j) / (2π)
    frequencies = []
    for z in roots:
        if abs(z) > 1e-10:
            f = np.angle(z) / (2 * np.pi)
            frequencies.append(f % 1.0)

    return sorted(frequencies)


# =========================================================================
# Method 2: ESPRIT (Estimation of Signal Parameters via Rotational Invariance)
# =========================================================================

def esprit_method(signal, num_freqs):
    """
    ESPRIT: subspace-based frequency estimation.
    More robust than Prony for noisy signals.
    """
    T = len(signal)
    M = num_freqs
    L = min(T // 2, max(2 * M + 1, 20))

    if T < L + M:
        return None

    # Build data matrix (Hankel)
    rows = T - L + 1
    X = np.zeros((L, rows))
    for i in range(L):
        X[i, :] = signal[i:i + rows]

    # SVD
    try:
        U, S, Vh = svd(X, full_matrices=False)
    except Exception:
        return None

    # Take the signal subspace (first M singular vectors)
    if len(S) < M:
        return None
    Us = U[:, :M]

    # Shift invariance: Us_1 and Us_2 are related by a rotation
    Us1 = Us[:-1, :]
    Us2 = Us[1:, :]

    # Solve Us1 @ Phi = Us2 for Phi
    try:
        Phi, _, _, _ = lstsq(Us1, Us2, rcond=None)
    except Exception:
        return None

    # Eigenvalues of Phi give the frequencies
    try:
        eigenvalues = np.linalg.eigvals(Phi)
    except Exception:
        return None

    frequencies = []
    for ev in eigenvalues:
        if abs(ev) > 1e-10:
            f = np.angle(ev) / (2 * np.pi)
            frequencies.append(f % 1.0)

    return sorted(frequencies)


# =========================================================================
# Method 3: FFT peak detection
# =========================================================================

def fft_peaks(signal, num_peaks=10):
    """
    Standard FFT to find dominant frequencies.
    """
    T = len(signal)
    # Remove DC
    signal_centered = signal - np.mean(signal)

    # Zero-pad for resolution
    N_fft = max(T, 4096)
    spectrum = np.abs(np.fft.fft(signal_centered, n=N_fft))

    # Only positive frequencies
    half = N_fft // 2
    spectrum = spectrum[:half]
    freqs = np.arange(half) / N_fft

    # Find peaks
    peaks = []
    for i in range(1, half - 1):
        if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
            peaks.append((spectrum[i], freqs[i]))

    peaks.sort(reverse=True)
    return [(f, amp) for amp, f in peaks[:num_peaks]]


# =========================================================================
# Method 4: Autocorrelation period detection
# =========================================================================

def autocorrelation_periods(signal, max_lag=None):
    """Detect periods via autocorrelation."""
    T = len(signal)
    if max_lag is None:
        max_lag = T // 2

    signal_centered = signal - np.mean(signal)
    var = np.var(signal_centered)
    if var < 1e-15:
        return []

    autocorr = np.correlate(signal_centered, signal_centered, mode='full')
    autocorr = autocorr[T-1:]  # positive lags only
    autocorr = autocorr / autocorr[0]  # normalize

    # Find peaks
    peaks = []
    for i in range(2, min(max_lag, len(autocorr) - 1)):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            if autocorr[i] > 0.05:  # Significance threshold
                peaks.append((i, autocorr[i]))

    peaks.sort(key=lambda x: -x[1])
    return peaks[:10]


# =========================================================================
# Check if detected frequencies relate to p-1 or q-1
# =========================================================================

def check_frequency_match(freqs, p, q, N, tolerance=0.01):
    """
    Check if any detected frequency f relates to (p-1) or (q-1).

    The modular exponential b^k mod N has "frequencies" related to
    the multiplicative orders of b mod p and mod q.

    If b has order r_p mod p and order r_q mod q, then
    b^k mod p cycles with period r_p, and b^k mod q cycles with period r_q.

    The resulting signal has frequencies at multiples of 1/r_p and 1/r_q.
    Since r_p | (p-1) and r_q | (q-1), these relate to the factorization.
    """
    matches = []
    target_periods = [p-1, q-1]
    # Also check divisors of p-1 and q-1
    for target in target_periods:
        divs = []
        for d in range(1, min(target + 1, 10000)):
            if target % d == 0:
                divs.append(d)
        for d in divs:
            target_freq = 1.0 / d
            for f in freqs:
                if isinstance(f, tuple):
                    f = f[0]
                if f > 0 and abs(f - target_freq) < tolerance:
                    matches.append((f, d, 'p-1' if target == p-1 else 'q-1'))
                # Also check multiples
                for mult in range(1, 10):
                    if abs(f * mult - target_freq) < tolerance:
                        matches.append((f, d, f'p-1/{mult}' if target == p-1 else f'q-1/{mult}'))

    return matches


# =========================================================================
# Main experiment
# =========================================================================

def run_experiment(bits_list=None, num_instances=20):
    if bits_list is None:
        bits_list = [16, 20, 24, 28, 32]

    print(f"{'='*75}")
    print(f"  FREQUENCY EXTRACTION FROM MODULAR EXPONENTIAL SIGNALS")
    print(f"  Can classical signal processing extract (p-1) or (q-1)?")
    print(f"{'='*75}")

    for bits in bits_list:
        print(f"\n{'='*75}")
        print(f"  {bits}-BIT SEMIPRIMES")
        print(f"{'='*75}")

        # Signal lengths to try: from poly(log N) to √N
        n = bits
        sqrt_N = 2 ** (bits // 2)
        T_values = sorted(set([
            n * 2, n * 5, n * 10, n * 20, n * 50,
            min(n * 100, sqrt_N),
            min(n * 500, sqrt_N * 2),
            min(sqrt_N, 100000),
        ]))
        T_values = [t for t in T_values if t <= 100000]

        prony_hits = {T: 0 for T in T_values}
        esprit_hits = {T: 0 for T in T_values}
        fft_hits = {T: 0 for T in T_values}
        autocorr_hits = {T: 0 for T in T_values}
        total = 0

        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)
            total += 1

            base = 2  # Use base 2

            for T in T_values:
                signal = modexp_signal(base, N, T)

                # Method 1: Prony
                num_f = min(10, T // 4)
                prony_freqs = prony_method(signal, num_f)
                if prony_freqs:
                    matches = check_frequency_match(prony_freqs, p, q, N)
                    if matches:
                        prony_hits[T] += 1

                # Method 2: ESPRIT
                esprit_freqs = esprit_method(signal, num_f)
                if esprit_freqs:
                    matches = check_frequency_match(esprit_freqs, p, q, N)
                    if matches:
                        esprit_hits[T] += 1

                # Method 3: FFT
                fft_freqs = fft_peaks(signal, num_peaks=20)
                if fft_freqs:
                    matches = check_frequency_match(fft_freqs, p, q, N)
                    if matches:
                        fft_hits[T] += 1

                # Method 4: Autocorrelation
                ac_peaks = autocorrelation_periods(signal, max_lag=min(T // 2, 5000))
                if ac_peaks:
                    # Check if any peak period divides (p-1) or (q-1)
                    for lag, strength in ac_peaks:
                        if (p - 1) % lag == 0 or (q - 1) % lag == 0:
                            autocorr_hits[T] += 1
                            break

            if (inst + 1) % 5 == 0:
                print(f"  [{inst+1}/{num_instances}]")

        # Report
        print(f"\n  Results ({total} instances):")
        print(f"  {'T':>8} {'T/√N':>8} {'T/n²':>6} | {'Prony':>8} {'ESPRIT':>8} {'FFT':>8} {'AutoCorr':>8}")
        print(f"  {'-'*8} {'-'*8} {'-'*6}-+-{'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for T in T_values:
            T_sqrtN = T / sqrt_N
            T_n2 = T / (n * n)
            pr = prony_hits[T] / total
            es = esprit_hits[T] / total
            ft = fft_hits[T] / total
            ac = autocorr_hits[T] / total
            print(f"  {T:8d} {T_sqrtN:8.4f} {T_n2:6.1f} | "
                  f"{pr:7.0%} {es:7.0%} {ft:7.0%} {ac:7.0%}")

    # Detailed example on one instance
    print(f"\n{'='*75}")
    print(f"  DETAILED EXAMPLE")
    print(f"{'='*75}")

    p, q = 137, 157
    N = p * q
    T = 2000
    base = 2

    print(f"  N = {N} = {p} × {q}")
    print(f"  Base = {base}, T = {T} samples")
    print(f"  True periods: p-1 = {p-1}, q-1 = {q-1}")
    print(f"  ord_p(2) = {multiplicative_order(base, p)}")
    print(f"  ord_q(2) = {multiplicative_order(base, q)}")

    signal = modexp_signal(base, N, T)

    # FFT analysis
    fft_f = fft_peaks(signal, num_peaks=15)
    print(f"\n  Top FFT frequencies:")
    for f, amp in fft_f:
        period = 1/f if f > 0 else float('inf')
        # Check if period is related to p-1 or q-1
        rel_p = ""
        rel_q = ""
        if period > 0:
            for d in range(1, 200):
                if abs(period * d - (p-1)) < 1:
                    rel_p = f" ← {d}×period ≈ p-1={p-1}"
                if abs(period * d - (q-1)) < 1:
                    rel_q = f" ← {d}×period ≈ q-1={q-1}"
        print(f"    f={f:.6f}  period={period:.1f}  amp={amp:.1f}{rel_p}{rel_q}")

    # Autocorrelation
    ac = autocorrelation_periods(signal, max_lag=1000)
    print(f"\n  Top autocorrelation peaks:")
    for lag, strength in ac[:10]:
        rel = ""
        if (p-1) % lag == 0:
            rel += f" divides p-1={p-1}"
        if (q-1) % lag == 0:
            rel += f" divides q-1={q-1}"
        print(f"    lag={lag:5d}  strength={strength:.4f}{rel}")

    # Prony
    prony_f = prony_method(signal, 8)
    if prony_f:
        print(f"\n  Prony frequencies:")
        for f in prony_f:
            period = 1/f if f > 0.001 else float('inf')
            print(f"    f={f:.6f}  period={period:.1f}")

    # Multi-base analysis
    print(f"\n  Multi-base signal (bases 2, 3, 5, 7):")
    bases = [2, 3, 5, 7]
    multi_sig = multi_base_signal(bases, N, T)
    # Cross-correlation between bases
    for i in range(len(bases)):
        for j in range(i+1, len(bases)):
            cc = np.corrcoef(multi_sig[:, i], multi_sig[:, j])[0, 1]
            print(f"    corr({bases[i]}, {bases[j]}) = {cc:.4f}")


def multiplicative_order(a, p):
    """Compute the multiplicative order of a mod p."""
    if math.gcd(a, p) > 1:
        return 0
    order = 1
    val = a % p
    while val != 1:
        val = (val * a) % p
        order += 1
        if order > p:
            return 0
    return order


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_experiment(bits_list=[16, 20, 24, 28], num_instances=15)
