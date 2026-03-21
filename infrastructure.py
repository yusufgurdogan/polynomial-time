#!/usr/bin/env python3
"""
Infrastructure exploration of Q(√N) for N = pq.

Enumerate the full principal cycle of reduced binary quadratic forms,
dump everything, and look for patterns that could lead to polynomial-time
factoring.

Key questions:
1. Where do ambiguous forms sit? (Theory says distance R/2)
2. Do form coefficients cluster near p or q?
3. Is there any poly-time computable signal correlated with
   the ambiguous form's position?
"""

import math
import random
import time
import sys
from typing import Optional, Tuple, List
from dataclasses import dataclass

sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# Binary Quadratic Forms: (a, b, c) represents ax² + bxy + cy²
# Discriminant Δ = b² - 4ac
# =============================================================================

@dataclass
class QF:
    """Binary quadratic form (a, b, c) with discriminant b²-4ac."""
    a: int
    b: int
    c: int

    @property
    def disc(self):
        return self.b * self.b - 4 * self.a * self.c

    def is_reduced(self):
        """A form is reduced if |b| ≤ a ≤ c, and if a = c then b ≥ 0,
        and if |b| = a then b ≥ 0. (Positive definite / Gauss reduced.)
        For indefinite forms (our case): |√Δ - 2|a|| < b < √Δ."""
        D = self.disc
        if D < 0:
            # Positive definite
            return (abs(self.b) <= self.a <= self.c and
                    (self.a != self.c or self.b >= 0) and
                    (abs(self.b) != self.a or self.b >= 0))
        else:
            # Indefinite
            sqrt_D = math.isqrt(D)
            return 0 < self.b < sqrt_D and sqrt_D - self.b < 2 * abs(self.a) < sqrt_D + self.b

    def is_ambiguous(self):
        """A form is ambiguous if a | b."""
        return self.a != 0 and self.b % self.a == 0

    def __repr__(self):
        return f"({self.a}, {self.b}, {self.c})"


def principal_form(N: int) -> QF:
    """The principal form of discriminant Δ = 4N (for odd N) or Δ = N."""
    # For discriminant Δ = 4N: principal form is (1, 0, -N)
    # For Δ = N (when N ≡ 1 mod 4): principal form is (1, 1, (1-N)/4)
    if N % 4 == 1:
        return QF(1, 1, (1 - N) // 4)
    else:
        return QF(1, 0, -N)


def reduce_form(f: QF) -> QF:
    """Reduce an indefinite form using the reduction algorithm (continued fraction step)."""
    D = f.disc
    sqrt_D = math.isqrt(D)
    # Ensure sqrt_D² ≤ D (isqrt gives floor)
    while (sqrt_D + 1) * (sqrt_D + 1) <= D:
        sqrt_D += 1

    a, b, c = f.a, f.b, f.c
    # Normalize: reduction for indefinite forms
    # Apply rho operator repeatedly until reduced
    max_iter = 10000
    for _ in range(max_iter):
        # Check if reduced
        if 0 < b < sqrt_D:
            two_abs_a = 2 * abs(a)
            if sqrt_D - b < two_abs_a < sqrt_D + b:
                return QF(a, b, c)

        # Apply rho: (a, b, c) → (c, -b + 2c*round((sqrt_D + b)/(2c)), ...)
        if c == 0:
            break
        # Next b: the unique b' with b' ≡ -b (mod 2|c|) and |b'| < sqrt_D
        # b' = -b + 2|c| * k where k = round((sqrt_D + b) / (2|c|))
        abs_c = abs(c)
        k = (sqrt_D + b + abs_c) // (2 * abs_c)  # round towards nearest
        b_new = -b + 2 * abs_c * k
        # Adjust sign of c
        if c < 0:
            b_new = -b + 2 * abs_c * k

        # New form: (c, b_new, (b_new² - D) / (4c))
        c_new = (b_new * b_new - D) // (4 * c)
        a, b, c = c, b_new, c_new

    return QF(a, b, c)


def rho_step(f: QF) -> QF:
    """One step of the reduction operator ρ: advance to next form in cycle.
    For indefinite form of discriminant Δ:
    ρ(a, b, c) = (c, b', c') where b' = -b mod 2|c| closest to √Δ,
    and c' = (b'² - Δ)/(4c).
    """
    D = f.disc
    sqrt_D = math.isqrt(D)
    # Correct sqrt_D
    if (sqrt_D + 1) * (sqrt_D + 1) <= D:
        sqrt_D += 1

    a, b, c = f.a, f.b, f.c
    if c == 0:
        return f

    abs_c = abs(c)
    # b' ≡ -b (mod 2|c|), 0 < b' < √Δ, and √Δ - b' < 2|c|
    # b' = -b + 2|c| * ceil((√Δ + b) / (2|c|))
    # Actually: b' is the unique value with b' ≡ -b (mod 2|c|) in (√Δ - 2|c|, √Δ)

    # Compute b' = -b mod 2*|c|, adjusted to be in the right range
    r = (-b) % (2 * abs_c)
    # We want b' in (sqrt_D - 2*abs_c, sqrt_D), with b' ≡ r (mod 2*abs_c)
    # Start from r and add multiples of 2*abs_c
    b_new = r
    while b_new <= sqrt_D - 2 * abs_c:
        b_new += 2 * abs_c
    if b_new >= sqrt_D:
        b_new -= 2 * abs_c

    # Ensure b_new is positive and in range
    if b_new <= 0:
        b_new += 2 * abs_c

    c_new = (b_new * b_new - D) // (4 * c)

    return QF(c, b_new, c_new)


def enumerate_cycle(N: int, max_forms: int = 100000) -> List[dict]:
    """
    Enumerate the full principal cycle of reduced forms for discriminant 4N.
    Returns list of {form, distance, is_ambiguous, is_square} for each form.
    """
    D = 4 * N  # discriminant
    sqrt_D = math.isqrt(D)

    # Start with principal form (1, 0, -N) and reduce it
    f0 = QF(1, 0, -N)
    f = reduce_form(f0)

    cycle = []
    distance = 0.0  # infrastructure distance (cumulative log)
    seen = set()

    for step in range(max_forms):
        key = (f.a, f.b, f.c)
        if key in seen and step > 0:
            break  # completed the cycle
        seen.add(key)

        # Record
        entry = {
            'form': QF(f.a, f.b, f.c),
            'a': f.a,
            'b': f.b,
            'c': f.c,
            'step': step,
            'distance': distance,
            'is_ambiguous': f.is_ambiguous(),
            'a_divides_N': (N % abs(f.a) == 0) if f.a != 0 else False,
            'c_divides_N': (N % abs(f.c) == 0) if f.c != 0 else False,
            'gcd_a_N': math.gcd(abs(f.a), N),
            'gcd_c_N': math.gcd(abs(f.c), N),
        }
        cycle.append(entry)

        # Advance: apply rho
        f_next = rho_step(f)

        # Compute distance increment
        # The infrastructure distance between consecutive reduced forms is
        # log(|a_{i+1}| / |a_i|) approximately, or more precisely:
        # δ = log((√Δ + b) / (2|a|)) where (a, b, c) is the current form
        if f.a != 0:
            val = (sqrt_D + f.b) / (2.0 * abs(f.a))
            if val > 0:
                distance += math.log(val)

        f = f_next

    return cycle


def analyze_cycle(N: int, p: int, q: int, verbose: bool = True):
    """Full analysis of the principal cycle for N = pq."""
    cycle = enumerate_cycle(N)
    R = cycle[-1]['distance'] if cycle else 0  # approximate regulator

    # Find ambiguous forms
    ambiguous = [e for e in cycle if e['is_ambiguous']]

    # Find forms whose a-value divides N (potential factor exposure)
    a_divides = [e for e in cycle if e['a_divides_N'] and abs(e['a']) > 1]

    # Find forms whose gcd(a, N) is non-trivial
    gcd_nontrivial = [e for e in cycle if 1 < e['gcd_a_N'] < N]

    if verbose:
        print(f"\n  N = {N} = {p} × {q} ({N.bit_length()} bits)")
        print(f"  Cycle length: {len(cycle)} forms")
        print(f"  Regulator R ≈ {R:.4f}")
        print(f"  √N = {math.isqrt(N)}, R/√N ≈ {R/math.sqrt(N):.4f}")

        if ambiguous:
            print(f"\n  Ambiguous forms ({len(ambiguous)}):")
            for e in ambiguous[:10]:
                print(f"    step={e['step']:5d} dist={e['distance']:10.4f} "
                      f"(dist/R={e['distance']/R:.4f} if R>0) "
                      f"form={e['form']} gcd(a,N)={e['gcd_a_N']}")

        if gcd_nontrivial:
            print(f"\n  Forms with non-trivial gcd(a, N) ({len(gcd_nontrivial)}):")
            for e in gcd_nontrivial[:10]:
                print(f"    step={e['step']:5d} dist={e['distance']:10.4f} "
                      f"form={e['form']} gcd(a,N)={e['gcd_a_N']}")

        # Distribution of a-values
        a_vals = [abs(e['a']) for e in cycle]
        print(f"\n  a-value stats: min={min(a_vals)}, max={max(a_vals)}, "
              f"mean={sum(a_vals)/len(a_vals):.1f}")
        print(f"  √(N/3) = {math.isqrt(N//3)} (theoretical bound on |a|)")

        # How close do a-values get to p and q?
        min_dist_p = min(abs(a - p) for a in a_vals)
        min_dist_q = min(abs(a - q) for a in a_vals)
        print(f"  Closest a to p={p}: distance {min_dist_p}")
        print(f"  Closest a to q={q}: distance {min_dist_q}")

        # Check if p or q appear directly as a-values
        p_in_a = sum(1 for a in a_vals if a == p)
        q_in_a = sum(1 for a in a_vals if a == q)
        print(f"  p appears as a-value: {p_in_a} times")
        print(f"  q appears as a-value: {q_in_a} times")

    return {
        'N': N, 'p': p, 'q': q,
        'cycle_length': len(cycle),
        'regulator': R,
        'num_ambiguous': len(ambiguous),
        'num_gcd_nontrivial': len(gcd_nontrivial),
        'ambiguous_positions': [(e['step'], e['distance']) for e in ambiguous],
        'gcd_positions': [(e['step'], e['gcd_a_N']) for e in gcd_nontrivial],
        'cycle': cycle,
    }


def compare_pq_vs_prime(bits: int = 20):
    """Compare infrastructure of N=pq vs N=prime to find distinguishing signals."""
    from harness import generate_semiprime
    from sympy import nextprime

    print("=" * 70)
    print(f"  COMPARING N=pq vs N=prime ({bits} bits)")
    print("=" * 70)

    # Generate semiprime
    n_pq, p, q = generate_semiprime(bits)

    # Generate a prime of similar size
    n_prime = nextprime(n_pq)
    while n_prime % 4 == 1:  # need N ≡ 3 mod 4 for simplicity, or just use 4N
        n_prime = nextprime(n_prime)

    print(f"\n--- N = {n_pq} = {p} × {q} (composite) ---")
    result_pq = analyze_cycle(n_pq, p, q, verbose=True)

    # For prime, we can't factor it, but we can still look at the cycle
    print(f"\n--- N = {n_prime} (prime) ---")
    cycle_prime = enumerate_cycle(n_prime)
    R_prime = cycle_prime[-1]['distance'] if cycle_prime else 0
    ambig_prime = [e for e in cycle_prime if e['is_ambiguous']]
    print(f"  Cycle length: {len(cycle_prime)} forms")
    print(f"  Regulator R ≈ {R_prime:.4f}")
    print(f"  Ambiguous forms: {len(ambig_prime)}")
    a_vals_prime = [abs(e['a']) for e in cycle_prime]
    if a_vals_prime:
        print(f"  a-value stats: min={min(a_vals_prime)}, max={max(a_vals_prime)}, "
              f"mean={sum(a_vals_prime)/len(a_vals_prime):.1f}")

    # Compare
    print(f"\n--- COMPARISON ---")
    print(f"  {'Metric':<30} | {'N=pq':>15} | {'N=prime':>15}")
    print(f"  {'-'*30}-+-{'-'*15}-+-{'-'*15}")
    print(f"  {'Cycle length':<30} | {result_pq['cycle_length']:>15} | {len(cycle_prime):>15}")
    print(f"  {'Regulator R':<30} | {result_pq['regulator']:>15.2f} | {R_prime:>15.2f}")
    print(f"  {'R / √N':<30} | {result_pq['regulator']/math.sqrt(n_pq):>15.4f} | {R_prime/math.sqrt(n_prime):>15.4f}")
    print(f"  {'Ambiguous forms':<30} | {result_pq['num_ambiguous']:>15} | {len(ambig_prime):>15}")
    print(f"  {'Non-trivial gcd(a,N)':<30} | {result_pq['num_gcd_nontrivial']:>15} | {'N/A':>15}")


def scaling_experiment():
    """How does the infrastructure behave as N grows?"""
    from harness import generate_semiprime

    print("=" * 70)
    print("  INFRASTRUCTURE SCALING")
    print("=" * 70)
    print(f"  {'bits':>5} | {'cycle':>7} | {'R':>10} | {'ambig':>6} | "
          f"{'gcd_hit':>7} | {'ambig_dist/R':>12} | {'time':>7}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*10}-+-{'-'*6}-+-"
          f"{'-'*7}-+-{'-'*12}-+-{'-'*7}")

    for bits in [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]:
        n, p, q = generate_semiprime(bits)
        t0 = time.time()
        result = analyze_cycle(n, p, q, verbose=False)
        elapsed = time.time() - t0

        # Position of first ambiguous form as fraction of R
        if result['ambiguous_positions'] and result['regulator'] > 0:
            first_ambig_dist = result['ambiguous_positions'][0][1]
            ratio = first_ambig_dist / result['regulator']
        else:
            ratio = float('nan')

        print(f"  {bits:5d} | {result['cycle_length']:7d} | {result['regulator']:10.2f} | "
              f"{result['num_ambiguous']:6d} | {result['num_gcd_nontrivial']:7d} | "
              f"{ratio:12.4f} | {elapsed:6.2f}s")

        if elapsed > 30:
            print("  stopping — too slow")
            break


if __name__ == "__main__":
    random.seed(42)

    # Test 1: Detailed cycle analysis for small N = pq
    print("=" * 70)
    print("  INFRASTRUCTURE CYCLE ANALYSIS")
    print("=" * 70)

    for (p, q) in [(3, 5), (7, 11), (13, 17), (23, 29), (101, 103)]:
        analyze_cycle(p * q, p, q)

    # Test 2: Compare N=pq vs N=prime
    compare_pq_vs_prime(20)

    # Test 3: Scaling
    print()
    scaling_experiment()
