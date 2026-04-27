#!/usr/bin/env python3
"""
order_failure_info.py — Harvey's insight: p-1 failure information

When Pollard p-1 FAILS to find a factor, the failure itself carries
information about p-1. This experiment quantifies how much information
each base reveals and whether accumulating constraints suffices to factor.

Harvey (2020) showed that accumulating this across many bases + lattice
reduction gives deterministic N^{1/5} factoring.
"""

import math
import random
import sys
from collections import defaultdict
from sympy import (
    isprime, factorint, gcd, nextprime, lcm,
    randprime
)


def generate_semiprime(bits):
    """Generate N = p*q where p, q are primes of roughly bits/2 each."""
    half = bits // 2
    lo = 1 << (half - 1)
    hi = 1 << half
    while True:
        p = randprime(lo, hi)
        q = randprime(lo, hi)
        if p != q:
            N = p * q
            if N.bit_length() == bits:
                return N, min(p, q), max(p, q)


def compute_order(a, p):
    """Compute multiplicative order of a mod p (p prime)."""
    if a % p == 0:
        return None
    pm1 = p - 1
    order = pm1
    for prime_factor, exp in factorint(pm1).items():
        for _ in range(exp):
            if pow(a, order // prime_factor, p) == 1:
                order //= prime_factor
            else:
                break
    return order


def phase1_info_extraction(N, p, q, bases, B_max):
    """
    Phase 1: For each base a, compute a^{lcm(1..B)} mod N for increasing B.
    Track when gcd becomes non-trivial.
    If it never does, the failure tells us about large prime factors of ord(a,p).
    """
    results = []
    for a in bases:
        if gcd(a, N) > 1:
            results.append({
                'base': a, 'trivial_factor': True,
                'factor_found_at_B': 0
            })
            continue

        # Compute a^{lcm(1,...,B)} mod N incrementally
        L = 1  # lcm(1,...,B)
        power = a % N  # will hold a^L mod N
        factor_found = False
        factor_B = None

        for B in range(2, B_max + 1):
            L_new = lcm(L, B)
            if L_new != L:
                exp = L_new // L
                power = pow(power, int(exp), N)
                L = L_new

            g = gcd(power - 1, N)
            if 1 < g < N:
                factor_found = True
                factor_B = B
                break

        # Compute ord(a, p) and ord(a, q) for analysis
        ord_p = compute_order(a, p)
        ord_q = compute_order(a, q)
        ord_p_factors = factorint(ord_p) if ord_p else {}
        ord_q_factors = factorint(ord_q) if ord_q else {}

        results.append({
            'base': a,
            'trivial_factor': False,
            'factor_found': factor_found,
            'factor_B': factor_B,
            'ord_p': ord_p,
            'ord_q': ord_q,
            'ord_p_factors': ord_p_factors,
            'ord_q_factors': ord_q_factors,
        })

    return results


def phase2_constraint_accumulation(results, p, q, B_max):
    """
    Phase 2: From each base, collect large prime factors of ord(a, p).
    These are constraints on p-1.
    """
    pm1 = p - 1
    qm1 = q - 1
    pm1_factors = factorint(pm1)
    qm1_factors = factorint(qm1)

    # Large primes: those > B_max
    pm1_large = {pf: e for pf, e in pm1_factors.items() if pf > B_max}
    qm1_large = {pf: e for pf, e in qm1_factors.items() if pf > B_max}

    # Collect discovered large factors from each base
    discovered_p_large = {}
    discovered_q_large = {}
    discovery_curve_p = []
    discovery_curve_q = []

    total_p_large_bits = sum(
        e * math.log2(pf) for pf, e in pm1_large.items()
    ) if pm1_large else 0
    total_q_large_bits = sum(
        e * math.log2(pf) for pf, e in qm1_large.items()
    ) if qm1_large else 0

    for i, r in enumerate(results):
        if r['trivial_factor'] or r.get('ord_p') is None:
            continue

        for pf, e in r.get('ord_p_factors', {}).items():
            if pf > B_max:
                if pf not in discovered_p_large:
                    discovered_p_large[pf] = 0
                discovered_p_large[pf] = max(discovered_p_large[pf], e)

        for pf, e in r.get('ord_q_factors', {}).items():
            if pf > B_max:
                if pf not in discovered_q_large:
                    discovered_q_large[pf] = 0
                discovered_q_large[pf] = max(discovered_q_large[pf], e)

        bits_p = sum(
            e * math.log2(pf) for pf, e in discovered_p_large.items()
        ) if discovered_p_large else 0
        bits_q = sum(
            e * math.log2(pf) for pf, e in discovered_q_large.items()
        ) if discovered_q_large else 0

        discovery_curve_p.append((i + 1, bits_p))
        discovery_curve_q.append((i + 1, bits_q))

    return {
        'pm1_factors': pm1_factors,
        'qm1_factors': qm1_factors,
        'pm1_large': pm1_large,
        'qm1_large': qm1_large,
        'total_p_large_bits': total_p_large_bits,
        'total_q_large_bits': total_q_large_bits,
        'discovered_p_large': discovered_p_large,
        'discovered_q_large': discovered_q_large,
        'discovery_curve_p': discovery_curve_p,
        'discovery_curve_q': discovery_curve_q,
    }


def phase4_blind_test(N, results, B_max):
    """
    Phase 4: Without knowing p, try to factor using only the constraints.

    Strategy: try primes ell > B_max one at a time. For each failed base a,
    check if gcd(a^{L*ell} - 1, N) is nontrivial.
    """
    # Compute lcm(1,...,B_max)
    L = 1
    for b in range(2, B_max + 1):
        L = lcm(L, b)

    failed_bases = [
        r['base'] for r in results
        if not r['trivial_factor'] and not r.get('factor_found', False)
    ]

    if not failed_bases:
        return {'success': True, 'method': 'direct_p-1', 'primes_tried': 0,
                'discovered_primes': []}

    search_limit = min(B_max * 20, 10000)

    discovered_primes = []
    M = int(L)

    ell = nextprime(B_max)
    primes_tried = 0

    while ell <= search_limit:
        primes_tried += 1
        for a in failed_bases[:5]:
            val = pow(a, M * int(ell), N)
            g = gcd(val - 1, N)
            if 1 < g < N:
                discovered_primes.append(int(ell))
                M = M * int(ell)
                break
        ell = nextprime(ell)

    # Final check
    final_factor = None
    for a in failed_bases[:5]:
        val = pow(a, M, N)
        g = gcd(val - 1, N)
        if 1 < g < N:
            final_factor = g
            break

    return {
        'success': final_factor is not None,
        'factor': int(final_factor) if final_factor else None,
        'method': 'blind_prime_search',
        'primes_tried': primes_tried,
        'discovered_primes': discovered_primes,
        'search_limit': search_limit,
    }


def run_experiment(bit_size, num_instances=20):
    """Run the full experiment for a given bit size."""
    print(f"\n{'='*70}")
    print(f"  SEMIPRIME SIZE: {bit_size} bits  ({num_instances} instances)")
    print(f"{'='*70}")

    # Generate first 30 primes as bases
    bases = []
    b = 2
    while len(bases) < 30:
        bases.append(b)
        b = nextprime(b)

    B_max = max(20, 1 << (bit_size // 4))
    print(f"  B_max (smooth bound): {B_max}")
    print(f"  Bases: first {len(bases)} primes up to {bases[-1]}")

    all_bases_needed = []
    all_coverage_at_5 = []
    all_coverage_at_10 = []
    all_coverage_at_20 = []
    blind_successes = 0
    direct_pm1_count = 0
    p_large_factor_counts = []

    for inst in range(num_instances):
        N, p, q = generate_semiprime(bit_size)

        # Phase 1
        results = phase1_info_extraction(N, p, q, bases, B_max)

        direct_finds = sum(
            1 for r in results
            if not r['trivial_factor'] and r.get('factor_found', False)
        )

        # Phase 2
        constraints = phase2_constraint_accumulation(results, p, q, B_max)

        pm1_large = constraints['pm1_large']
        p_large_factor_counts.append(len(pm1_large))

        if not pm1_large:
            all_bases_needed.append(0)
            direct_pm1_count += 1
        else:
            target = set(pm1_large.keys())
            found = set()
            bases_needed = None
            for i, r in enumerate(results):
                if r['trivial_factor'] or r.get('ord_p') is None:
                    continue
                for pf in r.get('ord_p_factors', {}):
                    if pf > B_max:
                        found.add(pf)
                if found >= target:
                    bases_needed = i + 1
                    break
            all_bases_needed.append(bases_needed if bases_needed else len(bases) + 1)

        # Coverage at various base counts
        curve = constraints['discovery_curve_p']
        total_bits = constraints['total_p_large_bits']

        def coverage_at(n_bases, curve=curve, total_bits=total_bits):
            if total_bits == 0:
                return 1.0
            for nb, bits in curve:
                if nb >= n_bases:
                    return bits / total_bits if total_bits > 0 else 1.0
            return (curve[-1][1] / total_bits) if curve and total_bits > 0 else 0.0

        all_coverage_at_5.append(coverage_at(5))
        all_coverage_at_10.append(coverage_at(10))
        all_coverage_at_20.append(coverage_at(20))

        # Phase 4
        blind = phase4_blind_test(N, results, B_max)
        if blind['success']:
            blind_successes += 1

        # Per-instance detail (first 3 instances)
        if inst < 3:
            print(f"\n  --- Instance {inst+1}: N={N} ({N.bit_length()}b), p={p}, q={q}")
            print(f"      p-1 = {p-1} = {dict(constraints['pm1_factors'])}")
            print(f"      q-1 = {q-1} = {dict(constraints['qm1_factors'])}")
            print(f"      Large factors of p-1 (>{B_max}): {dict(pm1_large)}")
            print(f"      Large factors of q-1 (>{B_max}): {dict(constraints['qm1_large'])}")
            print(f"      Direct p-1 finds: {direct_finds}/{len(bases)} bases")
            print(f"      Discovered large p-factors: {dict(constraints['discovered_p_large'])}")

            if curve:
                print(f"      Discovery curve (bases, bits_learned / {total_bits:.1f} total):")
                prev_bits = -1
                for nb, bits in curve[:15]:
                    if bits != prev_bits:
                        pct = 100 * bits / total_bits if total_bits > 0 else 100
                        print(f"        {nb:3d} bases -> {bits:.1f} bits ({pct:.0f}%)")
                        prev_bits = bits

            print(f"      Blind test: {'SUCCESS' if blind['success'] else 'FAILED'}"
                  f" (tried {blind.get('primes_tried', 0)} primes"
                  f", found {blind.get('discovered_primes', [])})")

    # Summary
    print(f"\n  {'─'*60}")
    print(f"  SUMMARY for {bit_size}-bit semiprimes:")
    print(f"  {'─'*60}")
    print(f"  p-1 already B_max-smooth: {direct_pm1_count}/{num_instances}"
          f" ({100*direct_pm1_count/num_instances:.0f}%)")

    if p_large_factor_counts:
        avg_large = sum(p_large_factor_counts) / len(p_large_factor_counts)
        print(f"  Avg large prime factors of p-1: {avg_large:.1f}")

    valid_needed = [x for x in all_bases_needed if x <= len(bases)]
    if valid_needed:
        print(f"  Bases to find ALL large factors of p-1: "
              f"mean={sum(valid_needed)/len(valid_needed):.1f}, "
              f"max={max(valid_needed)}")
    not_found = sum(1 for x in all_bases_needed if x > len(bases))
    if not_found:
        print(f"  Instances where 30 bases insufficient: {not_found}/{num_instances}")

    print(f"  Coverage after  5 bases: {100*sum(all_coverage_at_5)/len(all_coverage_at_5):.1f}%")
    print(f"  Coverage after 10 bases: {100*sum(all_coverage_at_10)/len(all_coverage_at_10):.1f}%")
    print(f"  Coverage after 20 bases: {100*sum(all_coverage_at_20)/len(all_coverage_at_20):.1f}%")
    print(f"  Blind factoring success: {blind_successes}/{num_instances}"
          f" ({100*blind_successes/num_instances:.0f}%)")


def main():
    print("=" * 70)
    print("  Harvey's Insight: Information from p-1 Failure")
    print("  When Pollard p-1 fails, the failure constrains p-1's structure")
    print("=" * 70)

    random.seed(42)

    for bits in [16, 20, 24, 28]:
        run_experiment(bits, num_instances=20)

    print(f"\n{'='*70}")
    print("  CONCLUSIONS")
    print(f"{'='*70}")
    print("""
  Key findings:
  1. Each base a reveals prime factors of ord(a,p) | (p-1).
     Failed p-1 attempts are NOT wasted — they constrain p-1.

  2. The "information curve" shows diminishing returns: early bases
     discover most large factors, later bases add less.

  3. Blind factoring (Phase 4) succeeds when we can enumerate the
     remaining unknown large primes of p-1 within search range.

  4. Harvey's approach: instead of searching primes one-by-one,
     use lattice reduction on the accumulated constraints to
     reconstruct p-1 in deterministic N^{1/5+epsilon} time.
""")


if __name__ == '__main__':
    main()
