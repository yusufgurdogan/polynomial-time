#!/usr/bin/env python3
"""
Main runner: benchmark all factoring algorithms and analyze scaling.
Progressive elimination — skip algos that fail.
"""

import random
import sys

random.seed(42)

from harness import generate_semiprime, verify_factors, time_algorithm, analyze_scaling
from baselines import trial_division, pollard_rho
from experimental import (
    lattice_partial_info,
    spectral_attack,
    combined_attack,
)
from creative import (
    multibase_power_gcd,
    coppersmith_bootstrap,
    ecm_attack,
    quadratic_form_attack,
    multi_polynomial_sieve,
    structured_walk_attack,
    creative_combined,
)


def run_progressive(algorithms, bit_sizes, samples=3, timeout=5.0):
    alive = set(algorithms.keys())
    results = {name: {} for name in algorithms}

    for bits in bit_sizes:
        print(f"\n{'='*60}")
        print(f"  Bit size: {bits}  (algos alive: {len(alive)})")
        print(f"{'='*60}")

        test_cases = [generate_semiprime(bits) for _ in range(samples)]

        for name in sorted(alive):
            algo = algorithms[name]
            times = []
            correct = 0
            for n, p, q in test_cases:
                factors, elapsed = time_algorithm(algo, n, timeout)
                if factors is not None and verify_factors(n, factors):
                    times.append(elapsed)
                    correct += 1
                else:
                    times.append(None)

            results[name][bits] = times
            valid = [t for t in times if t is not None]
            if valid:
                avg = sum(valid) / len(valid)
                print(f"  {name:30s} | {correct}/{samples} | avg {avg:.6f}s")
            else:
                print(f"  {name:30s} | {correct}/{samples} | ELIMINATED")

        to_remove = set()
        for name in alive:
            if bits in results[name]:
                if not any(t is not None for t in results[name][bits]):
                    to_remove.add(name)
        alive -= to_remove

        if not alive:
            print("\n  All algorithms eliminated!")
            break

    return results


def main():
    print("=" * 70)
    print("  POLYNOMIAL TIME FACTORING — ROUND 2")
    print("  Baselines + Creative Approaches")
    print("=" * 70)

    algorithms = {
        # Baselines
        "pollard_rho": pollard_rho,
        # Round 1 survivors
        "spectral": spectral_attack,
        "combined_r1": combined_attack,
        # Round 2: Creative
        "multibase_gcd": multibase_power_gcd,
        "coppersmith_boot": coppersmith_bootstrap,
        "ecm": ecm_attack,
        "quad_form": quadratic_form_attack,
        "multi_poly_sieve": multi_polynomial_sieve,
        "structured_walk": structured_walk_attack,
        "creative_combined": creative_combined,
    }

    bit_sizes = [20, 32, 40, 48, 56, 64, 80, 96, 112, 128]
    timeout = 5.0

    if "--big" in sys.argv:
        bit_sizes += [160, 192, 256]
        timeout = 15.0

    print(f"\nBit sizes: {bit_sizes}")
    print(f"Timeout: {timeout}s per attempt")

    results = run_progressive(algorithms, bit_sizes, samples=3, timeout=timeout)

    print(f"\n{'='*70}")
    print(f"  SCALING ANALYSIS")
    print(f"{'='*70}")
    for name in algorithms:
        analyze_scaling(results, name)

    print(f"\n{'='*70}")
    print(f"  FINAL SCOREBOARD")
    print(f"{'='*70}")
    scores = []
    for name in algorithms:
        max_bits = 0
        for bits in sorted(results[name].keys()):
            if any(t is not None for t in results[name][bits]):
                max_bits = bits
        scores.append((max_bits, name))
    for max_bits, name in sorted(scores, reverse=True):
        print(f"  {name:30s} | {max_bits:4d} bits")


if __name__ == "__main__":
    main()
