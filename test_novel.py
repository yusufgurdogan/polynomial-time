#!/usr/bin/env python3
"""Test the novel approaches against baselines."""

import os
import random
import time
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kills'))
random.seed(42)

from harness import generate_semiprime, verify_factors
from baselines import pollard_rho
from k01_novel import (
    power_residue_attack,
    coppersmith_attack,
    character_matrix_attack,
    algebraic_norm_attack,
    trace_sequence_attack,
    polynomial_ring_attack,
    multi_exponent_attack,
    novel_combined,
)

algorithms = {
    "pollard_rho": pollard_rho,
    "power_residue": power_residue_attack,
    "character_matrix": character_matrix_attack,
    "poly_ring": polynomial_ring_attack,
    "multi_exponent": multi_exponent_attack,
    "trace_sequence": trace_sequence_attack,
    "algebraic_norm": algebraic_norm_attack,
    "coppersmith": coppersmith_attack,
    "novel_combined": novel_combined,
}

bit_sizes = [20, 32, 40, 48, 56, 64, 80, 96, 112, 128]
samples = 3
timeout = 5.0

# Generate fixed test cases
test_cases = {}
for bits in bit_sizes:
    test_cases[bits] = [generate_semiprime(bits) for _ in range(samples)]

alive = set(algorithms.keys())

print("=" * 75)
print("  NOVEL APPROACHES BENCHMARK")
print("=" * 75)

results = {}
for name in algorithms:
    results[name] = {}

for bits in bit_sizes:
    print(f"\n{'='*60}")
    print(f"  Bit size: {bits}  (alive: {len(alive)})")
    print(f"{'='*60}")

    for name in sorted(alive):
        algo = algorithms[name]
        correct = 0
        times = []

        for n, p, q in test_cases[bits]:
            t0 = time.time()
            try:
                factors = algo(n)
            except Exception as e:
                factors = None
            elapsed = time.time() - t0

            if elapsed > timeout:
                times.append(None)
                continue

            if factors and verify_factors(n, factors):
                correct += 1
                times.append(elapsed)
            else:
                times.append(None)

        results[name][bits] = times
        valid = [t for t in times if t is not None]
        if valid:
            avg = sum(valid) / len(valid)
            print(f"  {name:25s} | {correct}/{samples} | avg {avg:.6f}s")
        else:
            print(f"  {name:25s} | {correct}/{samples} | ELIMINATED")

    # Eliminate
    to_remove = set()
    for name in alive:
        if not any(t is not None for t in results[name].get(bits, [])):
            to_remove.add(name)
    alive -= to_remove
    if not alive:
        print("\n  All eliminated!")
        break

# Final scoreboard
print(f"\n{'='*75}")
print(f"  SCOREBOARD")
print(f"{'='*75}")
scores = []
for name in algorithms:
    max_bits = 0
    for bits in sorted(results[name].keys()):
        if any(t is not None for t in results[name][bits]):
            max_bits = bits
    scores.append((max_bits, name))

for max_bits, name in sorted(scores, reverse=True):
    # Compute average time at max bits
    if max_bits > 0 and max_bits in results[name]:
        valid = [t for t in results[name][max_bits] if t is not None]
        avg = f"{sum(valid)/len(valid):.4f}s" if valid else "N/A"
    else:
        avg = "N/A"
    baseline = " (BASELINE)" if name == "pollard_rho" else ""
    print(f"  {name:25s} | {max_bits:4d} bits | avg@max: {avg}{baseline}")
