#!/usr/bin/env python3
"""Test novel2 approaches — focused on Frobenius ring methods."""

import os, random, time, sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kills'))
random.seed(42)

from harness import generate_semiprime, verify_factors
from baselines import pollard_rho
from k01_novel2 import (
    frobenius_walk,
    frobenius_systematic,
    iterated_frobenius,
    cubic_frobenius,
    multi_ring_attack,
    novel2_combined,
)

algorithms = {
    "pollard_rho": pollard_rho,
    "frob_walk": frobenius_walk,
    "frob_systematic": frobenius_systematic,
    "iter_frobenius": iterated_frobenius,
    "cubic_frob": cubic_frobenius,
    "multi_ring": multi_ring_attack,
    "novel2_combined": novel2_combined,
}

bit_sizes = [20, 32, 40, 48, 56, 64, 80, 96, 112, 128]
samples = 3
timeout = 10.0

test_cases = {}
for bits in bit_sizes:
    test_cases[bits] = [generate_semiprime(bits) for _ in range(samples)]

alive = set(algorithms.keys())
results = {name: {} for name in algorithms}

print("=" * 75)
print("  NOVEL v2: FROBENIUS RING METHODS")
print("=" * 75)

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
            except Exception:
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

    to_remove = set()
    for name in alive:
        if not any(t is not None for t in results[name].get(bits, [])):
            to_remove.add(name)
    alive -= to_remove
    if not alive:
        print("\n  All eliminated!")
        break

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
    tag = " (BASELINE)" if name == "pollard_rho" else " ** NOVEL **"
    if max_bits > 0 and max_bits in results[name]:
        valid = [t for t in results[name][max_bits] if t is not None]
        avg = f"{sum(valid)/len(valid):.4f}s" if valid else "N/A"
    else:
        avg = "N/A"
    print(f"  {name:25s} | {max_bits:4d} bits | avg@max: {avg}{tag}")
