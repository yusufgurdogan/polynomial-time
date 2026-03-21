#!/usr/bin/env python3
"""Push the ultimate algorithm to find its breaking point."""

import random, time, sys
sys.stdout.reconfigure(line_buffering=True)
random.seed(42)

from harness import generate_semiprime, verify_factors
from baselines import pollard_rho
from creative import ecm_attack
from novel2 import iterated_frobenius
from ultimate import ultimate_factor

algorithms = {
    "pollard_rho": pollard_rho,
    "ecm": ecm_attack,
    "iter_frobenius": iterated_frobenius,
    "ultimate": ultimate_factor,
}

bit_sizes = [32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256]
samples = 3
timeout = 30.0

test_cases = {}
for bits in bit_sizes:
    test_cases[bits] = [generate_semiprime(bits) for _ in range(samples)]

alive = set(algorithms.keys())
results = {name: {} for name in algorithms}

print("=" * 75)
print("  ULTIMATE FACTORING CHALLENGE")
print(f"  Bit sizes up to 256, timeout {timeout}s, {samples} samples each")
print("=" * 75)

for bits in bit_sizes:
    print(f"\n{'='*65}")
    print(f"  Bit size: {bits}  (alive: {len(alive)})")
    print(f"{'='*65}")

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
            print(f"  {name:25s} | {correct}/{samples} | avg {avg:.4f}s")
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
print(f"  FINAL SCOREBOARD")
print(f"{'='*75}")
scores = []
for name in algorithms:
    max_bits = 0
    for bits in sorted(results[name].keys()):
        if any(t is not None for t in results[name][bits]):
            max_bits = bits
    scores.append((max_bits, name))

for max_bits, name in sorted(scores, reverse=True):
    if max_bits > 0 and max_bits in results[name]:
        valid = [t for t in results[name][max_bits] if t is not None]
        avg = f"{sum(valid)/len(valid):.4f}s" if valid else "N/A"
    else:
        avg = "N/A"
    print(f"  {name:25s} | {max_bits:4d} bits | avg@max: {avg}")
