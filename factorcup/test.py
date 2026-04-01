#!/usr/bin/env python3
"""
FactorCup Autotester — Correctness verification.

Run: python test.py

Fix all errors before submitting to scoring.
Tests are incremental: later tests assume earlier ones pass.

Do not modify this file.
"""

import sys
import time
import importlib
from interface import validate_factors
from generate import generate_semiprime

# Try to import the entry
try:
    import entry
    factor = entry.factor
except ImportError:
    print("ERROR: No entry.py found.")
    print("  Copy the example:  cp example_entry.py entry.py")
    print("  Then edit entry.py with your algorithm.")
    sys.exit(1)
except AttributeError:
    print("ERROR: entry.py does not define a 'factor' function.")
    sys.exit(1)


def test(name, cases, timeout=10.0):
    """Run a test suite. Returns (passed, failed, errors)."""
    print(f"\n  Test: {name}")
    print(f"  {'-' * 60}")
    passed = 0
    failed = 0
    errors = 0

    for i, (N, p_expected, q_expected) in enumerate(cases):
        try:
            t0 = time.time()
            result = factor(N)
            elapsed = time.time() - t0

            if elapsed > timeout:
                print(f"    {i+1}. N={N} ({N.bit_length()}b): TIMEOUT ({elapsed:.1f}s > {timeout}s)")
                failed += 1
                continue

            ok, reason = validate_factors(N, result)
            if ok:
                print(f"    {i+1}. N={N} ({N.bit_length()}b): OK ({elapsed:.4f}s)")
                passed += 1
            else:
                print(f"    {i+1}. N={N} ({N.bit_length()}b): WRONG — {reason}")
                failed += 1

        except Exception as e:
            print(f"    {i+1}. N={N} ({N.bit_length()}b): ERROR — {e}")
            errors += 1

    total = passed + failed + errors
    status = "PASS" if failed == 0 and errors == 0 else "FAIL"
    print(f"  Result: {status} ({passed}/{total} passed)")
    return passed, failed, errors


def main():
    print("=" * 65)
    print("  FactorCup Autotester")
    print("=" * 65)

    total_pass = 0
    total_fail = 0
    total_err = 0

    # ---- Test 1: Tiny semiprimes ----
    tiny_cases = [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
        (1003, 17, 59),
    ]
    p, f, e = test("Tiny semiprimes (< 12 bits)", tiny_cases)
    total_pass += p
    total_fail += f
    total_err += e
    if f + e > 0:
        print("\n  STOP: Fix basic factoring before proceeding.")
        sys.exit(1)

    # ---- Test 2: Small semiprimes (12-20 bits) ----
    small_cases = []
    for bits in [12, 14, 16, 18, 20]:
        for i in range(3):
            small_cases.append(generate_semiprime(bits, seed=bits * 100 + i))
    p, f, e = test("Small semiprimes (12-20 bits)", small_cases)
    total_pass += p
    total_fail += f
    total_err += e
    if f + e > 0:
        print("\n  STOP: Fix small semiprime factoring before proceeding.")
        sys.exit(1)

    # ---- Test 3: Medium semiprimes (24-40 bits) ----
    medium_cases = []
    for bits in [24, 28, 32, 36, 40]:
        for i in range(2):
            medium_cases.append(generate_semiprime(bits, seed=bits * 100 + i))
    p, f, e = test("Medium semiprimes (24-40 bits)", medium_cases, timeout=30.0)
    total_pass += p
    total_fail += f
    total_err += e

    # ---- Test 4: Larger semiprimes (48-64 bits) ----
    large_cases = []
    for bits in [48, 56, 64]:
        large_cases.append(generate_semiprime(bits, seed=bits * 100))
    p, f, e = test("Larger semiprimes (48-64 bits)", large_cases, timeout=60.0)
    total_pass += p
    total_fail += f
    total_err += e

    # ---- Test 5: Consistency ----
    print(f"\n  Test: Consistency (same input → same output)")
    print(f"  {'-' * 60}")
    N, p_exp, q_exp = generate_semiprime(20, seed=9999)
    try:
        r1 = factor(N)
        r2 = factor(N)
        if r1 == r2:
            print(f"    factor({N}) = {r1} both times: OK")
            total_pass += 1
        else:
            print(f"    factor({N}) = {r1} then {r2}: INCONSISTENT")
            total_fail += 1
    except Exception as e:
        print(f"    ERROR: {e}")
        total_err += 1

    # ---- Summary ----
    total = total_pass + total_fail + total_err
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY: {total_pass}/{total} passed, {total_fail} failed, {total_err} errors")
    if total_fail + total_err == 0:
        print(f"  STATUS: ALL TESTS PASSED — ready for scoring (python score.py)")
    else:
        print(f"  STATUS: SOME TESTS FAILED — fix before scoring")
    print(f"{'=' * 65}")

    sys.exit(0 if total_fail + total_err == 0 else 1)


if __name__ == "__main__":
    main()
