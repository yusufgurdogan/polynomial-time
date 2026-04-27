#!/usr/bin/env python3
"""
GENUINELY NOVEL APPROACH: Factor n = p*q by solving the binary multiplication
as a system of polynomial equations.

The idea:
  1. Write p and q in binary: p = Σ pᵢ 2ⁱ, q = Σ qⱼ 2ʲ
  2. The equation p*q = n becomes:
     For each bit position m: Σ_{i+j=m} pᵢ*qⱼ + carry_in = nₘ + 2*carry_out
  3. After eliminating carries, we get quadratic equations in pᵢ, qⱼ
  4. Add Boolean constraints: pᵢ² = pᵢ (each bit is 0 or 1)
  5. Solve with Gröbner bases / linearization

If the multiplication structure makes the system "easy" (solvable in poly time),
we have a polynomial-time factoring algorithm.

This is fundamentally different from all smooth-order methods.
No number theory, no groups, no smoothness. Pure algebra.
"""

import math
import time
import random
import sys
from typing import Optional
from sympy import isprime, symbols, groebner, GF, ZZ, Poly, Symbol
from sympy import solve as sym_solve


def factor_algebraic(n: int, verbose: bool = False) -> Optional[list[int]]:
    """Factor n by solving the binary multiplication polynomial system."""
    if n < 4:
        return None
    if n % 2 == 0:
        f = 2
        while n % f == 0:
            n //= f
        result = [2]
        if n > 1:
            sub = factor_algebraic(n)
            if sub:
                result.extend(sub)
        return sorted(result)

    bits = n.bit_length()
    half = (bits + 1) // 2

    # Set up binary variables for p and q
    # p = 1 + 2*p1 + 4*p2 + ... (p is odd, so p0 = 1)
    # q = 1 + 2*q1 + 4*q2 + ... (q is odd, so q0 = 1)
    # p has at most 'half' bits, q has at most 'bits - half + 1' bits

    k = half  # number of bits for p (including p0 = 1)
    m = bits - half + 1  # number of bits for q

    # Create symbolic variables
    p_vars = [Symbol(f'p{i}') for i in range(k)]
    q_vars = [Symbol(f'q{j}') for j in range(m)]
    all_vars = p_vars + q_vars

    # p0 = 1, q0 = 1 (both odd)
    # Build polynomial expression for p and q
    p_expr = 1  # p0 = 1
    for i in range(1, k):
        p_expr += p_vars[i] * (2 ** i)

    q_expr = 1  # q0 = 1
    for j in range(1, m):
        q_expr += q_vars[j] * (2 ** j)

    # The main equation: p * q - n = 0
    product = p_expr * q_expr

    # Expand and collect
    # Instead of one big equation, break into bit-level equations
    # This gives more equations with fewer terms each

    equations = []

    # Boolean constraints: pᵢ² = pᵢ and qⱼ² = qⱼ (bits are 0 or 1)
    for i in range(1, k):
        equations.append(p_vars[i] ** 2 - p_vars[i])
    for j in range(1, m):
        equations.append(q_vars[j] ** 2 - q_vars[j])

    # The multiplication equation p*q = n
    # We break it down using carries
    n_bits = [(n >> i) & 1 for i in range(bits + 2)]

    # Compute partial products at each bit position
    # At position m: sum of pᵢ * qⱼ for i + j = m, plus carry in
    # This equals n_m + 2 * carry_out

    # We'll build equations by tracking the value at each bit position
    # val[m] = Σ_{i+j=m} pᵢ * qⱼ
    # where p₀ = q₀ = 1

    # Build the full product polynomial and extract bit equations
    # Actually, the simplest correct approach: p*q - n = 0
    main_eq = product - n
    equations.append(main_eq)

    # Additional constraint: p < q (WLOG)
    # This means p <= sqrt(n), so the top bit of p is determined
    # We know p < 2^k, which is already encoded in having k bits

    if verbose:
        num_vars = len(all_vars) - 2  # minus p0, q0 which are fixed
        print(f"  Variables: {num_vars} ({k-1} for p, {m-1} for q)")
        print(f"  Equations: {len(equations)} ({k-1} + {m-1} boolean + 1 main)")

    # Solve using Gröbner bases
    # Use variables p1, p2, ..., q1, q2, ... (skip p0, q0)
    solve_vars = [v for v in all_vars if str(v) not in ('p0', 'q0')]

    if not solve_vars:
        return None

    t0 = time.time()

    try:
        # Try SymPy's Gröbner basis
        G = groebner(equations, *solve_vars, order='grevlex', domain=ZZ)
        elapsed = time.time() - t0

        if verbose:
            print(f"  Gröbner basis computed in {elapsed:.3f}s, {len(G.polys)} polynomials")

        # Extract solutions from the Gröbner basis
        solutions = sym_solve(list(G.polys), solve_vars, dict=True)

        for sol in solutions:
            # Reconstruct p and q
            p_val = 1
            for i in range(1, k):
                bit = sol.get(p_vars[i], 0)
                if bit in (0, 1):
                    p_val += int(bit) * (2 ** i)
                else:
                    continue

            q_val = 1
            for j in range(1, m):
                bit = sol.get(q_vars[j], 0)
                if bit in (0, 1):
                    q_val += int(bit) * (2 ** j)
                else:
                    continue

            if p_val > 1 and q_val > 1 and p_val * q_val == n:
                factors = sorted([p_val, q_val])
                return factors

    except Exception as e:
        if verbose:
            print(f"  Gröbner failed: {e}")

    return None


def factor_algebraic_modular(n: int, verbose: bool = False) -> Optional[list[int]]:
    """
    Alternative: solve the system modulo a small prime first,
    then lift using Hensel's lemma.

    Over GF(p) for small p, Gröbner bases are much faster.
    """
    if n % 2 == 0:
        return [2] + (factor_algebraic(n // 2) or [n // 2])

    bits = n.bit_length()
    half = (bits + 1) // 2
    k = half
    m = bits - half + 1

    p_vars = [Symbol(f'p{i}') for i in range(1, k)]
    q_vars = [Symbol(f'q{j}') for j in range(1, m)]
    solve_vars = p_vars + q_vars

    # Build p and q expressions (p0 = q0 = 1)
    p_expr = 1
    for i, v in enumerate(p_vars):
        p_expr += v * (2 ** (i + 1))

    q_expr = 1
    for j, v in enumerate(q_vars):
        q_expr += v * (2 ** (j + 1))

    equations = []
    # Boolean constraints
    for v in solve_vars:
        equations.append(v ** 2 - v)
    # Main equation
    equations.append(p_expr * q_expr - n)

    if verbose:
        print(f"  Trying modular Gröbner over small fields...")

    # Try GF(p) for small primes
    for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        try:
            t0 = time.time()
            eqs_mod = [eq % prime for eq in equations]
            # Over GF(prime), solve
            field = GF(prime)
            G = groebner(equations, *solve_vars, order='grevlex', domain=field)
            elapsed = time.time() - t0

            if verbose:
                print(f"    GF({prime}): {len(G.polys)} polys in {elapsed:.3f}s")

            solutions = sym_solve(list(G.polys), solve_vars, dict=True)
            for sol in solutions:
                p_val = 1
                for i, v in enumerate(p_vars):
                    bit = sol.get(v, 0)
                    if bit in (0, 1):
                        p_val += int(bit) * (2 ** (i + 1))

                q_val = 1
                for j, v in enumerate(q_vars):
                    bit = sol.get(v, 0)
                    if bit in (0, 1):
                        q_val += int(bit) * (2 ** (j + 1))

                if p_val > 1 and q_val > 1 and p_val * q_val == n:
                    return sorted([p_val, q_val])

        except Exception as e:
            if verbose:
                print(f"    GF({prime}) failed: {e}")
            continue

    return None


def factor_algebraic_bitwise(n: int, verbose: bool = False) -> Optional[list[int]]:
    """
    Break the multiplication into bit-level equations with explicit carries.
    This gives MORE equations with FEWER variables per equation,
    which might be easier for Gröbner bases.
    """
    if n % 2 == 0:
        return [2] + (factor_algebraic(n // 2) or [n // 2])

    bits = n.bit_length()
    half = (bits + 1) // 2
    k = half
    m_bits = bits - half + 1

    # Variables for p bits (p0 = 1) and q bits (q0 = 1)
    p_syms = {0: 1}  # p0 = 1
    for i in range(1, k):
        p_syms[i] = Symbol(f'p{i}')

    q_syms = {0: 1}  # q0 = 1
    for j in range(1, m_bits):
        q_syms[j] = Symbol(f'q{j}')

    # Carry variables
    c_syms = {0: 0}  # c0 = 0
    for m in range(1, bits + 1):
        c_syms[m] = Symbol(f'c{m}')

    solve_vars = []
    for i in range(1, k):
        solve_vars.append(p_syms[i])
    for j in range(1, m_bits):
        solve_vars.append(q_syms[j])
    for m in range(1, bits + 1):
        solve_vars.append(c_syms[m])

    equations = []

    # Boolean constraints for p, q bits
    for i in range(1, k):
        equations.append(p_syms[i] ** 2 - p_syms[i])
    for j in range(1, m_bits):
        equations.append(q_syms[j] ** 2 - q_syms[j])

    # Bit-level multiplication equations
    n_bits = [(n >> pos) & 1 for pos in range(bits + 2)]

    for pos in range(bits + 1):
        # Sum of products at this position
        product_sum = c_syms.get(pos, 0)
        for i in range(k):
            j = pos - i
            if 0 <= j < m_bits:
                pi = p_syms.get(i, 0)
                qj = q_syms.get(j, 0)
                product_sum += pi * qj

        # product_sum = n_bit + 2 * carry_out
        # So: product_sum - n_bit - 2 * carry_out = 0
        carry_out = c_syms.get(pos + 1, 0)
        equations.append(product_sum - n_bits[pos] - 2 * carry_out)

    if verbose:
        print(f"  Bitwise formulation: {len(solve_vars)} vars, {len(equations)} eqs")

    t0 = time.time()
    try:
        G = groebner(equations, *solve_vars, order='grevlex', domain=ZZ)
        elapsed = time.time() - t0

        if verbose:
            print(f"  Gröbner: {len(G.polys)} polys in {elapsed:.3f}s")

        solutions = sym_solve(list(G.polys), solve_vars, dict=True)

        for sol in solutions:
            p_val = 1
            for i in range(1, k):
                bit = sol.get(p_syms[i], 0)
                p_val += int(bit) * (2 ** i)

            if p_val > 1 and n % p_val == 0:
                q_val = n // p_val
                if q_val > 1:
                    return sorted([p_val, q_val])

    except Exception as e:
        if verbose:
            print(f"  Failed: {e}")

    return None


# =============================================================================
# Test scaling
# =============================================================================

def test_scaling():
    """Test how Gröbner basis computation scales with bit size."""
    from harness import generate_semiprime, verify_factors

    print("=" * 70)
    print("  ALGEBRAIC FACTORING — SCALING TEST")
    print("  Solving p*q = n as polynomial system via Gröbner bases")
    print("=" * 70)

    methods = {
        "main_eq": factor_algebraic,
        "bitwise": factor_algebraic_bitwise,
    }

    # Start very small and increase
    for bits in [10, 12, 14, 16, 18, 20, 22, 24, 28, 32]:
        print(f"\n--- {bits} bits ---")
        n, p, q = generate_semiprime(bits)
        print(f"  n = {n} = {p} × {q}")

        for name, method in methods.items():
            t0 = time.time()
            try:
                result = method(n, verbose=True)
                elapsed = time.time() - t0
                if result and verify_factors(n, result):
                    print(f"  [{name}] SUCCESS in {elapsed:.3f}s: {result}")
                else:
                    print(f"  [{name}] FAILED in {elapsed:.3f}s (got {result})")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{name}] ERROR in {elapsed:.3f}s: {e}")

            # If it took > 30s, skip larger sizes for this method
            if elapsed > 30:
                print(f"  [{name}] Too slow, skipping for larger sizes")
                methods = {k: v for k, v in methods.items() if k != name}
                break

        if not methods:
            print("\nAll methods too slow, stopping.")
            break


if __name__ == "__main__":
    test_scaling()
