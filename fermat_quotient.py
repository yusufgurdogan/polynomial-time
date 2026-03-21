#!/usr/bin/env python3
"""
FERMAT QUOTIENTS IN Z/N²Z: The one structure nobody touched.

(Z/N²Z)* has a kernel {1+kN : k=0,...,N-1} ≅ (Z/NZ, +) over (Z/NZ)*.
The Fermat quotient q_p(a) = (a^{p-1} - 1)/p mod p is a LOGARITHM:
  q_p(ab) = q_p(a) + q_p(b) mod p

Computing F(a) = a^{N-1} mod N² captures this logarithmic structure.
The "lift" L(a) = (F(a) - 1) / N lives in Z/NZ and encodes Fermat quotients.

This is OUTSIDE the CRT merger analysis because:
- We compute in Z/N²Z, not Z/NZ
- Fermat quotients are ADDITIVE (logarithmic), not multiplicative
- The lift from (Z/NZ)* to (Z/N²Z)* is where Paillier hides information
"""

import math
import random
import sys
import numpy as np
from sympy import nextprime

sys.stdout.reconfigure(line_buffering=True)


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, hi))
    while q == p:
        q = nextprime(random.randint(lo, hi))
    if p > q: p, q = q, p
    return p * q, p, q


def fermat_lift(a, N):
    """
    Compute F(a) = a^{N-1} mod N² and extract the lift.
    L(a) = (F(a) - 1) / N mod N.

    By Euler's theorem generalization: for gcd(a, N) = 1,
    a^{λ(N)} ≡ 1 mod N, so a^{λ(N)} = 1 + k·N for some k.
    Then (a^{λ(N)} - 1)/N = k mod N.

    We use exponent N-1 instead of λ(N) since we don't know λ(N).
    Note: a^{N-1} mod N might not be 1 (N is not prime!).
    So F(a) = a^{N-1} mod N² might not have F(a) ≡ 1 mod N.
    The lift only makes sense when F(a) ≡ 1 mod N.

    Better: use exponent φ(N) = (p-1)(q-1)... but we don't know that.

    Actually: a^{lcm(p-1,q-1)} ≡ 1 mod N. So a^{lcm(p-1,q-1)} mod N² = 1 + k·N.
    We don't know lcm(p-1,q-1) either.

    What we CAN compute: a^{N-1} mod N².
    This equals a^{(N-1) mod λ(N)} × (a^{λ(N)})^{⌊(N-1)/λ(N)⌋} mod N².
    Hmm, this is complicated.

    Let's just compute it and see what structure emerges.
    """
    N2 = N * N
    F = pow(a, N - 1, N2)
    # Check if F ≡ 1 mod N
    F_mod_N = F % N
    if F_mod_N == 1:
        L = (F - 1) // N % N
        return F, L, True
    else:
        # F(a) mod N ≠ 1, meaning a^{N-1} ≢ 1 mod N (N is not prime)
        # This means N-1 is not a multiple of ord_N(a)
        return F, F_mod_N, False


def fermat_quotient_mod_p(a, p):
    """True Fermat quotient: q_p(a) = (a^{p-1} - 1)/p mod p."""
    val = pow(a, p - 1, p * p)
    return ((val - 1) // p) % p


def experiment(bits_list=None, num_instances=15, num_a=200):
    if bits_list is None:
        bits_list = [16, 20, 24, 28, 32]

    print(f"{'='*75}")
    print(f"  FERMAT QUOTIENTS IN Z/N²Z")
    print(f"  Exploring the lift from (Z/NZ)* to (Z/N²Z)*")
    print(f"{'='*75}")

    for bits in bits_list:
        print(f"\n{'='*75}")
        print(f"  {bits}-BIT SEMIPRIMES")
        print(f"{'='*75}")

        n_factored_gcd = 0
        n_factored_additive = 0
        total = 0
        lift_rate = []

        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)
            N2 = N * N
            total += 1

            # Compute F(a) = a^{N-1} mod N² for many random a
            lifts = []  # (a, L(a), is_liftable)
            Fs = []
            n_liftable = 0

            for _ in range(num_a):
                a = random.randint(2, N - 1)
                if math.gcd(a, N) > 1:
                    continue
                F, L, liftable = fermat_lift(a, N)
                Fs.append((a, F, L, liftable))
                if liftable:
                    lifts.append((a, L))
                    n_liftable += 1

            lift_rate.append(n_liftable / len(Fs) if Fs else 0)

            # --- Analysis 1: GCD of lift differences ---
            factored_gcd = False
            for i in range(len(lifts)):
                for j in range(i + 1, min(i + 50, len(lifts))):
                    a1, L1 = lifts[i]
                    a2, L2 = lifts[j]
                    diff = (L1 - L2) % N
                    if diff != 0:
                        g = math.gcd(diff, N)
                        if 1 < g < N:
                            factored_gcd = True
                            break
                if factored_gcd:
                    break

            if factored_gcd:
                n_factored_gcd += 1

            # --- Analysis 2: Additive structure ---
            # Fermat quotient is additive: q_p(ab) = q_p(a) + q_p(b) mod p
            # So L(ab) should relate to L(a) + L(b)... if L encodes Fermat quotients.
            # Check: L(a*b mod N) vs L(a) + L(b)
            factored_additive = False
            if len(lifts) >= 3:
                for i in range(min(len(lifts), 30)):
                    for j in range(i + 1, min(len(lifts), 30)):
                        a1, L1 = lifts[i]
                        a2, L2 = lifts[j]
                        ab = (a1 * a2) % N
                        if math.gcd(ab, N) > 1:
                            continue
                        F_ab, L_ab, liftable_ab = fermat_lift(ab, N)
                        if not liftable_ab:
                            continue

                        # Deviation from additivity
                        deviation = (L_ab - L1 - L2) % N
                        if deviation != 0:
                            g = math.gcd(deviation, N)
                            if 1 < g < N:
                                factored_additive = True
                                break
                    if factored_additive:
                        break

            if factored_additive:
                n_factored_additive += 1

            # --- Detailed output for first instances ---
            if inst < 2:
                print(f"\n  Instance {inst+1}: N = {N} = {p} × {q}")
                print(f"    Liftable (a^{{N-1}} ≡ 1 mod N): {n_liftable}/{len(Fs)} ({n_liftable/len(Fs):.0%})")

                # True Fermat quotients
                print(f"\n    True Fermat quotients q_p(a), q_q(a) vs lift L(a):")
                for a, L in lifts[:8]:
                    qp = fermat_quotient_mod_p(a, p)
                    qq = fermat_quotient_mod_p(a, q)
                    # L should encode (qp, qq) via CRT somehow
                    # L mod p and L mod q
                    L_mod_p = L % p
                    L_mod_q = L % q
                    print(f"      a={a:>6d}: q_p={qp:>5d} q_q={qq:>5d} | L={L:>8d} | "
                          f"L mod p={L_mod_p:>5d} L mod q={L_mod_q:>5d}")

                # Check: is L mod p proportional to q_p?
                if len(lifts) >= 5:
                    # Collect (q_p(a), L(a) mod p) pairs
                    qp_vals = [fermat_quotient_mod_p(a, p) for a, _ in lifts[:20]]
                    Lp_vals = [L % p for _, L in lifts[:20]]
                    qq_vals = [fermat_quotient_mod_p(a, q) for a, _ in lifts[:20]]
                    Lq_vals = [L % q for _, L in lifts[:20]]

                    # Check if L mod p = c * q_p mod p for some constant c
                    if qp_vals[0] != 0:
                        c_p = (Lp_vals[0] * pow(qp_vals[0], -1, p)) % p
                        matches_p = sum(1 for i in range(len(qp_vals))
                                       if (c_p * qp_vals[i]) % p == Lp_vals[i])
                        print(f"    L mod p = {c_p} × q_p mod p? matches {matches_p}/{len(qp_vals)}")
                    if qq_vals[0] != 0:
                        c_q = (Lq_vals[0] * pow(qq_vals[0], -1, q)) % q
                        matches_q = sum(1 for i in range(len(qq_vals))
                                       if (c_q * qq_vals[i]) % q == Lq_vals[i])
                        print(f"    L mod q = {c_q} × q_q mod q? matches {matches_q}/{len(qq_vals)}")

                # Check additivity deviation
                print(f"\n    Additivity check: L(ab) vs L(a)+L(b) mod N:")
                deviations_shown = 0
                for i in range(min(len(lifts), 5)):
                    for j in range(i+1, min(len(lifts), 5)):
                        a1, L1 = lifts[i]
                        a2, L2 = lifts[j]
                        ab = (a1 * a2) % N
                        if math.gcd(ab, N) > 1: continue
                        F_ab, L_ab, ok = fermat_lift(ab, N)
                        if not ok: continue
                        dev = (L_ab - L1 - L2) % N
                        g = math.gcd(dev, N) if dev != 0 else N
                        flag = f" ← gcd={g} FACTOR!" if 1 < g < N else ""
                        print(f"      L({a1}×{a2}) - L({a1}) - L({a2}) = {dev} mod N{flag}")
                        deviations_shown += 1
                    if deviations_shown >= 5: break

        mean_lift = np.mean(lift_rate)
        print(f"\n  Summary ({total} instances):")
        print(f"    Mean liftable rate: {mean_lift:.1%}")
        print(f"    Factored via GCD of lift diffs: {n_factored_gcd}/{total} ({n_factored_gcd/total:.0%})")
        print(f"    Factored via additivity deviation: {n_factored_additive}/{total} ({n_factored_additive/total:.0%})")

    print(f"\n{'='*75}")
    print(f"  THEORETICAL ANALYSIS")
    print(f"{'='*75}")
    print(f"""
  The lift L(a) = (a^{{N-1}} - 1)/N mod N when a^{{N-1}} ≡ 1 mod N.

  By the theory of p-adic logarithms:
    a^{{N-1}} mod N² = a^{{(p-1)(q-1) + (p+q-2)}} mod N²  (since N-1 = pq-1)

  Hmm, N-1 = pq-1 ≠ φ(N) = (p-1)(q-1) = pq - p - q + 1.
  So N-1 = φ(N) + (p + q - 2).

  a^{{N-1}} = a^{{φ(N)}} × a^{{p+q-2}} mod N².
  a^{{φ(N)}} ≡ 1 mod N (Euler), so a^{{φ(N)}} = 1 + kN for some k.
  Then a^{{N-1}} = (1 + kN) × a^{{p+q-2}} mod N².

  The lift L(a) captures BOTH the Fermat quotient (via k)
  AND the value a^{{p+q-2}} mod N.

  Since p+q is unknown, L(a) mixes Fermat quotient info with p+q info.
  If we could separate them, we'd know p+q → factor N.

  The ADDITIVITY CHECK: L(ab) - L(a) - L(b) mod N
  should be 0 if L were a pure homomorphism. The deviation
  encodes the non-additive part, which relates to p+q.
  If the deviation has a common factor with N → factoring!
    """)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    experiment(bits_list=[16, 20, 24, 28, 32], num_instances=15)
