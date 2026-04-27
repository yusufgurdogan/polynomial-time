#!/usr/bin/env python3
"""
ECM-COPPERSMITH BRIDGE

The idea: ECM fails when no curve order is fully B-smooth.
Coppersmith fails when you have zero bits of p.
But what if you combine them?

ECM with "large prime variation": find curves E where #E(F_p) = s * r,
where s is B-smooth and r is a single large prime. Computing [s]P gives
a point Q of order r on E mod p. This point Q encodes information about r,
which constrains p (since #E(F_p) = p + 1 - t, and t is bounded by Hasse).

If we collect constraints on p from MULTIPLE curves, each contributing
partial information, do they accumulate to the n/4 bits Coppersmith needs?

Specifically:
1. For each random curve E, compute [M]P where M = lcm(1,...,B) (stage 1 of ECM)
2. If [M]P ≠ O but [M]P has small order mod N, that's a "large prime hit"
3. The large prime r satisfies: #E(F_p) / gcd(#E(F_p), M) = r
4. From r and the Hasse bound: p + 1 - t = s*r, |t| ≤ 2√p
5. So p ≡ s*r - 1 + t mod (something), with |t| ≤ 2√p
6. Each curve gives: p ∈ [s*r - 1 - 2√p, s*r - 1 + 2√p]
   But we don't know s*r individually — we know s*r mod something.

Wait — we DO know something. If Q = [M]P has order r mod p,
then [r]Q ≡ O mod p. So gcd([r]Q_x denominator, N) might give p.
But we don't know r. We know Q and we know it has SOME order dividing
#E(F_p) / gcd(#E(F_p), M).

The real question: from Q (a point of unknown large order on E mod N),
can we extract bits of p?

Key insight: Q has order r mod p and order r' mod q (possibly different).
If r ≠ r', then for some multiple k, [k]Q ≡ O mod p but [k]Q ≢ O mod q.
Then gcd(denominator of [k]Q, N) = p. This is ECM stage 2.

But stage 2 needs to find k — which is the large prime r.
Standard stage 2 tries all primes in [B, B2].
What if instead of searching for r, we extract INFORMATION from Q
about what r must be, and feed that to Coppersmith?

Let's think about what Q tells us:
- Q = [M]P on E mod N
- Q mod p has order dividing #E(F_p)/gcd(#E(F_p), M)
- Q mod q has order dividing #E(F_q)/gcd(#E(F_q), M)
- The x-coordinate of Q is a specific value mod N

From Q, we can compute:
- [2]Q, [3]Q, [4]Q, ... mod N
- For any k, [k]Q mod N = CRT([k]Q mod p, [k]Q mod q)
- If [k]Q ≡ O mod p: the y-coordinate of [k]Q is 0 mod p
  So p | y_coordinate_of([k]Q)

The constraint on p: there exists k such that the y-coordinate of [k]Q ≡ 0 mod p.
This k is related to r = ord(Q mod p).

Can we extract information about r from the SEQUENCE of x-coordinates
of [1]Q, [2]Q, ..., [B2]Q mod N? This is what ECM stage 2 does.

THE BRIDGE EXPERIMENT:
1. Run ECM stage 1 with various B1 values
2. For points Q where stage 1 didn't factor but Q ≠ O:
   - Q has large prime order r_p mod p and r_q mod q
   - Compute a sequence of scalar multiples of Q
   - Extract constraints on p from the sequence
   - Feed constraints to Coppersmith
3. Compare: does ECM + Coppersmith beat standalone ECM?
"""

import math
import random
import sys
import time
import numpy as np
from sympy import nextprime

sys.stdout.reconfigure(line_buffering=True)


# =========================================================================
# Elliptic curve arithmetic mod N (Montgomery form for speed)
# =========================================================================

def ec_add(P, Q, N, a):
    """Add points on y^2 = x^3 + ax + b mod N. Returns None if inversion fails (= factor found)."""
    if P is None: return Q
    if Q is None: return P

    x1, y1 = P
    x2, y2 = Q

    if x1 == x2:
        if (y1 + y2) % N == 0:
            return None  # Point at infinity
        num = (3 * x1 * x1 + a) % N
        den = (2 * y1) % N
    else:
        num = (y2 - y1) % N
        den = (x2 - x1) % N

    g = math.gcd(den % N, N)
    if g > 1:
        if g < N:
            return ('FACTOR', g)
        return None  # Degenerate

    den_inv = pow(den, -1, N)
    lam = (num * den_inv) % N

    x3 = (lam * lam - x1 - x2) % N
    y3 = (lam * (x1 - x3) - y1) % N

    return (x3, y3)


def ec_mul(k, P, N, a):
    """Scalar multiplication [k]P on curve with parameter a mod N."""
    if k == 0:
        return None
    if k < 0:
        if P is None: return None
        P = (P[0], (-P[1]) % N)
        k = -k

    result = None
    addend = P

    while k > 0:
        if k & 1:
            result = ec_add(result, addend, N, a)
            if isinstance(result, tuple) and result[0] == 'FACTOR':
                return result
        addend = ec_add(addend, addend, N, a)
        if isinstance(addend, tuple) and addend[0] == 'FACTOR':
            return addend
        k >>= 1

    return result


# =========================================================================
# ECM Stage 1
# =========================================================================

def ecm_stage1(N, B1, curve_params=None):
    """
    ECM stage 1: pick a random curve, compute [M]P where M = lcm(1,...,B1).
    Returns (Q, curve_a, factor_or_None).
    """
    if curve_params is None:
        # Suyama parameterization for good curves
        sigma = random.randint(6, N - 1)
        u = (sigma * sigma - 5) % N
        v = (4 * sigma) % N

        g = math.gcd(u * v % N, N)
        if 1 < g < N:
            return None, None, g

        x0 = pow(u, 3, N)
        z0 = pow(v, 3, N)

        # Curve parameter a
        t = (v - u) % N
        a_num = (t * t * t * (3 * u + v)) % N
        a_den = (16 * x0 * v) % N
        g = math.gcd(a_den, N)
        if 1 < g < N:
            return None, None, g
        if g == N:
            return None, None, None

        a_den_inv = pow(a_den, -1, N)
        a = (a_num * a_den_inv - 2) % N

        # Starting point
        g = math.gcd(z0, N)
        if 1 < g < N:
            return None, None, g
        if g == N:
            return None, None, None
        z0_inv = pow(z0, -1, N)
        P = (x0 * z0_inv % N, 1)  # y doesn't matter for the test
    else:
        a, P = curve_params

    # Compute [M]P where M = product of prime powers up to B1
    Q = P
    # Use prime powers
    p = 2
    while p <= B1:
        pp = p
        while pp * p <= B1:
            pp *= p
        Q = ec_mul(pp, Q, N, a)
        if Q is None:
            return None, a, None  # Hit infinity = factored
        if isinstance(Q, tuple) and Q[0] == 'FACTOR':
            return None, a, Q[1]
        # Next prime
        p += 1
        while p <= B1 and not _is_prime_small(p):
            p += 1

    return Q, a, None


def _is_prime_small(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True


# =========================================================================
# ECM Stage 2 (standard: try primes in [B1, B2])
# =========================================================================

def ecm_stage2(Q, N, a, B1, B2):
    """Standard stage 2: check if [r]Q = O for any prime r in [B1, B2]."""
    if Q is None:
        return None

    # Compute [r]Q for primes r in [B1, B2]
    # Use baby-step-giant-step for efficiency
    # Simple version: just try each prime
    p = B1 + 1
    while p <= B2:
        if _is_prime_small(p):
            R = ec_mul(p, Q, N, a)
            if R is None:
                # [p]Q = O, meaning ord(Q) | p
                # But this means p | #E(F_p) or p | #E(F_q)
                # Try gcd
                pass
            if isinstance(R, tuple) and R[0] == 'FACTOR':
                return R[1]
        p += 1

    return None


# =========================================================================
# THE BRIDGE: Extract constraints from stage 1 remainder Q
# =========================================================================

def extract_constraints(Q, N, a, B1, num_curves=20):
    """
    After ECM stage 1, Q = [M]P has order r_p mod p and r_q mod q,
    where r_p and r_q are the "cofactors" (large primes if we're unlucky).

    What can we learn about p from Q?

    Idea 1: The ORDER of Q mod N is lcm(r_p, r_q). If we could compute
    this order, we'd know lcm(r_p, r_q). Combined with the Hasse bound,
    this constrains p.

    Idea 2: Compute [k]Q for k = 1, 2, 3, ... and check if [k]Q has
    special structure. If [k]Q ≡ O mod p (i.e., k = r_p), the
    x-coordinate of [k]Q will satisfy gcd(denom, N) = p.

    Idea 3: The SEQUENCE of x-coordinates of [1]Q, [2]Q, ..., [K]Q
    is a 1D signal on the elliptic curve. The "period" of this signal
    (when viewed mod p) is r_p. Same CRT merger problem as before!

    Idea 4: Multiple curves give multiple (Q_i, a_i) pairs. Each Q_i
    has order r_p^(i) mod p. The COMBINATION of these orders constrains p
    more than any single one.

    Let's test: given k curves each with a large prime cofactor, how
    much information about p accumulates?
    """
    # For each curve, we get Q with unknown order r_p mod p.
    # The Hasse bound says: #E_i(F_p) = p + 1 - t_i, |t_i| ≤ 2√p
    # And r_p^(i) | #E_i(F_p), with #E_i(F_p) / r_p^(i) = s_i (smooth part)
    # So p + 1 - t_i = s_i * r_p^(i)

    # If we knew s_i and r_p^(i), we'd know p + 1 - t_i exactly.
    # That gives p to within 4√p (the Hasse range).
    # For balanced semiprimes, 4√p ≈ 4·N^{1/4}.
    # That's EXACTLY the Coppersmith threshold!

    # So: ONE curve with known s_i and r_p^(i) gives p to within N^{1/4}
    # → Coppersmith factors.

    # The problem: we don't know s_i or r_p^(i) individually.
    # We know M (the stage 1 multiplier) and Q = [M]P.
    # We know s_i | M (since [s_i]([M/s_i]P) would be O mod p... not quite).

    # Actually: M = lcm(1,...,B1). If #E(F_p) = s * r with s | M and r prime > B1,
    # then [M]P ≡ [M/s · s]P ≡ [M/s · 0]P... no.
    # [M]P mod p: the order of P mod p is #E(F_p) = s*r.
    # [M]P mod p = [M mod (s*r)]P mod p.
    # Since s | M: M = s * k for some k. [M]P = [s*k]P.
    # [s*k]P mod p = [k mod r]([s]P) mod p (since [s]P has order r).
    # So [M]P mod p is NOT the identity (unless r | k, unlikely for random k, prime r > B1).
    # The ORDER of [M]P mod p divides r.
    # If r is prime: ord([M]P mod p) = r (unless [M]P ≡ O mod p, which means s*r | M).

    # So Q = [M]P has order r mod p (the large prime cofactor).
    # Similarly, Q has order r' mod q.
    # Q has order lcm(r, r') mod N.

    # To FIND r: we'd need to compute the order of Q mod N, then factor it.
    # Computing the order of Q mod N is at least as hard as factoring N (in general).

    # BUT: if r is "not too large" (say r < B2 for some stage 2 bound),
    # then [r]Q ≡ O mod p → gcd(denom of [r]Q, N) = p.
    # This is standard ECM stage 2.

    # THE BRIDGE IDEA: instead of trying all primes up to B2,
    # use the fact that r = (#E(F_p) - smooth_part) / smooth_part
    # and #E(F_p) ∈ [p+1-2√p, p+1+2√p].
    # For each candidate smooth_part s (there are few: s | M, s in right range),
    # r = (p + 1 - t) / s for some |t| ≤ 2√p.
    # This gives r ∈ [(p + 1 - 2√p)/s, (p + 1 + 2√p)/s].
    # An interval of width 4√p/s.

    # If s ≈ B1 (most of the smooth part absorbed), then r ≈ p/B1.
    # The interval has width 4√p/B1.

    # For each candidate r in this interval: check [r]Q mod N.
    # Number of candidates: 4√p/B1 ≈ 4·N^{1/4}/B1.
    # With B1 = N^{1/4}: only O(1) candidates!

    # But we need s | M AND s in the right range, and we don't know p.
    # We know N and the Hasse bound: p ∈ [√N - N^{1/4}, √N + N^{1/4}] roughly.
    # So #E(F_p) ∈ [√N - 3N^{1/4}, √N + 3N^{1/4}].
    # For each divisor s of M in the right range:
    #   r candidates ∈ [(√N - 3N^{1/4})/s, (√N + 3N^{1/4})/s]
    #   Width of r-interval: 6N^{1/4}/s

    return None  # Placeholder — the real computation follows


# =========================================================================
# The actual experiment
# =========================================================================

def bridge_experiment(bits_list=None, num_instances=20):
    if bits_list is None:
        bits_list = [24, 32, 40, 48]

    print(f"{'='*75}")
    print(f"  ECM-COPPERSMITH BRIDGE EXPERIMENT")
    print(f"{'='*75}")
    print(f"  Question: can partial smoothness from ECM stage 1,")
    print(f"  combined with Hasse bound, constrain p enough for Coppersmith?")
    print()
    print(f"  Key calculation:")
    print(f"  After ECM stage 1 with bound B1, Q = [M]P has order r ≈ p/B1 mod p.")
    print(f"  #E(F_p) = s*r where s|M (smooth) and r prime.")
    print(f"  p + 1 - t = s*r, |t| ≤ 2√p.")
    print(f"  For each divisor s of M: r-candidates in interval of width 6N^(1/4)/s.")
    print(f"  With B1 ≈ N^(1/4): width ≈ 6, i.e., O(1) candidates per s.")
    print(f"  Number of divisors of M: d(M) ≈ exp(O(√(B1/ln B1))).")
    print()

    for bits in bits_list:
        print(f"\n{'='*75}")
        print(f"  {bits}-BIT SEMIPRIMES")
        print(f"{'='*75}")

        n = bits
        sqrt_N = 2 ** (bits // 2)
        N_fourth = 2 ** (bits // 4)

        # B1 values to try
        B1_values = [
            max(10, bits),          # tiny
            max(20, bits * 2),      # small
            max(50, bits * 5),      # medium
            N_fourth // 4,          # quarter of N^{1/4}
            N_fourth,               # N^{1/4}
            N_fourth * 4,           # 4 * N^{1/4}
        ]
        B1_values = sorted(set(b for b in B1_values if b < sqrt_N))

        num_curves = 50  # curves per instance

        for B1 in B1_values:
            ecm_only_count = 0
            bridge_count = 0
            total = 0

            for inst in range(num_instances):
                N, p, q = generate_semiprime(bits)
                total += 1
                ecm_found = False
                bridge_found = False

                for curve in range(num_curves):
                    Q, a, factor = ecm_stage1(N, B1)

                    if factor and 1 < factor < N:
                        ecm_found = True
                        break

                    if Q is None:
                        continue

                    # Q exists and no factor from stage 1.
                    # Q has order r_p mod p where #E(F_p)/gcd(#E(F_p),M) = r_p.

                    # THE BRIDGE: compute M = lcm(1,...,B1), then enumerate
                    # divisors s of M near #E(F_p). For each s, check if
                    # [(p_approx + 1)/s] * Q ≡ O mod N yields a factor.

                    # We don't know p exactly, but p ≈ √N (balanced).
                    p_approx = int(math.sqrt(N))
                    hasse_range = int(4 * N ** 0.25) + 1

                    # For each candidate group order near p_approx + 1:
                    for t in range(-hasse_range, hasse_range + 1, max(1, hasse_range // 20)):
                        group_order_candidate = p_approx + 1 - t

                        if group_order_candidate <= 0:
                            continue

                        # The smooth part s = gcd(group_order_candidate, M)
                        # where M = lcm(1,...,B1)
                        # Instead of computing M exactly, compute the B1-smooth part
                        s = group_order_candidate
                        temp = s
                        for pp in range(2, B1 + 1):
                            if not _is_prime_small(pp):
                                continue
                            while temp % pp == 0:
                                temp //= pp
                        r_candidate = temp  # The remaining (hopefully prime) cofactor

                        if r_candidate <= 1 or r_candidate > N:
                            continue

                        # Check: does [r_candidate]Q give a factor?
                        R = ec_mul(r_candidate, Q, N, a)
                        if isinstance(R, tuple) and R[0] == 'FACTOR':
                            bridge_found = True
                            break
                        if R is None:
                            # [r]Q = O mod N, meaning r_candidate works mod both p and q
                            # Not directly useful, but try gcd
                            pass

                    if bridge_found:
                        break

                if ecm_found:
                    ecm_only_count += 1
                if bridge_found:
                    bridge_count += 1

            ecm_rate = ecm_only_count / total
            bridge_rate = bridge_count / total
            combined = (ecm_only_count + bridge_count -
                       min(ecm_only_count, bridge_count))  # rough

            print(f"  B1={B1:8d} (B1/N^{{1/4}}={B1/N_fourth:.2f}): "
                  f"ECM_stage1={ecm_rate:.0%}  "
                  f"bridge={bridge_rate:.0%}  "
                  f"{'<-- BRIDGE HELPS' if bridge_rate > ecm_rate else ''}")

    print(f"\n{'='*75}")
    print(f"  THEORETICAL ANALYSIS")
    print(f"{'='*75}")
    print(f"""
  The ECM-Coppersmith bridge works in principle:
  - ECM stage 1 with B1 ≈ N^{{1/4}} absorbs the B1-smooth part
  - The cofactor r ≈ p/B1 ≈ N^{{1/4}}
  - The Hasse bound constrains r to an interval of width O(N^{{1/4}}/B1) = O(1)
  - So O(1) candidates for r per curve

  BUT: the smooth part s is not exactly gcd(#E(F_p), M).
  It's the largest divisor of #E(F_p) all of whose prime factors are ≤ B1.
  Computing this requires knowing #E(F_p), which requires knowing p.

  We approximate: enumerate divisors of candidate group orders.
  This works when the approximation p ≈ √N is good enough,
  i.e., when |p - √N| < N^{{1/4}} (balanced primes).

  The number of candidates to check per curve: O(hasse_range / step) × 1.
  Total work: O(num_curves × hasse_range × EC_mul_cost).
  For B1 = N^{{1/4}}: work ≈ O(N^{{1/4}} × N^{{1/4}} × poly(n)) = O(N^{{1/2}}).
  This is NOT better than standard ECM or trial division.

  For the bridge to win: B1 must be smaller (so ECM stage 1 is cheaper)
  but the Hasse constraint must still narrow r to O(1) candidates.
  This requires B1 ≈ N^{{1/4}}, which makes the total work O(N^{{1/2}}).

  VERDICT: The ECM-Coppersmith bridge doesn't beat standalone ECM because
  the Hasse constraint width scales as N^{{1/4}}/B1, and you need B1 ≈ N^{{1/4}}
  to get O(1) candidates. The total work is Ω(N^{{1/4}}) regardless.
    """)


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, hi))
    while q == p:
        q = nextprime(random.randint(lo, hi))
    if p > q: p, q = q, p
    return p * q, p, q


if __name__ == "__main__":
    random.seed(42)
    bridge_experiment(bits_list=[24, 32, 40], num_instances=15)
