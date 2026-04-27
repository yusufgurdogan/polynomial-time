#!/usr/bin/env python3
"""
Belief propagation on the CRT factor graph for factoring.

Instead of the binary multiplication circuit (global carry coupling),
decompose via CRT:

For each small prime m: (p mod m) * (q mod m) ≡ N mod m

This gives a factor graph where:
- Variable nodes: bits of p (k binary variables)
- Factor nodes: one per small prime m, constraining (p mod m)
- Each factor only touches O(log m) bits of p (those that determine p mod m)

The CRT graph has MUCH weaker coupling than the binary circuit.
Question: does BP converge and recover bits of p?

Also test: Survey Propagation for the multi-modal case.
"""

import math
import random
import sys
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

sys.stdout.reconfigure(line_buffering=True)


def small_primes(limit):
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            for j in range(i*i, limit + 1, i):
                s[j] = False
    return [i for i in range(limit + 1) if s[i]]


# =============================================================================
# CRT Factor Graph
# =============================================================================
# Variable nodes: p_0, p_1, ..., p_{k-1} (bits of p, each ∈ {0, 1})
# Factor nodes: f_m for each small prime m
#   f_m constrains: the integer p = Σ p_i 2^i must satisfy
#   p mod m ∈ S_m, where S_m = {r : 1 ≤ r < m, gcd(r, m) coprime, (N/r) mod m valid}
#
# The key insight: p mod m only depends on p_0, ..., p_{ceil(log2(m))-1}
# WRONG — p mod m depends on ALL bits of p (because 2^i mod m wraps around)
# BUT: the weights 2^i mod m are periodic with period ord(2, m),
# so we can group bits by their residue class.
#
# Actually, p mod m = Σ p_i * (2^i mod m) mod m
# This is a LINEAR function of the bits mod m.
# Each factor f_m is a LINEAR constraint mod m on ALL bits.
# But the constraint is "soft" — it restricts p mod m to a set S_m.

class CRTFactorGraph:
    def __init__(self, N: int, k: int, primes: List[int]):
        """
        N: the composite to factor
        k: number of bits in p (p < 2^k)
        primes: small primes to use as CRT moduli
        """
        self.N = N
        self.k = k
        self.primes = primes
        self.num_vars = k  # bits of p

        # For each prime m, compute:
        # 1. The set S_m of valid residues for p mod m
        # 2. The weights w_i = 2^i mod m for each bit position i
        self.valid_residues = {}  # m -> set of valid p mod m
        self.weights = {}  # m -> list of 2^i mod m

        for m in primes:
            # Valid residues: r such that N*r^(-1) mod m is also nonzero
            # (both p mod m and q mod m must be nonzero)
            n_mod = N % m
            valid = set()
            for r in range(1, m):
                if math.gcd(r, m) != 1:
                    continue
                q_mod = (n_mod * pow(r, -1, m)) % m
                if q_mod > 0 and math.gcd(q_mod, m) == 1:
                    valid.add(r)
            self.valid_residues[m] = valid

            # Weights
            self.weights[m] = [pow(2, i, m) for i in range(k)]

        # Messages: factor → variable and variable → factor
        # msg_f2v[m][i] = P(p_i = 1 | factor m)  (as log-likelihood ratio)
        # msg_v2f[i][m] = P(p_i = 1 | all factors except m)
        self.msg_f2v = {m: [0.0] * k for m in primes}  # 0 = uniform
        self.msg_v2f = {i: {m: 0.0 for m in primes} for i in range(k)}

        # Prior: p is odd (p_0 = 1), p has k bits (p_{k-1} = 1)
        self.prior = [0.0] * k  # log(P(1)/P(0)), 0 = uniform
        self.prior[0] = 100.0   # p_0 = 1 (certainty)
        self.prior[k-1] = 100.0  # p_{k-1} = 1 (k-bit number)

    def bp_iterate(self, num_iters: int = 50, damping: float = 0.5):
        """Run loopy belief propagation."""
        k = self.k

        for iteration in range(num_iters):
            max_change = 0.0

            # Variable → Factor messages
            for i in range(k):
                for m in self.primes:
                    # msg_v2f[i][m] = prior[i] + Σ_{m' ≠ m} msg_f2v[m'][i]
                    total = self.prior[i]
                    for m2 in self.primes:
                        if m2 != m:
                            total += self.msg_f2v[m2][i]
                    old = self.msg_v2f[i][m]
                    self.msg_v2f[i][m] = damping * total + (1 - damping) * old
                    max_change = max(max_change, abs(self.msg_v2f[i][m] - old))

            # Factor → Variable messages
            for m in self.primes:
                valid = self.valid_residues[m]
                weights = self.weights[m]

                for target_i in range(k):
                    # msg_f2v[m][target_i] = log-likelihood ratio that p_{target_i} = 1
                    # based on the constraint p mod m ∈ valid,
                    # using beliefs about all other bits from msg_v2f

                    # Compute P(constraint satisfied | p_i = 0) and P(... | p_i = 1)
                    # This requires marginalizing over all other bits.
                    #
                    # For exact computation with m small:
                    # P(p mod m = r | p_i = b, other beliefs) can be computed
                    # by convolving the per-bit distributions mod m.

                    # Get beliefs about other bits (as probabilities of being 1)
                    other_probs = []
                    for j in range(k):
                        if j == target_i:
                            continue
                        llr = self.msg_v2f[j][m]
                        prob1 = 1.0 / (1.0 + math.exp(-llr)) if abs(llr) < 30 else (1.0 if llr > 0 else 0.0)
                        other_probs.append((j, prob1))

                    # Compute distribution of (Σ_{j≠i} p_j * w_j) mod m
                    # via convolution
                    dist = np.zeros(m)
                    dist[0] = 1.0  # start with 0

                    for j, prob1 in other_probs:
                        w = weights[j]
                        new_dist = np.zeros(m)
                        for r in range(m):
                            # p_j = 0 (prob 1-prob1): residue stays at r
                            new_dist[r] += dist[r] * (1 - prob1)
                            # p_j = 1 (prob prob1): residue shifts by w
                            new_dist[(r + w) % m] += dist[r] * prob1
                        dist = new_dist

                    # Now dist[r] = P(Σ_{j≠i} p_j w_j ≡ r mod m)
                    w_i = weights[target_i]

                    # P(constraint | p_i = 0) = Σ_{r ∈ valid} dist[r]
                    p_sat_0 = sum(dist[r] for r in valid)
                    # P(constraint | p_i = 1) = Σ_{r : (r + w_i) mod m ∈ valid} dist[r]
                    #                         = Σ_{r ∈ valid} dist[(r - w_i) mod m]
                    p_sat_1 = sum(dist[(r - w_i) % m] for r in valid)

                    # Log-likelihood ratio
                    if p_sat_0 > 1e-300 and p_sat_1 > 1e-300:
                        new_msg = math.log(p_sat_1 / p_sat_0)
                    elif p_sat_1 > 1e-300:
                        new_msg = 30.0
                    elif p_sat_0 > 1e-300:
                        new_msg = -30.0
                    else:
                        new_msg = 0.0

                    # Clip for stability
                    new_msg = max(-30.0, min(30.0, new_msg))

                    old = self.msg_f2v[m][target_i]
                    self.msg_f2v[m][target_i] = damping * new_msg + (1 - damping) * old
                    max_change = max(max_change, abs(self.msg_f2v[m][target_i] - old))

            if max_change < 1e-6:
                return iteration + 1  # converged

        return num_iters

    def get_beliefs(self) -> List[float]:
        """Get marginal belief P(p_i = 1) for each bit."""
        beliefs = []
        for i in range(self.k):
            llr = self.prior[i]
            for m in self.primes:
                llr += self.msg_f2v[m][i]
            prob1 = 1.0 / (1.0 + math.exp(-llr)) if abs(llr) < 30 else (1.0 if llr > 0 else 0.0)
            beliefs.append(prob1)
        return beliefs

    def decode_hard(self) -> int:
        """Hard decode: return the integer p from current beliefs."""
        beliefs = self.get_beliefs()
        p = 0
        for i in range(self.k):
            if beliefs[i] > 0.5:
                p |= (1 << i)
        return p


def test_bp_crt(k: int, num_primes: int, num_trials: int = 50,
                bp_iters: int = 100, damping: float = 0.3):
    """Test BP on CRT factor graph."""
    from sympy import nextprime

    primes = small_primes(200)[:num_primes]

    print(f"\n{'='*70}")
    print(f"  BP on CRT Factor Graph: k={k} bits, {num_primes} primes")
    print(f"  Primes: {primes[:15]}{'...' if len(primes) > 15 else ''}")
    print(f"  BP iters={bp_iters}, damping={damping}")
    print(f"{'='*70}")

    full_success = 0
    partial_70 = 0
    partial_50 = 0
    avg_correct_bits = 0
    avg_converge_iters = 0
    avg_max_belief = 0

    for trial in range(num_trials):
        # Generate instance
        lo, hi = 1 << (k - 1), (1 << k) - 1
        p = nextprime(random.randint(lo, hi))
        while p > hi:
            p = nextprime(random.randint(lo, hi))
        q = nextprime(random.randint(lo, hi))
        while q > hi or q == p:
            q = nextprime(random.randint(lo, hi))
        if p > q:
            p, q = q, p
        N = p * q

        # Build and run BP
        fg = CRTFactorGraph(N, k, primes)
        iters = fg.bp_iterate(num_iters=bp_iters, damping=damping)
        avg_converge_iters += iters

        beliefs = fg.get_beliefs()
        p_decoded = fg.decode_hard()

        # Count correct bits
        correct = sum(1 for i in range(k) if ((p >> i) & 1) == ((p_decoded >> i) & 1))
        avg_correct_bits += correct

        # Track belief strength
        max_non_fixed = max((abs(b - 0.5) for i, b in enumerate(beliefs)
                            if i not in [0, k-1]), default=0)
        avg_max_belief += max_non_fixed

        if p_decoded == p or p_decoded == q:
            full_success += 1
        if correct >= k * 0.7:
            partial_70 += 1
        if correct >= k * 0.5:
            partial_50 += 1

        # Detailed output for first few trials
        if trial < 3:
            print(f"\n  Trial {trial}: N={N}, p={p}, q={q}")
            print(f"  Decoded: {p_decoded} ({'CORRECT!' if p_decoded in [p,q] else 'wrong'})")
            print(f"  Correct bits: {correct}/{k} ({correct/k:.0%})")
            print(f"  Converged in {iters} iters")
            # Show beliefs for uncertain bits
            uncertain = [(i, beliefs[i]) for i in range(k)
                        if 0.1 < beliefs[i] < 0.9 and i not in [0, k-1]]
            if uncertain:
                print(f"  Uncertain bits: {[(i, f'{b:.3f}') for i, b in uncertain[:10]]}")
            fixed = [(i, beliefs[i]) for i in range(k)
                    if (beliefs[i] > 0.9 or beliefs[i] < 0.1) and i not in [0, k-1]]
            if fixed:
                correct_fixed = sum(1 for i, b in fixed
                                   if ((b > 0.5) == bool((p >> i) & 1)))
                print(f"  Confident bits: {len(fixed)} ({correct_fixed} correct)")

    avg_correct_bits /= num_trials
    avg_converge_iters /= num_trials
    avg_max_belief /= num_trials

    print(f"\n  --- Results ({num_trials} trials) ---")
    print(f"  Full recovery:  {full_success}/{num_trials} ({full_success/num_trials:.1%})")
    print(f"  70%+ bits:      {partial_70}/{num_trials} ({partial_70/num_trials:.1%})")
    print(f"  50%+ bits:      {partial_50}/{num_trials} ({partial_50/num_trials:.1%})")
    print(f"  Avg correct:    {avg_correct_bits:.1f}/{k} ({avg_correct_bits/k:.1%})")
    print(f"  Avg convergence: {avg_converge_iters:.1f} iters")
    print(f"  Avg max belief:  {avg_max_belief:.4f} (0=uniform, 0.5=certain)")

    return full_success, partial_70, avg_correct_bits / k


def scaling_test():
    """How does BP performance scale with k and number of primes?"""
    print(f"\n{'='*70}")
    print(f"  SCALING TEST: BP on CRT Factor Graph")
    print(f"{'='*70}")

    results = []
    for k in [8, 10, 12, 14, 16, 20]:
        for np_count in [10, 20, 40, 80]:
            if np_count > k * 5:
                continue
            full, partial, frac = test_bp_crt(k, np_count, num_trials=50,
                                               bp_iters=200, damping=0.3)
            results.append((k, np_count, full, partial, frac))

    print(f"\n{'='*70}")
    print(f"  SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'k':>4} | {'#primes':>7} | {'full':>5} | {'70%+':>5} | {'avg%':>6}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}")
    for k, np_count, full, partial, frac in results:
        print(f"  {k:4d} | {np_count:7d} | {full:5d} | {partial:5d} | {frac:5.1%}")


if __name__ == "__main__":
    random.seed(42)
    scaling_test()
