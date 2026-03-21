#!/usr/bin/env python3
"""
TENSOR STRUCTURE OF MULTIPLICATION: Entanglement entropy of factoring.

Represent N = p × q as a constraint satisfaction problem on bits.
The carry chain creates a 1D tensor network (Matrix Product State).
Measure the ENTANGLEMENT ENTROPY at each cut.

If entropy is polylog(n): tensor contraction methods (DMRG) work in poly time.
If entropy is Θ(n): exponential, no tensor shortcut.

This is a measurement nobody has done. It characterizes the INTRINSIC
difficulty of factoring from an information-theoretic perspective,
independent of any specific algorithm.

The state at position k: all valid (p[0:k], q[0:k], carry_k)
configurations consistent with N[0:k].

Bond dimension = number of reachable carry values at position k.
Entanglement entropy = entropy of the distribution over carries
(or over the full boundary state).
"""

import math
import random
import sys
import time
import numpy as np
from collections import defaultdict
from sympy import nextprime

sys.stdout.reconfigure(line_buffering=True)


def get_bit(x, i):
    return (x >> i) & 1


def generate_semiprime(bits):
    half = bits // 2
    lo, hi = 1 << (half - 1), (1 << half) - 1
    p = nextprime(random.randint(lo, hi))
    q = nextprime(random.randint(lo, hi))
    while q == p:
        q = nextprime(random.randint(lo, hi))
    if p > q: p, q = q, p
    return p * q, p, q


# =========================================================================
# EXACT STATE ENUMERATION
# =========================================================================

def enumerate_states(N, k_bits):
    """
    Enumerate ALL valid (p_bits, q_bits, carry) states at each position.

    At position i: we know N[0:i] and need to find all (p[0:i], q[0:i], carry_i)
    such that the multiplication constraints are satisfied.

    The constraint at position i:
      N_i = (sum_{j+l=i} p_j * q_l + carry_i) mod 2
      carry_{i+1} = (sum_{j+l=i} p_j * q_l + carry_i) // 2

    We track the "boundary state" = carry value at each position.
    """
    num_bits = N.bit_length() + 1
    N_bits = [(N >> i) & 1 for i in range(num_bits + k_bits)]

    # State: (carry_value) → count of valid (p[0:i], q[0:i]) configurations
    # We propagate from position 0 to position num_bits.

    # Initial state: carry_0 = 0, p[0] and q[0] both odd (for odd N)
    # At position 0: p_0 * q_0 + carry_0 = N_0 + 2 * carry_1
    # Since carry_0 = 0: p_0 * q_0 = N_0 + 2 * carry_1
    # N_0 = 1 (N is odd), so p_0 * q_0 = 1 + 2 * carry_1
    # p_0, q_0 ∈ {0, 1}. p_0 * q_0 ∈ {0, 1}.
    # 1 + 2*carry_1 = p_0*q_0. So p_0*q_0 = 1 (both must be 1) and carry_1 = 0.

    # More generally: at position i, the convolution term is
    # conv_i = sum_{j=0}^{min(i, k_bits-1)} p_j * q_{i-j}
    # where q_{i-j} is defined only for 0 ≤ i-j < k_bits.

    # For the exact enumeration, we track all possible states:
    # state = carry value → number of (p, q) prefix pairs giving that carry

    # Position 0
    states = defaultdict(int)  # carry → count

    # We track: carry → count of valid (p_prefix, q_prefix) pairs
    # At each step, for each current carry and each choice of new p_bit and q_bit,
    # compute the new carry and update.

    # But we need to track not just the carry but the FULL p and q prefixes
    # to compute the convolution at each position. The convolution at position i
    # depends on ALL previous bits of p and q.

    # For EXACT enumeration, we need to track (p_prefix, q_prefix) → carry.
    # But this is 2^{2k} states — exponential.

    # HOWEVER: for the CARRY, the only thing that matters from position i+1
    # onwards is the carry value and the bits p[0:i+1], q[0:i+1] (for future
    # convolution terms). This is still exponential in i.

    # SIMPLIFICATION: For k-bit p and q, at position i where i ≥ k,
    # no new p or q bits are introduced. The carry at position i depends on
    # the carry at position i-1 and the convolution term at position i.
    # The convolution term at position i depends on p[max(0,i-k+1):k] and q[0:min(i+1,k)].
    # For i ≥ k, the convolution term uses a "window" of k bits of p and q.

    # For SMALL k (the number of bits in p and q), we can enumerate.
    # k = bits/2. For bits=16, k=8: 2^16 prefix pairs at the worst position.

    # Let's do it exactly for small N.
    k = k_bits

    # Track: at each position i, the set of (p_prefix, q_prefix, carry) tuples
    # that are consistent with N[0:i].

    # More efficiently: track carry → set of (p_prefix, q_prefix)
    # But for counting, we just need carry → count.

    # Actually for entropy computation, we need the carry DISTRIBUTION:
    # P(carry_i = c) = (# of (p,q) pairs with that carry) / (total # valid (p,q) pairs)

    # FAST VERSION: just track carry → count, using the recurrence.
    # At position i: for each (carry_in, p_i, q_i) triple, the convolution adds
    # p_i * q_0 + p_{i-1} * q_1 + ... to the carry. But this depends on ALL
    # previous bits, not just the current ones.

    # THE ISSUE: the carry at position i depends on bits p[0:i+1] and q[0:i+1],
    # not just on carry_{i-1} and (p_i, q_i). The carry is NOT a Markov chain
    # on carry values alone.

    # So the exact computation requires tracking the full (p_prefix, q_prefix).
    # For k-bit factors: 4^k states, which is 4^8 = 65536 for 16-bit N.
    # Doable!

    # Let's enumerate ALL valid (p, q) pairs with N = p * q,
    # then compute the carry at each position for each pair,
    # and finally compute the entropy of the carry distribution.

    return None  # We'll use a different approach below


def compute_carry_entropy(N, p, q):
    """
    For a specific N = p * q, compute the carry at each position
    of the long multiplication, and the entropy of the carry distribution
    across ALL valid factorizations of N.

    For generic N with many factorizations, this measures the uncertainty
    about the carry (= the entanglement entropy).
    For N = pq (semiprime), there's only one factorization, so the carry
    is determined — entropy = 0.

    BUT: the question is about the CONSTRAINT PROPAGATION from partial
    information. Given N[0:k] (the first k bits), how many (p, q, carry_k)
    states are valid?
    """
    n = N.bit_length()
    k = (n + 1) // 2  # bits in each factor

    # The carry at position i in the multiplication p × q:
    # digit_i = (conv_i + carry_i) mod 2
    # carry_{i+1} = (conv_i + carry_i) // 2
    # where conv_i = sum_{j+l=i, 0≤j<k, 0≤l<k} p_j * q_l

    carries = [0] * (2 * k + 1)
    carry = 0
    for i in range(2 * k):
        conv = 0
        for j in range(max(0, i - k + 1), min(i + 1, k)):
            l = i - j
            if 0 <= l < k:
                conv += get_bit(p, j) * get_bit(q, l)
        total = conv + carry
        carries[i] = carry
        carry = total // 2

    return carries


def measure_boundary_entropy(N_bits, k, num_positions=None):
    """
    Given N (as bits), enumerate ALL (p, q) pairs with p*q matching N[0:pos]
    for each position pos. Measure:
    1. Number of valid (p_prefix, q_prefix) pairs at each position
    2. Carry distribution at each position
    3. Entanglement entropy at each position

    This is the EXACT measurement of the tensor network bond dimension.
    """
    if num_positions is None:
        num_positions = 2 * k

    n_bits = len(N_bits)

    # Forward propagation: enumerate all valid states at each position.
    # State = (p_prefix as int, q_prefix as int, carry as int)
    # But this is too many states. Instead, use the carry-only approximation.

    # Actually, let's do it right but limit to small k.
    if k > 12:
        return None  # Too large for exact enumeration

    # Enumerate by building up bit by bit.
    # At position i, we've assigned p[0:?] and q[0:?] bits.
    # The multiplication produces bits at position i from convolution of
    # all (p_j, q_l) with j + l = i.

    # The "boundary" at position i is the carry value.
    # But the carry depends on all previous bits, so we need to
    # track the full state.

    # EFFICIENT APPROACH: dynamic programming over the carry.
    # State at position i: carry value.
    # Transition: given carry_i, for each new pair of bits (p_bit, q_bit)
    # that could appear at this position, compute the new carry.

    # PROBLEM: at position i, the convolution term depends on bits p[j] and q[i-j]
    # for j = 0, ..., i. So at position i, we need to "remember" all previous p and q bits.

    # EXACT DP WITH FULL STATE:
    # State = (carry, p_value_so_far, q_value_so_far)
    # But p_value and q_value need i bits each → 2^{2i} states at position i.

    # For k ≤ 10: 2^{20} ≈ 1M states at the worst position. Doable.

    # Actually, let me think differently. We want to know, at each "cut"
    # position i, how many distinct (p, q) factorizations of N are consistent
    # with N[0:i], and what the carry distribution is.

    # Since N = pq has very few factorizations (typically 1-4 for semiprimes),
    # the entropy is always 0 or very low.

    # THE MORE INTERESTING QUESTION: forget that N is a semiprime.
    # For a GENERIC n-bit number N, how many (p, q) pairs with p*q = N
    # have a given carry profile? And what's the entropy of the carry at each cut?

    # For semiprimes: N = pq has exactly 1 non-trivial factorization (up to order).
    # So the carry is FULLY DETERMINED → entropy = 0 at every position.
    # This is trivially true and not useful.

    # THE RIGHT QUESTION: given only the FIRST i bits of N (not all of them),
    # how many (p, q) carry states are possible?
    # That is: what is the state space when you're PARTIALLY through the computation?

    # This is the "online" version: at each position, how much ambiguity remains?

    # State = carry value. For each carry value c at position i,
    # count the number of (p[0:i+1], q[0:i+1]) pairs that produce carry c.

    # DP:
    # At position 0: carry_in = 0. conv_0 = p_0 * q_0.
    #   total = p_0 * q_0. N_0 = total % 2. carry_1 = total // 2.
    #   For N_0 = 1: p_0 = q_0 = 1 (since 1*1 = 1). carry_1 = 0.
    #   For N_0 = 0: p_0*q_0 = 0 → (0,0), (0,1), (1,0). carry_1 = 0.

    # At position i: conv_i = sum_{j+l=i} p_j * q_l.
    #   total = conv_i + carry_i. N_i = total % 2. carry_{i+1} = total // 2.

    # The issue: conv_i depends on ALL p[0:i+1] and q[0:i+1],
    # not just on the new bits p_i and q_i.
    # So to compute conv_i, we need to know all previous bits.

    # RESOLUTION: Track the full p_prefix and q_prefix in the DP.
    # State = (p_int, q_int, carry) where p_int and q_int are the i-bit prefixes.
    # Transition: add bit p_i and q_i, compute new carry.

    # For k-bit factors at position i:
    # - p_int has i+1 bits (i ≤ k-1), q_int has min(i+1, k) bits
    # - At position i (i < k): introduce new p_i and q_i bits
    # - At position i (i ≥ k): no new bits, just propagate carry

    # The state space at position i (for i < k):
    # 2^{i+1} × 2^{i+1} × max_carry ≈ 4^i × i

    # For k = 8 (16-bit N): at worst position (i=7): 4^8 × 8 ≈ 500K states.
    # Doable.

    # For k = 10 (20-bit N): at worst position: 4^10 × 10 ≈ 10M states.
    # Tight but doable.

    # For k = 12 (24-bit N): 4^12 × 12 ≈ 200M. Too much.

    # Let's implement for k ≤ 10.

    # State representation: (p_int, q_int) → carry
    # At each position, for each existing state, try all (p_bit, q_bit) combinations.

    # Phase 1: positions 0 to k-1 (introduce new bits of p and q)
    # Phase 2: positions k to 2k-1 (no new bits, just carry propagation)

    # We need N_bits for the constraint.

    # Initialize
    # Position 0: p_0, q_0 must satisfy p_0*q_0 + 0 ≡ N_0 (mod 2)
    # and carry_1 = (p_0*q_0) // 2

    states = {}  # (p_int, q_int) → carry

    for p0 in range(2):
        for q0 in range(2):
            total = p0 * q0
            if total % 2 == N_bits[0]:
                carry = total // 2
                states[(p0, q0)] = carry

    carry_counts_by_pos = [defaultdict(int)]
    for (p_int, q_int), c in states.items():
        carry_counts_by_pos[0][c] += 1

    # Propagate
    for pos in range(1, min(2 * k, num_positions)):
        new_states = {}
        carry_counts = defaultdict(int)

        if pos < k:
            # Introduce new bits p_{pos} and q_{pos}
            for (p_int, q_int), carry_in in states.items():
                for p_bit in range(2):
                    for q_bit in range(2):
                        # New p and q values
                        new_p = p_int | (p_bit << pos)
                        new_q = q_int | (q_bit << pos)

                        # Compute convolution at this position
                        conv = 0
                        for j in range(pos + 1):
                            l = pos - j
                            if 0 <= l <= pos:
                                pj = (new_p >> j) & 1
                                ql = (new_q >> l) & 1
                                conv += pj * ql

                        total = conv + carry_in
                        if total % 2 == N_bits[pos]:
                            new_carry = total // 2
                            key = (new_p, new_q)
                            if key in new_states:
                                # Should have same carry (deterministic given prefix)
                                assert new_states[key] == new_carry
                            else:
                                new_states[key] = new_carry
                                carry_counts[new_carry] += 1
        else:
            # No new bits, just propagate carry
            for (p_int, q_int), carry_in in states.items():
                # Convolution at position pos uses p[j] and q[pos-j]
                # for j = max(0, pos-k+1) to min(pos, k-1)
                conv = 0
                for j in range(max(0, pos - k + 1), min(pos + 1, k)):
                    l = pos - j
                    if 0 <= l < k:
                        pj = (p_int >> j) & 1
                        ql = (q_int >> l) & 1
                        conv += pj * ql

                total = conv + carry_in
                if total % 2 == N_bits[pos]:
                    new_carry = total // 2
                    key = (p_int, q_int)
                    new_states[key] = new_carry
                    carry_counts[new_carry] += 1

        states = new_states
        carry_counts_by_pos.append(carry_counts)

        if not states:
            break

    return carry_counts_by_pos, len(states)


def entropy(counts):
    """Compute Shannon entropy of a distribution given by counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def log2_states(counts):
    """Log2 of total number of states."""
    total = sum(counts.values())
    return math.log2(total) if total > 0 else 0.0


# =========================================================================
# Main experiment
# =========================================================================

def experiment(bits_list=None, num_instances=20):
    if bits_list is None:
        bits_list = [12, 14, 16, 18, 20]

    print(f"{'='*75}")
    print(f"  TENSOR STRUCTURE OF MULTIPLICATION")
    print(f"  Entanglement entropy of the factoring constraint")
    print(f"{'='*75}")
    print(f"  At each position in the multiplication p × q = N,")
    print(f"  how many valid (p_prefix, q_prefix) states exist?")
    print(f"  And what is the carry entropy?")
    print()
    print(f"  If log2(#states) grows as O(log n): tensor methods work (poly time)")
    print(f"  If log2(#states) grows as O(n): exponential, no shortcut")

    for bits in bits_list:
        print(f"\n{'='*75}")
        print(f"  {bits}-BIT SEMIPRIMES")
        print(f"{'='*75}")

        k = bits // 2  # bits per factor
        if k > 10:
            print(f"  Skipping: k={k} too large for exact enumeration")
            continue

        all_entropies = []
        all_log_states = []

        t0 = time.time()
        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)
            N_bits = [(N >> i) & 1 for i in range(2 * k + 2)]

            result = measure_boundary_entropy(N_bits, k)
            if result is None:
                continue

            carry_counts, final_states = result

            # Compute entropy and state count at each position
            entropies = []
            log_states_list = []
            for pos_counts in carry_counts:
                e = entropy(pos_counts)
                ls = log2_states(pos_counts)
                entropies.append(e)
                log_states_list.append(ls)

            all_entropies.append(entropies)
            all_log_states.append(log_states_list)

            if inst < 3:
                print(f"\n  Instance {inst+1}: N = {N} = {p} × {q}")
                print(f"    Final valid states: {final_states}")
                print(f"    Position | carry_entropy | log2(#states) | #carry_vals")
                print(f"    ---------+---------------+---------------+------------")
                for pos in range(len(entropies)):
                    n_carry = len(carry_counts[pos])
                    total = sum(carry_counts[pos].values())
                    print(f"    {pos:8d} | {entropies[pos]:13.4f} | {log_states_list[pos]:13.4f} | {n_carry:10d}  (total={total})")

        elapsed = time.time() - t0
        print(f"\n  Computed in {elapsed:.1f}s")

        if not all_log_states:
            continue

        # Aggregate: average profile
        max_len = max(len(ls) for ls in all_log_states)
        avg_log_states = []
        avg_entropy = []
        for pos in range(max_len):
            vals = [ls[pos] for ls in all_log_states if pos < len(ls)]
            ent_vals = [e[pos] for e in all_entropies if pos < len(e)]
            if vals:
                avg_log_states.append(np.mean(vals))
                avg_entropy.append(np.mean(ent_vals))

        print(f"\n  Average profile across {len(all_log_states)} instances:")
        print(f"  Position | avg_carry_ent | avg_log2(#states)")
        print(f"  ---------+---------------+-----------------")
        for pos in range(len(avg_log_states)):
            print(f"  {pos:8d} | {avg_entropy[pos]:13.4f} | {avg_log_states[pos]:15.4f}")

        # Key metric: max log2(#states) — this is the "bond dimension"
        max_ls = max(avg_log_states)
        midpoint_ls = avg_log_states[k] if k < len(avg_log_states) else 0
        print(f"\n  Max log2(#states): {max_ls:.2f} (at position {avg_log_states.index(max_ls)})")
        print(f"  Midpoint log2(#states): {midpoint_ls:.2f} (position {k})")
        print(f"  k (bits per factor): {k}")
        print(f"  Ratio max/k: {max_ls/k:.3f}")

    # Scaling analysis
    print(f"\n{'='*75}")
    print(f"  SCALING: How does max log2(#states) grow with k?")
    print(f"{'='*75}")
    print(f"  If max ~ O(k) = O(n): bond dimension is exponential → no tensor shortcut")
    print(f"  If max ~ O(log k): bond dimension is polynomial → DMRG works!")
    print(f"  If max ~ O(√k): intermediate — sub-exponential tensor methods possible")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    experiment(bits_list=[10, 12, 14, 16, 18, 20], num_instances=15)
