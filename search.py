#!/usr/bin/env python3
"""
AI-guided algorithm search for factoring.

Instead of hand-crafting algorithms, we define a space of mathematical
operations and systematically search for compositions that factor integers.

The search space: sequences of operations on values mod n, where each
operation is drawn from a vocabulary of number-theoretic primitives.

A "candidate algorithm" is:
  1. Initialize some values (derived from n)
  2. Apply a sequence of operations
  3. Take gcd with n at each step

If gcd ever lands in (1, n), we found a factor.

The search strategy:
  - Start with known-working "seeds" (Pollard p-1, rho, etc.)
  - Mutate them: swap operations, change parameters, insert/delete steps
  - Crossover: combine parts of two working algorithms
  - Select: keep algorithms that factor more test cases or scale better
  - Novelty bonus: prefer algorithms that are structurally different from known ones
"""

import math
import random
import time
import json
from typing import Optional
from dataclasses import dataclass, field
from sympy import isprime

from harness import generate_semiprime, verify_factors


# =============================================================================
# Operation vocabulary — the building blocks
# =============================================================================

# Each operation takes (state, n) and returns new state.
# State is a dict of named integer values mod n.

OPERATIONS = {
    # Arithmetic
    "pow_k": lambda s, n, args: pow(s["a"], args.get("k", 2), n),
    "mul_ab": lambda s, n, args: (s["a"] * s["b"]) % n,
    "add_ab": lambda s, n, args: (s["a"] + s["b"]) % n,
    "sub_ab": lambda s, n, args: (s["a"] - s["b"]) % n,
    "sq": lambda s, n, args: (s["a"] * s["a"]) % n,
    "sq_plus_c": lambda s, n, args: (s["a"] * s["a"] + args.get("c", 1)) % n,
    "cube": lambda s, n, args: pow(s["a"], 3, n),
    "pow_n_derived": lambda s, n, args: pow(s["a"], _n_derived_exp(n, args), n),

    # Number-theoretic
    "gcd_a_n": lambda s, n, args: math.gcd(s["a"], n),
    "gcd_am1_n": lambda s, n, args: math.gcd(s["a"] - 1, n),
    "gcd_ap1_n": lambda s, n, args: math.gcd(s["a"] + 1, n),
    "jacobi": lambda s, n, args: _safe_jacobi(s["a"], n),
    "sqrt_mod": lambda s, n, args: pow(s["a"], (n + 1) // 4, n) if n % 4 == 3 else s["a"],

    # Sequence operations
    "lucas_step": lambda s, n, args: _lucas_step(s, n, args),
    "fib_step": lambda s, n, args: (s["b"], (s["a"] + s["b"]) % n),

    # Accumulation
    "mul_accum": lambda s, n, args: (s.get("acc", 1) * s["a"]) % n,
    "pow_chain": lambda s, n, args: pow(s["a"], args.get("k", 2), n),

    # Derived exponents — the key creative ingredient
    "pow_isqrt": lambda s, n, args: pow(s["a"], math.isqrt(n), n),
    "pow_bits": lambda s, n, args: pow(s["a"], n.bit_length(), n),
    "pow_bits_sq": lambda s, n, args: pow(s["a"], n.bit_length() ** 2, n),
    "pow_smooth": lambda s, n, args: _pow_smooth(s["a"], n, args.get("B", 20)),
}


def _n_derived_exp(n, args):
    """Compute an exponent derived from n."""
    mode = args.get("mode", 0)
    bits = n.bit_length()
    if mode == 0:
        return (n - 1) // 2
    elif mode == 1:
        return (n + 1) // 4
    elif mode == 2:
        return math.isqrt(n)
    elif mode == 3:
        return bits * bits
    elif mode == 4:
        return n % (1 << (bits // 2))
    return 2


def _safe_jacobi(a, n):
    if n <= 0 or n % 2 == 0:
        return a
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in [3, 5]:
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0


def _lucas_step(s, n, args):
    P = args.get("P", 3)
    v_prev = s.get("v_prev", 2)
    v_curr = s.get("a", P)
    new = (P * v_curr - v_prev) % n
    return new


def _pow_smooth(a, n, B):
    """Compute a^(lcm(1..B)) mod n."""
    result = a
    for p in _small_primes_up_to(B):
        pk = p
        while pk <= B:
            result = pow(result, p, n)
            pk *= p
    return result


def _small_primes_up_to(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(limit + 1) if sieve[i]]


# =============================================================================
# Candidate algorithm representation
# =============================================================================

@dataclass
class Candidate:
    """A candidate factoring algorithm as a sequence of operations."""
    # Initialization mode for state variables
    init_mode: str  # "random", "sqrt", "small", "n_derived"
    # Sequence of (operation_name, args_dict, target_var, source_vars)
    steps: list
    # How many times to repeat the main loop
    loop_count_mode: str  # "bits", "bits_sq", "bits_cube", "fixed"
    loop_count_param: int
    # When to check gcd
    gcd_check_interval: int
    # What to gcd: which variable minus what
    gcd_targets: list  # list of (var_name, offset) -> gcd(var - offset, n)

    # Metadata
    score: float = 0.0
    max_bits_solved: int = 0
    generation: int = 0
    parent_ids: list = field(default_factory=list)
    uid: str = ""

    def loop_count(self, n):
        bits = n.bit_length()
        if self.loop_count_mode == "bits":
            return bits * self.loop_count_param
        elif self.loop_count_mode == "bits_sq":
            return bits * bits * self.loop_count_param
        elif self.loop_count_mode == "bits_cube":
            return min(bits ** 3, 50000)
        else:
            return self.loop_count_param


def execute_candidate(candidate: Candidate, n: int) -> Optional[list[int]]:
    """Execute a candidate algorithm on input n."""
    if n < 4:
        return None
    if n % 2 == 0:
        return _factorize(n, 2)

    bits = n.bit_length()
    sqrt_n = math.isqrt(n)

    # Initialize state
    state = {}
    if candidate.init_mode == "random":
        state["a"] = random.randint(2, n - 1)
        state["b"] = random.randint(2, n - 1)
    elif candidate.init_mode == "sqrt":
        state["a"] = sqrt_n
        state["b"] = sqrt_n + 1
    elif candidate.init_mode == "small":
        state["a"] = 2
        state["b"] = 3
    elif candidate.init_mode == "n_derived":
        state["a"] = (n % (1 << (bits // 2))) or 2
        state["b"] = (n >> (bits // 2)) or 2
    else:
        state["a"] = random.randint(2, n - 1)
        state["b"] = random.randint(2, n - 1)

    state["acc"] = 1
    state["v_prev"] = 2
    state["prev_a"] = state["a"]

    loop_count = candidate.loop_count(n)
    loop_count = min(loop_count, 20000)  # hard cap — keeps eval fast

    for iteration in range(loop_count):
        # Execute steps
        for op_name, args, target, sources in candidate.steps:
            if op_name not in OPERATIONS:
                continue

            # Set up source mapping
            if sources:
                for src_name, state_var in sources:
                    if state_var in state:
                        state[src_name] = state[state_var]

            try:
                result = OPERATIONS[op_name](state, n, args)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

            if isinstance(result, tuple):
                # Some ops return multiple values (e.g., fib_step)
                if len(result) == 2:
                    state["a"], state["b"] = int(result[0]) % n, int(result[1]) % n
                continue

            result = int(result) % n
            state[target] = result

        # Check gcd periodically
        if iteration % candidate.gcd_check_interval == 0:
            for var_name, offset in candidate.gcd_targets:
                if var_name in state:
                    val = (state[var_name] - offset) % n
                    if val == 0:
                        continue
                    g = math.gcd(val, n)
                    if 1 < g < n:
                        return _factorize(n, g)

            # Also check accumulated product
            if "acc" in state and state["acc"] != 1:
                g = math.gcd(state["acc"], n)
                if 1 < g < n:
                    return _factorize(n, g)

        # Store previous value for rho-style cycle detection
        state["prev_a"] = state.get("a", 0)

    return None


def _factorize(n, factor):
    factors = []
    remaining = n
    for f in [factor, n // factor]:
        if f <= 1:
            continue
        if isprime(f):
            factors.append(f)
        else:
            # Simple recursive
            from baselines import pollard_rho
            sub = pollard_rho(f)
            if sub:
                factors.extend(sub)
            else:
                factors.append(f)
    # Verify multiplicities
    result = []
    temp = n
    for f in sorted(set(factors)):
        while temp % f == 0:
            result.append(f)
            temp //= f
    if temp > 1:
        result.append(temp)
    return sorted(result)


# =============================================================================
# Seed algorithms — encode known working methods as Candidates
# =============================================================================

def make_seeds():
    """Create seed candidates based on known algorithms."""
    seeds = []

    # Seed 1: Pollard p-1
    seeds.append(Candidate(
        init_mode="small",
        steps=[
            ("pow_smooth", {"B": 30}, "a", []),
        ],
        loop_count_mode="bits",
        loop_count_param=3,
        gcd_check_interval=1,
        gcd_targets=[("a", 1)],
        uid="seed_p_minus_1",
    ))

    # Seed 2: Pollard rho style
    seeds.append(Candidate(
        init_mode="random",
        steps=[
            ("sq_plus_c", {"c": 1}, "a", []),
            ("sq_plus_c", {"c": 1}, "b", []),
            ("sq_plus_c", {"c": 1}, "b", []),  # b advances twice
            ("sub_ab", {}, "diff", []),
        ],
        loop_count_mode="bits_sq",
        loop_count_param=2,
        gcd_check_interval=5,
        gcd_targets=[("diff", 0)],
        uid="seed_rho",
    ))

    # Seed 3: Fermat
    seeds.append(Candidate(
        init_mode="sqrt",
        steps=[
            ("add_ab", {}, "a", [("a", "a"), ("b", "one")]),
        ],
        loop_count_mode="bits_sq",
        loop_count_param=1,
        gcd_check_interval=1,
        gcd_targets=[("a", 0)],
        uid="seed_fermat",
    ))

    # Seed 4: Power GCD with n-derived exponent
    seeds.append(Candidate(
        init_mode="random",
        steps=[
            ("pow_n_derived", {"mode": 0}, "a", []),  # a^((n-1)/2)
        ],
        loop_count_mode="bits",
        loop_count_param=5,
        gcd_check_interval=1,
        gcd_targets=[("a", 1), ("a", -1)],
        uid="seed_euler",
    ))

    # Seed 5: Lucas sequence
    seeds.append(Candidate(
        init_mode="small",
        steps=[
            ("lucas_step", {"P": 3}, "a", []),
        ],
        loop_count_mode="bits_sq",
        loop_count_param=2,
        gcd_check_interval=10,
        gcd_targets=[("a", 2)],
        uid="seed_lucas",
    ))

    # Seed 6: Fibonacci walk
    seeds.append(Candidate(
        init_mode="small",
        steps=[
            ("fib_step", {}, "a", []),
        ],
        loop_count_mode="bits_sq",
        loop_count_param=2,
        gcd_check_interval=10,
        gcd_targets=[("a", 0), ("b", 0)],
        uid="seed_fib",
    ))

    # Seed 7: Multi-power chain
    seeds.append(Candidate(
        init_mode="random",
        steps=[
            ("pow_k", {"k": 2}, "a", []),
            ("mul_accum", {}, "acc", []),
        ],
        loop_count_mode="bits",
        loop_count_param=10,
        gcd_check_interval=20,
        gcd_targets=[("acc", 0), ("a", 1)],
        uid="seed_power_chain",
    ))

    # Seed 8: Smooth power with bits^2 bound
    seeds.append(Candidate(
        init_mode="small",
        steps=[
            ("pow_bits_sq", {}, "a", []),
        ],
        loop_count_mode="bits",
        loop_count_param=2,
        gcd_check_interval=1,
        gcd_targets=[("a", 1)],
        uid="seed_smooth_big",
    ))

    return seeds


# =============================================================================
# Mutation operators
# =============================================================================

def mutate(candidate: Candidate) -> Candidate:
    """Create a mutated copy of a candidate."""
    import copy
    c = copy.deepcopy(candidate)
    c.uid = f"mut_{random.randint(0, 999999):06d}"
    c.generation += 1
    c.parent_ids = [candidate.uid]
    c.score = 0
    c.max_bits_solved = 0

    mutation = random.choice([
        "change_init", "add_step", "remove_step", "swap_step",
        "change_args", "change_loop", "change_gcd_interval",
        "change_gcd_target", "duplicate_step", "change_op",
    ])

    op_names = list(OPERATIONS.keys())

    if mutation == "change_init":
        c.init_mode = random.choice(["random", "sqrt", "small", "n_derived"])

    elif mutation == "add_step":
        op = random.choice(op_names)
        args = _random_args(op)
        target = random.choice(["a", "b", "acc"])
        c.steps.insert(random.randint(0, len(c.steps)), (op, args, target, []))

    elif mutation == "remove_step" and len(c.steps) > 1:
        c.steps.pop(random.randint(0, len(c.steps) - 1))

    elif mutation == "swap_step" and len(c.steps) > 1:
        i, j = random.sample(range(len(c.steps)), 2)
        c.steps[i], c.steps[j] = c.steps[j], c.steps[i]

    elif mutation == "change_args" and c.steps:
        idx = random.randint(0, len(c.steps) - 1)
        op_name = c.steps[idx][0]
        c.steps[idx] = (op_name, _random_args(op_name), c.steps[idx][2], c.steps[idx][3])

    elif mutation == "change_loop":
        c.loop_count_mode = random.choice(["bits", "bits_sq", "bits_cube", "fixed"])
        c.loop_count_param = random.choice([1, 2, 3, 5, 10, 50])

    elif mutation == "change_gcd_interval":
        c.gcd_check_interval = random.choice([1, 2, 5, 10, 20, 50])

    elif mutation == "change_gcd_target":
        var = random.choice(["a", "b", "acc", "diff"])
        offset = random.choice([0, 1, -1, 2, -2])
        if random.random() < 0.5 and c.gcd_targets:
            c.gcd_targets[random.randint(0, len(c.gcd_targets) - 1)] = (var, offset)
        else:
            c.gcd_targets.append((var, offset))

    elif mutation == "duplicate_step" and c.steps:
        idx = random.randint(0, len(c.steps) - 1)
        c.steps.insert(idx + 1, c.steps[idx])

    elif mutation == "change_op" and c.steps:
        idx = random.randint(0, len(c.steps) - 1)
        new_op = random.choice(op_names)
        c.steps[idx] = (new_op, _random_args(new_op), c.steps[idx][2], c.steps[idx][3])

    # Cap step count
    if len(c.steps) > 10:
        c.steps = c.steps[:10]

    return c


def crossover(parent_a: Candidate, parent_b: Candidate) -> Candidate:
    """Combine two candidates."""
    import copy
    c = copy.deepcopy(parent_a)
    c.uid = f"cross_{random.randint(0, 999999):06d}"
    c.generation = max(parent_a.generation, parent_b.generation) + 1
    c.parent_ids = [parent_a.uid, parent_b.uid]
    c.score = 0
    c.max_bits_solved = 0

    # Mix steps
    if random.random() < 0.5:
        split = random.randint(0, len(c.steps))
        c.steps = c.steps[:split] + parent_b.steps[split:]
    else:
        # Interleave
        c.steps = []
        for i in range(max(len(parent_a.steps), len(parent_b.steps))):
            if i < len(parent_a.steps) and i < len(parent_b.steps):
                c.steps.append(random.choice([parent_a.steps[i], parent_b.steps[i]]))
            elif i < len(parent_a.steps):
                c.steps.append(parent_a.steps[i])
            else:
                c.steps.append(parent_b.steps[i])

    # Mix other params
    if random.random() < 0.5:
        c.init_mode = parent_b.init_mode
    if random.random() < 0.5:
        c.loop_count_mode = parent_b.loop_count_mode
        c.loop_count_param = parent_b.loop_count_param
    if random.random() < 0.5:
        c.gcd_targets = copy.deepcopy(parent_b.gcd_targets)

    if len(c.steps) > 10:
        c.steps = c.steps[:10]

    return c


def _random_args(op_name):
    """Generate random arguments for an operation."""
    if op_name in ("pow_k", "pow_chain"):
        return {"k": random.choice([2, 3, 5, 7, 11, 13])}
    elif op_name == "sq_plus_c":
        return {"c": random.randint(1, 10)}
    elif op_name == "pow_n_derived":
        return {"mode": random.randint(0, 4)}
    elif op_name == "lucas_step":
        return {"P": random.randint(2, 20)}
    elif op_name == "pow_smooth":
        return {"B": random.choice([10, 20, 30, 50, 100])}
    return {}


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_candidate(candidate: Candidate, test_suite: dict, timeout: float = 1.0) -> float:
    """
    Evaluate a candidate on the test suite.
    Relies on hard iteration cap in execute_candidate (no subprocess/signal overhead).
    """
    total_score = 0
    max_bits = 0

    for bits in sorted(test_suite.keys()):
        correct = 0
        total_time = 0

        for n, p, q in test_suite[bits]:
            start = time.time()
            try:
                factors = execute_candidate(candidate, n)
            except Exception:
                continue
            elapsed = time.time() - start

            if elapsed > timeout:
                continue  # too slow, skip

            if factors and verify_factors(n, factors):
                correct += 1
                total_time += elapsed
                max_bits = bits

        if correct > 0:
            total_score += correct * (bits ** 2)
            avg_time = total_time / correct
            if avg_time < 0.1:
                total_score += bits * 10
            elif avg_time < 1.0:
                total_score += bits * 5

    candidate.score = total_score
    candidate.max_bits_solved = max_bits
    return total_score


# =============================================================================
# Main evolution loop
# =============================================================================

def run_search(generations: int = 50, population_size: int = 30,
               elite_count: int = 5, test_bits: list = None):
    """Run the evolutionary search."""

    if test_bits is None:
        test_bits = [16, 24, 32, 40, 48, 56]

    # Generate test suite (fixed for consistency across generations)
    print("Generating test suite...")
    test_suite = {}
    for bits in test_bits:
        test_suite[bits] = [generate_semiprime(bits) for _ in range(3)]

    # Initialize population with seeds
    population = make_seeds()

    # Fill rest with mutations of seeds
    while len(population) < population_size:
        parent = random.choice(population[:len(make_seeds())])
        population.append(mutate(parent))

    print(f"Population: {len(population)} candidates")
    print(f"Test bits: {test_bits}")
    print(f"Generations: {generations}")
    print()

    best_ever = None
    best_ever_score = 0

    for gen in range(generations):
        gen_start = time.time()

        # Evaluate
        for candidate in population:
            if candidate.score == 0:  # only evaluate new candidates
                evaluate_candidate(candidate, test_suite, timeout=1.0)

        # Sort by score
        population.sort(key=lambda c: c.score, reverse=True)

        # Track best
        if population[0].score > best_ever_score:
            best_ever = population[0]
            best_ever_score = population[0].score

        gen_time = time.time() - gen_start
        top = population[0]
        print(f"Gen {gen:3d} | best: {top.score:8.0f} (max {top.max_bits_solved} bits) "
              f"| uid: {top.uid:20s} | {gen_time:.1f}s"
              f"| pop avg: {sum(c.score for c in population)/len(population):.0f}")

        # Print top 3
        if gen % 10 == 0:
            print(f"  Top candidates:")
            for i, c in enumerate(population[:5]):
                print(f"    {i+1}. {c.uid:25s} score={c.score:8.0f} "
                      f"max_bits={c.max_bits_solved:3d} "
                      f"steps={len(c.steps)} gen={c.generation}")
            print()

        # Selection + reproduction
        # Keep elite
        new_pop = population[:elite_count]

        # Generate children
        while len(new_pop) < population_size:
            r = random.random()
            if r < 0.4:
                # Mutation of a top candidate
                parent = population[random.randint(0, min(10, len(population) - 1))]
                child = mutate(parent)
                # Sometimes double-mutate
                if random.random() < 0.3:
                    child = mutate(child)
                new_pop.append(child)
            elif r < 0.7:
                # Crossover
                p1 = population[random.randint(0, min(10, len(population) - 1))]
                p2 = population[random.randint(0, min(15, len(population) - 1))]
                new_pop.append(crossover(p1, p2))
            else:
                # Fresh random from seeds
                seed = random.choice(make_seeds())
                child = mutate(mutate(seed))
                new_pop.append(child)

        population = new_pop

    # Final report
    print("\n" + "=" * 70)
    print("  SEARCH COMPLETE")
    print("=" * 70)

    # Re-evaluate top candidates on a harder test suite
    hard_bits = [32, 48, 64, 80, 96]
    hard_suite = {}
    for bits in hard_bits:
        hard_suite[bits] = [generate_semiprime(bits) for _ in range(3)]

    print("\nRe-evaluating top 10 on harder test suite...")
    finalists = population[:10]
    for c in finalists:
        c.score = 0
        evaluate_candidate(c, hard_suite, timeout=5.0)

    finalists.sort(key=lambda c: c.score, reverse=True)
    print(f"\n{'uid':>25s} | {'score':>8s} | {'max_bits':>8s} | steps")
    print("-" * 70)
    for c in finalists:
        print(f"{c.uid:>25s} | {c.score:8.0f} | {c.max_bits_solved:>8d} | {_describe_steps(c)}")

    # Print the best algorithm in detail
    if finalists[0].score > 0:
        print(f"\nBest algorithm detail:")
        _print_candidate(finalists[0])

    return finalists


def _describe_steps(c):
    return " -> ".join(s[0] for s in c.steps[:5])


def _print_candidate(c):
    print(f"  UID: {c.uid}")
    print(f"  Init: {c.init_mode}")
    print(f"  Loop: {c.loop_count_mode} × {c.loop_count_param}")
    print(f"  GCD check every {c.gcd_check_interval} iterations")
    print(f"  GCD targets: {c.gcd_targets}")
    print(f"  Steps:")
    for i, (op, args, target, sources) in enumerate(c.steps):
        print(f"    {i+1}. {op}({args}) -> {target}")
    print(f"  Score: {c.score}, Max bits: {c.max_bits_solved}")


if __name__ == "__main__":
    import sys
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_search(generations=gens, population_size=30)
