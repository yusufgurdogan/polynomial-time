#!/usr/bin/env python3
"""
Algorithm search v2: higher-level strategy space.

Instead of evolving sequences of arithmetic ops, we evolve PARAMETERIZED
STRATEGIES — each strategy is a known mathematical idea with tunable knobs.

The evolution tunes the knobs AND discovers new combinations of strategies.
"""

import math
import random
import time
import itertools
from typing import Optional
from dataclasses import dataclass, field
from sympy import isprime
from harness import generate_semiprime, verify_factors


# =============================================================================
# Strategy primitives — each is a parameterized factoring sub-routine
# =============================================================================

def strat_power_gcd(n: int, base: int, exponent: int) -> Optional[int]:
    """Compute gcd(base^exponent - 1, n)."""
    val = pow(base, exponent, n)
    g = math.gcd(val - 1, n)
    if 1 < g < n:
        return g
    g = math.gcd(val + 1, n)
    if 1 < g < n:
        return g
    return None


def strat_smooth_power(n: int, base: int, B: int) -> Optional[int]:
    """Compute base^(lcm(1..B)) mod n, check gcd."""
    B = min(B, 5000)
    power = base % n
    for p in _get_primes(B):
        pk = p
        while pk <= B:
            power = pow(power, p, n)
            pk *= p
    g = math.gcd(power - 1, n)
    if 1 < g < n:
        return g
    return None


def strat_lucas(n: int, P: int, steps: int) -> Optional[int]:
    """Lucas sequence V_k(P, 1) mod n, check gcd(V_k - 2, n)."""
    v_prev, v_curr = 2, P % n
    for k in range(2, steps + 2):
        v_prev, v_curr = v_curr, (P * v_curr - v_prev) % n
        if k % max(steps // 20, 1) == 0:
            g = math.gcd(v_curr - 2, n)
            if 1 < g < n:
                return g
    return None


def strat_rho(n: int, c: int, degree: int, steps: int) -> Optional[int]:
    """Pollard rho with x -> x^degree + c."""
    x = random.randint(2, n - 1)
    y = x
    for _ in range(steps):
        x = (pow(x, degree, n) + c) % n
        y = (pow(y, degree, n) + c) % n
        y = (pow(y, degree, n) + c) % n
        g = math.gcd(abs(x - y), n)
        if 1 < g < n:
            return g
        if g == n:
            return None
    return None


def strat_fermat(n: int, offset: int, steps: int) -> Optional[int]:
    """Fermat's method starting from isqrt(n) + offset."""
    a = math.isqrt(n) + offset
    for _ in range(steps):
        b2 = a * a - n
        if b2 >= 0:
            b = math.isqrt(b2)
            if b * b == b2:
                p = a - b
                if 1 < p < n and n % p == 0:
                    return p
        a += 1
    return None


def strat_fibonacci(n: int, steps: int) -> Optional[int]:
    """Fibonacci sequence mod n, check gcd."""
    a, b = 1, 1
    for k in range(steps):
        a, b = b, (a + b) % n
        if k % max(steps // 20, 1) == 0:
            g = math.gcd(a, n)
            if 1 < g < n:
                return g
    return None


def strat_euler_split(n: int, base: int) -> Optional[int]:
    """Use Euler's criterion: a^((n-1)/2) mod n. If n=pq this might split."""
    if n % 2 == 0:
        return None
    val = pow(base, (n - 1) // 2, n)
    for v in [val - 1, val + 1, val]:
        g = math.gcd(v % n, n)
        if 1 < g < n:
            return g
    # Try quarter
    if (n - 1) % 4 == 0:
        val = pow(base, (n - 1) // 4, n)
        for v in [val - 1, val + 1]:
            g = math.gcd(v % n, n)
            if 1 < g < n:
                return g
    return None


def strat_jacobi_scan(n: int, limit: int) -> Optional[int]:
    """Scan small values, use Jacobi symbol to find where characters split."""
    for a in range(2, limit):
        j = _jacobi(a, n)
        if j == 0:
            g = math.gcd(a, n)
            if 1 < g < n:
                return g
        elif j == -1:
            val = pow(a, (n - 1) // 2, n)
            g = math.gcd(val + 1, n)
            if 1 < g < n:
                return g
    return None


def strat_power_diff(n: int, base1: int, base2: int, exp: int) -> Optional[int]:
    """Compute gcd(base1^exp - base2^exp, n)."""
    v1 = pow(base1, exp, n)
    v2 = pow(base2, exp, n)
    g = math.gcd((v1 - v2) % n, n)
    if 1 < g < n:
        return g
    return None


def strat_multi_smooth(n: int, num_bases: int, B: int) -> Optional[int]:
    """Multiple bases with smooth power, accumulate gcds."""
    acc = 1
    for _ in range(num_bases):
        base = random.randint(2, n - 1)
        power = base
        p = 2
        while p <= B:
            pk = p
            while pk <= B:
                power = pow(power, p, n)
                pk *= p
            p = _next_prime(p)
        acc = (acc * (power - 1)) % n
        if acc == 0:
            continue
        g = math.gcd(acc, n)
        if 1 < g < n:
            return g
    return None


def strat_lehman(n: int) -> Optional[int]:
    """Lehman's method — O(n^(1/3)) but capped for speed."""
    cbrt = min(int(round(n ** (1/3))), 5000)
    for d in range(2, cbrt + 1):
        if n % d == 0:
            return d
    for k in range(1, min(cbrt + 1, 2000)):
        sqrt_4kn = math.isqrt(4 * k * n)
        limit = sqrt_4kn + min(int(cbrt / (4 * math.sqrt(k))) + 2, 50)
        for a in range(sqrt_4kn, limit):
            b2 = a * a - 4 * k * n
            if b2 >= 0:
                b = math.isqrt(b2)
                if b * b == b2:
                    g = math.gcd(a + b, n)
                    if 1 < g < n:
                        return g
    return None


# =============================================================================
# Algorithm = ordered list of strategy calls with parameters
# =============================================================================

# Each strategy spec: (name, param_generator, executor)
STRATEGY_CATALOG = {
    "smooth_power": {
        "params": lambda bits: {"B": random.choice([bits, bits*2, bits*3, bits*5, bits**2, bits**2*2])},
        "exec": lambda n, p: strat_smooth_power(n, random.randint(2, min(n-1, 100)), p["B"]),
    },
    "lucas": {
        "params": lambda bits: {"P": random.randint(2, 30), "steps": random.choice([bits**2, bits**3, bits*bits*2])},
        "exec": lambda n, p: strat_lucas(n, p["P"], min(p["steps"], 20000)),
    },
    "rho_quad": {
        "params": lambda bits: {"c": random.randint(1, 10), "steps": bits**2 * 2},
        "exec": lambda n, p: strat_rho(n, p["c"], 2, min(p["steps"], 20000)),
    },
    "rho_cubic": {
        "params": lambda bits: {"c": random.randint(1, 5), "steps": bits**2 * 2},
        "exec": lambda n, p: strat_rho(n, p["c"], 3, min(p["steps"], 20000)),
    },
    "fermat": {
        "params": lambda bits: {"offset": random.randint(0, bits*10), "steps": bits**2},
        "exec": lambda n, p: strat_fermat(n, p["offset"], min(p["steps"], 20000)),
    },
    "fibonacci": {
        "params": lambda bits: {"steps": bits**2 * 3},
        "exec": lambda n, p: strat_fibonacci(n, min(p["steps"], 20000)),
    },
    "euler_split": {
        "params": lambda bits: {"num_bases": random.choice([bits, bits*2, bits*5])},
        "exec": lambda n, p: _multi_euler(n, p["num_bases"]),
    },
    "jacobi_scan": {
        "params": lambda bits: {"limit": random.choice([bits*10, bits*50, bits**2])},
        "exec": lambda n, p: strat_jacobi_scan(n, min(p["limit"], 10000)),
    },
    "power_diff": {
        "params": lambda bits: {"exp_mode": random.choice(["n_half", "isqrt", "bits_sq", "random_smooth"])},
        "exec": lambda n, p: _power_diff_dispatch(n, p["exp_mode"]),
    },
    "multi_smooth": {
        "params": lambda bits: {"num_bases": random.choice([bits, bits*2]), "B": random.choice([bits*2, bits*3, bits**2])},
        "exec": lambda n, p: strat_multi_smooth(n, min(p["num_bases"], 200), min(p["B"], 5000)),
    },
    "lehman": {
        "params": lambda bits: {},
        "exec": lambda n, p: strat_lehman(n),
    },
}


def _multi_euler(n, num_bases):
    for _ in range(min(num_bases, 500)):
        base = random.randint(2, n - 1)
        r = strat_euler_split(n, base)
        if r:
            return r
    return None


def _power_diff_dispatch(n, mode):
    bits = n.bit_length()
    b1 = random.randint(2, min(n-1, 100))
    b2 = random.randint(2, min(n-1, 100))
    if mode == "n_half":
        exp = (n - 1) // 2
    elif mode == "isqrt":
        exp = math.isqrt(n)
    elif mode == "bits_sq":
        exp = bits * bits
    else:
        exp = 1
        for p in [2, 3, 5, 7, 11, 13]:
            exp *= p ** (bits // (p * 3) + 1)
    return strat_power_diff(n, b1, b2, exp)


@dataclass
class Algorithm:
    """An algorithm is an ordered list of strategies with parameters."""
    strategies: list  # [(strategy_name, frozen_params)]
    score: float = 0.0
    max_bits: int = 0
    uid: str = ""
    generation: int = 0

    def execute(self, n: int) -> Optional[list[int]]:
        if n < 4:
            return None
        if n % 2 == 0:
            return _make_factors(n, 2)

        for strat_name, params in self.strategies:
            if strat_name not in STRATEGY_CATALOG:
                continue
            try:
                factor = STRATEGY_CATALOG[strat_name]["exec"](n, params)
            except Exception:
                continue
            if factor is not None and 1 < factor < n:
                return _make_factors(n, factor)
        return None


def _make_factors(n, f):
    factors = []
    for part in [f, n // f]:
        if part <= 1:
            continue
        if isprime(part):
            factors.append(part)
        else:
            from baselines import pollard_rho
            sub = pollard_rho(part)
            if sub:
                factors.extend(sub)
            else:
                factors.append(part)
    result = []
    temp = n
    for f2 in sorted(set(factors)):
        while temp % f2 == 0:
            result.append(f2)
            temp //= f2
    if temp > 1:
        result.append(temp)
    return sorted(result)


# =============================================================================
# Evolution
# =============================================================================

def random_algorithm(bits_hint: int = 32) -> Algorithm:
    """Generate a random algorithm."""
    num_strats = random.randint(1, 5)
    strategies = []
    for _ in range(num_strats):
        name = random.choice(list(STRATEGY_CATALOG.keys()))
        params = STRATEGY_CATALOG[name]["params"](bits_hint)
        strategies.append((name, params))
    return Algorithm(
        strategies=strategies,
        uid=f"rnd_{random.randint(0,999999):06d}",
    )


def mutate_algo(algo: Algorithm) -> Algorithm:
    import copy
    a = copy.deepcopy(algo)
    a.uid = f"mut_{random.randint(0,999999):06d}"
    a.generation += 1
    a.score = 0
    a.max_bits = 0

    bits_hint = max(a.max_bits, 32)
    op = random.choice(["add", "remove", "swap", "replace", "reparametrize"])

    if op == "add":
        name = random.choice(list(STRATEGY_CATALOG.keys()))
        params = STRATEGY_CATALOG[name]["params"](bits_hint)
        pos = random.randint(0, len(a.strategies))
        a.strategies.insert(pos, (name, params))
    elif op == "remove" and len(a.strategies) > 1:
        a.strategies.pop(random.randint(0, len(a.strategies) - 1))
    elif op == "swap" and len(a.strategies) > 1:
        i, j = random.sample(range(len(a.strategies)), 2)
        a.strategies[i], a.strategies[j] = a.strategies[j], a.strategies[i]
    elif op == "replace":
        idx = random.randint(0, len(a.strategies) - 1)
        name = random.choice(list(STRATEGY_CATALOG.keys()))
        params = STRATEGY_CATALOG[name]["params"](bits_hint)
        a.strategies[idx] = (name, params)
    elif op == "reparametrize":
        idx = random.randint(0, len(a.strategies) - 1)
        name = a.strategies[idx][0]
        params = STRATEGY_CATALOG[name]["params"](bits_hint)
        a.strategies[idx] = (name, params)

    if len(a.strategies) > 8:
        a.strategies = a.strategies[:8]
    return a


def crossover_algo(a1: Algorithm, a2: Algorithm) -> Algorithm:
    import copy
    # Take front half of a1, back half of a2
    split1 = random.randint(0, len(a1.strategies))
    split2 = random.randint(0, len(a2.strategies))
    strats = a1.strategies[:split1] + a2.strategies[split2:]
    if not strats:
        strats = [random.choice(a1.strategies + a2.strategies)]
    return Algorithm(
        strategies=strats[:8],
        uid=f"xov_{random.randint(0,999999):06d}",
        generation=max(a1.generation, a2.generation) + 1,
    )


def evaluate(algo: Algorithm, test_suite: dict, timeout_per: float = 1.0):
    total = 0
    max_bits = 0
    for bits in sorted(test_suite.keys()):
        correct = 0
        total_time = 0
        for n, p, q in test_suite[bits]:
            t0 = time.time()
            try:
                factors = algo.execute(n)
            except Exception:
                continue
            elapsed = time.time() - t0
            if elapsed > timeout_per:
                continue
            if factors and verify_factors(n, factors):
                correct += 1
                total_time += elapsed
                max_bits = bits
        if correct > 0:
            total += correct * (bits ** 2)
            avg_t = total_time / correct
            if avg_t < 0.01:
                total += bits * 20
            elif avg_t < 0.1:
                total += bits * 10
            elif avg_t < 0.5:
                total += bits * 5
    algo.score = total
    algo.max_bits = max_bits


# =============================================================================
# Seeded algorithms
# =============================================================================

def make_seeds():
    seeds = []

    # Pure p-1 with various bounds
    for B in [30, 50, 100, 200]:
        seeds.append(Algorithm(
            strategies=[("smooth_power", {"B": B})],
            uid=f"seed_pm1_{B}",
        ))

    # Pure rho
    seeds.append(Algorithm(
        strategies=[("rho_quad", {"c": 1, "steps": 10000})],
        uid="seed_rho",
    ))

    # Lehman
    seeds.append(Algorithm(
        strategies=[("lehman", {})],
        uid="seed_lehman",
    ))

    # Combined: lehman -> p-1 -> rho
    seeds.append(Algorithm(
        strategies=[
            ("lehman", {}),
            ("smooth_power", {"B": 100}),
            ("rho_quad", {"c": 1, "steps": 10000}),
        ],
        uid="seed_combined",
    ))

    # Euler split + p-1
    seeds.append(Algorithm(
        strategies=[
            ("euler_split", {"num_bases": 50}),
            ("smooth_power", {"B": 100}),
        ],
        uid="seed_euler_pm1",
    ))

    # Lucas + fibonacci + rho
    seeds.append(Algorithm(
        strategies=[
            ("lucas", {"P": 3, "steps": 5000}),
            ("fibonacci", {"steps": 5000}),
            ("rho_cubic", {"c": 1, "steps": 5000}),
        ],
        uid="seed_sequence_mix",
    ))

    # Multi-smooth
    seeds.append(Algorithm(
        strategies=[("multi_smooth", {"num_bases": 50, "B": 200})],
        uid="seed_multi_smooth",
    ))

    return seeds


# =============================================================================
# Main
# =============================================================================

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    pop_size = 40

    print("=" * 70)
    print("  ALGORITHM SEARCH v2 — Strategy-level evolution")
    print("=" * 70)

    test_bits = [20, 28, 36, 44, 52, 60]
    samples = 3

    print("Generating test suite...")
    test_suite = {}
    for bits in test_bits:
        test_suite[bits] = [generate_semiprime(bits) for _ in range(samples)]

    # Harder suite for final eval
    hard_bits = [32, 48, 64, 80, 96, 112]
    hard_suite = {}
    for bits in hard_bits:
        hard_suite[bits] = [generate_semiprime(bits) for _ in range(samples)]

    # Init population
    pop = make_seeds()
    while len(pop) < pop_size:
        if random.random() < 0.5:
            pop.append(mutate_algo(random.choice(pop)))
        else:
            pop.append(random_algorithm())

    best_ever_score = 0
    best_ever = None
    stale_count = 0

    for gen in range(gens):
        t0 = time.time()
        for a in pop:
            if a.score == 0:
                evaluate(a, test_suite)
        pop.sort(key=lambda a: a.score, reverse=True)

        if pop[0].score > best_ever_score:
            best_ever_score = pop[0].score
            best_ever = pop[0]
            stale_count = 0
        else:
            stale_count += 1

        elapsed = time.time() - t0
        top = pop[0]
        strats_str = "+".join(s[0][:8] for s in top.strategies[:4])
        print(f"Gen {gen:3d} | best={top.score:7.0f} max={top.max_bits:2d}b "
              f"| {strats_str:40s} | {elapsed:5.1f}s | stale={stale_count}")

        if gen % 15 == 0 and gen > 0:
            print(f"\n  Top 5:")
            for i, a in enumerate(pop[:5]):
                strs = [f"{s[0]}({s[1]})" for s in a.strategies[:3]]
                print(f"    {i+1}. [{a.uid}] score={a.score:.0f} max={a.max_bits}b: {' | '.join(strs)}")
            print()

        # If stale too long, inject diversity
        if stale_count > 10:
            for _ in range(pop_size // 4):
                pop.append(random_algorithm())
            stale_count = 0

        # Next generation
        elite = pop[:5]
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            r = random.random()
            if r < 0.35:
                new_pop.append(mutate_algo(pop[random.randint(0, min(9, len(pop)-1))]))
            elif r < 0.6:
                p1 = pop[random.randint(0, min(9, len(pop)-1))]
                p2 = pop[random.randint(0, min(14, len(pop)-1))]
                new_pop.append(crossover_algo(p1, p2))
            elif r < 0.8:
                new_pop.append(mutate_algo(mutate_algo(random.choice(elite))))
            else:
                new_pop.append(random_algorithm())
        pop = new_pop

    # Final evaluation on hard suite
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION on harder suite")
    print("=" * 70)

    finalists = pop[:15]
    for a in finalists:
        a.score = 0
        evaluate(a, hard_suite, timeout_per=5.0)
    finalists.sort(key=lambda a: a.score, reverse=True)

    print(f"\n{'rank':>4} | {'uid':>20} | {'score':>7} | {'max':>4} | strategies")
    print("-" * 80)
    for i, a in enumerate(finalists[:10]):
        strats = " -> ".join(f"{s[0]}({_compact(s[1])})" for s in a.strategies)
        print(f"{i+1:4d} | {a.uid:>20} | {a.score:7.0f} | {a.max_bits:4d} | {strats}")

    if finalists[0].score > 0:
        print(f"\n  CHAMPION: {finalists[0].uid}")
        print(f"  Score: {finalists[0].score}, Max bits factored: {finalists[0].max_bits}")
        print(f"  Strategies:")
        for i, (name, params) in enumerate(finalists[0].strategies):
            print(f"    {i+1}. {name}: {params}")


def _compact(d):
    return ",".join(f"{k}={v}" for k,v in d.items()) if d else ""


_PRIMES_CACHE = None
def _get_primes(limit=5000):
    global _PRIMES_CACHE
    if _PRIMES_CACHE is None or _PRIMES_CACHE[-1] < limit:
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5)+1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        _PRIMES_CACHE = [i for i in range(limit+1) if sieve[i]]
    return _PRIMES_CACHE

def _next_prime(n):
    primes = _get_primes(max(n*2+100, 5000))
    for p in primes:
        if p > n:
            return p
    return n + 1


def _jacobi(a, n):
    if n <= 0 or n % 2 == 0:
        return 0
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


if __name__ == "__main__":
    main()
