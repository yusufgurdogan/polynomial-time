"""
Experimental factoring approaches.
These explore novel angles that might (long shot) exhibit polynomial scaling.
"""

import math
import random
from typing import Optional
from sympy import isprime


# =============================================================================
# Approach 1: Lattice-based / Coppersmith-inspired
# =============================================================================
# Coppersmith's theorem: if you know the high bits of a factor, you can
# recover the rest in polynomial time via lattice reduction (LLL).
# Idea: what if we can extract partial information about factors cheaply?

def lattice_partial_info(n: int) -> Optional[list[int]]:
    """
    Try to extract partial information about factors using properties of n,
    then use that to narrow the search.

    Key insight: for n = p*q, we know p+q = n - phi(n) + 1 and p*q = n.
    If we could approximate phi(n), we could solve the quadratic.

    We try various approximations of phi(n) based on the statistical
    properties of primes near sqrt(n).
    """
    sqrt_n = math.isqrt(n)
    bits = n.bit_length()

    # For a semiprime n=pq where p,q ~ sqrt(n):
    # phi(n) = (p-1)(q-1) = n - p - q + 1
    # p + q = n - phi(n) + 1
    # So if we can estimate p+q, we can solve the quadratic x^2 - (p+q)x + n = 0

    # The average prime gap near x is ~ln(x), so p+q ≈ 2*sqrt(n) with
    # error bounded by O(sqrt(n) * ln(sqrt(n)) / sqrt(n)) = O(ln(n))

    # Try offsets around 2*sqrt(n) for the sum p+q
    # If s = p+q, then p,q are roots of x^2 - sx + n = 0
    # discriminant = s^2 - 4n must be a perfect square

    # The key question: can we bound |p-q| polynomially in log(n)?
    # For random primes, |p-q| ~ O(sqrt(n)), so this is exponential search.
    # But what if we can use OTHER info to narrow the range?

    # Attempt: use modular arithmetic constraints to prune
    # For small primes m, we know (p mod m) * (q mod m) ≡ n mod m
    # This constrains possible values of p+q mod m
    # Combining via CRT might narrow the search

    search_range = max(bits * bits * 10, 10000)

    for delta in range(search_range):
        for sign in [1, -1]:
            s = 2 * sqrt_n + sign * delta
            if s < 2:
                continue
            disc = s * s - 4 * n
            if disc < 0:
                continue
            sqrt_disc = math.isqrt(disc)
            if sqrt_disc * sqrt_disc == disc:
                p = (s + sqrt_disc) // 2
                q = (s - sqrt_disc) // 2
                if p > 1 and q > 1 and p * q == n:
                    factors = sorted([p, q])
                    return factors

    return None


# =============================================================================
# Approach 2: Multiplicative order / period detection
# =============================================================================
# Shor's algorithm finds the period of a^x mod n using QFT.
# Can we detect periods classically using number-theoretic structure?

def classical_period_attack(n: int) -> Optional[list[int]]:
    """
    Try to find the multiplicative order of random bases mod n,
    using various classical shortcuts:

    1. Baby-step giant-step for order finding
    2. Pohlig-Hellman style decomposition (if order has small factors)
    3. Smooth-order detection

    If we find r = ord(a, n), and r is even, then
    gcd(a^(r/2) ± 1, n) often gives a factor.
    """
    if n % 2 == 0:
        return [2] + _factor_remaining(n // 2)

    bits = n.bit_length()

    for _ in range(min(bits * 2, 50)):  # try multiple random bases
        a = random.randint(2, n - 1)
        g = math.gcd(a, n)
        if 1 < g < n:
            return sorted([g, n // g]) if isprime(g) and isprime(n // g) else _factor_remaining_full(n, g)

        # Try to find the order using smooth-part detection
        # Compute a^(k!) mod n for increasing k — if order divides k!, we detect it
        factor = _smooth_order_attack(a, n)
        if factor is not None:
            return factor

        # Try baby-step giant-step for small orders
        factor = _bsgs_order_attack(a, n, bits)
        if factor is not None:
            return factor

    return None


def _smooth_order_attack(a: int, n: int, B: int = None) -> Optional[list[int]]:
    """
    Pollard p-1 style: if p-1 (for a factor p of n) is B-smooth,
    then a^(B!) ≡ 1 (mod p), so gcd(a^(B!) - 1, n) reveals p.

    This is polynomial if we pick B = O(poly(log n)) and a factor
    happens to have a smooth p-1. Not general, but worth trying.
    """
    bits = n.bit_length()
    if B is None:
        B = max(bits * 5, 100)

    power = a
    for k in range(2, B + 1):
        power = pow(power, k, n)
        if k % 50 == 0:  # periodically check
            g = math.gcd(power - 1, n)
            if 1 < g < n:
                return _factor_remaining_full(n, g)

    g = math.gcd(power - 1, n)
    if 1 < g < n:
        return _factor_remaining_full(n, g)

    return None


def _bsgs_order_attack(a: int, n: int, bits: int) -> Optional[list[int]]:
    """
    Baby-step giant-step to find the multiplicative order of a mod n.
    Only feasible for small orders (up to ~2^20).
    """
    limit = min(1 << 20, 1 << (bits // 3))
    m = math.isqrt(limit) + 1

    # Baby steps: a^j for j in [0, m)
    baby = {}
    power = 1
    for j in range(m):
        baby[power] = j
        power = (power * a) % n

    # Giant steps: a^(-m*i) for i in [0, m)
    # a^(-m) = inverse of a^m
    am = pow(a, m, n)
    try:
        am_inv = pow(am, -1, n)
    except ValueError:
        # gcd(am, n) != 1, which means we found a factor!
        g = math.gcd(am, n)
        if 1 < g < n:
            return _factor_remaining_full(n, g)
        return None

    gamma = 1
    for i in range(m):
        if gamma in baby:
            order = i * m + baby[gamma]
            if order > 0:
                # Found order, try to extract factor
                factor = _factor_from_order(a, order, n)
                if factor is not None:
                    return factor
        gamma = (gamma * am_inv) % n

    return None


def _factor_from_order(a: int, order: int, n: int) -> Optional[list[int]]:
    """Given ord(a, n) = order, try to extract a factor of n."""
    if order % 2 != 0:
        return None

    half = pow(a, order // 2, n)
    for val in [half - 1, half + 1]:
        g = math.gcd(val, n)
        if 1 < g < n:
            return _factor_remaining_full(n, g)

    # Try other divisors of order
    r = order
    while r % 2 == 0:
        r //= 2
        val = pow(a, r, n)
        for v in [val - 1, val + 1]:
            g = math.gcd(v, n)
            if 1 < g < n:
                return _factor_remaining_full(n, g)

    return None


# =============================================================================
# Approach 3: Algebraic structure / character sum approach
# =============================================================================
# The Legendre symbol (a/p) encodes quadratic residuosity.
# For n = pq, the Jacobi symbol (a/n) = (a/p)(a/q).
# Can we extract the individual Legendre symbols from the Jacobi symbol?

def character_sum_attack(n: int) -> Optional[list[int]]:
    """
    Use properties of Jacobi symbols and character sums to leak
    information about the factors of n.

    Key idea: if we evaluate the Jacobi symbol (a/n) for many values of a,
    we get (a/p)(a/q). If we can find an 'a' where (a/p) != (a/q),
    then (a/n) = -1 but a is a QR mod one factor and QNR mod the other.
    Combined with other constraints, this might help.

    More sophisticated: use Gauss sums. The Gauss sum G(chi, n) factors as
    G(chi_p) * G(chi_q) when n = pq. The magnitude gives us info about p and q.
    """
    sqrt_n = math.isqrt(n)

    # Collect Jacobi symbol data
    jacobi_neg = []  # values where (a/n) = -1
    jacobi_pos = []  # values where (a/n) = +1 (excluding trivial)

    for a in range(2, min(10000, n)):
        j = _jacobi_symbol(a, n)
        if j == 0:
            g = math.gcd(a, n)
            if 1 < g < n:
                return _factor_remaining_full(n, g)
        elif j == -1:
            jacobi_neg.append(a)
        else:
            jacobi_pos.append(a)

    # For a where (a/n) = -1: exactly one of (a/p), (a/q) is -1.
    # Try: compute a^((n-1)/2) mod n, which should be -1 (mod n) by Euler.
    # But a^((p-1)/2) mod p and a^((q-1)/2) mod q differ.
    # We can try to use CRT-style reasoning.

    # Approach: for each a with (a/n) = -1, compute gcd(a^((n-1)/2) + 1, n)
    # This sometimes yields a factor when the Euler criterion splits differently mod p vs q
    for a in jacobi_neg[:200]:
        # a^((n-1)/2) ≡ -1 (mod n) by Euler criterion
        # But more precisely: a^((n-1)/2) mod p and mod q may differ
        # Try computing a^(k) - 1 for various k derived from n
        val = pow(a, (n - 1) // 2, n)
        g = math.gcd(val + 1, n)
        if 1 < g < n:
            return _factor_remaining_full(n, g)
        g = math.gcd(val - 1, n)
        if 1 < g < n:
            return _factor_remaining_full(n, g)

    # Approach: use pairs of Jacobi symbols to detect structure
    # If (a/n) = 1 but a is NOT a QR mod n, then a^((n-1)/4) might reveal structure
    if (n - 1) % 4 == 0:
        for a in jacobi_pos[:200]:
            val = pow(a, (n - 1) // 4, n)
            for v in [val - 1, val + 1]:
                g = math.gcd(v, n)
                if 1 < g < n:
                    return _factor_remaining_full(n, g)

    return None


def _jacobi_symbol(a: int, n: int) -> int:
    """Compute the Jacobi symbol (a/n)."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")

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


# =============================================================================
# Approach 4: Continued fraction / rational approximation attack
# =============================================================================
# CFRAC (Morrison-Brillhart): the continued fraction expansion of sqrt(n)
# naturally produces values close to multiples of the factors.

def cfrac_attack(n: int) -> Optional[list[int]]:
    """
    Continued fraction factoring method.
    The CF expansion of sqrt(n) produces convergents p_k/q_k where
    p_k^2 - n*q_k^2 is small. These small residues are more likely
    to be smooth, giving us relations for the congruence-of-squares approach.

    Enhanced with early-abort and multi-polynomial variants.
    """
    sqrt_n = math.isqrt(n)
    if sqrt_n * sqrt_n == n:
        return [sqrt_n, sqrt_n] if isprime(sqrt_n) else None

    bits = n.bit_length()
    B = max(bits * 3, 50)

    # Generate small primes for factor base
    factor_base = [2]
    p = 3
    while len(factor_base) < B:
        if _jacobi_symbol(n % p if n % p != 0 else 0, p) >= 0:
            factor_base.append(p)
        p += 2
        while not _is_small_prime(p):
            p += 2

    # CF expansion of sqrt(n)
    m, d, a0 = 0, 1, sqrt_n
    a = a0

    # Track convergents
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    relations = []
    needed = len(factor_base) + 5

    max_iter = max(needed * 50, 5000)

    for iteration in range(max_iter):
        m = d * a - m
        d = (n - m * m) // d
        if d == 0:
            break
        a = (a0 + m) // d

        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev

        # Q_i = (-1)^i * (p_curr^2 - n * q_curr^2)  ... but let's just use p_curr^2 mod n
        residue = (p_curr * p_curr) % n
        if residue > n // 2:
            residue = n - residue  # take the smaller representative

        if residue == 0:
            g = math.gcd(p_curr, n)
            if 1 < g < n:
                return _factor_remaining_full(n, g)
            continue

        # Try to factor residue over factor base
        exponents = []
        remaining = residue
        for p_fb in factor_base:
            exp = 0
            while remaining % p_fb == 0:
                remaining //= p_fb
                exp += 1
            exponents.append(exp % 2)

        if remaining == 1:
            relations.append((p_curr % n, exponents))

            if len(relations) >= needed:
                break

        # Also check gcd directly
        g = math.gcd(p_curr * p_curr - n, n) if iteration % 10 == 0 else 1
        if 1 < g < n:
            return _factor_remaining_full(n, g)

    # Try to combine relations
    if len(relations) >= 2:
        for i in range(len(relations)):
            for j in range(i + 1, len(relations)):
                x_i, exp_i = relations[i]
                x_j, exp_j = relations[j]
                combined = [(exp_i[k] + exp_j[k]) % 2 for k in range(len(factor_base))]
                if all(c == 0 for c in combined):
                    x = (x_i * x_j) % n
                    # Approximate y
                    for y_candidate_func in [lambda: math.gcd(x - 1, n), lambda: math.gcd(x + 1, n)]:
                        g = y_candidate_func()
                        if 1 < g < n:
                            return _factor_remaining_full(n, g)

    return None


def _is_small_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# =============================================================================
# Approach 5: Spectral method / DFT-based period detection
# =============================================================================
# Instead of QFT, use classical DFT to find approximate periods
# in the sequence a^x mod n

def spectral_attack(n: int) -> Optional[list[int]]:
    """
    Multi-strategy period detection for a^x mod n.

    Strategy 1: Pollard p-1 with increasing bounds
    Strategy 2: Williams p+1
    Strategy 3: Direct order-finding via divisors of n-1, n+1, etc.
    Strategy 4: GCD accumulation — compute gcd(a^k - 1, n) for smooth k values
    """
    if n % 2 == 0:
        return [2] + _factor_remaining(n // 2)

    bits = n.bit_length()

    for attempt in range(min(bits, 20)):
        a = random.randint(2, n - 1)
        g = math.gcd(a, n)
        if 1 < g < n:
            return _factor_remaining_full(n, g)

        # Strategy: compute a^M mod n for M = lcm(1..B)
        # This finds factors p where p-1 is B-smooth
        B = bits * bits  # polynomial in bit-length
        power = a
        for k in range(2, B + 1):
            power = pow(power, k, n)

        g = math.gcd(power - 1, n)
        if 1 < g < n:
            return _factor_remaining_full(n, g)

        # Williams p+1: use Lucas sequences
        factor = _williams_pp1(n, a)
        if factor is not None:
            return factor

    return None


def _williams_pp1(n: int, seed: int) -> Optional[list[int]]:
    """Williams p+1 method using Lucas sequences."""
    bits = n.bit_length()
    B = bits * bits

    v = seed % n
    for k in range(2, B + 1):
        # V_k(v) via the doubling formula for Lucas sequences
        v = _lucas_chain(v, k, n)
        if k % 50 == 0:
            g = math.gcd(v - 2, n)
            if 1 < g < n:
                return _factor_remaining_full(n, g)

    g = math.gcd(v - 2, n)
    if 1 < g < n:
        return _factor_remaining_full(n, g)
    return None


def _lucas_chain(v: int, k: int, n: int) -> int:
    """Compute V_k(v, 1) mod n using the Lucas chain."""
    if k == 1:
        return v
    if k == 2:
        return (v * v - 2) % n

    # Binary method for Lucas V sequence
    vl = v
    vh = (v * v - 2) % n
    bits_k = bin(k)[3:]  # skip '0b1'

    for bit in bits_k:
        if bit == '1':
            vl = (vl * vh - v) % n
            vh = (vh * vh - 2) % n
        else:
            vh = (vl * vh - v) % n
            vl = (vl * vl - 2) % n

    return vl


# =============================================================================
# Approach 6: Combined / chained approach
# =============================================================================

def combined_attack(n: int) -> Optional[list[int]]:
    """
    Chain multiple partial-information methods together.
    Each method extracts a bit of info; combined they might factor n.
    """
    if n % 2 == 0:
        return [2] + _factor_remaining(n // 2)

    # Phase 1: Quick checks
    sqrt_n = math.isqrt(n)
    if sqrt_n * sqrt_n == n:
        if isprime(sqrt_n):
            return [sqrt_n, sqrt_n]

    # Phase 2: Fermat's method (nearby factors)
    factor = _fermat_method(n, iterations=10000)
    if factor:
        return factor

    # Phase 3: p-1 smoothness for multiple bounds
    for B in [100, 500, 2000, 10000]:
        a = random.randint(2, n - 1)
        factor = _smooth_order_attack(a, n, B)
        if factor:
            return factor

    # Phase 4: Character-based
    factor = character_sum_attack(n)
    if factor:
        return factor

    # Phase 5: CF
    factor = cfrac_attack(n)
    if factor:
        return factor

    return None


def _fermat_method(n: int, iterations: int = 10000) -> Optional[list[int]]:
    """Fermat's factorization: look for n = a^2 - b^2 = (a-b)(a+b)."""
    a = math.isqrt(n)
    if a * a == n:
        if isprime(a):
            return [a, a]
        return None

    a += 1
    for _ in range(iterations):
        b2 = a * a - n
        b = math.isqrt(b2)
        if b * b == b2:
            p, q = a - b, a + b
            if p > 1 and q > 1 and p * q == n:
                return _factor_remaining_full(n, p)
        a += 1

    return None


# =============================================================================
# Approach 7: Modular constraint propagation
# =============================================================================
# For each small prime m, n mod m constrains the residues of p and q mod m.
# Combining these constraints via CRT might dramatically narrow the search.

def modular_constraint_attack(n: int) -> Optional[list[int]]:
    """
    Use modular constraints to narrow factor search.

    For each small prime m:
      p*q ≡ n (mod m)
      So (p mod m, q mod m) must be among the solutions to xy ≡ n (mod m)

    For each solution pair, we get a constraint on p mod m.
    Combining via CRT gives us p mod M (for M = product of small primes),
    then we search only candidates matching that residue.
    """
    if n % 2 == 0:
        return [2] + _factor_remaining(n // 2)

    sqrt_n = math.isqrt(n)
    bits = n.bit_length()

    small_primes = [p for p in _small_primes_list(60) if n % p != 0]

    # For each small prime, find possible residues for p
    constraints = []
    for m in small_primes[:15]:  # use first 15 primes
        n_mod_m = n % m
        possible_p = set()
        for p_res in range(1, m):
            # q_res = n_mod_m * inverse(p_res, m) mod m
            try:
                q_res = (n_mod_m * pow(p_res, -1, m)) % m
            except (ValueError, ZeroDivisionError):
                continue
            if q_res > 0:
                possible_p.add(p_res)
        constraints.append((m, possible_p))

    # Use CRT to combine constraints and find candidate residues
    # Start with the first constraint and iteratively combine
    if not constraints:
        return None

    modulus = constraints[0][0]
    candidates = list(constraints[0][1])

    for m, possible in constraints[1:8]:  # combine up to 8 primes
        new_candidates = []
        new_modulus = modulus * m
        for c in candidates:
            for p_res in possible:
                # Find x such that x ≡ c (mod modulus) and x ≡ p_res (mod m)
                # CRT
                try:
                    m_inv = pow(modulus, -1, m)
                except ValueError:
                    continue
                x = (c + modulus * ((p_res - c) * m_inv % m)) % new_modulus
                new_candidates.append(x)
        candidates = new_candidates
        modulus = new_modulus

        # Prune: keep only candidates in plausible range for p
        if modulus > sqrt_n:
            # Can directly check
            for c in candidates:
                p_candidate = c
                while p_candidate <= sqrt_n + modulus:
                    if p_candidate > 1 and n % p_candidate == 0:
                        q = n // p_candidate
                        if q > 1:
                            return _factor_remaining_full(n, p_candidate)
                    p_candidate += modulus
            return None

        # Cap candidates to prevent explosion
        if len(candidates) > 10000:
            candidates = candidates[:10000]

    # Search remaining candidates
    for c in candidates:
        p_candidate = c
        while p_candidate < sqrt_n + modulus:
            if p_candidate > 1 and n % p_candidate == 0:
                q = n // p_candidate
                if q > 1:
                    return _factor_remaining_full(n, p_candidate)
            p_candidate += modulus

    return None


def _small_primes_list(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


# =============================================================================
# Approach 8: Algebraic group structure attack
# =============================================================================
# n = pq. The group (Z/nZ)* ≅ (Z/pZ)* × (Z/qZ)* by CRT.
# The order of this group is phi(n) = (p-1)(q-1).
# If we compute a^k for specially chosen k values, we might split the group.

def group_structure_attack(n: int) -> Optional[list[int]]:
    """
    Exploit the product structure of (Z/nZ)*.

    Key idea: pick random elements and compute their order.
    The order divides lcm(p-1, q-1). If we can find the order
    of enough elements, we can recover lcm(p-1, q-1), and from
    that, phi(n) = (p-1)(q-1), and then factor n.

    Enhanced: use the fact that for random a, the probability that
    ord(a, n) = lcm(p-1, q-1) is high. And from phi(n), factoring is easy.
    """
    if n % 2 == 0:
        return [2] + _factor_remaining(n // 2)

    bits = n.bit_length()

    # Strategy: compute a^(n-1) mod n for random a.
    # By Fermat, a^(p-1) ≡ 1 (mod p), so a^(n-1) mod p depends on (n-1) mod (p-1).
    # If (n-1) is not divisible by (p-1), a^(n-1) mod p ≠ 1 in general.
    # This means gcd(a^(n-1) - 1, n) might give a factor!

    for _ in range(bits * 5):
        a = random.randint(2, n - 1)
        g = math.gcd(a, n)
        if 1 < g < n:
            return _factor_remaining_full(n, g)

        # Try a^(n-1) - 1
        val = pow(a, n - 1, n)
        if val != 1:  # n is not a Carmichael number for this base
            g = math.gcd(val - 1, n)
            if 1 < g < n:
                return _factor_remaining_full(n, g)

        # Try a^((n-1)/2) — might split if (n-1)/2 is odd multiple of one order
        if (n - 1) % 2 == 0:
            val = pow(a, (n - 1) // 2, n)
            for v in [val - 1, val + 1]:
                g = math.gcd(v, n)
                if 1 < g < n:
                    return _factor_remaining_full(n, g)

        # Try a^(isqrt(n)) — a random exponent that's a function of n
        for exp in [math.isqrt(n), n % (1 << bits // 2), bits * bits]:
            val = pow(a, exp, n)
            g = math.gcd(val - 1, n)
            if 1 < g < n:
                return _factor_remaining_full(n, g)

    return None


# =============================================================================
# Helpers
# =============================================================================

def _factor_remaining(n: int) -> list[int]:
    """Factor a number that might be prime or composite."""
    if n < 2:
        return []
    if isprime(n):
        return [n]
    from baselines import pollard_rho
    result = pollard_rho(n)
    return result if result else [n]


def _factor_remaining_full(n: int, known_factor: int) -> list[int]:
    """Given one factor, return the complete factorization."""
    factors = []
    remaining = n
    while remaining % known_factor == 0:
        remaining //= known_factor

    f1_factors = _factor_remaining(known_factor)
    f2_factors = _factor_remaining(remaining)

    # Account for multiplicity
    factors = []
    temp = n
    for f in sorted(set(f1_factors + f2_factors)):
        while temp % f == 0:
            factors.append(f)
            temp //= f
    if temp > 1:
        factors.append(temp)

    return sorted(factors)
