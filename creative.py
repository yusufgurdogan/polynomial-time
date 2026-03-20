"""
Creative / novel factoring approaches — Round 2.

These go beyond standard textbook algorithms and explore less-trodden paths.
"""

import math
import random
from typing import Optional
from sympy import isprime


# =============================================================================
# Approach A: Multi-base power GCD accumulation
# =============================================================================
# Insight: for EACH base a, gcd(a^M - 1, n) reveals p iff ord_p(a) | M.
# With ONE base, we need M to be a multiple of ord_p(a).
# With MANY bases, the probability that at least one has a smooth order
# increases multiplicatively. Instead of raising the smoothness bound,
# we increase the number of bases — different scaling tradeoff.

def multibase_power_gcd(n: int) -> Optional[list[int]]:
    """
    Use many random bases with a SMALL smoothness bound.
    For each base a, compute a^M mod n where M = lcm(1..B).
    Accumulate gcd(a^M - 1, n) across bases.

    Key insight: even if no single base has smooth order,
    partial information from multiple bases might combine.
    We multiply residues: product of (a_i^M - 1) mod n,
    then take gcd with n.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    # Keep B polynomial in bits — this is the crucial design choice
    B = bits * 2

    # Precompute M_exponents: for each prime p <= B, highest power p^k <= B
    primes = _sieve(B)

    num_bases = bits * bits  # polynomial number of bases

    # Strategy 1: individual base checks
    for _ in range(num_bases):
        a = random.randint(2, n - 1)
        g = math.gcd(a, n)
        if 1 < g < n:
            return _finish(n, g)

        power = a
        for p in primes:
            pk = p
            while pk <= B:
                power = pow(power, p, n)
                pk *= p
            # Check periodically
            g = math.gcd(power - 1, n)
            if g == n:
                break  # useless, try next base
            if g > 1:
                return _finish(n, g)

    # Strategy 2: accumulate across bases
    # Product of (a_i^M - 1) mod n
    accumulated = 1
    for _ in range(num_bases):
        a = random.randint(2, n - 1)
        power = a
        for p in primes:
            pk = p
            while pk <= B:
                power = pow(power, p, n)
                pk *= p
        residue = (power - 1) % n
        if residue == 0:
            continue
        accumulated = (accumulated * residue) % n
        g = math.gcd(accumulated, n)
        if 1 < g < n:
            return _finish(n, g)

    return None


# =============================================================================
# Approach B: Lattice-based Coppersmith bootstrap
# =============================================================================
# Coppersmith: if you know the top ~half bits of p, you can recover the rest
# in polynomial time. But we don't know ANY bits.
#
# Bootstrap idea: use modular constraints from small primes to get a few bits,
# then use lattice methods to amplify. Even knowing p mod (small product)
# constrains the top bits when combined with the range [sqrt(n)/2, sqrt(n)*2].

def coppersmith_bootstrap(n: int) -> Optional[list[int]]:
    """
    Bootstrap Coppersmith's method by extracting partial factor info
    from modular constraints, then using polynomial root-finding.

    For n = pq, and known p0 = p mod M (for some M):
    p = p0 + M*k for some integer k
    So n = (p0 + M*k) * q
    Which means M*k*q ≡ n - p0*q (mod something useful)

    We set up a polynomial f(x) = p0 + M*x and look for small roots
    of f(x) mod p (which we don't know), using the fact that
    f(x) * (n/f(x)) = n.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    sqrt_n = math.isqrt(n)

    # Phase 1: collect modular info
    # For small primes m, find all possible p mod m
    small_primes = _sieve(min(bits * 3, 200))

    # For each small prime, n mod m constrains (p mod m, q mod m)
    # We enumerate and use CRT
    # But instead of full CRT (exponential), we use a meet-in-the-middle approach

    # Split primes into two halves
    half = len(small_primes) // 2
    primes_A = small_primes[:half]
    primes_B = small_primes[half:]

    M_A = 1
    for p in primes_A:
        M_A *= p
    M_B = 1
    for p in primes_B:
        M_B *= p

    # For group A: enumerate all possible p mod M_A
    candidates_A = _enumerate_residues(n, primes_A)
    # For group B: enumerate all possible p mod M_B
    candidates_B = _enumerate_residues(n, primes_B)

    if not candidates_A or not candidates_B:
        return None

    # Meet in the middle: for each candidate in A, compute what p mod M_B must be
    # given the constraint that p * (n/p) = n and p is in [2, sqrt_n]

    # Actually, we can be smarter: for each residue r_A mod M_A,
    # p = r_A + M_A * t, and p must divide n.
    # So n ≡ 0 (mod p), meaning n mod (r_A + M_A * t) = 0.
    # For small M_A, just search t.

    for r_A in candidates_A[:100]:  # cap to prevent blowup
        if r_A == 0:
            continue
        # p = r_A + M_A * t, p divides n, p <= sqrt_n
        # Search t
        max_t = (sqrt_n - r_A) // M_A + 1
        if max_t <= 0:
            continue

        # For very small M_A, we can search directly (still exponential)
        # But if M_A is large enough relative to sqrt(n), the search is small
        if max_t < bits * bits * bits:  # polynomial search
            p_cand = r_A
            while p_cand <= sqrt_n + M_A:
                if p_cand > 1 and n % p_cand == 0:
                    return _finish(n, p_cand)
                p_cand += M_A
        else:
            # Too many candidates — try lattice reduction
            # Set up the lattice: we want small (t, 1) such that r_A + M_A*t | n
            # Equivalent: find small x such that (r_A + M_A*x) | n
            # i.e., n ≡ 0 mod (r_A + M_A*x)
            # This is a variant of the divisor problem in a lattice
            pass

    return None


def _enumerate_residues(n, primes):
    """Enumerate possible values of p mod (product of primes)."""
    # Start with first prime
    if not primes:
        return [0]

    candidates = []
    m = primes[0]
    n_mod_m = n % m
    for r in range(1, m):
        if n_mod_m == 0 and r == 0:
            continue
        if (n_mod_m * pow(r, -1, m) if r != 0 else 0) % 1 == 0:
            # Check if r could be p mod m: need n/r mod m to also be valid
            try:
                q_res = (n_mod_m * pow(r, -1, m)) % m
                if q_res > 0:
                    candidates.append(r)
            except (ValueError, ZeroDivisionError):
                pass

    # CRT with remaining primes
    for prime in primes[1:]:
        new_candidates = []
        n_mod_p = n % prime
        modulus = 1
        for p in primes[:primes.index(prime)]:
            modulus *= p

        for c in candidates:
            for r in range(1, prime):
                try:
                    q_res = (n_mod_p * pow(r, -1, prime)) % prime
                except (ValueError, ZeroDivisionError):
                    continue
                if q_res > 0:
                    # CRT: combine c mod modulus with r mod prime
                    try:
                        m_inv = pow(modulus, -1, prime)
                    except ValueError:
                        continue
                    combined = (c + modulus * ((r - c % prime) * m_inv % prime)) % (modulus * prime)
                    new_candidates.append(combined)

        candidates = new_candidates
        if len(candidates) > 5000:
            candidates = candidates[:5000]  # prevent blowup

    return candidates


# =============================================================================
# Approach C: Elliptic curve method (ECM)
# =============================================================================
# Lenstra's ECM: random elliptic curves mod n. The group order mod p
# varies per curve, so we get many "bites at the apple" for smooth orders.
# Sub-exponential, but excellent for finding factors up to ~60 digits.

def ecm_attack(n: int) -> Optional[list[int]]:
    """
    Elliptic Curve Method. For each random curve E mod n,
    compute [M]P for M = lcm(1..B). If the group order mod p
    is B-smooth, we'll hit the identity mod p but not mod q,
    revealing p via gcd.
    """
    if n % 2 == 0:
        return _finish(n, 2)
    if n % 3 == 0:
        return _finish(n, 3)

    bits = n.bit_length()
    B1 = max(bits * 5, 100)   # stage 1 bound
    B2 = B1 * 10              # stage 2 bound
    num_curves = max(bits * 2, 20)

    primes = _sieve(B2)
    stage1_primes = [p for p in primes if p <= B1]

    for _ in range(num_curves):
        # Random curve in Montgomery form: By^2 = x^3 + Ax^2 + x
        # Random point (x, y) on curve, derive A
        sigma = random.randint(6, n - 1)
        u = (sigma * sigma - 5) % n
        v = (4 * sigma) % n

        try:
            v_inv = pow(v, -1, n)
        except ValueError:
            g = math.gcd(v, n)
            if 1 < g < n:
                return _finish(n, g)
            continue

        # Montgomery curve parameter
        x0 = (u * u * u) % n * pow(v * v * v, -1, n) % n
        A = ((v - u) ** 3 * (3 * u + v)) % n
        try:
            denom = pow(4 * u * u * u * v, -1, n)
        except ValueError:
            g = math.gcd(4 * u * u * u * v % n, n)
            if 1 < g < n:
                return _finish(n, g)
            continue
        A = (A * denom - 2) % n

        # Stage 1: scalar multiplication by lcm(1..B1)
        Qx, Qz = u * u * u % n * pow(v * v * v % n, -1, n) % n, 1

        for p in stage1_primes:
            pk = p
            while pk <= B1:
                Qx, Qz = _ec_multiply(Qx, Qz, p, A, n)
                if Qz == 0:
                    break
                g = math.gcd(Qz, n)
                if 1 < g < n:
                    return _finish(n, g)
                if g == n:
                    break
                pk *= p
            if Qz == 0:
                break

        if Qz == 0:
            continue

        g = math.gcd(Qz, n)
        if 1 < g < n:
            return _finish(n, g)

        # Stage 2: check primes in (B1, B2]
        # Standard continuation
        stage2_primes = [p for p in primes if B1 < p <= B2]
        Rx, Rz = Qx, Qz
        accumulated = 1
        for p in stage2_primes:
            Rx, Rz = _ec_multiply(Rx, Rz, p, A, n)
            if Rz == 0:
                break
            accumulated = (accumulated * Rz) % n
            if random.random() < 0.05:  # periodic gcd
                g = math.gcd(accumulated, n)
                if 1 < g < n:
                    return _finish(n, g)
                if g == n:
                    break

        g = math.gcd(accumulated, n)
        if 1 < g < n:
            return _finish(n, g)

    return None


def _ec_multiply(x: int, z: int, k: int, A: int, n: int) -> tuple[int, int]:
    """Montgomery ladder scalar multiplication on elliptic curve mod n."""
    if k == 0:
        return 0, 0
    if k == 1:
        return x, z

    # Binary method with Montgomery ladder
    r0x, r0z = x, z
    r1x, r1z = _ec_double(x, z, A, n)

    for bit in bin(k)[3:]:  # skip '0b1'
        if bit == '1':
            r0x, r0z = _ec_add(r0x, r0z, r1x, r1z, x, z, n)
            r1x, r1z = _ec_double(r1x, r1z, A, n)
        else:
            r1x, r1z = _ec_add(r0x, r0z, r1x, r1z, x, z, n)
            r0x, r0z = _ec_double(r0x, r0z, A, n)

    return r0x % n, r0z % n


def _ec_double(x: int, z: int, A: int, n: int) -> tuple[int, int]:
    """Point doubling on Montgomery curve."""
    u = (x + z) % n
    u = (u * u) % n
    v = (x - z) % n
    v = (v * v) % n
    diff = (u - v) % n
    rx = (u * v) % n
    rz = (diff * (v + ((A + 2) * pow(4, -1, n) % n) * diff)) % n
    return rx, rz


def _ec_add(x1: int, z1: int, x2: int, z2: int,
            x0: int, z0: int, n: int) -> tuple[int, int]:
    """Differential point addition on Montgomery curve."""
    u = ((x1 - z1) * (x2 + z2)) % n
    v = ((x1 + z1) * (x2 - z2)) % n
    add = (u + v) % n
    sub = (u - v) % n
    rx = (z0 * add * add) % n
    rz = (x0 * sub * sub) % n
    return rx, rz


# =============================================================================
# Approach D: Quadratic form / representation attack
# =============================================================================
# For various small D, check if n can be represented as x^2 + D*y^2.
# By genus theory, such representations (when they exist) leak info
# about the Legendre symbols (D/p) and (-D/p), constraining the factors.

def quadratic_form_attack(n: int) -> Optional[list[int]]:
    """
    Find representations n = x^2 + D*y^2 for various D.
    Each representation constrains the factors via quadratic reciprocity.

    For n = pq and n = x^2 + D*y^2:
    - (-D/n) = 1, which means (-D/p)(-D/q) = 1
    - So either both are 1 or both are -1.
    - If we find D where this splits differently, we get a factor.

    Also: Cornacchia's algorithm finds representations efficiently.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()
    sqrt_n = math.isqrt(n)

    # Collect quadratic character data
    # For each small D, compute Jacobi symbol (-D/n)
    constraints = []

    for D in range(1, min(bits * bits, 500)):
        neg_D_mod_n = (-D) % n
        jacobi = _jacobi(neg_D_mod_n, n)

        if jacobi == 0:
            g = math.gcd(D, n)
            if 1 < g < n:
                return _finish(n, g)
            continue

        if jacobi == 1:
            # (-D/p)(-D/q) = 1, so both are same sign
            # Try Cornacchia to find representation
            rep = _cornacchia(n, D)
            if rep is not None:
                x, y = rep
                # n = x^2 + D*y^2, verified
                # Now use this: gcd(x ± anything_derived, n)
                for val in [x, y, x + y, x - y, x * y, D * y]:
                    g = math.gcd(val, n)
                    if 1 < g < n:
                        return _finish(n, g)

            # Also try: if n = x^2 + D*y^2, then mod p:
            # x^2 + D*y^2 ≡ 0 (mod p)
            # So x/y ≡ ±sqrt(-D) (mod p)
            # If we can compute sqrt(-D) mod n, it splits into
            # different values mod p and mod q

        elif jacobi == -1:
            # (-D/p)(-D/q) = -1, so they differ
            # This means one of (-D/p), (-D/q) is 1 and other is -1
            # Try: a^((n-1)/2) mod n should be -1
            # But a^((p-1)/2) mod p differs from a^((q-1)/2) mod q
            a = neg_D_mod_n
            if a == 0:
                continue
            # Compute gcd(a^((n-1)/2) + 1, n) — this often factors!
            val = pow(a, (n - 1) // 2, n)
            for v in [val - 1, val + 1]:
                g = math.gcd(v % n, n)
                if 1 < g < n:
                    return _finish(n, g)

            # Also try a^((n+1)/4) if n ≡ 3 mod 4
            if n % 4 == 3:
                val = pow(a, (n + 1) // 4, n)
                for v in [val * val - a, val * val + a, val - 1, val + 1]:
                    g = math.gcd(v % n, n)
                    if 1 < g < n:
                        return _finish(n, g)

    # Combine constraints: use the character data to narrow down p mod 4D
    # for various D, then CRT
    return None


def _cornacchia(n: int, D: int) -> Optional[tuple[int, int]]:
    """
    Cornacchia's algorithm: find x, y such that x^2 + D*y^2 = n.
    Works when n is prime. For composite n, may fail — that's useful info too.
    """
    if D >= n:
        return None

    # Need sqrt(-D) mod n
    neg_D = (-D) % n
    try:
        r = _tonelli_shanks(neg_D, n)
    except (ValueError, Exception):
        return None

    if r is None:
        return None

    if r < n // 2:
        r = n - r

    # Extended Euclidean-like reduction
    sqrt_n = math.isqrt(n)
    a, b = n, r
    while b > sqrt_n:
        a, b = b, a % b

    # Check if (n - b^2) / D is a perfect square
    remainder = n - b * b
    if remainder < 0 or remainder % D != 0:
        return None

    c = remainder // D
    sqrt_c = math.isqrt(c)
    if sqrt_c * sqrt_c == c:
        return (b, sqrt_c)

    return None


def _tonelli_shanks(a: int, p: int) -> Optional[int]:
    """Compute sqrt(a) mod p using Tonelli-Shanks."""
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        return None

    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)

    # Factor p-1 = Q * 2^S
    Q, S = p - 1, 0
    while Q % 2 == 0:
        Q //= 2
        S += 1

    # Find quadratic non-residue
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    M = S
    c = pow(z, Q, p)
    t = pow(a, Q, p)
    R = pow(a, (Q + 1) // 2, p)

    while True:
        if t == 1:
            return R
        # Find least i such that t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
            if i >= M:
                return None

        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p


# =============================================================================
# Approach E: Generalized number field sieve (simplified)
# =============================================================================
# The NFS exploits smoothness in algebraic number fields.
# We implement a simplified version that uses multiple polynomial families.

def multi_polynomial_sieve(n: int) -> Optional[list[int]]:
    """
    Simplified multi-polynomial quadratic sieve.
    Key improvement over basic QS: use multiple polynomials Q(x) = (ax+b)^2 - n
    for different (a,b), each producing values more likely to be smooth.
    """
    if n % 2 == 0:
        return _finish(n, 2)

    sqrt_n = math.isqrt(n)
    if sqrt_n * sqrt_n == n:
        if isprime(sqrt_n):
            return [sqrt_n, sqrt_n]

    bits = n.bit_length()

    # Factor base
    fb_limit = max(int(math.exp(0.5 * math.sqrt(math.log(float(n)) * math.log(math.log(float(n)))))), 50)
    fb_limit = min(fb_limit, 2000)
    all_primes = _sieve(fb_limit)
    factor_base = [-1]  # include -1 for sign
    for p in all_primes:
        if p == 2 or _jacobi(n % p, p) == 1:
            factor_base.append(p)

    fb_size = len(factor_base)
    needed = fb_size + 5

    relations = []  # (value_mod_n, exponent_vector_mod_2)
    sieve_range = max(fb_size * 50, 5000)

    # Multi-polynomial: try different offsets
    for poly_idx in range(max(bits, 10)):
        if len(relations) >= needed:
            break

        offset = sqrt_n + poly_idx * sieve_range

        for x in range(1, sieve_range):
            val = (offset + x) ** 2 - n
            if val <= 0:
                continue

            original_val = val
            sign = 1
            if val < 0:
                val = -val
                sign = -1

            # Factor over factor base
            exponents = [1 if sign < 0 else 0]  # sign bit
            remaining = val
            for p in factor_base[1:]:  # skip -1
                exp = 0
                while remaining % p == 0:
                    remaining //= p
                    exp += 1
                exponents.append(exp % 2)

            if remaining == 1:  # smooth!
                relations.append(((offset + x) % n, exponents))
                if len(relations) >= needed:
                    break

    if len(relations) < 2:
        return None

    # Gaussian elimination over GF(2)
    return _gf2_solve(n, relations, factor_base)


def _gf2_solve(n, relations, factor_base):
    """GF(2) Gaussian elimination to find congruence of squares."""
    num_rels = len(relations)
    fb_size = len(factor_base)

    matrix = [rel[1][:] for rel in relations]
    # Pad if needed
    for row in matrix:
        while len(row) < fb_size:
            row.append(0)

    # Track combinations
    combo = [[1 if i == j else 0 for j in range(num_rels)] for i in range(num_rels)]

    pivot_row = 0
    for col in range(fb_size):
        # Find pivot
        found = None
        for row in range(pivot_row, num_rels):
            if matrix[row][col] == 1:
                found = row
                break
        if found is None:
            continue

        # Swap
        matrix[pivot_row], matrix[found] = matrix[found], matrix[pivot_row]
        combo[pivot_row], combo[found] = combo[found], combo[pivot_row]

        # Eliminate
        for row in range(num_rels):
            if row != pivot_row and matrix[row][col] == 1:
                for c in range(fb_size):
                    matrix[row][c] ^= matrix[pivot_row][c]
                for c in range(num_rels):
                    combo[row][c] ^= combo[pivot_row][c]

        pivot_row += 1

    # Find null space vectors
    for row in range(num_rels):
        if all(matrix[row][c] == 0 for c in range(fb_size)):
            indices = [j for j in range(num_rels) if combo[row][j] == 1]
            if len(indices) < 2:
                continue

            # Compute x = product of values, y = sqrt(product of Q values)
            x = 1
            y_sq = 1
            for idx in indices:
                x = (x * relations[idx][0]) % n
                qi = relations[idx][0] ** 2 - n
                y_sq *= abs(qi)

            y = math.isqrt(y_sq)
            if y * y != y_sq:
                continue

            y = y % n
            g = math.gcd((x - y) % n, n)
            if 1 < g < n:
                return _finish(n, g)
            g = math.gcd((x + y) % n, n)
            if 1 < g < n:
                return _finish(n, g)

    return None


# =============================================================================
# Approach F: Random walk / cycle detection in (Z/nZ)*
# =============================================================================
# Pollard's rho finds collisions in the iteration x -> f(x) mod n.
# But what if we use a STRUCTURED walk that's more likely to collide
# meaningfully — e.g., walks that respect the group structure?

def structured_walk_attack(n: int) -> Optional[list[int]]:
    """
    Instead of Pollard's x -> x^2 + c, use walks derived from
    the algebraic structure of n:
    - Walk on (Z/nZ)* using multiplication by carefully chosen elements
    - Walk using the Fibonacci recurrence mod n
    - Walk using polynomial evaluation chains
    """
    if n % 2 == 0:
        return _finish(n, 2)

    bits = n.bit_length()

    max_steps = bits * bits * 2  # keep it polynomial but bounded

    # Strategy 1: Fibonacci walk
    # F(k) mod p has period π(p) (Pisano period) dividing p^2-1 or 2(p+1)
    a, b = 0, 1
    for k in range(2, max_steps):
        a, b = b, (a + b) % n
        if k % 50 == 0:
            g = math.gcd(b, n)
            if 1 < g < n:
                return _finish(n, g)

    # Strategy 2: Lucas sequence walk
    for P in range(3, min(bits, 20)):
        v_prev, v_curr = 2, P
        for k in range(2, max_steps):
            v_prev, v_curr = v_curr, (P * v_curr - v_prev) % n
            if k % 50 == 0:
                g = math.gcd(v_curr - 2, n)
                if 1 < g < n:
                    return _finish(n, g)

    # Strategy 3: Polynomial chain walk (Pollard rho variant with higher degree)
    for d in [3, 5]:
        x = random.randint(2, n - 1)
        y = x
        for _ in range(max_steps):
            x = (pow(x, d, n) + 1) % n
            y = (pow(y, d, n) + 1) % n
            y = (pow(y, d, n) + 1) % n
            g = math.gcd(abs(x - y), n)
            if 1 < g < n:
                return _finish(n, g)
            if g == n:
                break

    return None


# =============================================================================
# Master combination: try all creative approaches in order of speed
# =============================================================================

def creative_combined(n: int) -> Optional[list[int]]:
    """Run all creative approaches, fastest first."""
    if n % 2 == 0:
        return _finish(n, 2)

    for algo in [
        multibase_power_gcd,
        quadratic_form_attack,
        structured_walk_attack,
        ecm_attack,
        multi_polynomial_sieve,
        coppersmith_bootstrap,
    ]:
        result = algo(n)
        if result is not None:
            return result

    return None


# =============================================================================
# Helpers
# =============================================================================

def _finish(n: int, factor: int) -> list[int]:
    """Given one factor, produce the full factorization."""
    factors = []
    remaining = n
    while remaining % factor == 0:
        remaining //= factor
    # Factor both parts
    for part in [factor, remaining]:
        if part <= 1:
            continue
        if isprime(part):
            factors.append(part)
        else:
            # Recurse with pollard
            from baselines import pollard_rho
            sub = pollard_rho(part)
            if sub:
                factors.extend(sub)
            else:
                factors.append(part)
    # Reconstruct with correct multiplicities
    result = []
    temp = n
    for f in sorted(set(factors)):
        while temp % f == 0:
            result.append(f)
            temp //= f
    if temp > 1:
        result.append(temp)
    return sorted(result)


def _sieve(limit: int) -> list[int]:
    if limit < 2:
        return []
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            for j in range(i*i, limit + 1, i):
                s[j] = False
    return [i for i in range(2, limit + 1) if s[i]]


def _jacobi(a: int, n: int) -> int:
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
