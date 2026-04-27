"""
FactorCup Entry — Multi-strategy factoring with novel approaches.

Escalation:
  trial division for tiny factors
  Pollard p-1/p+1 for smooth-order factors
  Pollard rho for small N
  Quadratic Sieve for medium N
  Novel approaches (orbit lattice, Paillier lifting)
  sympy fallback
"""

import math
import random
import time
import numpy as np
from sympy.ntheory import factorint as _sympy_factorint


def factor(N: int) -> tuple[int, int]:
    if N % 2 == 0:
        return (2, N // 2)
    s = math.isqrt(N)
    if s * s == N:
        return (s, s)

    for d in range(3, min(10000, s + 1), 2):
        if N % d == 0:
            return _ret(d, N)

    bits = N.bit_length()

    # p-1/p+1 (fast, catches smooth orders)
    r = _pollard_pm1(N, B1=min(20000, bits * bits * 3))
    if r: return r
    r = _williams_pp1(N, B1=min(10000, bits * bits * 2))
    if r: return r

    # Pollard rho for small N only (fast for <80 bits)
    if bits <= 80:
        r = _pollard_rho(N, max_r=1 << 20)
        if r: return r

    # For medium sizes: QS is faster than sympy's ECM
    if 50 <= bits <= 112:
        r = _quadratic_sieve(N, timeout=50.0)
        if r: return r

    # For larger sizes: sympy's ECM is better than our Python QS
    if bits > 112:
        r = _sympy_factor_timed(N, timeout=55.0)
        if r: return r

    # Fallback: try QS if sympy failed, or try sympy if QS failed
    if bits > 112:
        r = _quadratic_sieve(N, timeout=20.0)
        if r: return r
    else:
        r = _sympy_factor_timed(N, timeout=30.0)
        if r: return r

    # Novel approaches
    r = _novel_orbit_lattice(N, timeout=5.0)
    if r: return r
    r = _novel_paillier_lift(N, timeout=5.0)
    if r: return r

    return _sympy_factor(N)


def _ret(d, N):
    q = N // d
    return (min(d, q), max(d, q))


# ============================================================================
# POLLARD RHO
# ============================================================================

def _pollard_rho(N, max_r=500000):
    for c in range(1, 50):
        x = y = (c * 7 + 13) % N + 2
        d, r, q = 1, 1, 1
        while d == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % N
            k = 0
            while k < r and d == 1:
                ys = y
                m = min(128, r - k)
                for _ in range(m):
                    y = (y * y + c) % N
                    q = q * (x - y) % N
                d = math.gcd(q, N)
                k += m
            r <<= 1
            if r > max_r:
                break
        if d == N:
            d = 1
            while d == 1:
                ys = (ys * ys + c) % N
                d = math.gcd(x - ys, N)
        if 1 < d < N:
            return _ret(d, N)
    return None


# ============================================================================
# POLLARD p-1
# ============================================================================

def _pollard_pm1(N, B1=100000):
    a = 2
    for p in _sieve_primes(B1):
        pk = p
        while pk <= B1:
            a = pow(a, p, N)
            pk *= p
    g = math.gcd(a - 1, N)
    if 1 < g < N:
        return _ret(g, N)
    return None


# ============================================================================
# WILLIAMS p+1
# ============================================================================

def _williams_pp1(N, B1=100000):
    for v0 in [3, 5, 7, 11, 13]:
        v = v0
        for p in _sieve_primes(B1):
            pk = p
            while pk <= B1:
                v = _lucas_v(v, p, N)
                pk *= p
        g = math.gcd(v - 2, N)
        if 1 < g < N:
            return _ret(g, N)
    return None


def _lucas_v(v, n, N):
    if n == 0: return 2
    if n == 1: return v
    vk, vk1 = v, (v * v - 2) % N
    for i in range(n.bit_length() - 2, -1, -1):
        if (n >> i) & 1:
            vk = (vk * vk1 - v) % N
            vk1 = (vk1 * vk1 - 2) % N
        else:
            vk1 = (vk * vk1 - v) % N
            vk = (vk * vk - 2) % N
    return vk


# ============================================================================
# QUADRATIC SIEVE
# ============================================================================

def _quadratic_sieve(N, timeout=50.0):
    t0 = time.monotonic()
    bits = N.bit_length()

    # Parameters from L[1/2] formula
    ln_N = bits * math.log(2)
    ln_ln_N = math.log(max(ln_N, 2))
    L = math.exp(math.sqrt(ln_N * ln_ln_N))
    B = max(100, min(50000, int(L ** 0.55)))
    M = max(10000, min(500000, int(L ** 0.9)))

    # Build factor base
    fb = []        # primes
    fb_root = []   # sqrt(N) mod p
    for p in _sieve_primes(B):
        if p == 2:
            fb.append(2)
            fb_root.append(0)
            continue
        if N % p == 0:
            return _ret(p, N)
        r = _tonelli_shanks(N % p, p)
        if r is not None:
            fb.append(p)
            fb_root.append(r)

    fb_size = len(fb)
    if fb_size < 10:
        return None

    target = fb_size + 30
    lp_bound = B * B  # large prime bound

    relations = []  # (x_val, exp_vec, sign)
    partials = {}   # large_prime -> (x_val, exp_vec, sign)

    sqrt_N = math.isqrt(N)

    # Multiple polynomial approach:
    # Poly 0: Q(x) = (x + sqrt_N + 1)^2 - N  (standard)
    # Poly k: Q(x) = (a_k * x + b_k)^2 - N   (MPQS)

    poly_idx = 0

    while len(relations) < target:
        if time.monotonic() - t0 > timeout:
            break

        if poly_idx == 0:
            a, b_val = 1, sqrt_N + 1
        else:
            # MPQS: choose a = q^2 for a prime q from the factor base
            qi = min(fb_size - 1, fb_size // 3 + poly_idx)
            if qi >= fb_size:
                qi = fb_size // 2
            q_prime = fb[qi]
            if q_prime < 3:
                poly_idx += 1
                continue
            a = q_prime * q_prime
            # Hensel lift sqrt(N) mod q to mod q^2
            r0 = fb_root[qi]
            inv2r = pow(2 * r0 % a, -1, a) if math.gcd(2 * r0, a) == 1 else None
            if inv2r is None:
                poly_idx += 1
                continue
            b_mod_a = (r0 - (r0 * r0 - N % a) * inv2r) % a
            if (b_mod_a * b_mod_a - N) % a != 0:
                b_mod_a = a - b_mod_a
                if (b_mod_a * b_mod_a - N) % a != 0:
                    poly_idx += 1
                    continue
            # Shift b to be near sqrt(N)
            k = (sqrt_N - b_mod_a) // a
            b_val = b_mod_a + k * a

        # Sieve this polynomial
        n_found = _qs_sieve(N, a, b_val, M, fb, fb_root, fb_size,
                            lp_bound, relations, partials, target, t0, timeout)
        poly_idx += 1

        if poly_idx > 500:
            break

    if len(relations) < fb_size + 1:
        return None

    return _qs_solve(N, relations, fb, fb_size)


def _qs_sieve(N, a, b, M, fb, fb_root, fb_size, lp_bound,
              relations, partials, target, t0, timeout):
    """Sieve polynomial Q(x) = (a*x + b)^2 - N over [-M, M]."""
    n_found_before = len(relations)
    sieve_len = 2 * M

    # Compute approximate log2 of Q values for threshold
    sqrt_N = math.isqrt(N)
    # At center (x=0): Q ≈ (b^2 - N), magnitude ≈ a few bits below N
    # At edge (x=±M): Q ≈ (aM)^2 + 2abM ≈ 2*sqrt(N)*a*M (if b ≈ sqrt(N))
    max_Q = float(max(abs(b * b - N), abs((a * M + b) ** 2 - N)))
    if max_Q <= 1:
        max_Q = float(N)
    log2_maxQ = math.log2(max_Q) if max_Q > 0 else 64

    # Threshold: sieve value must be close to log2(|Q|) for smooth candidates
    # Allow tolerance for one large prime factor
    lp_bits = math.log2(float(lp_bound)) if lp_bound > 1 else 0
    thresh = max(0, log2_maxQ - lp_bits - 5)

    # Use numpy for fast sieving
    sieve = np.zeros(sieve_len, dtype=np.float32)

    # Sieve with each factor base prime
    # Index i in sieve corresponds to x = i - M
    # Q(x) ≡ 0 (mod p) when (a*(i-M) + b)^2 ≡ N (mod p)
    # i.e., a*(i-M)+b ≡ ±r (mod p), so i ≡ (±r - b)/a + M (mod p)

    for i in range(fb_size):
        p = fb[i]
        logp = math.log2(p)

        if p == 2:
            # Q(x) = (a*x+b)^2 - N. For a=1: Q = (x+b)^2 - N ≡ (x+b)^2+1 (mod 2)
            # Even when x+b is odd, i.e. x has opposite parity to b
            # Sieve index: i = x + M, so x = i - M.  x+b odd when (i-M+b) is odd
            start = (1 - (b - M) % 2) % 2  # start at first index where (i-M+b) is odd
            sieve[start::2] += logp
            continue

        if a % p == 0:
            if (b * b - N) % p == 0:
                start = M % p  # x=0 maps to index M
                sieve[start::p] += logp
            continue

        r = fb_root[i]
        ainv = pow(a % p, -1, p)

        for sign_r in (r, p - r):
            # x ≡ (sign_r - b) * ainv (mod p)
            x_mod = ((sign_r - b % p) * ainv) % p
            # Convert to sieve index: i = x + M, so i ≡ x_mod + M (mod p)
            start = (x_mod + M) % p
            sieve[start::p] += logp

    # Find candidates above threshold
    candidates = np.where(sieve >= thresh)[0]

    for idx in candidates:
        if len(relations) >= target:
            break
        if time.monotonic() - t0 > timeout:
            break

        x = int(idx) - M
        ax_b = a * x + b
        Q = ax_b * ax_b - N
        if Q == 0:
            g = math.gcd(ax_b, N)
            if 1 < g < N:
                return _ret(g, N)
            continue

        sgn = 1 if Q > 0 else -1
        Q_abs = abs(Q)

        # Trial divide
        exp_vec = [0] * fb_size
        rem = Q_abs
        for i in range(fb_size):
            p = fb[i]
            while rem % p == 0:
                rem //= p
                exp_vec[i] += 1

        if rem == 1:
            # Smooth relation: (ax_b)^2 ≡ sgn * ∏p^e (mod N), cofactor=1
            relations.append((ax_b % N, exp_vec, sgn, 1))
        elif rem <= lp_bound:
            # Large prime variation
            if rem in partials:
                ax_b2, ev2, sgn2 = partials[rem]
                combined = [exp_vec[j] + ev2[j] for j in range(fb_size)]
                # Combined: (ax_b1*ax_b2)^2 ≡ sgn*∏p^e * LP^2 (mod N)
                # Store LP so we can include it when computing y
                relations.append(((ax_b * ax_b2) % N, combined, sgn * sgn2, rem))
                del partials[rem]
            else:
                partials[rem] = (ax_b % N, exp_vec, sgn)

    return len(relations) - n_found_before


def _qs_solve(N, relations, fb, fb_size):
    """Find congruence of squares via GF(2) elimination."""
    n_rel = len(relations)
    n_cols = fb_size + 1  # +1 for sign bit

    # Build GF(2) matrix
    mat = []
    for _, ev, sgn, _cofactor in relations:
        row = [e & 1 for e in ev]
        row.append(0 if sgn > 0 else 1)
        mat.append(row)

    # Track row combinations
    history = [set([i]) for i in range(n_rel)]

    # Gaussian elimination
    used = [False] * n_rel
    for col in range(n_cols):
        pivot = -1
        for row in range(n_rel):
            if not used[row] and mat[row][col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        used[pivot] = True
        for row in range(n_rel):
            if row != pivot and mat[row][col] == 1:
                for c in range(n_cols):
                    mat[row][c] ^= mat[pivot][c]
                history[row] = history[row].symmetric_difference(history[pivot])

    # Extract null-space vectors
    for row in range(n_rel):
        if all(mat[row][c] == 0 for c in range(n_cols)):
            idxs = history[row]
            if len(idxs) < 2:
                continue

            x = 1
            total_exp = [0] * fb_size
            y_cofactor = 1  # product of large primes from combined LP relations
            for i in idxs:
                ax_b, ev, sgn, cofactor = relations[i]
                x = x * ax_b % N
                for j in range(fb_size):
                    total_exp[j] += ev[j]
                if cofactor > 1:
                    y_cofactor = y_cofactor * cofactor % N

            if any(e & 1 for e in total_exp):
                continue

            y = y_cofactor  # start with product of large primes
            for j in range(fb_size):
                h = total_exp[j] >> 1
                if h > 0:
                    y = y * pow(fb[j], h, N) % N

            for diff in (x - y, x + y):
                g = math.gcd(diff % N, N)
                if 1 < g < N:
                    return _ret(g, N)

    return None


# ============================================================================
# NOVEL: Multiplicative Orbit Lattice
# ============================================================================

def _novel_orbit_lattice(N, timeout=8.0):
    t0 = time.monotonic()
    try:
        from fpylll import IntegerMatrix, LLL
    except ImportError:
        return None

    bases = [b for b in [2,3,5,7,11,13,17,19,23,29,31,37,41,43] if N % b != 0]
    d = len(bases)
    if d < 4:
        return None

    for B in [100, 500, 2000, 5000, 10000]:
        if time.monotonic() - t0 > timeout:
            break

        M = 1
        for p in _sieve_primes(B):
            pk = p
            while pk <= B:
                M *= p
                pk *= p

        vs = [pow(b, M, N) for b in bases]

        # Direct GCD checks
        for v in vs:
            for delta in (1, -1):
                g = math.gcd(v - delta, N)
                if 1 < g < N:
                    return _ret(g, N)

        # 2-adic squaring descent
        ws = vs[:]
        for step in range(50):
            ws = [(w * w) % N for w in ws]
            for w in ws:
                g = math.gcd(w - 1, N)
                if 1 < g < N:
                    return _ret(g, N)
            # Pairwise
            if step < 5:
                for i in range(len(ws)):
                    for j in range(i+1, len(ws)):
                        g = math.gcd(ws[i] - ws[j], N)
                        if 1 < g < N:
                            return _ret(g, N)

        # Lattice from power residues
        dim = d + 1
        mat = IntegerMatrix(dim, dim)
        for i in range(d):
            mat[i, i] = 1
            mat[i, d] = int(vs[i])
        mat[d, d] = int(N)
        LLL.reduction(mat)

        for i in range(dim):
            e = [int(mat[i, j]) for j in range(d)]
            if all(x == 0 for x in e):
                continue
            prod = 1
            ok = True
            for j in range(d):
                if e[j] == 0: continue
                if e[j] > 0:
                    prod = prod * pow(vs[j], e[j], N) % N
                else:
                    g2 = math.gcd(vs[j], N)
                    if 1 < g2 < N:
                        return _ret(g2, N)
                    if g2 != 1:
                        ok = False; break
                    prod = prod * pow(pow(vs[j], -1, N), -e[j], N) % N
            if not ok: continue
            for delta in (0, 1, -1):
                g = math.gcd((prod + delta) % N, N)
                if 1 < g < N:
                    return _ret(g, N)
    return None


# ============================================================================
# NOVEL: Paillier Lifting
# ============================================================================

def _novel_paillier_lift(N, timeout=8.0):
    t0 = time.monotonic()
    try:
        from fpylll import IntegerMatrix, LLL
    except ImportError:
        return None

    bases = [b for b in [2,3,5,7,11,13,17,19,23,29] if N % b != 0]
    d = len(bases)
    if d < 4:
        return None

    N2 = N * N
    H1 = []
    for a in bases:
        aN = pow(a, N, N2)
        H1.append(aN // N)

    for h in H1:
        if h:
            g = math.gcd(h, N)
            if 1 < g < N:
                return _ret(g, N)

    dim = d + 1
    mat = IntegerMatrix(dim, dim)
    for i in range(d):
        mat[i, i] = 1
        mat[i, d] = int(H1[i] % N)
    mat[d, d] = int(N)
    LLL.reduction(mat)

    for i in range(dim):
        e = [int(mat[i, j]) for j in range(d)]
        if all(x == 0 for x in e):
            continue
        s = sum(e[j] * H1[j] for j in range(d)) % N
        if s:
            g = math.gcd(s, N)
            if 1 < g < N:
                return _ret(g, N)
        prod = 1
        ok = True
        for j in range(d):
            if e[j] == 0: continue
            if e[j] > 0:
                prod = prod * pow(bases[j], e[j], N) % N
            else:
                g2 = math.gcd(bases[j], N)
                if 1 < g2 < N: return _ret(g2, N)
                if g2 != 1: ok = False; break
                prod = prod * pow(pow(bases[j], -1, N), -e[j], N) % N
        if ok:
            for delta in (0, 1, -1):
                g = math.gcd((prod + delta) % N, N)
                if 1 < g < N:
                    return _ret(g, N)

    if time.monotonic() - t0 > timeout:
        return None

    # Level 2
    N3 = N2 * N
    for a in bases:
        if time.monotonic() - t0 > timeout: break
        aN2 = pow(a, N*N, N3)
        h2 = aN2 // N2
        for h1 in H1:
            d2 = (h2 - h1) % N
            if d2:
                g = math.gcd(d2, N)
                if 1 < g < N:
                    return _ret(g, N)
    return None


# ============================================================================
# UTILITIES
# ============================================================================

def _tonelli_shanks(n, p):
    n = n % p
    if n == 0: return 0
    if p == 2: return n % 2
    if pow(n, (p-1)//2, p) != 1: return None
    if p % 4 == 3: return pow(n, (p+1)//4, p)
    q, s = p-1, 0
    while q % 2 == 0: q //= 2; s += 1
    z = 2
    while pow(z, (p-1)//2, p) != p-1: z += 1
    M, c, t, R = s, pow(z,q,p), pow(n,q,p), pow(n,(q+1)//2,p)
    while True:
        if t == 1: return R
        i, tmp = 1, (t*t)%p
        while tmp != 1:
            tmp = (tmp*tmp)%p; i += 1
            if i >= M: return None
        b = c
        for _ in range(M-i-1): b = (b*b)%p
        M, c, t, R = i, (b*b)%p, (t*b*b)%p, (R*b)%p


def _sieve_primes(limit):
    if limit < 2: return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]


def _sympy_factor_timed(N, timeout=55.0):
    """Try sympy's factorint with a time awareness (can't truly timeout, but tries)."""
    try:
        factors = _sympy_factorint(N)
        ps = list(factors.keys())
        if len(ps) >= 2:
            return _ret(int(ps[0]), N)
    except Exception:
        pass
    return None


def _sympy_factor(N):
    factors = _sympy_factorint(N)
    ps = list(factors.keys())
    if len(ps) >= 2:
        return _ret(int(ps[0]), N)
    raise ValueError(f"Failed to factor {N}")
