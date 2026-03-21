#!/usr/bin/env python3
"""
Think simple. What's the DUMBEST thing that might work?

Every sophisticated approach died. 19 kills. What if the answer
is embarrassingly simple — something a child would try?

Idea 1: MULTI-BASE DIGIT ANALYSIS
  N in base B has digits that are the convolution of p,q digits WITH carries.
  Different bases B produce different carry patterns.
  The carry pattern depends on (p,q).
  Does looking at N in MANY bases simultaneously constrain (p,q)?

Idea 2: MULTIPLICATION TABLE PATTERN
  If you write out the long multiplication p × q in various bases,
  the intermediate partial products are p × q_i for each digit q_i.
  These partial products overlap and add with carries.
  But the DIGIT SUM of N in base B is related to N mod (B-1).
  And digit sums are carry-invariant! S_B(p*q) ≡ S_B(p) * S_B(q) mod (B-1).
  This gives: for every B, N mod (B-1) = (p mod (B-1)) * (q mod (B-1)) mod (B-1).
  This is just CRT again... but wait.

Idea 3: THE SIMPLE CONSTRAINT PROPAGATION
  p * q = N. Both p and q are odd. p < q. p > sqrt(N)/2 (balanced).
  What if we just... set up the multiplication as a constraint satisfaction
  problem in a NON-BINARY way? Not SAT (too slow), not BP (too weak).
  What about Gaussian elimination on the DIGITS of p and q?

  In base B, p = Σ p_i B^i, q = Σ q_j B^j, N = Σ n_k B^k.
  The constraint: Σ_{i+j=k} p_i q_j + carry_k = n_k + B * carry_{k+1}

  If we pick B large enough that most p_i, q_j are 0 or 1,
  the system becomes sparse. Can we solve it?

Idea 4: THE RATIO TRICK
  p/q is a rational number close to some value we can estimate.
  N = p*q, so p = N/q, q = N/p.
  p/q = p²/N. If we could estimate p/q to within 1/q², we'd know p/q exactly
  (by rational reconstruction / continued fractions).
  p/q is between 1/2 and 1 (balanced primes).
  The continued fraction of sqrt(N) gives convergents p_k/q_k with
  |p_k/q_k - sqrt(N)| < 1/q_k². But sqrt(N) ≈ sqrt(p*q), not p/q.

  HOWEVER: p/q = (p/√N)² / (q/√N)² ... hmm.

  What about: √(N/q²) = p/q · √(q²/q²) ... no.

  Actually: the continued fraction of N itself (as a rational!) is trivial.
  But the CF of √N gives SQUFOF-like info.

  What about the CF of N^{1/3}? Or N^{2/3}? Or cube_root(N)?
  These are irrational, and their CFs might have structure related to p,q.

Idea 5: THE BIRTHDAY PARADOX REVISITED
  Pollard's rho uses birthday paradox: random walk in Z/pZ, ~√p steps to collide.
  But what if we use a structured walk that collides FASTER?
  Not random — a walk that's designed to collide when two values are equal mod p.

  The issue: we don't know p. But we know N.
  What walk on Z/NZ has the property that collisions reveal factors?

  Standard: f(x) = x² + c mod N. Collision mod p after ~√p steps.
  Can we do better with a different f?

  What about: f(x) = x^x mod N? Or f(x) = x! mod N? Or f(x) = fibonacci(x) mod N?
  These involve MORE arithmetic, so they might hit special structure faster.

  Actually: x! mod N = 0 for x >= q (since q divides x!). So p | gcd(x! mod N, N)
  for x >= q but x < p... wait, p < q by convention. For x >= p, p | x!, so
  x! mod N ≡ 0 mod p. So gcd(x! mod N, N) = p for p ≤ x < q (if p doesn't divide
  other things). This is just trial division in disguise (factorial method).

Idea 6: BALANCED BINARY REPRESENTATION
  Instead of standard binary, use balanced ternary or balanced binary (digits -1, 0, 1).
  In balanced representation, p and q have roughly equal numbers of +1 and -1 digits.
  The product in balanced representation has DIFFERENT carry structure than binary.
  Does this help? The carries are smaller (bounded by digit count / 2).

Idea 7: THE DOUBLING MAP
  Consider the map T(x) = 2x mod N. Its orbit {x, 2x, 4x, ...} mod N
  has period ord_N(2). If we could find x such that the orbit has period
  dividing (p-1) but not (q-1), we'd factor.

  This is Pollard's p-1 in disguise. But what if we look at the GEOMETRY
  of the orbit? The orbit of x under T is a sequence in Z/NZ.
  By CRT, it's simultaneously an orbit in Z/pZ and Z/qZ.
  The two orbits have different periods. The INTERFERENCE between them
  creates a pattern in Z/NZ that encodes the factorization.

  Can we detect this interference pattern without knowing p or q?

Let's test the ideas that seem most novel.
"""

import math
import random
import sys
import time
import numpy as np
from sympy import nextprime, factorint

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


# ========================================================================
# IDEA 1: Multi-base digit pattern analysis
# ========================================================================

def multibase_analysis(N, p, q, max_base=100):
    """
    For each base B from 2 to max_base:
      - Compute digits of N in base B
      - Compute digits of p, q in base B
      - The digit pattern of N constrains (p,q)

    Key insight: N mod (B-1) = (p mod (B-1)) * (q mod (B-1)) mod (B-1)
    This is carry-invariant.

    But also: the NUMBER of digits of N in base B = floor(log_B(N)) + 1
    And the digit-wise structure (which digits are 0, which are large) depends on carries.

    Can we find a base B where the digit structure reveals something?
    """
    # For each base B, we get the constraint:
    # p mod (B-1) * q mod (B-1) ≡ N mod (B-1) mod (B-1)
    # This is equivalent to: p*q ≡ N mod (B-1), which is trivially true.
    # So digit sums don't help.

    # But what about INDIVIDUAL digits? In base B:
    # N_0 = (p_0 * q_0) mod B  (the lowest digit of N determines p_0*q_0 mod B)
    # N_0 is known. p_0 * q_0 ≡ N_0 mod B.
    # For prime B, this gives q_0 ≡ N_0 * p_0^{-1} mod B.
    # So if we know p mod B (which IS p_0 for single-digit), we know q mod B.

    # The constraint from base B is: p * q ≡ N mod B, i.e., q ≡ N * p^{-1} mod B.
    # This is just CRT. We knew this.

    # NEW IDEA: what about the CARRY from position 0 to position 1?
    # carry_1 = (p_0 * q_0) // B
    # N_1 = (p_0 * q_1 + p_1 * q_0 + carry_1) mod B
    # carry_2 = (p_0 * q_1 + p_1 * q_0 + carry_1) // B
    #
    # In base B = 10:
    # If p = 13, q = 17: p_0=3, p_1=1, q_0=7, q_1=1
    # carry_1 = (3*7)//10 = 2
    # digit_1 = (3*1 + 1*7 + 2) % 10 = 12 % 10 = 2
    # carry_2 = 12 // 10 = 1
    # digit_2 = (1*1 + carry_2) = 2
    # N = 221: digits 1,2,2 ✓

    # The carries are bounded: carry_i ≤ (number of terms contributing to position i) * (B-1)^2 / B
    # For position i with min(i+1, k) terms (k = number of digits in p or q):
    # carry_i ≤ k * (B-1)

    # In a LARGE base (B >> p, q), there are NO carries. N is just the convolution.
    # In a SMALL base (B = 2), carries dominate.
    # Is there a SWEET SPOT?

    # Let's try base B ≈ √p ≈ N^{1/4}. Then p has ~2 digits, q has ~2 digits.
    # N has ~4 digits. The convolution has at most 3 carries.
    # With only 3 carries (each bounded by B), there are ~B^3 ≈ N^{3/4} possibilities.
    # That's BETTER than brute force (N^{1/2}) but still exponential.

    # What about base B ≈ p^{1/k} for larger k?
    # p has k digits, each in [0, B). Product has 2k-1 positions.
    # Carry at position i bounded by (k * B) so carries take values in [0, k*B].
    # Total carry states: (k*B)^{2k} ≈ (k * N^{1/(2k)})^{2k} = k^{2k} * N.
    # That's worse! More digits = more carry ambiguity.

    # CONCLUSION: The optimal base is B ≈ max(p, q), which gives 0 carries
    # but requires knowing the factorization. Every other base has exponential ambiguity.

    return "multibase: sweet spot at B ≈ max(p,q), but that requires knowing the answer"


# ========================================================================
# IDEA 4: CF of N^{1/k} for various k
# ========================================================================

def cf_nth_root(N, p, q, max_terms=10000):
    """
    The continued fraction of √N relates to the infrastructure of Q(√N).
    What about ∛N, ⁴√N, etc.?

    For N = pq: √N = √p · √q.
    ∛N = ∛(pq) = p^{1/3} · q^{1/3}.

    The CF convergents p_k/q_k of √N satisfy |p_k/q_k - √N| < 1/q_k².
    When p_k² - N·q_k² = ±1, we've found the fundamental unit.
    This happens at the CF period, which is O(√N).

    For ∛N: the CF is related to the cubic field Q(∛N)?
    Not directly, but the convergents might hit special values.

    SIMPLE TEST: do the CF convergents of N^{1/k} ever give us
    something GCD-useful with N?
    """
    import mpmath
    mpmath.mp.dps = 50

    results = {}
    for k in [2, 3, 4, 5, 6]:
        root = mpmath.power(N, mpmath.mpf(1) / k)

        # Extract CF expansion
        x = root
        cf_terms = []
        prev_convergents = set()
        factor_found = None

        for i in range(max_terms):
            a = int(mpmath.floor(x))
            cf_terms.append(a)

            # Check convergent
            # Build convergent p_i/q_i
            if i == 0:
                h_prev, h_curr = 1, a
                k_prev, k_curr = 0, 1
            else:
                h_prev, h_curr = h_curr, a * h_curr + h_prev
                k_prev, k_curr = k_curr, a * k_curr + k_prev

            # Check if h_curr or k_curr share a factor with N
            for val in [h_curr, k_curr, h_curr * h_curr - N,
                        h_curr**k - N * k_curr**k if k_curr > 0 and k <= 4 else 0]:
                if val != 0:
                    g = math.gcd(abs(int(val)) % N if abs(int(val)) > N else abs(int(val)), N)
                    if 1 < g < N:
                        factor_found = (k, i, g, 'convergent')
                        break

            if factor_found:
                break

            frac = x - a
            if abs(float(frac)) < 1e-40:
                break
            x = 1 / frac

        results[k] = {
            'terms': len(cf_terms),
            'factor': factor_found,
        }

    return results


# ========================================================================
# IDEA 7: Doubling map orbit interference
# ========================================================================

def doubling_map_interference(N, p, q, num_points=5000, base=2):
    """
    The orbit of x under T(x) = base*x mod N visits points in Z/NZ.
    By CRT: T^k(x) mod p = base^k * x mod p (period divides ord_p(base))
            T^k(x) mod q = base^k * x mod q (period divides ord_q(base))

    The orbit in Z/NZ is the "product" of two orbits with different periods.
    The autocorrelation of the orbit sequence should have peaks at multiples
    of both periods.

    Can we detect these periods from the orbit alone?
    If yes → ord_p(base) → p-1 info → factor.
    """
    x = random.randint(2, N - 1)
    orbit = []
    for _ in range(num_points):
        x = (base * x) % N
        orbit.append(x)

    # Autocorrelation (mod N, so take differences)
    orbit_arr = np.array(orbit, dtype=np.float64)
    # Normalize to [0, 1)
    orbit_norm = orbit_arr / N

    # Simple autocorrelation: R(k) = mean(orbit[i] * orbit[i+k])
    max_lag = min(2000, num_points // 2)
    autocorr = np.zeros(max_lag)
    mean_val = np.mean(orbit_norm)
    var_val = np.var(orbit_norm)
    if var_val < 1e-15:
        return None

    for lag in range(max_lag):
        c = np.mean((orbit_norm[:num_points-lag] - mean_val) *
                     (orbit_norm[lag:] - mean_val))
        autocorr[lag] = c / var_val

    # Find peaks in autocorrelation (excluding lag 0)
    peaks = []
    for i in range(2, max_lag - 1):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            if autocorr[i] > 0.1:  # Significant peak
                peaks.append((i, autocorr[i]))

    peaks.sort(key=lambda x: -x[1])

    # The periods we're looking for
    ord_p = 1
    b = base
    while b != 1:
        b = (b * base) % p
        ord_p += 1
        if ord_p > p:
            break

    ord_q = 1
    b = base
    while b != 1:
        b = (b * base) % q
        ord_q += 1
        if ord_q > q:
            break

    # Check if any peak corresponds to a period
    period_found = False
    for lag, strength in peaks[:10]:
        g = math.gcd(lag, N)
        # Check if lag relates to ord_p or ord_q
        if ord_p > 0 and lag % ord_p == 0:
            period_found = True
        if ord_q > 0 and lag % ord_q == 0:
            period_found = True

    return {
        'top_peaks': peaks[:5],
        'ord_p': ord_p,
        'ord_q': ord_q,
        'period_found': period_found,
        'peak_at_ord': any(abs(lag - ord_p) < 3 or abs(lag - ord_q) < 3
                          for lag, _ in peaks[:20]) if peaks else False,
    }


# ========================================================================
# IDEA 8: THE SIMPLEST POSSIBLE THING — ROUNDING
# ========================================================================

def rounding_attack(N, p, q):
    """
    The SIMPLEST idea: N = p * q. So p = N/q and q = N/p.
    √N is between p and q (for balanced primes).

    What if we just... try rounding √N to nearby "nice" numbers?

    p and q are primes. Primes have specific patterns:
    - They're odd (except 2)
    - They're not divisible by 3 (except 3)
    - They end in 1, 3, 7, or 9 in base 10
    - By Dirichlet, they're equidistributed in residue classes

    For N = p*q with p < q:
    - p < √N < q
    - q = N/p, so if p is close to √N, q is also close to √N
    - The gap: q - p. For balanced primes, q - p can be as large as √N.

    What if we compute round(√N) and then search NEARBY for factors?
    This IS trial division near √N, which takes O(q - p) steps.
    For balanced primes with q - p ~ √N, this is O(N^{1/4}) = SQUFOF-level.

    But here's a twist: what if we round √N to the nearest number
    with specific MODULAR properties? Like: find x near √N such that
    x is 1 mod 6 (possible prime). Then test N mod x.

    This is just trial division with a 3x speedup. Not interesting.

    MORE INTERESTING: the continued fraction of N/floor(√N) or N/(floor(√N)+1).
    N/x for x near √N is near √N from the other side.
    The CF of N/x converges to q when x = p (since N/p = q exactly).
    For x ≈ p, N/x ≈ q + small correction.
    The CF convergents of N/x for x near √N will have q as an early convergent
    when x is close to p.

    But "close to p" means within O(1) of p, which requires knowing p.
    """
    sqrt_N = math.isqrt(N)

    # Try: for various x near √N, compute CF of N/x
    # and check if any convergent divides N
    results = []
    search_radius = min(1000, int(N**0.25))

    for delta in range(-search_radius, search_radius + 1):
        x = sqrt_N + delta
        if x < 2 or x >= N:
            continue
        if N % x == 0:
            results.append(('direct', delta, x))
            break

        # Ratio N/x
        # CF expansion of N/x (rational, finite)
        # Actually N/x is rational only if x divides N.
        # For general x, N/x is irrational... wait, N and x are integers,
        # so N/x is rational (= N/x as a fraction).
        # Its CF terminates. The convergents are h_k/k_k.
        # We want h_k * k_k close to N.

        # Actually, just check gcd(x, N) for each x.
        # This is trial division. Skip.

    return results


# ========================================================================
# IDEA 9: MODULAR SQUARE ROOTS AS FINGERPRINTS
# ========================================================================

def sqrt_fingerprint(N, p, q, num_tests=200):
    """
    For a random a, the equation x² ≡ a mod N has:
    - 0 solutions if a is QNR mod p or QNR mod q
    - 4 solutions if a is QR mod both p and q

    We can't compute sqrt(a) mod N without factoring. But we CAN compute:
    - Jacobi symbol (a/N) — this is 1 for both QR-QR and QNR-QNR cases
    - a^{(N-1)/2} mod N — should be ±1 (Euler criterion for N)

    The Jacobi symbol is computable in poly time and gives:
    (a/N) = (a/p)(a/q).

    If (a/N) = 1: either (a/p)=(a/q)=1 or (a/p)=(a/q)=-1
    If (a/N) = -1: (a/p) ≠ (a/q) — factoring!

    But we can't distinguish (a/p)=1,(a/q)=1 from (a/p)=-1,(a/q)=-1
    using only (a/N). Unless...

    WAIT. (a/N) = -1 means EXACTLY that (a/p) ≠ (a/q).
    If we find such an a, then:
      a^{(p-1)/2} ≡ 1 mod p and a^{(q-1)/2} ≡ -1 mod q (or vice versa)
      So a^{(N-1)/4} (if (N-1)/4 is integer) might give useful info.

    Actually, if we compute a^{(N-1)/2} mod N:
      = a^{(N-1)/2} mod p × a^{(N-1)/2} mod q  (by CRT)

    (N-1)/2 = (pq-1)/2.
    (pq-1)/2 mod (p-1) = ((p-1)(q-1) + (p-1) + (q-1))/2 mod (p-1)
                        = (q-1)/2 mod (p-1)  ... if p is odd

    Hmm, this gets complicated. Let me just test: does a^{(N-1)/2} mod N
    distinguish QR-QR from QNR-QNR cases?
    """
    jacobi_plus_euler_data = []

    for _ in range(num_tests):
        a = random.randint(2, N - 1)
        if math.gcd(a, N) > 1:
            continue

        # Jacobi symbol
        j = jacobi_symbol(a, N)

        # Euler pseudoprime test
        euler = pow(a, (N - 1) // 2, N)

        # True Legendre symbols (using known p, q)
        lp = pow(a, (p - 1) // 2, p)
        lq = pow(a, (q - 1) // 2, q)
        lp = 1 if lp == 1 else -1
        lq = 1 if lq == 1 else -1

        jacobi_plus_euler_data.append({
            'a': a, 'jacobi': j, 'euler': euler,
            'lp': lp, 'lq': lq,
            'case': f"({'+' if lp==1 else '-'},{'+' if lq==1 else '-'})",
        })

    # Analyze: for (a/N) = 1 cases, can euler distinguish (+,+) from (-,-)?
    j1_cases = [d for d in jacobi_plus_euler_data if d['jacobi'] == 1]
    pp_cases = [d for d in j1_cases if d['case'] == '(+,+)']
    mm_cases = [d for d in j1_cases if d['case'] == '(-,-)']

    # Look at euler values
    pp_eulers = set(d['euler'] for d in pp_cases)
    mm_eulers = set(d['euler'] for d in mm_cases)

    return {
        'total': len(jacobi_plus_euler_data),
        'j=1': len(j1_cases),
        'pp_count': len(pp_cases),
        'mm_count': len(mm_cases),
        'pp_euler_values': pp_eulers,
        'mm_euler_values': mm_eulers,
        'euler_distinguishes': pp_eulers != mm_eulers and len(pp_eulers) > 0 and len(mm_eulers) > 0,
    }


def jacobi_symbol(a, n):
    if n <= 0 or n % 2 == 0:
        return 0
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0


# ========================================================================
# IDEA 10: THE MULTIPLICATION TENSOR
# ========================================================================

def multiplication_tensor_idea(N, p, q):
    """
    N = p × q. Think of this as a BILINEAR FORM.

    In binary: N = (Σ p_i 2^i)(Σ q_j 2^j) = Σ p_i q_j 2^{i+j}

    The "multiplication tensor" T_{i,j,k} = [i+j == k] encodes HOW
    the bits interact. The carry structure is what makes this hard.

    But here's a thought: what if we linearize?

    Define variables x_{ij} = p_i * q_j for all (i,j).
    These satisfy:
    1. x_{ij} ∈ {0, 1} (since p_i, q_j ∈ {0,1})
    2. Σ_{i+j=k} x_{ij} + carry_k = N_k + 2 * carry_{k+1}  (the carry equations)
    3. x_{ij} * x_{ij} = x_{ij} (Boolean)
    4. x_{i,j} * x_{i,j'} = x_{i,j} * x_{i,j'} (consistency: p_i is shared)
       Specifically: x_{i,j} * x_{i,j'} = p_i * q_j * p_i * q_{j'} = p_i * q_j * q_{j'}
       And x_{i,j} + x_{i,j'} - x_{i,j}*x_{i,j'} = p_i*(q_j + q_{j'} - q_j*q_{j'})
       = p_i * (1 - (1-q_j)(1-q_{j'}))

    This is a system of quadratic equations over {0,1}.
    It's NP-hard in general (it includes SAT).

    But maybe for MULTIPLICATION specifically, the structure is special?
    The quadratic system has a very specific pattern: rank-1 outer product + carries.

    Can Gröbner bases solve this specific structure faster than general?

    For small instances: yes (but that's just brute force).
    For large instances: unknown, but likely exponential.
    """
    return "multiplication_tensor: reduces to quadratic system, likely NP-hard"


# ========================================================================
# MAIN: Run the testable ideas
# ========================================================================

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print(f"{'='*70}")
    print(f"  SIMPLE IDEAS EXPERIMENT")
    print(f"  Testing the dumbest things that might work")
    print(f"{'='*70}")

    # Test Idea 4: CF of N^{1/k}
    print(f"\n{'='*70}")
    print(f"  IDEA 4: CF of N^{{1/k}} for k = 2, 3, 4, 5, 6")
    print(f"{'='*70}")

    for bits in [16, 20, 24, 28, 32]:
        factored_by = {}
        num_instances = 50
        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)
            results = cf_nth_root(N, p, q, max_terms=5000)
            for k, info in results.items():
                if info['factor']:
                    factored_by[k] = factored_by.get(k, 0) + 1

        print(f"\n  {bits}-bit semiprimes ({num_instances} instances):")
        for k in [2, 3, 4, 5, 6]:
            count = factored_by.get(k, 0)
            print(f"    CF of N^{{1/{k}}}: {count}/{num_instances} factored ({count/num_instances:.0%})")

    # Test Idea 7: Doubling map interference
    print(f"\n{'='*70}")
    print(f"  IDEA 7: Doubling map orbit interference")
    print(f"{'='*70}")

    for bits in [16, 20, 24]:
        detected = 0
        peak_found = 0
        num_instances = 30
        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)
            result = doubling_map_interference(N, p, q, num_points=3000)
            if result:
                if result['period_found']:
                    detected += 1
                if result['peak_at_ord']:
                    peak_found += 1

        print(f"\n  {bits}-bit semiprimes ({num_instances} instances):")
        print(f"    Period in autocorrelation: {detected}/{num_instances}")
        print(f"    Peak near true order: {peak_found}/{num_instances}")

    # Test Idea 9: Sqrt fingerprint
    print(f"\n{'='*70}")
    print(f"  IDEA 9: Euler criterion distinguishes QR cases?")
    print(f"{'='*70}")

    for bits in [16, 20, 24, 28, 32]:
        distinguishes = 0
        num_instances = 50
        for inst in range(num_instances):
            N, p, q = generate_semiprime(bits)
            result = sqrt_fingerprint(N, p, q, num_tests=100)
            if result['euler_distinguishes']:
                distinguishes += 1

        print(f"\n  {bits}-bit semiprimes ({num_instances} instances):")
        print(f"    Euler distinguishes (+,+) from (-,-): {distinguishes}/{num_instances}")

        # Show one example
        N, p, q = generate_semiprime(bits)
        result = sqrt_fingerprint(N, p, q, num_tests=50)
        pp_e = result['pp_euler_values']
        mm_e = result['mm_euler_values']
        print(f"    Example: N={N}, pp_eulers={pp_e}, mm_eulers={mm_e}")

    # Summary of all ideas
    print(f"\n{'='*70}")
    print(f"  SUMMARY OF SIMPLE IDEAS")
    print(f"{'='*70}")
    print(f"""
  Idea 1 (Multi-base): Reduces to CRT. Dead by design.
  Idea 2 (Digit sums): = N mod (B-1). Trivially true. Dead.
  Idea 3 (Constraint propagation): = quadratic system. NP-hard.
  Idea 4 (CF of N^{{1/k}}): Test results above.
  Idea 5 (Birthday paradox variants): = Pollard rho variants.
  Idea 6 (Balanced ternary): Changes carry structure but doesn't eliminate it.
  Idea 7 (Doubling map interference): Test results above.
  Idea 8 (Rounding): = trial division near √N.
  Idea 9 (Euler fingerprint): Test results above.
  Idea 10 (Multiplication tensor): = quadratic system. NP-hard.
    """)
