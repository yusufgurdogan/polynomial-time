#!/usr/bin/env python3
"""
Non-abelian (matrix) arithmetic over Z/NZ for factoring.

Tests whether GL_2(Z/NZ) carries MORE factoring information than scalar Z/NZ.

Four experiments:
  1. Matrix Pollard Rho vs scalar Pollard Rho
  2. Commutator structure — does [A,B] detect factors?
  3. Eigenvalue splitting — discriminant-based factoring
  4. Matrix powering (p-1 analog) vs scalar powering

For semiprimes N = pq at 12, 16, 20, 24 bits, 100 instances each.
"""

import random
import math
import time
from sympy import nextprime, gcd, jacobi_symbol, isprime

# ── Matrix helpers over Z/nZ ──────────────────────────────────────────────

def mat_mul(A, B, n):
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % n,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % n],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % n,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % n],
    ]

def mat_add(A, B, n):
    return [
        [(A[0][0]+B[0][0]) % n, (A[0][1]+B[0][1]) % n],
        [(A[1][0]+B[1][0]) % n, (A[1][1]+B[1][1]) % n],
    ]

def mat_sub(A, B, n):
    return [
        [(A[0][0]-B[0][0]) % n, (A[0][1]-B[0][1]) % n],
        [(A[1][0]-B[1][0]) % n, (A[1][1]-B[1][1]) % n],
    ]

def mat_det(A, n):
    return (A[0][0]*A[1][1] - A[0][1]*A[1][0]) % n

def mat_trace(A, n):
    return (A[0][0] + A[1][1]) % n

def mat_inv(A, n):
    d = mat_det(A, n)
    g = math.gcd(d, n)
    if g != 1:
        return None
    d_inv = pow(d, -1, n)
    return [
        [(A[1][1] * d_inv) % n, ((-A[0][1]) * d_inv) % n],
        [((-A[1][0]) * d_inv) % n, (A[0][0] * d_inv) % n],
    ]

def mat_sq(A, n):
    return mat_mul(A, A, n)

def rand_mat(n):
    return [
        [random.randrange(n), random.randrange(n)],
        [random.randrange(n), random.randrange(n)],
    ]

def rand_gl2(n):
    for _ in range(100):
        M = rand_mat(n)
        if math.gcd(mat_det(M, n), n) == 1:
            return M
    return None

def identity():
    return [[1, 0], [0, 1]]


# ── Semiprime generation ──────────────────────────────────────────────────

def gen_semiprime(bits):
    half = bits // 2
    lo = 1 << (half - 1)
    hi = 1 << half
    while True:
        p = nextprime(random.randrange(lo, hi))
        q = nextprime(random.randrange(lo, hi))
        if p != q:
            N = p * q
            if N.bit_length() >= bits - 1 and N.bit_length() <= bits + 1:
                return int(N), int(p), int(q)


# ── Experiment 1: Matrix Pollard Rho ──────────────────────────────────────

def scalar_pollard_rho(N, max_steps=10000):
    c = random.randrange(2, N)
    x = random.randrange(2, N)
    y = x
    for step in range(1, max_steps + 1):
        x = (x * x + c) % N
        y = (y * y + c) % N
        y = (y * y + c) % N
        d = math.gcd(abs(x - y), N)
        if 1 < d < N:
            return step, d
    return None, None

def matrix_pollard_rho(N, max_steps=10000):
    C = rand_mat(N)
    M = rand_mat(N)
    M2 = [row[:] for row in M]
    for step in range(1, max_steps + 1):
        M = mat_add(mat_sq(M, N), C, N)
        M2 = mat_add(mat_sq(M2, N), C, N)
        M2 = mat_add(mat_sq(M2, N), C, N)
        diff = mat_sub(M, M2, N)
        det_diff = mat_det(diff, N)
        if det_diff == 0:
            for i in range(2):
                for j in range(2):
                    d = math.gcd(diff[i][j], N)
                    if 1 < d < N:
                        return step, d
            continue
        d = math.gcd(det_diff, N)
        if 1 < d < N:
            return step, d
    return None, None


def run_experiment1(bits_list, n_trials=100):
    print("=" * 70)
    print("EXPERIMENT 1: Matrix Pollard Rho vs Scalar Pollard Rho")
    print("=" * 70)
    print(f"{'Bits':>6} | {'Scalar success':>14} {'Avg steps':>10} | {'Matrix success':>14} {'Avg steps':>10}")
    print("-" * 70)

    for bits in bits_list:
        scalar_wins = 0
        scalar_steps_list = []
        matrix_wins = 0
        matrix_steps_list = []

        for _ in range(n_trials):
            N, p, q = gen_semiprime(bits)

            steps_s, factor_s = scalar_pollard_rho(N)
            if factor_s is not None:
                scalar_wins += 1
                scalar_steps_list.append(steps_s)

            steps_m, factor_m = matrix_pollard_rho(N)
            if factor_m is not None:
                matrix_wins += 1
                matrix_steps_list.append(steps_m)

        avg_s = sum(scalar_steps_list) / len(scalar_steps_list) if scalar_steps_list else float('inf')
        avg_m = sum(matrix_steps_list) / len(matrix_steps_list) if matrix_steps_list else float('inf')

        print(f"{bits:>6} | {scalar_wins:>6}/{n_trials:<6} {avg_s:>10.1f} | {matrix_wins:>6}/{n_trials:<6} {avg_m:>10.1f}")

    print()


# ── Experiment 2: Commutator Structure ────────────────────────────────────

def run_experiment2(bits_list, n_trials=100):
    print("=" * 70)
    print("EXPERIMENT 2: Commutator Factor Detection")
    print("=" * 70)
    print("For random A, B in GL_2(Z/NZ), compute [A,B] = ABA^-1 B^-1")
    print("Check gcd(tr([A,B]) - 2, N) and gcd(entries of [A,B]-I, N).")
    print("Compare to scalar Euler criterion: gcd(a^((N-1)/2) - jacobi(a,N), N).")
    print()
    hdr = f"{'Bits':>6} | {'Comm trace':>11} {'Rate':>7} | {'Comm entry':>11} {'Rate':>7} | {'Scalar Euler':>13} {'Rate':>7}"
    print(hdr)
    print("-" * len(hdr))

    n_samples = 50

    for bits in bits_list:
        comm_factor = 0
        scalar_factor = 0
        entry_factor = 0
        total = 0

        for _ in range(n_trials):
            N, p, q = gen_semiprime(bits)

            for __ in range(n_samples):
                total += 1
                A = rand_gl2(N)
                B = rand_gl2(N)
                if A is None or B is None:
                    continue

                A_inv = mat_inv(A, N)
                B_inv = mat_inv(B, N)
                if A_inv is None or B_inv is None:
                    continue

                # Commutator [A, B] = A B A^-1 B^-1
                AB = mat_mul(A, B, N)
                AiB = mat_mul(A_inv, B_inv, N)
                comm = mat_mul(AB, AiB, N)

                tr_comm = mat_trace(comm, N)
                d = math.gcd((tr_comm - 2) % N, N)
                if 1 < d < N:
                    comm_factor += 1

                # Check all entries of comm - I
                I = identity()
                diff = mat_sub(comm, I, N)
                found_entry = False
                for i in range(2):
                    for j in range(2):
                        d2 = math.gcd(diff[i][j], N)
                        if 1 < d2 < N:
                            found_entry = True
                if found_entry:
                    entry_factor += 1

                # Scalar comparison: Euler criterion
                a = random.randrange(2, N)
                if math.gcd(a, N) == 1:
                    js = jacobi_symbol(a, N)
                    half_pow = pow(a, (N - 1) // 2, N)
                    d3 = math.gcd((half_pow - js) % N, N)
                    if 1 < d3 < N:
                        scalar_factor += 1

        rate_c = comm_factor / total if total > 0 else 0
        rate_s = scalar_factor / total if total > 0 else 0
        rate_e = entry_factor / total if total > 0 else 0
        print(f"{bits:>6} | {comm_factor:>5}/{total:<5} {rate_c:>6.4f} | {entry_factor:>5}/{total:<5} {rate_e:>6.4f} | {scalar_factor:>6}/{total:<6} {rate_s:>6.4f}")

    print()


# ── Experiment 3: Eigenvalue Splitting ────────────────────────────────────

def run_experiment3(bits_list, n_trials=100):
    print("=" * 70)
    print("EXPERIMENT 3: Eigenvalue Splitting via Discriminant")
    print("=" * 70)
    print("For random M in GL_2(Z/NZ), D = tr(M)^2 - 4*det(M).")
    print("If (D/p) != (D/q), gcd(D^{(N-1)/2} - jacobi(D,N), N) factors N.")
    print("Compare to scalar Euler criterion on random a.")
    print()
    hdr = f"{'Bits':>6} | {'Matrix D-fac':>13} {'Rate':>7} | {'Scalar Euler':>13} {'Rate':>7} | {'Split cnt':>10}"
    print(hdr)
    print("-" * len(hdr))

    n_samples = 50

    for bits in bits_list:
        mat_factor = 0
        scalar_factor = 0
        split_count = 0
        total = 0

        for _ in range(n_trials):
            N, p, q = gen_semiprime(bits)
            e = (N - 1) // 2

            for __ in range(n_samples):
                total += 1

                M = rand_gl2(N)
                if M is None:
                    continue

                tr_M = mat_trace(M, N)
                det_M = mat_det(M, N)
                delta = (tr_M * tr_M - 4 * det_M) % N

                if math.gcd(delta, N) != 1:
                    d0 = math.gcd(delta, N)
                    if 1 < d0 < N:
                        mat_factor += 1
                else:
                    lp = jacobi_symbol(int(delta % p), int(p))
                    lq = jacobi_symbol(int(delta % q), int(q))
                    if lp != lq:
                        split_count += 1

                    j = jacobi_symbol(int(delta), int(N))
                    dp = pow(int(delta), int(e), int(N))
                    d = math.gcd((dp - j) % N, N)
                    if 1 < d < N:
                        mat_factor += 1

                # Scalar comparison
                a = random.randrange(2, N)
                if math.gcd(a, N) == 1:
                    j_a = jacobi_symbol(a, N)
                    ap = pow(a, int(e), int(N))
                    d_a = math.gcd((ap - j_a) % N, N)
                    if 1 < d_a < N:
                        scalar_factor += 1

        rate_m = mat_factor / total if total > 0 else 0
        rate_s = scalar_factor / total if total > 0 else 0
        print(f"{bits:>6} | {mat_factor:>6}/{total:<6} {rate_m:>6.4f} | {scalar_factor:>6}/{total:<6} {rate_s:>6.4f} | {split_count:>10}")

    print()


# ── Experiment 4 (bonus): Matrix order detection ─────────────────────────

def run_experiment4(bits_list, n_trials=100):
    print("=" * 70)
    print("EXPERIMENT 4 (bonus): Matrix Powering -- Order-Based Factoring")
    print("=" * 70)
    print("Compute M^e mod N for smooth exponent e = lcm(1..B).")
    print("If ord_p(M) | e but ord_q(M) !| e, then M^e = I mod p not mod q.")
    print("Then gcd(entries of M^e - I, N) = p.")
    print()
    print(f"{'Bits':>6} | {'Bound B':>8} | {'Matrix p-1':>11} {'Rate':>7} | {'Scalar p-1':>11} {'Rate':>7}")
    print("-" * 65)

    def mat_pow(M, e, n):
        result = identity()
        base = [row[:] for row in M]
        e = int(e)
        while e > 0:
            if e & 1:
                result = mat_mul(result, base, n)
            base = mat_sq(base, n)
            e >>= 1
        return result

    def smooth_exp(B):
        e = 1
        for pp in range(2, B + 1):
            if isprime(pp):
                pk = pp
                while pk * pp <= B:
                    pk *= pp
                e *= pk
        return e

    for bits in bits_list:
        mat_wins = 0
        scalar_wins = 0
        total = 0

        B = max(20, int(2 ** (bits / 4 + 2)))
        if B > 500:
            B = 500
        e = smooth_exp(B)

        for _ in range(n_trials):
            N, p, q = gen_semiprime(bits)
            n_samples = 20

            for __ in range(n_samples):
                total += 1

                # Matrix power test
                M = rand_gl2(N)
                if M is not None:
                    Me = mat_pow(M, e, N)
                    I = identity()
                    diff = mat_sub(Me, I, N)
                    det_d = mat_det(diff, N)
                    d = math.gcd(det_d, N)
                    found = False
                    if 1 < d < N:
                        mat_wins += 1
                        found = True
                    if not found:
                        for i in range(2):
                            for j in range(2):
                                d2 = math.gcd(diff[i][j], N)
                                if 1 < d2 < N:
                                    mat_wins += 1
                                    found = True
                                    break
                            if found:
                                break

                # Scalar power test (p-1 method analog)
                a = random.randrange(2, N)
                if math.gcd(a, N) == 1:
                    ae = pow(a, int(e), int(N))
                    d_s = math.gcd((ae - 1) % N, N)
                    if 1 < d_s < N:
                        scalar_wins += 1

        rate_m = mat_wins / total if total > 0 else 0
        rate_s = scalar_wins / total if total > 0 else 0
        print(f"{bits:>6} | {B:>8} | {mat_wins:>5}/{total:<5} {rate_m:>6.4f} | {scalar_wins:>5}/{total:<5} {rate_s:>6.4f}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    bits_list = [12, 16, 20, 24]
    n_trials = 100

    print()
    print("=" * 70)
    print("  NON-ABELIAN TRACE EXPERIMENT")
    print("  Does GL_2(Z/NZ) carry more factoring info than Z/NZ*?")
    print("=" * 70)
    print()

    t0 = time.time()

    run_experiment1(bits_list, n_trials)
    run_experiment2(bits_list, n_trials)
    run_experiment3(bits_list, n_trials)
    run_experiment4(bits_list, n_trials)

    elapsed = time.time() - t0

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s")
    print()
    print("Key questions answered:")
    print("  1. Does matrix Pollard rho find factors faster (fewer steps)?")
    print("  2. Do commutators leak factors via trace structure?")
    print("  3. Does eigenvalue splitting (discriminant) beat scalar Euler?")
    print("  4. Does matrix p-1 analog find more factors than scalar p-1?")
    print()
    print("If matrix rates ~ scalar rates: non-abelian structure adds NO info.")
    print("If matrix rates > scalar rates: GL_2 carries EXTRA factoring signal.")
    print()


if __name__ == "__main__":
    main()
