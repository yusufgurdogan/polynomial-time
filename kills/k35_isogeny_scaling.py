#!/usr/bin/env python3
"""
Scaling study: how does isogeny-factoring rate decay with bit size?

If rate ~ N^{-alpha}, extract alpha. alpha < 0.25 beats Pollard rho, alpha = 0
is polynomial. alpha ~ 0.33 is ECM-class. alpha ~ 0.5 is birthday.

Budget: 200 j-tries per N; 30 instances per bit size; bit sizes 14..44.
"""
import math, random, sys, time
from k35_isogeny import try_isogeny_factor, generate_semiprime
sys.stdout.reconfigure(line_buffering=True)


def run(bits, n_instances, n_tries):
    factors = 0
    total_tries_used = 0
    reasons = {}
    for inst in range(n_instances):
        N, p, q = generate_semiprime(bits)
        hit = False
        for t in range(n_tries):
            j = random.randint(2, N - 2)
            res, reason = try_isogeny_factor(N, j)
            if res:
                factors += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                total_tries_used += t + 1
                hit = True
                break
        if not hit:
            total_tries_used += n_tries
    # Average tries per factor (if factors > 0)
    avg_tries = total_tries_used / max(factors, 1)
    rate_per_try = factors / (n_instances * n_tries) if n_instances * n_tries > 0 else 0
    return factors, n_instances, rate_per_try, avg_tries, reasons


if __name__ == "__main__":
    random.seed(99)
    print("Isogeny-factoring rate scaling\n")
    print(f"  {'bits':>4}  {'factored':>10}  {'rate/try':>12}  {'log2(1/rate)':>14}  reasons")

    data = []
    for bits in [14, 18, 22, 26, 30, 34, 38, 42]:
        f, n, rate, avg, reasons = run(bits, 30, 100)
        if rate > 0:
            log_inv_rate = -math.log2(rate)
        else:
            log_inv_rate = float('inf')
        print(f"  {bits:>4}  {f:>5}/{n:<4}  {rate:>10.5f}  {log_inv_rate:>14.2f}  {reasons}")
        data.append((bits, rate))

    # Fit alpha in rate ~ 2^{-alpha * bits}
    print(f"\nScaling fit: rate ≈ 2^(-alpha * bits)")
    print(f"Pairwise alphas:")
    for i in range(len(data) - 1):
        b1, r1 = data[i]
        b2, r2 = data[i + 1]
        if r1 > 0 and r2 > 0:
            alpha = -(math.log2(r2) - math.log2(r1)) / (b2 - b1)
            print(f"  bits {b1}->{b2}: alpha = {alpha:.3f}")

    print(f"\nReference:")
    print(f"  alpha = 0.00 : polynomial time")
    print(f"  alpha = 0.25 : Pollard rho class")
    print(f"  alpha = 0.33 : ECM class (sub-exp in p)")
    print(f"  alpha = 0.50 : birthday (random gcd)")
