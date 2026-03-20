#!/usr/bin/env python3
"""Push ECM to find its breaking point."""

import random
import time
random.seed(42)

from harness import generate_semiprime, verify_factors
from creative import ecm_attack

bit_sizes = [64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 320, 384, 512]
timeout = 30.0

print(f"{'bits':>6} | {'result':>8} | {'time':>10} | {'note'}")
print("-" * 55)

for bits in bit_sizes:
    n, p, q = generate_semiprime(bits)

    start = time.time()

    import multiprocessing as mp
    result_q = mp.Queue()
    def worker(n, q):
        try:
            q.put(ecm_attack(n))
        except Exception:
            q.put(None)

    proc = mp.Process(target=worker, args=(n, result_q))
    proc.start()
    proc.join(timeout=timeout)
    elapsed = time.time() - start

    if proc.is_alive():
        proc.kill()
        proc.join()
        print(f"{bits:>6} | {'TIMEOUT':>8} | {elapsed:>9.3f}s | gave up after {timeout}s")
        # If it times out, no point trying larger
        remaining = [b for b in bit_sizes if b > bits]
        for b in remaining:
            print(f"{b:>6} | {'SKIP':>8} |           | (would be slower)")
        break
    else:
        try:
            factors = result_q.get_nowait()
        except:
            factors = None

        if factors and verify_factors(n, factors):
            print(f"{bits:>6} | {'OK':>8} | {elapsed:>9.3f}s | factors: {factors[0]}... × {factors[-1]}...")
        else:
            print(f"{bits:>6} | {'FAIL':>8} | {elapsed:>9.3f}s | returned {factors}")
            # Try 2 more times with different seeds
            retry_ok = False
            for retry in range(2):
                n2, p2, q2 = generate_semiprime(bits)
                proc2 = mp.Process(target=worker, args=(n2, result_q))
                proc2.start()
                proc2.join(timeout=timeout)
                if proc2.is_alive():
                    proc2.kill()
                    proc2.join()
                    continue
                try:
                    f2 = result_q.get_nowait()
                except:
                    f2 = None
                if f2 and verify_factors(n2, f2):
                    retry_ok = True
                    break
            if not retry_ok:
                print(f"       | ECM WALL | hit at {bits} bits — fails consistently")
                remaining = [b for b in bit_sizes if b > bits]
                for b in remaining:
                    print(f"{b:>6} | {'SKIP':>8} |           |")
                break
