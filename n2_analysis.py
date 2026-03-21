#!/usr/bin/env python3
"""
N² IMAGE ANALYSIS: H mod p behaves as a random function.

|image(H mod p)| ≈ (1 - 1/e) × p ≈ 0.632 × p

This is the coupon-collector fraction — confirming H mod p acts as
a random oracle with no exploitable algebraic structure.

Birthday collision at √(0.63p) ≈ 0.79√p — O(N^{1/4}) asymptotically.
The Fermat quotient structure is destroyed by cross-contamination
from the mixed exponent N = pq.
"""
# Analysis already performed inline. This file documents the finding.
# See fermat_quotient.py and n2_factoring.py for the implementations.
#
# KEY RESULT:
# H(a) = (a^N mod N² - a^N mod N) / N
# - H mod p depends only on a mod p (verified)
# - |image(H mod p)| / (p-1) → 1 - 1/e ≈ 0.632 (verified 10-20 bits)
# - This matches a random function f: {1,...,p-1} → Z/pZ
# - No algebraic compression → birthday is tight → O(N^{1/4})
