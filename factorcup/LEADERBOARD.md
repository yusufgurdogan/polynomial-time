# FactorCup Leaderboard

Verified scores from `factorcup/score.py` against every standalone factoring function in this repo. Each entry runs in a spawn-isolated subprocess with parent-side timing and `os.urandom`-seeded test cases. The scorer walks bit sizes 32 → 1024 with 5 random balanced semiprimes per size and a 60s per-case timeout, declaring an entry dead at the first bit size where it solves fewer than 3 of 5.

**Columns**
- `max_bits`: largest bit size where at least 3 of 5 cases solved
- `model`: best-fit scaling (POLY, SUBEXP=L[1/3], EXP), chosen by R² on log-log regression
- `R²`: goodness-of-fit for the chosen model
- `poly_k`: exponent under the polynomial fit when meaningful
- `time@max`: median time on the 5 cases at `max_bits`

**Caveat on model selection.** With balanced semiprimes capped at 160 bits in our run, the fits across POLY/SUBEXP/EXP are often within a few percent of each other. Treat the `model` column as suggestive, not authoritative — `max_bits` and `time@max` are the load-bearing numbers.

## Live entries

| # | Algorithm | max_bits | model | R² | poly_k | time@max |
|---|-----------|---------:|:-----:|---:|-------:|---------:|
| 1 | `entry` | 160 | EXP | 0.619 | 2.81 | 32.09s |
| 2 | `k02_ultimate` | 128 | EXP | 0.919 | 3.20 | 19.30s |
| 3 | `ecm_attack` | 112 | EXP | 0.588 | 1.21 | 2.68s |
| 4 | `pollard_rho` | 96 | EXP | 0.679 | 1.99 | 5.07s |
| 5 | `k01b_frobenius_walk` | 64 | POLY | 0.122 | 0.07 | 0.34s |
| 6 | `k01b_multi_ring` | 48 | POLY | — | — | 0.31s |
| 7 | `k01b_iterated_frobenius` | 48 | POLY | — | — | 0.31s |
| 8 | `lattice_partial_info` | 48 | POLY | — | — | 0.32s |
| 9 | `creative_combined` | 48 | POLY | — | — | 0.33s |
| 10 | `k01b_cubic_frobenius` | 48 | POLY | — | — | 0.33s |
| 11 | `k01b_novel2_combined` | 48 | POLY | — | — | 0.33s |
| 12 | `spectral_attack` | 48 | POLY | — | — | 0.34s |
| 13 | `multibase_power_gcd` | 48 | POLY | — | — | 0.35s |
| 14 | `structured_walk_attack` | 48 | POLY | — | — | 0.37s |
| 15 | `combined_attack` | 48 | POLY | — | — | 0.38s |
| 16 | `multi_polynomial_sieve` | 48 | POLY | — | — | 0.42s |
| 17 | `trial_division` | 48 | POLY | — | — | 0.69s |
| 18 | `classical_period_attack` | 32 | POLY | — | — | 0.31s |
| 19 | `modular_constraint_attack` | 32 | POLY | — | — | 0.33s |
| 20 | `k01_coppersmith` | 32 | POLY | — | — | 0.38s |
| 21 | `k01_novel_combined` | 32 | POLY | — | — | 0.42s |
| 22 | `k01_trace_sequence` | 32 | POLY | — | — | 0.42s |
| 23 | `k01_character_matrix` | 32 | POLY | — | — | 0.43s |

## Dead at 32 bits

These functions failed to solve 3 of 5 random 32-bit balanced semiprimes within 60s each. Most are research probes — they target specific structure (Coppersmith hints, partial info, smooth orders) that random semiprimes do not have.

- `cfrac_attack`
- `character_sum_attack`
- `coppersmith_bootstrap`
- `group_structure_attack`
- `k01_algebraic_norm`
- `k01_multi_exponent`
- `k01_polynomial_ring`
- `k01_power_residue`
- `k01b_frobenius_systematic`
- `k03_coppersmith_pipeline`
- `k03_factor_algebraic`
- `k03_factor_algebraic_bitwise`
- `k03_factor_algebraic_modular`
- `quadratic_form_attack`
- `quadratic_sieve`

---

**How to reproduce.** From `factorcup/`: `pip install -r requirements.txt && python run_leaderboard.py`. Runs ~30 minutes wall-clock with 4-way parallelism on an 8-core box.

**Anti-cheat.** Spawn isolation (no shared parent memory), parent-side `time.monotonic` (immune to child clock manipulation), `os.urandom` test-case seeds (unpredictable per run), and a `<50ms at 64+ bits` suspicion flag for impossibly fast solves.
