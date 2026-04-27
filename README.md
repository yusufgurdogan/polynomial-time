# polynomial-time

A research log of attempts at finding a polynomial-time classical integer factoring algorithm, plus a competition to crowdsource the search.

## Why

In March 2026, George Hotz wrote [*Polynomial Time Factoring Algorithm*](https://geohot.github.io/blog/jekyll/update/2026/03/16/polynomial-time-factoring.html). The argument is short: nobody has proven factoring is hard, quantum factoring is polynomial via Shor, so a classical polynomial algorithm probably exists, and AI is well-suited to find it.

I read the blog four days later and started this repo.

The thesis I am testing: if such an algorithm exists, AI can find it. Each direction below is a real experiment, not a hand-wave. So far each one has either failed to scale or revealed where a barrier sits, but I have not stopped looking.

## What's here

Three things sit alongside each other:

1. **A barrier taxonomy.** Every approach I have tried lands on at least one of five barriers: CRT blindness (ring operations on Z/NZ are symmetric in p and q), the genus hyperplane obstruction (LLL on multiplicative relation lattices is trapped in the principal genus), birthday/smoothness density bounds, dimensionality lower bounds, and the noise barrier on classical Regev-style dual lattice sampling.
2. **A kill list.** 38 distinct experiments, each one a specific approach with a specific failure mode, indexed below.
3. **[FactorCup](factorcup/).** A scored competition where entries are graded on scaling behavior across bit sizes 32 to 1024. The scorer classifies submissions as POLY, EXP, or SUBEXP via log-log regression on median times. Anti-cheat: os.urandom seeding, lookup-table detection, spawn isolation, parent-side timing. If you build something that looks polynomial, this is where you prove it.

FactorCup is also an AI benchmark. The task is well-specified and the scoring is mechanical.

## FactorCup

See [`factorcup/README.md`](factorcup/README.md) for the full spec.

```bash
cd factorcup
pip install -r requirements.txt
cp example_entry.py entry.py
python test.py
python score.py
```

Current entry in this repo (`factorcup/entry.py`) reaches 160 bits via a tuned escalation pipeline (trial division, Pollard p-1 / Williams p+1, Pollard rho, MPQS, sympy ECM, plus orbit-lattice and Paillier-lift novel attempts).

## The kill list

Each numbered kill has a dedicated file in [`kills/`](kills/) named `kNN_topic.py`. Kills #1 through #11 were not numbered in the original commit messages (counting started at #12); they are renumbered chronologically here.

- **#1** [Frobenius ring](kills/k01_novel.py) ([helper](kills/k01_novel2.py)): novel ring methods showed promise but no poly-time path
- **#2** [cyclotomic + ECM](kills/k02_ultimate.py): combined attack reaches 144 bits but is sub-exp, not poly
- **#3** [Coppersmith pipeline](kills/k03_coppersmith.py) ([algebraic](kills/k03_algebraic.py)): Coppersmith with algebraic enhancements, no breakthrough
- **#4** [Schoof mod N](kills/k04_schoof.py): Schoof's algorithm over composite, hits the same CRT-asymmetry wall
- **#5** [genus geometry](kills/k05_genus_geometry.py): strong negative result, foreshadows the genus hyperplane theorem
- **#6** [BP on CRT factor graph](kills/k06_bp_crt.py): belief propagation converges independent of N
- **#7** [Direction A: structured bases](kills/k07_structured_bases.py): structured bases do not flip genus parity
- **#8** [Direction B1: cycle position](kills/k08_infra_correlate.py): no poly-time feature correlates with cycle position
- **#9** [spectral scout](kills/k09_spectral_scout.py): vanishing signal is a finite-size artifact
- **#10** [Direction B2: biquadratic](kills/k10_biquadratic.py): biquadratic lift WORKS but fades, the most interesting early result
- **#11** [B2 sweep](kills/k11_biquad_sweep.py): success rate collapses, no property of m predicts success
- **#12** [neural factoring](kills/k12_neural_factor.py): memorization, no generalization
- **#13** [uncarrying](kills/k13_uncarry.py): carry-removal does not preserve factor info
- **#14** [dual_sample](kills/k14_dual_sample.py): six classical L*/Z^d sampling strategies, all noise ratio 1.0
- **#15** [class_number](kills/k15_class_number.py): factoring pipeline works but is circular
- **#16** [soft_coppersmith](kills/k16_soft_coppersmith.py): n/4-bit barrier universal across info types
- **#17** [partial_regev](kills/k17_partial_regev.py): smooth relations to noisy dual samples are genus-trapped
- **#18** [hyperbolic_lattice](kills/k18_hyperbolic_lattice.py): Gaussian/Fermat/Pell lattices, all decay to 0%
- **#19** [poly_factor_bridge](kills/k19_poly_factor_bridge.py): polynomial factoring vs integer factoring, no new structure
- **#20** [simple_ideas](kills/k20_simple_ideas.py): 10 back-to-basics approaches, all dead
- **#21** [l_function_precision](kills/k21_l_function_precision.py): h·R route dead
- **#22** [ecm_coppersmith_bridge](kills/k22_ecm_coppersmith_bridge.py): works but same complexity as ECM
- **#23** [gap_geometry](kills/k23_gap_geometry.py): three-distance theorem on mod-exp orbits
- **#24** [cf_partial_quotients](kills/k24_cf_partial_quotients.py): statistical signals but no factoring
- **#25** [carry propagation](https://github.com/yusufgurdogan/polynomial-time/commit/bf3365d): midpoint carry grows linearly not uniquely (no dedicated file, lived inside modifications)
- **#26** [lattice_recovery](kills/k26_lattice_recovery.py): works at 12 bits, dead by 28
- **#27** [PCA on multi-base orbit](https://github.com/yusufgurdogan/polynomial-time/commit/f1debfb): torus is not a plane (no dedicated file, lived inside modifications)
- **#28** [fermat_quotient](kills/k28_fermat_quotient.py): Z/N²Z liftable rate 0%, exponent problem
- **#29** [additive_jacobi](kills/k29_additive_jacobi.py): Hadamard decomposition via spectral methods
- **#30** [nonabelian_trace](kills/k30_nonabelian_trace.py): GL₂ non-abelian traces
- **#31** [order_failure_info](kills/k31_order_failure_info.py): Harvey-style order-failure information
- **#32** [info_scan](kills/k32_info_scan.py): no poly-time function predicts p
- **#33** [h_lattice](kills/k33_h_lattice_v2.py) ([scaling](kills/k33_h_lattice_scaling.py)): Fermat-quotient lattice bypasses genus but is worse than random
- **#34** [binomial](kills/k34_binomial.py): AKS-discrepancy is birthday-class
- **#35** [isogeny](kills/k35_isogeny.py) ([scaling](kills/k35_isogeny_scaling.py)): Φ₂(j, Y) GCD is real signal but sub-exp α ≈ 0.3
- **#36** [modpoly_cascade](kills/k36_modpoly_cascade.py): Φ₂/Φ₃ correlated, no compounding
- **#37** [division_poly](kills/k37_division_poly.py): ψ₃ weaker than Φ₂
- **#38** [miller_rabin](kills/k38_miller_rabin.py): MR sqrt-of-1 requires Fermat liar, dead at scale

## Repo navigation

- [`factorcup/`](factorcup/): the competition package, scorer, baselines, entry.
- [`kills/`](kills/): every numbered kill, named `kNN_topic.py`. Run any of them directly (e.g., `python kills/k35_isogeny_scaling.py`).
- [`paper_draft.md`](paper_draft.md), [`genus_hyperplane_lemma.md`](genus_hyperplane_lemma.md): the structural argument for why LLL on the multiplicative relation lattice cannot factor (the genus hyperplane obstruction).
- [`harness.py`](harness.py), [`baselines.py`](baselines.py), [`run.py`](run.py): shared infrastructure and runners.
- [`quantum_microscope.py`](quantum_microscope.py), [`quantum_viz.html`](quantum_viz.html), [`quantum_plot.svg`](quantum_plot.svg): visualizations of what Regev's quantum sampler produces.

## Other people working on this

- [geohot/factoring](https://github.com/geohot/factoring): George Hotz's own repo, started about a month after this one. Currently a 150-bit MPQS implementation plus toy AKS and brute-force order-finding scripts.
- [BigPolarBear1/factorization](https://github.com/BigPolarBear1/factorization): p-adic sieve and binomial-expansion sieving, also chasing polynomial time.

If you are running similar experiments and want to compare notes, open an issue.

## Status

No polynomial-time algorithm yet. The search continues. If you have one, submit to FactorCup.
