# polynomial-time

A research log of attempts at finding a polynomial-time classical integer factoring algorithm, plus a competition to crowdsource the search.

## Why

In March 2026, George Hotz wrote [*Polynomial Time Factoring Algorithm*](https://geohot.github.io/blog/jekyll/update/2026/03/16/polynomial-time-factoring.html). The argument is short: nobody has proven factoring is hard, quantum factoring is polynomial via Shor, so a classical polynomial algorithm probably exists, and AI is well-suited to find it.

I read the blog four days later and started this repo.

The thesis I am testing: if such an algorithm exists, AI can find it. The corollary, which I did not expect to find as strong as it has turned out: most of the natural directions are blocked by structural barriers that can be stated precisely. Each blocked direction below is a real experiment, not a hand-wave.

## What's here

Three things sit alongside each other:

1. **A barrier taxonomy.** Every approach I have tried lands on at least one of five barriers: CRT blindness (ring operations on Z/NZ are symmetric in p and q), the genus hyperplane obstruction (LLL on multiplicative relation lattices is trapped in the principal genus), birthday/smoothness density bounds, dimensionality lower bounds, and the noise barrier on classical Regev-style dual lattice sampling.
2. **A kill list.** 38 distinct experiments, each one a specific approach with a specific failure mode, indexed below.
3. **[FactorCup](factorcup/).** A scored competition where entries are graded on scaling behavior across bit sizes 32 to 1024. The scorer classifies submissions as POLY, EXP, or SUBEXP via log-log regression on median times. Anti-cheat: os.urandom seeding, lookup-table detection, spawn isolation, parent-side timing. If you build something that looks polynomial, this is where you prove it.

FactorCup is also, accidentally, an AI benchmark. The task is well-specified, the scoring is mechanical, and a polynomial-time entry would be a structurally novel discovery.

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

Each kill has a commit (run `git log --grep='kill #'` for the full set, or look for "KILLED" / "DEAD" in the early ones) and a dedicated file in [`kills/`](kills/). Files are named `kNN_topic.py` where NN is the kill number.

Kills #1 through #11 were not numbered in the original commit messages (the user started counting at #12). They are renumbered chronologically here.

- **#1** [Frobenius ring](kills/k01_novel.py) ([helper](kills/k01_novel2.py)): novel ring methods showed promise but no poly-time path (ebe267b)
- **#2** [cyclotomic + ECM](kills/k02_ultimate.py): combined attack reaches 144 bits but is sub-exp, not poly (1d8015a)
- **#3** [Coppersmith pipeline](kills/k03_coppersmith.py) ([algebraic](kills/k03_algebraic.py)): Coppersmith with algebraic enhancements, no breakthrough (5c008a9)
- **#4** [Schoof mod N](kills/k04_schoof.py): Schoof's algorithm over composite, hits the same CRT-asymmetry wall (bc143cc)
- **#5** [genus geometry](kills/k05_genus_geometry.py): strong negative result, foreshadows the genus hyperplane theorem (cdb663c)
- **#6** [BP on CRT factor graph](kills/k06_bp_crt.py): belief propagation converges independent of N (05459c7)
- **#7** [Direction A: structured bases](kills/k07_structured_bases.py): structured bases do not flip genus parity (87d350b)
- **#8** [Direction B1: cycle position](kills/k08_infra_correlate.py): no poly-time feature correlates with cycle position (f965220)
- **#9** [spectral scout](kills/k09_spectral_scout.py): vanishing signal is a finite-size artifact (d3b16ed)
- **#10** [Direction B2: biquadratic](kills/k10_biquadratic.py): biquadratic lift WORKS but fades, the most interesting early result (6065109)
- **#11** [B2 sweep](kills/k11_biquad_sweep.py): success rate collapses, no property of m predicts success (f8423d2)
- **#12** [neural factoring](kills/k12_neural_factor.py): memorization, no generalization (0fda433)
- **#13** [uncarrying](kills/k13_uncarry.py): carry-removal does not preserve factor info (7b766af)
- **#14** [dual_sample](kills/k14_dual_sample.py): six classical L*/Z^d sampling strategies, all noise ratio 1.0 (39f2b45)
- **#15** [class_number](kills/k15_class_number.py): factoring pipeline works but is circular (39f2b45)
- **#16** [soft_coppersmith](kills/k16_soft_coppersmith.py): n/4-bit barrier universal across info types (39f2b45)
- **#17** [partial_regev](kills/k17_partial_regev.py): smooth relations to noisy dual samples are genus-trapped (39f2b45)
- **#18** [hyperbolic_lattice](kills/k18_hyperbolic_lattice.py): Gaussian/Fermat/Pell lattices, all decay to 0% (39f2b45)
- **#19** [poly_factor_bridge](kills/k19_poly_factor_bridge.py): polynomial factoring vs integer factoring, no new structure (39f2b45)
- **#20** [simple_ideas](kills/k20_simple_ideas.py): 10 back-to-basics approaches, all dead (2693817)
- **#21** [l_function_precision](kills/k21_l_function_precision.py): h·R route dead (b2d68fe)
- **#22** [ecm_coppersmith_bridge](kills/k22_ecm_coppersmith_bridge.py): works but same complexity as ECM (bebdaa8)
- **#23** [gap_geometry](kills/k23_gap_geometry.py): three-distance theorem on mod-exp orbits (f7e7d97)
- **#24** [cf_partial_quotients](kills/k24_cf_partial_quotients.py): statistical signals but no factoring (375ac4c)
- **#25**: carry propagation, midpoint carry grows linearly not uniquely (bf3365d, embedded in modifications)
- **#26** [lattice_recovery](kills/k26_lattice_recovery.py): works at 12 bits, dead by 28 (f1e3550)
- **#27**: PCA on multi-base orbit, torus is not a plane (f1debfb, embedded in modifications)
- **#28** [fermat_quotient](kills/k28_fermat_quotient.py): Z/N²Z liftable rate 0%, exponent problem (de50d6f)
- **#29** [additive_jacobi](kills/k29_additive_jacobi.py): Hadamard decomposition via spectral methods (6ef14e2)
- **#30** [nonabelian_trace](kills/k30_nonabelian_trace.py): GL₂ non-abelian traces (1f9c743)
- **#31** [order_failure_info](kills/k31_order_failure_info.py): Harvey-style order-failure information (1f9c743)
- **#32** [info_scan](kills/k32_info_scan.py): no poly-time function predicts p (8b0c470)
- **#33** [h_lattice](kills/k33_h_lattice_v2.py) ([scaling](kills/k33_h_lattice_scaling.py)): Fermat-quotient lattice bypasses genus but is worse than random (f500959)
- **#34** [binomial](kills/k34_binomial.py): AKS-discrepancy is birthday-class (af2994c)
- **#35** [isogeny](kills/k35_isogeny.py) ([scaling](kills/k35_isogeny_scaling.py)): Φ₂(j, Y) GCD is real signal but sub-exp α ≈ 0.3 (74a71f9)
- **#36** [modpoly_cascade](kills/k36_modpoly_cascade.py): Φ₂/Φ₃ correlated, no compounding (da61f77)
- **#37** [division_poly](kills/k37_division_poly.py): ψ₃ weaker than Φ₂ (30c407a)
- **#38** [miller_rabin](kills/k38_miller_rabin.py): MR sqrt-of-1 requires Fermat liar, dead at scale (ed4a131)

The numbering is not always sequential because earlier exploration was not numbered consistently. Kills #25 and #27 were experiments embedded in modifications to existing files rather than dedicated new files.

## Repo navigation

- `factorcup/`: the competition package, scorer, baselines, entry.
- `paper_draft.md`, `genus_hyperplane_lemma.md`: the structural argument for why LLL on the multiplicative relation lattice cannot factor (the genus hyperplane obstruction).
- [`kills/`](kills/): every numbered kill that has a dedicated file, named `kNN_topic.py`. Run any of them directly (e.g., `python kills/k35_isogeny_scaling.py`).
- Root `.py` files: shared infrastructure (`harness.py`, `baselines.py`, `run.py`) and earlier exploratory experiments that predate the kill numbering.
- `quantum_microscope.py`, `quantum_viz.html`, `quantum_plot.svg`: visualizations of what Regev's quantum sampler produces, for intuition.

If you want to read one thing, read [`paper_draft.md`](paper_draft.md). If you want to compete, go to [`factorcup/`](factorcup/).

## Other people working on this

- [geohot/factoring](https://github.com/geohot/factoring): George Hotz's own repo, started about a month after this one. Currently a 150-bit MPQS implementation plus toy AKS and brute-force order-finding scripts.
- [BigPolarBear1/factorization](https://github.com/BigPolarBear1/factorization): p-adic sieve and binomial-expansion sieving, also chasing polynomial time.

If you are running similar experiments and want to compare notes, open an issue.

## Status

I do not have a polynomial-time algorithm. After 38 kills I have something almost as useful: a sharper picture of where the walls are. Every classical mechanism I have tested lands in one of three regimes: birthday (rate 2/√N), sub-exponential (L[1/2] or L[1/3]), or pathological-input-only (Carmichael, smooth-order primes, etc.). Nothing flat.

If you can produce a single experiment that shows a flat success rate (constant, not decaying with bits) on random balanced semiprimes, you have something genuinely new. Submit to FactorCup.
