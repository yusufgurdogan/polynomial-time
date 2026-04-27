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

Each kill is a commit. Run `git log --grep='kill #'` for the full set, or click through.

Early phase (genus theorem and lattice obstructions):
- **Direction A**: structured bases do not flip genus parity (87d350b)
- **Direction B1**: no poly-time feature correlates with cycle position (f965220)
- **Spectral scout**: vanishing signal is a finite-size artifact (d3b16ed)
- **B2 biquadratic sweep**: success collapses, no property of m predicts success (f8423d2)
- **BP on CRT factor graph**: belief propagation converges independent of N (05459c7)
- **Genus geometry**: strong negative result (cdb663c)

Numbered kills:
- **#12 #13**: uncarrying and neural factoring, both dead (7b766af)
- **#14**: dual_sample, six classical strategies for L*/Z^d sampling all give noise ratio 1.0 (39f2b45)
- **#15**: class_number, factoring pipeline works but is circular (39f2b45)
- **#16**: soft Coppersmith, n/4-bit barrier universal across information types (39f2b45)
- **#17**: partial Regev, smooth relations into noisy dual samples are genus-trapped (39f2b45)
- **#18**: hyperbolic_lattice (Gaussian/Fermat/Pell), all decay to 0% (39f2b45)
- **#19**: poly_factor_bridge, polynomial factoring vs integer factoring, no new structure (39f2b45)
- **#20**: 10 back-to-basics simple ideas, all dead (2693817)
- **#21**: L-function precision, the h·R route is dead (b2d68fe)
- **#22**: ECM-Coppersmith bridge, works but same complexity as ECM (bebdaa8)
- **#23**: gap geometry via three-distance theorem on mod-exp orbits (f7e7d97)
- **#24**: continued fraction partial quotients, statistical signals but no factoring (375ac4c)
- **#25**: carry propagation, midpoint carry grows linearly not uniquely (bf3365d)
- **#26**: lattice recovery from classical orbit, works at 12 bits, dead by 28 (f1e3550)
- **#27**: PCA on multi-base orbit, torus is not a plane (f1debfb)
- **#28**: Fermat quotients in Z/N²Z, liftable rate 0% due to exponent problem (de50d6f)
- **#29**: additive Jacobi matrix, Hadamard decomposition via spectral methods (6ef14e2)
- **#30 #31**: GL₂ non-abelian traces and Harvey-style order-failure information (1f9c743)
- **#32**: information scan, no poly-time function predicts p (8b0c470)
- **#33**: H-lattice (Fermat-quotient lattice), bypasses genus but is worse than random (f500959)
- **#34**: binomial AKS-discrepancy, birthday-class (af2994c)
- **#35**: isogeny j-root-gcd via Φ₂(j, Y), real signal but sub-exp α ≈ 0.3 (74a71f9)
- **#36**: Φ₂/Φ₃ cascade, correlated, no compounding (da61f77)
- **#37**: division polynomial ψ₃, weaker than Φ₂ (30c407a)
- **#38**: Miller-Rabin sqrt-of-1, requires Fermat liar so dead at scale (ed4a131)

The numbering is not always sequential because earlier exploration was not numbered consistently.

## Repo navigation

- `factorcup/`: the competition package, scorer, baselines, entry.
- `paper_draft.md`, `genus_hyperplane_lemma.md`: the structural argument for why LLL on the multiplicative relation lattice cannot factor (the genus hyperplane obstruction).
- Root `.py` files: each one is roughly a kill or a building block. `harness.py` is the shared runner. The numbered-kill files map to the commits above.
- `quantum_microscope.py`, `quantum_viz.html`, `quantum_plot.svg`: visualizations of what Regev's quantum sampler produces, for intuition.

If you want to read one thing, read [`paper_draft.md`](paper_draft.md). If you want to compete, go to [`factorcup/`](factorcup/).

## Other people working on this

- [geohot/factoring](https://github.com/geohot/factoring): George Hotz's own repo, started about a month after this one. Currently a 150-bit MPQS implementation plus toy AKS and brute-force order-finding scripts.
- [BigPolarBear1/factorization](https://github.com/BigPolarBear1/factorization): p-adic sieve and binomial-expansion sieving, also chasing polynomial time.

If you are running similar experiments and want to compare notes, open an issue.

## Status

I do not have a polynomial-time algorithm. After 38 kills I have something almost as useful: a sharper picture of where the walls are. Every classical mechanism I have tested lands in one of three regimes: birthday (rate 2/√N), sub-exponential (L[1/2] or L[1/3]), or pathological-input-only (Carmichael, smooth-order primes, etc.). Nothing flat.

If you can produce a single experiment that shows a flat success rate (constant, not decaying with bits) on random balanced semiprimes, you have something genuinely new. Submit to FactorCup.
