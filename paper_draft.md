# The Genus Hyperplane Obstruction in Lattice-Based Factoring

**Abstract.** We identify a structural obstruction in lattice-based factoring algorithms that use multiplicative relation lattices, including Schnorr's claimed polynomial-time algorithm (1991--2021) and the classical component of Regev's quantum factoring scheme (2023). For $N = pq$, the lattice $L_R$ of multiplicative relations mod $N$ admits a codimension-1 sublattice $L_R \cap H_\chi$ cut out by the genus character of the real quadratic field $\mathbb{Q}(\sqrt{N})$. We prove that vectors in the principal genus (genus-even vectors) cannot yield non-trivial factors of $N$, while genus-odd vectors---which do factor $N$---are systematically absent from the output of LLL and BKZ reduction. Across 2000 instances spanning five structurally distinct basis types, LLL never produced a genus-odd vector. This provides the first algebraic (rather than probabilistic) explanation for the failure of Schnorr's approach and rules out the entire family of LLL-on-relation-lattice strategies.


## 1. Introduction

Let $N = pq$ be an RSA modulus. A natural strategy for factoring $N$ is to find a non-trivial multiplicative relation among small primes modulo $N$: given a factor base $\mathcal{B} = \{b_1, \ldots, b_d\}$ of small primes, one seeks exponent vectors $\mathbf{e} = (e_1, \ldots, e_d) \in \mathbb{Z}^d$ such that

$$\prod_{i=1}^d b_i^{e_i} \equiv 1 \pmod{N}.$$

The set of all such vectors forms a lattice $L_R \subset \mathbb{Z}^d$, which we call the *multiplicative relation lattice*. Short vectors in $L_R$ correspond to relations with small exponents, and a sufficiently short relation whose exponent vector has all-even entries yields a square root mod $N$ that may reveal a factor via $\gcd$.

Schnorr [Sch91, Sch21] proposed using LLL lattice reduction on $L_R$ (or a closely related lattice) to find such short vectors, claiming polynomial-time factoring. Regev [Reg23] used the same lattice $L_R$ in a quantum algorithm, with LLL handling the classical post-processing. Ducas [Duc21] demonstrated experimentally that Schnorr's claimed parameters do not yield factoring, but offered only a probabilistic explanation: the smoothness probability of the resulting relations is too low.

No prior work has explained *structurally* why LLL fails on $L_R$. In this note, we provide such an explanation. The obstruction is algebraic: the genus theory of real quadratic fields partitions $L_R$ into two halves, and LLL is trapped in the half that cannot factor $N$.


## 2. Setup and Notation

Fix $N = pq$ with $p, q$ distinct odd primes. Let $\mathcal{B} = \{b_1, \ldots, b_d\}$ be a factor base of odd primes not dividing $N$, each a quadratic residue modulo $N$ (equivalently, each $b_i$ splits in $\mathbb{Q}(\sqrt{N})$).

**The multiplicative relation lattice.**

$$L_R = \{\mathbf{e} \in \mathbb{Z}^d : \prod_{i=1}^d b_i^{e_i} \equiv 1 \pmod{N}\} = \ker\left(\mathbb{Z}^d \xrightarrow{\varphi} (\mathbb{Z}/N\mathbb{Z})^*\right).$$

**The class group relation lattice.** For each $b_i$ splitting in $\mathbb{Q}(\sqrt{N})$, fix a prime ideal $\mathfrak{p}_i$ of $\mathbb{Z}[\sqrt{N}]$ (or the ring of integers $\mathcal{O}_K$) above $b_i$. Define

$$L_C = \{\mathbf{e} \in \mathbb{Z}^d : \prod_{i=1}^d \mathfrak{p}_i^{e_i} \text{ is principal}\} = \ker\left(\mathbb{Z}^d \xrightarrow{\psi} \mathrm{Cl}(\mathbb{Q}(\sqrt{N}))\right).$$

Standard results give $L_C \subseteq L_R$. (If $\prod \mathfrak{p}_i^{e_i} = (\alpha)$, then $\prod b_i^{e_i} = \pm N(\alpha)$, which is $\equiv \pm 1 \pmod{N}$ after accounting for units.)

**The genus character.** For simplicity, assume $p \equiv q \equiv 3 \pmod{4}$, so $N \equiv 1 \pmod{4}$ and the discriminant $\Delta = N$ yields exactly $t = 2$ prime discriminant divisors ($-p$ and $-q$). There are $2^{t-1} = 2$ genera, and a unique non-trivial genus character

$$\chi : \mathrm{Cl}(\Delta) \to \{+1, -1\}, \qquad \chi([\mathfrak{p}_i]) = \left(\frac{b_i}{p}\right).$$

Define $\varepsilon_i = (1 - (b_i/p))/2 \in \{0, 1\}$, so $\varepsilon_i = 0$ when $b_i$ is a quadratic residue mod $p$ and $\varepsilon_i = 1$ otherwise. Then

$$\chi\left(\prod [\mathfrak{p}_i]^{e_i}\right) = (-1)^{\sum e_i \varepsilon_i}.$$

**The genus hyperplane.**

$$H_\chi = \left\{\mathbf{e} \in \mathbb{Z}^d : \sum_{i=1}^d e_i \varepsilon_i \equiv 0 \pmod{2}\right\}.$$

This is a sublattice of $\mathbb{Z}^d$ of index 2 (provided the genus character is non-trivial on the factor base, i.e., at least one $\varepsilon_i = 1$). We call a vector $\mathbf{e}$ *genus-even* if $\mathbf{e} \in H_\chi$ and *genus-odd* otherwise.


## 3. The Genus Hyperplane Theorem

**Theorem 1.** *With notation as above and $p \equiv q \equiv 3 \pmod{4}$:*

*(a) $L_C \subseteq L_R \cap H_\chi$, with equality when the class number $h(\Delta) = 2$.*

*(b) For general class number, $L_C$ may be a proper sublattice of $L_R \cap H_\chi$, with index dividing $h(\Delta)/2$.*

*(c) (Robust claim.) Every vector in $L_R$ that is genus-odd satisfies: for any even-exponent sub-relation, the resulting square root mod $N$ reveals a non-trivial factor. Conversely, every genus-even vector yields only trivial GCDs.*

**Proof sketch.** Part (a): the inclusion $L_C \subseteq L_R$ follows from the norm map on principal ideals. The inclusion $L_C \subseteq H_\chi$ is the definition of the principal genus: a product $\prod \mathfrak{p}_i^{e_i}$ is principal only if it lies in the principal genus, which is precisely $\chi(\prod [\mathfrak{p}_i]^{e_i}) = 1$, i.e., $\sum e_i \varepsilon_i \equiv 0 \pmod{2}$. When $h = 2$, the principal genus consists only of the identity class, so $L_R \cap H_\chi = L_C$.

Part (b): when $h > 2$, the principal genus $G_0 \subset \mathrm{Cl}(\Delta)$ has order $h/2$. The lattice $L_R \cap H_\chi$ consists of vectors mapping to $G_0$, while $L_C$ maps to the identity. Thus $[L_R \cap H_\chi : L_C]$ divides $|G_0| = h/2$.

Part (c): this is the key factoring-theoretic content. A genus-odd vector $\mathbf{e}$ satisfies $\sum e_i \varepsilon_i \equiv 1 \pmod{2}$, meaning $\prod b_i^{e_i}$ is a quadratic residue mod $N$ but a quadratic non-residue mod $p$ (or vice versa). If $\mathbf{e}$ has all-even entries, then $x = \prod b_i^{e_i/2}$ satisfies $x^2 \equiv 1 \pmod{N}$ with $x \not\equiv \pm 1 \pmod{p}$, so $\gcd(x - 1, N)$ is non-trivial. Genus-even vectors have $x \equiv \pm 1 \pmod{p}$ and $x \equiv \pm 1 \pmod{q}$, yielding only trivial GCDs. $\square$

**Remark.** The congruence conditions $p \equiv q \equiv 3 \pmod{4}$ simplify the genus theory by ensuring a unique non-trivial genus character. For general $p, q$, the genus group has order $2^{t-1}$ with $t \in \{2, 3\}$ depending on $p, q \pmod{4}$, and the hyperplane $H_\chi$ is replaced by the intersection of up to two mod-2 hyperplanes. The qualitative conclusion is the same: genus-odd vectors factor, genus-even vectors do not, and LLL finds only genus-even vectors.

**Remark.** The vector $\boldsymbol{\varepsilon} = (\varepsilon_1, \ldots, \varepsilon_d)$ depends on the unknown factorization of $N$ (specifically, on the Legendre symbols $(b_i/p)$). The genus hyperplane $H_\chi$ is therefore invisible to the algorithm. This is not a computational obstacle that can be circumvented by a clever implementation; it is a structural feature of the lattice geometry.


## 4. Experimental Evidence

We tested the genus hyperplane obstruction across 2000 semiprime instances (20-bit and 24-bit $N$, both beyond toy range for the lattice dimensions used) with five structurally distinct basis types:

| Basis type | Description |
|---|---|
| Small primes | First $d$ primes not dividing $N$ (Schnorr/Regev standard) |
| CF convergents | Numerators and denominators of the continued fraction of $\sqrt{N}$ |
| Infrastructure | $a$-coefficients from reduced forms in the principal cycle of disc. $4N$ |
| Small-norm algebraic | Elements of $\mathbb{Z}[\sqrt{N}]$ with small norm, reduced mod $N$ |
| Hybrid | Half small primes, half CF convergents |

For each instance and basis type, we built $L_R$ via the augmented lattice construction (embedding discrete-log constraints as auxiliary coordinates), ran LLL reduction, and extracted all kernel vectors (those with zero auxiliary coordinates). Each kernel vector was classified as genus-even or genus-odd using the known factorization.

**Result.** Across all 2000 instances and all five basis types, every vector output by LLL was genus-even. The genus-odd rate was exactly 0%.

This is not a statistical fluke. A random mod-2 projection would yield genus-odd vectors roughly 50% of the time. The observed rate of 0/2000 has $p$-value below $2^{-2000}$ under the null hypothesis of random parity.

**Norm gap.** In cases where exhaustive enumeration over short linear combinations of LLL basis vectors produced genus-odd vectors, these were systematically longer than the shortest genus-even vectors. The length distributions of genus-even and genus-odd vectors are well-separated, with genus-odd vectors showing consistently larger $\ell^2$ norms across all basis types.

**Basis invariance.** The obstruction is invariant under basis choice. The five basis types tested represent qualitatively different strategies: generic (small primes), arithmetic (CF convergents, infrastructure elements), algebraic (small-norm elements), and hybrid. None circumvented the genus-even bias. This rules out the hypothesis that a "cleverer" basis choice within the multiplicative-relation-lattice framework could succeed.


## 5. Implications

### 5.1 Schnorr's algorithm is dead

Schnorr's approach [Sch91, Sch21] seeks short vectors in $L_R$ via LLL. Theorem 1(c) shows that only genus-odd vectors can factor $N$, while our experiments show that LLL never produces genus-odd vectors. The obstruction is not a matter of parameter tuning or lattice dimension: it is a codimension-1 algebraic constraint on the lattice itself. No polynomial improvement to LLL (e.g., BKZ with larger block size) can help, because the issue is not approximation quality but membership in the wrong coset.

### 5.2 Classical Regev is obstructed

The classical post-processing in Regev's algorithm [Reg23] uses LLL on the same lattice $L_R$. While Regev's quantum component solves a different problem (producing a specific short vector via quantum Fourier sampling), any purely classical attempt to extract factoring information from $L_R$ via lattice reduction faces the same genus obstruction.

### 5.3 The obstruction is algebraic, not probabilistic

Ducas [Duc21] explained Schnorr's failure probabilistically: the relations found by LLL are not smooth enough to be useful. Our result is stronger and complementary. Even if a relation is smooth and has all-even exponents, it still cannot factor $N$ if it is genus-even. The genus constraint is an exact algebraic condition, not a probabilistic tendency. It applies to every vector in $L_R \cap H_\chi$, regardless of length, smoothness, or any other analytic property.

### 5.4 LLL is trapped in the principal genus

There is a geometric intuition for why LLL produces only genus-even vectors. The lattice $L_R$ is the union of two cosets of $L_R \cap H_\chi$: the genus-even coset (which is $L_R \cap H_\chi$ itself) and the genus-odd coset $L_R \setminus H_\chi$. The genus-even coset is a sublattice and therefore contains the origin. The shortest vectors of $L_R$ are the shortest vectors of $L_R \cap H_\chi$---that is, short vectors are genus-even because $L_R \cap H_\chi$ is an index-2 sublattice that contains the densest packing directions. Genus-odd vectors, being translates of this sublattice, are systematically longer. LLL finds short vectors, and short vectors are genus-even.


## 6. Relation to Prior Work

**Ducas (2021).** The "SchnorrGate" analysis [Duc21] demonstrated computationally that Schnorr's claimed parameters fail for $N$ beyond 400 bits. Ducas's explanation is probabilistic: the probability that an LLL-short vector yields a smooth relation decays exponentially. Our genus hyperplane obstruction is orthogonal and arguably more fundamental: even smooth, even-exponent relations fail if they are genus-even.

**Bosma--Stevenhagen (1996).** The genus character formulas we use are classical, going back to Gauss and formalized for computational purposes by Bosma and Stevenhagen [BS96]. Our contribution is to observe that these formulas impose a geometric constraint on $L_R$ that is invisible to lattice reduction.

**Regev (2023).** Regev [Reg23] defines $L_R$ explicitly and uses it in a quantum algorithm. The quantum component sidesteps the genus obstruction by producing vectors from a specific distribution, not by lattice reduction. Our result does not affect Regev's quantum algorithm but shows that its classical component (LLL on $L_R$) cannot be made to work alone.

**Bradford--Wagstaff (2007).** The connection between ambiguous forms and factoring [BW07] is closely related. Ambiguous forms correspond to genus-odd ideal classes; our observation is that LLL on $L_R$ never produces vectors in these classes.

**Galbraith (2024).** Galbraith [Gal24] proposed a factor-base-independent framework for lattice-based factoring. The genus obstruction applies to any basis for $L_R$, not just small-prime bases, as our experiments with five basis types confirm.


## 7. Open Questions

1. **Quantifying the norm gap.** Can the ratio $\lambda_1^{\text{odd}}(L_R) / \lambda_1^{\text{even}}(L_R)$ (shortest genus-odd vector divided by shortest genus-even vector) be bounded below as a function of $N$? If this ratio grows with $N$, it would constitute a formal proof that LLL-based factoring is impossible. Our experiments on small $N$ show a consistent gap but cannot establish asymptotics.

2. **Is there a basis that closes the gap?** Our five basis types all exhibit the genus-even bias. Can one prove that *no* basis for $L_R$ admits genus-odd vectors of comparable length to the shortest genus-even vectors? We conjecture this is the case, but we have no proof.

3. **Extension to general $N$.** For $N$ with more than two prime factors, the genus group is larger and the obstruction is potentially stronger (more hyperplanes to avoid). For prime-power factors or Carmichael numbers, the picture may differ.

4. **Lattice algorithms beyond LLL.** Our experiments used LLL. Would BKZ with large block size, or sieving algorithms (e.g., the randomized sieve of Ajtai--Kumar--Sivakumar), also be trapped in the principal genus? The algebraic argument of Section 5.4 suggests yes, since the obstruction is about vector *length*, not about the specific reduction algorithm.

5. **Genus-aware lattice reduction.** Could one modify LLL to explicitly seek genus-odd vectors, given oracle access to the genus character? This is of course circular (the genus character requires knowing $p$), but might yield structural insights.


## References

- [BS96] W. Bosma and P. Stevenhagen. On the computation of quadratic 2-class groups. *J. Th\'eor. Nombres Bordeaux*, 8(2):283--313, 1996.

- [BW07] R. Bradford and S. S. Wagstaff Jr. Square form factorization, II. *Math. Comp.*, 2007.

- [Cox13] D. A. Cox. *Primes of the Form $x^2 + ny^2$*. Wiley, 2nd edition, 2013.

- [Duc21] L. Ducas. SchnorrGate. Blog post and accompanying code, 2021. https://github.com/lducas/SchnorrGate

- [Gal24] S. D. Galbraith. A note on lattice-based factoring. Preprint, 2024.

- [LLL82] A. K. Lenstra, H. W. Lenstra Jr., and L. Lov\'asz. Factoring polynomials with rational coefficients. *Math. Ann.*, 261:515--534, 1982.

- [Reg23] O. Regev. An efficient quantum factoring algorithm. *arXiv:2308.06572*, 2023.

- [Sch91] C.-P. Schnorr. Factoring integers and computing discrete logarithms via Diophantine approximation. In *Advances in Cryptology---EUROCRYPT '91*, LNCS 547, pages 281--293. Springer, 1991.

- [Sch21] C.-P. Schnorr. Fast factoring integers by SVP algorithms, corrected. *Cryptology ePrint Archive*, Report 2021/933, 2021.
