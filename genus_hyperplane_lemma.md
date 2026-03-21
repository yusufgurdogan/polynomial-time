# The Genus Hyperplane Lemma

## Draft — for verification and refinement

### Setup

Let N = pq with p, q distinct odd primes, p < q.
Let Δ = 4N (the discriminant of the order Z[√N] in Q(√N)).

Let b₁, ..., b_d be distinct odd primes not dividing N, each splitting in Q(√N)
(equivalently, (Δ/bᵢ) = 1, i.e., N is a quadratic residue mod bᵢ).

**Regev's lattice:**
$$L_R = \{(e_1, \ldots, e_d) \in \mathbb{Z}^d : \prod_{i=1}^d b_i^{e_i} \equiv 1 \pmod{N}\}$$

This is the kernel of the group homomorphism
φ: Z^d → (Z/NZ)*, (e₁,...,e_d) ↦ ∏ bᵢ^{eᵢ} mod N.

**Class group lattice:**
For each bᵢ that splits in Q(√N), let 𝔭ᵢ be a prime ideal of Z[√N] above bᵢ.
Define:
$$L_C = \{(e_1, \ldots, e_d) \in \mathbb{Z}^d : \prod_{i=1}^d \mathfrak{p}_i^{e_i} \text{ is principal in } \mathrm{Cl}(\mathbb{Q}(\sqrt{N}))\}$$

This is the kernel of the class group map
ψ: Z^d → Cl(Q(√N)), (e₁,...,e_d) ↦ ∏ [𝔭ᵢ]^{eᵢ}.

### The genus character

For Δ = 4pq, the discriminant has two odd prime divisors (p and q) plus the
factor of 4. By genus theory (Gauss), the group of genus characters has order
2^(t-1) where t is the number of prime discriminant divisors of Δ.

For Δ = 4pq:
- If p ≡ q ≡ 1 (mod 4): Δ = 4pq, prime discriminant divisors are {-4, p*, q*}
  where p* = (-1)^{(p-1)/2} p, giving t = 3 and 4 genera.
- If p ≡ 3 (mod 4) or q ≡ 3 (mod 4): the count of independent genus characters
  may be 1 or 2 depending on exact congruences mod 4 and mod 8.

**Simplest case:** p ≡ q ≡ 3 (mod 4).
Then N ≡ 1 (mod 4), and we can take discriminant Δ = N instead of 4N.
The prime discriminant divisors are p* = -p and q* = -q (since p, q ≡ 3 mod 4).
There are t = 2 prime discriminant divisors, giving 2^(t-1) = 2 genera.

The unique non-trivial genus character is:
$$\chi: \mathrm{Cl}(\Delta) \to \{+1, -1\}$$
defined on a form class represented by (a, b, c) with gcd(a, Δ) = 1 by:
$$\chi([a, b, c]) = \left(\frac{a}{p}\right) = \left(\frac{a}{q}\right)$$
(The last equality holds because (a/p)(a/q) = (a/N) = (a/Δ) for forms of
discriminant Δ, and this Jacobi symbol equals 1 for represented values.)

### The encoding

Define εᵢ ∈ {0, 1} by:
$$\varepsilon_i = \frac{1 - (b_i / p)}{2}$$
where (bᵢ/p) is the Legendre symbol. So εᵢ = 0 if bᵢ is a QR mod p,
and εᵢ = 1 if bᵢ is a QNR mod p.

The genus character evaluated on the ideal class [𝔭ᵢ] is:
$$\chi([\mathfrak{p}_i]) = (b_i / p) = (-1)^{\varepsilon_i}$$

For a product of ideal classes:
$$\chi\left(\prod [\mathfrak{p}_i]^{e_i}\right) = \prod (b_i/p)^{e_i} = (-1)^{\sum e_i \varepsilon_i}$$

### The Lemma

**Lemma (Genus Hyperplane).** With notation as above, and assuming p ≡ q ≡ 3 (mod 4):

(a) L_C ⊆ L_R.

(b) L_C = L_R ∩ H_χ, where H_χ = {e ∈ Z^d : Σᵢ eᵢ εᵢ ≡ 0 (mod 2)}.

(c) If the genus character χ is non-trivial on Im(ψ) (i.e., some [𝔭ᵢ] is
    in the non-principal genus), then [L_R : L_C] = 2.

**Proof sketch.**

(a) If ∏ 𝔭ᵢ^{eᵢ} is principal, say = (α), then the norm N(α) = ∏ bᵢ^{eᵢ}
    (up to sign and units). Since α ∈ Q(√N), its norm lies in Z, and
    reducing mod N shows ∏ bᵢ^{eᵢ} ≡ ±N(α)²/... ≡ ... mod N.

    More precisely: if ∏ 𝔭ᵢ^{eᵢ} = (α), then ∏ 𝔭̄ᵢ^{eᵢ} = (ᾱ),
    and ∏ (bᵢ)^{eᵢ} = ∏ (𝔭ᵢ 𝔭̄ᵢ)^{eᵢ} = (α)(ᾱ) = (αᾱ) = (N(α)).
    So ∏ bᵢ^{eᵢ} = N(α) · u for some unit u ∈ Z[√N]*.
    In (Z/NZ)*, N(α) ≡ αᾱ mod N, which is well-defined.
    Actually, N(α) is a rational integer, and ∏ bᵢ^{eᵢ} = ±N(α),
    so ∏ bᵢ^{eᵢ} ≡ ±N(α) ≡ ... ≡ 1 mod N?

    [This step needs more care — the map from principal ideals to
    multiplicative relations mod N involves the unit group and signs.
    The inclusion L_C ⊆ L_R should follow from the surjection
    (Z/NZ)* → Cl(Q(√N)) → Cl(Q(√N))/2Cl having the right kernel.]

(b) The key claim. An element e ∈ L_R satisfies ∏ bᵢ^{eᵢ} ≡ 1 mod N.
    The product ideal ∏ 𝔭ᵢ^{eᵢ} maps to the trivial class in Cl iff
    it's principal. It maps to the principal GENUS iff χ(∏ [𝔭ᵢ]^{eᵢ}) = 1,
    i.e., iff Σ eᵢ εᵢ ≡ 0 mod 2.

    We need: e ∈ L_R and Σ eᵢ εᵢ ≡ 0 mod 2 ⟹ e ∈ L_C.

    This would mean: a multiplicative relation mod N whose ideal-class
    image is in the principal genus must actually be principal.
    This is NOT automatic — the principal genus contains non-principal classes
    (unless h = 2, which would mean every genus-even relation is principal).

    **CAVEAT**: This step likely requires h(Δ) = 2 (class number 2),
    which is NOT true in general. For general class number, L_C is a
    proper sublattice of L_R ∩ H_χ.

    For the EXPERIMENTAL observation (all short L_R vectors are genus-even),
    the explanation might be: short vectors correspond to relations with
    small exponents, and such relations tend to land in the principal class
    (not just the principal genus) because the class group element has
    small "height" in some sense.

(c) Follows from (b) when the genus character is surjective onto {±1}
    from the factor base.

### Corrected Statement

The experiments actually show something slightly different and potentially
stronger than the clean L_C = L_R ∩ H_χ:

**Observation**: For all test cases (N = pq, 7 ≤ N ≤ 9797), every vector
in L_R found by LLL (i.e., every SHORT vector) satisfies Σ eᵢ εᵢ ≡ 0 mod 2.

This could mean:
(A) L_C = L_R ∩ H_χ exactly (clean index-2 decomposition), OR
(B) Short vectors in L_R happen to be genus-even, while genus-odd vectors
    exist but are longer (the "geometric" explanation), OR
(C) For these small N, L_R ∩ H_χ = L_R (genus character is trivial on L_R),
    meaning ALL relations happen to be genus-even.

The experiments couldn't distinguish (A) from (B) from (C) because we never
found a genus-odd vector at all. Distinguishing requires:
- Finding the SHORTEST genus-odd vector and comparing to shortest genus-even
- Computing the actual index [L_R : L_C] and [L_R ∩ H_χ : L_C]
- Testing with larger N where more relations are available

### What Needs to Be Done

1. **Verify part (a)** carefully: the map from principal ideals to multiplicative
   relations mod N, accounting for units and signs.

2. **Determine if (b) holds exactly or only approximately**: is L_C = L_R ∩ H_χ,
   or is L_C a proper sublattice of L_R ∩ H_χ with index = class number / 2?

3. **Specify the exact hypotheses**: which congruence conditions on p, q mod 4
   are needed, and how many independent genus characters exist for each case.

4. **Connect to Schnorr's failure**: if LLL finds only genus-even vectors,
   and genus-odd vectors factor N, then genus theory explains why Schnorr's
   approach cannot work with LLL alone — regardless of lattice parameters.

### References

- Bosma, Stevenhagen: "On the computation of quadratic 2-class groups" (JTNB 1996)
- Kopp, Lagarias: principal genus theorem restatement
- Regev: "An Efficient Quantum Factoring Algorithm" (arXiv:2308.06572)
- Bradford, Wagstaff: "Square Form Factorization, II"
- Gauss: Disquisitiones Arithmeticae, §228–§292 (genus theory)
- Cox: "Primes of the Form x² + ny²" (genus theory exposition)
