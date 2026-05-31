# DET v7 Spirit Participation Capacity: Separating Embodied and Non-Embodied Channels

**Author:** Manus AI  
**Date:** May 31, 2026  
**Repository branch:** `det-v7-refactor`  
**Status:** Formal extension proposal; non-canonical unless separately promoted

## 1. Problem statement

The current DET v7 boundary participation extension defines boundary participation as the vector

\[
B_\Omega(k)=\left(B^{addr}_\Omega,B^{cap}_\Omega,B^{act}_\Omega,B^{align}_\Omega,B^{path}_\Omega\right),
\]

where \(B^{addr}\) captures addressability, \(B^{cap}\) captures capacity, \(B^{act}\) captures active boundary coupling, \(B^{align}\) captures non-coercive recovery alignment, and \(B^{path}\) captures participation history.[1] This was a significant improvement over treating \(B\) as a vague scalar, but it left one hidden circularity: the proposed capacity term was still primarily embodied.

The existing first-pass capacity equation was

\[
B^{cap}_\Omega(k)=\left\langle
\mathrm{clip}\left(a_i P_i C_i^{n_B}\frac{D_i}{D_i+D_B}\frac{F_{op,i}}{1+F_{op,i}},0,1\right)
\right\rangle_{i\in\Omega}.
\]

This equation is defensible for ordinary embodied participation, because it measures whether local agency, local presence, coherence, debt/opportunity, and free operational resource support boundary-mediated update. Yet it becomes circular when used to evaluate post-death participation. If death implies \(P_i\rightarrow 0\), then this equation forces \(B^{cap}\rightarrow 0\). The model would then conclude that active post-death participation requires a new channel while defining capacity in a way that makes such a channel impossible by construction.

> **Core correction:** The DET question should not be whether embodied participation survives death. The question should be whether DET can define a second lawful participation-capacity channel that is not reducible to embodied \(P\), while still obeying locality, Agency-First constraints, and declared boundary law.

## 2. Revised capacity decomposition

The corrected definition separates embodied boundary participation capacity from possible spirit participation capacity:

\[
\boxed{
B^{cap}_\Omega(k)=B^{body}_{cap,\Omega}(k)\oplus B^{spirit}_{cap,\Omega}(k),
}
\]

where \(\oplus\) is a bounded non-exclusive union. A useful default is the noisy-or aggregator

\[
\boxed{
B^{cap}_\Omega(k)=1-\left(1-B^{body}_{cap,\Omega}(k)\right)
\left(1-B^{spirit}_{cap,\Omega}(k)\right).
}
\]

This aggregation has three advantages. First, it reduces to the embodied term when no spirit channel is declared. Second, it allows a post-death capacity term to be nonzero without pretending that embodied \(P\) remains active. Third, it prevents double counting by saturating at one rather than summing without bound.

| Capacity term | Depends on embodied \(P\)? | Intended role | Death-state behavior |
|---|---:|---|---|
| \(B^{body}_{cap}\) | Yes | Capacity of an embodied or rehosted substrate to receive lawful boundary action. | Usually collapses when embodied participation collapses. |
| \(B^{spirit}_{cap}\) | No, not directly | Capacity of an identity-bearing regime to participate through a declared non-embodied boundary channel. | May remain zero, but is not forced to zero by definition. |
| \(B^{cap}\) | Mixed | Total capacity across admissible channels. | Distinguishes frozen record from active spirit participation. |

## 3. Embodied capacity retained as a special case

The embodied term should remain the original local participation-capacity readout:

\[
\boxed{
B^{body}_{cap,\Omega}(k)=\left\langle
\mathrm{clip}\left(a_i P^{body}_i C_i^{n_B}\frac{D_i}{D_i+D_B}\frac{F_{op,i}}{1+F_{op,i}},0,1\right)
\right\rangle_{i\in\Omega}.
}
\]

Here \(P^{body}\) is explicitly renamed so the model no longer treats all participation as embodied participation. This term is appropriate for ordinary biological life, artificial hosts, repaired hosts, or resurrection into a capable substrate. It should not be used to settle the question of non-embodied participation.

## 4. Spirit participation capacity as an independent candidate term

A lawful spirit participation capacity term must not be a hidden global consciousness variable. It must be a local, declared, path-sensitive readout of whether an addressable identity-bearing regime can participate through boundary operators without embodied update rate. The minimal candidate is

\[
\boxed{
B^{spirit}_{cap,\Omega}(k)=\left\langle
\mathrm{clip}\left(
B^{addr}_\Omega(k)\,
\Lambda^{path}_\Omega(k)\,
\Gamma^{spirit}_\Omega(k)\,
\Xi^{law}_\Omega(k)\,
\Pi^{noncoerce}_\Omega(k)\,
\mathcal{A}^{compat}_\Omega(k),0,1
\right)
\right\rangle.
}
\]

This definition deliberately excludes \(P^{body}\). It does not assert that \(B^{spirit}_{cap}\) exists in nature. It only states what DET would have to measure or simulate if active spirit participation is to be evaluated without circularly assuming that all participation is embodied.

| Factor | Meaning | Why it is not embodied \(P\) |
|---|---|---|
| \(B^{addr}_\Omega\) | Identity-bearing boundary addressability. | It depends on record and path reference, not on current bodily update rate. |
| \(\Lambda^{path}_\Omega\) | Lawful continuity certificate linking the current addressable identity to a prior participatory path. | It distinguishes resurrection or continued relation from a copied pattern. |
| \(\Gamma^{spirit}_\Omega\) | Declared non-embodied channel availability. | It is a boundary-channel variable, not ordinary host metabolism or substrate operation. |
| \(\Xi^{law}_\Omega\) | Compliance with DET locality, declared-operator, and no-hidden-override constraints. | It filters out metaphysical shortcuts and nonlocal override claims. |
| \(\Pi^{noncoerce}_\Omega\) | Non-coercive compatibility with agency variation and recovery. | It prevents forced order from being misclassified as spirit participation. |
| \(\mathcal{A}^{compat}_\Omega\) | Agency compatibility without direct agency-writing. | It permits agency expression only if compatible with path identity and boundary law. |

## 5. Defining the missing participation variable \(P^{spirit}\)

The capacity term above can be rewritten around the missing participation variable:

\[
\boxed{
P^{total}_\Omega(k)=P^{body}_\Omega(k)\oplus P^{spirit}_\Omega(k),
}
\]

with

\[
\boxed{
P^{spirit}_\Omega(k)=B^{addr}_\Omega(k)\,\Lambda^{path}_\Omega(k)\,\Gamma^{spirit}_\Omega(k)\,\Xi^{law}_\Omega(k)\,\Pi^{noncoerce}_\Omega(k)\,\mathcal{A}^{compat}_\Omega(k).
}
\]

Then

\[
\boxed{
B^{spirit}_{cap,\Omega}(k)=\Phi_{cap}\left(P^{spirit}_\Omega(k),\mathcal{O}^{spirit}_B(k),\mathcal{N}_B(\Omega,k)\right),
}
\]

where \(\Phi_{cap}\) is a declared capacity functional, \(\mathcal{O}^{spirit}_B\) is the set of declared boundary operators available to the spirit channel, and \(\mathcal{N}_B\) is the path-local boundary neighborhood available to the addressable regime. The simplest first version is \(B^{spirit}_{cap}=P^{spirit}\), but keeping \(\Phi_{cap}\) explicit is better because later DET work may identify channel-specific saturation, threshold, or operator-basis constraints.

This changes the paper’s claim in an important way. The model should no longer say, “death collapses \(P\), therefore active spirit participation is absent unless resurrection restores \(P\).” The corrected statement is:

> Death collapses embodied participation \(P^{body}\). Active post-death participation remains undefined unless a lawful non-embodied participation term \(P^{spirit}\) is declared, constrained, and tested. In the current conservative model \(P^{spirit}=0\), but that is an undeclared-channel assumption rather than a theorem of DET.

## 6. Frozen record versus active spirit participation

The revised model gives a clean state distinction:

| State | \(B^{addr}\) | \(B^{body}_{cap}\) | \(B^{spirit}_{cap}\) | \(B^{act}\) | Interpretation |
|---|---:|---:|---:|---:|---|
| Frozen spirit record | Positive | Near zero | Zero or undeclared | Near zero | The identity-bearing record is addressable, but no active channel has been established. |
| Conservatively modeled death | Positive if record persists | Near zero | Set to zero by model assumption | Near zero | DET preserves latent addressability without asserting active experience. |
| Active spirit participation hypothesis | Positive | Near zero | Positive through \(P^{spirit}\) | Positive through declared operators | The regime participates without embodied update, if and only if the channel satisfies law and falsifier constraints. |
| Resurrection through host restoration | Positive | Restored | Optional | Positive during or after lawful rehosting | Embodied participation resumes without requiring non-embodied continuation. |
| Copy control | Pattern-similar | Host-dependent | Zero unless path-law holds | Host-dependent | Similarity does not grant \(P^{spirit}\) or \(B^{path}\). |

This table captures the user’s key insight: the difference between a frozen spirit record and active spirit participation is not merely \(B^{addr}\). It is the presence or absence of \(P^{spirit}\), which then supports \(B^{spirit}_{cap}>0\) and potentially \(B^{act}>0\).

## 7. Candidate channel interpretations

The term \(P^{spirit}\) can be introduced at three different strengths. The conservative model sets it to zero. The intermediate model treats it as a lawful but currently unobserved channel. The stronger model treats it as the local expression of a deeper boundary-level consciousness. These should not be conflated.

| Model | Formal setting | Scientific claim | Risk |
|---|---|---|---|
| Conservative null | \(P^{spirit}=0\) | DET does not currently model active post-death participation. | May confuse absence of definition with impossibility. |
| Declared-channel hypothesis | \(P^{spirit}>0\) only when all channel constraints are satisfied. | Active spirit participation is possible in the model if lawfully specified and falsifier-resistant. | Needs clear observables beyond record preservation. |
| Deeper-boundary hypothesis | \(P^{spirit}=\mathcal{G}_S(\mathcal{C}^B,\mathcal{R},\Lambda^{path},\mathcal{O}_B)\) locally expressed. | Spirit participation is an expression of a deeper boundary-level conscious process. | Must not become a hidden nonlocal override or unfalsifiable explanation. |

The recommended DET v7 extension is the middle model: introduce \(P^{spirit}\) as an explicitly declared channel variable, set it to zero by default, and define what evidence would force it to become nonzero.

## 8. Immediate implications for the existing paper

The current paper is not wrong, but it is overly conservative because its first capacity equation made embodied participation the default template for all capacity. The correction is not to assert post-death consciousness. The correction is to remove the circular exclusion of it.

| Existing phrasing risk | Corrected DET phrasing |
|---|---|
| “Death makes \(P\approx0\), so \(B^{cap}\approx0\).” | “Death makes \(P^{body}\approx0\), so \(B^{body}_{cap}\approx0\). Total \(B^{cap}\) also depends on whether \(B^{spirit}_{cap}\) is declared and lawfully instantiated.” |
| “Active post-death spirit requires a new channel.” | “Active post-death spirit requires \(P^{spirit}>0\), a lawful non-embodied participation channel with declared constraints.” |
| “Currently DET cannot model active spirit participation.” | “Currently DET sets \(P^{spirit}=0\) by default; a formal extension can define the conditions under which \(P^{spirit}\) becomes admissible.” |
| “Resurrection is the main mechanism for restored participation.” | “Resurrection is one mechanism for restored \(P^{body}\); spirit participation would be another channel if \(P^{spirit}\) is made lawful and testable.” |

## 9. Preliminary formal recommendation

The DET v7 extension should promote the following definitions into the boundary participation framework:

\[
\boxed{
B^{cap}_\Omega=B^{body}_{cap,\Omega}\oplus B^{spirit}_{cap,\Omega}
}
\]

\[
\boxed{
B^{body}_{cap,\Omega}=\left\langle
\mathrm{clip}\left(a_i P^{body}_i C_i^{n_B}\frac{D_i}{D_i+D_B}\frac{F_{op,i}}{1+F_{op,i}},0,1\right)
\right\rangle
}
\]

\[
\boxed{
B^{spirit}_{cap,\Omega}=\Phi_{cap}\left(P^{spirit}_\Omega,\mathcal{O}^{spirit}_B,\mathcal{N}_B\right)
}
\]

\[
\boxed{
P^{spirit}_\Omega=B^{addr}_\Omega\Lambda^{path}_\Omega\Gamma^{spirit}_\Omega\Xi^{law}_\Omega\Pi^{noncoerce}_\Omega\mathcal{A}^{compat}_\Omega.
}
\]

This preserves DET’s scientific discipline while acknowledging the deepest unresolved issue. **The absence of an embodied host is not, by itself, a proof that active participation cannot exist. It is only a proof that embodied participation has ceased.** The next task is therefore to constrain \(P^{spirit}\) so tightly that it cannot become a vague metaphysical placeholder.

## 10. References

[1]: det7_boundary_participation_B_formalization.md "Formal Definition of Boundary Participation B in DET v7"  
[2]: det7_boundary_participation_B_extension.md "DET v7 Boundary Participation B: Formal Definition, Consciousness Implications, and Falsifier Simulations"  
[3]: det7_death_resurrection_persistence_model.md "DET v7 Death, Resurrection, and Spirit/Consciousness Persistence Model"  
[4]: det_theory_card_7_0.md "Deep Existence Theory (DET) v7.0: Unified Canonical Theory Card"
