# Method

## Problem

A user `u` with training history `H_u` receives a ranked top-K list `L_u = (j_1, …, j_K)`. The user (or an
operator acting for the user) asks to remove one recommended item `t ∈ L_u`. EPIC-Rec answers the question

> *Which of the user's own interactions would have to be absent for `t` to leave the list — while the rest of
> the list and the reasons for it remain intact?*

A **mask** `M ⊆ D_u(t) ⊆ H_u` is a set of the user's own interactions that are removed from the scoring input.
Nothing else is changed: no other user's data, no knowledge-graph edge, no model weight. EPIC-Rec looks for the
**smallest** mask that satisfies the requested constraints.

| constraint | meaning |
| --- | --- |
| **C1** target removal | `t ∉ top-K(s_M)` |
| **C2** list preservation | every other item of `L_u` stays in `top-K(s_M)` (one outsider takes the freed slot) |
| **C3** evidence preservation | every retained item keeps at least one *qualifying, faithful* evidence path |

| setting | constraints | config name |
| --- | --- | --- |
| A | C1 | `A` |
| B | C1 + C2 | `B` |
| C | C1 + C2 + C3 | `C` |
| ablation "without C2" | C1 + C3 (every original non-target item keeps evidence) | `A3` |

The settings are nested: every C mask is a B mask and every B mask is an A mask, so minimum sizes satisfy
`|M*_A| ≤ |M*_B| ≤ |M*_C|`, and infeasibility propagates upwards.

## Path-additive recommender

`epic_rec.models.PathAdditiveRecommender` scores candidates by

```
s(u, j) = β · log(1 + pop_j) + (1 / |H_u|) · Σ_{i ∈ H_u} W[i, j]
W = Σ_r θ_r · S_r + θ_co · S_co
S_r[i, j]  = Σ_{e : i -r- e -r- j} 1 / deg_r(e)                 (knowledge-graph paths)
S_co[i, j] = Σ_{v ≠ u : v interacted with i and j} 1 / sqrt(pop_i · pop_j)   (collaborative paths)
```

Every term of `W[i, j]` is one **path** `p = (i, r, x, j)` with origin `i ∈ H_u`, relation `r`, intermediate node
`x` (an entity or another user) and contribution `c(p) = θ_r / (deg_r(x) · |H_u|)` (KG) or
`θ_co / (sqrt(pop_i · pop_j) · |H_u|)` (collaborative). Hence

* the score **decomposes exactly** into the popularity term plus the sum of path contributions
  (`score_from_paths` recomputes it path by path; the tests require agreement to `1e-9`);
* masking interaction `i` removes exactly the paths that start at `i`: `s_M = s − Σ_{i ∈ M} W[i] / |H_u|`, with the
  normaliser `|H_u|` frozen, i.e. the explanation refers to the deployed model, not to a retrained one;
* masking never raises a score.

The non-negative weights `θ` and `β` are fitted on the training split with a leave-one-out BPR objective
(L-BFGS-B): for a held-out positive `p` of user `u`, the co-occurrence and popularity statistics are corrected for
`p` (`co(i, p) − 1`, `pop_p − 1`) so that the model cannot see the pair it is scored on. Validation and test
interactions are never used to fit anything. Accuracy (Recall@K, NDCG@K) is reported against a popularity
baseline, ItemKNN (neighbourhood size chosen on validation) and BPR matrix factorisation (epochs chosen on
validation by early stopping), see `epic-rec train`.

## Ranking rule and margin

* **P (exact ranking):** candidates (items not in `H_u`) are ordered by masked score, descending, ties broken by
  item id, ascending. This is the rule used everywhere a list is produced.
* **P_ε (margin problem):** every ordering a setting needs must hold with a gap of at least `ε` (default
  `1e-6`, configurable: `experiment.eps`). Every P_ε-feasible mask is P-feasible; the converse fails only when a
  mask relies on a gap smaller than `ε` or on an exact tie.

The exact solver solves P_ε, and **all methods are held to the same problem**: a mask is a success only if the
independent verifier confirms the requested constraints under both P and P_ε (`verification.passed`). The
P-only result (`passed_P`) is stored so that the difference between the two semantics can be quantified
(`p_vs_p_eps.csv`; enumeration on small domains with `experiment.enumerate_max`).

## Mask domain

`D_u(t) = {i ∈ H_u : A[i, t] > 0}`, the interactions that contribute to the target's score (`scope: target`,
default). Masking an interaction outside `D_u(t)` cannot lower `s(t)` and can only lower other items, so it
never helps C1 and can only hurt C2; `scope: history` lets the solver use all of `H_u` instead.

## Evidence (C3)

For retained item `j`, let `P[i, j]` be the largest single-path contribution from origin `i` to `j`, and

```
τ_j = max(α · max_i P[i, j], φ),        Q_j = {i ∈ H_u : P[i, j] ≥ τ_j}.
```

A path from an origin in `Q_j` with contribution `≥ τ_j` is **qualifying**. It is **faithful** if disabling it
lowers the recomputed score of `j` by at least `φ` and by a positive amount. C3 holds when every retained item
keeps at least one qualifying faithful path after masking, i.e. some origin in `Q_j` is unmasked. `α` (relative
strength, default 0.5) and `φ` (absolute floor, default 0) are configurable (`experiment.alpha`,
`experiment.phi`). The evidence unit is the individual path, not the interaction. A retained item without any
qualifying path ("evidence-free") makes C3 unsatisfiable; such cases are reported as their own infeasibility
cause.

## Exact solver

`epic_rec.optimization.solve_milp` (HiGHS, `highspy==1.15.1`). Binary `m_i` for `i ∈ D_u(t)`; masked scores are
linear in `m`, so every required ordering is a linear row.

* **A / A3:** binary `y_c` per candidate that can end above `t`; `y_c = 1` forces `s_M(c) ≥ s_M(t) + ε`;
  `Σ y_c ≥ K − (#candidates above t under every mask)`.
* **B / C:** exactly one replacement outsider `r` (`Σ z_r = 1`) ends above `t`; every retained `j` ends above `t`
  and above every other outsider `c` (relaxed by big-M when `c` is the replacement).
* **C3:** `Σ_{i ∈ Q_j} (1 − m_i) ≥ 1` for every retained `j` whose qualifying origins all lie in `D_u(t)`.
* **Constraint generation.** The "retained above outsider" rows are dense; the first model contains them for the
  `THREAT_INIT = 40` highest-scoring outsiders; violated outsiders are added and the model is re-solved. Every
  intermediate model is a relaxation, so its infeasibility proves infeasibility and its optimum is a lower bound.
* **Replacement pruning.** An outsider `r` can be the replacement only if `s(r) ≥ max_c lb(c)`, with `lb(c)` the
  lowest possible masked score of `c` (proof in the module docstring); tests compare pruning on and off.
* **Outcomes:** `optimal` (verified mask with matching lower bound), `feasible` (verified mask, optimality not
  proven within the time limit; bound and gap reported), `infeasible` (proven for P_ε), `timeout` (neither).
  Each solve runs in a forked child process with a hard wall-clock limit (`run_isolated`); a killed solve is
  recorded as `timeout` with `killed = true`.
* Masks returned by HiGHS are re-checked in NumPy (gaps `≥ ε − 1e-9`, the solver's feasibility tolerance) and
  then verified independently.
* **Cross-check** (`epic_rec.optimization.certify`; the field is still named `certification`). Optimality and infeasibility claims rest on a single
  branch-and-bound run, and on one MovieLens pair the study configuration (presolve off for B / C, feasibility
  tolerances 1e-9) reported a non-minimal mask as optimal. Every claim is therefore re-solved under a second HiGHS
  configuration (presolve on, primal tolerance 1e-7, MIP tolerance 1e-6), 60 s per solve, on the model that would
  refute it: an optimum of size k is re-solved with `|M| ≤ k − 1`, an infeasibility claim without a size cap.
  Infeasible → `confirmed`. A returned mask replaces the result only after passing the independent verifier, and the
  new size is cross-checked again (`corrected`). Anything else → `undecided`, and the result is downgraded (optimum to
  `feasible`, infeasibility to `timeout`). Size-0 optima and infeasibility found by the structural pre-checks
  involve no search (`structural`). The problem definitions are unchanged. Two configurations of HiGHS are not two
  solvers and are not a formal optimality certificate. `epic-rec certify` applies the same procedure to stored records.

For infeasible instances, `infeasibility_report` names the cause (C1 unreachable, list entanglement, evidence
entanglement, evidence-free item, joint conflict) by solving the weaker settings, and lists certificates, the
retained items whose evidence is blocked by the mask domain, and entanglement statistics (how many of the target's
supporting interactions also carry the evidence of retained items).

## Approximate explainers

All approximate methods search under P_ε and return a mask that is then verified exactly like the MILP's.

| method id | description |
| --- | --- |
| `greedy:violation` | adds the interaction that most reduces a hinge surrogate of the constraint violation |
| `greedy:margin` | adds the interaction maximising `A[i, t] − A[i, r]`, `r` = current rank-K+1 item |
| `greedy:target_score` | static order by contribution to the target score (attribution) |
| `greedy:list_aware` | static order by contribution to the target minus contribution to the retained items |
| `greedy:popularity` / `greedy:random` | control orders |
| `greedy:<criterion>:c1_stop` | stops as soon as C1 holds and only *reports* whether C2 / C3 happen to hold |
| `beam` | beam search over masks (width 5) with the violation surrogate |
| `continuous:<sigmoid|gumbel>:<attribution|random>:<repair|norepair>` | gradient-based relaxed mask with sparsity penalty (Adam, analytic gradients), discretised by thresholding, then optionally verify-and-repair |

`repair` restores feasibility (adds interactions greedily) and then prunes redundant ones. Exact versus
approximate comparisons (`exact_vs_approximate.csv`) report success rates, the share of exact optima reached,
mean excess size over the exact optimum, and — as consistency checks that must be zero — masks smaller than the
exact optimum and successes on instances the MILP proved infeasible.

## Independent verification

`epic_rec.evaluation.verify` does not reuse the solver's matrices. It (1) recomputes all masked scores with
`model.score(u, M)` and re-ranks with the tie-break; (2) recomputes the target's and every retained item's score
path by path (`score_from_paths`) and requires agreement to `1e-9`; (3) for C3, enumerates the surviving paths of
every retained item and tests faithfulness by disabling them. A result counts as a success **only** if every
requested constraint passes here; a solver mask that fails is reported with `status = "verification_failed"`
and never counted.
