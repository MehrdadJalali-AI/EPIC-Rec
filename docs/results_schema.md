# Output files

```
results/
  <dataset>/
    accuracy.json, accuracy.csv           recommendation accuracy of the backbone and baselines
    <experiment>/
      sample.json                         sampled users, strata, resolved config
      records.jsonl                       one record per (user, target, configuration, setting, method)
      prince_records.jsonl                PRINCE runs (prince experiments only)
  report/
    tables/*.csv                          every table, computed from records.jsonl
    figures/*.{pdf,png}                   every figure, computed from the tables
    report_manifest.json
```

## Explanation object

`epic_rec.explainers.explain(...)` returns an `Explanation` (also written by `epic-rec explain --out` and by the
demo as JSON):

| field | meaning |
| --- | --- |
| `user`, `target`, `rank`, `K` | the request: remove `target` (at `rank`) from the user's top-`K` list |
| `setting`, `setting_label` | `A` / `B` / `C` / `A3` and its constraints, e.g. `"C1 + C2 + C3"` |
| `method` | `exact`, `beam`, `greedy:<criterion>[:c1_stop]`, `continuous` |
| `eps`, `alpha`, `phi` | score margin and evidence thresholds |
| `status` | exact: `optimal` / `feasible` / `infeasible` / `timeout`; approximate: `found` / `not_found`; any method: `verification_failed` |
| `mask`, `size` | the interactions (item ids of the user's history) to mask, and their number |
| `success` | `true` only if the independent verifier confirmed every requested constraint |
| `bound`, `gap` | exact only: proven lower bound on the minimum size and relative optimality gap |
| `verification` | the verifier's report (below) |
| `infeasibility` | for infeasible exact results: cause, weaker-setting outcomes, certificates, blocking items, entanglement |
| `details` | method-specific information (iterations, generated constraints, discretisation and repair steps, …) |

`verification` contains `passed` (P and P_ε), `passed_P`, `passed_P_eps`, `c1`, `c2`, `c3`, `c1_margin`,
`c1_c2_margin`, `independent` (scores recomputed from the model), `decomposition_error`, `top_k_after`, `entered`
(items that joined the list), `overlap` (share of retained items still in the list), `kendall_retained` (rank
correlation of the retained items before and after), `evidence` (per retained item: surviving qualifying paths,
the strongest one, faithfulness checks) and `evidence_survival` (share of retained items that keep faithful
evidence).

**Reading an explanation.** *"If interactions M had not been in your history, item t would not be in your top-K;
the other items would still be recommended (C2), each still supported by the listed paths (C3)."* An
`infeasible` result is also informative: it certifies that no set of the user's own interactions can remove `t`
under the requested guarantees, and the cause says why (for example, the target and a retained item rely on the
same interactions).

## Record fields (`records.jsonl`)

Identification: `dataset`, `experiment`, `user`, `stratum` (training-history quartile, 0 = shortest),
`hist_len`, `K`, `alpha`, `phi`, `eps`, `rank`, `target`, `setting`, `setting_label`, `method`.
Instance: `domain_size` (|D_u(t)|), `evidence_free`, `ent_*` (entanglement statistics).
Outcome: `status`, `success`, `size`, `mask`, `bound`, `gap`, `runtime` (s), `killed`, `iterations`, `calls`.
Verification summary: `c1`, `c2`, `c3`, `c1_margin`, `c1_c2_margin`, `passed_P`, `passed_P_eps`, `overlap`,
`kendall`, `evidence_survival`, `surviving_paths_mean`, `strongest_min`, `decomposition_error`.
Exact only, optional: `enum_P`, `enum_Peps` (enumerated optimum under P and P_ε, −1 = infeasible),
`infeasibility_cause`, `certificates`, `evidence_blocking_items`.

## Tables (`epic-rec report`)

All shares are percentages of pairs; `pairs` and `users` give the denominators; `*_lo` / `*_hi` are 95 %
user-level stratified cluster-bootstrap intervals (users resampled within history quartiles; all targets of a user
move together).

| file | content |
| --- | --- |
| `feasibility_by_setting.csv` | exact-solver outcome shares per setting, mask existence with CI, killed solves, runtimes, mean gap |
| `mask_size_distribution.csv` | distribution of minimum mask sizes (optimal pairs) |
| `paired_mask_size_differences.csv` | size increase between settings (B − A, C − B, …) on pairs optimal in both, with CI |
| `feasibility_by_rank.csv`, `feasibility_by_history_quartile.csv`, `feasibility_by_entanglement_quartile.csv` | mask existence by stratum |
| `exact_vs_approximate.csv` | per method: success (and C1, C1 + C2, C1 + C2 + C3 shares), exact-size share, excess size; consistency counters `smaller_than_optimum` and `success_on_infeasible` (must be 0) |
| `evidence_survival.csv` | evidence survival, surviving path counts and strengths, list overlap, Kendall τ of returned masks |
| `runtime_scalability.csv` | exact-solve runtime and timeouts by mask-domain size |
| `p_vs_p_eps.csv` | P versus P_ε on enumerated pairs; MILP agreement with enumeration |
| `infeasibility_causes.csv` | causes of infeasibility per setting |
| `cross_dataset.csv` | mask existence per dataset and setting |

Sensitivity analyses (K, α, φ, ε) and ablations are rows of the same tables, distinguished by the `experiment`,
`K`, `alpha`, `phi`, `eps`, `setting` and `method` columns.
