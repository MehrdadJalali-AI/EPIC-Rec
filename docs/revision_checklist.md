# Revision checklist

Each item says where it was changed and how it was checked. Items that were not completed as new experiments say so.

1. **Figure 1 numbers.** The old chart labels 2.19, 0.35 and 27.5 are not in `manuscript/fig1_charts.svg` or its PDF. Bars and whiskers are computed by `scripts/fig1_from_results.py` from `results/report/tables`. All-pairs C1+C2+C3 is 73.10%. All-pairs C1-stop success is 21.80% [19.90, 23.67]. The conditional rate 27.5% is stated in the caption of Figure 2 and in the abstract, and is not a bar. Paired costs are 1.348 [1.185, 1.524] and 0.313 [0.250, 0.377]. Check: `tests/test_fig1_numbers.py`.

2. **Denominators.** Exact existence rates use all 2,000 pairs. Heuristic success in Table 7 uses all 2,000 pairs (21.80). `MLCStopAll` (27.5) divides the joint successes by the edits that reached C1. The abstract and the chart caption say which is which.

3. **Reproducibility.** `scripts/manuscript_from_report.py` writes `numbers.tex` and the tables. The charts and the worked example have their own scripts. Aggregated tables and compact exact statuses are in `results/report/tables` and `results/published`. Full `records.jsonl` files stay out of git (about 80 MB each). Raw data is not uploaded. The availability paragraph names the repository. `manuscript/REVISION` records the content commit. Not every sentence of prose is generated; the paragraph says so.

4. **Published comparisons.** Not run. GREASE, LXR, ACCENT, PRINCE, TRACE and Zhang et al. (counterfactual support sets) are different tasks and backbones, and their official code was not integrated. The limitations section states that blocker. No in-house heuristic is labelled with a published name. Zhang et al. remains a preprint citation; the publisher venue and DOI were not confirmable. The name CoSRec also belongs to an unrelated SIGIR 2025 dataset.

5. **Optimality language.** The manuscript no longer calls two HiGHS configurations a certificate or a second solver. Verification of a mask, a solver status, and the second configuration are distinguished in the method section. The known size-4 then size-3 MovieLens case is described without calling it a solver bug. The K=5 worked example was also enumerated: P and Pε both have minimum size 1, and that size matches the solver and the stored record.

6. **Scope.** P, Pε and D_u(t) are distinguished. Infeasibility of Pε is not infeasibility of P. Infeasibility on D_u(t) is not infeasibility over the whole history. Covering constraints omit an item whose qualifying origins are not all inside D_u(t), which is the fixed-unmasked case. The 190-pair enumeration is described as a sample, not a universal proof.

7. **Runtime.** The 20-second value is the HiGHS clock, which is soft. The recorded maximum on MovieLens setting A is 24.53 seconds. The hard kill is 20 seconds plus a 40-second grace. Recorded runtime excludes the second-configuration time and the separate verifier pass. Medians, maxima and killed counts are the stored macros. No end-to-end deployment latency is claimed. The original study machine's CPU model is not in the result files, so it is not invented here.

8. **MovieLens versus Last.fm.** From stored results: the gap is already present at C1+C2 (83.30% versus 30.05%). Last.fm infeasibility of the full constraint is list entanglement for 91.2%, evidence entanglement for 0.4%, joint for 2.4%. β is 0.00 versus 0.48, fit separately. D_u(t) covers 100% versus 70.3% of history interactions. Evidence-free pairs are 0 and 0. No retrained popularity ablation and no expanded-domain re-solve were run. The text says that.

9. **Editorial.** φ = 0 is defined as the contribution floor. Continuous-mask loss, Adam settings, threshold 0.5 and repair are stated from `continuous.py`. Setting A3 (C1+C3) is defined with the other settings. Faithfulness is limited to the additive score. Zero-contribution paths are excluded. Authorship was not changed. No journal class was imposed. "First method" is not claimed.

10. **Worked example.** Section "Worked example", Figure `fig_worked`, Table `tab:worked`, data in `case_study/worked_k5.json`. User 4739, K=5, target *Return of the Jedi*, mask size 1 (*The Empire Strikes Back*). Status optimal, verifier passed, enumeration of the 12-interaction domain agrees for P and Pε, stored K-sweep record matches. The removal-only heuristic also satisfies C2 and C3 on this pair, so no collateral-damage story was added. Entity names are rehashed ids.

11. **Checks.** `pytest tests/test_fig1_numbers.py tests/test_certify.py` passed. `latexmk` produced `manuscript/epic_rec.pdf` (19 pages) with no undefined citations. One URL line was tightened with `xurl`.
