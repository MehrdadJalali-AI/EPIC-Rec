# Baselines

Two kinds of comparison are made, and they are kept strictly apart.

1. **In-house approximate explainers** (greedy variants, beam search, continuous masks, random and popularity
   controls) run on the same model and the same problem as the exact solver. They are named after what they do
   (`greedy:target_score`, `continuous:gumbel:random:repair`, …) and never after a published method.
2. **Published counterfactual explainers** are run only from their official code, pinned to a commit. When this
   was not possible, the reason is documented below and no number is reported for that method.

The registry with the same information is `epic_rec.baselines.registry.PUBLISHED`.

## Recommendation-accuracy baselines

`epic-rec train` reports Recall@10/20 and NDCG@10/20 on validation and test for: the path-additive model, the same
model without collaborative paths (KG only), popularity, ItemKNN (cosine, neighbourhood size selected on
validation) and BPR matrix factorisation (early stopping on validation NDCG). Results are written to
`results/<dataset>/accuracy.{json,csv}`.

## Published explainers

| method | official code (commit) | licence | status |
| --- | --- | --- | --- |
| PRINCE (Ghazimatin et al., WSDM 2020) | `azinmatin/prince@570e5e2` | none stated | **run** (adapter) |
| ACCENT (Tran et al., SIGIR 2021) | `hieptk/accent@8c6652e` | GPL-3.0 | not reproducible here |
| GREASE (Chen et al., 2022 / ACM TORS 2025) | none found | – | not reproducible |
| CoSRec (Zhang et al., ICONIP 2026, accepted) | `zhang-haichao/CoSRec@6b63e0b` | none stated | not attempted in this release |
| CF-GNNExplainer (Lucic et al., AISTATS 2022) | `a-lucic/cf-gnnexplainer@b643562` | MIT | not applicable (node classification) |

### PRINCE — run

* **Code.** `cfe_generator.py` and `local_push.py` are downloaded at run time from the pinned commit into
  `~/.cache/epic_rec/prince/` and checked against SHA-256 values (the repository has no licence, so the files are
  not redistributed). The code is Python 2; the adapter's only modification is converting `print` statements to
  function calls, applied in memory.
* **Model.** PRINCE explains a personalised-PageRank recommender on the user–item–entity graph. It is run on that
  native model (`epic_rec.baselines.ppr.PPRRecommender`, built from the same training split and KG links),
  because PRINCE's algorithm is specific to PageRank.
* **Goal.** PRINCE finds a minimum set of the user's actions whose removal replaces the **top-1** item with
  another item. The adapter records PRINCE's own goal (`prince_goal`), and separately recomputes PageRank to check
  whether the target also left the top-K list (C1) and whether the other items stayed (C2). Because PRINCE
  replaces the top item rather than removing it from the list, C1 can fail even when PRINCE succeeds. C3 is not
  defined for PageRank scores (no additive path decomposition) and is reported as `null`.
* **Run.** `epic-rec prince --config configs/experiments/movielens_prince.yaml` writes `prince_records.jsonl`.
  PRINCE's tolerance `epsilon` is configurable (`prince.epsilon`); its default `1 / (nodes · 10^4)` is slow on
  the real graphs.

### ACCENT — not reproducible here

Repository `hieptk/accent`, commit `8c6652ed5e0c9716d33b3b29b03e2b25ab429ce3`, GPL-3.0. Requirements: Python 3.7,
TensorFlow 2.2, tensorflow-addons 0.11 (or the authors' Ubuntu VM). TensorFlow 2.2 was released for Python 3.5–3.8
on x86 only and was not installed in this project's environment. More fundamentally, ACCENT's influence
functions are defined for NCF / RCF models trained by gradient descent; the path-additive model has frozen graph
statistics and a few non-negative weights, so ACCENT's influence estimates would not refer to the same model.
Porting it would produce a new method, not ACCENT; no ACCENT number is reported.

### GREASE — no official code

No official implementation was found (GitHub search, November 2026). A re-implementation from the paper would not
be GREASE and is not provided.

### CoSRec — not attempted in this release

Repository `zhang-haichao/CoSRec`, commit `6b63e0bc6abeeca57ae70bd96e7224a8918dcad3`, no licence. CoSRec learns masks
for its own trained LightGCN checkpoints (PyTorch, PyTorch Geometric, torch-sparse). LightGCN scores have no path
decomposition, so C3 cannot be evaluated, and the missing licence does not permit adaptation. A comparison on C1
and C2 would require running the authors' pipeline end to end on our splits; this is future work.

### CF-GNNExplainer — not applicable

Repository `a-lucic/cf-gnnexplainer`, commit `b643562cfd503b5e6970cc3fefc5bebc01189448`, MIT. The method explains
node-classification predictions of a GCN and has no notion of a ranked list. The in-house continuous-mask
explainer is an edge-perturbation optimiser in the same spirit but is not CF-GNNExplainer and is never labelled as
such.
