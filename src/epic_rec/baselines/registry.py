"""Published counterfactual explainers: what was examined, and whether a reliable comparison is possible.

Nothing in this package re-implements a published method under its name. A published baseline is either run from
its official code (PRINCE) or documented here with the reason it could not be compared. The full discussion is in
``docs/baselines.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublishedBaseline:
    name: str
    reference: str
    repository: str
    commit: str | None
    licence: str
    status: str  # "adapter" | "not_reproducible"
    model: str
    reason: str


PUBLISHED: dict[str, PublishedBaseline] = {
    "prince": PublishedBaseline(
        name="PRINCE",
        reference="Ghazimatin et al., WSDM 2020, doi:10.1145/3336191.3371824",
        repository="https://github.com/azinmatin/prince",
        commit="570e5e25e40f82a0f7d5f2c20185f1ff8ad7f7a4",
        licence="none stated (all rights reserved by default); code is downloaded at run time, not redistributed",
        status="adapter",
        model="personalised PageRank on the user-item-entity graph (PRINCE's native model)",
        reason=(
            "Run from the official code on its native recommender. PRINCE explains the replacement of the top-1 "
            "item; masks are evaluated for that goal and for C1 and C2 by recomputing PageRank. C3 is not defined "
            "for PageRank scores (no additive path decomposition) and is reported as not applicable. The code is "
            "Python 2; the adapter's only change is converting print statements to print() calls."
        ),
    ),
    "accent": PublishedBaseline(
        name="ACCENT",
        reference="Tran et al., SIGIR 2021, doi:10.1145/3404835.3463005",
        repository="https://github.com/hieptk/accent",
        commit="8c6652ed5e0c9716d33b3b29b03e2b25ab429ce3",
        licence="GPL-3.0",
        status="not_reproducible",
        model="NCF and RCF trained in TensorFlow 2.2",
        reason=(
            "The official code requires Python 3.7, TensorFlow 2.2 and tensorflow-addons 0.11 (or the authors' "
            "Ubuntu VM). TensorFlow 2.2 was released for Python 3.5-3.8 on x86 only, so it was not installed in "
            "this project's environment. ACCENT's influence functions are defined for NCF / RCF models trained by "
            "gradient descent; our path-additive backbone has frozen graph statistics and a few non-negative path "
            "weights, so ACCENT's influence estimates would not refer to the same model. Porting it would be a new "
            "method, not ACCENT."
        ),
    ),
    "grease": PublishedBaseline(
        name="GREASE",
        reference="Chen et al., arXiv:2208.04222 (2022); ACM TORS 4(2), 2025, doi:10.1145/3731683",
        repository="no official repository found (GitHub search, Nov 2026)",
        commit=None,
        licence="n/a",
        status="not_reproducible",
        model="GNN recommender with a surrogate model on the l-hop neighbourhood",
        reason="No official implementation was found; a re-implementation from the paper would not be GREASE.",
    ),
    "cosrec": PublishedBaseline(
        name="CoSRec",
        reference="Zhang et al., ICONIP 2026 (accepted; proceedings not yet published)",
        repository="https://github.com/zhang-haichao/CoSRec",
        commit="6b63e0bc6abeeca57ae70bd96e7224a8918dcad3",
        licence="none stated (all rights reserved by default)",
        status="not_reproducible",
        model="LightGCN (PyTorch + PyTorch Geometric + torch-sparse) on MovieLens-100K / 1M",
        reason=(
            "CoSRec learns masks for its own trained LightGCN checkpoints; LightGCN scores have no path "
            "decomposition, so C3 cannot be evaluated, and the repository has no licence permitting adaptation. "
            "A comparison on C1 and C2 would require running the authors' pipeline end to end on our splits; this "
            "was not attempted in this release and is listed as future work."
        ),
    ),
    "cf-gnnexplainer": PublishedBaseline(
        name="CF-GNNExplainer",
        reference="Lucic et al., AISTATS 2022, PMLR 151:4499-4511",
        repository="https://github.com/a-lucic/cf-gnnexplainer",
        commit="b643562cfd503b5e6970cc3fefc5bebc01189448",
        licence="MIT",
        status="not_reproducible",
        model="GCN node classification",
        reason=(
            "The method explains node-classification predictions of a GCN; it has no notion of a ranked list. "
            "The in-house continuous-mask explainer (epic_rec.explainers.continuous) is an edge-perturbation "
            "optimiser in the same spirit but is not CF-GNNExplainer and is never labelled as such."
        ),
    ),
}
