"""Train the path-additive backbone and the accuracy baselines; select baseline hyper-parameters on validation.

Outputs (paths from the config's ``model`` and ``results_dir`` keys):
* the fitted backbone (``model.path``, .npz);
* ``<results_dir>/accuracy.json`` and ``accuracy.csv``: Recall@K / NDCG@K on validation and test for every model,
  the selected hyper-parameters and the learned path weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from epic_rec.data.dataset import InteractionDataset
from epic_rec.evaluation.accuracy import evaluate
from epic_rec.models.baselines import BPRMF, ItemKNN, Popularity
from epic_rec.models.path_additive import PathAdditiveRecommender
from epic_rec.utils.io import atomic_write_json
from epic_rec.utils.logging import get_logger

log = get_logger(__name__)


def model_path(cfg: dict) -> Path:
    return Path(cfg.get("model", {}).get("path", f"models/{cfg['dataset']['name']}/path_additive.npz"))


def load_model(cfg: dict, ds: InteractionDataset) -> PathAdditiveRecommender:
    p = model_path(cfg)
    if p.exists():
        return PathAdditiveRecommender.load(p)
    log.info("no trained model at %s; training", p)
    return PathAdditiveRecommender.fit(ds, {**cfg.get("model", {}), "seed": cfg.get("seed", 0)})


def train_and_evaluate(cfg: dict, ds: InteractionDataset, baselines: bool = True) -> dict[str, Any]:
    mcfg = {**cfg.get("model", {}), "seed": cfg.get("seed", 0)}
    ks = tuple(cfg.get("evaluation", {}).get("ks", [10, 20]))
    sel = f"ndcg@{max(ks)}"
    res: dict[str, Any] = {}
    model = PathAdditiveRecommender.fit(ds, mcfg)
    model.save(model_path(cfg))
    res["path_additive"] = dict(val=evaluate(model, ds, "val", ks), test=evaluate(model, ds, "test", ks),
                                theta=model.theta, beta=model.beta, fit=getattr(model, "fit_info", {}))
    if baselines:
        kg = PathAdditiveRecommender.fit(ds, {**mcfg, "collaborative": False})
        res["path_additive_kg_only"] = dict(val=evaluate(kg, ds, "val", ks), test=evaluate(kg, ds, "test", ks),
                                            theta=kg.theta, beta=kg.beta)
        X = ds.matrix("train")
        pop = Popularity(X)
        res["popularity"] = dict(val=evaluate(pop, ds, "val", ks), test=evaluate(pop, ds, "test", ks))
        best = None
        for k in cfg.get("baselines", {}).get("itemknn_k", [None, 50, 200]):
            m = ItemKNN(X, k)
            v = evaluate(m, ds, "val", ks)
            if best is None or v[sel] > best[1][sel]:
                best = (k, v, m)
        res["itemknn"] = dict(val=best[1], test=evaluate(best[2], ds, "test", ks), k=best[0])
        bc = cfg.get("baselines", {}).get("bprmf", {})
        mf = BPRMF(X, dim=bc.get("dim", 64), lr=bc.get("lr", 0.05), reg=bc.get("reg", 1e-4), seed=cfg.get("seed", 0))
        state: dict[str, Any] = dict(best=-1.0, epoch=0, bad=0, P=None, Q=None)
        every, patience = bc.get("eval_every", 5), bc.get("patience", 2)

        def cb(m: BPRMF, ep: int) -> bool:
            if (ep + 1) % every:
                return False
            v = evaluate(m, ds, "val", ks)[sel]
            if v > state["best"]:
                state.update(best=v, epoch=ep + 1, bad=0, P=m.P.copy(), Q=m.Q.copy())
            else:
                state["bad"] += 1
            return state["bad"] >= patience

        mf.fit(bc.get("max_epochs", 100), callback=cb)
        if state["P"] is not None:
            mf.P, mf.Q = state["P"], state["Q"]
        res["bprmf"] = dict(val=evaluate(mf, ds, "val", ks), test=evaluate(mf, ds, "test", ks),
                            epochs=state["epoch"], dim=mf.dim)
    out = Path(cfg.get("results_dir", f"results/{ds.name}"))
    atomic_write_json(out / "accuracy.json", dict(dataset=ds.name, stats=ds.stats(), models=res))
    rows = [dict(model=k, split=s, **{m: v for m, v in d[s].items() if m != "users"})
            for k, d in res.items() for s in ("val", "test")]
    pd.DataFrame(rows).to_csv(out / "accuracy.csv", index=False)
    log.info("accuracy written to %s", out / "accuracy.json")
    return res
