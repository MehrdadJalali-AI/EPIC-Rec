"""Deterministic preprocessing: raw files -> :class:`InteractionDataset`.

Pipeline (every step is configured in ``configs/<dataset>.yaml`` and recorded in the manifest):

1. read positive interactions (explicit ratings are binarised with ``positive.min_rating``; play counts with
   ``positive.min_weight``);
2. optionally keep only items that have a knowledge-graph entity (``require_kg_link``);
3. iterative k-core filtering (``k_core.min_user`` / ``k_core.min_item``) until no user or item changes;
4. item-entity links: triples with exactly one item endpoint become (item, relation, entity) links; triples
   between two items are counted and ignored (the backbone uses item-entity-item paths only); relations are
   renamed with ``kg.relation_map``, dropped with ``kg.drop_relations``, and filtered by
   ``kg.min_links_per_relation``; entities linked to fewer than ``kg.min_items_per_entity`` items (they create no
   path) or more than ``kg.max_items_per_entity`` items are removed;
5. per-user random split into train / validation / test with a seed (:func:`split_per_user`).

The k-core filter uses all interactions before the split, as is standard; every statistic the recommender
learns from (co-occurrence, popularity, model weights) is computed from the training split only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epic_rec.data.dataset import InteractionDataset, split_per_user
from epic_rec.data.sources import SOURCES
from epic_rec.utils.io import sha256_file
from epic_rec.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------- raw readers
def _read_movielens(raw: Path, cfg: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    r = pd.read_csv(raw / "ratings.dat", sep="::", engine="python", header=None, names=["user", "item", "rating", "ts"])
    r = r[r.rating >= cfg.get("positive", {}).get("min_rating", 4)]
    i2e = pd.read_csv(raw / "item_index2entity_id_rehashed.txt", sep="\t", header=None, names=["item", "entity"])
    kg = pd.concat(
        pd.read_csv(raw / f, sep="\t", header=None, names=["h", "r", "t"])
        for f in ("kg_part1_rehashed.txt", "kg_part2_rehashed.txt")
    )
    return r[["user", "item"]], dict(zip(i2e.item, i2e.entity)), kg


def _read_lastfm(raw: Path, cfg: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    r = pd.read_csv(raw / "user_artists.dat", sep="\t")
    r.columns = [c.strip() for c in r.columns]
    r = r.rename(columns={"userID": "user", "artistID": "item"})
    r = r[r.weight >= cfg.get("positive", {}).get("min_weight", 1)]
    i2e = pd.read_csv(raw / "item_index2entity_id.txt", sep="\t", header=None, names=["item", "entity"])
    kg = pd.read_csv(raw / "kg.txt", sep="\t", header=None, names=["h", "r", "t"])
    return r[["user", "item"]], dict(zip(i2e.item, i2e.entity)), kg


def _read_amazon_book(raw: Path, cfg: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    for f in ("train.txt", "test.txt"):
        for line in (raw / f).read_text().splitlines():
            parts = line.split()
            if len(parts) > 1:
                u = int(parts[0])
                rows.extend((u, int(i)) for i in parts[1:])
    r = pd.DataFrame(rows, columns=["user", "item"]).drop_duplicates()
    items = pd.read_csv(raw / "item_list.txt", sep=" ")
    rel = pd.read_csv(raw / "relation_list.txt", sep=" ")
    names = {int(k): str(v).replace("http://rdf.freebase.com/ns/", "") for v, k in zip(rel.org_id, rel.remap_id)}
    kg = pd.read_csv(raw / "kg_final.txt", sep=" ", header=None, names=["h", "r", "t"])
    kg["r"] = kg.r.map(names)
    # KGAT convention: item remap id i is entity remap id i
    return r, {int(i): int(i) for i in items.remap_id}, kg


READERS = {"movielens-1m": _read_movielens, "lastfm": _read_lastfm, "amazon-book": _read_amazon_book}


# ---------------------------------------------------------------------- steps
def k_core(r: pd.DataFrame, min_user: int, min_item: int, max_iter: int = 100) -> tuple[pd.DataFrame, int]:
    for it in range(1, max_iter + 1):
        n0 = len(r)
        uc = r.user.value_counts()
        r = r[r.user.isin(uc[uc >= min_user].index)]
        ic = r.item.value_counts()
        r = r[r.item.isin(ic[ic >= min_item].index)]
        if len(r) == n0:
            return r, it
    raise RuntimeError("k-core filtering did not converge")


def item_entity_links(
    kg: pd.DataFrame, item2ent: dict, items: np.ndarray, kcfg: dict
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (item, relation, entity) links for the kept items and a dict of filter counts."""
    ent2item = {e: i for i, e in item2ent.items() if i in set(items.tolist())}
    all_item_ents = set(item2ent.values())
    h_item = kg.h.map(ent2item)
    t_item = kg.t.map(ent2item)
    h_is_any_item = kg.h.isin(all_item_ents)
    t_is_any_item = kg.t.isin(all_item_ents)
    a = kg[h_item.notna() & ~t_is_any_item]
    b = kg[t_item.notna() & ~h_is_any_item]
    ie = pd.concat(
        [
            pd.DataFrame({"item_orig": a.h.map(ent2item).astype(np.int64).values, "relation": a.r.values,
                          "entity": a.t.values}),
            pd.DataFrame({"item_orig": b.t.map(ent2item).astype(np.int64).values, "relation": b.r.values,
                          "entity": b.h.values}),
        ]
    )
    counts: dict[str, Any] = dict(item_item_triples_ignored=int((h_is_any_item & t_is_any_item).sum()))
    rmap = kcfg.get("relation_map") or {}
    if rmap:
        mapped = ie.relation.map(rmap)
        if not kcfg.get("keep_unmapped", False):
            counts["links_unmapped_dropped"] = int(mapped.isna().sum())
            ie = ie[mapped.notna()].assign(relation=mapped[mapped.notna()])
        else:
            ie = ie.assign(relation=mapped.fillna(ie.relation))
    drop = set(kcfg.get("drop_relations") or [])
    if drop:
        counts["links_dropped_relations"] = int(ie.relation.isin(drop).sum())
        ie = ie[~ie.relation.isin(drop)]
    ie = ie.drop_duplicates()
    lo, hi = kcfg.get("min_items_per_entity", 2), kcfg.get("max_items_per_entity")
    deg = ie.groupby(["relation", "entity"]).item_orig.transform("nunique")
    keep = deg >= lo
    if hi:
        counts["links_high_degree_dropped"] = int((deg > hi).sum())
        keep &= deg <= hi
    ie = ie[keep]
    rc = ie.relation.value_counts()
    small = rc[rc < kcfg.get("min_links_per_relation", 1)].index
    counts["relations_too_small"] = sorted(map(str, small))
    ie = ie[~ie.relation.isin(small)]
    return ie, counts


def prepare(cfg: dict) -> InteractionDataset:
    """Build and save the processed dataset described by the ``dataset`` section of a config."""
    dcfg = cfg["dataset"]
    name = dcfg["name"]
    raw = Path(dcfg.get("raw_dir", "data/raw")) / name
    out = Path(dcfg.get("processed_dir", f"data/processed/{name}"))
    if name not in READERS:
        raise KeyError(f"unknown dataset {name}; available: {sorted(READERS)}")
    r, item2ent, kg = READERS[name](raw, dcfg)
    n_raw = len(r)
    if dcfg.get("require_kg_link", True):
        r = r[r.item.isin(set(item2ent))]
    n_linked = len(r)
    kc = dcfg.get("k_core", {})
    r, iters = k_core(r, kc.get("min_user", 10), kc.get("min_item", 5))
    users = np.sort(r.user.unique())
    items = np.sort(r.item.unique())
    ie, kg_counts = item_entity_links(kg, item2ent, items, dcfg.get("kg", {}))
    uid = {u: k for k, u in enumerate(users)}
    iid = {i: k for k, i in enumerate(items)}
    pairs = np.column_stack([r.user.map(uid).values, r.item.map(iid).values]).astype(np.int32)
    s = dcfg.get("split", {})
    train, val, test = split_per_user(pairs, len(users), s.get("val_frac", 0.1), s.get("test_frac", 0.2),
                                      s.get("seed", 0))
    ent_codes, ent_uniques = pd.factorize(ie.entity, sort=True)
    item_entity = (
        pd.DataFrame({"item": ie.item_orig.map(iid).astype(np.int64).values, "relation": ie.relation.astype(str).values,
                      "entity": ent_codes})
        .sort_values(["relation", "entity", "item"])
        .reset_index(drop=True)
    )
    src = SOURCES[name]
    meta = dict(
        source=dict(description=src.description, linkage=src.linkage, licence=src.licence, notes=list(src.notes)),
        raw_files={p.name: sha256_file(p) for p in sorted(raw.iterdir()) if p.is_file() and p.suffix != ".json"},
        config=dcfg,
        filtering=dict(interactions_raw_positive=n_raw, interactions_with_kg_link=n_linked,
                       interactions_after_k_core=int(len(r)), k_core_iterations=iters, **kg_counts),
    )
    ds = InteractionDataset(name, len(users), len(items), train, val, test, item_entity, users, items, meta)
    ds.check_no_leakage()
    ds.save(out)
    log.info("%s: %s", name, ds.stats())
    return ds


def load_or_prepare(cfg: dict) -> InteractionDataset:
    dcfg = cfg["dataset"]
    if dcfg["name"] == "synthetic":
        from epic_rec.data.synthetic import make_synthetic

        return make_synthetic(**dcfg.get("synthetic", {}))
    out = Path(dcfg.get("processed_dir", f"data/processed/{dcfg['name']}"))
    if (out / "manifest.json").exists():
        return InteractionDataset.load(out)
    return prepare(cfg)
