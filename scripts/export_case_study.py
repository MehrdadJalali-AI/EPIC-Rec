"""Export three case studies from the stored MovieLens main run.

Reuses the fitted path-additive model, the HiGHS solver, the ranking-margin
heuristic and the independent verifier. Does not modify stored results.

The Satori release shipped with this study is rehashed: entity ids have no
names. Relation labels are mapped to actor/director/writer/genre/language/
country/rating/collaborative. Collaborative vias are labelled "co-user".
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from epic_rec.data.preprocess import load_or_prepare
from epic_rec.evaluation.verify import verify
from epic_rec.explainers.greedy import greedy
from epic_rec.models.path_additive import COLLAB
from epic_rec.optimization.diagnose import evidence_blocking_items, infeasibility_report
from epic_rec.optimization.instance import EPS, ExplanationInstance
from epic_rec.optimization.milp import solve_milp
from epic_rec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "movielens-1m" / "main" / "records.jsonl"
MOVIES = ROOT / "data" / "raw" / "movielens-1m" / "movies.dat"
OUT = ROOT / "case_study"

FAMOUS = {
    "toy story", "the godfather", "the godfather: part ii", "pulp fiction",
    "schindler's list", "the shawshank redemption", "forrest gump",
    "the silence of the lambs", "the matrix", "jurassic park", "titanic",
    "star wars: episode iv - a new hope", "star wars: episode v - the empire strikes back",
    "star wars: episode vi - return of the jedi", "raiders of the lost ark",
    "back to the future", "casablanca", "terminator 2: judgment day",
    "the usual suspects", "fargo", "goodfellas", "psycho", "2001: a space odyssey",
    "e.t. the extra-terrestrial", "the lion king", "aladdin", "saving private ryan",
    "braveheart", "gladiator", "american beauty", "the sixth sense", "alien",
    "die hard", "the princess bride", "groundhog day", "apollo 13",
}

RELATION = {
    "film.actor.film": "actor",
    "film.person_or_entity_appearing_in_film.film": "actor",
    "film.film.star": "actor",
    "film.director.film": "director",
    "film.film.director": "director",
    "film.writer.film": "writer",
    "film.film.writer": "writer",
    "film.film.genre": "genre",
    "film.film.language": "language",
    "film.film.country": "country",
    "film.film.rating": "rating",
    COLLAB: "collaborative",
}


def load_titles(ds) -> dict[int, str]:
    raw: dict[int, str] = {}
    for line in MOVIES.read_text(encoding="latin-1").splitlines():
        mid, title, _genres = line.split("::", 2)
        raw[int(mid)] = title
    out = {}
    missing = []
    for i, ext in enumerate(ds.item_ids):
        title = raw.get(int(ext))
        if title is None:
            missing.append(int(ext))
            title = f"UNRESOLVED ml-id {int(ext)}"
        out[i] = title
    if missing:
        print(f"WARNING: {len(missing)} item ids have no title in movies.dat")
    return out


def year_of(title: str) -> int | None:
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


def bare(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip().lower()


def item_ref(titles: dict[int, str], i: int) -> dict:
    title = titles[int(i)]
    return {"item_id": int(i), "title": title, "year": year_of(title)}


def load_exact() -> dict[tuple[int, int], dict]:
    """Certified exact records, keyed by (user, target)."""
    pairs: dict[tuple[int, int], dict] = {}
    with RECORDS.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("method") != "exact" or r.get("K") != 10 or r.get("alpha") != 0.5:
                continue
            if abs(r.get("eps", 0) - 1e-6) > 1e-12:
                continue
            pairs.setdefault((r["user"], r["target"]), {})[r["setting"]] = r
    return pairs


def proven(rec: dict | None, status: str) -> bool:
    """A stored claim counts when certification confirmed it, or the solver proved it structurally."""
    if not rec or rec.get("status") != status:
        return False
    return rec.get("certification") in ("confirmed", "structural")


def success_pool(pairs, titles) -> tuple[list[dict], list[str]]:
    notes = []
    success = []
    for (u, t), by in pairs.items():
        c = by.get("C")
        if not proven(c, "optimal"):
            continue
        if not (2 <= c.get("size", 99) <= 4 and 15 <= c.get("hist_len", 0) <= 40):
            continue
        title = titles.get(t, "")
        success.append({
            "user": u, "target": t, "title": title, "size": c["size"],
            "hist": c["hist_len"], "rank": c.get("rank"), "famous": bare(title) in FAMOUS,
        })
    success.sort(key=lambda r: (not r["famous"], r["size"], r["hist"], r["rank"] or 99))
    famous = [r for r in success if r["famous"]]
    notes.append(
        f"success candidates with C optimal, |M| in 2..4, history 15..40, certification confirmed: {len(success)}"
    )
    notes.append(f"of which the target title is on the well-known list: {len(famous)}")
    for r in (famous or success)[:8]:
        notes.append(
            f"  candidate user {r['user']} target {r['target']} '{r['title']}' "
            f"rank {r['rank']} |M_C|={r['size']} history {r['hist']}"
        )
    if not success:
        raise SystemExit("no success candidate in the stored main run")
    # Well-known titles first, then the rest, so a heuristic-failure screen can fall through.
    return famous + [r for r in success if not r["famous"]], notes


def infeasible_for(pairs, titles, user: int, avoid: int, hist: int) -> tuple[dict, list[str]]:
    notes = []
    same = []
    for (u, t), by in pairs.items():
        if u != user or t == avoid:
            continue
        a, b, c = by.get("A"), by.get("B"), by.get("C")
        if not proven(a, "optimal"):
            continue
        if proven(b, "infeasible") or proven(c, "infeasible"):
            same.append({
                "user": u, "target": t, "title": titles.get(t, ""), "rank": a.get("rank"),
                "hist": a["hist_len"], "B": b.get("status") if b else None,
                "C": c.get("status") if c else None, "same_user": True,
                "famous": bare(titles.get(t, "")) in FAMOUS,
            })
    if same:
        same.sort(key=lambda r: (not r["famous"], r["rank"] or 99))
        notes.append(f"same user {user} has {len(same)} other targets with A optimal and B or C proven infeasible")
        for r in same[:5]:
            notes.append(f"  infeasible candidate target {r['target']} '{r['title']}' rank {r['rank']} B={r['B']} C={r['C']}")
        return same[0], notes
    notes.append(f"user {user} has no other target with A optimal and B or C proven infeasible")
    best = None
    for (u, t), by in pairs.items():
        a, b, c = by.get("A"), by.get("B"), by.get("C")
        if not proven(a, "optimal") or not (15 <= a.get("hist_len", 0) <= 40):
            continue
        if not (proven(b, "infeasible") or proven(c, "infeasible")):
            continue
        title = titles.get(t, "")
        key = (abs(a["hist_len"] - hist), bare(title) not in FAMOUS, a.get("rank") or 99)
        cand = {"user": u, "target": t, "title": title, "rank": a.get("rank"), "hist": a["hist_len"],
                "B": b.get("status"), "C": c.get("status"), "same_user": False, "famous": bare(title) in FAMOUS}
        if best is None or key < best[0]:
            best = (key, cand)
    if best is None:
        raise SystemExit("no infeasible candidate")
    notes.append(
        f"infeasible fallback user {best[1]['user']} target {best[1]['target']} '{best[1]['title']}' "
        f"history {best[1]['hist']} B={best[1]['B']} C={best[1]['C']}"
    )
    return best[1], notes


def ranked(inst: ExplanationInstance, s: np.ndarray, n: int) -> list[dict]:
    ids = inst.topk(s, K=n)
    rows = []
    for rank, j in enumerate(ids, start=1):
        rows.append({"rank": rank, "item_id": int(j), "title": None, "score": round(float(s[j]), 6)})
    return rows


def fill_titles(rows: list[dict], titles: dict[int, str]) -> list[dict]:
    for r in rows:
        r["title"] = titles[r["item_id"]]
    return rows


def paths_for(model, titles, u, j, tau: float, mask: set[int] | None) -> list[dict]:
    """Qualifying paths into j. Entity names are not in the rehashed Satori files."""
    groups = model.path_groups(u, j)
    rows = []
    for g in groups:
        rel = RELATION.get(g.relation, g.relation)
        for p in g.paths():
            if p.contribution + 1e-15 < tau:
                continue
            via = "co-user" if p.via_type == "user" else f"rehashed-entity-{int(p.via)}"
            rows.append({
                "origin_item_id": int(p.origin),
                "origin_title": titles[int(p.origin)],
                "relation": rel,
                "relation_raw": g.relation,
                "via": via,
                "via_id": int(p.via),
                "entity_name_available": False,
                "contribution": round(float(p.contribution), 6),
                "survives_under_C": None if mask is None else int(p.origin) not in mask,
            })
    rows.sort(key=lambda r: -r["contribution"])
    return rows


def decomposition(inst, titles, j: int, mask: set[int], n: int = 6) -> list[dict]:
    col = inst.A[:, j]
    order = list(np.argsort(-col)[:n])
    # Keep a masked interaction visible even when it is not among the six largest.
    for i in mask:
        k = inst.pos.get(int(i))
        if k is not None and k not in order and col[k] > 0:
            order.append(k)
    rows = []
    for k in order:
        if col[k] <= 0:
            continue
        i = int(inst.H[k])
        rows.append({
            "item_id": i,
            "title": titles[i],
            "A_ij_before": round(float(col[k]), 6),
            "A_ij_after": 0.0 if i in mask else round(float(col[k]), 6),
            "masked": i in mask,
        })
    return rows


PERSON = {"actor", "director", "writer"}


def shared_with_target(model, titles, u, t, items: list[int]) -> list[dict]:
    """Links that also support the target. Person links are kept ahead of genre or country."""
    target_keys = {}
    for g in model.path_groups(u, t):
        for p in g.paths():
            target_keys.setdefault((g.relation, int(p.via)), float(p.contribution))
    out = []
    for j in items:
        best: dict[tuple, dict] = {}
        for g in model.path_groups(u, j):
            rel = RELATION.get(g.relation, g.relation)
            for p in g.paths():
                key = (g.relation, int(p.via))
                if key not in target_keys:
                    continue
                via = "co-user" if p.via_type == "user" else f"rehashed-entity-{int(p.via)}"
                row = {
                    "relation": rel,
                    "relation_raw": g.relation,
                    "via": via,
                    "origin_title": titles[int(p.origin)],
                    "contribution_to_j": round(float(p.contribution), 6),
                    "contribution_to_t": round(target_keys[key], 6),
                }
                k = (rel, via)
                if k not in best or row["contribution_to_j"] > best[k]["contribution_to_j"]:
                    best[k] = row
        person = [r for r in best.values() if r["relation"] in PERSON]
        other = [r for r in best.values() if r["relation"] not in PERSON]
        person.sort(key=lambda r: -r["contribution_to_j"])
        other.sort(key=lambda r: -r["contribution_to_j"])
        out.append({
            "item_id": int(j),
            "title": titles[int(j)],
            "shared_person": person[:6],
            "shared_other": other[:3],
        })
    return out


def items_whose_removal_releases(inst: ExplanationInstance, titles: dict[int, str]) -> list[dict]:
    """Retained items j such that C1+C2 becomes feasible once j is allowed to leave the list.

    The instance's retained set is restored before returning. This does not change the solver.
    """
    original = np.array(inst.retained, dtype=np.int64)
    found = []
    try:
        for j in original.tolist():
            inst.retained = np.array([x for x in original.tolist() if x != j], dtype=np.int64)
            res = solve_milp(inst, "B", eps=EPS, time_limit=20.0, threads=1)
            if res["status"] not in ("optimal", "feasible"):
                continue
            mask = norm_mask(res["mask"])
            report = verify(inst, mask, "B", eps=EPS)
            found.append({
                "item_id": int(j),
                "title": titles[int(j)],
                "status_if_this_item_may_leave": res["status"],
                "mask_size": res["size"],
                "verifier_passed": bool(report["passed"]),
            })
    finally:
        inst.retained = original
    return found


def norm_mask(mask) -> list[int]:
    return sorted(int(i) for i in mask) if mask else []


def solve_settings(inst, stored_by: dict) -> tuple[dict, list[str]]:
    mismatches = []
    out = {}
    for name in ("A", "B", "C"):
        # Study configuration: presolve off, feasibility tolerances 1e-9 (solve_milp defaults).
        res = solve_milp(inst, name, eps=EPS, time_limit=20.0, threads=1)
        mask = norm_mask(res["mask"])
        report = verify(inst, mask, name, eps=EPS) if res["status"] in ("optimal", "feasible") else None
        stored = stored_by.get(name, {})
        same_status = stored.get("status") == res["status"]
        same_size = stored.get("size") == res["size"]
        same_mask = norm_mask(stored.get("mask")) == mask
        if not (same_status and same_size):
            mismatches.append(
                f"setting {name}: re-run status {res['status']} size {res['size']} mask {mask}; "
                f"stored status {stored.get('status')} size {stored.get('size')} mask {stored.get('mask')}"
            )
        elif not same_mask:
            mismatches.append(
                f"setting {name}: same status and size, different mask. re-run {mask}; stored {stored.get('mask')}"
            )
        out[name] = {
            "status": res["status"],
            "size": res["size"],
            "runtime_s": round(float(res["time"]), 4),
            "mask_ids": mask,
            "lower_bound": res.get("bound"),
            "verifier_passed": None if report is None else bool(report["passed"]),
            "passed_P": None if report is None else bool(report["passed_P"]),
            "passed_P_eps": None if report is None else bool(report["passed_P_eps"]),
            "c1": None if report is None else bool(report["c1"]),
            "c2": None if report is None else report["c2"],
            "c3": None if report is None else report["c3"],
            "matches_stored_status_and_size": same_status and same_size,
            "matches_stored_mask": same_mask,
            "stored_status": stored.get("status"),
            "stored_size": stored.get("size"),
            "stored_mask": stored.get("mask"),
            "stored_runtime_s": stored.get("runtime"),
            "scores": res.get("scores"),
        }
        if res["status"] in ("optimal", "feasible") and report is not None and not report["passed"]:
            mismatches.append(f"setting {name}: solver returned {res['status']} but the verifier rejected the mask")
    return out, mismatches


def export_case(model, titles, spec, pairs, kind: str) -> tuple[dict, list[str]]:
    u, t = spec["user"], spec["target"]
    inst = ExplanationInstance(model, u, K=10, alpha=0.5, phi=0.0).set_target(t)
    stored = pairs[(u, t)]
    settings, mismatches = solve_settings(inst, stored)
    # The narrative mask is setting C when it is a real mask, else the strongest solved mask.
    if settings["C"]["status"] in ("optimal", "feasible"):
        mask_for_evidence: set[int] | None = set(settings["C"]["mask_ids"])
        mask_role = "C"
    elif settings["A"]["status"] in ("optimal", "feasible"):
        # Setting C has no mask. Decompose the setting-A mask, which is the one that removes t.
        mask_for_evidence = set(settings["A"]["mask_ids"])
        mask_role = "A"
    else:
        mask_for_evidence = None
        mask_role = "none"

    history = [item_ref(titles, i) for i in inst.H.tolist()]
    original = fill_titles(ranked(inst, inst.s, 13), titles)
    topk = original[:10]
    outsiders = original[10:13]

    def pack_setting(name: str) -> dict:
        block = settings[name]
        mask = block["mask_ids"]
        s = inst.masked_scores(mask) if block["status"] in ("optimal", "feasible", "infeasible") else inst.s
        # An infeasible result has an empty mask; show the unchanged list.
        new = fill_titles(ranked(inst, s if mask or block["status"] != "infeasible" else inst.s, 10), titles)
        if block["status"] == "infeasible":
            new = fill_titles(ranked(inst, inst.s, 10), titles)
        old_ids = {r["item_id"] for r in topk}
        entering = [r for r in new if r["item_id"] not in old_ids]
        return {
            "setting": name,
            "label": {"A": "C1", "B": "C1+C2", "C": "C1+C2+C3"}[name],
            "status": block["status"],
            "runtime_s": block["runtime_s"],
            "mask": [item_ref(titles, i) for i in mask],
            "new_topk": new,
            "entering": entering,
            "verifier_passed": block["verifier_passed"],
            "matches_stored_status_and_size": block["matches_stored_status_and_size"],
            "matches_stored_mask": block["matches_stored_mask"],
            "stored_status": block["stored_status"],
            "stored_size": block["stored_size"],
            "stored_runtime_s": block["stored_runtime_s"],
        }

    packed = {name: pack_setting(name) for name in ("A", "B", "C")}
    retained = [int(j) for j in inst.retained.tolist()]
    evidence = []
    for k, j in enumerate(retained):
        tau = float(inst.tau[k])
        paths = paths_for(model, titles, u, j, tau, mask_for_evidence if mask_role == "C" else None)
        evidence.append({
            "item_id": j,
            "title": titles[j],
            "tau": round(tau, 6),
            "n_qualifying_paths": len(paths),
            "paths": paths,
        })
    t_paths = paths_for(model, titles, u, t, tau=0.0, mask=mask_for_evidence if mask_role == "C" else None)
    # tau=0 keeps every positive path; take the five strongest.
    t_paths = sorted(t_paths, key=lambda r: -r["contribution"])[:5]
    applied = mask_for_evidence or set()
    for p in t_paths:
        p["removed_by_mask"] = int(p["origin_item_id"]) in applied
        p["removed_by_setting"] = mask_role
    decomp = {
        "mask_setting": mask_role,
        "target": {"item_id": t, "title": titles[t], "top": decomposition(inst, titles, t, applied)},
        "retained": [
            {"item_id": j, "title": titles[j], "top": decomposition(inst, titles, j, applied)}
            for j in retained
        ],
    }
    doc = {
        "kind": kind,
        "dataset": "movielens-1m",
        "user_id": u,
        "history_length": int(len(inst.H)),
        "K": 10,
        "alpha": 0.5,
        "eps": EPS,
        "entity_names": "unavailable: the Satori KG in this study is the rehashed RippleNet release, which stores integer ids only",
        "history": history,
        "original_topk": topk,
        "top3_outsiders": outsiders,
        "target": {"item_id": t, "title": titles[t], "rank": int(inst.rank)},
        "settings": packed,
        "evidence": evidence,
        "target_evidence_top5": t_paths,
        "score_decomposition": decomp,
        "mismatches": mismatches,
    }
    if kind == "infeasible":
        failing = "C" if packed["C"]["status"] == "infeasible" else "B"
        report = infeasibility_report(
            inst, failing, eps=EPS,
            outcomes={n: packed[n]["status"] for n in ("A", "B", "C")},
            time_limit=20.0,
        )
        a_ids = {r["item_id"] for r in packed["A"]["new_topk"]}
        dropped_by_a = [item_ref(titles, j) for j in retained if j not in a_ids and j != t]
        releases = items_whose_removal_releases(inst, titles)
        focus = [r["item_id"] for r in releases] or [r["item_id"] for r in dropped_by_a]
        # The sequel and the films the C1 mask actually pushes out are the comparisons worth listing.
        doc["infeasibility"] = {
            "failing_setting": failing,
            "cause": report.get("cause"),
            "certificates": report.get("certificates"),
            "retained_dropped_by_setting_A": dropped_by_a,
            "retained_whose_removal_makes_B_feasible": releases,
            "evidence_blocking_items_structural": [item_ref(titles, j) for j in evidence_blocking_items(inst)],
            "shared_evidence_with_target": shared_with_target(model, titles, u, t, focus),
        }
    return doc, mismatches


def heuristic_case(model, titles, u: int, t: int) -> tuple[dict, list[str]]:
    """Ranking-margin, stop at C1. This mode was not stored in the main run."""
    inst = ExplanationInstance(model, u, K=10, alpha=0.5, phi=0.0).set_target(t)
    t0 = time.perf_counter()
    res = greedy(inst, "C", criterion="margin", mode="c1_stop", eps=EPS)
    elapsed = time.perf_counter() - t0
    mask = norm_mask(res.get("mask"))
    report_c = verify(inst, mask, "C", eps=EPS)
    report_a = verify(inst, mask, "A", eps=EPS)
    s = inst.masked_scores(mask)
    new = fill_titles(ranked(inst, s, 10), titles)
    old = {int(j) for j in inst.L.tolist()}
    retained = [int(j) for j in inst.retained.tolist()]
    dropped = [item_ref(titles, j) for j in retained if j not in {r["item_id"] for r in new}]
    lost = []
    for k, j in enumerate(retained):
        q = set(int(i) for i in inst.Q[k].tolist())
        if q and q.issubset(set(mask)):
            lost.append(item_ref(titles, j))
    notes = []
    if not report_a["passed"]:
        notes.append("ranking-margin C1-stop did not satisfy C1 on this pair")
    if report_c["passed"]:
        notes.append("ranking-margin C1-stop also satisfied C2 and C3; it is not a heuristic failure on this pair")
    else:
        notes.append(f"ranking-margin C1-stop: C1 passed={report_a['passed']}, "
                     f"C2={report_c['c2']}, C3={report_c['c3']}")
    return {
        "kind": "heuristic_failure",
        "method": "greedy:margin:c1_stop",
        "stored_in_main_run": False,
        "note": "The stored main run recorded ranking-margin only in constrained mode, and C1-stop only for the target-score criterion. This block is a fresh call of greedy(), not a stored record.",
        "user_id": u,
        "target": {"item_id": t, "title": titles[t], "rank": int(inst.rank)},
        "mask": [item_ref(titles, i) for i in mask],
        "runtime_s": round(elapsed, 4),
        "new_topk": new,
        "entering": [r for r in new if r["item_id"] not in old],
        "verifier_setting_A": {"passed": bool(report_a["passed"]), "c1": bool(report_a["c1"])},
        "verifier_setting_C": {
            "passed": bool(report_c["passed"]), "c1": bool(report_c["c1"]),
            "c2": report_c["c2"], "c3": report_c["c3"],
        },
        "retained_dropped_from_list": dropped,
        "retained_lost_all_evidence": lost,
    }, notes


def sentence_list(rows: list[dict]) -> str:
    return ", ".join(f"{r['title']}" for r in rows) if rows else "none"


def write_summary(cases: list[dict], notes: list[str]) -> None:
    lines = [
        "# Case studies",
        "",
        "Numbers below are copied from `case_1.json`, `case_2.json` and `case_3.json`.",
        "The main run used here is the certified MovieLens-1M study: 2,000 user–target pairs, "
        "200 users, K = 10, alpha = 0.5. It is not the earlier 800-pair prototype.",
        "Entity names are not available: the Satori file in this study is the rehashed RippleNet "
        "graph, so a shared person or genre is identified by its relation and rehashed id. "
        "The graph stores each link in both directions, so two paths can name the same entity.",
        "",
        "## How the pairs were chosen",
        "",
    ]
    lines.extend(f"- {n}" for n in notes)
    lines.append("")
    for doc in cases:
        if doc["kind"] == "heuristic_failure":
            v = doc["verifier_setting_C"]
            lines += [
                "## Case 3 — ranking-margin stopped at C1",
                "",
                f"Same user and target as case 1: user {doc['user_id']}, {doc['target']['title']}, "
                f"originally rank {doc['target']['rank']}.",
                f"The ranking-margin heuristic in C1-stop mode masked {sentence_list(doc['mask'])} "
                f"({len(doc['mask'])} items) in {doc['runtime_s']} s.",
                f"The verifier on setting A (C1 only) passed: {doc['verifier_setting_A']['passed']}. "
                f"On setting C, C1 = {v['c1']}, C2 = {v['c2']}, C3 = {v['c3']}, passed = {v['passed']}.",
                f"Retained items that left the top-10: {sentence_list(doc['retained_dropped_from_list'])}.",
                f"Retained items that lost every qualifying evidence path: {sentence_list(doc['retained_lost_all_evidence'])}.",
                f"Items that entered the top-10: {sentence_list(doc['entering'])}.",
                doc["note"],
                "",
            ]
            continue
        tgt = doc["target"]
        top = ", ".join(f"{r['rank']}. {r['title']} ({r['score']})" for r in doc["original_topk"][:5])
        why = ", ".join(
            f"{p['origin_title']} via {p['relation_raw']} ({p['via']}, contribution {p['contribution']})"
            for p in doc["target_evidence_top5"][:3]
        )
        lines += [
            f"## Case {'1' if doc['kind']=='success' else '2'} — {doc['kind']}",
            "",
            f"User {doc['user_id']} has a training history of {doc['history_length']} movies. "
            f"K = {doc['K']}, alpha = {doc['alpha']}.",
            f"The target is {tgt['title']}, originally rank {tgt['rank']}.",
            f"The original top of the list: {top}.",
            f"The strongest paths into the target are {why}.",
        ]
        for name in ("A", "B", "C"):
            b = doc["settings"][name]
            lines.append(
                f"Setting {name} ({b['label']}): status {b['status']}, mask size {len(b['mask'])}, "
                f"solve time {b['runtime_s']} s, verifier passed {b['verifier_passed']}, "
                f"matches stored status and size {b['matches_stored_status_and_size']} "
                f"(stored status {b['stored_status']}, stored size {b['stored_size']}, "
                f"stored time {b['stored_runtime_s']} s). "
                f"Masked: {sentence_list(b['mask'])}. Entering: {sentence_list(b['entering'])}."
            )
        top_support = ", ".join(
            f"{r['title']} ({r['A_ij_before']})" for r in doc["score_decomposition"]["target"]["top"][:3]
        )
        lines.append(f"The three history movies that contribute most to the target's score are {top_support}.")
        if doc["kind"] == "success":
            kept = doc["settings"]["C"]["new_topk"]
            retained_ids = {e["item_id"] for e in doc["evidence"]}
            stayed = [r for r in kept if r["item_id"] in retained_ids]
            lines.append(f"After the setting-C mask the top-10 is: {sentence_list(kept)}.")
            lines.append(f"The retained movies that stay are: {sentence_list(stayed)}.")
        if doc.get("infeasibility"):
            inf = doc["infeasibility"]
            lines.append(f"Infeasibility of setting {inf['failing_setting']}: {inf['cause']}.")
            lines.append(
                "The optimal C1 mask removes the target and also pushes out: "
                f"{sentence_list(inf['retained_dropped_by_setting_A'])}."
            )
            release_ids = {r["item_id"] for r in inf["retained_whose_removal_makes_B_feasible"]}
            collateral = [r for r in inf["retained_dropped_by_setting_A"] if r["item_id"] not in release_ids]
            if collateral:
                noun = "that movie" if len(collateral) == 1 else "those movies"
                lines.append(
                    f"{sentence_list(collateral)} also leaves under this particular C1 mask, "
                    f"but allowing {noun} to leave does not by itself make C1+C2 feasible."
                )
            releases = inf["retained_whose_removal_makes_B_feasible"]
            if releases:
                bits = ", ".join(
                    f"{r['title']} (B then {r['status_if_this_item_may_leave']}, size {r['mask_size']}, "
                    f"verifier {r['verifier_passed']})"
                    for r in releases
                )
                lines.append(
                    "C1+C2 becomes feasible if any one of these retained movies is allowed to leave: " + bits + "."
                )
            else:
                lines.append("Allowing any single retained movie to leave does not make C1+C2 feasible.")
            for row in inf["shared_evidence_with_target"]:
                if row["shared_person"]:
                    bits = ", ".join(
                        f"{s['relation']} {s['via']} from {s['origin_title']} "
                        f"(contribution to this movie {s['contribution_to_j']}, to the target {s['contribution_to_t']})"
                        for s in row["shared_person"][:3]
                    )
                    lines.append(f"{row['title']} shares a person with the target: {bits}.")
                else:
                    extra = ""
                    if row["shared_other"]:
                        s = row["shared_other"][0]
                        extra = (
                            f" The strongest overlapping link is {s['relation']} {s['via']} from {s['origin_title']} "
                            f"(contribution to this movie {s['contribution_to_j']}, to the target {s['contribution_to_t']})."
                        )
                    lines.append(
                        f"{row['title']} shares no actor, director or writer with the target in this graph."
                        + extra
                    )
        if doc["mismatches"]:
            lines.append("Mismatches against the stored main run:")
            lines.extend(f"- {m}" for m in doc["mismatches"])
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serialisable: {type(obj)}")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "movielens.yaml")
    ds = load_or_prepare(cfg)
    titles = load_titles(ds)
    from epic_rec.experiments.training import load_model
    model = load_model(cfg, ds)
    pairs = load_exact()
    print(f"exact pairs in the stored main run: {len(pairs)}", flush=True)
    pool, notes = success_pool(pairs, titles)
    for n in notes:
        print(n, flush=True)
    chosen = None
    fallback = None
    for cand in pool[:40]:
        h, hnotes = heuristic_case(model, titles, cand["user"], cand["target"])
        fails = h["verifier_setting_A"]["passed"] and not h["verifier_setting_C"]["passed"]
        line = (f"heuristic screen user {cand['user']} '{cand['title']}': "
                f"C1={h['verifier_setting_A']['passed']} C2={h['verifier_setting_C']['c2']} "
                f"C3={h['verifier_setting_C']['c3']}")
        print(line, flush=True)
        notes.append(line)
        if not fails:
            continue
        if fallback is None:
            fallback = (cand, h, hnotes)
        same = any(
            uu == cand["user"] and tt != cand["target"] and proven(by.get("A"), "optimal")
            and (proven(by.get("B"), "infeasible") or proven(by.get("C"), "infeasible"))
            for (uu, tt), by in pairs.items()
        )
        if same:
            notes.append("this candidate also has a same-user infeasible target; selecting it")
            chosen = (cand, h, hnotes)
            break
    if chosen is None and fallback is not None:
        notes.append("no heuristic failure shared a user with a proven infeasible target; using the first failure")
        chosen = fallback
    if chosen is None:
        notes.append("no screened candidate was a ranking-margin C1-stop failure; using the top success candidate")
        h, hnotes = heuristic_case(model, titles, pool[0]["user"], pool[0]["target"])
        chosen = (pool[0], h, hnotes)
    success, heur, hnotes = chosen
    infeasible, inotes = infeasible_for(pairs, titles, success["user"], success["target"], success["hist"])
    notes.extend(inotes)
    notes.extend(hnotes)
    notes.append(f"SELECTED success: user {success['user']} '{success['title']}' |M_C|={success['size']}")
    notes.append(f"SELECTED infeasible: user {infeasible['user']} '{infeasible['title']}' "
                 f"same_user={infeasible['same_user']} B={infeasible['B']} C={infeasible['C']}")
    for n in notes[-6:]:
        print(n, flush=True)
    OUT.mkdir(exist_ok=True)
    case1, mis1 = export_case(model, titles, success, pairs, "success")
    case2, mis2 = export_case(model, titles, infeasible, pairs, "infeasible")
    case1["mismatches"] = mis1
    case2["mismatches"] = mis2
    for n, doc in ((1, case1), (2, case2), (3, heur)):
        path = OUT / f"case_{n}.json"
        path.write_text(json.dumps(doc, indent=2, default=_json_default) + "\n")
        print(f"wrote {path}")
    write_summary([case1, case2, heur], notes)
    unresolved = [h["title"] for h in case1["history"] + case2["history"] if h["title"].startswith("UNRESOLVED")]
    print("unresolved titles:", unresolved or "none")
    print("case 1 mismatches:", mis1 or "none")
    print("case 2 mismatches:", mis2 or "none")
    print("heuristic notes:", hnotes)


if __name__ == "__main__":
    main()
