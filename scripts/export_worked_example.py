"""Export one K=5 MovieLens example for the manuscript's worked example.

The pair is chosen from the stored K-sweep (exact method, setting C, cross-check
outcome confirmed). It is illustrative. The script re-solves it, checks the mask
with the independent verifier, and enumerates P and Pε when the mask domain is
small enough. It does not change stored results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from epic_rec.data.preprocess import load_or_prepare
from epic_rec.evaluation.verify import verify
from epic_rec.explainers.greedy import greedy
from epic_rec.experiments.training import load_model
from epic_rec.optimization.exhaustive import exhaustive
from epic_rec.optimization.instance import EPS, ExplanationInstance
from epic_rec.optimization.milp import solve_milp
from epic_rec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "movielens-1m" / "k_sweep" / "records.jsonl"
MOVIES = ROOT / "data" / "raw" / "movielens-1m" / "movies.dat"
OUT_DIR = ROOT / "case_study"
SVG = ROOT / "manuscript" / "fig_worked.svg"
TEX = ROOT / "manuscript" / "case_worked.tex"

FAMOUS = {
    "pulp fiction", "the godfather", "jurassic park", "the matrix", "forrest gump",
    "star wars: episode iv - a new hope", "star wars: episode v - the empire strikes back",
    "star wars: episode vi - return of the jedi", "die hard", "terminator 2: judgment day",
    "the silence of the lambs", "schindler's list", "the shawshank redemption",
    "raiders of the lost ark", "back to the future", "toy story", "the lion king",
    "saving private ryan", "gladiator", "alien", "goodfellas", "fargo", "titanic",
    "casablanca", "the usual suspects", "braveheart", "aladdin", "e.t. the extra-terrestrial",
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
    "co": "collaborative",
}


def titles_for(ds) -> dict[int, str]:
    raw = {}
    for line in MOVIES.read_text(encoding="latin-1").splitlines():
        mid, title, _g = line.split("::", 2)
        raw[int(mid)] = title
    out = {}
    missing = 0
    for i, ext in enumerate(ds.item_ids):
        title = raw.get(int(ext))
        if title is None:
            missing += 1
            title = f"UNRESOLVED {int(ext)}"
        out[i] = title
    if missing:
        raise SystemExit(f"{missing} titles did not resolve")
    return out


def bare(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip().lower()


def short(title: str, n: int = 22) -> str:
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    return title if len(title) <= n else title[: n - 1] + "…"


def select(titles: dict[int, str]) -> dict:
    by: dict[tuple[int, int], dict] = {}
    with RECORDS.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("method") != "exact" or r.get("K") != 5 or abs(r.get("alpha", 1) - 0.5) > 1e-12:
                continue
            by.setdefault((r["user"], r["target"]), {})[r["setting"]] = r
    pool = []
    for (u, t), settings in by.items():
        c = settings.get("C")
        if not c or c.get("status") != "optimal":
            continue
        if c.get("certification") != "confirmed":
            continue
        if c.get("size") not in (1, 2) or not (12 <= c.get("hist_len", 0) <= 40):
            continue
        if c.get("domain_size", 99) > 16:
            continue
        title = titles[t]
        pool.append({
            "user": u, "target": t, "title": title, "famous": bare(title) in FAMOUS,
            "size": c["size"], "hist": c["hist_len"], "rank": c["rank"],
            "domain": c["domain_size"], "mask": c["mask"], "settings": {
                name: {"status": settings[name]["status"], "size": settings[name]["size"], "mask": settings[name]["mask"]}
                for name in ("A", "B", "C") if name in settings
            },
        })
    if not pool:
        raise SystemExit("no enumerable K=5 example in the stored sweep")
    pool.sort(key=lambda r: (not r["famous"], r["size"] != 1, abs((r["rank"] or 5) - 4), r["hist"]))
    print(f"enumerable confirmed K=5 candidates: {len(pool)}; famous: {sum(r['famous'] for r in pool)}")
    for r in pool[:8]:
        print(f"  user {r['user']} '{r['title']}' rank {r['rank']} |M|={r['size']} history {r['hist']} domain {r['domain']}")
    return pool[0]


def top_rows(inst: ExplanationInstance, s: np.ndarray) -> list[dict]:
    rows = []
    for rank, j in enumerate(inst.topk(s, K=5), start=1):
        rows.append({"rank": rank, "item_id": int(j), "score": round(float(s[int(j)]), 4)})
    return rows


def one_path(model, u: int, j: int, tau: float, mask: set[int]) -> dict | None:
    best = None
    for g in model.path_groups(u, j):
        rel = RELATION.get(g.relation, g.relation)
        for p in g.paths():
            if p.contribution + 1e-15 < tau:
                continue
            if int(p.origin) in mask:
                continue
            row = {
                "origin": int(p.origin),
                "relation": rel,
                "relation_raw": g.relation,
                "via": "co-user" if p.via_type == "user" else f"rehashed-entity-{int(p.via)}",
                "contribution": round(float(p.contribution), 6),
            }
            if best is None or row["contribution"] > best["contribution"]:
                best = row
    return best


def svg(doc: dict) -> str:
    """Three panels: before, masked interactions, after. Colours match Figure 1."""
    def panel(x0, title, rows, marks):
        bits = [f'<text x="{x0}" y="36" font-size="18" font-weight="700" fill="#111">{title}</text>']
        for i, row in enumerate(rows):
            y = 64 + i * 36
            bits.append(f'<rect x="{x0}" y="{y}" width="250" height="30" rx="6" fill="{row["fill"]}"/>')
            bits.append(
                f'<text x="{x0+8}" y="{y+20}" font-size="14" fill="{row["ink"]}" font-weight="700">'
                f'{row["text"]}</text>'
            )
            if i < len(marks) and marks[i]:
                bits.append(f'<text x="{x0+258}" y="{y+20}" font-size="16" fill="#2e8b57" font-weight="700">{marks[i]}</text>')
        return bits

    before = []
    for r in doc["original_topk"]:
        target = r["item_id"] == doc["target"]["item_id"]
        before.append({
            "fill": "#d1495b" if target else "#d9e6f2",
            "ink": "#fff" if target else "#1f3c5a",
            "text": f'{r["rank"]}. {short(r["title"])}  {r["score"]:.3f}',
        })
    mask_rows = []
    for r in doc["mask"]:
        mask_rows.append({
            "fill": "#f6d5d9",
            "ink": "#8c1d2c",
            "text": short(r["title"], 24),
        })
    mask_rows.append({
        "fill": "#fff",
        "ink": "#333",
        "text": f'status {doc["settings"]["C"]["status"]}, |M| = {doc["settings"]["C"]["size"]}',
    })
    after = []
    old = {r["item_id"] for r in doc["original_topk"]}
    retained = {r["item_id"] for r in doc["original_topk"] if r["item_id"] != doc["target"]["item_id"]}
    marks = []
    for r in doc["settings"]["C"]["new_topk"]:
        if r["item_id"] == doc["target"]["item_id"]:
            fill, ink, mark = "#fff", "#d1495b", ""
        elif r["item_id"] not in old:
            fill, ink, mark = "#f2c14e", "#5a4210", "new"
        else:
            fill, ink, mark = "#d9e6f2", "#1f3c5a", "✓" if r["item_id"] in retained else ""
        after.append({"fill": fill, "ink": ink, "text": f'{r["rank"]}. {short(r["title"])}  {r["score"]:.3f}'})
        marks.append(mark)
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 280" width="980" height="280" font-family="Helvetica, Arial, sans-serif">',
        '<rect width="980" height="280" fill="#fff"/>',
        *panel(16, "Before", before, [""] * 5),
        '<text x="300" y="150" font-size="22" fill="#444">→</text>',
        *panel(330, "Masked history", mask_rows, [""] * len(mask_rows)),
        '<text x="640" y="150" font-size="22" fill="#444">→</text>',
        *panel(690, "After", after, marks),
        "</svg>",
    ]
    return "\n".join(body) + "\n"


def tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def write_tex(doc: dict) -> None:
    tgt = doc["target"]
    c = doc["settings"]["C"]
    masked = ", ".join(tex_escape(m["title"]) for m in doc["mask"])
    orig = ", ".join(
        f"{r['rank']}.\\ \\emph{{{tex_escape(r['title'])}}} ({r['score']:.4f})" for r in doc["original_topk"]
    )
    new = ", ".join(
        f"{r['rank']}.\\ \\emph{{{tex_escape(r['title'])}}} ({r['score']:.4f})" for r in c["new_topk"]
    )
    ev_rows = []
    for e in doc["retained_evidence"]:
        p = e["surviving_path"]
        via = tex_escape(p["via"]) if p else "none"
        rel = tex_escape(p["relation"]) if p else "none"
        origin = tex_escape(p["origin_title"]) if p else "none"
        contrib = f"{p['contribution']:.6f}" if p else "--"
        ev_rows.append(
            f"{tex_escape(e['title'])} & {e['tau']:.6f} & {rel} & {origin} & {via} & {contrib} & yes \\\\"
        )
    heur = doc["heuristic"]
    heur_sentence = (
        f"On this pair the target-score heuristic that stops at removal masks "
        f"{', '.join(tex_escape(m['title']) for m in heur['mask']) or 'nothing'}. "
        f"The verifier reports C1 = {heur['c1']}, C2 = {heur['c2']}, C3 = {heur['c3']}."
    )
    if heur["c1"] and not (heur["c2"] and heur["c3"]):
        dropped = ", ".join(tex_escape(m["title"]) for m in heur["dropped"]) or "none"
        lost = ", ".join(tex_escape(m["title"]) for m in heur["lost_evidence"]) or "none"
        heur_sentence += f" Collateral damage on this pair: displaced retained movies: {dropped}; retained movies that lose every qualifying path: {lost}."
    else:
        heur_sentence += " This particular removal happens to satisfy C2 and C3 as well, so it is not an example of collateral damage. The main-sample rate, not this pair, is the evidence that removal-only edits usually do not."
    enum = doc["enumeration"]
    if enum["ran"]:
        mini = (
            f"Enumeration of every subset of $D_u(t)$ ({enum['domain_size']} interactions) finds the same "
            f"minimum size under $P$ ({enum['P_size']}) and under $P_\\varepsilon$ ({enum['Peps_size']}). "
            f"No smaller permitted mask satisfies C1+C2+C3 under either formulation."
        )
    else:
        mini = (
            "The mask domain was too large to enumerate. The size is the HiGHS optimum for $P_\\varepsilon$ "
            "on $D_u(t)$ after the second configuration found no smaller verified mask. That is not an "
            "enumeration of $P$, and it is not a formal optimality certificate."
        )
    plain_mask = " and ".join(tex_escape(m["title"]) for m in doc["mask"])
    lines = [
        r"\subsection{Worked example: removing one recommendation while preserving the others}",
        r"\label{sec:worked}",
        "",
        "The example is one MovieLens pair from the stored $K=5$ sweep (exact method, $\\alpha=0.5$, "
        "$\\varepsilon=10^{-6}$, mask domain $D_u(t)$). It was chosen because setting C is recorded as optimal "
        f"with a mask of size {c['size']}, the history has {doc['history_length']} movies, the mask domain has "
        f"{doc['domain_size']} interactions, and the cross-check status is confirmed. "
        f"{'The target is a widely known title. ' if doc['famous'] else ''}"
        "It illustrates the procedure. It is not an estimate of average mask size or feasibility.",
        "",
        f"User {doc['user_id']} is recommended, in order, {orig}. "
        f"The target is \\emph{{{tex_escape(tgt['title'])}}}, originally rank {tgt['rank']}, score {tgt['score']:.4f}. "
        f"The three history movies that contribute most to its score are "
        + ", ".join(f"\\emph{{{tex_escape(r['title'])}}} ({r['A_ij_before']:.4f})" for r in doc["target_contributions"][:3])
        + ".",
        "",
    ]
    top_masked = [r for r in doc["target_contributions"][:3] if r.get("masked")]
    if top_masked:
        lines.append(
            "The masked "
            + ("interaction is" if len(doc["mask"]) == 1 else "interactions include")
            + " "
            + ", ".join(f"\\emph{{{tex_escape(r['title'])}}}" for r in top_masked)
            + ", which is among those three contributors. Its contribution is removed and the others stay."
        )
        lines.append("")
    lines += [
        f"Under C1+C2+C3 the solver returns $M$ of size {c['size']}: {masked}. "
        f"The study configuration reports status {c['status']} in {c['runtime_s']} seconds. "
        f"The independent verifier accepts the mask under both $P$ and $P_\\varepsilon$ "
        f"(C1 = {c['verifier']['c1']}, C2 = {c['verifier']['c2']}, C3 = {c['verifier']['c3']}). "
        f"The stored $K=5$ record has status {c['stored_status']}, size {c['stored_size']}, "
        f"and the masks {'match' if c['matches_stored_mask'] else 'do not match'}. "
        + mini,
        "",
        f"After the mask the top five are {new}. "
        f"\\emph{{{tex_escape(tgt['title'])}}} is not among them. "
        f"The other four original movies remain. "
        f"The replacement is \\emph{{{tex_escape(doc['entering'][0]['title'])}}} "
        f"(rank {doc['entering'][0]['rank']}, score {doc['entering'][0]['score']:.4f}). "
        f"Kendall's $\\tau_a$ on the four retained movies, before against after, is {doc['kendall_retained']:.3f}.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Surviving evidence for the four movies that stay in the worked example. "
        r"Each row is one original path whose contribution is at least $\tau_j$ and whose origin is not masked. "
        r"Entity names are not in the rehashed Satori release, so the intermediate node is a relation and a rehashed id, or ``co-user''.}",
        r"\label{tab:worked}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Retained movie & $\tau_j$ & relation & origin & via & $c(p)$ & survives \\",
        r"\midrule",
        *ev_rows,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
        f"Masking {plain_mask} removes \\emph{{{tex_escape(tgt['title'])}}} from the top-five recommendations. "
        "The remaining four original movies stay in the list, and each retains at least one original evidence "
        "path meeting its contribution threshold. The edit is a change to the inputs of the scorer, followed by "
        "re-ranking. It is not a change to the interface, and this example is not evidence about privacy, "
        "conversion or trust.",
        "",
        heur_sentence,
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{fig_worked.pdf}",
        r"\caption{Worked example, $K=5$. Left: the original list; the target is red. "
        r"Centre: the masked history interaction. Right: the list after re-ranking. "
        r"Blue rows are retained movies, the gold row is the replacement, and a tick marks a retained movie "
        r"that keeps a qualifying path. Scores are the path-additive scores.}",
        r"\label{fig:worked}",
        r"\end{figure}",
        "",
    ]
    TEX.write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "movielens.yaml")
    ds = load_or_prepare(cfg)
    titles = titles_for(ds)
    spec = select(titles)
    u, t = spec["user"], spec["target"]
    inst = ExplanationInstance(model := load_model(cfg, ds), u, K=5, alpha=0.5, phi=0.0).set_target(t)
    settings = {}
    mismatches = []
    for name in ("A", "B", "C"):
        res = solve_milp(inst, name, eps=EPS, time_limit=20.0, threads=1)
        mask = sorted(int(i) for i in (res["mask"] or []))
        report = verify(inst, mask, name, eps=EPS) if res["status"] in ("optimal", "feasible") else None
        stored = spec["settings"][name]
        same = stored["status"] == res["status"] and stored["size"] == res["size"] and sorted(stored["mask"] or []) == mask
        if not same:
            mismatches.append(f"{name}: rerun {res['status']} {mask}; stored {stored}")
        new = top_rows(inst, inst.masked_scores(mask) if mask else inst.s)
        for row in new:
            row["title"] = titles[row["item_id"]]
        settings[name] = {
            "status": res["status"], "size": res["size"], "runtime_s": round(float(res["time"]), 4),
            "mask": [{"item_id": i, "title": titles[i]} for i in mask],
            "new_topk": new,
            "verifier": None if report is None else {
                "passed": bool(report["passed"]), "c1": bool(report["c1"]),
                "c2": bool(report["c2"]), "c3": report["c3"],
            },
            "stored_status": stored["status"], "stored_size": stored["size"],
            "matches_stored_mask": same,
            "kendall_retained": None if report is None else round(float(report["kendall_retained"]), 4),
        }
        if report is not None and not report["passed"]:
            mismatches.append(f"{name}: verifier rejected the mask")
    cmask = {m["item_id"] for m in settings["C"]["mask"]}
    original = top_rows(inst, inst.s)
    for row in original:
        row["title"] = titles[row["item_id"]]
    retained = []
    for k, j in enumerate(inst.retained.tolist()):
        path = one_path(model, u, int(j), float(inst.tau[k]), cmask)
        if path:
            path["origin_title"] = titles[path["origin"]]
        retained.append({
            "item_id": int(j), "title": titles[int(j)], "tau": round(float(inst.tau[k]), 6),
            "surviving_path": path,
        })
    if any(e["surviving_path"] is None for e in retained):
        mismatches.append("a retained movie has no surviving qualifying path in the export")
    col = inst.A[:, t]
    order = np.argsort(-col)[:6]
    contributions = []
    for k in order:
        if col[k] <= 0:
            continue
        i = int(inst.H[k])
        contributions.append({
            "item_id": i, "title": titles[i],
            "A_ij_before": round(float(col[k]), 6),
            "A_ij_after": 0.0 if i in cmask else round(float(col[k]), 6),
            "masked": i in cmask,
        })
    enum_p = exhaustive(inst, "C", max_n=16, margin=0.0)
    enum_e = exhaustive(inst, "C", max_n=16, margin=EPS)
    enumeration = {"ran": enum_p is not None and enum_e is not None, "domain_size": int(len(inst.scope))}
    if enumeration["ran"]:
        enumeration["P_size"] = enum_p["size"]
        enumeration["Peps_size"] = enum_e["size"]
        enumeration["P_mask"] = enum_p["mask"]
        enumeration["Peps_mask"] = enum_e["mask"]
        if enum_e["size"] != settings["C"]["size"]:
            mismatches.append(f"enumeration Peps size {enum_e['size']} != solver size {settings['C']['size']}")
    h = greedy(inst, "C", criterion="target_score", mode="c1_stop", eps=EPS)
    hmask = sorted(int(i) for i in (h.get("mask") or []))
    hv = verify(inst, hmask, "C", eps=EPS)
    hs = inst.masked_scores(hmask)
    htop = {int(j) for j in inst.topk(hs).tolist()}
    dropped = [{"item_id": int(j), "title": titles[int(j)]} for j in inst.retained.tolist() if int(j) not in htop]
    lost = []
    hset = set(hmask)
    for k, j in enumerate(inst.retained.tolist()):
        q = {int(i) for i in inst.Q[k].tolist()}
        if q and q.issubset(hset):
            lost.append({"item_id": int(j), "title": titles[int(j)]})
    old_ids = {r["item_id"] for r in original}
    entering = [r for r in settings["C"]["new_topk"] if r["item_id"] not in old_ids]
    doc = {
        "kind": "worked_example_k5",
        "illustrative": True,
        "source": "results/movielens-1m/k_sweep exact records, re-solved",
        "dataset": "movielens-1m",
        "user_id": u,
        "history_length": int(len(inst.H)),
        "domain_size": int(len(inst.scope)),
        "K": 5,
        "alpha": 0.5,
        "phi": 0.0,
        "eps": EPS,
        "famous": spec["famous"],
        "history": [{"item_id": int(i), "title": titles[int(i)]} for i in inst.H.tolist()],
        "original_topk": original,
        "target": {"item_id": t, "title": titles[t], "rank": int(inst.rank),
                   "score": round(float(inst.s[t]), 4)},
        "mask": settings["C"]["mask"],
        "target_contributions": contributions,
        "settings": settings,
        "entering": entering,
        "kendall_retained": settings["C"]["kendall_retained"],
        "retained_evidence": retained,
        "enumeration": enumeration,
        "heuristic": {
            "method": "greedy:target_score:c1_stop",
            "stored": False,
            "mask": [{"item_id": i, "title": titles[i]} for i in hmask],
            "c1": bool(hv["c1"]), "c2": bool(hv["c2"]), "c3": hv["c3"],
            "passed_setting_C": bool(hv["passed"]),
            "dropped": dropped, "lost_evidence": lost,
        },
        "mismatches": mismatches,
        "entity_names": "unavailable in the rehashed Satori release",
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "worked_k5.json").write_text(json.dumps(doc, indent=2) + "\n")
    SVG.write_text(svg(doc))
    write_tex(doc)
    print("SELECTED", spec["user"], spec["title"], "size", spec["size"], "famous", spec["famous"])
    print("mismatches:", mismatches or "none")
    print("enumeration", enumeration)
    print("heuristic", doc["heuristic"]["c1"], doc["heuristic"]["c2"], doc["heuristic"]["c3"],
          [m["title"] for m in dropped])


if __name__ == "__main__":
    main()
