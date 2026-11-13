"""Configurable, resumable experiment runner.

One *job* is (user, K, alpha, phi). A job builds the user's problem instance once and, for every target rank,
score margin eps, constraint setting and method in the config, writes one record (a JSON line) to
``<output_dir>/records.jsonl``. Jobs run in parallel worker processes; every exact solve additionally runs in its
own child process with a hard wall-clock limit (:func:`epic_rec.optimization.run_isolated`). A restarted run
skips jobs whose records are already in the file.

The record schema is documented in ``docs/results_schema.md``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from epic_rec.evaluation.verify import verify
from epic_rec.explainers.api import explain
from epic_rec.optimization.certify import certify_isolated
from epic_rec.optimization.diagnose import infeasibility_report
from epic_rec.optimization.exhaustive import exhaustive
from epic_rec.optimization.instance import ExplanationInstance, get_setting
from epic_rec.utils.io import JsonlWriter, atomic_write_json, dumps
from epic_rec.utils.logging import get_logger
from epic_rec.utils.seeding import rng_for

log = get_logger(__name__)
_MODEL: Any = None
_CFG: dict = {}
_CERT_FIELDS = ("certification", "original_status", "original_size", "certify_time")


def method_id(spec: dict) -> str:
    name = spec["name"]
    if name == "greedy":
        return f"greedy:{spec.get('criterion', 'violation')}" + (":c1_stop" if spec.get("mode") == "c1_stop" else "")
    if name == "continuous":
        return ":".join(["continuous", spec.get("relaxation", "sigmoid"), spec.get("init", "attribution"),
                         "repair" if spec.get("repair", True) else "norepair"])
    return name


def _method_call(spec: dict) -> tuple[str, dict]:
    name = spec["name"]
    kw = {k: v for k, v in spec.items() if k not in ("name", "criterion", "mode")}
    if name == "greedy":
        return f"greedy:{spec.get('criterion', 'violation')}:{spec.get('mode', 'constrained')}", kw
    return name, kw


def sample_users(model: Any, per_stratum: int | None, seed: int, min_history: int = 1
                 ) -> tuple[list[tuple[int, int]], list[float]]:
    """Users stratified by training-history quartile; ``per_stratum=None`` takes every user."""
    hist = model.hist_len
    cuts = np.quantile(hist, [0.25, 0.5, 0.75]).tolist()
    strata = np.digitize(hist, cuts, right=True)
    out = []
    for q in range(4):
        pool = np.flatnonzero((strata == q) & (hist >= min_history))
        if per_stratum is not None and len(pool) > per_stratum:
            pool = np.sort(rng_for(seed, "users", q).choice(pool, per_stratum, replace=False))
        out += [(int(u), q) for u in pool]
    return out, cuts


def _summarise_ver(ver: dict | None) -> dict[str, Any]:
    if ver is None:
        return dict(c1=None, c2=None, c3=None, c1_margin=None, c1_c2_margin=None, passed_P=None, overlap=None,
                    kendall=None, evidence_survival=None,
                    surviving_paths_mean=None, strongest_min=None, decomposition_error=None, passed_P_eps=None)
    ev = ver.get("evidence") or []
    sp_ = [e["surviving_paths"] for e in ev if e.get("surviving_paths") is not None]
    st_ = [e["strongest"] for e in ev if e.get("strongest") is not None]
    return dict(c1=ver["c1"], c2=ver["c2"], c3=ver["c3"], c1_margin=ver["c1_margin"],
                c1_c2_margin=ver["c1_c2_margin"], passed_P=ver["passed_P"], overlap=ver["overlap"],
                kendall=ver["kendall_retained"],
                evidence_survival=ver["evidence_survival"],
                surviving_paths_mean=float(np.mean(sp_)) if sp_ else None,
                strongest_min=float(np.min(st_)) if st_ else None,
                decomposition_error=ver["decomposition_error"], passed_P_eps=ver["passed_P_eps"])


def run_job(job: tuple) -> list[dict]:
    u, stratum, K, alpha, phi = job
    cfg = _CFG
    ex = cfg["experiment"]
    model = _MODEL
    inst = ExplanationInstance(model, u, K=K, alpha=alpha, phi=phi, scope=ex.get("scope", "target"))
    ranks = ex.get("ranks", "all")
    ranks = list(range(1, K + 1)) if ranks == "all" else [r for r in ranks if r <= K]
    settings = [get_setting(s) for s in ex.get("settings", ["A", "B", "C"])]
    tl = float(ex.get("time_limit", 20.0))
    float(ex.get("hard_kill_grace", 40.0))
    enum_max = int(ex.get("enumerate_max", 0))
    records = []
    base = dict(dataset=cfg["dataset"]["name"], experiment=ex["name"], user=u, stratum=stratum,
                hist_len=int(model.hist_len[u]), K=K, alpha=alpha, phi=phi)
    for rank in ranks:
        t = int(inst.L[rank - 1])
        inst.set_target(t)
        ent = inst.entanglement()
        pair = dict(base, rank=rank, target=t, domain_size=len(inst.scope), evidence_free=inst.evidence_free,
                    **{f"ent_{k}": v for k, v in ent.items()})
        for eps in ex.get("eps", [1e-6]):
            exact_status: dict[str, str] = {}
            for st in settings:
                for spec in ex.get("methods", [{"name": "exact"}]):
                    mid = method_id(spec)
                    if spec.get("settings") and st.name not in spec["settings"]:
                        continue
                    name, kw = _method_call(spec)
                    rng = rng_for(cfg.get("seed", 0), "method", mid, u, t, st.name)
                    if name == "exact":
                        e = explain(inst, st, "exact", eps=eps, time_limit=tl, isolate=True, diagnose=False,
                                    evidence=True, certify_limit=float(ex.get("certify_time_limit", 60.0)))
                        exact_status[st.name] = e.status
                    else:
                        kw = {k: v for k, v in kw.items() if k != "settings"}
                        e = explain(inst, st, name, eps=eps, rng=rng, evidence=True, **kw)
                    rec = dict(pair, eps=eps, setting=st.name, setting_label=st.label, method=mid,
                               status=e.status, success=e.success, size=e.size, bound=e.bound, gap=e.gap,
                               runtime=round(e.runtime, 4), killed=bool(e.details.get("killed", False)),
                               iterations=e.details.get("iterations"), calls=e.details.get("calls"),
                               mask=e.mask, **_summarise_ver(e.verification))
                    if "details" in e.details:
                        rec["method_details"] = e.details["details"]
                    if name == "exact":
                        if "certification" in e.details:
                            rec.update({k: e.details[k] for k in _CERT_FIELDS})
                            rec["certify_time"] = round(rec["certify_time"], 4)
                        if enum_max and len(inst.scope) <= enum_max:
                            for sem, mg in (("P", 0.0), ("Peps", eps)):
                                r = exhaustive(inst, st, enum_max, margin=mg)
                                rec[f"enum_{sem}"] = r["size"] if r["status"] == "optimal" else -1
                        if e.status == "infeasible" and ex.get("diagnose", True):
                            rep = infeasibility_report(inst, st, eps=eps, outcomes=dict(exact_status),
                                                       time_limit=min(tl, 10.0))
                            rec["infeasibility_cause"] = rep["cause"]
                            rec["certificates"] = [c["kind"] for c in rep["certificates"]]
                            rec["evidence_blocking_items"] = len(rep["evidence_blocking_items"])
                    records.append(rec)
    return records


def _worker(job: tuple, path: str) -> None:
    recs = run_job(job)
    tmp = path + ".part"
    with open(tmp, "w") as f:
        f.write("".join(dumps(r) + "\n" for r in recs))
    os.replace(tmp, path)


def run_experiment(cfg: dict, model: Any) -> Path:
    """Run (or resume) the experiment in ``cfg['experiment']``; returns the records path."""
    global _MODEL, _CFG
    _MODEL, _CFG = model, cfg
    ex = cfg["experiment"]
    out = Path(ex.get("output_dir", f"results/{cfg['dataset']['name']}/{ex['name']}"))
    out.mkdir(parents=True, exist_ok=True)
    users, cuts = sample_users(model, ex.get("users_per_stratum"), ex.get("user_seed", cfg.get("seed", 0)))
    if ex.get("max_users"):
        users = users[: ex["max_users"]]
    grid = [(u, q, K, a, p) for K in ex.get("K", [10]) for a in ex.get("alpha", [0.5]) for p in ex.get("phi", [0.0])
            for u, q in users]
    atomic_write_json(out / "sample.json", dict(users=users, history_quartile_cuts=cuts, config=cfg))
    writer = JsonlWriter(out / "records.jsonl")
    done = writer.done_keys(("user", "K", "alpha", "phi"))
    jobs = [j for j in grid if (j[0], j[2], j[3], j[4]) not in done]
    log.info("%s: %d jobs (%d already done), %d users", ex["name"], len(jobs), len(grid) - len(jobs), len(users))
    workers = int(os.environ.get("EPIC_WORKERS", ex.get("workers", 1)))
    tmpdir = out / "tmp"
    tmpdir.mkdir(exist_ok=True)
    t0 = time.time()
    if workers <= 1 or "fork" not in mp.get_all_start_methods():
        for k, job in enumerate(jobs, 1):
            writer.write(run_job(job))
            log.info("%d/%d jobs done (%.0fs)", k, len(jobs), time.time() - t0)
        return out / "records.jsonl"
    ctx = mp.get_context("fork")
    running: dict[Any, str] = {}
    queue = list(jobs)
    k = 0
    while queue or running:
        while queue and len(running) < workers:
            job = queue.pop(0)
            fn = str(tmpdir / f"job_{job[0]}_{job[2]}_{job[3]}_{job[4]}.jsonl")
            p = ctx.Process(target=_worker, args=(job, fn))
            p.start()
            running[p] = fn
        time.sleep(0.05)
        for p in list(running):
            if not p.is_alive():
                fn = running.pop(p)
                p.join()
                if os.path.exists(fn):
                    with open(fn) as f:
                        writer.write(json.loads(line) for line in f if line.strip())
                    os.remove(fn)
                    k += 1
                    log.info("%d/%d jobs done (%.0fs)", k, len(jobs), time.time() - t0)
                else:
                    log.error("job for %s failed (exit code %s); it will be retried on resume", fn, p.exitcode)
    return out / "records.jsonl"


def run_prince(cfg: dict, ds: Any) -> Path:
    """PRINCE on its native PPR recommender for a stratified user sample (top-1 targets)."""
    from epic_rec.baselines.ppr import PPRRecommender
    from epic_rec.baselines.prince import PrinceAdapter

    pc = cfg.get("prince", {})
    ex = cfg["experiment"]
    out = Path(ex.get("output_dir", f"results/{cfg['dataset']['name']}/{ex['name']}"))
    out.mkdir(parents=True, exist_ok=True)
    rec = PPRRecommender(ds, alpha=pc.get("alpha", 0.15))
    adapter = PrinceAdapter(rec, epsilon=pc.get("epsilon"))
    hist = np.diff(ds.matrix("train").indptr)
    cuts = np.quantile(hist, [0.25, 0.5, 0.75])
    strata = np.digitize(hist, cuts, right=True)
    writer = JsonlWriter(out / "prince_records.jsonl")
    done = {k[0] for k in writer.done_keys(("user",))}
    for q in range(4):
        pool = np.flatnonzero(strata == q)
        n = min(pc.get("users_per_stratum", 5), len(pool))
        for u in np.sort(rng_for(cfg.get("seed", 0), "prince-users", q).choice(pool, n, replace=False)):
            if int(u) in done:
                continue
            r = adapter.explain_top1(int(u), K=ex.get("K", [10])[0])
            r.update(dataset=cfg["dataset"]["name"], stratum=q, hist_len=int(hist[u]), method="prince")
            writer.write([r])
            log.info("PRINCE user %d: found=%s |M|=%s (%.1fs)", u, r["found"], r.get("size"), r["runtime"])
    return out / "prince_records.jsonl"


def _certify_group(recs: list[dict], scope: str, time_limit: float) -> list[dict]:
    """Certify the exact records of one (user, K, alpha, phi) job; returns the updated records."""
    r0 = recs[0]
    inst = ExplanationInstance(_MODEL, r0["user"], K=r0["K"], alpha=r0["alpha"], phi=r0["phi"], scope=scope)
    out = []
    for r in recs:
        r = dict(r)
        inst.set_target(r["target"])
        st = get_setting(r["setting"])
        raw = dict(status=r["status"], size=r["size"], mask=r["mask"], bound=r["bound"], gap=r["gap"],
                   iterations=r.get("iterations"))
        c = certify_isolated(inst, st, raw, eps=r["eps"], time_limit=time_limit)
        r.update({k: c[k] for k in _CERT_FIELDS})
        r["certify_time"] = round(r["certify_time"], 4)
        changed = c["mask"] != r["mask"]
        r.update(status=c["status"], size=c["size"], mask=c["mask"], bound=c["bound"], gap=c["gap"])
        if changed:
            ver = verify(inst, c["mask"], st, eps=r["eps"], with_evidence=True) if c["mask"] is not None else None
            r.update(_summarise_ver(ver), success=bool(ver and ver["passed"]))
            if ver is not None and not ver["passed"]:
                r["status"] = "verification_failed"
        if r["status"] != "infeasible":
            for k in ("infeasibility_cause", "certificates", "evidence_blocking_items"):
                r.pop(k, None)
        out.append(r)
    return out


def _certify_worker(recs: list[dict], scope: str, time_limit: float, path: str) -> None:
    out = _certify_group(recs, scope, time_limit)
    with open(path + ".part", "w") as f:
        f.write("".join(dumps(r) + "\n" for r in out))
    os.replace(path + ".part", path)


def certify_stored(out_dir: str | Path, cfg: dict, model: Any, workers: int = 1, time_limit: float = 60.0) -> dict:
    """Certify the stored exact optimality and infeasibility claims of an experiment in place.

    Records that already carry a ``certification`` field are left as they are. The uncertified file is kept as
    ``records.uncertified.jsonl``; partial results are kept in ``certify_tmp/`` so an interrupted run resumes.
    Returns the counts of the certification outcomes."""
    global _MODEL
    _MODEL = model
    out_dir = Path(out_dir)
    path = out_dir / "records.jsonl"
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    backup = out_dir / "records.uncertified.jsonl"
    if not backup.exists():
        shutil.copy2(path, backup)
    scope = cfg["experiment"].get("scope", "target")
    groups: dict[tuple, list[int]] = {}
    for k, r in enumerate(records):
        if r["method"] == "exact" and "certification" not in r:
            groups.setdefault((r["user"], r["K"], r["alpha"], r["phi"]), []).append(k)
    tmpdir = out_dir / "certify_tmp"
    tmpdir.mkdir(exist_ok=True)
    todo = []
    for key, idx in groups.items():
        fn = str(tmpdir / ("cert_" + "_".join(map(str, key)) + ".jsonl"))
        if not os.path.exists(fn):
            todo.append((key, idx, fn))
    log.info("%s: certifying %d exact records in %d jobs (%d jobs already done)", out_dir,
             sum(len(i) for i in groups.values()), len(todo), len(groups) - len(todo))
    t0 = time.time()
    if workers <= 1 or "fork" not in mp.get_all_start_methods():
        for n, (_, idx, fn) in enumerate(todo, 1):
            _certify_worker([records[k] for k in idx], scope, time_limit, fn)
            log.info("%d/%d certification jobs done (%.0fs)", n, len(todo), time.time() - t0)
    else:
        ctx = mp.get_context("fork")
        running: dict[Any, str] = {}
        queue = list(todo)
        n = 0
        while queue or running:
            while queue and len(running) < workers:
                _, idx, fn = queue.pop(0)
                p = ctx.Process(target=_certify_worker, args=([records[k] for k in idx], scope, time_limit, fn))
                p.start()
                running[p] = fn
            time.sleep(0.05)
            for p in list(running):
                if not p.is_alive():
                    running.pop(p)
                    p.join()
                    n += 1
                    if n % 50 == 0 or n == len(todo):
                        log.info("%d/%d certification jobs done (%.0fs)", n, len(todo), time.time() - t0)
    for key, idx in groups.items():
        fn = tmpdir / ("cert_" + "_".join(map(str, key)) + ".jsonl")
        if not fn.exists():
            raise RuntimeError(f"certification job {key} failed; rerun to resume")
        with open(fn) as f:
            new = [json.loads(line) for line in f if line.strip()]
        for k, r in zip(idx, new, strict=True):
            records[k] = r
    tmp = path.with_suffix(".jsonl.part")
    with open(tmp, "w") as f:
        f.write("".join(dumps(r) + "\n" for r in records))
    os.replace(tmp, path)
    for fn in tmpdir.iterdir():
        fn.unlink()
    tmpdir.rmdir()
    counts: dict[str, int] = {}
    for r in records:
        if r["method"] == "exact":
            counts[r.get("certification", "none")] = counts.get(r.get("certification", "none"), 0) + 1
    return counts


def verify_records(records: list[dict], cfg: dict, model: Any) -> int:
    """Re-verify every stored successful mask (used by tests and ``epic-rec report --reverify``)."""
    bad = 0
    for r in records:
        if not r.get("success") or r.get("mask") is None or r["method"].endswith("c1_stop"):
            continue
        inst = ExplanationInstance(model, r["user"], K=r["K"], alpha=r["alpha"], phi=r["phi"])
        inst.set_target(r["target"])
        bad += not verify(inst, r["mask"], r["setting"], eps=r["eps"])["passed"]
    return bad
