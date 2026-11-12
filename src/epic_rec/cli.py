"""Command-line interface: ``epic-rec <command> [options]`` (also ``python -m epic_rec``).

Commands
--------
download  fetch and checksum the raw files of a dataset
prepare   build the processed dataset (filters, KG links, train/val/test split)
train     fit the path-additive backbone and the accuracy baselines; write accuracy tables
explain   explain one (user, target) and print or save the verified explanation object
run       run (or resume) an experiment config; writes records.jsonl
prince    run the PRINCE baseline (official code) on its native PPR recommender
report    build CSV tables and PDF/PNG figures from saved records
demo      end-to-end example on the synthetic dataset (CPU, about a minute)
pipeline  prepare, train, run and report for every experiment listed in a pipeline config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from epic_rec.utils.config import load_config
from epic_rec.utils.io import atomic_write_json
from epic_rec.utils.logging import get_logger

log = get_logger("epic_rec")


def _cfg(a: argparse.Namespace) -> dict:
    return load_config(a.config, a.set)


def cmd_download(a: argparse.Namespace) -> None:
    from epic_rec.data.download import download

    print(download(a.dataset, a.raw_root, force=a.force))


def cmd_prepare(a: argparse.Namespace) -> None:
    from epic_rec.data.preprocess import prepare

    ds = prepare(_cfg(a))
    print(json.dumps(ds.stats(), indent=2))


def cmd_train(a: argparse.Namespace) -> None:
    from epic_rec.data.preprocess import load_or_prepare
    from epic_rec.experiments.training import train_and_evaluate

    cfg = _cfg(a)
    res = train_and_evaluate(cfg, load_or_prepare(cfg), baselines=not a.no_baselines)
    for name, r in res.items():
        print(f"{name:24s} " + "  ".join(f"{k}={v:.4f}" for k, v in r["test"].items() if k != "users"))


def cmd_explain(a: argparse.Namespace) -> None:
    from epic_rec.data.preprocess import load_or_prepare
    from epic_rec.experiments.training import load_model
    from epic_rec.explainers.api import explain
    from epic_rec.optimization.instance import ExplanationInstance
    from epic_rec.paths.evidence import top_paths

    cfg = _cfg(a)
    model = load_model(cfg, load_or_prepare(cfg))
    inst = ExplanationInstance(model, a.user, K=a.K, alpha=a.alpha, phi=a.phi)
    inst.set_target(int(inst.L[a.rank - 1]))
    out = []
    for s in a.setting:
        e = explain(inst, s, a.method, eps=a.eps, time_limit=a.time_limit)
        print(e.summary())
        d = e.to_dict()
        d["target_paths_before"] = [p.__dict__ for p in top_paths(model, a.user, int(inst.t), None, 5)]
        out.append(d)
    if a.out:
        atomic_write_json(a.out, out)
        print(f"written {a.out}")


def cmd_run(a: argparse.Namespace) -> None:
    from epic_rec.data.preprocess import load_or_prepare
    from epic_rec.experiments.runner import run_experiment
    from epic_rec.experiments.training import load_model

    cfg = _cfg(a)
    print(run_experiment(cfg, load_model(cfg, load_or_prepare(cfg))))


def cmd_prince(a: argparse.Namespace) -> None:
    from epic_rec.data.preprocess import load_or_prepare
    from epic_rec.experiments.runner import run_prince

    cfg = _cfg(a)
    print(run_prince(cfg, load_or_prepare(cfg)))


def cmd_report(a: argparse.Namespace) -> None:
    from epic_rec.reporting import all_figures, all_tables, load_records, write_tables

    files = []
    for r in a.results:
        p = Path(r)
        files += sorted(p.rglob("records.jsonl")) if p.is_dir() else [p]
    if not files:
        sys.exit(f"no records.jsonl under {a.results}")
    df = load_records(files)
    tables = all_tables(df, n_boot=a.n_boot, seed=a.seed)
    out = Path(a.out)
    paths = write_tables(tables, out / "tables")
    paths += all_figures(tables, df, out / "figures")
    atomic_write_json(out / "report_manifest.json", dict(records=[str(f) for f in files], rows=len(df),
                                                         n_boot=a.n_boot, outputs=[str(p) for p in paths]))
    print(f"{len(df)} records from {len(files)} files -> {len(paths)} outputs in {out}")


def cmd_demo(a: argparse.Namespace) -> None:
    import numpy as np

    from epic_rec.data.synthetic import make_synthetic
    from epic_rec.explainers.api import explain
    from epic_rec.models.path_additive import PathAdditiveRecommender
    from epic_rec.optimization.instance import ExplanationInstance
    from epic_rec.paths.evidence import top_paths

    out = Path(a.out)
    ds = make_synthetic(seed=a.seed)
    model = PathAdditiveRecommender.fit(ds, {"seed": a.seed})
    print(f"synthetic data: {ds.stats()}")
    print(f"learned weights: theta={ {k: round(v, 4) for k, v in model.theta.items()} }, beta={model.beta:.4f}")
    order = abs(model.hist_len - np.median(model.hist_len)).argsort(kind="stable")
    results = []

    def show(u: int, rank: int, note: str) -> None:
        inst = ExplanationInstance(model, u, K=a.K)
        inst.set_target(int(inst.L[rank - 1]))
        print(f"\n{note}\nuser {u}: top-{a.K} = {inst.L.tolist()}; target item {inst.t} at rank {rank}")
        for p in top_paths(model, u, int(inst.t), None, 3):
            print(f"  evidence path: item {p.origin} -[{p.relation}]-> {p.via_type} {p.via} -> item {p.item} "
                  f"(contribution {p.contribution:.4f})")
        for s in ("A", "B", "C"):
            for m in ("exact", "greedy:violation", "beam"):
                e = explain(inst, s, m, time_limit=10)
                print(" ", e.summary())
                results.append(e.to_dict())

    u0 = int(order[0])
    show(u0, a.rank, f"Example 1: the median-history user's rank-{a.rank} item")
    for u in order[1:]:
        inst = ExplanationInstance(model, int(u), K=a.K)
        hit = next((r for r in range(1, a.K + 1)
                    if explain(inst.set_target(int(inst.L[r - 1])), "C", "exact", diagnose=False).success), None)
        if hit:
            show(int(u), hit, "Example 2: the first target (users ordered by distance to the median history) "
                              "with a verified C1 + C2 + C3 explanation")
            break
    atomic_write_json(out / "explanations.json", results)
    print(f"\nexplanation objects written to {out / 'explanations.json'}")


def cmd_pipeline(a: argparse.Namespace) -> None:
    from epic_rec.data.preprocess import load_or_prepare
    from epic_rec.experiments.runner import run_experiment, run_prince
    from epic_rec.experiments.training import load_model, model_path, train_and_evaluate

    pipe = load_config(a.config, a.set)
    root = Path(a.config).parent
    results = []
    for entry in pipe["experiments"]:
        cfg = load_config(root / entry, a.set)
        ds = load_or_prepare(cfg)
        if not model_path(cfg).exists():
            train_and_evaluate(cfg, ds, baselines=pipe.get("baselines", True))
        if cfg.get("experiment", {}).get("kind") == "prince":
            run_prince(cfg, ds)
        else:
            results.append(run_experiment(cfg, load_model(cfg, ds)).parent)
    a.results = [str(r) for r in results]
    a.out = pipe.get("report_dir", "results/report")
    cmd_report(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="epic-rec", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_cfg(name: str, help_: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--config", required=True)
        s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
        return s

    s = sub.add_parser("download", help="fetch raw files")
    s.add_argument("dataset", choices=["movielens-1m", "lastfm", "amazon-book"])
    s.add_argument("--raw-root", default="data/raw")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_download)
    with_cfg("prepare", "build processed dataset").set_defaults(fn=cmd_prepare)
    s = with_cfg("train", "fit backbone and baselines")
    s.add_argument("--no-baselines", action="store_true")
    s.set_defaults(fn=cmd_train)
    s = with_cfg("explain", "explain one target")
    s.add_argument("--user", type=int, required=True)
    s.add_argument("--rank", type=int, default=1)
    s.add_argument("--K", type=int, default=10)
    s.add_argument("--alpha", type=float, default=0.5)
    s.add_argument("--phi", type=float, default=0.0)
    s.add_argument("--eps", type=float, default=1e-6)
    s.add_argument("--setting", nargs="+", default=["A", "B", "C"])
    s.add_argument("--method", default="exact")
    s.add_argument("--time-limit", type=float, default=20.0)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_explain)
    with_cfg("run", "run an experiment").set_defaults(fn=cmd_run)
    with_cfg("prince", "run PRINCE").set_defaults(fn=cmd_prince)
    s = sub.add_parser("report", help="tables and figures from records")
    s.add_argument("results", nargs="+", help="records.jsonl files or directories containing them")
    s.add_argument("--out", default="results/report")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("demo", help="synthetic end-to-end example")
    s.add_argument("--out", default="results/demo")
    s.add_argument("--K", type=int, default=10)
    s.add_argument("--rank", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_demo)
    s = with_cfg("pipeline", "full pipeline")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_pipeline)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
