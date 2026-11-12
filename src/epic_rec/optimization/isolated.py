"""Run a solver call in a forked child process with a hard wall-clock limit.

HiGHS checks its clock only periodically, and version 1.16.0 was observed to hang on some models. Running each
solve in a child process guarantees that one instance cannot stall an experiment: if the child does not return
within ``time_limit + grace`` seconds it is killed and the outcome is recorded as ``timeout`` with
``killed=True``. On platforms without ``fork`` the call runs in-process.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from typing import Any


def _child(fn: Callable[..., dict], args: tuple, kwargs: dict, q: Any) -> None:
    try:
        q.put(("ok", fn(*args, **kwargs)))
    except Exception as e:  # reported to the parent
        q.put(("error", repr(e)))


def run_isolated(fn: Callable[..., dict], *args: Any, hard_limit: float, **kwargs: Any) -> dict[str, Any]:
    if "fork" not in mp.get_all_start_methods():
        return fn(*args, **kwargs)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(fn, args, kwargs, q))
    t0 = time.time()
    p.start()
    try:
        kind, res = q.get(timeout=hard_limit)
    except Exception:
        kind, res = "killed", None
    p.join(1)
    if p.is_alive():
        p.kill()
        p.join()
    if kind == "ok":
        res.setdefault("killed", False)
        return res
    if kind == "error":
        raise RuntimeError(f"solver raised in child process: {res}")
    return dict(status="timeout", size=None, mask=None, bound=None, gap=None, time=time.time() - t0,
                iterations=None, threats=None, killed=True)
