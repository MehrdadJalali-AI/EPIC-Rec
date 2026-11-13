"""File helpers: checksums, atomic writes and append-only JSON-lines checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_default, sort_keys=False)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def atomic_write_json(path: str | os.PathLike, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, default=_default, indent=indent) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    """Yield records; a truncated last line (interrupted run) is skipped."""
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class JsonlWriter:
    """Append-only JSON-lines file used as an experiment checkpoint.

    Every record is flushed and fsynced, so an interrupted run loses at most the record being written;
    :meth:`done_keys` lets a restarted run skip finished jobs.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def done_keys(self, key_fields: Iterable[str]) -> set[tuple]:
        fields = tuple(key_fields)
        return {tuple(r.get(k) for k in fields) for r in read_jsonl(self.path)}

    def write(self, records: Iterable[dict]) -> None:
        text = "".join(dumps(r) + "\n" for r in records)
        if not text:
            return
        with open(self.path, "a") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
