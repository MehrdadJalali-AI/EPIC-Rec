"""Download raw files with checksum verification and write a provenance manifest."""

from __future__ import annotations

import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from epic_rec.data.sources import SOURCES, DatasetSource, RemoteFile
from epic_rec.utils.io import atomic_write_json, sha256_file
from epic_rec.utils.logging import get_logger

log = get_logger(__name__)


class ChecksumError(RuntimeError):
    pass


def _fetch(f: RemoteFile, dest: Path, retries: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            log.info("downloading %s", f.url)
            with urllib.request.urlopen(f.url, timeout=120) as r, open(tmp, "wb") as out:
                shutil.copyfileobj(r, out, 1 << 20)
            os.replace(tmp, dest)
            return
        except OSError as e:  # network errors
            log.warning("download failed (%s), attempt %d/%d", e, attempt, retries)
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {f.url}")


def download(name: str, raw_root: str | Path = "data/raw", force: bool = False) -> Path:
    """Fetch every file of dataset ``name`` into ``raw_root/name`` and verify its SHA-256.

    Returns the dataset's raw directory. A manifest ``download_manifest.json`` records URL, checksum, size and
    download time of every file.
    """
    src: DatasetSource = SOURCES[name]
    raw = Path(raw_root) / name
    raw.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in src.files:
        dest = raw / f.name
        if force or not dest.exists():
            _fetch(f, dest)
        digest = sha256_file(dest)
        if digest != f.sha256:
            raise ChecksumError(
                f"{dest}: SHA-256 {digest} does not match the recorded {f.sha256}. The upstream file has "
                "changed or the download is corrupt; delete it and retry, or update sources.py deliberately."
            )
        for member in f.extract:
            out = raw / Path(member).name
            if force or not out.exists():
                with zipfile.ZipFile(dest) as z:
                    candidates = [m for m in z.namelist() if m == member or m.endswith("/" + member)]
                    if not candidates:
                        raise FileNotFoundError(f"{member} not in {dest}")
                    with z.open(candidates[0]) as zf, open(out, "wb") as o:
                        shutil.copyfileobj(zf, o)
        entries.append(
            dict(name=f.name, url=f.url, sha256=digest, bytes=dest.stat().st_size, extracted=list(f.extract))
        )
    atomic_write_json(
        raw / "download_manifest.json",
        dict(dataset=name, linkage=src.linkage, licence=src.licence, notes=list(src.notes), files=entries),
    )
    return raw
