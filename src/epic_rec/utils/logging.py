"""Logging setup shared by the command-line tools."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "epic_rec", level: int | str | None = None, logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger("epic_rec").handlers:
        root = logging.getLogger("epic_rec")
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
        root.addHandler(h)
        root.setLevel(logging.INFO)
        root.propagate = False
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter(_FMT))
        logging.getLogger("epic_rec").addHandler(fh)
    if level is not None:
        logging.getLogger("epic_rec").setLevel(level)
    return logger
