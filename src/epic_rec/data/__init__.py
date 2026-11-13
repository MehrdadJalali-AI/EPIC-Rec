"""Data download, preprocessing and the processed dataset container."""

from epic_rec.data.dataset import InteractionDataset, split_per_user
from epic_rec.data.download import download
from epic_rec.data.preprocess import load_or_prepare, prepare
from epic_rec.data.sources import SOURCES
from epic_rec.data.synthetic import make_synthetic

__all__ = ["SOURCES", "InteractionDataset", "download", "load_or_prepare", "make_synthetic", "prepare",
           "split_per_user"]
