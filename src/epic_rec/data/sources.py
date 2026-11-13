"""Registry of raw data sources: URLs, pinned commits, SHA-256 checksums and licence notes.

Interaction data are downloaded from their official distributors (GroupLens); knowledge-graph linkages are taken
from the public releases of RippleNet, KGCN and KGAT at fixed commits. Nothing here is redistributed by this
repository: :mod:`epic_rec.data.download` fetches the files into the user's ``data/raw`` directory and refuses any
file whose checksum differs from the value recorded below.

Linkage provenance
------------------
* MovieLens-1M and Last.fm use the Microsoft Satori linkage released by Wang et al. with RippleNet (CIKM 2018) and
  KGCN (WWW 2019). The original Satori service is not publicly accessible; these releases are the reproducible
  public form of that linkage and are used unchanged.
* Amazon-Book uses the KB4Rec Freebase linkage as preprocessed and released by Wang et al. with KGAT (KDD 2019).
"""

from __future__ import annotations

from dataclasses import dataclass, field

RIPPLENET_COMMIT = "46a364d6aa4e346111043924a4741a21d2ff09b6"
KGCN_COMMIT = "2363ca2ddc66a79860629b2e0ec6ad23c57697c8"
KGAT_COMMIT = "c03737be46ac26a0b5431efe149828e982ffbbfb"

_RIPPLENET = f"https://raw.githubusercontent.com/hwwang55/RippleNet/{RIPPLENET_COMMIT}/data/movie/"
_KGCN = f"https://raw.githubusercontent.com/hwwang55/KGCN/{KGCN_COMMIT}/data/music/"
_KGAT = (
    "https://raw.githubusercontent.com/xiangwang1223/knowledge_graph_attention_network/"
    f"{KGAT_COMMIT}/Data/amazon-book/"
)


@dataclass(frozen=True)
class RemoteFile:
    """One downloadable file. ``extract`` lists archive members to unpack after the checksum check."""

    name: str
    url: str
    sha256: str
    extract: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSource:
    name: str
    description: str
    files: tuple[RemoteFile, ...]
    licence: str
    linkage: str
    notes: tuple[str, ...] = field(default_factory=tuple)


SOURCES: dict[str, DatasetSource] = {
    "movielens-1m": DatasetSource(
        name="movielens-1m",
        description="MovieLens-1M ratings (GroupLens) with the Satori knowledge-graph linkage of RippleNet",
        files=(
            RemoteFile(
                "ml-1m.zip",
                "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
                "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
                extract=("ml-1m/ratings.dat", "ml-1m/movies.dat"),
            ),
            RemoteFile(
                "item_index2entity_id_rehashed.txt",
                _RIPPLENET + "item_index2entity_id_rehashed.txt",
                "fd1b78c58a7afbcf13a408168118fbd1ac53c7bde6efc404abf9588d2302e6f9",
            ),
            RemoteFile("kg_part1_rehashed.txt", _RIPPLENET + "kg_part1_rehashed.txt", "d7dac12962063dee14132cce8fca2e8c7b8ca299176430e503d4e33ce2d3e1e9"),
            RemoteFile("kg_part2_rehashed.txt", _RIPPLENET + "kg_part2_rehashed.txt", "3cb493bfddb10bf01b991f34a56560aa1af265285db8b10d164b76db0486b602"),
        ),
        licence=(
            "MovieLens: GroupLens research licence (no redistribution, see ml-1m/README). "
            "RippleNet linkage files: MIT licence (hwwang55/RippleNet)."
        ),
        linkage=f"Satori, via hwwang55/RippleNet@{RIPPLENET_COMMIT[:7]} data/movie",
        notes=(
            "RippleNet's data/movie/ratings.dat is byte-identical to ml-1m/ratings.dat "
            "(SHA-256 506d64ca...e0f); the GroupLens file is used.",
        ),
    ),
    "lastfm": DatasetSource(
        name="lastfm",
        description="Last.fm HetRec-2011 listening data (GroupLens) with the Satori linkage of KGCN",
        files=(
            RemoteFile(
                "hetrec2011-lastfm-2k.zip",
                "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
                "6738f48195667ff03caaab4d32ca9a3133d8cc026b7c3cdaf6ce1010e913c59c",
                extract=("user_artists.dat", "artists.dat"),
            ),
            RemoteFile("item_index2entity_id.txt", _KGCN + "item_index2entity_id.txt", "f9693bdde538f755d09b9931b0be662a98965e538658a00c2050912c0adf487b"),
            RemoteFile("kg.txt", _KGCN + "kg.txt", "f62bbc8f717c9f68e66dff6faeaa71025092acc11a0f1b4878b953eb05f946a9"),
        ),
        licence=(
            "HetRec-2011 Last.fm: GroupLens / Last.fm terms (research use, no redistribution). "
            "KGCN linkage files: MIT licence (hwwang55/KGCN)."
        ),
        linkage=f"Satori, via hwwang55/KGCN@{KGCN_COMMIT[:7]} data/music",
        notes=(
            "KGCN's data/music/user_artists.dat equals the GroupLens file after CRLF->LF conversion "
            "(SHA-256 254272fa...416); the GroupLens file is used.",
        ),
    ),
    "amazon-book": DatasetSource(
        name="amazon-book",
        description="Amazon-Book as released with KGAT (KB4Rec Freebase linkage, 10-core)",
        files=(
            RemoteFile("train.txt", _KGAT + "train.txt", "4d10106bb06a77e11df660429903c0067aeb85b86431807836b0d8b7edc30c38"),
            RemoteFile("test.txt", _KGAT + "test.txt", "f47eff9d35fb95c07d74251a991478a84095ce6c5969b0dafcb4afe99fc716ec"),
            RemoteFile("kg_final.txt.zip", _KGAT + "kg_final.txt.zip", "ae1adad6c59ac2faf46066fb03b680a8a93db60588fcc63fdb0670df0f32161a", extract=("kg_final.txt",)),
            RemoteFile("item_list.txt", _KGAT + "item_list.txt", "efea8518cd68e515584fdfbfca95f318b25ef551c64498c5935897c2b02e436f"),
            RemoteFile("relation_list.txt", _KGAT + "relation_list.txt", "50fe9c6560224b20d8ace78ae8ca0b8dad39248f6dc900f6744b633ba8c33f4b"),
        ),
        licence=(
            "Amazon review data: research use under the terms of its distributors (McAuley et al.). "
            "KGAT release: MIT licence (xiangwang1223/knowledge_graph_attention_network)."
        ),
        linkage=f"KB4Rec Freebase, via KGAT@{KGAT_COMMIT[:7]} Data/amazon-book",
        notes=(
            "KGAT's train/test files are merged and re-split with EPIC-Rec's own deterministic split so that all "
            "datasets share one protocol; KGAT's original split is not used.",
            "In KGAT's release, item remap id i is entity remap id i in kg_final.txt.",
        ),
    ),
}
