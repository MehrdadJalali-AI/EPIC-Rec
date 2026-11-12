# Data

No dataset is included in this repository. `epic-rec download <dataset>` fetches the files from their official
distributors into `data/raw/<dataset>/`, verifies each file's SHA-256 against the value pinned in
`src/epic_rec/data/sources.py` (a mismatch aborts with `ChecksumError`), extracts the needed archive members and
writes `download_manifest.json` (URL, checksum, size). `epic-rec prepare --config configs/<dataset>.yaml` then
builds `data/processed/<dataset>/` with a `manifest.json` that records the raw-file checksums, every filter
setting, the filtering counts and the dataset statistics.

## Sources

| dataset | interactions | knowledge-graph linkage | licence of the interaction data |
| --- | --- | --- | --- |
| MovieLens-1M (`movielens-1m`) | GroupLens `ml-1m.zip` | Microsoft Satori, released with RippleNet, `hwwang55/RippleNet@46a364d` `data/movie` (MIT) | GroupLens research licence, no redistribution |
| Last.fm (`lastfm`) | GroupLens HetRec-2011 `hetrec2011-lastfm-2k.zip` | Microsoft Satori, released with KGCN, `hwwang55/KGCN@2363ca2` `data/music` (MIT) | GroupLens / Last.fm terms, research use, no redistribution |
| Amazon-Book (`amazon-book`, optional) | KGAT release `xiangwang1223/knowledge_graph_attention_network@c03737b` | KB4Rec Freebase, as preprocessed by KGAT (MIT) | Amazon review data terms (McAuley et al.) |

**Satori linkage.** The Microsoft Satori service is not publicly accessible. The item-to-entity files and
knowledge-graph triples released by the RippleNet and KGCN authors are the public, citable form of that
linkage; they are downloaded at pinned commits and used unchanged. If these files disappear or change upstream,
the checksum check fails and the pipeline stops; it never substitutes another linkage silently.

**Provenance checks.** RippleNet's `data/movie/ratings.dat` is byte-identical to GroupLens' `ml-1m/ratings.dat`;
KGCN's `data/music/user_artists.dat` equals GroupLens' file after CRLF→LF conversion. In both cases the GroupLens
original is used.

## Preprocessing (`src/epic_rec/data/preprocess.py`)

1. **Positives.** MovieLens: rating ≥ 4. Last.fm: every listened artist (play count ≥ 1). Amazon-Book: KGAT's
   train and test files are merged (KGAT's split is not used).
2. **KG items.** Only items with a linked entity are kept (`require_kg_link: true`).
3. **k-core.** Users with fewer than `min_user` and items with fewer than `min_item` interactions are removed,
   repeated until nothing changes. MovieLens 10 / 5; Last.fm 5 / 3; Amazon-Book 20 / 10. In Last.fm only about
   a fifth of the interactions involve one of the 3,846 linked artists, so a 10 / 5 core would leave too few
   items to be informative (see the comment in `configs/lastfm.yaml`).
4. **Item–entity links.** Triples with exactly one linked-item endpoint become (item, relation, entity) links;
   triples between two items are counted and ignored (the model uses item–entity–item paths). Entities linked to
   fewer than `min_items_per_entity` (2) kept items create no path and are dropped; optionally very generic
   entities (`max_items_per_entity`) and rare relations (`min_links_per_relation`) are dropped. All counts are in
   the manifest.
5. **Split.** Per user, interactions are shuffled with a seeded generator and split 70 / 10 / 20 into train /
   validation / test (`split.seed`, default 0); every user keeps at least one training interaction.
   `check_no_leakage` asserts that no (user, item) pair occurs twice within a split or in two splits.

The k-core is computed on all interactions before splitting, as is standard. Every statistic the recommender
learns from (co-occurrence, popularity, path weights) is computed from the training split only; validation is used
for baseline hyper-parameters, and test only for reported accuracy. Explanations are computed for the training
history, which is the input of the deployed model.

**Difference from the manuscript draft.** The manuscript's draft experiments used an 80 / 20 train / test split
without a validation set. This implementation adds a 10 % validation split so that baseline hyper-parameters are
not tuned on test data; numbers produced here are therefore not expected to equal the draft's.

## Synthetic data

`epic_rec.data.synthetic.make_synthetic` generates a small dataset (users with group preferences, items with
genre and creator entities) with a fixed seed. It needs no download and is used by the tests, CI, the demo and
the smoke experiment (`configs/synthetic.yaml`). It contains many exact score ties, which exercises the P versus
P_ε distinction.
