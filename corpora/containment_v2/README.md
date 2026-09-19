# containment_v2/ — side corpora, deliberately OUT of the loader's glob

`retrieval_core_local.load_corpus()` globs `corpora/corpus_{kind}_*.json`, and later files
overwrite earlier records case-id-wise. These v2-containment side corpora MUST NOT match that
glob: containment is a scorer feature and a relevance key, so if they leak in, every retrieval
number silently shifts (caught 2026-09-16: a baselines run picked them up from `corpora/` and
tab7 (ii) moved 0.972 → 0.911 before the reproduction gate flagged it).

Keep this directory out of `corpora/` proper. The frozen tables use the degenerate ('boundary')
containment of the frozen corpora; these v2 files exist for the grader packet and the Table 11
recompute, to be adopted deliberately — never implicitly.
