"""Stage 9 — the official evaluation pool and its piece positions.

Derivation:
  * start from the TEST sequences of split_official_v3_noval.json (the family-closed official test set),
  * map our name to LIGHT's name (grab chunks `s10_x_1_0` are base-named `s10_x_1` in LIGHT),
  * keep those whose LIGHT name is in LIGHT/InterAct/stats/test_seq_orig.txt (the official InterAct test list),
  * drop intercap (LIGHT never trained/released intercap) and any sequence LIGHT's released tree has no
    data.npz for (no ground truth in the baselines' format -> not comparable).
Every pool sequence has exactly ONE piece in segments_cap300.json.

Outputs <out>/interact_test.json  {"rows": [{name, light_name, subset, obj, text}]} sorted by name: the test list
                                  evaluate.py and refine.py --seqs read (object name from the prepared data,
                                  caption = the sequence's first caption in texts.json)
        <out>/pieces_test.txt     positions of the pool sequences inside the TEST-filtered piece list
                                  (segments_cap300.json order restricted to split["test"] == the
                                  T2HOIPieceDataset(seqs=split["test"]) index order == generate.py --pieces)
"""
import json
import os
import re

import numpy as np

from . import paths as P

_CHUNK_RE = re.compile(r"^(.*)_\d+$")


def light_name(seq, subset):
    if subset == "grab":
        m = _CHUNK_RE.match(seq)
        if m:
            return m.group(1)
    return seq


def build_pool(split_json, light_root=P.LIGHT_ROOT):
    sp = json.load(open(split_json)); s2s = sp["meta"]["seq_subset"]
    orig = {l.strip().split("/")[-1] for l in open(f"{light_root}/stats/test_seq_orig.txt") if l.strip()}
    pool = []
    for s in sorted(sp["test"]):
        sub = s2s[s]; ln = light_name(s, sub)
        if ln not in orig or sub == "intercap":
            continue
        if not os.path.exists(f"{light_root}/{sub}/sequences_canonical/{ln}/data.npz"):
            continue
        pool.append({"name": s, "light_name": ln, "subset": sub})
    return pool


def piece_positions(pool, segments_json, split_json, latents=None):
    """--pieces positions of the pool inside the test-filtered piece order. If `latents` (an encoded
    archive) is given, its seg_seq order is asserted equal to the segments file (same source of truth)."""
    seg = [g["seq"] for g in json.load(open(segments_json))["segments"]]
    if latents:
        with np.load(latents, allow_pickle=True) as z:
            zs = [str(s) for s in z["seg_seq"]]
        assert zs == seg, f"{latents} seg_seq order != {segments_json}"
    test = set(json.load(open(split_json))["test"])
    keep = [i for i in range(len(seg)) if seg[i] in test]
    names = {r["name"] for r in pool}
    pos = [k for k, i in enumerate(keep) if seg[i] in names]
    got = [seg[keep[k]] for k in pos]
    multi = sorted({n for n in got if got.count(n) > 1}); missing = sorted(names - set(got))
    if multi or missing:
        raise SystemExit(f"pool not addressable: {len(missing)} missing {missing[:5]} | {len(multi)} multi-piece {multi[:5]}")
    return pos


def add_obj_text(pool, texts_json, out_root=P.PAMI_DATA_ROOT):
    """obj (the prepared object_6dof.npz) and text (first caption of texts.json) for every pool row, in place."""
    t = json.load(open(texts_json))
    for r in pool:
        with np.load(f"{out_root}/output/{r['name']}/object_6dof.npz", allow_pickle=True) as o:
            r["obj"] = str(np.atleast_1d(o["names"])[0])
        r["text"] = t["texts"][t["captions"][r["name"]]]
    return pool


def run(out, split_json=None, segments_json=None, light_root=P.LIGHT_ROOT, latents=None, expect=None):
    split_json = split_json or P.SPLIT_V3
    segments_json = segments_json or f"{P.T2HOI_OUT}/segments_cap300.json"
    os.makedirs(out, exist_ok=True)
    pool = build_pool(split_json, light_root)
    if expect and len(pool) != expect:
        raise SystemExit(f"pool has {len(pool)} sequences, expected {expect}")
    add_obj_text(pool, os.path.join(os.path.dirname(segments_json), "texts.json"))
    json.dump({"rows": pool}, open(f"{out}/interact_test.json", "w"), indent=1)
    pos = piece_positions(pool, segments_json, split_json, latents)
    open(f"{out}/pieces_test.txt", "w").write(" ".join(map(str, pos)))
    import collections
    print(f"[test_pool] {len(pool)} sequences {dict(collections.Counter(r['subset'] for r in pool))} -> {out}")
    return pool, pos
