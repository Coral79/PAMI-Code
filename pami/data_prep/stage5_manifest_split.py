"""Stage 5 — window manifest + official splits (v1, v2 and the v3_noval rule).

manifest_w64_s32.{npz,csv}: every 64-frame window (stride 32) of every packaged sequence with >= 64
frames; per window the ACTIVE object indices (moves > V_THR or touches < C_THR inside the window) and a
tier (-1 no active object, 0 one object/one part, 1 one object/several parts, 2.. multi-object).

Splits (all keyed by SEQUENCE; seq_subset recorded in meta):
  split_official.json (v1)      test = seqs named in LIGHT/InterAct stats/{test_seq.txt, test_seq_orig.txt}
                                (grab: suffix-stripped), val = 10% of the rest (tier-stratified within
                                subset, seed 0), train = rest.   -> the population of ALL stats files (stage 6)
  split_official_v2.json        chunk-FAMILY closure: a family (s, s_<int>, s_<int>_<int>) goes to test iff
                                any member is in the official union; families are atomic in train/val.
  split_official_v3_noval.json  train = v2 train U v2 val (LIGHT/InterAct have no val set), test = v2 test,
                                val = random.Random(0).sample(train, 1000) (a TRAIN subset used only for
                                metric logging / checkpoint selection). -> the split every final model uses.
"""
import csv
import datetime
import glob
import json
import os
import random
import re

import numpy as np

from . import paths as P
from .subsets import seq_subset_map

WIN, STRIDE = 64, 32
V_THR, C_THR = 0.001, 0.010                      # m/frame ; m — identical to the tae_sample rule
RATIO_VAL, SEED = 0.10, 0
INT = re.compile(r"^\d{1,3}$")
N_VAL_V3 = 1000


def tier_of(n_active, n_parts):
    if n_active == 0: return -1
    if n_active == 1: return 0 if n_parts <= 1 else 1
    if n_active == 2: return 2
    if n_active == 3: return 3
    return 4


def build_manifest(out_root=P.PAMI_DATA_ROOT):
    rows, win_meta, tier_hist = [], [], {}
    s2s = seq_subset_map()
    files = sorted(glob.glob(f"{out_root}/output/*/tae_sample.npz"))
    for f in files:
        seq = os.path.basename(os.path.dirname(f))
        if seq not in s2s:
            print(f"  [skip] {seq}: not in any InterAct subset"); continue
        ts = np.load(f, allow_pickle=True)
        T = int(ts["T"]); an = [str(x) for x in ts["active_names"]]
        M = len(an)
        spd = [ts[f"speed_{mi}"] for mi in range(M)]
        cd = [ts[f"contact_d_{mi}"] for mi in range(M)]
        w = [ts[f"w_{mi}"] for mi in range(M)]
        if T < WIN:
            continue
        for s in range(0, T - WIN + 1, STRIDE):
            e = s + WIN
            active, parts = [], set()
            for mi in range(M):
                moves = spd[mi][s:e].max() > V_THR
                touches = cd[mi][s:e].min() < C_THR
                if moves or touches:
                    active.append(mi)
                    pv = w[mi][s:e][:, :6].sum(0)
                    for pi in range(6):
                        if pv[pi] > 0: parts.add(pi)
            tier = tier_of(len(active), len(parts))
            tier_hist[tier] = tier_hist.get(tier, 0) + 1
            rows.append([seq, s2s[seq], s, e, len(active), len(parts), tier, ";".join(an[mi] for mi in active)])
            win_meta.append((seq, s, e, np.array(active, dtype=np.int16), tier, s2s[seq]))
    tag = f"_w{WIN}_s{STRIDE}"
    with open(f"{out_root}/manifest{tag}.csv", "w", newline="") as fh:
        wc = csv.writer(fh)
        wc.writerow(["seq", "subset", "start", "end", "n_active", "n_parts", "tier", "active_objs"])
        wc.writerows(rows)
    tmp = f"{out_root}/manifest{tag}.npz.tmp.npz"
    np.savez(tmp,
             seq=np.array([m[0] for m in win_meta], dtype=object),
             start=np.array([m[1] for m in win_meta], dtype=np.int32),
             end=np.array([m[2] for m in win_meta], dtype=np.int32),
             tier=np.array([m[4] for m in win_meta], dtype=np.int8),
             active=np.array([m[3] for m in win_meta], dtype=object),
             subset=np.array([m[5] for m in win_meta], dtype=object))
    os.replace(tmp, f"{out_root}/manifest{tag}.npz")
    print(f"[manifest] {len(rows)} windows from {len(set(r[0] for r in rows))} seqs (of {len(files)} tae_sample files)")
    for t in sorted(tier_hist):
        print(f"  tier {t}: {tier_hist[t]:6d} ({100*tier_hist[t]/len(rows):.1f}%)")
    return f"{out_root}/manifest{tag}.npz"


def _seq2tier(manifest):
    man = np.load(manifest, allow_pickle=True)
    seq2tier = {}
    for s, t in zip(man["seq"].tolist(), man["tier"].tolist()):
        s = str(s); seq2tier[s] = max(seq2tier.get(s, -1), int(t))      # sequence tier = MAX over windows
    return seq2tier


def _official_names(light_root, lists):
    official = set()
    for fn in lists:
        for line in open(f"{light_root}/stats/{fn}"):
            line = line.strip()
            if line: official.add(line.split("/")[-1])
    return official


def build_split_official_v1(out_root=P.PAMI_DATA_ROOT, manifest=None, light_root=P.LIGHT_ROOT,
                            out_name="split_official.json", grab_suffix_strip=False):
    """v1. grab_suffix_strip=False reproduces the STORED split_official.json (built by the
    builder version WITHOUT the grab `_<k>` suffix strip, so grab has 0 official-test seqs there and
    its 112 official test sequences sit in train/val — the known v1 leak that v2 fixed). It is kept
    because this file's TRAIN list is the population of every stats file (stage 6)."""
    manifest = manifest or f"{out_root}/manifest_w{WIN}_s{STRIDE}.npz"
    seq2tier = _seq2tier(manifest); s2s = seq_subset_map()
    official = _official_names(light_root, ("test_seq.txt", "test_seq_orig.txt"))

    def _is_official(s):
        return s in official or (grab_suffix_strip and s2s[s] == "grab" and s.rsplit("_", 1)[0] in official)
    test = sorted(s for s in seq2tier if _is_official(s))
    rest = {s: t for s, t in seq2tier.items() if not _is_official(s)}
    rng = np.random.RandomState(SEED)
    train, val, per_subset = [], [], {}
    for subset in sorted({s2s[s] for s in seq2tier}):
        sub = {s: t for s, t in rest.items() if s2s[s] == subset}
        tr, va = [], []
        for tier in sorted(set(sub.values())):
            seqs = sorted([s for s, t in sub.items() if t == tier])
            rng.shuffle(seqs)
            n_va = int(round(len(seqs) * RATIO_VAL))
            va += seqs[:n_va]; tr += seqs[n_va:]
        train += tr; val += va
        per_subset[subset] = dict(train=len(tr), val=len(va), test=sum(1 for s in test if s2s[s] == subset))
    split = {"train": sorted(train), "val": sorted(val), "test": test,
             "ratio": [1 - RATIO_VAL, RATIO_VAL, None], "seed": SEED, "manifest": os.path.basename(manifest),
             "meta": {"dataset": "interact_7sub_official", "split_by": "official_test_lists",
                      "test_source": "LIGHT/InterAct/stats/test_seq(.orig).txt union", "subsets": per_subset,
                      "seq_subset": {s: s2s[s] for s in sorted(seq2tier)}}}
    out = f"{out_root}/{out_name}"
    json.dump(split, open(out, "w"), indent=2)
    print(f"[split v1] train {len(train)} | val {len(val)} | test {len(test)} -> {out}")
    return out


def build_split_official_v2(out_root=P.PAMI_DATA_ROOT, manifest=None, light_root=P.LIGHT_ROOT,
                            lists=("test_seq.txt", "test_seq_orig.txt"), out_name="split_official_v2.json"):
    manifest = manifest or f"{out_root}/manifest_w{WIN}_s{STRIDE}.npz"
    seq2tier = _seq2tier(manifest); s2s = seq_subset_map()
    allseq = set(seq2tier)
    official = _official_names(light_root, lists)

    def base_of(s):
        h, _, t = s.rpartition("_")
        return h if (INT.match(t) and (h in allseq or h in official)) else None

    def root_of(s):
        while True:
            b = base_of(s)
            if b is None:
                return s
            s = b

    fams = {}
    for s in allseq:
        fams.setdefault(root_of(s), []).append(s)
    test, rest_f = [], {}
    for root, members in fams.items():
        if root in official or any(m in official for m in members):
            test += members
        else:
            rest_f[root] = members
    rng = np.random.RandomState(SEED)
    train, val, per_subset = [], [], {}
    for subset in sorted({s2s[s] for s in allseq}):
        sub_f = {r: m for r, m in rest_f.items() if s2s[m[0]] == subset}
        tr, va = [], []
        for tier in sorted({max(seq2tier[s] for s in m) for m in sub_f.values()}):
            roots = sorted(r for r, m in sub_f.items() if max(seq2tier[s] for s in m) == tier)
            t_target = int(round(sum(len(sub_f[r]) for r in roots) * RATIO_VAL))
            rng.shuffle(roots)
            n_va = 0
            for r in roots:
                if n_va < t_target:
                    va += sub_f[r]; n_va += len(sub_f[r])
                else:
                    tr += sub_f[r]
        train += tr; val += va
        per_subset[subset] = dict(train=len(tr), val=len(va), test=sum(1 for s in test if s2s[s] == subset))
    split = {"train": sorted(train), "val": sorted(val), "test": sorted(test),
             "ratio": [1 - RATIO_VAL, RATIO_VAL, None], "seed": SEED, "manifest": os.path.basename(manifest),
             "meta": {"dataset": "interact_7sub_official_v2", "split_by": "official_test_lists+chunk_families",
                      "test_source": f"LIGHT/InterAct/stats/{list(lists)} union",
                      "rule": "transitive chunk-family atomicity; family->test if any member or root in "
                              "official union (all subsets, int-suffix guard)",
                      "builder": "pami.data_prep.stage5_manifest_split.build_split_official_v2",
                      "built": datetime.datetime.now().isoformat(timespec="seconds"),
                      "subsets": per_subset, "seq_subset": {s: s2s[s] for s in sorted(allseq)}}}
    out = f"{out_root}/{out_name}"
    json.dump(split, open(out, "w"), indent=2)
    print(f"[split v2] train {len(train)} | val {len(val)} | test {len(test)} -> {out}")
    return out


def build_split_v3_noval(out_root=P.PAMI_DATA_ROOT, v2_path=None, out_name="split_official_v3_noval.json"):
    v2 = json.load(open(v2_path or f"{out_root}/split_official_v2.json"))
    train = sorted(v2["train"] + v2["val"])
    val = sorted(random.Random(SEED).sample(train, N_VAL_V3))
    meta = dict(v2["meta"])
    meta["note"] = ("v3_noval: train = v2 train U v2 val (everything except the family-closed official-test "
                    "closure), matching LIGHT/InterAct which have no val. 'val' is a 1000-seq RANDOM SUBSET OF "
                    "TRAIN (random.Random(0).sample) used for metric logging and checkpoint selection only.")
    split = {"train": train, "val": val, "test": list(v2["test"]), "ratio": v2["ratio"], "seed": SEED, "meta": meta}
    out = f"{out_root}/{out_name}"
    json.dump(split, open(out, "w"), indent=2)
    print(f"[split v3_noval] train {len(train)} | val(train subset) {len(val)} | test {len(split['test'])} -> {out}")
    return out


def run(out_root=P.PAMI_DATA_ROOT, light_root=P.LIGHT_ROOT):
    man = build_manifest(out_root)
    build_split_official_v1(out_root, man, light_root)
    v2 = build_split_official_v2(out_root, man, light_root)
    build_split_v3_noval(out_root, v2)
