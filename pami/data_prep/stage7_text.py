"""Stage 7 — text + segment ("piece") files for the generator, with the test-truncation filter.

TEXT + PIECE RULES FOLLOW LIGHT (data_loaders/humanml/data/dataset.py):
  - captions come from <seq_dir>/text.txt ("caption#tokens#f_tag#to_tag", tags in seconds; a 2-field
    "caption#tokens" line = untagged); if a sequence has no text.txt, from LIGHT's released data.npz.
  - an UNTAGGED caption -> whole-sequence piece [0, T4] carrying ALL untagged captions of the sequence
    (one random caption per sample at train time); a TAGGED caption -> its own cropped piece.
    (No text.txt in the corpus carries a real time tag.)
  - length filter 36 <= len <= 300 frames @ 30 fps (== LIGHT's 24..200 @ 20 fps, inclusive).
  - InterAct's own parent rule: a 301..400-frame sequence WITHOUT chunk children `<seq>_<int>` is kept
    and TRUNCATED to [0,300] ("ok_truncated"); a parent WITH chunk children is dropped (the children
    are the pieces); > 400-frame parents without children are dropped.
  - lengths are 4-aligned (latent step = 4 frames); T4 = (T // 4) * 4.
  - InterAct has no per-segment action taxonomy: action_segs is EMPTY (the action token is never fed).

OUTPUT <out>/
  texts.json          {texts:[""(null caption), ...], captions:{seq->tid}, action_segs:{}, T4:{seq->T4},
                       seq_subset:{seq->subset}, fps}
  segments_cap{300,448,600}.json  {cap, fps, segments:[{seq,f0,f1,cap_tid,cap_tids}]}
  coverage.json       per-seq {part, subset, T4, text_src, reason, n_pieces, chunk_children, light_train}

v2nt corpus (the corpus the final PamiGen was trained on) = the above ("v2") minus the TRAIN pieces
whose reason is "ok_truncated" (847 truncated 301-400-frame parents); test pieces untouched.
"""
import collections
import csv
import json
import os
import re
import shutil

import numpy as np

from . import paths as P
from .subsets import TEXT_FALLBACK

CAPS = (300, 448, 600)
MIN_SEC = 24.0 / 20.0     # 1.2 s   LIGHT min_motion_len (24 @ 20 fps)
MAX_SEC = 200.0 / 20.0    # 10.0 s  LIGHT max (200 @ 20 fps) == InterAct's 300-frame chunk length
TRUNC_SEC = 400.0 / 30.0  # InterAct keeps seqs up to 400 frames (30 fps) and truncates them to 300
_CHUNK_RE = re.compile(r"^(.*)_\d+$")


def _r4(x):
    return int(round(x / 4)) * 4


def parse_humanml_text(path, stats=None):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("#")
            cap = parts[0].strip()
            if not cap:
                continue
            f_tag = to_tag = 0.0
            if len(parts) >= 4:
                try:
                    f_tag = float(parts[2]); to_tag = float(parts[3])
                except ValueError:
                    if stats is not None: stats["tag_unparsable"] = stats.get("tag_unparsable", 0) + 1
                    f_tag = to_tag = 0.0
            elif stats is not None:
                stats["untagged_short_line"] = stats.get("untagged_short_line", 0) + 1
            if np.isnan(f_tag): f_tag = 0.0
            if np.isnan(to_tag): to_tag = 0.0
            out.append((cap, f_tag, to_tag))
    return out


def light_dir(seq, sub, light_root):
    """Our seq name -> LIGHT's released dir (or None). grab is base-named in LIGHT (`s10_x_1_0` -> `s10_x_1`)."""
    cands = [seq]
    if sub == "grab":
        m = _CHUNK_RE.match(seq)
        if m:
            cands.append(m.group(1))
    for c in cands:
        d = f"{light_root}/{sub}/sequences_canonical/{c}"
        if os.path.isdir(d):
            return d
    return None


def _light_captions(d):
    p = os.path.join(d or "", "data.npz")
    if not d or not os.path.exists(p):
        return []
    try:
        with np.load(p, allow_pickle=True) as z:
            return [str(e.get("caption", "")).strip() for e in z["text"] if str(e.get("caption", "")).strip()]
    except Exception as e:
        print(f"  [light npz] {p}: {e}")
        return []


def build_text(split_json, out, fps=30, out_root=P.PAMI_DATA_ROOT, index_csv=P.INDEX_CSV, light_root=P.LIGHT_ROOT):
    FPS = fps
    min_frames = int(round(MIN_SEC * FPS)); max_frames = int(round(MAX_SEC * FPS)); trunc_max = int(round(TRUNC_SEC * FPS))
    os.makedirs(out, exist_ok=True)
    split = json.load(open(split_json))
    seq_subset = split.get("meta", {}).get("seq_subset", {})
    test_seqs = set(split.get("test", []))
    split_seqs = set(split.get("train", [])) | set(split.get("val", [])) | test_seqs
    light_test_p = f"{light_root}/stats/test_seq.txt"
    light_test = {l.strip().split("/")[-1] for l in open(light_test_p) if l.strip()} if os.path.exists(light_test_p) else set()
    _children = collections.defaultdict(int)
    for q in split_seqs:
        m = _CHUNK_RE.match(q)
        if m and m.group(1) in split_seqs:
            _children[m.group(1)] += 1

    Ttot, seq_dir = {}, {}
    for r in csv.DictReader(open(index_csv)):
        Ttot[r["seq_name"]] = int(r["end_frame"]) - int(r["start_frame"])
        seq_dir[r["seq_name"]] = os.path.dirname(r["source_path"])

    texts, tidx = [""], {"": 0}

    def tid(t):
        t = (t or "").strip()
        if not t: return 0
        if t not in tidx:
            tidx[t] = len(texts); texts.append(t)
        return tidx[t]

    captions, T4s, subset_of = {}, {}, {}
    pieces = []
    cov = collections.Counter()
    pstats = {}
    per_seq = {}
    for seq in sorted(split_seqs):
        sub = seq_subset.get(seq, "?")
        part = "test" if seq in test_seqs else "train"
        if seq not in Ttot or not os.path.isdir(f"{out_root}/output/{seq}"):
            per_seq[seq] = {"part": part, "subset": sub, "T4": None, "text_src": "none",
                            "reason": "not_in_index_or_pkg", "n_pieces": 0}
            continue
        T4 = (Ttot[seq] // 4) * 4
        if T4 < 8:
            per_seq[seq] = {"part": part, "subset": sub, "T4": T4, "text_src": "none", "reason": "too_short", "n_pieces": 0}
            continue
        T4s[seq] = T4; subset_of[seq] = sub; cov["n"] += 1
        row = {"part": part, "subset": sub, "T4": T4, "chunk_children": _children.get(seq, 0), "T4_eq_max": T4 == max_frames}
        per_seq[seq] = row

        _td = seq_dir.get(seq, "")
        if not os.path.exists(os.path.join(_td, "text.txt")):
            for _alt in TEXT_FALLBACK.get(sub, []):
                if os.path.exists(os.path.join(_alt, seq, "text.txt")):
                    _td = os.path.join(_alt, seq); break
        tpath = os.path.join(_td, "text.txt")
        _n0 = pstats.get("untagged_short_line", 0)
        lines = parse_humanml_text(tpath, pstats)
        row["text_src"] = "txt" if lines else "none"
        if lines and pstats.get("untagged_short_line", 0) > _n0:
            row["text_src"] = "txt_2field"; cov["text_2field"] += 1
        ld = light_dir(seq, sub, light_root)
        row["light_train"] = bool(ld and os.path.exists(os.path.join(ld, "data.npz")) and os.path.basename(ld) not in light_test)
        if not lines:
            lcaps = _light_captions(ld)
            if lcaps:
                lines = [(c, 0.0, 0.0) for c in lcaps]
                row["text_src"] = "light_npz"; cov["text_light_npz"] += 1
        elif ld is not None:
            lcaps = _light_captions(ld)
            if lcaps and len(lcaps) != len(lines):
                cov["cap_count_mismatch_vs_light"] += 1
        if not lines:
            cov["no_text"] += 1
            captions[seq] = 0
            row["reason"] = "no_caption_anywhere"; row["n_pieces"] = 0
            continue
        cov["with_text"] += 1
        captions[seq] = tid(lines[0][0])

        whole_tids = []
        n_before = len(pieces); n_tag_drop = 0
        for cap, f_tag, to_tag in lines:
            ct = tid(cap)
            if f_tag == 0.0 and to_tag == 0.0:
                whole_tids.append(ct); cov["whole"] += 1
                continue
            f0 = _r4(max(0, int(f_tag * FPS)))
            f1 = _r4(min(T4, int(to_tag * FPS)))
            cov["tagged"] += 1
            if f1 - f0 < min_frames or (f1 - f0) > max_frames:
                cov["dropped_len"] += 1; n_tag_drop += 1
                continue
            pieces.append((seq, int(f0), int(f1), [int(ct)]))
        whole_state = None
        if whole_tids:
            if T4 < min_frames:
                cov["dropped_len"] += 1; whole_state = "too_short"
            elif T4 > max_frames:
                if T4 <= trunc_max and not _children.get(seq, 0):
                    pieces.append((seq, 0, int(max_frames), whole_tids))
                    cov["truncated"] += 1; whole_state = "ok_truncated"
                else:
                    cov["dropped_len"] += 1
                    whole_state = "parent_chunked" if _children.get(seq, 0) else "parent_gt_trunc"
            else:
                pieces.append((seq, 0, T4, whole_tids))
        row["n_pieces"] = len(pieces) - n_before
        if whole_state == "ok_truncated":
            row["reason"] = "ok_truncated"
        elif row["n_pieces"]:
            row["reason"] = "ok"
        elif whole_state:
            row["reason"] = whole_state
        else:
            row["reason"] = "tagged_all_dropped_len" if n_tag_drop else "tagged_none"

    json.dump({"texts": texts, "captions": captions, "action_segs": {}, "T4": T4s, "seq_subset": subset_of, "fps": FPS},
              open(f"{out}/texts.json", "w"))
    for cap in CAPS:
        segs = [{"seq": s, "f0": f0, "f1": f1, "cap_tid": tids[0], "cap_tids": tids}
                for (s, f0, f1, tids) in pieces if (f1 - f0) <= cap]
        json.dump({"cap": cap, "fps": FPS, "segments": segs}, open(f"{out}/segments_cap{cap}.json", "w"))
        print(f"[cap {cap}] {len(segs)} pieces")
    print(f"[texts] fps={FPS} | {cov['n']} seqs, {len(texts)} unique texts, {len(pieces)} pieces "
          f"(whole {cov['whole']}, tagged {cov['tagged']}, dropped-by-len {cov['dropped_len']}, truncated {cov['truncated']}, "
          f"2-field text.txt {cov['text_2field']}, LIGHT caption fallback {cov['text_light_npz']}) -> {out}")
    json.dump({"min_frames": min_frames, "max_frames": max_frames, "trunc_max": trunc_max, "legacy": False,
               "split": split_json, "light_root": light_root, "seqs": per_seq}, open(f"{out}/coverage.json", "w"))
    OK = ("ok", "ok_truncated")
    for part in ("train", "test"):
        rows = {s: r for s, r in per_seq.items() if r["part"] == part}
        c = collections.Counter(r["reason"] for r in rows.values())
        print(f"  [{part}] {len(rows)} seqs, with piece {sum(c[x] for x in OK)}: {dict(c)}")
    return out


def make_nt(v2_dir, out, cap=300):
    """v2nt = v2 minus the TRAIN 'ok_truncated' pieces (from coverage.json). segments_cap{cap}.json is
    filtered; texts.json / coverage.json / text_feats.npz (if present) are copied unchanged."""
    os.makedirs(out, exist_ok=True)
    cov = json.load(open(f"{v2_dir}/coverage.json"))["seqs"]
    drop = {s for s, r in cov.items() if r["part"] == "train" and r.get("reason") == "ok_truncated"}
    sj = json.load(open(f"{v2_dir}/segments_cap{cap}.json"))
    segs = [g for g in sj["segments"] if g["seq"] not in drop]
    json.dump({"cap": sj["cap"], "fps": sj["fps"], "segments": segs,
               "note": "v2 minus TRAIN ok_truncated parents (301-400-frame seqs cut to 300); test pieces untouched"},
              open(f"{out}/segments_cap{cap}.json", "w"))
    for fn in ("texts.json", "coverage.json", "text_feats.npz"):
        src = f"{v2_dir}/{fn}"
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(f"{out}/{fn}"):
            shutil.copyfile(src, f"{out}/{fn}")
    print(f"[v2nt] {len(sj['segments'])} -> {len(segs)} pieces (dropped {len(drop)} truncated train parents) -> {out}")
    return out


def run(split_json=None, out=None, fps=30, out_root=P.PAMI_DATA_ROOT, index_csv=P.INDEX_CSV, light_root=P.LIGHT_ROOT):
    split_json = split_json or f"{out_root}/split_official_v3_noval.json"
    out = out or f"{out_root}/gen_corpus"      # not P.T2HOI_OUT: that may point at the frozen corpus of the released models
    v2 = build_text(split_json, out + "_v2", fps, out_root, index_csv, light_root)
    make_nt(v2, out)
