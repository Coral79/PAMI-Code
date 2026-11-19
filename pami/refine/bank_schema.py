"""The versioned contract between build_real_bank.py (producer) and data_m3.py (consumer).

A bank directory = one item npz per window + `build_spec.json` (written by --init), one
`manifest.part-<tag>.json` per --append shard, and `manifest.json` (written by --finalize).
`check_bank_dir` refuses anything this code cannot consume: a schema / mask version other than the
one below, an unfinalized or torn build, a manifest whose counts disagree with the directory, shard
parts that do not match the spec, and (deterministically, every item) an item whose stamp triple
differs from the manifest. Two dirs may be loaded together only if COMPARED_FIELDS agree.
"""
import glob
import hashlib
import json
import os
import sys

BANK_SCHEMA = "m3-bank/3"
MASK_VERSION = "mask/v4"
MANIFEST_NAME = "manifest.json"
BUILD_SPEC_NAME = "build_spec.json"
PART_PREFIX = "manifest.part-"
PART_GLOB = PART_PREFIX + "*.json"
# name -> (shape, dtype). W = window length, S = number of contact segments, M = 1 object.
BANK_FIELDS = {
    "h272": ("(W,272)", "float32"), "rot6h": ("(W,30,6)", "float32"),
    "root_wf_pos": ("(W,3)", "float32"), "root_wf_rot6": ("(W,6)", "float32"),
    "p_obj": ("(M,W,3)", "float32"), "r6_obj": ("(M,W,6)", "float32"),
    "contact_intent": ("(W,5)", "bool"), "reachable_mask": ("(W,5)", "bool"),
    "segment_id": ("(W,5)", "int32"), "segment_anchor": ("(S,10)", "float32"),
    "foot_contact_intent": ("(W,2)", "bool"), "ik_err_mm": ("()", "float"),
    "seq": ("()", "str"), "f0": ("()", "int"), "meta": ("()", "str (json)"),
    "bank_schema": ("()", "str"), "mask_version": ("()", "str"), "mask_id": ("()", "str"),
    "hand_mode": ("()", "str"), "hand_source": ("()", "str"),
}
COMPARED_FIELDS = ("bank_schema", "mask_version", "mask_id", "stream", "hand_mode",
                   "root_dialect", "window", "stride", "varlen", "floor_strategy", "mask_params")
PART_SPEC_FIELDS = ("bank_schema", "mask_version", "mask_id", "stream", "floor_strategy",
                    "hand_mode", "window", "stride", "varlen", "mask_params")
ITEM_STAMP_FIELDS = ("bank_schema", "mask_version", "mask_id")
SKIP_SCAN_ENV = "M3_BANK_SKIP_ITEM_SCAN"        # =1 degrades the full item scan to a 16-item sample


def mask_id(floor_strategy, params, version=MASK_VERSION):
    """The COMPARED mask identity: version + floor strategy + a hash of every threshold."""
    blob = json.dumps(params, sort_keys=True, default=float).encode()
    return f"{version}|floor={floor_strategy}|p={hashlib.sha1(blob).hexdigest()[:10]}"


def count_items(d):
    return len(glob.glob(os.path.join(d, "*.npz")))


def atomic_write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def item_stamps(path):
    import numpy as np
    with np.load(path, allow_pickle=True) as z:
        return tuple(str(z[k]) if k in z.files else None for k in ITEM_STAMP_FIELDS)


def items_digest(files):
    """Order-independent sha256 over sorted `basename|bank_schema|mask_version|mask_id` lines."""
    h = hashlib.sha256()
    for f in sorted(files, key=os.path.basename):
        s = item_stamps(f)
        h.update(("|".join([os.path.basename(f)] + [x if x else "" for x in s]) + "\n").encode())
    return h.hexdigest()


def verify_items(d, want, files=None, sample=None):
    """Compare item stamp triples with `want`; sample=None opens every item. Returns (n_checked, n)."""
    files = sorted(files if files is not None else glob.glob(os.path.join(d, "*.npz")))
    n = len(files)
    if not n:
        return 0, 0
    if sample is None:
        idx = range(n)
    else:
        k = min(sample, n)
        idx = sorted({0, n - 1} | {i * (n - 1) // max(1, k - 1) for i in range(k)})
    checked = 0
    for i in idx:
        got = item_stamps(files[i])
        checked += 1
        if got != want:
            raise RuntimeError(f"bank dir {d}: item {os.path.basename(files[i])} is stamped {got}, the manifest "
                               f"says {want}. The directory mixes two builds; rebuild into a fresh directory.")
    return checked, n


_SCAN_CACHE = {}


def _scan_items(d, m, files=None):
    p = os.path.join(d, MANIFEST_NAME)
    want = tuple(m.get(k) for k in ITEM_STAMP_FIELDS)
    files = sorted(files if files is not None else glob.glob(os.path.join(d, "*.npz")))
    key = (os.path.abspath(d), os.path.getmtime(p), len(files), want)
    if key in _SCAN_CACHE:
        return _SCAN_CACHE[key]
    skip = os.environ.get(SKIP_SCAN_ENV, "") in ("1", "true", "yes")
    checked, n = verify_items(d, want, files=files, sample=16 if skip else None)
    if skip:
        print(f"[{SKIP_SCAN_ENV}] {d}: item stamps SAMPLED ({checked}/{n} opened), not verified",
              file=sys.stderr, flush=True)
    elif m.get("items_sha256"):
        got = items_digest(files)
        if got != m["items_sha256"]:
            raise RuntimeError(f"bank dir {d}: item-stamp digest {got[:12]}... != manifest's "
                               f"{m['items_sha256'][:12]}...: the set of items changed after finalize.")
    _SCAN_CACHE[key] = (checked, n)
    return checked, n


def verify_parts(d, m=None):
    """Re-verify the shard part manifests against build_spec.json at load time."""
    sp = os.path.join(d, BUILD_SPEC_NAME)
    parts = sorted(glob.glob(os.path.join(d, PART_GLOB)))
    if not os.path.isfile(sp):
        return 0
    if not parts:
        raise RuntimeError(f"bank dir {d}: {BUILD_SPEC_NAME} present but no {PART_GLOB}; rebuild.")
    spec = json.load(open(sp))
    listed = m.get("shard_parts") if m else None
    if listed is not None and sorted(listed) != sorted(os.path.basename(x) for x in parts):
        raise RuntimeError(f"bank dir {d}: shard manifests on disk differ from the finalized list; re-run --finalize.")
    spec_files = set(spec.get("gen_files", []))
    seen, total = {}, 0
    for p in parts:
        j = json.load(open(p))
        bad = [k for k in PART_SPEC_FIELDS
               if json.dumps(j.get(k), sort_keys=True, default=float) != json.dumps(spec.get(k), sort_keys=True, default=float)]
        if bad:
            raise RuntimeError(f"bank dir {d}: {os.path.basename(p)} disagrees with {BUILD_SPEC_NAME} on {bad}.")
        alien = [f for f in j.get("gen_files", []) if f not in spec_files]
        if alien:
            raise RuntimeError(f"bank dir {d}: {os.path.basename(p)} claims {len(alien)} gen npz outside the spec (e.g. {alien[0]}).")
        for f in j.get("gen_files", []):
            if f in seen:
                raise RuntimeError(f"bank dir {d}: {f} is claimed by two shards ({seen[f]}, {os.path.basename(p)}).")
            seen[f] = os.path.basename(p)
        total += int(j.get("n_windows", 0))
    if m is not None and m.get("n_windows") is not None and total != m["n_windows"]:
        raise RuntimeError(f"bank dir {d}: shard manifests sum to {total} windows, manifest says {m['n_windows']}.")
    return len(parts)


def check_bank_dir(d, expect=None, check_items=True):
    """Hard-fail unless <d>/manifest.json declares a complete bank this code can consume.
    expect: optional dict of COMPARED_FIELDS values to pin (e.g. stream="roundtrip"). Returns the manifest."""
    p = os.path.join(d, MANIFEST_NAME)
    if not os.path.isfile(p):
        parts = glob.glob(os.path.join(d, PART_GLOB))
        if parts:
            raise RuntimeError(f"bank dir {d} has {len(parts)} shard manifests but no {MANIFEST_NAME}: run "
                               f"build_bank.py --finalize --out {d}")
        raise RuntimeError(f"bank dir {d} has no {MANIFEST_NAME}: not a {BANK_SCHEMA} bank; rebuild with build_bank.py")
    m = json.load(open(p))
    got = (m.get("bank_schema"), m.get("mask_version"))
    if got != (BANK_SCHEMA, MASK_VERSION):
        raise RuntimeError(f"bank dir {d}: manifest says bank_schema={got[0]!r} mask_version={got[1]!r}, "
                           f"code expects {BANK_SCHEMA!r}/{MASK_VERSION!r}.")
    if not m.get("complete"):
        raise RuntimeError(f"bank dir {d}: manifest has no `complete: true` (interrupted finalize); re-run --finalize.")
    on_disk = count_items(d)
    claimed = m.get("n_items_on_disk", m.get("n_windows"))
    if claimed != on_disk:
        raise RuntimeError(f"bank dir {d}: manifest claims {claimed} items, {on_disk} *.npz present; re-run --finalize.")
    n_win = m.get("n_windows")
    if n_win is not None and m.get("n_items_on_disk") is not None and n_win != on_disk:
        raise RuntimeError(f"bank dir {d}: n_windows={n_win} != n_items_on_disk={m['n_items_on_disk']}.")
    verify_parts(d, m)
    if check_items:
        _scan_items(d, m)
    for k, v in (expect or {}).items():
        if m.get(k) != v:
            raise RuntimeError(f"bank dir {d}: manifest {k}={m.get(k)!r}, caller requires {v!r}.")
    return m


def check_banks_agree(manifests, dirs, fields=COMPARED_FIELDS):
    """Every dir loaded together must agree on the mask identity and the pipeline dialect."""
    for k in fields:
        vals = {json.dumps(bool(m.get(k)) if k == "varlen" else m.get(k), sort_keys=True, default=float)
                for m in manifests}
        if len(vals) > 1:
            detail = ", ".join(f"{d}: {m.get(k)!r}" for d, m in zip(dirs, manifests))
            raise RuntimeError(f"bank dirs disagree on {k!r} -- {detail}; they are different builds.")
