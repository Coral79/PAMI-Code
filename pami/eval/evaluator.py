"""LIGHT-exact evaluation of exported PAMI samples (port of the paper's ``light_exact_eval.py``).

Metrics = exactly the LIGHT harness set (LIGHT ``eval/eval_t2hoi_wguide.py``):
  semantic : R-precision Top-1/2/3, multimodal (matching) distance, FID, diversity -- LIGHT's
             evaluator networks (``assets/evaluator.ckpt``) on LIGHT's 549-d features
             (156 joint coords + 9 object pose + 384 BPS deltas) of the FULL (untrimmed) sequence.
  geometric: foot-skating ratio | contact micro P/R/F1 + contact ratio (52 joints vs 340 object
             points, 5 cm) and the hand block (min over joints 22:52) | penetration (52 joints in
             the predicted object's rest frame -> SDF, mean |negative|). All on the first
             ``T-4`` frames (LIGHT's ``lengths-4`` trim), sequences capped at 300 frames.

Everything numeric is either LIGHT's function (imported from LIGHT_ROOT, unmodified) or the
exact statement of the original script; see docs/EVAL.md.
"""
import os, json, glob, hashlib, datetime
import numpy as np
import torch

from . import light_env

MAXT = 300                      # LIGHT caps every sequence at 300 frames
MIN_FRAMES = 8                  # rows shorter than this are skipped (original filter)
ENC_CHUNK = 24                  # motion-encoder chunk size (padding group); kept for bit-exactness
from pami.data_prep import paths as _P
DEFAULT_TEST_LIST = os.path.join(_P.PAMI_DATA_ROOT, "test_pool", "interact_test.json")      # prepare_data.py test_pool


def sha1_file(path, n=12):
    try:
        return hashlib.sha1(open(path, "rb").read()).hexdigest()[:n]
    except Exception:
        return None


# --------------------------------------------------------------------------------------------
# LIGHT pieces (loaded lazily, after light_env.setup_light_root)
# --------------------------------------------------------------------------------------------
class LightEvaluator:
    """Evaluator networks + feature builder, all from LIGHT_ROOT."""

    def __init__(self, device, evaluator_ckpt="assets/evaluator.ckpt"):
        from bps_torch.bps import bps_torch
        from tma.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
        self.root = light_env.light_root()
        self.dev = device
        self.ckpt_path = evaluator_ckpt if os.path.isabs(evaluator_ckpt) else os.path.join(self.root, evaluator_ckpt)
        sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        self._sd = sd
        self.me = ActorAgnosticEncoder(nfeats=549, vae=True, latent_dim=256, ff_size=1024, num_layers=4).to(device).eval()
        self.me.load_state_dict({k[14:]: v for k, v in sd.items() if k.startswith("motionencoder")})
        self.bps = bps_torch()
        self.basis = torch.from_numpy(np.load(os.path.join(self.root, "InterAct/stats/bps_basis_set_128_1.npy"))).float().to(device)
        self._te = None
        self._canrot = {}

    # -- text tower (DistilBERT actor-agnostic encoder of the SAME checkpoint) --
    def text_encoder(self):
        if self._te is None:
            from tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
            te = DistilbertActorAgnosticEncoder("distilbert-base-uncased", latent_dim=256, ff_size=1024, num_layers=4).to(self.dev).eval()
            te.load_state_dict({k[12:]: v for k, v in self._sd.items() if k.startswith("textencoder")})
            self._te = te
        return self._te

    @torch.no_grad()
    def embed_texts(self, texts, bs=64):
        te = self.text_encoder()
        return torch.cat([te(texts[i:i + bs]).loc.float().cpu() for i in range(0, len(texts), bs)])

    def canrot(self, subset, obj):
        if (subset, obj) not in self._canrot:
            p = os.path.join(self.root, f"InterAct/{subset}/objects/{obj}/{obj}_can.npy")
            self._canrot[(subset, obj)] = torch.from_numpy(np.load(p)).float().to(self.dev)
        return self._canrot[(subset, obj)]

    def feats549(self, j156, R, t, obj_points, cr):
        """LIGHT evaluator features: joints (T,156) | object 6d-rot (in canonical frame) + transl | BPS deltas (384)."""
        from pytorch3d.transforms import matrix_to_rotation_6d
        T = j156.shape[0]; R, t = R.float().to(self.dev), t.float().to(self.dev)
        opt = torch.from_numpy(obj_points.astype(np.float32)).to(self.dev)
        posed = torch.bmm(opt[None].expand(T, -1, -1), R.transpose(1, 2)) + t[:, None]
        deltas = self.bps.encode(x=posed, feature_type=["deltas"],
                                 custom_basis=self.basis[None].expand(T, -1, -1) + t[:, None])["deltas"]
        o9 = torch.cat([matrix_to_rotation_6d(torch.einsum("tij,jk->tik", R, cr.T)), t], -1)
        return torch.cat([torch.from_numpy(j156.astype(np.float32)).to(self.dev), o9, deltas.reshape(T, 384)], -1)

    @torch.no_grad()
    def encode_motions(self, feats):
        """(list of (T_i,549) tensors) -> (N,256) motion embeddings, chunked exactly like the original."""
        E = []
        for i in range(0, len(feats), ENC_CHUNK):
            ch = feats[i:i + ENC_CHUNK]; Ls = [f.shape[0] for f in ch]
            pad = torch.zeros(len(ch), max(Ls), 549, device=self.dev)
            for k, f in enumerate(ch): pad[k, :f.shape[0]] = f
            E.append(self.me(pad, Ls).loc.float().cpu())
        return torch.cat(E)


# --------------------------------------------------------------------------------------------
# Text-embedding cache
# --------------------------------------------------------------------------------------------
def load_text_embeddings(ev, rows, cache_path, evaluator_sha1):
    """(N,256) text embeddings of the test rows, in row order.

    Cache formats accepted:
      * release format: dict(temb, texts, names, ckpt_sha1) -- looked up by sequence name, text checked;
      * the paper's historical pool cache: dict(temb, texts, ...) built over the 1449-row pool --
        looked up by ``pool_index``, text checked (reusing it reproduces the stored numbers bit-exactly).
    Missing -> built with the text tower of the same evaluator checkpoint and saved.
    """
    if cache_path and os.path.exists(cache_path):
        C = torch.load(cache_path, map_location="cpu", weights_only=False)
        texts = list(C["texts"]); temb = C["temb"]
        if "names" in C:
            idx = {n: i for i, n in enumerate(C["names"])}
            sel = [idx.get(r["name"]) for r in rows]
        else:
            sel = [r.get("pool_index") if r.get("pool_index", 10**9) < len(texts) else None for r in rows]
        ok = all(s is not None and texts[s] == r["text"] for s, r in zip(sel, rows))
        if ok:
            print(f"[text-emb] reusing cache {cache_path} ({len(texts)} rows)", flush=True)
            return temb[torch.tensor(sel)], cache_path
        print(f"[text-emb] cache {cache_path} does not cover the test list -> rebuilding", flush=True)
    temb = ev.embed_texts([r["text"] for r in rows])
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(dict(temb=temb, texts=[r["text"] for r in rows], names=[r["name"] for r in rows],
                        ckpt=ev.ckpt_path, ckpt_sha1=evaluator_sha1), cache_path)
        print(f"[text-emb] built {tuple(temb.shape)} -> {cache_path}", flush=True)
    return temb, cache_path


# --------------------------------------------------------------------------------------------
# Row loaders  ->  (J52 (T,52,3) np.float32, R (T,3,3) torch, t (T,3) torch) or None
# --------------------------------------------------------------------------------------------
def load_gt(row, maxt=MAXT):
    from pytorch3d.transforms import rotation_6d_to_matrix
    p = os.path.join(light_env.light_root(), f"InterAct/{row['subset']}/sequences_canonical/{row['light_name']}/data.npz")
    dz = np.load(p, allow_pickle=True)
    m = np.asarray(dz["motion"], dtype=np.float32)[:maxt]
    return (m[:, :156].reshape(-1, 52, 3), rotation_6d_to_matrix(torch.from_numpy(m[:, 186:192])),
            torch.from_numpy(m[:, 192:195]))


def gt_obj_points(row):
    p = os.path.join(light_env.light_root(), f"InterAct/{row['subset']}/sequences_canonical/{row['light_name']}/data.npz")
    return np.asarray(np.load(p, allow_pickle=True)["obj_points"])[:, :3]     # 340 points, LIGHT's contact cloud


def pred_path(pred_dir, row):
    return os.path.join(pred_dir, f"refined_{row['name']}.npz")


def load_pred(pred_dir, row, maxt=MAXT):
    """Exported sample: npz with joints52 (T,52,3), R_obj (T,3,3), p_obj (T,3) in LIGHT's canonical frame."""
    fo = pred_path(pred_dir, row)
    if not os.path.exists(fo): return None
    oz = np.load(fo, allow_pickle=True)
    J = np.asarray(oz["joints52"], dtype=np.float32)[:maxt]
    return (J, torch.from_numpy(np.asarray(oz["R_obj"], dtype=np.float32))[:maxt],
            torch.from_numpy(np.asarray(oz["p_obj"], dtype=np.float32))[:maxt])


# --------------------------------------------------------------------------------------------
# R-precision / matching score
# --------------------------------------------------------------------------------------------
def rprecision(tm, E, protocol="light64", bs=None, lengths=None, seed=0, passes=5):
    """Top-1/2/3 R-precision and matching (multimodal) distance between text (tm) and motion (E) embeddings.

    protocol "light64" (PAPER, default): rows shuffled with a fixed seed, ``passes`` passes, batches of
      ``bs`` (64), remainder of each pass dropped; ranking by torch.argsort of the cdist matrix.
      Reproduces LIGHT's revised Table 1 protocol ("R-Precision uses batch size 64").
    protocol "light" (LIGHT's public code, pre-revision): rows sorted by length ascending, no shuffle,
      consecutive batches of ``bs`` (256), the LAST batch filled by repeating its own items
      (t2hoi_collate), np.argsort ranking, one pass. Numbers are NOT comparable across protocols.
    Returns dict(r1, r1_std, r2, r3, matching).
    """
    n = len(tm)
    if bs is None: bs = 64 if protocol == "light64" else 256
    BS = min(int(bs), n)
    R1, R2, R3, match = [], [], [], []
    if protocol == "light":
        assert lengths is not None
        order = np.argsort(np.asarray(lengths), kind="stable")
        tmn, En = tm.numpy(), E.numpy()
        for s0 in range(0, len(order), BS):
            idx = list(order[s0:s0 + BS])
            if len(idx) < BS: idx = (idx * (-(-BS // len(idx))))[:BS]      # LIGHT t2hoi_collate repeat-fill
            idx = np.array(idx)
            dm = np.linalg.norm(tmn[idx][:, None, :] - En[idx][None, :, :], axis=-1)   # euclidean_distance_matrix
            match.append(float(np.trace(dm) / BS))
            rank = np.argsort(dm, axis=1)                                     # calculate_R_precision (np.argsort)
            hit = rank[:, :3] == np.arange(BS)[:, None]
            R1.append(float(hit[:, :1].any(1).mean())); R2.append(float(hit[:, :2].any(1).mean())); R3.append(float(hit.any(1).mean()))
    elif protocol == "light64":
        g = torch.Generator().manual_seed(seed)
        for _ in range(passes):
            perm = torch.randperm(n, generator=g)
            for s0 in range(0, len(perm) - BS + 1, BS):
                idx = perm[s0:s0 + BS]
                d = torch.cdist(tm[idx], E[idx])
                match.append(float(d.diag().mean()))
                rank = d.argsort(1)
                hit = rank[:, :3] == torch.arange(BS)[:, None]
                R1.append(float(hit[:, :1].any(1).float().mean()))
                R2.append(float(hit[:, :2].any(1).float().mean()))
                R3.append(float(hit.any(1).float().mean()))
    else:
        raise ValueError(protocol)
    return dict(r1=float(np.mean(R1)), r1_std=float(np.std(R1)), r2=float(np.mean(R2)),
                r3=float(np.mean(R3)), matching=float(np.mean(match)))


# --------------------------------------------------------------------------------------------
# One method row
# --------------------------------------------------------------------------------------------
def score_rows(ev, rows, temb, loader, skip_pen=False, limit=0, log_name="", per_seq=False):
    """Per-sequence geometric metrics + motion embeddings for one method.

    loader(row) -> (J, R, t) or None (row skipped). Returns dict with kept indices, embeddings, lists."""
    from utils.eval_t2m_utils import calculate_skating_ratio, contact_metric_micro, compute_penetration_metric2
    embs, keep, tembs_row = [], [], []
    sk_list, pen_list, pen_skip = [], [], 0
    cf1, cp_, cr_, cpct = [], [], [], []
    hf1, hp_, hr_, hpct = [], [], [], []
    names, lens = [], []
    with torch.no_grad():
        for i, r in enumerate(rows):
            got = loader(r)
            if got is None: continue
            J, R_, t_ = got
            gt_ = load_gt(r)
            Tm = min(len(J), len(gt_[0]))
            if Tm < MIN_FRAMES: continue
            L = Tm - 4                                   # LIGHT's lengths-4 trim
            op = gt_obj_points(r)
            # semantic feats (full length, untrimmed)
            embs.append(ev.feats549(J.reshape(len(J), -1), R_, t_, op, ev.canrot(r["subset"], r["obj"])))
            keep.append(i); tembs_row.append(temb[i]); names.append(r["name"]); lens.append(int(len(J)))
            # --- geometric, all on [:L] ---
            jt = torch.from_numpy(J[:L]).float()
            opt = torch.from_numpy(op.astype(np.float32))
            po = torch.bmm(opt[None].expand(L, -1, -1), R_[:L].float().transpose(1, 2)) + t_[:L].float()[:, None]
            gj = torch.from_numpy(gt_[0][:L]).float()
            go = torch.bmm(opt[None].expand(L, -1, -1), gt_[1][:L].float().transpose(1, 2)) + gt_[2][:L].float()[:, None]
            # LIGHT divides the skate-frame count by the PADDED batch length (n_frames=300), not the
            # sequence's own length -- pad to 300 to reproduce.
            m22p = torch.zeros(1, 22, 3, MAXT); m22p[..., :L] = jt[:, :22].permute(1, 2, 0)[None]
            mskp = torch.zeros(1, 1, 1, MAXT); mskp[..., :L] = 1.0
            sk_list.append(float(calculate_skating_ratio(m22p, mskp)[0][0]))
            cm = contact_metric_micro(gj, go, jt, po, contact_threh=0.05)
            cf1.append(cm["micro_f1"]); cp_.append(cm["micro_precision"])
            cr_.append(cm["micro_recall"]); cpct.append(cm["mean_pred_contact_percent"])
            hf1.append(cm["micro_f1_hand"]); hp_.append(cm["micro_precision_hand"])
            hr_.append(cm["micro_recall_hand"]); hpct.append(cm["mean_pred_contact_percent_hand"])
            if skip_pen:
                pen_skip += 1
            else:
                try:
                    pen = compute_penetration_metric2(0, r["subset"], r["obj"], jt.to(ev.dev),
                                                      t_[:L].float().to(ev.dev), R_[:L].float().to(ev.dev))
                    pen_list.append(float(pen))
                except Exception as e:   # noqa: BLE001 -- counted; fatal at the end unless allowed
                    print(f"[{log_name}] penetration failed on {r['name']}: {e!r}", flush=True); pen_skip += 1
            if limit and len(keep) >= limit: break
            if (i + 1) % 300 == 0: print(f"[{log_name}] {i+1}/{len(rows)}", flush=True)
        E = ev.encode_motions(embs)
    out = dict(keep=np.array(keep), E=E, tm=torch.stack(tembs_row), lengths=[f.shape[0] for f in embs],
               sk=sk_list, pen=pen_list, pen_skip=pen_skip,
               c_f1=cf1, c_prec=cp_, c_rec=cr_, c_pct=cpct, ch_f1=hf1, ch_prec=hp_, ch_rec=hr_, ch_pct=hpct)
    if per_seq:
        out["per_seq"] = dict(names=names, L=lens, fsr=[float(x) for x in sk_list], c_f1=[float(x) for x in cf1],
                              ch_f1=[float(x) for x in hf1], c_pct=[float(x) for x in cpct], ch_pct=[float(x) for x in hpct],
                              pen=[float(x) for x in pen_list], c_prec=[float(x) for x in cp_], c_rec=[float(x) for x in cr_],
                              ch_prec=[float(x) for x in hp_], ch_rec=[float(x) for x in hr_])
    return out


def summarize(S, gtS, protocol, rprec_bs=None, seed=0, passes=5):
    """Aggregate one method's per-sequence results into the paper's row dict (same keys as the stored JSONs)."""
    from utils.eval_t2m_utils import calculate_diversity, calculate_activation_statistics, calculate_frechet_distance
    ki, E = S["keep"], S["E"]
    rp = rprecision(S["tm"], E, protocol=protocol, bs=rprec_bs, lengths=S["lengths"], seed=seed, passes=passes)
    gtki = gtS["keep"]
    assert np.isin(ki, gtki).all(), "kept rows are not a subset of the GT rows (--limit hazard)"
    sel = np.searchsorted(gtki, ki)                      # FID reference = GT embeddings of the SAME rows
    np.random.seed(0)                                    # calculate_diversity uses unseeded np.random.choice
    mu1, s1 = calculate_activation_statistics(E.numpy())
    mu2, s2 = calculate_activation_statistics(gtS["E"][sel].numpy())
    fid = float(calculate_frechet_distance(mu2, s2, mu1, s1))
    dt = 300 if len(E) > 300 else max(2, len(E) // 2)
    div = float(calculate_diversity(E.numpy(), dt))
    row = dict(rprec_protocol=protocol, n=int(len(ki)), **rp, fid=fid, diversity=div,
               foot_skate=float(np.mean(S["sk"])), pen=float(np.mean(S["pen"])) if S["pen"] else None,
               pen_n=len(S["pen"]), pen_skip=S["pen_skip"],
               c_f1=float(np.mean(S["c_f1"])), c_prec=float(np.mean(S["c_prec"])), c_rec=float(np.mean(S["c_rec"])),
               c_pct=float(np.mean(S["c_pct"])), ch_f1=float(np.mean(S["ch_f1"])), ch_prec=float(np.mean(S["ch_prec"])),
               ch_rec=float(np.mean(S["ch_rec"])), ch_pct=float(np.mean(S["ch_pct"])))
    # explicit names: c_f1 = 52-joint micro F1 (not in the paper); ch_f1 = hand-block F1 = the paper's C_F1
    row["all_joint_f1"], row["paper_hand_c_f1"] = row["c_f1"], row["ch_f1"]
    if "per_seq" in S: row["per_seq"] = S["per_seq"]
    return row


def print_row(name, row):
    pen = row["pen"] if row["pen"] is None else round(row["pen"], 4)
    print(f"{name:28s} n={row['n']:4d} R@1 {row['r1']:.3f} R@2 {row['r2']:.3f} R@3 {row['r3']:.3f} "
          f"Match {row['matching']:.3f} FID {row['fid']:.3f} Div {row['diversity']:.2f} | "
          f"skate {row['foot_skate']:.4f} pen {pen} | "
          f"cF1 {row['c_f1']:.3f} P {row['c_prec']:.3f} R {row['c_rec']:.3f} pct {row['c_pct']:.3f} | "
          f"hand F1 {row['ch_f1']:.3f} P {row['ch_prec']:.3f} R {row['ch_rec']:.3f} pct {row['ch_pct']:.3f}", flush=True)


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------
def evaluate(pred_dir, out_path, light_root, test_list=DEFAULT_TEST_LIST, name=None, rprec="light64", rprec_bs=None,
             rprec_seed=0, rprec_passes=5, skip_pen=False, allow_pen_skip=False, text_emb_cache=None,
             evaluator_ckpt="assets/evaluator.ckpt", limit=0, per_seq=False, device=None):
    """Evaluate one export directory against GT. Writes ``out_path`` (JSON: _manifest, gt, <name>) and returns it."""
    pred_dir = os.path.abspath(pred_dir); out_path = os.path.abspath(out_path)
    test_list = os.path.abspath(test_list)
    if text_emb_cache: text_emb_cache = os.path.abspath(text_emb_cache)
    if evaluator_ckpt and not os.path.isabs(evaluator_ckpt) and os.path.exists(evaluator_ckpt):
        evaluator_ckpt = os.path.abspath(evaluator_ckpt)
    light_env.setup_light_root(light_root)
    name = name or os.path.basename(pred_dir.rstrip("/"))
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not skip_pen:
        raise SystemExit("penetration (LIGHT's SDF sampler) needs CUDA; run on a GPU or pass --skip_pen")

    rows = json.load(open(test_list))["rows"]
    n_files = len(glob.glob(os.path.join(pred_dir, "refined_*.npz")))
    print(f"[eval] {len(rows)} test rows | pred_dir={pred_dir} ({n_files} refined_*.npz) | rprec={rprec} | device={device}", flush=True)
    ev = LightEvaluator(device, evaluator_ckpt)
    ev_sha = sha1_file(ev.ckpt_path)
    temb, cache_used = load_text_embeddings(ev, rows, text_emb_cache, ev_sha)
    assert temb.shape[0] == len(rows)

    out = {"_manifest": dict(pred_dir=pred_dir, test_list=test_list, test_list_sha1=sha1_file(test_list),
                             light_root=light_root, evaluator_ckpt=ev.ckpt_path, evaluator_sha1=ev_sha,
                             text_emb_cache=cache_used, rprec_protocol=rprec,
                             rprec_bs=int(rprec_bs) if rprec_bs else (64 if rprec == "light64" else 256),
                             rprec_seed=rprec_seed, rprec_passes=rprec_passes, skip_pen=bool(skip_pen), limit=limit,
                             device=device, script_sha1=sha1_file(os.path.abspath(__file__)),
                             date=datetime.datetime.now().isoformat(timespec="minutes"))}
    gtS = score_rows(ev, rows, temb, lambda r: load_gt(r), skip_pen=skip_pen, limit=limit, log_name="gt", per_seq=per_seq)
    out["gt"] = summarize(gtS, gtS, rprec, rprec_bs, rprec_seed, rprec_passes); print_row("gt", out["gt"])
    S = score_rows(ev, rows, temb, lambda r: load_pred(pred_dir, r), skip_pen=skip_pen, limit=limit, log_name=name, per_seq=per_seq)
    out[name] = summarize(S, gtS, rprec, rprec_bs, rprec_seed, rprec_passes); print_row(name, out[name])
    for k in ("gt", name):
        if out[k]["pen_skip"] and not (skip_pen or allow_pen_skip):
            raise SystemExit(f"Error: {k}: penetration failed on {out[k]['pen_skip']} rows (pass --allow_pen_skip to tolerate)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}", flush=True)
    return out
