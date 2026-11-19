"""Checkpoint -> (network, dataset): the ONE supported way to load a PamiRefiner checkpoint.

A refiner checkpoint records everything its consumers need: `args` (d, layers, surf, dense_surf,
dense_seed), `sensor` (K, radius, tips, the cloud length the sensor saw), `arch_version`. The guards
refuse a checkpoint whose training regime, sensor configuration or surface clouds do not match what
this code would build -- those mismatches load without error and produce wrong numbers silently.
"""
import math

ARCH_VERSION = 4          # forward-pass / supervision regime this code implements
ARCH_NOTES = {4: "root dy channel supervised + measured foot band; single-object 54-token refiner"}


def check_arch(ck, path):
    v = ck.get("arch_version", 1)
    if v != ARCH_VERSION:
        raise SystemExit(f"{path}: arch_version {v} but this code is v{ARCH_VERSION} "
                         f"({ARCH_NOTES[ARCH_VERSION]}); the forward pass may load and still be wrong.")
    if ck.get("arm", "A2") != "A2":
        raise SystemExit(f"{path}: arm {ck.get('arm')!r} is an ablation arm; this code implements A2 (full body).")
    caps = ck.get("caps_obj") or {}
    if (float(caps.get("cm", 5.0)), float(caps.get("deg", 10.0))) != (5.0, 10.0):
        raise SystemExit(f"{path}: non-default object authority {caps}; this code carries 5 cm / 10 deg only.")
    return v


def dataset_cfg_from_ckpt(ck):
    """ck -> the BankWindows kwargs the checkpoint was trained with: {surf, dense_surf, dense_seed}."""
    ca = ck.get("args") or {}
    return dict(surf=int(ca.get("surf", 512)), dense_surf=int(ca.get("dense_surf", 0)),
                dense_seed=int(ca.get("dense_seed", 0)))


def build_m3_dataset(ck, dirs, stream, path="<ckpt>"):
    """checkpoint + bank dirs -> a BankWindows configured exactly as the trainer's was."""
    from pami.refine.data_m3 import BankWindows
    cfg = dataset_cfg_from_ckpt(ck)
    ds = BankWindows(dirs, cfg["surf"], stream=stream, dense_surf=cfg["dense_surf"], dense_seed=cfg["dense_seed"])
    check_dataset(ck, path, ds=ds)
    return ds


def check_dataset(ck, path, ds=None, batch=None):
    """Refuse a dataset / batch whose surface clouds disagree with the checkpoint's record."""
    from pami.refine.sensor_m3 import sensor_cfg_from_ckpt
    cfg = dataset_cfg_from_ckpt(ck)
    want_sensor = cfg["dense_surf"] or cfg["surf"]
    _, rec_surf = sensor_cfg_from_ckpt(ck)
    if int(rec_surf) != int(want_sensor):
        raise SystemExit(f"{path}: ck['sensor']['surf']={rec_surf} but args say dense_surf={cfg['dense_surf']} "
                         f"surf={cfg['surf']}: the checkpoint contradicts itself.")
    if ds is not None:
        got = dict(surf=int(ds.surf), dense_surf=int(ds.dense_surf), dense_seed=int(ds.dense_seed))
        if got != cfg:
            raise SystemExit(f"{path}: dataset config {got} != the checkpoint's {cfg}.")
    if batch is not None:
        from pami.refine.data_m3 import sensor_cloud
        n_canon = int(batch["canon"].shape[-2])
        n_sensor = int(sensor_cloud(batch)[0].shape[-2])
        rec_probe = (ck.get("sensor") or {}).get("probe_surf")
        if rec_probe is not None and n_canon != int(rec_probe):
            raise SystemExit(f"{path}: probe cloud is {n_canon} points but the checkpoint trained on {int(rec_probe)}.")
        if n_sensor != want_sensor or n_canon > cfg["surf"]:
            raise SystemExit(f"{path}: data carries canon={n_canon} / sensor cloud={n_sensor} points, checkpoint "
                             f"trained with surf={cfg['surf']} / sensor cloud {want_sensor}.")
    return cfg


def build_m3_net(ck, path, dev, batch=None, verbose=True):
    """checkpoint -> RefineNetM3 in eval mode, weights loaded strictly. Pass one collated batch so the
    surface clouds the data actually carries are checked against the checkpoint's record."""
    from pami.refine.features_m3 import D3_BODY, D3_FINGER, D3_OBJ, D3_SCENE
    from pami.refine.net_m3 import RefineNetM3
    from pami.refine.sensor_m3 import sensor_cfg_from_ckpt
    check_arch(ck, path)
    check_dataset(ck, path, batch=batch)
    ca = ck.get("args") or {}
    s_kw, s_surf = sensor_cfg_from_ckpt(ck)
    net = RefineNetM3(D3_BODY, D3_FINGER, D3_OBJ, D3_SCENE, d=ca.get("d", 256), layers=ca.get("layers", 4),
                      cap_obj_m=0.05, cap_obj_rad=math.radians(10.0), sensor_kw=s_kw).to(dev)
    s = net.sensor
    got = dict(K=s.K, radius=s.radius, tips=s.tips)
    want = dict(K=int(s_kw["K"]), radius=float(s_kw["radius"]), tips=bool(s_kw["tips"]))
    if got != want:
        raise SystemExit(f"{path}: sensor config mismatch -- checkpoint {want}, built {got}.")
    if ck.get("periodic"):
        print(f"[arch_guard] periodic checkpoint (it={ck.get('it')}), saved on the iteration schedule "
              f"(not gate-admitted).", flush=True)
    sd = dict(ck["model"] if "model" in ck else ck)
    sd.pop("body_mask", None)                    # A2 = all joints active; the old all-ones mask buffer is gone
    net.load_state_dict(sd)                      # strict: no silent partial load
    net.eval()
    if verbose:
        print(f"[refiner] {path}: it={ck.get('it')} sensor K={want['K']} r={want['radius']} tips={want['tips']} "
              f"cloud={s_surf}", flush=True)
    return net
