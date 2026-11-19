# Evaluation

`evaluate.py` scores a folder of refined results with the InterAct text-to-HOI protocol of
[LIGHT](https://github.com/wzyabcas/LIGHT). LIGHT's evaluator networks, metric functions and data are imported from
the LIGHT clone.

## Requirements

- the evaluation packages of [`environment/README.md`](../environment/README.md) §2
- `LIGHT_ROOT`: the LIGHT clone with its processed InterAct data and `assets/evaluator.ckpt` (steps 3 and 5 of the
  LIGHT README)
- the test list of `python prepare_data.py test_pool`
- a GPU for the penetration metric; `--skip_pen --device cpu` computes everything else on CPU

## Usage

```bash
export LIGHT_ROOT=$PWD/data/LIGHT
export OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python evaluate.py --pred_dir outputs/refined_demo --out outputs/eval/refined_demo.json
```

`--pred_dir` holds one `refined_<sequence>.npz` per test sequence, as written by `refine.py`: `joints52 (T,52,3)`,
`R_obj (T,3,3)`, `p_obj (T,3)`. Missing sequences are skipped. Other options: `--test_list`, `--skip_pen`,
`--per_seq`, `--limit N`, `--device`.

## Output

The JSON has a `gt` row and one row for the prediction folder:

| key | metric |
|---|---|
| `r1`, `r2`, `r3` | R-precision Top-1 / 2 / 3 |
| `matching` | multimodal distance |
| `fid` | FID |
| `diversity` | diversity |
| `foot_skate` | foot-skating ratio |
| `pen` | penetration (×100 in the paper) |
| `c_pct` | contact ratio |
| `ch_prec`, `ch_rec`, `ch_f1` | hand contact precision / recall / F1 |
