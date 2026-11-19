# extras/ — visualisation

Nothing here is needed to train, generate, refine or evaluate.

`render_aitviewer/render_hoi_aitv.py` renders ground-truth sequences and exported results of `refine.py`
(`refined_<sequence>.npz`) side by side into an mp4 video with [aitviewer](https://github.com/eth-ait/aitviewer), headless
(`PYOPENGL_PLATFORM=egl` on a machine without a display).
It needs `aitviewer` and `ffmpeg` in addition to the model environment and reads the same environment
variables as `refine.py`. See the docstring of the script for the options.

```bash
python extras/render_aitviewer/render_hoi_aitv.py --seq sub17_woodchair_018 --gt \
    --panels outputs/refined_rep3 --labels PAMI --out_dir outputs/videos
```
