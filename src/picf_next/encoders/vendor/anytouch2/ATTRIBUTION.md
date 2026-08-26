This directory contains a minimal vendored subset derived from the upstream AnyTouch2 repository:
https://github.com/GeWu-Lab/AnyTouch2 at commit
`82c5677d9cf0176d97a1fe04745f63cd02dd6f54`.

Vendored components:
- `tactile_mae.py`
- `util/pos_embed.py`
- `CLIP-B-16/config.json`

These files are adapted only as needed for local packaging, import paths and
runtime assertions. The upstream license is copied into this directory.
