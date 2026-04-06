Vendored from `vjepa2-main` in `TEMP_REPO/vjepa2-main (1).zip`:

- `app/vjepa_2_1/models/vision_transformer.py`
- `app/vjepa_2_1/models/utils/modules.py`
- `app/vjepa_2_1/models/utils/patch_embed.py`
- `src/utils/tensors.py`
- `src/masks/utils.py`

These files were copied into `openpi.picf.vjepa.vendor` and minimally patched to:

- resolve imports inside the OpenPI package
- disable repo-wide Ruff linting on vendor sources
- keep the encoder-only inference path local, without hub download logic
