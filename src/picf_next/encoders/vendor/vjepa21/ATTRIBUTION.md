Vendored from Meta's official V-JEPA2 source and pinned for this repository at
commit `204698b45b3712590f06245fbfba32d3be539812`:

- `app/vjepa_2_1/models/vision_transformer.py`
- `app/vjepa_2_1/models/utils/modules.py`
- `app/vjepa_2_1/models/utils/patch_embed.py`
- `src/utils/tensors.py`
- `src/masks/utils.py`

These files were first exercised in the legacy PICF implementation, then copied
into `picf_next.encoders.vendor.vjepa21` and minimally patched to:

- resolve imports inside the PICF Next package
- disable repo-wide Ruff linting on vendor sources
- keep the encoder-only inference path local, without hub download logic
