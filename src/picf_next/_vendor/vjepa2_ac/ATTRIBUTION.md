# V-JEPA 2 source transplant

This directory is a complete copy of the upstream `src/` tree from
`facebookresearch/vjepa2` at commit
`204698b45b3712590f06245fbfba32d3be539812`.

The only edits are mechanical Python import-root rewrites from `src.*` to
`picf_next._vendor.vjepa2_ac.*`, required to avoid collision with this
project's own source root. No model equation, tensor operation, default,
attention mask, RoPE behavior or compatibility quirk was changed. The original
MIT license is included as `LICENSE`; the complete immutable upstream archive
is `references/source_archives/vjepa2-204698b45b3712590f06245fbfba32d3be539812.tar.gz`.
