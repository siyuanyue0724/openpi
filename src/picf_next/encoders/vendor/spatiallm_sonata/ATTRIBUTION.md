This directory contains the SpatialLM-adapted Point Transformer V3/Sonata
encoder and serialization primitives exercised by the pinned production
checkpoint.

- SpatialLM source: https://github.com/manycore-research/SpatialLM at commit
  `8913c44d84a450c53e9340b13317f8cf7144a738`
- Sonata architecture source: https://github.com/facebookresearch/sonata at
  commit `18c09ff8d713494f78a8213792262b910977a65d`
- Legacy parity source commit:
  `afddce6b0369e4c294cc428dbeceded581198a8f`

Local changes are limited to package paths, explicit tensor/index runtime
assertions, and correction of the invalid-dimension Hilbert decoder error
message's missing format argument. The Apache-2.0 license is copied into this
directory.
