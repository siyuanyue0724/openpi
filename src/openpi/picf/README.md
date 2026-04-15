# PICF README

The current canonical PICF handoff and deployment document is:

- [`README_semantic_prefix_primary_2048_refactor.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_semantic_prefix_primary_2048_refactor.md)

The current CALVIN validation and rollout document is:

- [`CALVIN_VALIDATION_README.md`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md)

The current formal contract is:

- [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)

## Current One-Paragraph Summary

PICF now uses a language-free physical core together with a full
semantic-prefix-primary control and conditioned-future path. The physical
posterior, physical predictive cache, and next-step innovation remain
language-free. The full PaliGemma token sequence now enters the control and
conditioned-future trunks at the native `2048` width, while `posterior.tokens`
and `posterior.global_post` are appended as structured world-state context.

Do not use the obsolete sidecar-primary notes or the earlier 384-width
semantic-prefix notes as the current spec.
