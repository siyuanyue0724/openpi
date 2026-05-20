# Paper-Code Snapshot Manifest

Date: 2026-05-20

This manifest records the local paper-code snapshots used by
`docs/PICF_AQR_OWM_LATEST_SLOT_FINAL_AUDIT_20260520_TEMP.md` and
`scripts/picf_latest_slot_deployment_audit.py`.

The full external repositories are intentionally not vendored into the main
OpenPI repository.  They are local audit inputs, not runtime dependencies.  A
fresh checkout can reproduce the audit by cloning these references into the
listed paths, or by using this manifest as the provenance record for the
documented design comparison.

| Local Path | Remote | Commit |
| --- | --- | --- |
| `temp/paper_code_20260518/MetaSlot` | `https://github.com/lhj-lhj/MetaSlot.git` | `ba5d214c7fc619650cfd14ca9d70902defb836d3` |
| `temp/paper_code_20260518/AdaSlot` | `https://github.com/amazon-science/AdaSlot.git` | `6a60387f4ee985e55b254274f41974e1aa5130e8` |
| `temp/paper_code_20260518/object-centric-learning-framework` | `https://github.com/amazon-science/object-centric-learning-framework.git` | `0a97292cb0dbb173777d8137ac0032957b4f0a6c` |
| `temp/paper_code_20260518/slot-attention-video` | `https://github.com/google-research/slot-attention-video.git` | `ba8f15ee19472c6f9425c9647daf87910f17b605` |
| `temp/external_repos/SlotLifter` | `https://github.com/YuLiu-LY/SlotLifter.git` | `0ea13606101430e6371608284c38f897934b961e` |
| `temp/external_repos/slot_refs_20260520/vit-object-binding` | `https://github.com/liyihao0302/vit-object-binding.git` | `014c66b45ea262f9b6eec83ff388a1e1c10dfcaa` |

Design-use rule:

```text
External code snapshots inform architecture audits only.
They are not imported as runtime dependencies unless a later implementation
document explicitly rewrites the module boundary and license handling.
```
