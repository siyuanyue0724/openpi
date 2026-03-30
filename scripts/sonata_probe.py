"""
One-off probe for OpenPI Sonata integration (PyTorch).

Why this lives in scripts/:
  - Keeps debug instrumentation out of core library code.
  - Easy to delete cleanly after validation.

This probe can:
  1) Inspect observation.point_clouds['pointcloud'] and masks
  2) Inspect observation.tokenized_prompt
  3) Verify "point window tokens" (start/end) exist EXACTLY ONCE per sample
     - If model doesn't expose point_start_id/point_end_id, it will infer them from tokenized_prompt.
  4) Optionally run Sonata encode once (run_encode=True) and report token_len distribution.

Delete cleanly after validation:
  rm scripts/sonata_probe.py
"""

from __future__ import annotations

from typing import Any
import logging

import torch


def _infer_point_window_ids_from_prompt(
    tok_long: torch.Tensor,
    *,
    topk: int = 128,
) -> tuple[int, int] | None:
    """
    Infer (start_id, end_id) from tokenized_prompt when model doesn't expose them.

    Heuristics (robust + conservative):
      - Special point window tokens are typically very large IDs (often vocab_size-2 / vocab_size-1).
      - They should appear exactly once per sample.
      - Their positions should satisfy start_pos < end_pos, usually adjacent (gap=1).

    We search among the top-k largest UNIQUE token IDs in the prompt.
    First, we prioritize pairs (sid, sid+1) to match the common "last two IDs" contract.
    If not found, we fall back to a general pair search among candidates.
    """
    if tok_long.ndim != 2:
        return None
    # unique token IDs
    uniq = torch.unique(tok_long)
    if uniq.numel() < 2:
        return None

    # top-k largest ids
    uniq_sorted = uniq.sort(descending=True).values
    cand = uniq_sorted[: min(int(topk), int(uniq_sorted.numel()))]
    cand_list = [int(x) for x in cand.tolist()]
    cand_set = set(cand_list)

    # Precompute per-candidate (count[B], pos[B])
    stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for cid in cand_list:
        m = (tok_long == int(cid))
        cnt = m.sum(dim=1)  # [B]
        pos = m.to(torch.int64).argmax(dim=1)  # [B] (meaningful only when cnt>0)
        stats[int(cid)] = (cnt, pos)

    def _score_pair(sid: int, eid: int) -> tuple[float, float, int, int] | None:
        s_cnt, s_pos = stats[sid]
        e_cnt, e_pos = stats[eid]
        ok = (s_cnt == 1) & (e_cnt == 1) & (s_pos < e_pos)
        ok_frac = float(ok.to(torch.float32).mean().item())
        if ok_frac <= 0.0:
            return None
        gap = (e_pos - s_pos).to(torch.int64)
        adjacency_frac = float((gap == 1).to(torch.float32).mean().item())
        # median gap over ok samples (smaller is better, typically 1)
        gap_ok = gap[ok]
        gap_med = int(gap_ok.median().item()) if gap_ok.numel() > 0 else 1_000_000
        # return score: higher ok_frac, higher adjacency, smaller gap, larger ids
        return (ok_frac, adjacency_frac, -gap_med, sid + eid)

    best: tuple[int, int] | None = None
    best_score: tuple[float, float, int, int] | None = None

    # Pass 1: prefer consecutive IDs (sid, sid+1)
    for sid in cand_list:
        eid = sid + 1
        if eid not in cand_set:
            continue
        sc = _score_pair(sid, eid)
        if sc is None:
            continue
        if best_score is None or sc > best_score:
            best_score = sc
            best = (sid, eid)

    # Pass 2: fallback to all pairs among candidates
    if best is None:
        for sid in cand_list:
            if sid == 0:
                continue
            for eid in cand_list:
                if eid == sid or eid == 0:
                    continue
                sc = _score_pair(sid, eid)
                if sc is None:
                    continue
                if best_score is None or sc > best_score:
                    best_score = sc
                    best = (sid, eid)

    return best


@torch.no_grad()
def probe_sonata_integration(
    model,
    observation,
    *,
    run_encode: bool = False,
    max_print: int = 8,
) -> dict[str, Any]:
    """
    Debug probe for Sonata integration.

    Probe policy:
      - Default: DOES NOT run encoder (fast).
      - If run_encode=True: runs encoder once and restores model._pt_cache and sonata train/eval afterwards.
    """
    logger = logging.getLogger("openpi")

    info: dict[str, Any] = {
        "enable_sonata": bool(getattr(model, "enable_sonata", False)),
        "sonata_mode": str(getattr(model, "sonata_mode", "")),
        "require_cuda": bool(getattr(model, "require_cuda", True)),
        "sonata_ckpt": getattr(model, "sonata_ckpt", None),
        "sonata_projector_ckpt": getattr(model, "sonata_projector_ckpt", None),
        "sonata_in_channels": int(getattr(model, "sonata_in_channels", getattr(model, "point_feat_dim", -1))),
        "point_feat_dim_config": int(getattr(model, "point_feat_dim", -1)),
        "point_token_cap": int(getattr(model, "point_token_cap", -1)),
        "sonata_auto_pad_feat": bool(getattr(model, "sonata_auto_pad_feat", False)),
        "sonata_validate": bool(getattr(model, "sonata_validate", False)),
        "point_start_id": None if getattr(model, "point_start_id", None) is None else int(model.point_start_id),
        "point_end_id": None if getattr(model, "point_end_id", None) is None else int(model.point_end_id),
        "pc_projector_built": getattr(model, "pc_projector", None) is not None,
        "pc_projector_loaded": bool(getattr(model, "_pc_projector_loaded", False)),
    }

    # ---- Pointcloud tensor ----
    pcs = None
    if hasattr(observation, "point_clouds"):
        try:
            pcs = observation.point_clouds.get("pointcloud", None)
        except Exception:
            pcs = None

    if pcs is None:
        info["pcs_shape"] = None
    else:
        if not isinstance(pcs, torch.Tensor):
            pcs = torch.as_tensor(pcs)
        info["pcs_shape"] = tuple(pcs.shape)
        info["pcs_dtype"] = str(pcs.dtype)
        info["pcs_device"] = str(pcs.device)
        if pcs.ndim == 3 and pcs.shape[-1] >= 3:
            info["obs_fd"] = int(pcs.shape[-1] - 3)
            info["exp_fd"] = int(info["sonata_in_channels"])
            info["feat_dim_match"] = bool(info["obs_fd"] == info["exp_fd"])
        else:
            info["obs_fd"] = None
            info["exp_fd"] = int(info["sonata_in_channels"])
            info["feat_dim_match"] = False

    # ---- Pointcloud masks ----
    pmask = None
    if hasattr(observation, "point_cloud_masks"):
        try:
            pmask = observation.point_cloud_masks.get("pointcloud", None)
        except Exception:
            pmask = None

    if pmask is None:
        info["pmask_shape"] = None
    else:
        if not isinstance(pmask, torch.Tensor):
            pmask = torch.as_tensor(pmask)
        info["pmask_shape"] = tuple(pmask.shape)
        info["pmask_dtype"] = str(pmask.dtype)
        if pmask.ndim == 1:
            info["pmask_true"] = int(pmask.to(torch.bool).sum().item())
        elif pmask.ndim == 2:
            info["pmask_true_per_sample"] = pmask.to(torch.bool).sum(dim=1).cpu().tolist()[:max_print]

    # ---- Prompt tokens: window pair ----
    tok = getattr(observation, "tokenized_prompt", None)
    if tok is None and hasattr(model, "_preprocess_observation"):
        # fallback (probe-only)
        try:
            _, _, tok, _, _ = model._preprocess_observation(observation, train=False)
            info["prompt_from_preprocess"] = True
        except Exception as e:
            info["prompt_probe_error"] = repr(e)
            tok = None

    if isinstance(tok, torch.Tensor) and tok.ndim == 2:
        B = tok.shape[0]
        info["prompt_shape"] = tuple(tok.shape)
        tok_long = tok if tok.dtype == torch.long else tok.to(torch.long)

        start_id_used = info.get("point_start_id", None)
        end_id_used = info.get("point_end_id", None)

        # Infer if missing
        if start_id_used is None or end_id_used is None:
            inferred = _infer_point_window_ids_from_prompt(tok_long, topk=128)
            if inferred is not None:
                start_id_used, end_id_used = inferred
                info["point_start_id_inferred"] = int(start_id_used)
                info["point_end_id_inferred"] = int(end_id_used)

        if start_id_used is not None and end_id_used is not None:
            start_count = (tok_long == int(start_id_used)).sum(dim=1)
            end_count = (tok_long == int(end_id_used)).sum(dim=1)
            info["start_count"] = start_count.cpu().tolist()[:max_print]
            info["end_count"] = end_count.cpu().tolist()[:max_print]

            s_any = start_count > 0
            e_any = end_count > 0
            s_pos = torch.where(
                s_any,
                (tok_long == int(start_id_used)).to(torch.int64).argmax(dim=1),
                torch.full((B,), -1, dtype=torch.int64, device=tok.device),
            )
            e_pos = torch.where(
                e_any,
                (tok_long == int(end_id_used)).to(torch.int64).argmax(dim=1),
                torch.full((B,), -1, dtype=torch.int64, device=tok.device),
            )
            info["start_pos"] = s_pos.cpu().tolist()[:max_print]
            info["end_pos"] = e_pos.cpu().tolist()[:max_print]

            ok_pair = (start_count == 1) & (end_count == 1) & (s_pos >= 0) & (e_pos >= 0) & (s_pos < e_pos)
            info["ok_pair_frac"] = float(ok_pair.to(torch.float32).mean().item())

            gap = (e_pos - s_pos)
            info["adjacent_frac"] = float((gap == 1).to(torch.float32).mean().item())
            # median gap over ok samples
            gap_ok = gap[ok_pair]
            if gap_ok.numel() > 0:
                info["gap_median"] = int(gap_ok.to(torch.int64).median().item())

            tok_mask = getattr(observation, "tokenized_prompt_mask", None)
            if tok_mask is not None:
                if not isinstance(tok_mask, torch.Tensor):
                    tok_mask = torch.as_tensor(tok_mask)
                tok_mask = tok_mask.to(device=tok.device)

                if tok_mask.ndim == 2 and tuple(tok_mask.shape) == tuple(tok_long.shape):
                    empty_window_ok = ok_pair.clone()
                    for b in range(B):
                        if bool(ok_pair[b].item()):
                            mid_visible = tok_mask[
                                b,
                                int(s_pos[b].item()) + 1 : int(e_pos[b].item())
                            ].to(torch.bool)
                            if bool(mid_visible.any().item()):
                                empty_window_ok[b] = False
                    info["empty_window_frac"] = float(empty_window_ok.to(torch.float32).mean().item())
                else:
                    info["empty_window_frac"] = None
                    info["prompt_mask_missing_or_shape_mismatch"] = True
            else:
                info["empty_window_frac"] = None
                info["prompt_mask_missing_or_shape_mismatch"] = True
        else:
            info["point_window_ids_missing"] = True
    else:
        info["prompt_shape"] = None
        info["prompt_missing_or_not_tensor"] = True

    # ---- Optional: run encoding once ----
    if run_encode and info["enable_sonata"] and info["sonata_mode"] in ("projector", "all"):
        old_cache = getattr(model, "_pt_cache", None)
        old_sonata_training = None
        if getattr(model, "sonata", None) is not None:
            old_sonata_training = bool(model.sonata.training)

        try:
            model.torch_sonata_encode_batch(observation, train=False)
            if getattr(model, "_pt_cache", None) is not None:
                pt_feat_raw, pt_mask = model._pt_cache
                info["pt_feat_raw_shape"] = tuple(pt_feat_raw.shape)
                info["pt_mask_shape"] = tuple(pt_mask.shape)
                info["pt_tokens_per_sample"] = pt_mask.to(torch.int64).sum(dim=1).cpu().tolist()[:max_print]
        except Exception as e:
            info["encode_error"] = repr(e)
        finally:
            # Restore artifacts
            model._pt_cache = old_cache
            if old_sonata_training is not None and getattr(model, "sonata", None) is not None:
                model.sonata.train(old_sonata_training)

    # ---- Logging ----
    compact = {k: v for k, v in info.items() if k not in (
        "start_count", "end_count", "start_pos", "end_pos",
        "pmask_true_per_sample", "pt_tokens_per_sample",
    )}
    logger.info("[Sonata probe] %s", compact)

    if "start_count" in info:
        logger.info(
            "[Sonata probe] start_count[:%d]=%s end_count[:%d]=%s",
            max_print, info.get("start_count"), max_print, info.get("end_count"),
        )
        logger.info(
            "[Sonata probe] start_pos[:%d]=%s end_pos[:%d]=%s ok_pair_frac=%s adjacent_frac=%s gap_median=%s",
            max_print, info.get("start_pos"), max_print, info.get("end_pos"),
            info.get("ok_pair_frac"), info.get("adjacent_frac"), info.get("gap_median", None),
        )

    if "pt_tokens_per_sample" in info:
        logger.info("[Sonata probe] pt_tokens_per_sample[:%d]=%s", max_print, info.get("pt_tokens_per_sample"))

    return info
