#!/usr/bin/env python3
"""Offline weak IsSameObject probe for PICF anchor debug artifacts.

This script intentionally uses only exported debug artifacts. It is a diagnostic
probe, not a training loss. The goal is to decide whether current AQR/MVTrack
evidence already contains a decodable same-object signal, or whether the next
engineering step must add stronger temporal evidence such as real tracklets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch


def _finite_vec(xs: Any, n: int) -> list[float] | None:
    if not isinstance(xs, list) or len(xs) < n:
        return None
    out: list[float] = []
    for x in xs[:n]:
        if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            return None
        out.append(float(x))
    return out


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _sparse(items: Any) -> dict[int, float]:
    out: dict[int, float] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        weight = item.get("weight")
        if isinstance(idx, int) and isinstance(weight, (int, float)) and math.isfinite(float(weight)):
            out[idx] = out.get(idx, 0.0) + max(float(weight), 0.0)
    return out


def _cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a).intersection(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _dense_vec(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            return []
        out.append(float(item))
    return out


def _dense_cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    aa = a[:n]
    bb = b[:n]
    dot = sum(x * y for x, y in zip(aa, bb, strict=True))
    na = math.sqrt(sum(x * x for x in aa))
    nb = math.sqrt(sum(y * y for y in bb))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _topk_jaccard(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa.intersection(sb)) / max(1, len(sa.union(sb)))


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    labeled = [(x, 1) for x in pos] + [(x, 0) for x in neg]
    labeled.sort(key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    idx = 0
    while idx < len(labeled):
        j = idx + 1
        while j < len(labeled) and labeled[j][0] == labeled[idx][0]:
            j += 1
        avg_rank = 0.5 * (rank + rank + (j - idx) - 1)
        rank_sum_pos += avg_rank * sum(label for _score, label in labeled[idx:j])
        rank += j - idx
        idx = j
    n_pos = len(pos)
    n_neg = len(neg)
    return (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _anchor_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    ad = row.get("anchor_debug", {})
    obs = ad.get("observation", {})
    mapg = ad.get("mapg", {})
    xyzs = obs.get("xyz") or []
    pixels = obs.get("pixel") or []
    roles = obs.get("role_ids") or []
    visual_topk = ((mapg.get("visual_priors") or {}).get("topk") or [])
    point_topk = ((mapg.get("point_priors") or {}).get("topk") or [])
    support_signatures = obs.get("support_signature") or []
    binding_signatures = obs.get("binding_signature") or []
    anchors: list[dict[str, Any]] = []
    count = min(len(xyzs), len(pixels), len(roles), len(visual_topk), len(point_topk))
    for idx in range(count):
        xyz = _finite_vec(xyzs[idx], 3)
        pixel = _finite_vec(pixels[idx], 2)
        role = roles[idx]
        if xyz is None or pixel is None or not isinstance(role, int):
            continue
        anchors.append(
            {
                "index": idx,
                "role": role,
                "xyz": xyz,
                "pixel": pixel,
                "visual": _sparse(visual_topk[idx]),
                "point": _sparse(point_topk[idx]),
                "support_signature": _dense_vec(support_signatures[idx]) if idx < len(support_signatures) else [],
                "binding_signature": _dense_vec(binding_signatures[idx]) if idx < len(binding_signatures) else [],
            }
        )
    return anchors


def _overlay_anchor_rows(payload: dict[str, Any], *, source: str) -> list[dict[str, Any]]:
    raw_anchors = payload.get("anchors") or []
    anchors: list[dict[str, Any]] = []
    if not isinstance(raw_anchors, list):
        return anchors
    for item in raw_anchors:
        if not isinstance(item, dict) or str(item.get("source", "")) != source:
            continue
        xyz = _finite_vec(item.get("world_xyz"), 3)
        pixel_value = item.get("pixel_xy")
        if pixel_value is None:
            # Invisible/off-camera anchors are still useful for lifecycle audits,
            # but the weak IsSameObject probe needs comparable image-space
            # proximity labels to avoid marking two invisible anchors as same.
            continue
        pixel = _finite_vec(pixel_value, 2)
        role = item.get("role")
        if xyz is None or pixel is None or not isinstance(role, int):
            continue
        anchors.append(
            {
                "index": int(item.get("index", len(anchors))),
                "role": role,
                "xyz": xyz,
                "pixel": pixel,
                "visual": {},
                "point": {},
                "support_signature": _dense_vec(item.get("support_signature")),
                "binding_signature": _dense_vec(item.get("binding_signature")),
            }
        )
    return anchors


def _load_anchor_debug_frames(path: Path) -> list[tuple[Any, Any, Any, list[dict[str, Any]]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [(row.get("episode"), row.get("step"), row.get("goal"), _anchor_rows(row)) for row in rows]


def _load_overlay_frames(path: Path, *, source: str) -> list[tuple[Any, Any, Any, list[dict[str, Any]]]]:
    files = sorted(path.glob("step_*.json")) if path.is_dir() else [path]
    frames: list[tuple[Any, Any, Any, list[dict[str, Any]]]] = []
    for file_path in files:
        if not file_path.is_file():
            continue
        payload = json.loads(file_path.read_text())
        if not isinstance(payload, dict):
            continue
        frames.append(
            (
                payload.get("segment_id", 0),
                payload.get("step"),
                payload.get("prompt"),
                _overlay_anchor_rows(payload, source=source),
            )
        )
    return frames


def _combined_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    visual = _cosine(a["visual"], b["visual"])
    point = _cosine(a["point"], b["point"])
    binding = _dense_cosine(a.get("binding_signature", []), b.get("binding_signature", []))
    xyz_d = _dist(a["xyz"], b["xyz"])
    pixel_d = _dist(a["pixel"], b["pixel"])
    geom = math.exp(-xyz_d / 0.08) * math.exp(-pixel_d / 30.0)
    if a.get("binding_signature") and b.get("binding_signature"):
        return 0.25 * visual + 0.25 * point + 0.25 * binding + 0.25 * geom
    return 0.35 * visual + 0.35 * point + 0.30 * geom


class _DiagonalQuadraticProbe(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(dim))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x * y * self.weight).sum(dim=-1) + self.bias


class _LowRankQuadraticProbe(torch.nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        rank = max(int(rank), 1)
        self.left = torch.nn.Linear(dim, rank, bias=False)
        self.right = torch.nn.Linear(dim, rank, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.scale = 1.0 / math.sqrt(float(rank))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = (self.left(x) * self.right(y)).sum(dim=-1)
        yx = (self.left(y) * self.right(x)).sum(dim=-1)
        return (0.5 * (xy + yx) * self.scale) + self.bias


class _FullQuadraticProbe(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(dim, dim))
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.scale = 1.0 / math.sqrt(float(dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = 0.5 * (self.weight + self.weight.T)
        return ((x @ weight) * y).sum(dim=-1) * self.scale + self.bias


def _normalize_dense(vectors: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vectors, p=2.0, dim=-1, eps=1e-8)


def _split_balanced_indices(labels: torch.Tensor, *, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    pos = [int(i) for i in torch.nonzero(labels > 0.5, as_tuple=False).reshape(-1).tolist()]
    neg = [int(i) for i in torch.nonzero(labels <= 0.5, as_tuple=False).reshape(-1).tolist()]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    train_pos = max(1, int(round(len(pos) * train_fraction))) if len(pos) > 1 else len(pos)
    train_neg = max(1, int(round(len(neg) * train_fraction))) if len(neg) > 1 else len(neg)
    train = pos[:train_pos] + neg[:train_neg]
    test = pos[train_pos:] + neg[train_neg:]
    if not test:
        test = train
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _train_quadratic_probe(
    pairs: list[dict[str, Any]],
    *,
    feature: str,
    mode: str,
    rank: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    train_fraction: float,
    max_pairs: int,
    seed: int,
    max_full_dim: int,
) -> dict[str, Any]:
    usable: list[tuple[list[float], list[float], int]] = []
    for pair in pairs:
        left = pair.get(f"{feature}_a")
        right = pair.get(f"{feature}_b")
        label = pair.get("label")
        if not isinstance(left, list) or not isinstance(right, list) or label not in (0, 1):
            continue
        dim = min(len(left), len(right))
        if dim <= 0:
            continue
        usable.append((left[:dim], right[:dim], int(label)))
    pos_count = sum(label for _left, _right, label in usable)
    neg_count = len(usable) - pos_count
    if pos_count == 0 or neg_count == 0:
        return {"status": "insufficient_classes", "examples": len(usable), "positive": pos_count, "negative": neg_count}

    rng = random.Random(seed)
    pos = [item for item in usable if item[2] == 1]
    neg = [item for item in usable if item[2] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    per_class = min(len(pos), len(neg), max(1, int(max_pairs) // 2))
    balanced = pos[:per_class] + neg[:per_class]
    rng.shuffle(balanced)
    dim = min(len(left) for left, _right, _label in balanced)
    if dim <= 0:
        return {"status": "empty_feature_dim", "examples": len(balanced)}
    x = torch.tensor([left[:dim] for left, _right, _label in balanced], dtype=torch.float32)
    y = torch.tensor([right[:dim] for _left, right, _label in balanced], dtype=torch.float32)
    labels = torch.tensor([label for _left, _right, label in balanced], dtype=torch.float32)
    x = _normalize_dense(x)
    y = _normalize_dense(y)

    if mode == "diag_quadratic":
        model: torch.nn.Module = _DiagonalQuadraticProbe(dim)
    elif mode == "low_rank_quadratic":
        model = _LowRankQuadraticProbe(dim, rank=rank)
    elif mode == "full_quadratic":
        if dim > int(max_full_dim):
            return {"status": "skipped_full_dim_too_large", "examples": len(balanced), "dim": dim, "max_full_dim": int(max_full_dim)}
        model = _FullQuadraticProbe(dim)
    else:
        return {"status": "unknown_mode", "mode": str(mode)}

    train_idx, test_idx = _split_balanced_indices(labels, train_fraction=float(train_fraction), seed=int(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = torch.nn.BCEWithLogitsLoss()
    for _epoch in range(max(int(epochs), 0)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[train_idx], y[train_idx])
        loss = criterion(logits, labels[train_idx])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = model(x[test_idx], y[test_idx]).detach().cpu().tolist()
        test_labels = labels[test_idx].detach().cpu().tolist()
        train_loss = float(criterion(model(x[train_idx], y[train_idx]), labels[train_idx]).detach().cpu().item())
    pos_scores = [float(score) for score, label in zip(test_logits, test_labels, strict=True) if label > 0.5]
    neg_scores = [float(score) for score, label in zip(test_logits, test_labels, strict=True) if label <= 0.5]
    auc = _auc(pos_scores, neg_scores)
    accuracy = None
    if pos_scores and neg_scores:
        correct = sum(1 for score in pos_scores if score > 0.0) + sum(1 for score in neg_scores if score <= 0.0)
        accuracy = correct / float(len(pos_scores) + len(neg_scores))
    return {
        "status": "ok",
        "mode": str(mode),
        "feature": str(feature),
        "dim": int(dim),
        "rank": int(rank) if mode == "low_rank_quadratic" else None,
        "examples_balanced": int(len(balanced)),
        "examples_train": int(len(train_idx)),
        "examples_test": int(len(test_idx)),
        "positive_total": int(pos_count),
        "negative_total": int(neg_count),
        "positive_used": int(per_class),
        "negative_used": int(per_class),
        "train_loss": train_loss,
        "auc": auc,
        "accuracy_at_zero": accuracy,
        "pos_mean_logit": _mean_or_none(pos_scores),
        "neg_mean_logit": _mean_or_none(neg_scores),
    }


def run_probe_from_frames(
    frames: list[tuple[Any, Any, Any, list[dict[str, Any]]]],
    *,
    artifact: str,
    source_format: str,
    pos_xyz_m: float,
    neg_xyz_m: float,
    pos_px: float,
    neg_px: float,
    quadratic_probe_modes: tuple[str, ...] = (),
    quadratic_probe_feature: str = "binding_signature",
    quadratic_probe_rank: int = 16,
    quadratic_probe_epochs: int = 200,
    quadratic_probe_lr: float = 0.03,
    quadratic_probe_weight_decay: float = 0.001,
    quadratic_probe_train_fraction: float = 0.7,
    quadratic_probe_max_pairs: int = 20000,
    quadratic_probe_seed: int = 0,
    quadratic_probe_max_full_dim: int = 256,
) -> dict[str, Any]:
    scores: dict[str, dict[str, list[float]]] = {
        "visual_cos": {"pos": [], "neg": []},
        "point_cos": {"pos": [], "neg": []},
        "support_mean": {"pos": [], "neg": []},
        "support_signature_cos": {"pos": [], "neg": []},
        "binding_signature_cos": {"pos": [], "neg": []},
        "geometry": {"pos": [], "neg": []},
        "combined": {"pos": [], "neg": []},
    }
    examples = {"positive": 0, "negative": 0, "ambiguous": 0}
    probe_pairs: list[dict[str, Any]] = []

    for (ep_a, _step_a, _goal_a, anchors_a), (ep_b, _step_b, _goal_b, anchors_b) in zip(frames, frames[1:]):
        if ep_a != ep_b:
            continue
        for a in anchors_a:
            for b in anchors_b:
                if a["role"] != b["role"]:
                    continue
                xyz_d = _dist(a["xyz"], b["xyz"])
                pixel_d = _dist(a["pixel"], b["pixel"])
                is_pos = xyz_d <= pos_xyz_m and pixel_d <= pos_px
                is_neg = xyz_d >= neg_xyz_m or pixel_d >= neg_px
                if not is_pos and not is_neg:
                    examples["ambiguous"] += 1
                    continue
                bucket = "pos" if is_pos else "neg"
                examples["positive" if is_pos else "negative"] += 1
                visual = _cosine(a["visual"], b["visual"])
                point = _cosine(a["point"], b["point"])
                support_sig = _dense_cosine(a.get("support_signature", []), b.get("support_signature", []))
                binding_sig = _dense_cosine(a.get("binding_signature", []), b.get("binding_signature", []))
                geom = math.exp(-xyz_d / 0.08) * math.exp(-pixel_d / 30.0)
                scores["visual_cos"][bucket].append(visual)
                scores["point_cos"][bucket].append(point)
                scores["support_mean"][bucket].append(0.5 * (visual + point))
                if a.get("support_signature") and b.get("support_signature"):
                    scores["support_signature_cos"][bucket].append(support_sig)
                if a.get("binding_signature") and b.get("binding_signature"):
                    scores["binding_signature_cos"][bucket].append(binding_sig)
                scores["geometry"][bucket].append(geom)
                scores["combined"][bucket].append(_combined_score(a, b))
                probe_pairs.append(
                    {
                        "label": 1 if is_pos else 0,
                        "support_signature_a": a.get("support_signature", []),
                        "support_signature_b": b.get("support_signature", []),
                        "binding_signature_a": a.get("binding_signature", []),
                        "binding_signature_b": b.get("binding_signature", []),
                    }
                )

    duplicate_pairs = 0
    same_role_pairs = 0
    duplicate_visual_cos: list[float] = []
    duplicate_point_cos: list[float] = []
    duplicate_binding_signature_cos: list[float] = []
    for _ep, _step, _goal, anchors in frames:
        for i, a in enumerate(anchors):
            for b in anchors[i + 1 :]:
                if a["role"] != b["role"]:
                    continue
                same_role_pairs += 1
                xyz_d = _dist(a["xyz"], b["xyz"])
                pixel_d = _dist(a["pixel"], b["pixel"])
                if xyz_d <= pos_xyz_m and pixel_d <= pos_px:
                    duplicate_pairs += 1
                    duplicate_visual_cos.append(_cosine(a["visual"], b["visual"]))
                    duplicate_point_cos.append(_cosine(a["point"], b["point"]))
                    if a.get("binding_signature") and b.get("binding_signature"):
                        duplicate_binding_signature_cos.append(
                            _dense_cosine(a.get("binding_signature", []), b.get("binding_signature", []))
                        )

    metrics: dict[str, Any] = {
        "path": str(artifact),
        "source_format": str(source_format),
        "frames": len(frames),
        "pair_examples": examples,
        "same_role_pairs_within_frame": same_role_pairs,
        "duplicate_candidate_pairs_within_frame": duplicate_pairs,
        "duplicate_candidate_fraction_within_frame": duplicate_pairs / same_role_pairs if same_role_pairs else None,
        "thresholds": {
            "pos_xyz_m": pos_xyz_m,
            "neg_xyz_m": neg_xyz_m,
            "pos_px": pos_px,
            "neg_px": neg_px,
        },
    }
    for name, buckets in scores.items():
        metrics[f"{name}_auc"] = _auc(buckets["pos"], buckets["neg"])
        metrics[f"{name}_pos_mean"] = sum(buckets["pos"]) / len(buckets["pos"]) if buckets["pos"] else None
        metrics[f"{name}_neg_mean"] = sum(buckets["neg"]) / len(buckets["neg"]) if buckets["neg"] else None
    metrics["duplicate_visual_cos_mean"] = (
        sum(duplicate_visual_cos) / len(duplicate_visual_cos) if duplicate_visual_cos else None
    )
    metrics["duplicate_point_cos_mean"] = (
        sum(duplicate_point_cos) / len(duplicate_point_cos) if duplicate_point_cos else None
    )
    metrics["duplicate_binding_signature_cos_mean"] = (
        sum(duplicate_binding_signature_cos) / len(duplicate_binding_signature_cos)
        if duplicate_binding_signature_cos
        else None
    )

    combined_auc = metrics.get("combined_auc")
    binding_auc = metrics.get("binding_signature_cos_auc")
    dup_frac = metrics.get("duplicate_candidate_fraction_within_frame")
    if isinstance(binding_auc, float) and binding_auc >= 0.70 and isinstance(dup_frac, float) and dup_frac >= 0.20:
        decision = "binding_subspace_decodable_but_assignment_duplicates_candidates"
    elif isinstance(combined_auc, float) and combined_auc >= 0.75 and isinstance(dup_frac, float) and dup_frac >= 0.20:
        decision = "same_object_signal_decodable_but_assignment_duplicates_candidates"
    elif isinstance(binding_auc, float) and binding_auc >= 0.70:
        decision = "binding_subspace_decodable"
    elif isinstance(combined_auc, float) and combined_auc >= 0.75:
        decision = "same_object_signal_decodable"
    else:
        decision = "same_object_signal_weak_or_debug_features_insufficient"
    metrics["decision"] = decision
    if quadratic_probe_modes:
        probe_metrics: dict[str, Any] = {}
        for mode in quadratic_probe_modes:
            if mode == "none":
                continue
            mode_metrics = _train_quadratic_probe(
                probe_pairs,
                feature=str(quadratic_probe_feature),
                mode=str(mode),
                rank=int(quadratic_probe_rank),
                epochs=int(quadratic_probe_epochs),
                lr=float(quadratic_probe_lr),
                weight_decay=float(quadratic_probe_weight_decay),
                train_fraction=float(quadratic_probe_train_fraction),
                max_pairs=int(quadratic_probe_max_pairs),
                seed=int(quadratic_probe_seed),
                max_full_dim=int(quadratic_probe_max_full_dim),
            )
            probe_metrics[str(mode)] = mode_metrics
            if mode_metrics.get("status") == "ok":
                prefix = f"{quadratic_probe_feature}_{mode}"
                metrics[f"{prefix}_trained_auc"] = mode_metrics.get("auc")
                metrics[f"{prefix}_trained_accuracy_at_zero"] = mode_metrics.get("accuracy_at_zero")
                metrics[f"{prefix}_trained_train_loss"] = mode_metrics.get("train_loss")
        metrics["trained_quadratic_probes"] = probe_metrics
    return metrics


def run_probe(
    path: Path,
    *,
    pos_xyz_m: float,
    neg_xyz_m: float,
    pos_px: float,
    neg_px: float,
    quadratic_probe_modes: tuple[str, ...] = (),
    quadratic_probe_feature: str = "binding_signature",
    quadratic_probe_rank: int = 16,
    quadratic_probe_epochs: int = 200,
    quadratic_probe_lr: float = 0.03,
    quadratic_probe_weight_decay: float = 0.001,
    quadratic_probe_train_fraction: float = 0.7,
    quadratic_probe_max_pairs: int = 20000,
    quadratic_probe_seed: int = 0,
    quadratic_probe_max_full_dim: int = 256,
) -> dict[str, Any]:
    return run_probe_from_frames(
        _load_anchor_debug_frames(path),
        artifact=str(path),
        source_format="anchor_debug_jsonl",
        pos_xyz_m=pos_xyz_m,
        neg_xyz_m=neg_xyz_m,
        pos_px=pos_px,
        neg_px=neg_px,
        quadratic_probe_modes=quadratic_probe_modes,
        quadratic_probe_feature=quadratic_probe_feature,
        quadratic_probe_rank=quadratic_probe_rank,
        quadratic_probe_epochs=quadratic_probe_epochs,
        quadratic_probe_lr=quadratic_probe_lr,
        quadratic_probe_weight_decay=quadratic_probe_weight_decay,
        quadratic_probe_train_fraction=quadratic_probe_train_fraction,
        quadratic_probe_max_pairs=quadratic_probe_max_pairs,
        quadratic_probe_seed=quadratic_probe_seed,
        quadratic_probe_max_full_dim=quadratic_probe_max_full_dim,
    )


def run_overlay_probe(
    path: Path,
    *,
    source: str,
    pos_xyz_m: float,
    neg_xyz_m: float,
    pos_px: float,
    neg_px: float,
    quadratic_probe_modes: tuple[str, ...] = (),
    quadratic_probe_feature: str = "binding_signature",
    quadratic_probe_rank: int = 16,
    quadratic_probe_epochs: int = 200,
    quadratic_probe_lr: float = 0.03,
    quadratic_probe_weight_decay: float = 0.001,
    quadratic_probe_train_fraction: float = 0.7,
    quadratic_probe_max_pairs: int = 20000,
    quadratic_probe_seed: int = 0,
    quadratic_probe_max_full_dim: int = 256,
) -> dict[str, Any]:
    return run_probe_from_frames(
        _load_overlay_frames(path, source=source),
        artifact=str(path),
        source_format=f"anchor_overlay_json:{source}",
        pos_xyz_m=pos_xyz_m,
        neg_xyz_m=neg_xyz_m,
        pos_px=pos_px,
        neg_px=neg_px,
        quadratic_probe_modes=quadratic_probe_modes,
        quadratic_probe_feature=quadratic_probe_feature,
        quadratic_probe_rank=quadratic_probe_rank,
        quadratic_probe_epochs=quadratic_probe_epochs,
        quadratic_probe_lr=quadratic_probe_lr,
        quadratic_probe_weight_decay=quadratic_probe_weight_decay,
        quadratic_probe_train_fraction=quadratic_probe_train_fraction,
        quadratic_probe_max_pairs=quadratic_probe_max_pairs,
        quadratic_probe_seed=quadratic_probe_seed,
        quadratic_probe_max_full_dim=quadratic_probe_max_full_dim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--anchor-debug", type=Path, help="Evaluation anchor_debug.jsonl artifact.")
    source.add_argument("--anchor-overlays", type=Path, help="Training anchor_overlays directory or one step_*.json file.")
    parser.add_argument("--overlay-source", default="posterior", choices=["posterior", "graph"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pos-xyz-m", type=float, default=0.04)
    parser.add_argument("--neg-xyz-m", type=float, default=0.12)
    parser.add_argument("--pos-px", type=float, default=12.0)
    parser.add_argument("--neg-px", type=float, default=35.0)
    parser.add_argument(
        "--quadratic-probe",
        default="none",
        help=(
            "Comma-separated trained probe modes: none, diag_quadratic, low_rank_quadratic, full_quadratic, all. "
            "These reimplement the IsSameObject pairwise/quadratic protocol natively for exported PICF artifacts."
        ),
    )
    parser.add_argument("--quadratic-probe-feature", default="binding_signature", choices=["binding_signature", "support_signature"])
    parser.add_argument("--quadratic-probe-rank", type=int, default=16)
    parser.add_argument("--quadratic-probe-epochs", type=int, default=200)
    parser.add_argument("--quadratic-probe-lr", type=float, default=0.03)
    parser.add_argument("--quadratic-probe-weight-decay", type=float, default=0.001)
    parser.add_argument("--quadratic-probe-train-fraction", type=float, default=0.7)
    parser.add_argument("--quadratic-probe-max-pairs", type=int, default=20000)
    parser.add_argument("--quadratic-probe-seed", type=int, default=0)
    parser.add_argument("--quadratic-probe-max-full-dim", type=int, default=256)
    args = parser.parse_args()

    modes = tuple(part.strip() for part in str(args.quadratic_probe).split(",") if part.strip())
    if "all" in modes:
        modes = ("diag_quadratic", "low_rank_quadratic", "full_quadratic")

    if args.anchor_debug is not None:
        metrics = run_probe(
            args.anchor_debug,
            pos_xyz_m=args.pos_xyz_m,
            neg_xyz_m=args.neg_xyz_m,
            pos_px=args.pos_px,
            neg_px=args.neg_px,
            quadratic_probe_modes=modes,
            quadratic_probe_feature=str(args.quadratic_probe_feature),
            quadratic_probe_rank=int(args.quadratic_probe_rank),
            quadratic_probe_epochs=int(args.quadratic_probe_epochs),
            quadratic_probe_lr=float(args.quadratic_probe_lr),
            quadratic_probe_weight_decay=float(args.quadratic_probe_weight_decay),
            quadratic_probe_train_fraction=float(args.quadratic_probe_train_fraction),
            quadratic_probe_max_pairs=int(args.quadratic_probe_max_pairs),
            quadratic_probe_seed=int(args.quadratic_probe_seed),
            quadratic_probe_max_full_dim=int(args.quadratic_probe_max_full_dim),
        )
    else:
        metrics = run_overlay_probe(
            args.anchor_overlays,
            source=str(args.overlay_source),
            pos_xyz_m=args.pos_xyz_m,
            neg_xyz_m=args.neg_xyz_m,
            pos_px=args.pos_px,
            neg_px=args.neg_px,
            quadratic_probe_modes=modes,
            quadratic_probe_feature=str(args.quadratic_probe_feature),
            quadratic_probe_rank=int(args.quadratic_probe_rank),
            quadratic_probe_epochs=int(args.quadratic_probe_epochs),
            quadratic_probe_lr=float(args.quadratic_probe_lr),
            quadratic_probe_weight_decay=float(args.quadratic_probe_weight_decay),
            quadratic_probe_train_fraction=float(args.quadratic_probe_train_fraction),
            quadratic_probe_max_pairs=int(args.quadratic_probe_max_pairs),
            quadratic_probe_seed=int(args.quadratic_probe_seed),
            quadratic_probe_max_full_dim=int(args.quadratic_probe_max_full_dim),
        )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
