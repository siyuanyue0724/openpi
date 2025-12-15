#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 CALVIN 数据集根目录，列出 task_*、每个 split 以及 episode 数量，
用于确认你应该把 --src 指到哪里。
用法：python scripts/calvin/calvin_discover.py --root ~/datasets/calvin/dataset
"""
import os, re, argparse, json
from pathlib import Path
EP_RE = re.compile(r"^episode_(\d+)\.npz$")

def count_episodes(d: Path) -> int:
    if not d.exists() or not d.is_dir():
        return 0
    n = 0
    for name in os.listdir(d):
        if EP_RE.match(name):
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser(description="发现 CALVIN task/split/episode 概况")
    ap.add_argument("--root", required=True, help="CALVIN 数据集根目录，例如 ~/datasets/calvin/dataset")
    args = ap.parse_args()
    root = Path(os.path.expanduser(args.root)).resolve()
    if not root.exists():
        raise SystemExit(f"[ERR] 不存在：{root}")

    tasks = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("task_"):
            tasks.append(p)

    if not tasks:
        raise SystemExit(f"[ERR] 在 {root} 下没发现 task_* 目录，请确认下载/解压。")

    report = {}
    for tdir in tasks:
        item = {"splits": {}}
        # 候选 split 名称（尽量覆盖不同分布）
        cand_splits = [
            "training","train","training_lang","training_no_lang",
            "validation","val","validation_lang","validation_seen","validation_unseen"
        ]
        # 实际存在的子目录
        found = []
        for d in sorted(tdir.iterdir()):
            if d.is_dir():
                found.append(d.name)
        # 统计每个存在的 split
        for name in found:
            spdir = tdir / name
            n = count_episodes(spdir)
            if n>0:
                has_hydra = (spdir / ".hydra").exists()
                has_ann   = (spdir / "auto_lang_ann.npy").exists()
                item["splits"][name] = {
                    "path": str(spdir),
                    "episodes": n,
                    "has_.hydra": has_hydra,
                    "has_auto_lang_ann": has_ann
                }
        report[tdir.name] = item

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n[HINT] 若要创建子集，--src 应指向具体某个 task_*/（例如 task_D_D）。")
    print("       如果没有 standard 'training/validation'，可以在子集脚本里用 --train-split/--val-split 手动指定。")

if __name__ == "__main__":
    main()
