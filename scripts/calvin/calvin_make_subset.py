#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从某个 CALVIN task 目录（如 .../dataset/task_D_D）里，按 episode id 生成【最小子集】目录。
优先符号链接，失败回退硬链接，再失败复制；并把 validation/.hydra 与 auto_lang_ann.npy 一并就位。
能自动识别 split 名称；如识别不到，可通过 --train-split/--val-split 指定。

示例（D→D）：
  python scripts/calvin/calvin_make_subset.py \
    --src ~/datasets/calvin/dataset/task_D_D \
    --out ~/datasets/calvin/workspace/task_D_D_subset \
    --train "0-999" \
    --val "0-199"
"""
import os, re, json, shutil, argparse
from pathlib import Path
from typing import List, Tuple, Optional

EP_RE = re.compile(r"^episode_(\d+)\.npz$")

def parse_id_selector(sel: str, available: List[int]) -> List[int]:
    """解析 '0-99,200,300-320' 或一个包含 id 的文本文件路径。返回排序后的有效 id 列表。"""
    avail = set(available)
    if sel is None:
        return sorted(avail)
    p = Path(sel)
    ids: List[int] = []
    if p.exists() and p.is_file():
        for line in p.read_text().splitlines():
            line=line.strip()
            if line:
                ids.append(int(line))
        return [i for i in ids if i in avail]
    for tok in sel.split(","):
        tok = tok.strip()
        if not tok: 
            continue
        if "-" in tok:
            a,b = tok.split("-"); a,b = int(a), int(b)
            for i in range(a, b+1):
                if i in avail: ids.append(i)
        else:
            i = int(tok)
            if i in avail: ids.append(i)
    return sorted(set(ids))

def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
        return
    except Exception:
        pass
    try:
        os.link(src, dst)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)

def collect_episode_ids(dirpath: Path) -> List[int]:
    ids: List[int] = []
    if not dirpath.exists() or not dirpath.is_dir():
        return ids
    for name in os.listdir(dirpath):
        m = EP_RE.match(name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)

def pick_split_dir(task_root: Path, prefer_names: List[str]) -> Optional[Path]:
    """按优先级返回第一个存在且包含至少一个 episode 的 split 目录；找不到返回 None。"""
    for name in prefer_names:
        d = task_root / name
        if d.exists() and d.is_dir():
            if collect_episode_ids(d):
                return d
    return None

def copy_if_exists(src: Path, dst: Path):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    elif src.is_dir():
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)

def make_subset(src_root: Path, out_root: Path,
                train_sel: str, val_sel: str,
                train_split: Optional[str]=None,
                val_split: Optional[str]=None) -> Tuple[int,int,Path,dict]:
    # 自动选择 split 目录
    if train_split:
        train_src = src_root / train_split
        if not train_src.exists():
            raise SystemExit(f"[ERR] 指定的 --train-split 不存在：{train_src}")
    else:
        train_src = pick_split_dir(src_root, ["training","train","training_lang","training_no_lang"])
    if val_split:
        val_src = src_root / val_split
        if not val_src.exists():
            raise SystemExit(f"[ERR] 指定的 --val-split 不存在：{val_src}")
    else:
        val_src = pick_split_dir(src_root, ["validation","val","validation_lang","validation_seen","validation_unseen"])

    if train_src is None or val_src is None:
        # 打印可用目录帮助定位
        found = [d.name for d in src_root.iterdir() if d.is_dir()]
        raise SystemExit(f"[ERR] 未找到可用的 training/validation 目录。\n"
                         f" task根：{src_root}\n 发现的子目录：{found}\n"
                         f" 你也可以手动指定 --train-split/--val-split。")

    train_ids_all = collect_episode_ids(train_src)
    val_ids_all   = collect_episode_ids(val_src)
    if not train_ids_all:
        raise SystemExit(f"[ERR] {train_src} 下找不到 episode_*.npz")
    if not val_ids_all:
        raise SystemExit(f"[ERR] {val_src} 下找不到 episode_*.npz")

    train_ids = parse_id_selector(train_sel, train_ids_all)
    val_ids   = parse_id_selector(val_sel,   val_ids_all)

    train_out = out_root / "training"
    val_out   = out_root / "validation"

    # 链接 episodes
    for eid in train_ids:
        src = train_src / f"episode_{eid:06d}.npz"
        dst = train_out / f"episode_{eid:06d}.npz"
        safe_link_or_copy(src, dst)
    for eid in val_ids:
        src = val_src / f"episode_{eid:06d}.npz"
        dst = val_out / f"episode_{eid:06d}.npz"
        safe_link_or_copy(src, dst)

    # 元数据：优先 val/.hydra；如果没有就全局搜一遍 .hydra
    hydra_src = val_src / ".hydra"
    if not hydra_src.exists():
        cand = src_root / "validation" / ".hydra"
        if cand.exists():
            hydra_src = cand
    hydra_out = val_out / ".hydra"
    copy_if_exists(hydra_src, hydra_out)

    # auto_lang_ann（若存在则拷贝）
    ann_train = train_src / "auto_lang_ann.npy"
    ann_val   = val_src   / "auto_lang_ann.npy"
    copy_if_exists(ann_train, train_out / "auto_lang_ann.npy")
    copy_if_exists(ann_val,   val_out   / "auto_lang_ann.npy")

    # 清单
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "train_episodes.txt").write_text("\n".join(map(str, train_ids))+"\n", encoding="utf-8")
    (out_root / "val_episodes.txt").write_text("\n".join(map(str, val_ids))+"\n", encoding="utf-8")

    manifest = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "chosen_train_split": str(train_src.name),
        "chosen_val_split": str(val_src.name),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_first5": train_ids[:5],
        "val_first5": val_ids[:5],
        "notes": "优先符号链接，失败回退硬链接，再失败复制；已复制 validation/.hydra（若存在）与 auto_lang_ann.npy（若存在）。"
    }
    with open(out_root/"manifest.json","w",encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return len(train_ids), len(val_ids), out_root, manifest

def main():
    ap = argparse.ArgumentParser(description="CALVIN 子集构建（按 episode id，符号链接优先）")
    ap.add_argument("--src", help="task 目录，如 ~/datasets/calvin/dataset/task_D_D")
    ap.add_argument("--root", help="（可选）dataset 根目录，如 ~/datasets/calvin/dataset（与 --task 搭配）")
    ap.add_argument("--task", help="（可选）task 名，如 task_D_D；若提供 --src 则忽略本项")
    ap.add_argument("--out", required=True, help="输出子集根，如 ~/datasets/calvin/workspace/task_D_D_subset")
    ap.add_argument("--train", required=True, help="训练 episode 选择，如 '0-999' 或 文件路径")
    ap.add_argument("--val",   required=True, help="验证 episode 选择，如 '0-199' 或 文件路径")
    ap.add_argument("--train-split", default=None, help="（可选）训练 split 目录名，如 'training_no_lang'")
    ap.add_argument("--val-split",   default=None, help="（可选）验证 split 目录名，如 'validation_seen'")
    args = ap.parse_args()

    if args.src:
        src_root = Path(os.path.expanduser(args.src)).resolve()
    else:
        if not args.root or not args.task:
            raise SystemExit("[ERR] 需要 --src 或 (--root + --task) 之一。")
        src_root = (Path(os.path.expanduser(args.root)).resolve() / args.task).resolve()

    out_root = Path(os.path.expanduser(args.out)).resolve()

    print(f"[PLAN] src={src_root}")
    print(f"[PLAN] out={out_root}")
    tr_n, va_n, outdir, manifest = make_subset(
        src_root, out_root, args.train, args.val,
        train_split=args.train_split, val_split=args.val_split
    )
    print(f"[DONE] 训练集 episodes: {tr_n}，验证集 episodes: {va_n}")
    print(f"[DONE] 子集目录就绪：{outdir}")
    print(f"[HINT] 上游脚本的 dataset_path 指向：{outdir}")
    print(f"[HINT] 选用的 splits：train='{manifest['chosen_train_split']}', val='{manifest['chosen_val_split']}'")

if __name__ == "__main__":
    main()
