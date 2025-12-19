# scripts/bench_calvin_loader_throughput.py
import argparse
import dataclasses
import os
import time

from openpi.training.config import get_config
from openpi.training.data_loader import create_data_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_calvin_sonata")
    ap.add_argument("--calvin-root", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--warmup-batches", type=int, default=10)
    ap.add_argument("--num-batches", type=int, default=200)
    args = ap.parse_args()

    if args.calvin_root:
        os.environ["CALVIN_ZIP"] = args.calvin_root

    cfg = get_config(args.config)
    if args.num_workers is not None:
        cfg = dataclasses.replace(cfg, num_workers=args.num_workers)
    if args.batch_size is not None:
        cfg = dataclasses.replace(cfg, batch_size=args.batch_size)

    # warmup
    dl = create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=args.warmup_batches, skip_norm_stats=False)
    for _ in dl:
        pass

    # measure
    dl = create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=args.num_batches, skip_norm_stats=False)
    t0 = time.time()
    n_samples = 0
    for obs, act in dl:
        n_samples += int(obs.state.shape[0])
    dt = time.time() - t0

    print(f"[bench] cfg={args.config} batch_size={cfg.batch_size} num_workers={cfg.num_workers}")
    print(f"[bench] batches={args.num_batches} samples={n_samples} dt={dt:.2f}s")
    print(f"[bench] sec/batch={dt/args.num_batches:.3f} samples/s={n_samples/dt:.2f}")


if __name__ == "__main__":
    main()
