from __future__ import annotations

import argparse
import base64
import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


CDN_IPS = (
    "47.130.32.119",
    "13.229.7.180",
    "3.0.71.232",
    "18.136.219.244",
    "47.131.129.243",
    "54.254.163.78",
    "54.169.134.11",
    "52.221.162.66",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-base64", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args()
    if args.size <= 0 or not 1 <= args.connections <= len(CDN_IPS):
        raise ValueError("invalid parallel range-download dimensions")
    url = base64.b64decode(args.url_base64, validate=True).decode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    part_size = (args.size + args.connections - 1) // args.connections

    def download(index: int) -> Path:
        start = index * part_size
        stop = min(args.size, start + part_size) - 1
        part = args.output.with_name(f"{args.output.name}.part-{index:02d}")
        expected = stop - start + 1
        if part.is_file() and part.stat().st_size == expected:
            return part
        part.unlink(missing_ok=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "8",
                "--retry-all-errors",
                "--connect-timeout",
                "8",
                "--resolve",
                f"us.aws.cdn.hf.co:443:{CDN_IPS[index]}",
                "--range",
                f"{start}-{stop}",
                "--output",
                str(part),
                url,
            ],
            check=True,
        )
        if part.stat().st_size != expected:
            raise RuntimeError(f"range {index} has {part.stat().st_size} bytes, expected {expected}")
        return part

    with ThreadPoolExecutor(max_workers=args.connections) as pool:
        parts = list(pool.map(download, range(args.connections)))
    temporary = args.output.with_name(f"{args.output.name}.assembling")
    with temporary.open("wb") as destination:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    if temporary.stat().st_size != args.size:
        raise RuntimeError("assembled download has the wrong byte count")
    observed = _sha256(temporary)
    if observed != args.sha256:
        raise RuntimeError(f"download SHA-256 mismatch: expected {args.sha256}, observed {observed}")
    temporary.replace(args.output)
    for part in parts:
        part.unlink()
    print(f"verified {args.output} {args.size} {observed}")


if __name__ == "__main__":
    main()
