"""Download benchmark CSVs from the official thuml/Time-Series-Library dataset (CC BY 4.0).

Run explicitly; nothing in the package downloads data. Files go to data/ (Git-ignored).
The SHA-256 of each file is printed so runs can record exactly which data they used.

    python tools/download_data.py ETTh1 electricity
"""
import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

SOURCE = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/"
FILES = {
    "ETTh1": "ETT-small/ETTh1.csv",
    "ETTh2": "ETT-small/ETTh2.csv",
    "ETTm1": "ETT-small/ETTm1.csv",
    "ETTm2": "ETT-small/ETTm2.csv",
    "weather": "weather/weather.csv",
    "electricity": "electricity/electricity.csv",
    "traffic": "traffic/traffic.csv",
    "exchange_rate": "exchange_rate/exchange_rate.csv",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", choices=sorted(FILES))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        target = args.out_dir / f"{name}.csv"
        if target.exists() and not args.force:
            print(f"{name}: exists ({target.stat().st_size} bytes)")
        else:
            partial = target.with_suffix(".csv.part")
            with urllib.request.urlopen(SOURCE + FILES[name]) as response, open(partial, "wb") as f:
                expected = response.headers.get("Content-Length")
                shutil.copyfileobj(response, f)
            if expected is not None and partial.stat().st_size != int(expected):
                raise SystemExit(f"{name}: truncated download ({partial.stat().st_size} of {expected} bytes); retry")
            partial.replace(target)
            print(f"{name}: downloaded {target.stat().st_size} bytes from {SOURCE + FILES[name]}")
        print(f"  sha256 {sha256(target)}")


if __name__ == "__main__":
    main()
