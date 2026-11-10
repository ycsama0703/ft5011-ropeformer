#!/usr/bin/env python
"""
Reproducible data builder for FT5011 project (learnable RoPE frequencies for TS forecasting).

Usage (from the repository root; needs pandas, numpy, requests, pyarrow):
    python study/build_data.py --out data [--only ett,lstnet,weather,crypto,pems] [--max-ffill 3]

Downloads raw files into  <DATA>/raw/, verifies them (shape / checksums), writes final CSVs
into <DATA>/ and regenerates <DATA>/SOURCES.md and <DATA>/manifest.json.

Final CSV contract: first column `date`, all other columns numeric, no NaN, chronological order.
Only CPU, modest parallelism (<= 6 download threads). No GPU, no git.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[1] / "data"  # overridden by --out
RAW = DATA / "raw"
TODAY = date.today().isoformat()
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "ft5011-data-builder/1.0 (+student project)"
MANIFEST: dict = {"build_date": TODAY, "datasets": {}}


# ----------------------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 4, allow_404: bool = False) -> bool:
    """Download url -> dest (skip if exists). Returns False on 404 when allow_404."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=120, stream=True)
            if r.status_code == 404 and allow_404:
                return False
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return True
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return True


def check_final(df: pd.DataFrame, name: str) -> None:
    assert df.columns[0] == "date", f"{name}: first column must be date"
    d = pd.to_datetime(df["date"])
    assert d.is_monotonic_increasing, f"{name}: dates not chronological"
    vals = df.iloc[:, 1:]
    assert all(pd.api.types.is_numeric_dtype(t) for t in vals.dtypes), f"{name}: non-numeric columns"
    assert not vals.isna().any().any(), f"{name}: NaN present"
    assert np.isfinite(vals.to_numpy(dtype=float)).all(), f"{name}: inf present"


def record(name: str, final_csv: Path, raw_files: list[Path], sources: list[str], steps: list[str],
           extra: dict | None = None) -> None:
    df = pd.read_csv(final_csv)
    check_final(df, name)
    MANIFEST["datasets"][name] = {
        "status": "OK",
        "final_csv": str(final_csv.relative_to(DATA)),
        "final_sha256": sha256(final_csv),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "date_start": str(df["date"].iloc[0]),
        "date_end": str(df["date"].iloc[-1]),
        "sources": sources,
        "download_date": TODAY,
        "raw_files": [{"path": str(p.relative_to(DATA)), "sha256": sha256(p), "bytes": p.stat().st_size}
                      for p in raw_files],
        "steps": steps,
        **(extra or {}),
    }
    print(f"[OK] {name}: {df.shape[0]} rows x {df.shape[1]} cols, {df['date'].iloc[0]} .. {df['date'].iloc[-1]}")


# ----------------------------------------------------------------------------- 1. ETT
ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
ETT_EXPECT = {"ETTh1": 17420, "ETTh2": 17420, "ETTm1": 69680, "ETTm2": 69680}
ETT_COLS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def build_ett() -> None:
    for name, n in ETT_EXPECT.items():
        url = ETT_BASE + f"{name}.csv"
        raw = RAW / "ETT-small" / f"{name}.csv"
        download(url, raw)
        df = pd.read_csv(raw)
        assert list(df.columns) == ETT_COLS, f"{name}: columns {list(df.columns)}"
        assert df.shape[0] == n, f"{name}: expected {n} rows, got {df.shape[0]}"
        final = DATA / f"{name}.csv"
        shutil.copyfile(raw, final)  # byte-identical to the official file
        record(name, final, [raw], [url],
               ["Downloaded official CSV from zhouhaoyi/ETDataset (branch main).",
                f"Verified header == {ETT_COLS} and exactly {n} rows, no NaN, chronological.",
                "Copied byte-for-byte to the final CSV (no modification)."])


# ----------------------------------------------------------------------------- 2. LSTNet datasets
LSTNET_BASE = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/"
LSTNET = {
    # name: (relative path, expected rows, expected value cols, start, freq)
    "electricity": ("electricity/electricity.txt.gz", 26304, 321, "2016-07-01 02:00:00", "h"),
    "traffic": ("traffic/traffic.txt.gz", 17544, 862, "2016-07-01 02:00:00", "h"),
    "exchange_rate": ("exchange_rate/exchange_rate.txt.gz", 7588, 8, "1990-01-01 00:00:00", "D"),
    "solar": ("solar-energy/solar_AL.txt.gz", 52560, 137, "2006-01-01 00:00:00", "10min"),
}
FREQ_NAME = {"h": "hourly", "D": "daily", "10min": "10-minute"}
XCHECK_AVAILABLE = {"electricity", "traffic", "exchange_rate"}  # datasets hosted on thuml/Time-Series-Library


AUTOFORMER_XCHECK = ("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/"
                     "2b66e59ee19dac8f6f19fb5d4997f289fdfea357/{n}/{n}.csv")


def build_lstnet() -> None:
    for name, (rel, n_rows, n_cols, start, freq) in LSTNET.items():
        url = LSTNET_BASE + rel
        raw = RAW / rel
        download(url, raw)
        with gzip.open(raw, "rb") as f:
            df = pd.read_csv(f, header=None)
        assert df.shape == (n_rows, n_cols), f"{name}: shape {df.shape} != ({n_rows},{n_cols})"
        steps = ["Downloaded gzip'd comma-separated matrix (no header) from laiguokun/multivariate-time-series-data (branch master).",
                 f"Verified shape exactly ({n_rows}, {n_cols}) and no NaN."]
        if name == "exchange_rate":
            # The Autoformer exchange_rate.csv (used by Autoformer/DLinear/PatchTST) has the last two LSTNet
            # columns swapped: Autoformer `OT` == LSTNet column 6, Autoformer `6` == LSTNet column 7.
            # Follow the Autoformer file so results are comparable with the literature.
            df = df[[*range(n_cols - 2), n_cols - 1, n_cols - 2]]
            steps.append("Reordered the last two columns to match the Autoformer exchange_rate.csv exactly "
                         "(Autoformer `OT` = LSTNet column index 6, Autoformer `6` = LSTNet column index 7).")
        cols = [str(i) for i in range(n_cols - 1)] + ["OT"]
        df.columns = cols
        idx = pd.date_range(start, periods=n_rows, freq=freq)
        df = pd.concat([pd.Series(idx.strftime("%Y-%m-%d %H:%M:%S"), name="date"), df], axis=1)
        final = DATA / f"{name}.csv"
        df.to_csv(final, index=False)
        steps += [f"Added `date` column: {FREQ_NAME[freq]} index starting {start} (Autoformer convention).",
                  f"Named value columns 0..{n_cols-2} and the last one `OT` (Autoformer convention).",
                  "Values written unchanged (pandas default float repr)."]
        if name not in XCHECK_AVAILABLE:
            record(name, final, [raw], [url], steps)
            continue
        # Cross-check against the Autoformer CSV hosted on the official THUML HuggingFace mirror.
        xurl = AUTOFORMER_XCHECK.format(n=name)
        xraw = RAW / "autoformer_xcheck" / f"{name}.csv"
        download(xurl, xraw)
        x = pd.read_csv(xraw)
        assert x.shape == df.shape and list(x.columns) == list(df.columns), f"{name}: xcheck shape/cols differ"
        assert (pd.to_datetime(x["date"]).values == pd.to_datetime(df["date"]).values).all(), f"{name}: xcheck dates differ"
        assert np.allclose(x.iloc[:, 1:].to_numpy(float), df.iloc[:, 1:].to_numpy(float), rtol=0, atol=1e-9), \
            f"{name}: xcheck values differ"
        steps.append("Verified the final CSV equals the Autoformer CSV from thuml/Time-Series-Library (HF, commit 2b66e59): "
                     "same columns, same dates, values equal within 1e-9 (its date strings are 'YYYY/M/D H:MM' for exchange_rate).")
        record(name, final, [raw, xraw], [url, xurl], steps)


# ----------------------------------------------------------------------------- 3. weather
WEATHER_URL = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv"
WEATHER_COMMIT = "2b66e59ee19dac8f6f19fb5d4997f289fdfea357"  # HF repo commit observed on download day
WEATHER_PINNED = WEATHER_URL.replace("/resolve/main/", f"/resolve/{WEATHER_COMMIT}/")


def build_weather() -> None:
    raw = RAW / "weather" / "weather.csv"
    download(WEATHER_PINNED, raw)
    # The original Autoformer file header is latin-1 encoded ("W/m²", "µmol"), so decode explicitly.
    df = pd.read_csv(raw, encoding="latin-1")
    assert df.shape == (52696, 22), f"weather: shape {df.shape} != (52696, 22)"
    assert df.columns[0] == "date" and df.columns[-1] == "OT", list(df.columns)
    final = DATA / "weather.csv"
    df.to_csv(final, index=False, encoding="utf-8")
    record("weather", final, [raw], [WEATHER_PINNED, WEATHER_URL],
           ["Downloaded Autoformer/Time-Series-Library weather.csv from the official THUML HuggingFace mirror "
            f"(thuml/Time-Series-Library, commit {WEATHER_COMMIT}).",
            "Verified exactly 52696 rows x 22 columns, first column `date`, last column `OT`, no NaN, chronological.",
            "Header decoded as latin-1 and re-written as UTF-8 (column names 'SWDR (W/m²)', 'PAR (µmol/m²/s)', "
            "'max. PAR (µmol/m²/s)' are now valid UTF-8); numeric values unchanged.",
            "Only the encoding of the header differs from the raw file."],
           extra={"columns": list(df.columns)})


# ----------------------------------------------------------------------------- 4. crypto hourly (Binance)
BINANCE = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{ym}.zip"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "SOLUSDT"]
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
MAX_FFILL_HOURS = 3


def months(first="2021-01", last="2026-10") -> list[str]:
    out, (y, m) = [], map(int, first.split("-"))
    ly, lm = map(int, last.split("-"))
    while (y, m) <= (ly, lm):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch_month(sym: str, ym: str) -> tuple[str, str, bool, Path | None]:
    url = BINANCE.format(s=sym, ym=ym)
    zdest = RAW / "binance" / sym / f"{sym}-1h-{ym}.zip"
    cdest = zdest.with_suffix(".zip.CHECKSUM")
    ok = download(url, zdest, allow_404=True)
    if not ok:
        return sym, ym, False, None
    download(url + ".CHECKSUM", cdest)
    expected = cdest.read_text().split()[0].lower()
    actual = sha256(zdest)
    if expected != actual:
        zdest.unlink(missing_ok=True)
        cdest.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch {sym} {ym}: {expected} != {actual}")
    return sym, ym, True, zdest


def read_kline_zip(zpath: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        assert len(names) == 1, names
        raw = z.read(names[0])
    first = raw.split(b"\n", 1)[0]
    has_header = not first.split(b",")[0].strip().isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    df.columns = KLINE_COLS[: df.shape[1]]
    assert df.shape[1] == 12, df.shape
    # From 2025 Binance spot timestamps are microseconds; detect per row (> 1e14 => microseconds).
    for c in ("open_time", "close_time"):
        t = df[c].astype("int64").to_numpy()
        micro = t > 1e14
        t = np.where(micro, t // 1000, t)
        df[c] = t
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    return df[["date", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]


def build_crypto() -> None:
    # Probe reachability first.
    try:
        r = SESSION.head(BINANCE.format(s="BTCUSDT", ym="2021-01"), timeout=30)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        MANIFEST["datasets"]["crypto_hourly"] = {"status": "MISSING", "reason": f"data.binance.vision unreachable: {e}"}
        print("[MISSING] crypto_hourly: data.binance.vision unreachable", e)
        return

    all_months = months()
    results: dict[str, dict[str, Path | None]] = {s: {} for s in SYMBOLS}
    jobs = [(s, ym) for s in SYMBOLS for ym in all_months]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch_month, s, ym) for s, ym in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            s, ym, ok, p = fut.result()
            results[s][ym] = p if ok else None
            if i % 50 == 0:
                print(f"  ... {i}/{len(jobs)} monthly files done")

    # Last month available for every symbol (files are published a few days after month end).
    available = {s: [ym for ym in all_months if results[s][ym] is not None] for s in SYMBOLS}
    last_common = min(max(v) for v in available.values())
    missing_report = {s: [ym for ym in all_months if ym <= last_common and results[s][ym] is None] for s in SYMBOLS}
    assert all(len(v) == 0 for v in missing_report.values()), f"internal gaps in monthly files: {missing_report}"
    use_months = [ym for ym in all_months if ym <= last_common]
    raw_files = [results[s][ym] for s in SYMBOLS for ym in use_months]
    raw_files += [p.with_suffix(".zip.CHECKSUM") for p in raw_files]

    per_sym: dict[str, pd.DataFrame] = {}
    for s in SYMBOLS:
        parts = [read_kline_zip(results[s][ym]) for ym in use_months]
        df = pd.concat(parts, ignore_index=True).drop_duplicates("date").sort_values("date").set_index("date")
        assert (df.index.minute == 0).all() and (df.index.second == 0).all(), f"{s}: non-hourly open_time"
        per_sym[s] = df

    start = max(df.index.min() for df in per_sym.values())
    end = min(df.index.max() for df in per_sym.values())
    full_idx = pd.date_range(start, end, freq="h")

    gap_report: dict[str, dict] = {}
    long_gaps_end = start  # earliest time after which every symbol has only short (<=3h) gaps
    filled: dict[str, pd.DataFrame] = {}
    for s, df in per_sym.items():
        rx = df.reindex(full_idx)
        missing = rx["close"].isna()
        # identify contiguous missing runs
        runs = []
        if missing.any():
            m = missing.to_numpy()
            edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                runs.append((full_idx[a], full_idx[b - 1], int(b - a)))
        short = [r for r in runs if r[2] <= MAX_FFILL_HOURS]
        long_ = [r for r in runs if r[2] > MAX_FFILL_HOURS]
        if long_:
            long_gaps_end = max(long_gaps_end, max(r[1] for r in long_) + pd.Timedelta(hours=1))
        gap_report[s] = {
            "missing_hours_total": int(missing.sum()),
            "short_gaps_filled": len(short),
            "hours_forward_filled": int(sum(r[2] for r in short)),
            "long_gaps": [{"from": str(a), "to": str(b), "hours": n} for a, b, n in long_],
        }
        filled[s] = rx

    # Choose final start: after the last long gap of any symbol (if any), then forward-fill short gaps.
    final_start = long_gaps_end
    final_idx = full_idx[full_idx >= final_start]
    lrv_cols, lqv_cols = {}, {}
    for s in SYMBOLS:
        rx = filled[s].loc[final_idx]
        rx = rx.ffill(limit=MAX_FFILL_HOURS)
        assert not rx["close"].isna().any(), f"{s}: NaN remain after limited ffill"
        hl = np.log(rx["high"].to_numpy(dtype=float) / rx["low"].to_numpy(dtype=float))
        park = hl / (2.0 * math.sqrt(math.log(2.0)))
        with np.errstate(divide="ignore"):
            lrv = np.log(park)
        lrv = np.where(hl > 0, lrv, math.log(1e-6))
        lrv = np.maximum(lrv, math.log(1e-6))
        lqv = np.log1p(rx["quote_volume"].to_numpy(dtype=float))
        lrv_cols[f"{s}_lrv"] = lrv
        lqv_cols[f"{s}_lqv"] = lqv
        gap_report[s]["hours_forward_filled_in_final_range"] = int(
            filled[s].loc[final_idx, "close"].isna().sum())
        gap_report[s]["high_eq_low_hours"] = int((hl <= 0).sum())

    out = pd.DataFrame({"date": final_idx.strftime("%Y-%m-%d %H:%M:%S"), **lrv_cols, **lqv_cols})
    final = DATA / "crypto_hourly.csv"
    out.to_csv(final, index=False, float_format="%.8f")
    record("crypto_hourly", final, raw_files,
           [BINANCE.format(s="{SYMBOL}", ym="{YYYY-MM}"), BINANCE.format(s="{SYMBOL}", ym="{YYYY-MM}") + ".CHECKSUM"],
           [f"Downloaded monthly 1h spot kline zips for {SYMBOLS} for months {use_months[0]}..{use_months[-1]} "
            f"({len(use_months)} months x 8 symbols = {len(use_months)*8} zips) plus their .CHECKSUM files; "
            "every zip's sha256 was verified against the published checksum.",
            "Parsed headerless kline CSVs (12 columns). Timestamps > 1e14 are treated as microseconds and "
            "divided by 1000 (Binance switched spot timestamps to microseconds from 2025-01).",
            "open_time converted to naive UTC datetime, used as `date` (bar open time).",
            f"Built the hourly UTC index over the common range of all symbols: {start} .. {end}.",
            f"Gaps <= {MAX_FFILL_HOURS}h forward-filled (raw OHLCV carried forward before feature computation); "
            "longer gaps are reported and the output starts after the last long gap "
            f"(final start = {final_start}).",
            "{SYM}_lrv = log( ln(high/low) / (2*sqrt(ln 2)) ), floored at log(1e-6) when high == low.",
            "{SYM}_lqv = log(1 + quote_volume).",
            "Column order: date, all *_lrv, then all *_lqv. Values written with 8 decimals."],
           extra={"symbols": SYMBOLS, "months_used": [use_months[0], use_months[-1]],
                  "last_common_month": last_common,
                  "months_not_yet_published": {s: [ym for ym in all_months if results[s][ym] is None] for s in SYMBOLS},
                  "common_range": [str(start), str(end)], "final_start": str(final_start),
                  "gap_report": gap_report})


# ----------------------------------------------------------------------------- 5. PEMS04 / PEMS08 (traffic flow, 5-min)
# thuml/Time-Series-Library (commit 2b66e59) does NOT contain PEMS/PEMS04.npz or PEMS/PEMS08.npz (checked 2026-11-09;
# it only hosts the UEA PEMS-SF .ts files). The STSGCN-format .npz files are taken from the HF repos below and
# cross-checked bit-for-bit against the STSGCN author's own HF mirror (Davidham3/pems-stsgcn, parquet).
PEMS = {
    # name: (npz url pinned to commit, expected T, expected N, nominal start)
    "PEMS04": ("https://huggingface.co/datasets/jimmygao3218/PEMS04/resolve/0d0e451662ac964505dd6fcea4a739f416b4ee25/PEMS04.npz",
               16992, 307, "2018-01-01 00:00:00"),
    "PEMS08": ("https://huggingface.co/datasets/jimmygao3218/PEMS08/resolve/7d5f30e2f1b538973ff805fc84ac42a0bd7422aa/PEMS08.npz",
               17856, 170, "2016-07-01 00:00:00"),
}
PEMS_XCHECK = ("https://huggingface.co/datasets/Davidham3/pems-stsgcn/resolve/"
               "c4a988ceca7f330a2283db6aeddf765dd6eac63f/{n}/{n}_series.parquet")


def build_pems() -> None:
    for name, (url, T, N, start) in PEMS.items():
        raw = RAW / "pems" / "jimmygao3218" / f"{name}.npz"
        download(url, raw)
        with np.load(raw) as z:
            assert z.files == ["data"], z.files
            arr = z["data"]
        assert arr.shape[:2] == (T, N) and arr.ndim == 3, f"{name}: shape {arr.shape} != ({T},{N},F)"
        flow = arr[:, :, 0].astype(np.float64)
        assert not np.isnan(flow).any(), f"{name}: NaN in flow"
        # cross-check with the STSGCN author's parquet mirror (flow column = list<float64>[N] per timestep)
        xurl = PEMS_XCHECK.format(n=name)
        xraw = RAW / "pems" / "davidham3" / f"{name}_series.parquet"
        download(xurl, xraw)
        px = pd.read_parquet(xraw, columns=["timestep", "flow"])
        assert (px["timestep"].to_numpy() == np.arange(T)).all(), f"{name}: parquet timesteps not 0..T-1"
        xflow = np.stack(px["flow"].to_numpy())
        assert xflow.shape == flow.shape and np.array_equal(xflow, flow), f"{name}: npz flow != author parquet flow"
        cols = [str(i) for i in range(N - 1)] + ["OT"]
        idx = pd.date_range(start, periods=T, freq="5min")
        df = pd.concat([pd.Series(idx.strftime("%Y-%m-%d %H:%M:%S"), name="date"),
                        pd.DataFrame(flow, columns=cols)], axis=1)
        final = DATA / f"{name}.csv"
        df.to_csv(final, index=False)
        record(name, final, [raw, xraw], [url, xurl],
               [f"Downloaded STSGCN-format `{name}.npz` from HuggingFace (jimmygao3218/{name}, pinned commit) — "
                "thuml/Time-Series-Library does not host PEMS04/PEMS08.",
                f"Verified npz key `data` with shape ({T}, {N}, {arr.shape[2]}) [T, N, F], float64, no NaN.",
                "Kept feature index 0 (traffic flow) only; occupancy/speed discarded.",
                "Verified flow equals bit-for-bit the `flow` column of the STSGCN author's HF mirror "
                "Davidham3/pems-stsgcn (<name>_series.parquet, commit c4a988c, which documents verification against "
                "STSGCN GitHub commit 3f825f6).",
                f"Added NOMINAL `date` column: 5-minute index starting {start} ({T} steps = {T//288} days). "
                "The npz carries no timestamps; the start date follows the standard dataset description "
                f"({'Jan–Feb 2018' if name=='PEMS04' else 'Jul–Aug 2016'}) and is nominal, not verified against Caltrans PeMS.",
                f"Named value columns 0..{N-2} and the last sensor `OT` (Autoformer convention); values written unchanged."],
               extra={"nominal_dates": True, "npz_shape": list(arr.shape)})


# ----------------------------------------------------------------------------- SOURCES.md
def write_sources() -> None:
    (DATA / "manifest.json").write_text(json.dumps(MANIFEST, indent=2))
    L = [f"# Data sources and provenance\n",
         f"Build date (download date): **{MANIFEST['build_date']}**. Built by `build_data.py` on aidf-gpu04 "
         f"(python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}).\n",
         "All final CSVs: first column `date`, all other columns numeric, no NaN, chronological order.\n"]
    for name, m in MANIFEST["datasets"].items():
        L.append(f"\n## {name}\n")
        if m.get("status") != "OK":
            L.append(f"**Status: {m['status']}** — {m.get('reason','')}\n")
            continue
        L.append(f"- Final file: `{m['final_csv']}` — sha256 `{m['final_sha256']}`")
        L.append(f"- Shape: **{m['rows']} rows × {m['cols']} columns** (incl. `date`)")
        L.append(f"- Date range: {m['date_start']} → {m['date_end']}"
                 + ("  **(nominal dates — see steps)**" if m.get("nominal_dates") else ""))
        L.append(f"- Download date: {m['download_date']}")
        L.append("- Source URL(s):")
        for u in m["sources"]:
            L.append(f"  - `{u}`")
        if name == "crypto_hourly":
            L.append(f"- Symbols: {', '.join(m['symbols'])}; months used {m['months_used'][0]}..{m['months_used'][1]} "
                     f"(last month published for all symbols: {m['last_common_month']})")
            npub = {k: v for k, v in m["months_not_yet_published"].items() if v}
            if npub:
                L.append(f"- Months requested but not published: {npub}")
            L.append(f"- Common raw range {m['common_range'][0]} .. {m['common_range'][1]}; final start {m['final_start']}")
            L.append("- Gap report per symbol:")
            for s, g in m["gap_report"].items():
                L.append(f"  - {s}: {g['missing_hours_total']} missing hours in common range; "
                         f"{g['hours_forward_filled_in_final_range']} hours forward-filled in final range "
                         f"({g['short_gaps_filled']} short gaps ≤3h); long gaps (>3h): "
                         f"{g['long_gaps'] if g['long_gaps'] else 'none'}; high==low hours: {g['high_eq_low_hours']}")
            L.append(f"- Raw files: {len(m['raw_files'])} (zips + CHECKSUMs) under `raw/binance/`; "
                     "sha256 of each zip is listed in `manifest.json` and equals its published `.CHECKSUM`.")
        else:
            L.append("- Raw file(s):")
            for rf in m["raw_files"]:
                L.append(f"  - `{rf['path']}` ({rf['bytes']} bytes) sha256 `{rf['sha256']}`")
        L.append("- Processing steps:")
        for i, s in enumerate(m["steps"], 1):
            L.append(f"  {i}. {s}")
    (DATA / "SOURCES.md").write_text("\n".join(L) + "\n")
    print("wrote SOURCES.md and manifest.json")


def main() -> None:
    global MAX_FFILL_HOURS, DATA, RAW
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA, help="output directory for raw files and final CSVs")
    ap.add_argument("--only", default="ett,lstnet,weather,crypto,pems")
    ap.add_argument("--max-ffill", type=int, default=MAX_FFILL_HOURS,
                    help="crypto: forward-fill gaps up to this many hours (default 3). The only long gap in "
                         "2021-01..2026-10 is a 4h exchange-wide outage on 2021-08-13; --max-ffill 4 keeps 2021-01 onward.")
    args = ap.parse_args()
    MAX_FFILL_HOURS = args.max_ffill
    DATA, RAW = args.out.resolve(), args.out.resolve() / "raw"
    RAW.mkdir(parents=True, exist_ok=True)
    todo = set(args.only.split(","))
    mpath = DATA / "manifest.json"
    if mpath.exists():  # keep entries for datasets not rebuilt in this run
        MANIFEST["datasets"].update(json.loads(mpath.read_text()).get("datasets", {}))
    for key, fn in [("ett", build_ett), ("lstnet", build_lstnet), ("weather", build_weather), ("crypto", build_crypto),
                    ("pems", build_pems)]:
        if key in todo:
            t0 = time.time()
            fn()
            print(f"  [{key}] done in {time.time()-t0:.0f}s")
    # order datasets deterministically
    order = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic", "solar", "crypto_hourly", "exchange_rate", "weather",
             "PEMS04", "PEMS08"]
    MANIFEST["datasets"] = {k: MANIFEST["datasets"][k] for k in order if k in MANIFEST["datasets"]}
    write_sources()


if __name__ == "__main__":
    main()
