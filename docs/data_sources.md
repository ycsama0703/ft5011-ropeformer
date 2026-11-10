# Data sources and provenance

Build date (download date): **2026-11-09**. Built by `build_data.py` on aidf-gpu04 (python 3.12.13, pandas 3.1.1, numpy 2.5.4).

All final CSVs: first column `date`, all other columns numeric, no NaN, chronological order.


## ETTh1

- Final file: `ETTh1.csv` — sha256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`
- Shape: **17420 rows × 8 columns** (incl. `date`)
- Date range: 2016-07-01 00:00:00 → 2018-06-26 19:00:00
- Download date: 2026-11-09
- Source URL(s):
  - `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`
- Raw file(s):
  - `raw/ETT-small/ETTh1.csv` (2589657 bytes) sha256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`
- Processing steps:
  1. Downloaded official CSV from zhouhaoyi/ETDataset (branch main).
  2. Verified header == ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'] and exactly 17420 rows, no NaN, chronological.
  3. Copied byte-for-byte to the final CSV (no modification).

## ETTh2

- Final file: `ETTh2.csv` — sha256 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`
- Shape: **17420 rows × 8 columns** (incl. `date`)
- Date range: 2016-07-01 00:00:00 → 2018-06-26 19:00:00
- Download date: 2026-11-09
- Source URL(s):
  - `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv`
- Raw file(s):
  - `raw/ETT-small/ETTh2.csv` (2417960 bytes) sha256 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`
- Processing steps:
  1. Downloaded official CSV from zhouhaoyi/ETDataset (branch main).
  2. Verified header == ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'] and exactly 17420 rows, no NaN, chronological.
  3. Copied byte-for-byte to the final CSV (no modification).

## ETTm1

- Final file: `ETTm1.csv` — sha256 `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`
- Shape: **69680 rows × 8 columns** (incl. `date`)
- Date range: 2016-07-01 00:00:00 → 2018-06-26 19:45:00
- Download date: 2026-11-09
- Source URL(s):
  - `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv`
- Raw file(s):
  - `raw/ETT-small/ETTm1.csv` (10360719 bytes) sha256 `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`
- Processing steps:
  1. Downloaded official CSV from zhouhaoyi/ETDataset (branch main).
  2. Verified header == ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'] and exactly 69680 rows, no NaN, chronological.
  3. Copied byte-for-byte to the final CSV (no modification).

## ETTm2

- Final file: `ETTm2.csv` — sha256 `db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd`
- Shape: **69680 rows × 8 columns** (incl. `date`)
- Date range: 2016-07-01 00:00:00 → 2018-06-26 19:45:00
- Download date: 2026-11-09
- Source URL(s):
  - `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv`
- Raw file(s):
  - `raw/ETT-small/ETTm2.csv` (9677236 bytes) sha256 `db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd`
- Processing steps:
  1. Downloaded official CSV from zhouhaoyi/ETDataset (branch main).
  2. Verified header == ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'] and exactly 69680 rows, no NaN, chronological.
  3. Copied byte-for-byte to the final CSV (no modification).

## electricity

- Final file: `electricity.csv` — sha256 `066b25eb3de3df8d9941692a59d5d012d7b25327e3cec2c06e024a037cb7eb10`
- Shape: **26304 rows × 322 columns** (incl. `date`)
- Date range: 2016-07-01 02:00:00 → 2019-07-02 01:00:00
- Download date: 2026-11-09
- Source URL(s):
  - `https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz`
  - `https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/electricity/electricity.csv`
- Raw file(s):
  - `raw/electricity/electricity.txt.gz` (17993794 bytes) sha256 `3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa`
  - `raw/autoformer_xcheck/electricity.csv` (95581762 bytes) sha256 `7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad`
- Processing steps:
  1. Downloaded gzip'd comma-separated matrix (no header) from laiguokun/multivariate-time-series-data (branch master).
  2. Verified shape exactly (26304, 321) and no NaN.
  3. Added `date` column: hourly index starting 2016-07-01 02:00:00 (Autoformer convention).
  4. Named value columns 0..319 and the last one `OT` (Autoformer convention).
  5. Values written unchanged (pandas default float repr).
  6. Verified the final CSV equals the Autoformer CSV from thuml/Time-Series-Library (HF, commit 2b66e59): same columns, same dates, values equal within 1e-9 (its date strings are 'YYYY/M/D H:MM' for exchange_rate).

## traffic

- Final file: `traffic.csv` — sha256 `78325acb321bc868ce5785021319aecf5f11e6c9c5e8afae2f252f2c27d98d3b`
- Shape: **17544 rows × 863 columns** (incl. `date`)
- Date range: 2016-07-01 02:00:00 → 2018-07-02 01:00:00
- Download date: 2026-11-09
- Source URL(s):
  - `https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz`
  - `https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/traffic/traffic.csv`
- Raw file(s):
  - `raw/traffic/traffic.txt.gz` (29310512 bytes) sha256 `c7be5a00519d344a5ec0eabdbfec5ea0c7dd1eed5f9b1a3843a93bb88086a56d`
  - `raw/autoformer_xcheck/traffic.csv` (136478119 bytes) sha256 `cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16`
- Processing steps:
  1. Downloaded gzip'd comma-separated matrix (no header) from laiguokun/multivariate-time-series-data (branch master).
  2. Verified shape exactly (17544, 862) and no NaN.
  3. Added `date` column: hourly index starting 2016-07-01 02:00:00 (Autoformer convention).
  4. Named value columns 0..860 and the last one `OT` (Autoformer convention).
  5. Values written unchanged (pandas default float repr).
  6. Verified the final CSV equals the Autoformer CSV from thuml/Time-Series-Library (HF, commit 2b66e59): same columns, same dates, values equal within 1e-9 (its date strings are 'YYYY/M/D H:MM' for exchange_rate).

## solar

- Final file: `solar.csv` — sha256 `0adbe03788b99705c7a2459d5cd89fbcf8791980599deadf9d34cb6570cf35d7`
- Shape: **52560 rows × 138 columns** (incl. `date`)
- Date range: 2006-01-01 00:00:00 → 2006-12-31 23:50:00
- Download date: 2026-11-09
- Source URL(s):
  - `https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz`
- Raw file(s):
  - `raw/solar-energy/solar_AL.txt.gz` (8748993 bytes) sha256 `1ae9329c3e1032acbb8dd080bfc397c44507006812a715d2788f172820489a1c`
- Processing steps:
  1. Downloaded gzip'd comma-separated matrix (no header) from laiguokun/multivariate-time-series-data (branch master).
  2. Verified shape exactly (52560, 137) and no NaN.
  3. Added `date` column: 10-minute index starting 2006-01-01 00:00:00 (Autoformer convention).
  4. Named value columns 0..135 and the last one `OT` (Autoformer convention).
  5. Values written unchanged (pandas default float repr).

## crypto_hourly

- Final file: `crypto_hourly.csv` — sha256 `f816c609f20d9f5cee942a5bbc4ab0d0ff2ffa2aed92a9e94a7e0e932a280486`
- Shape: **45738 rows × 17 columns** (incl. `date`)
- Date range: 2021-08-13 06:00:00 → 2026-10-31 23:00:00
- Download date: 2026-11-09
- Source URL(s):
  - `https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1h/{SYMBOL}-1h-{YYYY-MM}.zip`
  - `https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1h/{SYMBOL}-1h-{YYYY-MM}.zip.CHECKSUM`
- Symbols: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LTCUSDT, SOLUSDT; months used 2021-01..2026-10 (last month published for all symbols: 2026-10)
- Common raw range 2021-01-01 00:00:00 .. 2026-10-31 23:00:00; final start 2021-08-13 06:00:00
- Gap report per symbol:
  - BTCUSDT: 14 missing hours in common range; 3 hours forward-filled in final range (6 short gaps ≤3h); long gaps (>3h): [{'from': '2021-08-13 02:00:00', 'to': '2021-08-13 05:00:00', 'hours': 4}]; high==low hours: 2
  - ETHUSDT: 14 missing hours in common range; 3 hours forward-filled in final range (6 short gaps ≤3h); long gaps (>3h): [{'from': '2021-08-13 02:00:00', 'to': '2021-08-13 05:00:00', 'hours': 4}]; high==low hours: 2
  - BNBUSDT: 14 missing hours in common range; 3 hours forward-filled in final range (6 short gaps ≤3h); long gaps (>3h): [{'from': '2021-08-13 02:00:00', 'to': '2021-08-13 05:00:00', 'hours': 4}]; high==low hours: 2
  - XRPUSDT: 14 missing hours in common range; 3 hours forward-filled in final range (6 short gaps ≤3h); long gaps (>3h): [{'from': '2021-08-13 02:00:00', 'to': '2021-08-13 05:00:00', 'hours': 4}]; high==low hours: 2
  - ADAUSDT: 14 missing hours in common range; 3 hours forward-filled in final range (6 short gaps ≤3h); long gaps (>3h): [{'from': '2021-08-13 02:00:00', 'to': '2021-08-13 05:00:00', 'hours': 4}]; high==low hours: 2
  - DOGEUSDT: 14 missing hours in common range; 3 hours forward-filled in final range (6 short gaps ≤3h); long gaps (>3h): [{'from': '2021-08-13 02:00:00', 'to': '2021-08-13 05:00:00', 'hours': 4}]; high==low hours: 2
  - LTCUSDT: 14 missing hours in common range; 3 hours forward-filled in final range (6 short gaps ≤3h); long gaps (>3h): [{'from': '2021-08-13 02:00:00', 'to': '2021-08-13 05:00:00', 'hours': 4}]; high==low hours: 2
  - SOLUSDT: 14 missing hours in common range; 3 hours forward-filled in final range (6 short gaps ≤3h); long gaps (>3h): [{'from': '2021-08-13 02:00:00', 'to': '2021-08-13 05:00:00', 'hours': 4}]; high==low hours: 2
- Raw files: 1120 (zips + CHECKSUMs) under `raw/binance/`; sha256 of each zip is listed in `manifest.json` and equals its published `.CHECKSUM`.
- Processing steps:
  1. Downloaded monthly 1h spot kline zips for ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LTCUSDT', 'SOLUSDT'] for months 2021-01..2026-10 (70 months x 8 symbols = 560 zips) plus their .CHECKSUM files; every zip's sha256 was verified against the published checksum.
  2. Parsed headerless kline CSVs (12 columns). Timestamps > 1e14 are treated as microseconds and divided by 1000 (Binance switched spot timestamps to microseconds from 2025-01).
  3. open_time converted to naive UTC datetime, used as `date` (bar open time).
  4. Built the hourly UTC index over the common range of all symbols: 2021-01-01 00:00:00 .. 2026-10-31 23:00:00.
  5. Gaps <= 3h forward-filled (raw OHLCV carried forward before feature computation); longer gaps are reported and the output starts after the last long gap (final start = 2021-08-13 06:00:00).
  6. {SYM}_lrv = log( ln(high/low) / (2*sqrt(ln 2)) ), floored at log(1e-6) when high == low.
  7. {SYM}_lqv = log(1 + quote_volume).
  8. Column order: date, all *_lrv, then all *_lqv. Values written with 8 decimals.

## exchange_rate

- Final file: `exchange_rate.csv` — sha256 `7ebe539df3f6d67a44dd68268ee9ec92f4b16ba29d116cb22a7c57d7fe8876a6`
- Shape: **7588 rows × 9 columns** (incl. `date`)
- Date range: 1990-01-01 00:00:00 → 2010-10-10 00:00:00
- Download date: 2026-11-09
- Source URL(s):
  - `https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz`
  - `https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/exchange_rate/exchange_rate.csv`
- Raw file(s):
  - `raw/exchange_rate/exchange_rate.txt.gz` (177142 bytes) sha256 `6a4fbb561dcdee43f071a83badbeec9223a335552278348656738b556f1c2e9d`
  - `raw/autoformer_xcheck/exchange_rate.csv` (637800 bytes) sha256 `48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842`
- Processing steps:
  1. Downloaded gzip'd comma-separated matrix (no header) from laiguokun/multivariate-time-series-data (branch master).
  2. Verified shape exactly (7588, 8) and no NaN.
  3. Reordered the last two columns to match the Autoformer exchange_rate.csv exactly (Autoformer `OT` = LSTNet column index 6, Autoformer `6` = LSTNet column index 7).
  4. Added `date` column: daily index starting 1990-01-01 00:00:00 (Autoformer convention).
  5. Named value columns 0..6 and the last one `OT` (Autoformer convention).
  6. Values written unchanged (pandas default float repr).
  7. Verified the final CSV equals the Autoformer CSV from thuml/Time-Series-Library (HF, commit 2b66e59): same columns, same dates, values equal within 1e-9 (its date strings are 'YYYY/M/D H:MM' for exchange_rate).

## weather

- Final file: `weather.csv` — sha256 `ec3d3f996fa958fe8639e07fc5f30b1471b8c40c55220d3effe46ef8492d9cf9`
- Shape: **52696 rows × 22 columns** (incl. `date`)
- Date range: 2020-01-01 00:10:00 → 2021-01-01 00:00:00
- Download date: 2026-11-09
- Source URL(s):
  - `https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/weather/weather.csv`
  - `https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv`
- Raw file(s):
  - `raw/weather/weather.csv` (7235425 bytes) sha256 `34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63`
- Processing steps:
  1. Downloaded Autoformer/Time-Series-Library weather.csv from the official THUML HuggingFace mirror (thuml/Time-Series-Library, commit 2b66e59ee19dac8f6f19fb5d4997f289fdfea357).
  2. Verified exactly 52696 rows x 22 columns, first column `date`, last column `OT`, no NaN, chronological.
  3. Header decoded as latin-1 and re-written as UTF-8 (column names 'SWDR (W/m²)', 'PAR (µmol/m²/s)', 'max. PAR (µmol/m²/s)' are now valid UTF-8); numeric values unchanged.
  4. Only the encoding of the header differs from the raw file.

## PEMS04

- Final file: `PEMS04.csv` — sha256 `779d6afe2f7778bbffcb6510cc9c3a6cd0fc5e8b2a08c8aa1595e01a9ac57522`
- Shape: **16992 rows × 308 columns** (incl. `date`)
- Date range: 2018-01-01 00:00:00 → 2018-02-28 23:55:00  **(nominal dates — see steps)**
- Download date: 2026-11-09
- Source URL(s):
  - `https://huggingface.co/datasets/jimmygao3218/PEMS04/resolve/0d0e451662ac964505dd6fcea4a739f416b4ee25/PEMS04.npz`
  - `https://huggingface.co/datasets/Davidham3/pems-stsgcn/resolve/c4a988ceca7f330a2283db6aeddf765dd6eac63f/PEMS04/PEMS04_series.parquet`
- Raw file(s):
  - `raw/pems/jimmygao3218/PEMS04.npz` (32956284 bytes) sha256 `95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a`
  - `raw/pems/davidham3/PEMS04_series.parquet` (20354128 bytes) sha256 `39de0b3d83ece4543c9bd500e692dbe983cce5e6c1648dd7895b27222de2413d`
- Processing steps:
  1. Downloaded STSGCN-format `PEMS04.npz` from HuggingFace (jimmygao3218/PEMS04, pinned commit) — thuml/Time-Series-Library does not host PEMS04/PEMS08.
  2. Verified npz key `data` with shape (16992, 307, 3) [T, N, F], float64, no NaN.
  3. Kept feature index 0 (traffic flow) only; occupancy/speed discarded.
  4. Verified flow equals bit-for-bit the `flow` column of the STSGCN author's HF mirror Davidham3/pems-stsgcn (<name>_series.parquet, commit c4a988c, which documents verification against STSGCN GitHub commit 3f825f6).
  5. Added NOMINAL `date` column: 5-minute index starting 2018-01-01 00:00:00 (16992 steps = 59 days). The npz carries no timestamps; the start date follows the standard dataset description (Jan–Feb 2018) and is nominal, not verified against Caltrans PeMS.
  6. Named value columns 0..305 and the last sensor `OT` (Autoformer convention); values written unchanged.

## PEMS08

- Final file: `PEMS08.csv` — sha256 `a42123e43340ecfe67244a3f8494afcd62d8f2cd3f55defc0562c3e6b369e07f`
- Shape: **17856 rows × 171 columns** (incl. `date`)
- Date range: 2016-07-01 00:00:00 → 2016-08-31 23:55:00  **(nominal dates — see steps)**
- Download date: 2026-11-09
- Source URL(s):
  - `https://huggingface.co/datasets/jimmygao3218/PEMS08/resolve/7d5f30e2f1b538973ff805fc84ac42a0bd7422aa/PEMS08.npz`
  - `https://huggingface.co/datasets/Davidham3/pems-stsgcn/resolve/c4a988ceca7f330a2283db6aeddf765dd6eac63f/PEMS08/PEMS08_series.parquet`
- Raw file(s):
  - `raw/pems/jimmygao3218/PEMS08.npz` (18525419 bytes) sha256 `e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075`
  - `raw/pems/davidham3/PEMS08_series.parquet` (11956120 bytes) sha256 `7d9143f9271f1819217de56940959e8b14a23c23824dc6cb7cc4ef641976b1b8`
- Processing steps:
  1. Downloaded STSGCN-format `PEMS08.npz` from HuggingFace (jimmygao3218/PEMS08, pinned commit) — thuml/Time-Series-Library does not host PEMS04/PEMS08.
  2. Verified npz key `data` with shape (17856, 170, 3) [T, N, F], float64, no NaN.
  3. Kept feature index 0 (traffic flow) only; occupancy/speed discarded.
  4. Verified flow equals bit-for-bit the `flow` column of the STSGCN author's HF mirror Davidham3/pems-stsgcn (<name>_series.parquet, commit c4a988c, which documents verification against STSGCN GitHub commit 3f825f6).
  5. Added NOMINAL `date` column: 5-minute index starting 2016-07-01 00:00:00 (17856 steps = 62 days). The npz carries no timestamps; the start date follows the standard dataset description (Jul–Aug 2016) and is nominal, not verified against Caltrans PeMS.
  6. Named value columns 0..168 and the last sensor `OT` (Autoformer convention); values written unchanged.
