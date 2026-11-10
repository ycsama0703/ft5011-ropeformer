"""Build finance CSVs in the benchmark format (date column + numeric channels) and score daily periodicity.

btc_hourly.csv: BTCUSDT 1h Binance spot klines 2019-2024.
  channels: log_volume = log(base-asset volume), log_trades = log(number of trades),
            abs_ret = |log close-to-close return| (hourly realised-volatility proxy)
epf_np.csv / epf_pjm.csv: day-ahead hourly electricity prices (epftoolbox EPF benchmark, Zenodo 4624805), price only.
"""
import csv, glob, math, sys
from datetime import datetime, timezone
from pathlib import Path

raw, out = Path(sys.argv[1]), Path(sys.argv[2])

rows = []
for f in sorted(glob.glob(str(raw / "btc" / "BTCUSDT-1h-*.csv"))):
    with open(f) as fh:
        for r in csv.reader(fh):
            if not r or not r[0].isdigit():
                continue
            ts = int(r[0])
            ts = ts // 1000 if ts > 10**14 else ts  # microsecond timestamps in newer files
            rows.append((ts, float(r[4]), float(r[5]), int(r[8])))
rows.sort()
dedup = {}
for r in rows:
    dedup[r[0]] = r
rows = [dedup[k] for k in sorted(dedup)]
gaps = sum(1 for a, b in zip(rows, rows[1:]) if b[0] - a[0] != 3_600_000)
with open(out / "btc_hourly.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["date", "log_volume", "log_trades", "abs_ret"])
    for prev, cur in zip(rows, rows[1:]):
        d = datetime.fromtimestamp(cur[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        w.writerow([d, f"{math.log(cur[2] + 1e-9):.6f}", f"{math.log(cur[3] + 1):.6f}",
                    f"{abs(math.log(cur[1] / prev[1])):.8f}"])
print(f"btc_hourly.csv: {len(rows) - 1} rows, {gaps} gaps (non-1h steps)")

for name in ("NP", "PJM"):
    with open(raw / "epf" / f"{name}.csv") as fh, open(out / f"epf_{name.lower()}.csv", "w", newline="") as g:
        r, w = csv.reader(fh), csv.writer(g)
        next(r)
        w.writerow(["date", "price"])
        n = 0
        for row in r:
            w.writerow([row[0], row[1].strip()]); n += 1
    print(f"epf_{name.lower()}.csv: {n} rows")
