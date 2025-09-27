import os, sys, traceback
import pandas as pd
import yfinance as yf
from google.cloud import bigquery

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "stock-data-portfolio")
DATASET = "us_stock"
TABLE   = "daily_prices"
TARGET  = f"{PROJECT_ID}.{DATASET}.{TABLE}"
STAGING = f"{TARGET}__staging"

def log(msg): print(f"[ETL] {msg}", flush=True)

def get_last_1y(ticker: str) -> pd.DataFrame:
    log(f"Fetching {ticker} 1y...")
    t = yf.Ticker(ticker)
    df = t.history(period="1y", auto_adjust=False).reset_index()
    if df.empty:
        log("WARN: yfinance returned empty frame")
        return df
    drop_cols = [c for c in ["Dividends","Stock Splits","Adj Close"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.rename(columns={
        "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    })
    df["symbol"] = ticker
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for c in ["open","high","low","close"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    if "volume" in df.columns: df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    df = df[["symbol","date","open","high","low","close","volume"]]
    log(f"Fetched rows: {len(df)}; head: {df.head(2).to_dict(orient='records')}")
    return df

import numpy as np
import random

def make_dirty(
    df: pd.DataFrame,
    *,
    miss_rate=0.01,         # 欠損（1%）
    outlier_rate=0.01,      # 外れ値（1%）
    dup_rate=0.01,          # 重複（1%）
    case_rate=0.05,         # 表記ゆれ（5%）
    swap_rate=0.01,         # Low>High 不整合（1%）
    future_rows=5,          # 未来日付の注入行数
    future_max_days=30      # 未来に最大+30日
) -> pd.DataFrame:
    x = df.copy()

    n = len(x)
    if n == 0:
        return x

    rng = np.random.default_rng(seed=int(pd.Timestamp.utcnow().timestamp()) % (2**32))
    idx_all = np.arange(n)

    # 1) 欠損（Open/High/Low/Close/VolumeのどれかをNaNに）
    miss_cols = ["open","high","low","close","volume"]
    k = max(1, int(n * miss_rate))
    miss_rows = rng.choice(idx_all, size=min(k, n), replace=False)
    for r in miss_rows:
        c = rng.choice(miss_cols)
        x.at[r, c] = np.nan

    # 2) 外れ値（急騰 or 急落：closeを±3〜5倍、high/lowも崩す）
    k = max(1, int(n * outlier_rate))
    out_rows = rng.choice(idx_all, size=min(k, n), replace=False)
    for r in out_rows:
        mult = float(rng.choice([3, 4, 5, 1/3, 1/4, 1/5]))
        if pd.notna(x.at[r, "close"]):
            x.at[r, "close"] = x.at[r, "close"] * mult
        if pd.notna(x.at[r, "open"]):
            x.at[r, "open"] = x.at[r, "open"] * mult
        # high/low を雑にずらす（不自然さを残すため）
        if pd.notna(x.at[r, "high"]):
            x.at[r, "high"] = x.at[r, "high"] * (mult * 0.9)
        if pd.notna(x.at[r, "low"]):
            x.at[r, "low"] = x.at[r, "low"] * (mult * 1.1)

    # 3) 重複（ランダム行をそのまま複製して末尾に追加）
    k = max(1, int(n * dup_rate))
    dup_rows = rng.choice(idx_all, size=min(k, n), replace=False)
    x = pd.concat([x, x.iloc[dup_rows].copy()], ignore_index=True)

    # 4) 表記ゆれ（symbol を一部だけランダムな大小混在に）
    k = max(1, int(len(x) * case_rate))
    case_rows = rng.choice(np.arange(len(x)), size=min(k, len(x)), replace=False)
    def jitter_case(s: str) -> str:
        # ランダムに大小を混ぜる
        return "".join(ch.upper() if random.random() < 0.5 else ch.lower() for ch in s)
    for r in case_rows:
        if isinstance(x.at[r, "symbol"], str):
            x.at[r, "symbol"] = jitter_case(x.at[r, "symbol"])

    # 5) 不整合（Low > High を意図的に発生）
    k = max(1, int(len(x) * swap_rate))
    swap_rows = rng.choice(np.arange(len(x)), size=min(k, len(x)), replace=False)
    for r in swap_rows:
        lo, hi = x.at[r, "low"], x.at[r, "high"]
        if pd.notna(lo) and pd.notna(hi):
            # low を少し持ち上げ、high を少し下げて逆転させる
            x.at[r, "low"]  = lo * 1.05
            x.at[r, "high"] = hi * 0.95
            # 逆転していなければ強制スワップ
            if pd.notna(x.at[r, "low"]) and pd.notna(x.at[r, "high"]) and x.at[r, "low"] <= x.at[r, "high"]:
                x.at[r, "low"], x.at[r, "high"] = x.at[r, "high"] + 1e-6, x.at[r, "low"] - 1e-6

    # 6) 未来日付（直近行をコピーして未来日に）
    if future_rows > 0:
        latest = x.sort_values("date").iloc[-1:].copy()
        futs = []
        for i in range(future_rows):
            d = latest.copy()
            add_days = int(rng.integers(1, future_max_days+1))
            d["date"] = pd.to_datetime(d["date"]) + pd.to_timedelta(add_days, unit="D")
            d["date"] = d["date"].dt.date
            futs.append(d)
        x = pd.concat([x] + futs, ignore_index=True)

    # 型崩れの最終調整（BigQueryに入るように）
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.date
    for c in ["open","high","low","close"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
    if "volume" in x.columns:
        # pandas Int64 は NA サポート
        x["volume"] = pd.to_numeric(x["volume"], errors="coerce").astype("Int64")

    return x



def ensure_dataset_table(client: bigquery.Client):
    log("Ensuring dataset/table exist...")
    # データセット作成（あればスキップ）
    ds_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET}")
    try: client.create_dataset(ds_ref, exists_ok=True)
    except Exception as e: log(f"Dataset ensure warn: {e}")
    # テーブル作成（あればスキップ）
    schema = [
        bigquery.SchemaField("symbol","STRING"),
        bigquery.SchemaField("date","DATE"),
        bigquery.SchemaField("open","FLOAT"),
        bigquery.SchemaField("high","FLOAT"),
        bigquery.SchemaField("low","FLOAT"),
        bigquery.SchemaField("close","FLOAT"),
        bigquery.SchemaField("volume","INT64"),
    ]
    try:
        client.create_table(bigquery.Table(TARGET, schema=schema))
        log("Created table (first time).")
    except Exception:
        pass

def load_upsert(df: pd.DataFrame):
    client = bigquery.Client(project=PROJECT_ID)
    ensure_dataset_table(client)
    schema = [
        bigquery.SchemaField("symbol","STRING"),
        bigquery.SchemaField("date","DATE"),
        bigquery.SchemaField("open","FLOAT"),
        bigquery.SchemaField("high","FLOAT"),
        bigquery.SchemaField("low","FLOAT"),
        bigquery.SchemaField("close","FLOAT"),
        bigquery.SchemaField("volume","INT64"),
    ]

    # 本番テーブルに直接上書きロード（MERGE禁止のため）
    log("Loading directly into target (truncate)...")
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=schema
    )
    client.load_table_from_dataframe(df, TARGET, job_config=job_config).result()
    log(f"Replaced table with {len(df)} rows")

if __name__ == "__main__":
    try:
        df = get_last_1y("AAPL")
        if df.empty:
            raise SystemExit("No data fetched from yfinance")
        load_upsert(df)
        log("DONE")
    except Exception as e:
        log("ERROR occurred:")
        traceback.print_exc()
        sys.exit(1)
