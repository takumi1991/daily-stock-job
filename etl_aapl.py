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
    miss_rate=0.2,
    outlier_rate=0.01,
    dup_rate=0.01,
    case_rate=0.01,
    swap_rate=0.01,
    future_rows=5,
    future_max_days=30
) -> tuple[pd.DataFrame, dict]:
    x = df.copy()
    stats = {
        "missing": 0,
        "outliers": 0,
        "duplicates": 0,
        "case_jitter": 0,
        "low_gt_high": 0,
        "future_rows": 0,
    }

    n = len(x)
    if n == 0:
        return x, stats

    rng = np.random.default_rng(seed=int(pd.Timestamp.utcnow().timestamp()) % (2**32))
    idx_all = np.arange(n)

    # 1) 欠損
    miss_cols = ["open","high","low","close","volume"]
    k = max(1, int(n * miss_rate))
    miss_rows = rng.choice(idx_all, size=min(k, n), replace=False)
    for r in miss_rows:
        c = rng.choice(miss_cols)
        x.at[r, c] = np.nan
        stats["missing"] += 1

    # 2) 外れ値
    k = max(1, int(n * outlier_rate))
    out_rows = rng.choice(idx_all, size=min(k, n), replace=False)
    for r in out_rows:
        stats["outliers"] += 1
        mult = float(rng.choice([3, 4, 5, 1/3, 1/4, 1/5]))
        if pd.notna(x.at[r, "close"]): x.at[r, "close"] *= mult
        if pd.notna(x.at[r, "open"]):  x.at[r, "open"] *= mult
        if pd.notna(x.at[r, "high"]):  x.at[r, "high"] *= mult * 0.9
        if pd.notna(x.at[r, "low"]):   x.at[r, "low"]  *= mult * 1.1

    # 3) 重複
    k = max(1, int(n * dup_rate))
    dup_rows = rng.choice(idx_all, size=min(k, n), replace=False)
    if len(dup_rows) > 0:
        stats["duplicates"] = len(dup_rows)
        x = pd.concat([x, x.iloc[dup_rows].copy()], ignore_index=True)

    # 4) 表記ゆれ
    k = max(1, int(len(x) * case_rate))
    case_rows = rng.choice(np.arange(len(x)), size=min(k, len(x)), replace=False)
    def jitter_case(s: str) -> str:
        return "".join(ch.upper() if random.random()<0.5 else ch.lower() for ch in s)
    for r in case_rows:
        if isinstance(x.at[r,"symbol"], str):
            x.at[r,"symbol"] = jitter_case(x.at[r,"symbol"])
            stats["case_jitter"] += 1

    # 5) Low>High
    k = max(1, int(len(x) * swap_rate))
    swap_rows = rng.choice(np.arange(len(x)), size=min(k, len(x)), replace=False)
    for r in swap_rows:
        lo, hi = x.at[r,"low"], x.at[r,"high"]
        if pd.notna(lo) and pd.notna(hi):
            x.at[r,"low"], x.at[r,"high"] = hi+1e-6, lo-1e-6
            stats["low_gt_high"] += 1

    # 6) 未来日付
    if future_rows > 0:
        latest = x.sort_values("date").iloc[-1:].copy()
        futs=[]
        for i in range(future_rows):
            d=latest.copy()
            add_days=int(rng.integers(1,future_max_days+1))
            d["date"]=pd.to_datetime(d["date"])+pd.to_timedelta(add_days,"D")
            d["date"]=d["date"].dt.date
            futs.append(d)
        x=pd.concat([x]+futs,ignore_index=True)
        stats["future_rows"]=len(futs)

    # 型を整える
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.date
    for c in ["open","high","low","close"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
    if "volume" in x.columns:
        x["volume"] = pd.to_numeric(x["volume"], errors="coerce").astype("Int64")

    return x, stats

from datetime import datetime, timezone
from google.cloud import bigquery

LOG_TABLE = f"{PROJECT_ID}.{DATASET}.etl_dirty_log"

def ensure_log_table(client: bigquery.Client):
    """監査ログ用テーブルを作成（なければ）"""
    schema = [
        bigquery.SchemaField("run_ts", "TIMESTAMP"),
        bigquery.SchemaField("ticker", "STRING"),
        bigquery.SchemaField("rows_before", "INT64"),
        bigquery.SchemaField("rows_after", "INT64"),
        bigquery.SchemaField("missing", "INT64"),
        bigquery.SchemaField("outliers", "INT64"),
        bigquery.SchemaField("duplicates", "INT64"),
        bigquery.SchemaField("case_jitter", "INT64"),
        bigquery.SchemaField("low_gt_high", "INT64"),
        bigquery.SchemaField("future_rows", "INT64"),
        bigquery.SchemaField("github_run_id", "STRING"),
        bigquery.SchemaField("github_run_number", "STRING"),
        bigquery.SchemaField("note", "STRING"),
    ]
    try:
        client.create_table(bigquery.Table(LOG_TABLE, schema=schema))
        log("Created log table (first time).")
    except Exception:
        pass

def log_stats_to_bq(ticker: str, rows_before: int, rows_after: int, stats: dict, note: str = ""):
    """汚し件数を BigQuery に1行追記"""
    client = bigquery.Client(project=PROJECT_ID)
    ensure_dataset_table(client)   # データセットが無い場合に備える
    ensure_log_table(client)

    payload = pd.DataFrame([{
        "run_ts": datetime.now(timezone.utc),
        "ticker": ticker,
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "missing": int(stats.get("missing", 0)),
        "outliers": int(stats.get("outliers", 0)),
        "duplicates": int(stats.get("duplicates", 0)),
        "case_jitter": int(stats.get("case_jitter", 0)),
        "low_gt_high": int(stats.get("low_gt_high", 0)),
        "future_rows": int(stats.get("future_rows", 0)),
        # GitHub Actions から来るときは環境変数が入る（ローカル/Cloud Shell でも空でOK）
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "github_run_number": os.environ.get("GITHUB_RUN_NUMBER", ""),
        "note": note,
    }])

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",   # 追記
        create_disposition="CREATE_IF_NEEDED",
    )
    client.load_table_from_dataframe(payload, LOG_TABLE, job_config=job_config).result()
    log(f"Appended log row into {LOG_TABLE}")



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

from google.cloud import bigquery
from datetime import datetime, timezone

LOG_TABLE = "etl_run_log"  # 監査テーブル名

def save_run_log(
    client: bigquery.Client,
    project_id: str,
    dataset: str,
    stats: dict,
    *,
    symbol: str,
    rows_before: int,
    rows_after_dirty: int,
    source: str = "yfinance",
    run_id: str | None = None,
):
    """汚し件数などの監査ログを BQ に1行追記（サンドボックス対応：WRITE_APPEND）"""
    table_id = f"{project_id}.{dataset}.{LOG_TABLE}"

    # スキーマ定義（存在しなければ create + append）
    schema = [
        bigquery.SchemaField("run_ts", "TIMESTAMP"),
        bigquery.SchemaField("run_id", "STRING"),
        bigquery.SchemaField("source", "STRING"),
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("rows_before", "INT64"),
        bigquery.SchemaField("rows_after_dirty", "INT64"),
        bigquery.SchemaField("missing", "INT64"),
        bigquery.SchemaField("outliers", "INT64"),
        bigquery.SchemaField("duplicates", "INT64"),
        bigquery.SchemaField("case_jitter", "INT64"),
        bigquery.SchemaField("low_gt_high", "INT64"),
        bigquery.SchemaField("future_rows", "INT64"),
    ]
    try:
        client.create_table(bigquery.Table(table_id, schema=schema))
    except Exception:
        pass  # 既にあればスキップ

    row = [{
        "run_ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id or os.getenv("GITHUB_RUN_ID") or "",
        "source": source,
        "symbol": symbol,
        "rows_before": int(rows_before),
        "rows_after_dirty": int(rows_after_dirty),
        "missing": int(stats.get("missing", 0)),
        "outliers": int(stats.get("outliers", 0)),
        "duplicates": int(stats.get("duplicates", 0)),
        "case_jitter": int(stats.get("case_jitter", 0)),
        "low_gt_high": int(stats.get("low_gt_high", 0)),
        "future_rows": int(stats.get("future_rows", 0)),
    }]

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_json(row, table_id, job_config=job_config)
    job.result()
    print(f"[ETL] Logged run to {table_id}: {row[0]}")


if __name__ == "__main__":
    try:
        df = get_last_1y("AAPL")
        if df.empty:
            raise SystemExit("No data fetched from yfinance")

        rows_before = len(df)
        df_dirty, stats = make_dirty(df)  # ← 件数付きで受け取る
        log(f"Injected dirt: {stats}")

        # 保存（WRITE_TRUNCATEで本体テーブルを置換）
        load_upsert(df_dirty)

        # 監査ログを追記
        bq_client = bigquery.Client(project=PROJECT_ID)
        save_run_log(
            bq_client, PROJECT_ID, DATASET, stats,
            symbol="AAPL",
            rows_before=rows_before,
            rows_after_dirty=len(df_dirty),
        )

        log("DONE")
    except Exception as e:
        log("ERROR occurred:")
        traceback.print_exc()
        sys.exit(1)
