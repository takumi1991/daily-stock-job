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
    # staging作成 or 置換
    try: client.create_table(bigquery.Table(STAGING, schema=schema))
    except Exception: pass
    log("Loading into staging (truncate)...")
    client.load_table_from_dataframe(
        df, STAGING,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    ).result()
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
