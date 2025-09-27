import os
import pandas as pd
import yfinance as yf
from google.cloud import bigquery

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "stock-data-portfolio")
DATASET = "us_stock"
TABLE   = "daily_prices"

def get_last_1y(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(period="1y", auto_adjust=False).reset_index()
    df = df.drop(columns=[c for c in ["Dividends","Stock Splits","Adj Close"] if c in df.columns], errors="ignore")
    df = df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df["symbol"] = ticker
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for c in ["open","high","low","close"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    if "volume" in df.columns: df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    return df[["symbol","date","open","high","low","close","volume"]]

def load_to_bq(df: pd.DataFrame):
    client = bigquery.Client(project=PROJECT_ID)
    target = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    staging = f"{target}__staging"

    schema = [
        bigquery.SchemaField("symbol","STRING"),
        bigquery.SchemaField("date","DATE"),
        bigquery.SchemaField("open","FLOAT"),
        bigquery.SchemaField("high","FLOAT"),
        bigquery.SchemaField("low","FLOAT"),
        bigquery.SchemaField("close","FLOAT"),
        bigquery.SchemaField("volume","INT64"),
    ]

    # ステージングテーブルを作成（なければ）
    try:
        client.create_table(bigquery.Table(staging, schema=schema))
    except Exception:
        pass

    # ステージングを毎回リセットしてロード
    client.load_table_from_dataframe(
        df, staging,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    ).result()

    # 本番テーブルへ MERGE（UPSERT）
    client.query(f"""
      MERGE `{target}` T
      USING `{staging}` S
      ON T.symbol=S.symbol AND T.date=S.date
      WHEN MATCHED THEN UPDATE SET
        open=S.open, high=S.high, low=S.low, close=S.close, volume=S.volume
      WHEN NOT MATCHED THEN
        INSERT (symbol,date,open,high,low,close,volume)
        VALUES (S.symbol,S.date,S.open,S.high,S.low,S.close,S.volume)
    """).result()

    print(f"Upserted {len(df)} rows into {target}")

if __name__ == "__main__":
    d = get_last_1y("AAPL")
    if d.empty: raise SystemExit("No data fetched")
    load_to_bq(d)
