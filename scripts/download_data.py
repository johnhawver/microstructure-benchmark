from dotenv import load_dotenv
load_dotenv()
import os
import databento as db
import polars as pl
from src.data_io import write_parquet
from src.config import DATASET, SYMBOL, DATA_START, DATA_END, BUDGET_USD

client = db.Historical(os.environ["DATABENTO_API_KEY"])

def main():
    """Download data from Databento and save to Parquet files."""
    datasets = client.metadata.list_datasets()
    print(datasets)

    result = client.symbology.resolve(dataset="GLBX.MDP3", 
                            symbols=["MNQ.c.0"], 
                            stype_in="continuous", 
                            stype_out="instrument_id", 
                            start_date="2025-09-02", 
                            end_date="2025-09-03")
    print(result)

    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=["MNQ.c.0"],
        stype_in="continuous",
        schema="mbp-1",
        start="2025-09-02T13:30:00",
        end="2025-09-02T13:40:00",
    )
    print(f"Estimated cost: ${cost:.6f}")
    assert cost < 1.0, f"Cost ${cost:.6f} is not under $1 — aborting"

    for schema in ["mbp-1", "trades", "ohlcv-1m"]:
        print(schema, estimate_cost(DATA_START, DATA_END, schema))

    window_cost = estimate_cost(DATA_START, DATA_END, "mbp-1")
    if window_cost > BUDGET_USD:
        raise RuntimeError(f"MBP-1 window cost ${window_cost:.6f} is greater than budget ${BUDGET_USD:.2f} — aborting")
        
    df = download_mbp1(start="2025-09-02T13:30:00", end="2025-09-02T13:40:00")
    write_parquet(df, "data/parquet/mnq_mbp1_2025-09-02_smoke.parquet")


def download_mbp1(start: str, end: str, 
              symbol: str = "MNQ.c.0") -> pl.DataFrame:
    """Download an MBP-1 dataset from Databento and return a Polars DataFrame."""
    dbns = client.timeseries.get_range(dataset="GLBX.MDP3", 
                                        schema="mbp-1", symbols=[symbol], 
                                        stype_in="continuous", start=start, 
                                        end=end)
    
    df = dbns.to_df()
    polars_df = pl.from_pandas(df.reset_index())
    return polars_df

def estimate_cost(start: str, end: str, schema: str) -> float:
    """Return the estimated USD cost of an MNQ continuous request for the given schema."""
    return client.metadata.get_cost(dataset=DATASET, symbols=[SYMBOL], schema=schema, stype_in="continuous", start=start, end=end)


if __name__ == "__main__":
    main()