"""Re-registers the best model in the MLflow Model Registry (for Docker compose).

Reads the pre-built evaluation CSV, picks the best model by recall@k, and
logs it as a pyfunc in the MLflow server (MLFLOW_TRACKING_URI env must point
to the server, e.g. http://mlflow:5000).

Prints are allowed in scripts/ per project conventions.
"""

import json

import pandas as pd

from src.config import get_settings
from src.pipeline.evaluate import _register


def main() -> None:
    """Registers the best model in the MLflow Model Registry."""
    settings = get_settings()
    csv_path = settings.models_dir / "evaluation" / "metrics_comparison.csv"
    df = pd.read_csv(csv_path).set_index("model")
    best = str(df["recall_at_k"].idxmax())
    with open(settings.processed_data_dir / "split_meta.json") as f:
        meta = json.load(f)
    _register(df, best, meta["dataset_hash"], settings)
    print(f"Registered '{best}' in MLflow Model Registry.")


if __name__ == "__main__":
    main()
