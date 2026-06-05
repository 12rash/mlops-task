"""
MLOps Batch Job - Rolling Mean Signal Pipeline
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import io


# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────

def setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to both file and stdout."""
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────
# Config loading + validation
# ──────────────────────────────────────────────

def load_config(config_path: str, logger: logging.Logger) -> dict:
    """Load and validate YAML config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping.")

    required_fields = {"seed": int, "window": int, "version": str}
    for field, expected_type in required_fields.items():
        if field not in config:
            raise KeyError(f"Missing required config field: '{field}'")
        if not isinstance(config[field], expected_type):
            raise TypeError(
                f"Config field '{field}' must be {expected_type.__name__}, "
                f"got {type(config[field]).__name__}"
            )

    if config["window"] < 1:
        raise ValueError(f"Config 'window' must be >= 1, got {config['window']}")

    logger.info(
        "Config loaded — version=%s | seed=%s | window=%s",
        config["version"], config["seed"], config["window"]
    )
    return config


# ──────────────────────────────────────────────
# Dataset loading + validation
# ──────────────────────────────────────────────

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load and validate the CSV dataset."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # First attempt: standard read
    try:
        df = pd.read_csv(path, quotechar='"')
    except Exception:
        # Fall back to a tolerant reparse below
        df = None

    # Handle edge-case where entire lines are quoted (one column with commas)
    if df is None or (len(df.columns) == 1 and df.columns[0].count(',') > 0):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_lines = f.read().splitlines()

            # Strip a single leading+trailing quote for each line when present
            cleaned = []
            for ln in raw_lines:
                if ln.startswith('"') and ln.endswith('"'):
                    cleaned.append(ln[1:-1])
                else:
                    cleaned.append(ln)

            content = "\n".join(cleaned)
            df = pd.read_csv(io.StringIO(content))
        except Exception as e:
            raise ValueError(f"Failed to parse CSV (after cleaning): {e}") from e

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Coerce close to numeric; flag if all NaN
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df["close"].isna().all():
        raise ValueError("Column 'close' contains no valid numeric values.")

    logger.info("Dataset loaded — rows=%d | columns=%s", len(df), list(df.columns))
    return df


# ──────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────

def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute rolling mean on 'close'.
    The first (window-1) rows produce NaN and are excluded from signal computation.
    """
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()
    nan_count = df["rolling_mean"].isna().sum()
    logger.info(
        "Rolling mean computed — window=%d | warm-up rows (NaN)=%d", window, nan_count
    )
    return df


def compute_signal(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Generate binary signal: 1 if close > rolling_mean, else 0.
    Rows where rolling_mean is NaN are excluded (signal = NaN, then dropped).
    """
    df = df.copy()
    valid = df["rolling_mean"].notna()
    df.loc[valid, "signal"] = (df.loc[valid, "close"] > df.loc[valid, "rolling_mean"]).astype(int)
    signal_rows = int(valid.sum())
    logger.info(
        "Signal generated — valid rows=%d | signal=1 count=%d | signal=0 count=%d",
        signal_rows,
        int(df["signal"].sum()),
        int(signal_rows - df["signal"].sum()),
    )
    return df


# ──────────────────────────────────────────────
# Metrics output
# ──────────────────────────────────────────────

def write_metrics(output_path: str, payload: dict, logger: logging.Logger) -> None:
    """Write metrics dict to JSON file and print to stdout."""
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Metrics written to %s", output_path)
    print(json.dumps(payload, indent=2))


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLOps Rolling Mean Signal Pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV")
    parser.add_argument("--config",   required=True, help="Path to YAML config")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file", help="Path for log file")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    start_time = time.time()
    logger.info("=== Job started ===")

    version = "unknown"  # fallback for error output before config is loaded

    try:
        # 1. Load + validate config
        config = load_config(args.config, logger)
        version = config["version"]
        seed    = config["seed"]
        window  = config["window"]

        # 2. Set random seed for reproducibility
        np.random.seed(seed)
        logger.debug("Random seed set to %d", seed)

        # 3. Load + validate dataset
        df = load_dataset(args.input, logger)

        # 4. Rolling mean
        logger.info("Computing rolling mean (window=%d)…", window)
        df = compute_rolling_mean(df, window, logger)

        # 5. Signal generation
        logger.info("Generating binary signal…")
        df = compute_signal(df, logger)

        # 6. Metrics
        valid_df   = df[df["rolling_mean"].notna()]
        rows_proc  = len(valid_df)
        signal_rate = round(float(valid_df["signal"].mean()), 4)
        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Metrics summary — rows_processed=%d | signal_rate=%.4f | latency_ms=%d",
            rows_proc, signal_rate, latency_ms
        )

        payload = {
            "version":        version,
            "rows_processed": rows_proc,
            "metric":         "signal_rate",
            "value":          signal_rate,
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }
        write_metrics(args.output, payload, logger)
        logger.info("=== Job completed successfully ===")
        sys.exit(0)

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        error_payload = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
        }
        try:
            write_metrics(args.output, error_payload, logger)
        except Exception as write_exc:
            logger.error("Could not write error metrics: %s", write_exc)
        sys.exit(1)


if __name__ == "__main__":
    main()