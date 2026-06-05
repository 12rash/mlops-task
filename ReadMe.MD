# MLOps Batch Job — Rolling Mean Signal Pipeline

A minimal, reproducible MLOps-style batch pipeline that computes a rolling-mean-based
binary trading signal from OHLCV data.

---

## Project Structure

```
mlops-task/
├── run.py           # Main pipeline script
├── config.yaml      # Job configuration (seed, window, version)
├── data.csv         # Input OHLCV dataset (10,000 rows)
├── requirements.txt # Python dependencies
├── Dockerfile       # Container definition
├── metrics.json     # Sample output from a successful run
├── run.log          # Sample log from a successful run
└── README.md        # This file
```

---

## Local Run

### Prerequisites
- Python 3.9+
- pip

### Setup
```bash
pip install -r requirements.txt
```

### Run
```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

---

## Docker Build & Run

```bash
# Build the image
docker build -t mlops-task .

# Run the container
docker run --rm mlops-task
```

The container:
- Bundles `data.csv` and `config.yaml` at build time
- Prints the final metrics JSON to stdout
- Writes `metrics.json` and `run.log` inside the container
- Exits with code `0` on success, non-zero on failure

---

## Configuration (`config.yaml`)

| Field     | Type   | Description                        |
|-----------|--------|------------------------------------|
| `seed`    | int    | NumPy random seed for reproducibility |
| `window`  | int    | Rolling mean window size           |
| `version` | string | Pipeline version tag               |

```yaml
seed: 42
window: 5
version: "v1"
```

---

## Signal Logic

1. **Rolling mean** — computed on `close` with the configured `window`.  
   The first `window - 1` rows produce `NaN` and are **excluded** from signal computation.
2. **Signal** — `1` if `close > rolling_mean`, otherwise `0`.

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 55,
  "seed": 42,
  "status": "success"
}
```

> `rows_processed` = 9996 because the first 4 rows (window - 1 = 4) are warm-up NaNs excluded from signal computation.

### Error output format
```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

---

## Validation & Error Handling

The pipeline validates:
- Config file exists and contains `seed`, `window`, `version` with correct types
- Input CSV file exists and is parseable
- Input is non-empty
- `close` column is present and contains valid numeric values

All errors are logged and written to `metrics.json` before exiting with code `1`.

---

## Reproducibility

- `numpy.random.seed(seed)` is set from config before any processing
- All logic is deterministic given the same `config.yaml` and `data.csv`
- Re-running produces identical `metrics.json` output