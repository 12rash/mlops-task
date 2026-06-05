FROM python:3.9-slim

WORKDIR /app

# Copy dependency file first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + data
COPY run.py .
COPY config.yaml .
COPY data.csv .

# Default command — runs the pipeline with fixed internal paths
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
