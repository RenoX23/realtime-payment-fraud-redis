# Production Dockerfile for Real-Time Fraud Pipeline Microservice
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY src/ ./src/
COPY pipeline/ ./pipeline/
COPY dashboards/ ./dashboards/
COPY data/ ./data/
COPY models/ ./models/
COPY .env.example ./.env

EXPOSE 8000 8501

# Default command runs FastAPI service
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
