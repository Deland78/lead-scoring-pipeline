# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=5001

# Install system dependencies (including curl for healthcheck)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create models directory (artifacts should be volume-mounted in prod)
RUN mkdir -p models

COPY main.py .
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

# Do NOT copy model artifacts at build time; mount at runtime
# EXPOSE port and set healthcheck
EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/v2/health || exit 1

CMD ["/app/start.sh"]