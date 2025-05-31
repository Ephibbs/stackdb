FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py .
COPY stackdb/ stackdb/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -e .

COPY api/requirements.txt ./api/
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ ./api/

RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
    
RUN mkdir -p /app/data && \
    chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

VOLUME ["/app/data"]

CMD ["python", "api/main.py", "--host", "0.0.0.0", "--port", "8000", "--persistence-path", "/app/data"] 