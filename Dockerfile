# Dockerfile (for Hugging Face Spaces)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (optional; uncomment if your requirements need them)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential curl && \
#     rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . /app

# Hugging Face sets $PORT (default 7860). Bind to it.
ENV PORT=7860
EXPOSE 7860

# Use uvicorn to serve FastAPI app defined in app.py as variable "app"
CMD ["bash", "-lc", "chmod +x /app/entrypoint.sh && /app/entrypoint.sh"]
