FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "from transformers import AutoModelForTDT; print('Parakeet TDT ready')"

COPY . .

ENV VOICES_DIR=/app/voices
ENV MODELS_DIR=/app/models

EXPOSE 8883

CMD ["python", "serve.py"]
