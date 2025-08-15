FROM python:3.13-slim

# deps básicas (e libs que o Chromium pode precisar)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Instala deps Python + Playwright + Chromium dentro da imagem
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m playwright install --with-deps chromium

COPY . .

ENV PORT=10000
# opcional: desligar o headless fallback se precisar
# ENV USE_PLAYWRIGHT=0

CMD ["bash", "start.sh"]
