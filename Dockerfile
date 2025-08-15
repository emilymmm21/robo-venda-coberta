FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deps de sistema (lxml + playwright)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# deps Python
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# instala o chromium do Playwright (com dependências do SO)
RUN python -m playwright install --with-deps chromium

# código
COPY . .

# porta padrão do Render
ENV PORT=10000

# healthcheck opcional no container local
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["bash", "start.sh"]
