# usa Python 3.13 mesmo, sem compilar pandas (porque subimos pra 2.2.3)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# instala deps do sistema só se precisar (comente se não precisar de nada nativo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# ajuste o módulo se o seu app não for main:app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
