FROM python:3.13-slim

WORKDIR /app

# dependências nativas pro lxml e build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# porta que o Render detecta
ENV PORT=10000
EXPOSE 10000

CMD ["./start.sh"]
