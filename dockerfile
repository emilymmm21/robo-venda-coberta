# Usa imagem oficial do Python 3.11
FROM python:3.11-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . .

# Instala dependências
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Expõe a porta que o Render usará
EXPOSE 10000

# Comando para rodar o app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
