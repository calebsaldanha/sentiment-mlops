FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY modelo_sentimento_dsa_v1.joblib .

# O Render define a porta dinamicamente via variável de ambiente PORT
EXPOSE 8000

# Executa o script python que lê a porta correta automaticamente
CMD ["python", "app.py"]
