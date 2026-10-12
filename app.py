import os
import re
import unicodedata
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel

# Inicializar a aplicação FastAPI
app = FastAPI(
    title="API de MLOps - Análise de Sentimentos",
    description="Serviço de inferência para classificação de reviews de e-commerce",
    version="1.0",
)

# Configuração de CORS (Essencial para permitir que a Vercel acesse o Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite pedidos de qualquer origem (ou ajuste para o seu domínio da Vercel)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar o pipeline/modelo treinado a partir do disco
model = joblib.load("modelo_sentimento_dsa_v1.joblib")


# Função de limpeza de texto idêntica à do treino
def dsa_limpa_texto(texto: str) -> str:
  if not isinstance(texto, str):
    return ""
  texto_sem_acentos = "".join(
      c
      for c in unicodedata.normalize("NFKD", texto)
      if unicodedata.category(c) != "Mn"
  )
  texto_limpo = texto_sem_acentos.lower()
  texto_limpo = re.sub(r"[^a-z\s]", "", texto_limpo)
  texto_limpo = re.sub(r"\s+", " ", texto_limpo).strip()
  return texto_limpo


# Estrutura de dados esperada no Payload
class ReviewRequest(BaseModel):
  reviews: list[str]


@app.get("/")
def home():
  return {"mensagem": "A API de MLOps está online e a funcionar com sucesso! 🚀"}


@app.post("/predict")
def predict(payload: ReviewRequest):
  textos_limpos = [dsa_limpa_texto(r) for r in payload.reviews]
  previsoes = model.predict(textos_limpos)
  sentimentos = ["Negativo" if p == 0 else "Positivo" for p in previsoes]

  resultados = []
  for review, sentimento in zip(payload.reviews, sentimentos):
    resultados.append({"review": review, "sentimento": sentimento})

  return {"resultados": resultados}


# Execução local ou compatível com nuvens que utilizam portas dinâmicas (como o Render)
if __name__ == "__main__":
  import uvicorn

  port = int(os.environ.get("PORT", 8000))
  uvicorn.run("app:app", host="0.0.0.0", port=port)
