import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. Carregamento Seguro da Chave de API ---

# Esta linha carrega as variáveis do seu arquivo .env para o ambiente
load_dotenv()

# Agora, pegamos a chave de API da variável de ambiente que acabamos de carregar
# O os.getenv() lê a variável "GOOGLE_API_KEY"
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, api_key=google_api_key)

resposta = llm.invoke("Quem é você? Seja criativo na resposta.")
print(resposta.content)

