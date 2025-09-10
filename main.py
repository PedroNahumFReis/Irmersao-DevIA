# -*- coding: utf-8 -*-
"""
Script para triagem de chamados de Service Desk utilizando a API Google Gemini e LangChain.
O script classifica as mensagens dos usuários em categorias para otimizar o atendimento.
"""

# 1. IMPORTS
# Organizando os imports: bibliotecas padrão primeiro, depois as de terceiros.
import os
from typing import List, Literal, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 2. CONSTANTES E CONFIGURAÇÃO
# Definir constantes em maiúsculas torna o código mais legível.

# O prompt principal que define a tarefa para o modelo de IA.
TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?"). A urgência é geralmente BAIXA.\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral"). A urgência é MEDIA.\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH."). A urgência geralmente é ALTA.\n'
    "Analise a mensagem e decida a ação mais apropriada."
)

# 3. SCHEMAS DE DADOS (PYDANTIC)
# Define a estrutura de dados esperada da resposta do modelo.

class TriagemOut(BaseModel):
    """Define o formato de saída estruturado para a triagem do Service Desk."""
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(
        default_factory=list,
        description="Campos ou informações que faltam na mensagem do usuário."
    )

# 4. FUNÇÃO PRINCIPAL
# Encapsula a lógica principal do script.

def main():
    """
    Função principal que carrega a configuração, executa a triagem
    em uma lista de testes e imprime os resultados.
    """
    print("Iniciando o script de triagem de Service Desk...")

    # Carregamento da chave de API
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("A chave de API do Google não foi encontrada. Verifique seu arquivo .env.")

    # Inicialização do modelo LLM (FEITA APENAS UMA VEZ)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, api_key=google_api_key)

    # Criação da "Chain" do LangChain com a saída estruturada
    triagem_chain = llm.with_structured_output(TriagemOut)

    # Lista de mensagens para testar a triagem
    testes = [
        "Posso reembolsar a internet?",
        "Quero mais 5 dias de trabalho remoto. Como faço?",
        "Posso reembolsar cursos ou treinamentos da Alura?",
        "Quantas capivaras tem no Rio Pinheiros?",
        "Preciso de ajuda.",
        "Favor abrir um chamado sobre o meu acesso."
    ]

    print("\n--- Iniciando Testes de Triagem ---\n")
    for msg_teste in testes:
        try:
            # Invoca a chain com o prompt do sistema e a mensagem do usuário
            resultado: TriagemOut = triagem_chain.invoke([
                SystemMessage(content=TRIAGEM_PROMPT),
                HumanMessage(content=msg_teste)
            ])
            # Converte o resultado Pydantic para um dicionário para impressão
            print(f"Pergunta: '{msg_teste}'\n -> Resposta: {resultado.model_dump()}\n")

        except Exception as e:
            print(f"Pergunta: '{msg_teste}'\n -> Ocorreu um erro: {e}\n")


# 5. BLOCO DE EXECUÇÃO
# Garante que a função main() seja chamada apenas quando o script for executado diretamente.
if __name__ == "__main__":
    main()