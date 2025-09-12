# -*- coding: utf-8 -*-
"""
Script final para triagem de chamados e consulta de políticas via RAG
utilizando a API Google Gemini e LangChain.
Versão corrigida para lidar com os limites da API (rate limits).
"""

# 1. IMPORTS
import os
import time  # Importado para adicionar delays e evitar rate limit
from typing import List, Literal, Dict
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 2. CONSTANTES E SCHEMAS

# Prompt para a tarefa de triagem
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

# Prompt para a tarefa de RAG
PROMPT_RAG = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda apenas 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

class TriagemOut(BaseModel):
    """Define o formato de saída estruturado para a triagem do Service Desk."""
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(
        default_factory=list,
        description="Campos ou informações que faltam na mensagem do usuário."
    )

# 3. FUNÇÕES AUXILIARES

def setup_rag_chain(llm: ChatGoogleGenerativeAI, embeddings: GoogleGenerativeAIEmbeddings, pdf_folder: str):
    """
    Carrega os PDFs, cria os chunks, gera embeddings de forma segura e monta a chain de RAG.
    Retorna o retriever e a document_chain.
    """
    print("\n--- Configurando o sistema de RAG (Consulta de Documentos) ---")
    docs = []
    for file in Path(pdf_folder).glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(file))
            docs.extend(loader.load())
            print(f"Carregado: {file}")
        except Exception as e:
            print(f"Erro ao carregar {file}: {e}")

    if not docs:
        print("Nenhum documento foi carregado. A função de RAG será desativada.")
        return None, None

    print(f"Total de documentos carregados: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    print(f"Documentos divididos em {len(chunks)} chunks.")

    # Em vez de criar o vectorstore de uma só vez, faremos isso em um loop
    # para evitar o rate limit do modelo de embedding.
    print("Iniciando a criação dos embeddings um por um para evitar rate limits...")
    vectorstore = None
    for i, chunk in enumerate(chunks):
        try:
            # Pega o texto e os metadados do chunk
            chunk_text = [chunk.page_content]
            chunk_metadata = [chunk.metadata]

            if vectorstore is None:
                # Cria o vectorstore com o primeiro chunk
                vectorstore = FAISS.from_texts(
                    texts=chunk_text,
                    embedding=embeddings,
                    metadatas=chunk_metadata
                )
            else:
                # Adiciona os chunks restantes ao vectorstore existente
                vectorstore.add_texts(
                    texts=chunk_text,
                    embedding=embeddings,
                    metadatas=chunk_metadata
                )
            print(f"  -> Embedding do chunk {i+1}/{len(chunks)} criado com sucesso.")
        except Exception as e:
            print(f"  -> Erro ao criar embedding para o chunk {i+1}: {e}")
        
        # Adiciona um pequeno delay para não sobrecarregar a API de embedding
        time.sleep(1)
    
    print("Criação dos embeddings finalizada.")

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4}
    )
    document_chain = create_stuff_documents_chain(llm, PROMPT_RAG)
    return retriever, document_chain


def perguntar_politica_rag(pergunta: str, retriever, document_chain) -> Dict:
    """Executa uma pergunta contra a chain de RAG."""
    docs_relacionados = retriever.invoke(pergunta)

    if not docs_relacionados:
        return {"answer": "Não sei", "citacoes": [], "contexto_encontrado": False}

    answer = document_chain.invoke({
        "input": pergunta,
        "context": docs_relacionados
    })

    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei", "citacoes": [], "contexto_encontrado": True}

    return {"answer": txt, "citacoes": docs_relacionados, "contexto_encontrado": True}

# 4. FUNÇÃO PRINCIPAL

def main():
    """
    Função principal que carrega a configuração, executa a triagem
    e os testes de RAG, imprimindo os resultados.
    """
    print("Iniciando o script...")

    # --- CONFIGURAÇÃO INICIAL (FEITA APENAS UMA VEZ) ---
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("A chave de API do Google não foi encontrada. Verifique seu arquivo .env.")

    # Inicializa o modelo LLM e os Embeddings uma única vez com os nomes corretos
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, api_key=google_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=google_api_key)

    # Lista de mensagens para testar ambas as funcionalidades
    testes = [
        "Posso reembolsar a internet?",
        "Quero mais 5 dias de trabalho remoto. Como faço?",
        "Posso reembolsar cursos ou treinamentos da Alura?",
        "Quantas capivaras tem no Rio Pinheiros?",
        "Preciso de ajuda.",
        "Favor abrir um chamado sobre o meu acesso."
    ]

    # --- PARTE 1: TESTES DE TRIAGEM ---
    print("\n--- Iniciando Testes de Triagem ---")
    triagem_chain = llm.with_structured_output(TriagemOut)
    for msg_teste in testes:
        try:
            resultado: TriagemOut = triagem_chain.invoke([
                SystemMessage(content=TRIAGEM_PROMPT),
                HumanMessage(content=msg_teste)
            ])
            print(f"Pergunta: '{msg_teste}'\n -> Resposta: {resultado.model_dump()}\n")
        except Exception as e:
            print(f"Pergunta: '{msg_teste}'\n -> Ocorreu um erro: {e}\n")
        
        # Espera 6 segundos para evitar o rate limit do modelo de CHAT
        print("Aguardando 6 segundos para evitar o rate limit...")
        time.sleep(6)


    # --- PARTE 2: TESTES DE RAG (CONSULTA DE DOCUMENTOS) ---
    retriever, document_chain = setup_rag_chain(llm, embeddings, "docs")
    
    if retriever and document_chain:
        print("\n--- Iniciando Testes de RAG (Consulta de Documentos) ---")
        for msg_teste in testes:
            try:
                resposta = perguntar_politica_rag(msg_teste, retriever, document_chain)
                print(f"Pergunta: '{msg_teste}'\n -> Resposta: {resposta['answer']}")
                if resposta.get("citacoes"):
                    print("Citações encontradas.")
                print("\n---\n")
            except Exception as e:
                print(f"Pergunta: '{msg_teste}'\n -> Ocorreu um erro: {e}\n")
            
            # Espera 6 segundos para evitar o rate limit do modelo de CHAT
            print("Aguardando 6 segundos para evitar o rate limit...")
            time.sleep(6)
    
    print("Script finalizado.")


# 5. BLOCO DE EXECUÇÃO
if __name__ == "__main__":
    main()