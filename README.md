🤖 Agente de IA para Service Desk (Imersão Alura + Google)
Este projeto, desenvolvido durante a Imersão Dev: Agentes de IA da Alura em parceria com o Google, consiste em um agente de IA inteligente para automatizar o atendimento de Service Desk, focado em políticas internas de uma empresa.

O agente utiliza o poder do Google Gemini e a flexibilidade do LangGraph para criar um fluxo de trabalho que não apenas responde perguntas, mas também toma decisões e executa ações com base na solicitação do usuário. A interação é feita através de uma interface de chat amigável construída com Streamlit.

✨ Funcionalidades Principais
Triagem Inteligente: O agente primeiro analisa a intenção do usuário, classificando a solicitação em AUTO_RESOLVER, PEDIR_INFO ou ABRIR_CHAMADO.

RAG (Retrieval-Augmented Generation): Para perguntas sobre políticas, o agente consulta uma base de conhecimento de documentos PDF, garantindo respostas precisas e fundamentadas.

Fluxo de Decisão com LangGraph: O agente opera com base em um grafo de estados, permitindo uma lógica complexa de fallback. Se a busca nos documentos falha, ele pode reavaliar a situação e decidir entre pedir mais informações ou abrir um chamado.

Interface de Chat Interativa: Uma interface web construída com Streamlit permite que os usuários interajam com o agente de forma natural.

🛠️ Tecnologias Utilizadas
Linguagem: Python

Modelos de IA: Google Gemini (via API)

Orquestração: LangChain & LangGraph

Busca Vetorial (RAG): FAISS & GoogleGenerativeAIEmbeddings

Interface Web: Streamlit

Manipulação de Documentos: PyMuPDFLoader

🚀 Como Executar o Projeto
Siga os passos abaixo para rodar o agente na sua máquina local.

Pré-requisitos
Python 3.9+

Uma chave de API do Google Gemini. Você pode obter uma no Google AI Studio.

Passos
Clone o repositório:

git clone [https://github.com/seu-usuario/nome-do-repositorio.git](https://github.com/seu-usuario/nome-do-repositorio.git)
cd nome-do-repositorio

Crie um ambiente virtual (recomendado):

python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

Instale as dependências:

pip install -r requirements.txt

Configure sua chave de API:

Renomeie o arquivo .env.example para .env.

Abra o arquivo .env e cole sua chave de API do Google:

GOOGLE_API_KEY="SUA_CHAVE_DE_API_AQUI"

Adicione os Documentos:

Crie uma pasta chamada docs na raiz do projeto.

Coloque os arquivos PDF que o agente deve usar como base de conhecimento dentro desta pasta.

Execute a aplicação Streamlit:

streamlit run app.py

Abra seu navegador e acesse o endereço http://localhost:8501.
