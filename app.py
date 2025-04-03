import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import chromadb

# Insira sua chave da API do Gemini aqui (melhor usar segredos do Streamlit Cloud - veja a observação abaixo)
os.environ["GOOGLE_API_KEY"] = "SUA_CHAVE_DA_API"

# Inicialização (executado apenas uma vez por sessão)
@st.cache_resource
def load_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    return llm

@st.cache_resource
def load_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Certifique-se que o path para a pasta chroma_db esteja correto
    persist_directory = "./chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    db = client.get_collection("langchain")
    retriever = db.as_retriever()
    return retriever

llm = load_llm()
retriever = load_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

st.title("Chatbot da Pró-Reitoria de Extensão - IFCE")
st.write("Faça suas perguntas sobre os documentos da PROEXT.")

# Inicialização do estado da sessão para as mensagens do chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar hoje?"}]

# Exibição das mensagens do chat
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Entrada de texto do usuário
if prompt := st.chat_input():
    # Adiciona a mensagem do usuário ao histórico
    st.session_state["messages"].append({"role": "user", "content": prompt})
    # Exibe a mensagem do usuário na interface
    st.chat_message("user").write(prompt)

    # Obtém a resposta do chatbot
    result = qa.run(prompt)
    # Adiciona a resposta do chatbot ao histórico
    st.session_state["messages"].append({"role": "assistant", "content": result})
    # Exibe a resposta do chatbot na interface
    st.chat_message("assistant").write(result)