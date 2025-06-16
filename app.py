import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA

VECTORSTORE_DIR = "./chroma_store"
FILES_FOLDER = "./files"
USE_OPENAI = False  # Set to True to use OpenAI, False for local Ollama

st.title("📄 Local PDF/DOCX Chatbot")

# ---- Helper Functions ----

def load_and_split_files(file_paths):
    all_docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        all_docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(all_docs)

def build_vectorstore_from_files(folder, use_openai):
    file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".pdf", ".docx"))]
    docs = load_and_split_files(file_paths)
    if use_openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(docs, embedding=embeddings, persist_directory=VECTORSTORE_DIR)

def load_vectorstore_from_disk(directory, use_openai):
    if use_openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=directory, embedding_function=embeddings)

def build_vectorstore_from_uploaded(uploaded_file, use_openai):
    ext = uploaded_file.name.split('.')[-1].lower()
    temp_path = f'./temp_uploaded.{ext}'
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    docs = load_and_split_files([temp_path])
    if use_openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    os.remove(temp_path)
    return vectorstore

def get_llm(use_openai):
    if use_openai:
        return ChatOpenAI(model="gpt-3.5-turbo")
    else:
        return ChatOllama(model="llama3")

# ---- Vectorstore Initialization ----

def initialize_vectorstore():
    if os.path.exists(os.path.join(VECTORSTORE_DIR, 'index')) or os.path.exists(os.path.join(VECTORSTORE_DIR, 'chroma.sqlite3')):
        return load_vectorstore_from_disk(VECTORSTORE_DIR, USE_OPENAI)
    else:
        return build_vectorstore_from_files(FILES_FOLDER, USE_OPENAI)

if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = initialize_vectorstore()
    st.session_state['kb_files'] = set([f for f in os.listdir(FILES_FOLDER) if f.endswith(('.pdf', '.docx'))])

# ---- Main App ----

st.info("The chatbot is loaded with PDFs and DOCX files from the ./files folder.")

uploaded_file = st.file_uploader(
    "Or upload a PDF or DOCX to chat with (not added to knowledge base)",
    type=["pdf", "docx"]
)
question = st.text_input("Ask a question:")

# Handle uploaded file chat
if uploaded_file and question:
    if (
        'uploaded_vectorstore' not in st.session_state or
        st.session_state.get('last_uploaded') != uploaded_file.name
    ):
        st.session_state['uploaded_vectorstore'] = build_vectorstore_from_uploaded(uploaded_file, USE_OPENAI)
        st.session_state['last_uploaded'] = uploaded_file.name
    vectorstore = st.session_state['uploaded_vectorstore']
    llm = get_llm(USE_OPENAI)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    with st.spinner("Thinking..."):
        answer = qa.run(question)
    st.markdown(f"**Answer:** {answer}")

# Handle knowledge base chat
elif question:
    # Check if files have changed, and rebuild if so
    current_files = set([f for f in os.listdir(FILES_FOLDER) if f.endswith(('.pdf', '.docx'))])
    if current_files != st.session_state['kb_files']:
        with st.spinner("Updating knowledge base..."):
            st.session_state['vectorstore'] = build_vectorstore_from_files(FILES_FOLDER, USE_OPENAI)
            st.session_state['kb_files'] = current_files
    vectorstore = st.session_state['vectorstore']
    llm = get_llm(USE_OPENAI)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    with st.spinner("Thinking..."):
        answer = qa.run(question)
    st.markdown(f"**Answer:** {answer}")

st.caption(
    "Files in knowledge base: " +
    ", ".join(sorted([f for f in os.listdir(FILES_FOLDER) if f.endswith(('.pdf', '.docx'))]))
)
