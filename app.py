import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain

VECTORSTORE_DIR = "./chroma_store"
FILES_FOLDER = "./files"
USE_OPENAI = False  # Set to True to use OpenAI, False for local Ollama

# st.set_page_config(layout="wide") 
st.title("💬 Chat with Your PDFs/DOCX")
st.markdown("""
<style>
.user-message {
    background-color: #d0eaff;
    color: #003366;
    padding: 10px 16px;
    border-radius: 16px;
    margin-bottom: 8px;
    margin-left: 30%;
    margin-right: 0;
    text-align: right;
    width: fit-content;
    float: right;
    clear: both;
}
.bot-message {
    background-color: #f2f2f2;
    color: #222;
    padding: 10px 16px;
    border-radius: 16px;
    margin-bottom: 8px;
    margin-right: 30%;
    margin-left: 0;
    text-align: left;
    width: fit-content;
    float: left;
    clear: both;
}
</style>
""", unsafe_allow_html=True)


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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
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

def initialize_vectorstore():
    if os.path.exists(os.path.join(VECTORSTORE_DIR, 'index')) or os.path.exists(os.path.join(VECTORSTORE_DIR, 'chroma.sqlite3')):
        return load_vectorstore_from_disk(VECTORSTORE_DIR, USE_OPENAI)
    else:
        return build_vectorstore_from_files(FILES_FOLDER, USE_OPENAI)

# ---- Session State Initialization ----

if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = initialize_vectorstore()
    st.session_state['kb_files'] = set([f for f in os.listdir(FILES_FOLDER) if f.endswith(('.pdf', '.docx'))])

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'uploaded_vectorstore' not in st.session_state:
    st.session_state['uploaded_vectorstore'] = None
    st.session_state['last_uploaded'] = None
if 'upload_chat_history' not in st.session_state:
    st.session_state['upload_chat_history'] = []

# ---- UI ----

st.info("The chatbot is loaded with PDFs and DOCX files from the ./files folder.")

tab1, tab2 = st.tabs(["Knowledge Base Chat", "Chat with Uploaded File"])

# ---- Knowledge Base Chat ----

with tab1:
    st.markdown("#### Chat with the Knowledge Base (files in ./files)")
    # Check if files have changed, and rebuild if so
    current_files = set([f for f in os.listdir(FILES_FOLDER) if f.endswith(('.pdf', '.docx'))])
    if current_files != st.session_state['kb_files']:
        with st.spinner("Updating knowledge base..."):
            st.session_state['vectorstore'] = build_vectorstore_from_files(FILES_FOLDER, USE_OPENAI)
            st.session_state['kb_files'] = current_files

    # Display chat history
    for i, (user, bot) in enumerate(st.session_state['chat_history']):
        st.markdown(f'<div class="user-message">{user}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-message">{bot}</div>', unsafe_allow_html=True)


    user_input = st.text_input("Your question:", key="kb_input", placeholder="Ask about your knowledge base...")
    if st.button("Send", key="kb_send") and user_input.strip():
        llm = get_llm(USE_OPENAI)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            st.session_state['vectorstore'].as_retriever(search_kwargs={"k": 8}),
            return_source_documents=False
        )

        with st.spinner("Thinking..."):
            result = qa_chain(
                {"question": user_input, "chat_history": st.session_state['chat_history']}
            )
        answer = result["answer"]
        st.session_state['chat_history'].append((user_input, answer))
        st.rerun()

    st.caption(
        "Files in knowledge base: " +
        ", ".join(sorted([f for f in os.listdir(FILES_FOLDER) if f.endswith(('.pdf', '.docx'))]))
    )

# ---- Uploaded File Chat ----

with tab2:
    st.markdown("#### Chat with a Single Uploaded File (not added to knowledge base)")
    uploaded_file = st.file_uploader(
        "Upload a PDF or DOCX",
        type=["pdf", "docx"],
        key="upload_file"
    )

    if uploaded_file:
        if (
            st.session_state['uploaded_vectorstore'] is None or
            st.session_state['last_uploaded'] != uploaded_file.name
        ):
            with st.spinner("Processing uploaded file..."):
                st.session_state['uploaded_vectorstore'] = build_vectorstore_from_uploaded(uploaded_file, USE_OPENAI)
                st.session_state['last_uploaded'] = uploaded_file.name
                st.session_state['upload_chat_history'] = []

        # Display chat history
        for i, (user, bot) in enumerate(st.session_state['chat_history']):
            st.markdown(f'<div class="user-message">{user}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message">{bot}</div>', unsafe_allow_html=True)



        upload_input = st.text_input(
            "Your question (uploaded file):",
            key="upload_input",
            placeholder="Ask about your uploaded document..."
        )
        if st.button("Send", key="upload_send") and upload_input.strip():
            llm = get_llm(USE_OPENAI)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                st.session_state['uploaded_vectorstore'].as_retriever(),
                return_source_documents=False
            )
            with st.spinner("Thinking..."):
                result = qa_chain(
                    {"question": upload_input, "chat_history": st.session_state['upload_chat_history']}
                )
            answer = result["answer"]
            st.session_state['upload_chat_history'].append((upload_input, answer))
            st.rerun()
    else:
        st.info("Upload a file to start chatting.")

