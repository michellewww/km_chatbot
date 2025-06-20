import os
import csv
import streamlit as st
import pandas as pd
from docx import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangchainDocument

VECTORSTORE_DIR = "./chroma_store"
FILES_FOLDER = "./Shared/Marketing/Project Descriptions/TECH 2B Long Form"
SUMMARY_DOCX = "./Shared/Marketing/Project Descriptions/K&M TECH-2B Short Form Compendium.docx"
SUMMARY_CSV = "./Shared/Marketing/Project Descriptions/short_form_compendium.csv"
USE_OPENAI = False  # Set to True to use OpenAI, False for local Ollama

st.title("Chat with Your Project DOCX/PDFs (With Shortlist & Summaries!)")
st.caption("Files in: Shared/Marketing/Project Descriptions/TECH 2B Long Form")

# ---- 1. Extract summary table from DOCX to CSV (if not exists) ----
def docx_table_to_csv(docx_path, csv_path):
    doc = Document(docx_path)
    table = doc.tables[0]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in table.rows:
            writer.writerow([cell.text.strip() for cell in row.cells])

if not os.path.exists(SUMMARY_CSV):
    docx_table_to_csv(SUMMARY_DOCX, SUMMARY_CSV)

# ---- 2. Load summary CSV into DataFrame and dict ----
summary_df = pd.read_csv(SUMMARY_CSV)
id_column = summary_df.columns[0]  # "Job #" usually

# Build summary_dict: {job_id: [row1, row2, ...]}
summary_dict = {}
for _, row in summary_df.iterrows():
    job_id = str(row[id_column]).strip()
    if not job_id or job_id == "nan":
        continue
    summary_dict.setdefault(job_id, []).append(row.to_dict())

# ---- 3. Helper: Find files for each job_id ----
def get_files_for_job_id(job_id):
    files = []
    for fname in os.listdir(FILES_FOLDER):
        if fname.startswith(str(job_id)) and fname.lower().endswith(('.pdf', '.docx')):
            files.append(os.path.join(FILES_FOLDER, fname))
    return files

# ---- 4. Build vectorstore ONLY for files matching IDs in summary ----
def build_vectorstore_from_shortlist(use_openai=USE_OPENAI):
    shortlist_ids = set(summary_dict.keys())
    matched_files = []
    for job_id in shortlist_ids:
        matched_files.extend(get_files_for_job_id(job_id))
    matched_files = list(set(matched_files))
    progress_bar = st.progress(0, text="Starting file processing...")
    all_chunks = []
    total_files = len(matched_files)
    for idx, path in enumerate(matched_files):
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        chunks = splitter.split_documents(docs)
        filename = os.path.basename(path)
        job_id = filename.split()[0].split("_")[0]
        for chunk in chunks:
            chunk.metadata = {
                **chunk.metadata,
                "filename": filename,
                "job_id": job_id
            }
        all_chunks.extend(chunks)
        progress_bar.progress((idx + 1) / total_files, text=f"Processing {filename} ({idx+1}/{total_files})")
    
    # Also add CSV data to vectorstore for better retrieval
    for job_id, rows in summary_dict.items():
        for row in rows:
            text = "\n".join([f"{col}: {row[col]}" for col in summary_df.columns])
            csv_doc = LangchainDocument(
                page_content=text,
                metadata={
                    "source": "summary_csv",
                    "job_id": job_id,
                    "filename": "short_form_compendium.csv"
                }
            )
            all_chunks.append(csv_doc)
    
    if use_openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    progress_bar.progress(1.0, text="Building vectorstore...")
    vs = Chroma.from_documents(all_chunks, embedding=embeddings, persist_directory=VECTORSTORE_DIR)
    progress_bar.empty()
    return vs

def load_vectorstore_from_disk(directory, use_openai=USE_OPENAI):
    if use_openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=directory, embedding_function=embeddings)

def get_llm(use_openai=USE_OPENAI):
    if use_openai:
        return ChatOpenAI(model="gpt-3.5-turbo")
    else:
        return ChatOllama(model="llama3")

def initialize_vectorstore():
    if os.path.exists(os.path.join(VECTORSTORE_DIR, 'index')) or os.path.exists(os.path.join(VECTORSTORE_DIR, 'chroma.sqlite3')):
        return load_vectorstore_from_disk(VECTORSTORE_DIR, USE_OPENAI)
    else:
        return build_vectorstore_from_shortlist(USE_OPENAI)

# ---- 5. Find relevant CSV rows for a query ----
def find_relevant_csv_rows(query, threshold=0.2):
    """Find CSV rows relevant to the query using semantic search"""
    # First try exact job ID match if query contains a number
    import re
    potential_job_ids = re.findall(r'\b\d+\b', query)
    direct_matches = []
    
    for job_id in potential_job_ids:
        if job_id in summary_dict:
            direct_matches.extend(summary_dict[job_id])
    
    if direct_matches:
        return direct_matches, [job_id for job_id in potential_job_ids if job_id in summary_dict]
    
    # Otherwise use the vectorstore to find semantic matches
    vs = st.session_state['vectorstore']
    results = vs.similarity_search_with_score(
        query, 
        k=10,
        filter={"source": "summary_csv"}
    )
    
    relevant_rows = []
    relevant_ids = set()
    
    for doc, score in results:
        if score < threshold:  # Lower score is better in similarity search
            job_id = doc.metadata.get('job_id')
            if job_id and job_id in summary_dict:
                relevant_ids.add(job_id)
                relevant_rows.extend(summary_dict[job_id])
    
    # If no good matches, return empty
    if not relevant_rows:
        # Fall back to keyword matching as a last resort
        for job_id, rows in summary_dict.items():
            for row in rows:
                row_text = " ".join([str(v) for v in row.values()]).lower()
                if any(word.lower() in row_text for word in query.lower().split() if len(word) > 3):
                    relevant_ids.add(job_id)
                    relevant_rows.append(row)
    
    return relevant_rows, list(relevant_ids)

# ---- Custom prompt that prioritizes CSV data ----
csv_prompt_template = """
You are an assistant that helps users find information about technical projects.

First, examine the CSV data provided below to see if it contains the answer to the user's question:

CSV DATA:
{csv_context}

If the CSV data doesn't fully answer the question, use the following additional context from project documents:

DOCUMENT CONTEXT:
{context}

Answer the user's question based on the information above. Be specific and detailed.
If the information is in the CSV data, make sure to mention that explicitly.
If you need to use information from the documents, cite which files you referenced.

Question: {question}
Previous conversation: {chat_history}

Answer:
"""

# ---- Session State Initialization ----
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = initialize_vectorstore()
    st.session_state['kb_files'] = set([f for f in os.listdir(FILES_FOLDER) if f.endswith(('.pdf', '.docx'))])

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# ---- UI ----
st.info("The chatbot is loaded with shortlisted PDFs and DOCX files only.")
st.markdown("Chat with the Knowledge Base (shortlisted files only)")

# Add Update Database button
if st.button("Update Database"):
    with st.spinner("Updating knowledge base from shortlist..."):
        st.session_state['vectorstore'] = build_vectorstore_from_shortlist(USE_OPENAI)
        st.session_state['kb_files'] = set([f for f in os.listdir(FILES_FOLDER) if f.endswith(('.pdf', '.docx'))])
    st.success("Knowledge base updated!")

# Display chat history
for i, (user, bot) in enumerate(st.session_state['chat_history']):
    st.text(f"User: {user}")
    st.text(f"Bot: {bot}")

user_input = st.text_input("Your question:", key="kb_input", placeholder="Ask about your knowledge base...")
if st.button("Send", key="kb_send") and user_input.strip():
    # 1. First check CSV data for relevant information
    relevant_rows, relevant_ids = find_relevant_csv_rows(user_input)
    
    # 2. Format CSV data as structured context
    csv_context = ""
    if relevant_rows:
        csv_context = "Relevant project information:\n\n"
        for i, row in enumerate(relevant_rows[:5]):  # Limit to top 5 most relevant rows
            csv_context += f"Project {i+1}:\n"
            for col in summary_df.columns:
                if str(row[col]) != "nan" and str(row[col]).strip():
                    csv_context += f"- {col}: {row[col]}\n"
            csv_context += "\n"
    
    # 3. Set up document retriever with relevant job IDs
    retriever = st.session_state['vectorstore'].as_retriever(
        search_kwargs={
            "k": 8,  # Reduced from 12 to focus on most relevant
            "filter": {"job_id": {"$in": relevant_ids}} if relevant_ids else {}
        }
    )
    
    # 4. Set up custom prompt that prioritizes CSV data
    prompt = PromptTemplate(
        template=csv_prompt_template,
        input_variables=["csv_context", "context", "question", "chat_history"]
    )
    
    # 5. Set up QA chain with custom prompt
    llm = get_llm(USE_OPENAI)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    # 6. Process the query
    with st.spinner("Thinking..."):
        result = qa_chain(
            {
                "question": user_input,
                "csv_context": csv_context,
                "chat_history": st.session_state['chat_history']
            }
        )
    answer = result["answer"]

    # 7. Gather all unique file names actually used in the answer
    referenced_files = set()
    for doc in result.get("source_documents", []):
        filename = doc.metadata.get('filename', '')
        if filename and filename != "short_form_compendium.csv":
            referenced_files.add(filename)
    
    # 8. Compose the full response
    full_response = answer
    if referenced_files:
        full_response += "\n\nReference files:\n" + "\n".join(sorted(referenced_files))

    st.session_state['chat_history'].append((user_input, full_response))
    st.rerun()

st.caption(
    "Shortlisted files in knowledge base: " +
    ", ".join(sorted([f for f in os.listdir(FILES_FOLDER) if f.endswith(('.pdf', '.docx'))]))
)
