import os
import json
import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangchainDocument
import re
from typing import List, Dict, Any

VECTORSTORE_DIR = "./chroma_store"
PROJECT_JSON = "./Shared/Marketing/Project Descriptions/combined_project_data.json"
USE_OPENAI = False  # Set to True to use OpenAI, False for local Ollama

st.title("K&M Project Knowledge Assistant")
st.caption("Powered by comprehensive project database")

# ---- 1. Load and process JSON project data ----
@st.cache_data
def load_project_data():
    """Load and process the combined project data JSON"""
    try:
        with open(PROJECT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading project data: {e}")
        return None

def create_project_search_index(projects):
    """Create searchable text content for each project"""
    search_index = {}
    
    for project in projects:
        project_id = project.get('project_id', '') or project.get('job_number', '')
        if not project_id:
            continue
            
        # Build comprehensive search text
        search_text_parts = []
        
        # Basic project info
        if project.get('project_name'):
            search_text_parts.append(f"Project: {project['project_name']}")
        
        if project.get('client'):
            search_text_parts.append(f"Client: {project['client']}")
            
        if project.get('country'):
            search_text_parts.append(f"Country: {project['country']}")
            
        # Services
        if project.get('services'):
            search_text_parts.append(f"Services: {', '.join(project['services'])}")
            
        # Regions
        if project.get('regions'):
            search_text_parts.append(f"Regions: {', '.join(project['regions'])}")
            
        # Sectors
        if project.get('sectors'):
            search_text_parts.append(f"Sectors: {', '.join(project['sectors'])}")
            
        # Technologies
        if project.get('technologies'):
            tech_list = []
            for tech in project['technologies']:
                if isinstance(tech, dict):
                    tech_list.append(f"{tech.get('type', '')} ({tech.get('capacity', '')})")
                else:
                    tech_list.append(str(tech))
            search_text_parts.append(f"Technologies: {', '.join(tech_list)}")
            
        # Contract details
        if project.get('contract_value'):
            search_text_parts.append(f"Contract Value: {project['contract_value']}")
            
        if project.get('mw_total'):
            search_text_parts.append(f"Total MW: {project['mw_total']}")
            
        # Years
        if project.get('year_start') or project.get('year_end'):
            years = f"{project.get('year_start', '')} - {project.get('year_end', '')}"
            search_text_parts.append(f"Duration: {years}")
            
        # DOCX specific data if available
        if project.get('docx_data'):
            docx_data = project['docx_data']
            if docx_data.get('assignment', {}).get('name'):
                search_text_parts.append(f"Assignment: {docx_data['assignment']['name']}")
            if docx_data.get('assignment', {}).get('description'):
                search_text_parts.append(f"Description: {docx_data['assignment']['description']}")
            if docx_data.get('role'):
                search_text_parts.append(f"Role: {docx_data['role']}")
                
        search_index[project_id] = {
            'project': project,
            'search_text': '\n'.join(search_text_parts)
        }
    
    return search_index

def find_relevant_projects(query: str, search_index: Dict, max_results: int = 10) -> List[Dict]:
    """Find projects relevant to the query using keyword matching and semantic analysis"""
    query_lower = query.lower()
    relevant_projects = []
    
    # Extract potential job IDs from query
    potential_job_ids = re.findall(r'\b\d+\b', query)
    
    # First, check for direct job ID matches
    for job_id in potential_job_ids:
        if job_id in search_index:
            relevant_projects.append({
                'project': search_index[job_id]['project'],
                'relevance_score': 1.0,
                'match_type': 'direct_id'
            })
    
    # If we found direct matches, return those first
    if relevant_projects:
        return relevant_projects[:max_results]
    
    # Otherwise, do keyword-based search
    query_words = [word for word in query_lower.split() if len(word) > 2]
    
    for project_id, data in search_index.items():
        search_text_lower = data['search_text'].lower()
        
        # Calculate relevance score based on keyword matches
        score = 0
        matches = []
        
        for word in query_words:
            if word in search_text_lower:
                # Give higher weight to exact matches in important fields
                if word in data['project'].get('project_name', '').lower():
                    score += 3
                elif word in data['project'].get('client', '').lower():
                    score += 2
                elif word in data['project'].get('country', '').lower():
                    score += 2
                else:
                    score += 1
                matches.append(word)
        
        # Bonus for multiple matches
        if len(matches) > 1:
            score += len(matches) * 0.5
            
        if score > 0:
            relevant_projects.append({
                'project': data['project'],
                'relevance_score': score,
                'match_type': 'keyword',
                'matched_terms': matches
            })
    
    # Sort by relevance score
    relevant_projects.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return relevant_projects[:max_results]

def format_project_for_context(project_data: Dict) -> str:
    """Format project data for use in LLM context"""
    project = project_data['project']
    context_parts = []
    
    # Header
    project_id = project.get('project_id') or project.get('job_number')
    project_name = project.get('project_name', 'Unnamed Project')
    context_parts.append(f"=== PROJECT {project_id}: {project_name} ===")
    
    # Basic info
    if project.get('client'):
        context_parts.append(f"Client: {project['client']}")
    if project.get('country'):
        context_parts.append(f"Country: {project['country']}")
    if project.get('contract_value'):
        context_parts.append(f"Contract Value: {project['contract_value']}")
    if project.get('year_start') or project.get('year_end'):
        years = f"{project.get('year_start', '')} - {project.get('year_end', '')}"
        context_parts.append(f"Project Duration: {years}")
    
    # Technical details
    if project.get('mw_total'):
        context_parts.append(f"Total Capacity: {project['mw_total']} MW")
        
    if project.get('technologies'):
        tech_list = []
        for tech in project['technologies']:
            if isinstance(tech, dict):
                tech_list.append(f"{tech.get('type', '')} ({tech.get('capacity', '')})")
            else:
                tech_list.append(str(tech))
        context_parts.append(f"Technologies: {', '.join(tech_list)}")
    
    # Services and sectors
    if project.get('services'):
        context_parts.append(f"Services: {', '.join(project['services'])}")
    if project.get('sectors'):
        context_parts.append(f"Sectors: {', '.join(project['sectors'])}")
    if project.get('regions'):
        context_parts.append(f"Regions: {', '.join(project['regions'])}")
    
    # DOCX data if available
    if project.get('docx_data'):
        docx_data = project['docx_data']
        context_parts.append("--- Detailed Information ---")
        if docx_data.get('assignment', {}).get('name'):
            context_parts.append(f"Assignment: {docx_data['assignment']['name']}")
        if docx_data.get('assignment', {}).get('description'):
            context_parts.append(f"Description: {docx_data['assignment']['description']}")
        if docx_data.get('role'):
            context_parts.append(f"K&M Role: {docx_data['role']}")
        if docx_data.get('duration', {}).get('original'):
            context_parts.append(f"Duration Details: {docx_data['duration']['original']}")
    
    # Match information
    if project_data.get('match_type') == 'keyword' and project_data.get('matched_terms'):
        context_parts.append(f"[Matched terms: {', '.join(project_data['matched_terms'])}]")
    
    return '\n'.join(context_parts) + '\n\n'

# ---- 2. Build vectorstore with JSON data only ----
def build_vectorstore_from_json(use_openai=USE_OPENAI):
    """Build vectorstore from JSON project data only"""
    progress_bar = st.progress(0, text="Loading project data...")
    
    # Load project data
    project_data = load_project_data()
    if not project_data:
        st.error("Could not load project data")
        return None
    
    projects = project_data.get('projects', [])
    all_chunks = []
    
    # Add JSON project data to vectorstore
    progress_bar.progress(0.2, text="Processing project database...")
    for i, project in enumerate(projects):
        project_id = project.get('project_id') or project.get('job_number', f'project_{i}')
        
        # Create document from project data
        formatted_project = format_project_for_context({'project': project})
        
        project_doc = LangchainDocument(
            page_content=formatted_project,
            metadata={
                "source": "project_database",
                "project_id": project_id,
                "filename": "combined_project_data.json",
                "project_name": project.get('project_name', ''),
                "client": project.get('client', ''),
                "country": project.get('country', ''),
                "data_source": project.get('source', 'unknown')
            }
        )
        all_chunks.append(project_doc)
        
        # Update progress
        progress_bar.progress(0.2 + 0.6 * (i / len(projects)), 
                            text=f"Processing project {i+1}/{len(projects)}")
    
    # Build vectorstore
    progress_bar.progress(0.9, text="Building vector database...")
    
    if use_openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vs = Chroma.from_documents(all_chunks, embedding=embeddings, persist_directory=VECTORSTORE_DIR)
    progress_bar.progress(1.0, text="Complete!")
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
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    else:
        return ChatOllama(model="llama3", temperature=0.1)

def initialize_vectorstore():
    if os.path.exists(os.path.join(VECTORSTORE_DIR, 'index')) or os.path.exists(os.path.join(VECTORSTORE_DIR, 'chroma.sqlite3')):
        try:
            return load_vectorstore_from_disk(VECTORSTORE_DIR, USE_OPENAI)
        except:
            # If loading fails, rebuild
            return build_vectorstore_from_json(USE_OPENAI)
    else:
        return build_vectorstore_from_json(USE_OPENAI)

# ---- 3. Enhanced prompt template ----
enhanced_prompt_template = """
You are a knowledgeable assistant for K&M Engineering & Consulting Corporation, specializing in their project portfolio and capabilities.

RELEVANT K&M PROJECTS:
{project_context}

ADDITIONAL CONTEXT FROM DOCUMENTS:
{context}

Based on the above information, please answer the user's question comprehensively. 

Guidelines:
1. Prioritize information from the project database as it contains the most structured and up-to-date project information
2. Use specific project details like job numbers, client names, locations, and technical specifications
3. If mentioning projects, include relevant details like project ID, client, country, and key technical information
4. For capability questions, reference multiple relevant projects as examples
5. Be specific about K&M's role and the services provided
6. If the information is not available in the provided context, clearly state that

Previous conversation: {chat_history}
Question: {question}

Answer:
"""

# ---- Session State Initialization ----
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = initialize_vectorstore()

if 'project_data' not in st.session_state:
    st.session_state['project_data'] = load_project_data()

if 'search_index' not in st.session_state and st.session_state['project_data']:
    st.session_state['search_index'] = create_project_search_index(
        st.session_state['project_data'].get('projects', [])
    )

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'current_question' not in st.session_state:
    st.session_state['current_question'] = ""

# ---- UI ----
if st.session_state['project_data']:
    project_info = st.session_state['project_data'].get('document_info', {})
    st.info(f"📊 Loaded {project_info.get('total_merged_projects', 0)} projects from combined database")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Excel Projects", project_info.get('excel_projects', 0))
    with col2:
        st.metric("DOCX Projects", project_info.get('docx_projects', 0))
    with col3:
        st.metric("Total Projects", project_info.get('total_merged_projects', 0))

# Add Update Database button
if st.button("🔄 Update Database"):
    with st.spinner("Updating knowledge base..."):
        # Clear cache and reload
        st.cache_data.clear()
        st.session_state['vectorstore'] = build_vectorstore_from_json(USE_OPENAI)
        st.session_state['project_data'] = load_project_data()
        if st.session_state['project_data']:
            st.session_state['search_index'] = create_project_search_index(
                st.session_state['project_data'].get('projects', [])
            )
    st.success("✅ Knowledge base updated!")

# Display chat history
for i, (user, bot) in enumerate(st.session_state['chat_history']):
    with st.container():
        st.markdown(f"**👤 User:** {user}")
        st.markdown(f"**🤖 Assistant:** {bot}")
        st.divider()

# Chat input - use session state for the value
user_input = st.text_input("Ask about K&M's projects and capabilities:", 
                          value=st.session_state['current_question'],
                          key="kb_input", 
                          placeholder="e.g., 'What renewable energy projects has K&M worked on in Asia?'")

if st.button("Send", key="kb_send") and user_input.strip():
    with st.spinner("🔍 Searching project database..."):
        # 1. Find relevant projects from JSON data
        relevant_projects = find_relevant_projects(
            user_input, 
            st.session_state['search_index'], 
            max_results=8
        )
        
        # 2. Format project context
        project_context = ""
        if relevant_projects:
            project_context = "RELEVANT K&M PROJECTS:\n\n"
            for proj_data in relevant_projects:
                project_context += format_project_for_context(proj_data)
        
        # 3. Get additional context from vectorstore
        retriever = st.session_state['vectorstore'].as_retriever(search_kwargs={"k": 6})
        
        # 4. Set up enhanced prompt
        prompt = PromptTemplate(
            template=enhanced_prompt_template,
            input_variables=["project_context", "context", "question", "chat_history"]
        )
        
        # 5. Set up QA chain
        llm = get_llm(USE_OPENAI)
        
        # Create a custom chain that includes project context
        def custom_qa_chain(question, chat_history):
            # Get documents from retriever
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format the prompt
            formatted_prompt = prompt.format(
                project_context=project_context,
                context=context,
                question=question,
                chat_history=chat_history
            )
            
            # Get response from LLM
            response = llm.invoke(formatted_prompt)
            
            return {
                "answer": response.content,
                "source_documents": docs
            }

    with st.spinner("🤔 Generating response..."):
        # 6. Process the query
        result = custom_qa_chain(user_input, st.session_state['chat_history'])
        
        answer = result["answer"]
        
        # 7. Add source information
        source_info = []
        if relevant_projects:
            project_ids = [p['project'].get('project_id') or p['project'].get('job_number') 
                          for p in relevant_projects[:3]]
            source_info.append(f"📊 Referenced projects: {', '.join(filter(None, project_ids))}")
        
        # 8. Compose final response
        full_response = answer
        if source_info:
            full_response += "\n\n---\n" + "\n".join(source_info)

    st.session_state['chat_history'].append((user_input, full_response))
    # Clear the current question after sending
    st.session_state['current_question'] = ""
    st.rerun()

# Sidebar with example questions only
with st.sidebar:
    st.subheader("💡 Example Questions")
    example_questions = [
        "What renewable energy projects has K&M completed?",
        "Show me projects in Asia",
        "What is K&M's experience with solar power?",
        "Tell me about project 12345",
        "What services does K&M provide?",
        "Show projects with USAID as client",
        "What wind energy projects has K&M worked on?",
        "Tell me about K&M's hydropower experience",
        "Show me projects in Africa",
        "What transmission line projects has K&M done?"
    ]
    
    for q in example_questions:
        if st.button(q, key=f"example_{hash(q)}", use_container_width=True):
            st.session_state['current_question'] = q
            st.rerun()

st.caption("🚀 Powered by K&M's comprehensive project database")
