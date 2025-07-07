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
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import docx
import PyPDF2
from io import BytesIO
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import shutil
# import pysqlite3

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Add this at the top of your script, before other imports
load_dotenv()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

VECTORSTORE_DIR = "./chroma_store"
PROJECT_JSON = "./Shared/Marketing/Project Descriptions/combined_project_data.json"
WEBSITE_MD = "./Shared/website.md"

USE_OPENAI = True  # Set to True to use OpenAI, False for local Ollama
def get_openai_api_key():
    """Get OpenAI API key from environment or Streamlit secrets"""
    api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        try:
            api_key = os.getenv('OPENAI_API_KEY')
        except Exception:
            pass
    return api_key

OPENAI_API_KEY = get_openai_api_key()
if USE_OPENAI and not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables or Streamlit secrets.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

st.title("K&M Project Knowledge Assistant")
st.caption("Powered by comprehensive project database and website content")

# ---- CV PROCESSING FUNCTIONS ----

def extract_text_from_docx(file_content):
    try:
        doc = docx.Document(BytesIO(file_content))
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    cell_text = []
                    for paragraph in cell.paragraphs:
                        if paragraph.text.strip():
                            cell_text.append(paragraph.text.strip())
                    if cell_text:
                        row_text.append(' '.join(cell_text))
                if row_text:
                    full_text.append(' | '.join(row_text))
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return None

def extract_text_from_pdf(file_content):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text.append(page_text.strip())
        return '\n'.join(text)
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def separate_cv_sections(cv_text: str) -> tuple:
    work_undertaken_patterns = [
        r'Work\s+Undertaken\s+that\s+Best\s+Illustrates\s+Capability\s+to\s+Handle\s+the\s+Tasks\s+Assigned',
        r'Work\s+Undertaken\s+that\s+Best\s+Illustrates',
        r'WORK\s+UNDERTAKEN\s+THAT\s+BEST\s+ILLUSTRATES',
        r'Work\s+undertaken\s+that\s+best\s+illustrates',
        r'Project\s+Experience',
        r'PROJECT\s+EXPERIENCE',
        r'Key\s+Projects',
        r'Selected\s+Projects'
    ]
    project_section_start = None
    for pattern in work_undertaken_patterns:
        match = re.search(pattern, cv_text, re.IGNORECASE)
        if match:
            project_section_start = match.start()
            break
    if project_section_start is not None:
        bio_text = cv_text[:project_section_start].strip()
        projects_text = cv_text[project_section_start:].strip()
    else:
        project_patterns = [
            r'Project\s+Director[,:]',
            r'Project\s+Manager[,:]',
            r'Team\s+Leader[,:]',
            r'Lead\s+Consultant[,:]',
            r'Senior\s+\w+\s+Expert[,:]'
        ]
        matches = []
        for pattern in project_patterns:
            for match in re.finditer(pattern, cv_text, re.IGNORECASE):
                matches.append(match.start())
        if matches:
            project_section_start = min(matches)
            bio_text = cv_text[:project_section_start].strip()
            projects_text = cv_text[project_section_start:].strip()
        else:
            split_point = int(len(cv_text) * 0.6)
            bio_text = cv_text[:split_point].strip()
            projects_text = cv_text[split_point:].strip()
    return bio_text, projects_text

def parse_projects_section(projects_text: str) -> List[str]:
    project_separators = [
        r'Project\s+Director[,:]',
        r'Project\s+Manager[,:]',
        r'Team\s+Leader[,:]',
        r'Lead\s+Consultant[,:]',
        r'Senior\s+\w+\s+Expert[,:]',
        r'PPP\s+Financial\s+Expert[,:]',
        r'Energy\s+Economist[,:]',
        r'Technical\s+Manager[,:]',
        r'LNG\s+Expert[,:]',
        r'Quality\s+Control[,:]',
        r'Quality\s+Assurance[,:]'
    ]
    project_starts = []
    for pattern in project_separators:
        for match in re.finditer(pattern, projects_text, re.IGNORECASE):
            project_starts.append((match.start(), match.group()))
    project_starts.sort(key=lambda x: x[0])
    if not project_starts:
        lines = projects_text.split('\n')
        projects = []
        current_project = []
        for line in lines:
            line = line.strip()
            if line:
                if re.match(r'^[A-Z][^,]*,\s*[A-Z][^(]*\([0-9]{4}', line):
                    if current_project:
                        projects.append('\n'.join(current_project))
                        current_project = []
                current_project.append(line)
        if current_project:
            projects.append('\n'.join(current_project))
        return projects
    projects = []
    for i, (start_pos, _) in enumerate(project_starts):
        if i < len(project_starts) - 1:
            end_pos = project_starts[i + 1][0]
            project_text = projects_text[start_pos:end_pos].strip()
        else:
            project_text = projects_text[start_pos:].strip()
        if project_text:
            projects.append(project_text)
    return projects

def find_relevant_projects_with_gpt(projects_text: str, user_description: str, use_gpt: bool = True) -> List[str]:
    if not use_gpt:
        st.info("GPT processing disabled - returning empty project list")
        return []
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return []
    try:
        prompt = f"""
        You are an expert CV analyst. I will provide you with a projects section from a CV and a job description. 
        Your task is to identify ALL projects from the CV that are relevant to the job requirements.

        JOB DESCRIPTION:
        {user_description}

        CV PROJECTS SECTION:
        {projects_text}

        Please:
        1. Carefully read through all the projects in the CV
        2. Identify which projects are relevant to the job description based on:
           - Similar locations
           - Similar technologies/sectors
           - Similar services/roles
           - Similar project types
           - Similar skills required
        3. For each relevant project, copy and paste the EXACT project description from the CV
        4. Return ONLY the relevant project descriptions, each separated by "---PROJECT_SEPARATOR---"

        Do not summarize or modify the project descriptions - copy them exactly as they appear in the CV.
        If no projects are relevant, return "NO_RELEVANT_PROJECTS"
        """
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an expert CV analyst specializing in matching project experience to job requirements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        result = response.choices[0].message.content.strip()
        if result == "NO_RELEVANT_PROJECTS":
            return []
        relevant_projects = [proj.strip() for proj in result.split("---PROJECT_SEPARATOR---") if proj.strip()]
        return relevant_projects
    except Exception as e:
        st.error(f"Error calling OpenAI API for project matching: {e}")
        return []

def generate_qualification_paragraphs_with_gpt(bio_text: str, relevant_projects: List[str], user_description: str, use_gpt: bool = True) -> str:
    if not use_gpt:
        st.info("GPT processing disabled - returning basic qualification text")
        return "Qualification generation disabled. Please enable GPT processing to generate tailored qualification paragraphs."
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return "Unable to generate qualifications - OpenAI API not available."
    try:
        projects_combined = "\n\n".join(relevant_projects) if relevant_projects else "No directly relevant projects identified."
        prompt = f"""
        You are an expert proposal writer. I will provide you with:
        1. Bio section from a CV
        2. Relevant project experience 
        3. Job/work description
        4. Themes of the proposal

        Your task is to write a four-paragraph professional bio for this individual. Use confident, professional language without exaggeration or flattery.

        BIO SECTION:
        {bio_text}

        RELEVANT PROJECT EXPERIENCE:
        {projects_combined}

        JOB/WORK DESCRIPTION:
        {user_description}
        
        Themes of the proposal:
        {st.session_state['extracted_themes']}
        
        The four paragraphs should be structured as follows:
        1. Paragraph 1 should introduce the individual, summarizing their current role, core areas of expertise, experience with relevant power generation technologies (e.g., LNG, renewables, storage), and countries or regions where they have worked that are relevant to the proposal.
        2. Paragraph 2 should focus on a specific theme that aligns with the needs of the proposal. After paragraph 2, insert a section titled ===Relevant Project Experience for Theme 2=== and then output a table as follows: For each of exactly 2-3 relevant project experiences (copied verbatim from the RELEVANT PROJECT EXPERIENCE section above) that reinforce the theme stated in paragraph 2, output a line starting with >>>PROJECT_START<<< and ending with >>>PROJECT_END<<<, with the project description in between. Each project should be clearly separated and include the exact position, project name, location, year, and a short description as in the CV.
        3. Paragraph 3 should focus on another specific theme. After paragraph 3, insert a section titled ===Relevant Project Experience for Theme 3=== and then output a table as follows: For each of exactly 2-3 relevant project experiences (copied verbatim from the RELEVANT PROJECT EXPERIENCE section above) that reinforce the theme stated in paragraph 3, output a line starting with >>>PROJECT_START<<< and ending with >>>PROJECT_END<<<, with the project description in between. Each project should be clearly separated and include the exact position, project name, location, year, and a short description as in the CV (if the tense in the project description is not past tense, change it to past tense, but keep the exact project description as in the CV).
        4. Paragraph 4 should conclude the bio by summarizing how the individual's experience aligns with the proposed assignment. It should also include a brief statement of their academic background, including degrees earned, fields of study, and the institutions attended. Ensure the writing is cohesive, clear, and appropriate for inclusion in a technical or commercial proposal.

        Write in third person (he/she) and make it compelling but factual.
        """
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an expert proposal writer specializing in highlighting candidate qualifications for infrastructure and consulting projects."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API for qualification generation: {e}")
        return f"Unable to generate qualifications due to API error: {str(e)}"

# ---- MOVE: Button Mode Selection to sidebar top ----
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'cv'  # Default to search

# Initialize separate chat histories for each tab
if 'chat_history_general' not in st.session_state:
    st.session_state['chat_history_general'] = []

if 'chat_history_search' not in st.session_state:
    st.session_state['chat_history_search'] = []

if 'chat_history_cv' not in st.session_state:
    st.session_state['chat_history_cv'] = []

# Keep backward compatibility
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    

def get_current_chat_history():
    """Get chat history for current mode"""
    mode = st.session_state.get('mode', 'search')
    if mode == 'general':
        return st.session_state['chat_history_general']
    elif mode == 'search':
        return st.session_state['chat_history_search']
    elif mode == 'cv':
        return st.session_state['chat_history_cv']
    else:
        return st.session_state['chat_history']

def add_to_current_chat_history(user_msg, bot_msg):
    """Add message to current mode's chat history"""
    mode = st.session_state.get('mode', 'search')
    if mode == 'general':
        st.session_state['chat_history_general'].append((user_msg, bot_msg))
    elif mode == 'search':
        st.session_state['chat_history_search'].append((user_msg, bot_msg))
    elif mode == 'cv':
        st.session_state['chat_history_cv'].append((user_msg, bot_msg))
    else:
        st.session_state['chat_history'].append((user_msg, bot_msg))



with st.sidebar:
    st.subheader("Mode Selection")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("General", key="general_btn", 
                    type="primary" if st.session_state['mode'] == 'general' else "secondary",
                    use_container_width=True):
            st.session_state['mode'] = 'general'
            st.rerun()
    with col2:
        if st.button("Search", key="search_btn", 
                    type="primary" if st.session_state['mode'] == 'search' else "secondary",
                    use_container_width=True):
            st.session_state['mode'] = 'search'
            st.rerun()
    with col3:
        if st.button("CV", key="cv_btn", 
                    type="primary" if st.session_state['mode'] == 'cv' else "secondary",
                    use_container_width=True):
            st.session_state['mode'] = 'cv'
            st.rerun()
    st.caption(f"**Current Mode:** {st.session_state['mode'].title()}")
    st.divider()

# ---- CV MODE UI ----
if st.session_state['mode'] == 'cv':
    st.subheader("📋 AI-Powered CV Analysis & Proposal Generation")
    
    with st.expander("ℹ️ About AI-Powered CV Analysis"):
        st.markdown("""
        **This CV analysis uses OpenAI GPT-4.1-mini to:**

        1. **Intelligent Project Matching**: AI reads through all projects in the CV and identifies those most relevant to your work description

        2. **Exact Text Extraction**: Relevant project descriptions are copied exactly from the CV (no summarization or modification)

        3. **Smart Qualification Writing**: AI combines the candidate's bio and relevant projects to write compelling qualification paragraphs
        """)
        
    st.info("Upload a CV (DOCX or PDF) and describe the work requirements. AI will identify relevant projects and generate key qualification paragraphs.")

    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key required for CV analysis. Please set OPENAI_API_KEY in your environment variables or Streamlit secrets.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload CV File",
        type=['docx', 'pdf'],
        help="Upload a DOCX or PDF file containing the CV to analyze"
    )

    work_description = st.text_area(
        "Describe the Work/Project Requirements",
        placeholder="e.g., 'We need a senior consultant for feasibility study of a 100MW solar project in Kenya, including technical assessment, financial modeling, and regulatory analysis.'",
        height=100,
        help="Provide a detailed description of the work requirements. AI will match this against the CV projects."
    )

    # NEW: Prompt revision textbox
    default_qual_prompt = """You are an expert proposal writer. I will provide you with:
1. Bio section from a CV
2. Relevant project experience 
3. Job/work description
4. Themes of the proposal

Your task is to write a four-paragraph professional bio for this individual. Use confident, professional language without exaggeration or flattery.

BIO SECTION:
{{bio_text}}

RELEVANT PROJECT EXPERIENCE:
{{relevant_projects}}

JOB/WORK DESCRIPTION:
{{user_description}}

Themes of the proposal:
{{themes}}

The four paragraphs should be structured as follows:
1. Paragraph 1 should introduce the individual, summarizing their current role, core areas of expertise, experience with relevant power generation technologies (e.g., LNG, renewables, storage), and countries or regions where they have worked that are relevant to the proposal.
2. Paragraph 2 should focus on a specific theme that aligns with the needs of the proposal. After paragraph 2, insert a section titled ===Relevant Project Experience for Theme 2=== and then output a table as follows: For each of exactly 2-3 relevant project experiences (copied verbatim from the RELEVANT PROJECT EXPERIENCE section above) that reinforce the theme stated in paragraph 2, output a line starting with >>>PROJECT_START<<< and ending with >>>PROJECT_END<<<, with the project description in between. Each project should be clearly separated and include the exact position, project name, location, year, and a short description as in the CV.
3. Paragraph 3 should focus on another specific theme. After paragraph 3, insert a section titled ===Relevant Project Experience for Theme 3=== and then output a table as follows: For each of exactly 2-3 relevant project experiences (copied verbatim from the RELEVANT PROJECT EXPERIENCE section above) that reinforce the theme stated in paragraph 3, output a line starting with >>>PROJECT_START<<< and ending with >>>PROJECT_END<<<, with the project description in between. Each project should be clearly separated and include the exact position, project name, location, year, and a short description as in the CV (if the tense in the project description is not past tense, change it to past tense, but keep the exact project description as in the CV).
4. Paragraph 4 should conclude the bio by summarizing how the individual's experience aligns with the proposed assignment. It should also include a brief statement of their academic background, including degrees earned, fields of study, and the institutions attended. Ensure the writing is cohesive, clear, and appropriate for inclusion in a technical or commercial proposal.

Write in third person (he/she) and make it compelling but factual.
"""
    qual_prompt = st.text_area(
        "Revise the Answer Structuring Prompt (Optional)",
        value=default_qual_prompt,
        height=180,
        help="You can revise how the answer is structured. The default is a strong proposal-style prompt."
    )

    # NEW: Extract themes button
    extract_themes_btn = st.button("🔍 Extract Themes from Requirements", type="secondary", use_container_width=True, key="extract_themes_btn")

    # NEW: Placeholder for themes extraction and revision
    # Initialize themes in session state if not exists
    if 'extracted_themes' not in st.session_state:
        st.session_state['extracted_themes'] = "(Click 'Extract Themes' above to get themes from your requirements)"
    
    themes_text = st.text_area(
        "Themes to Highlight (Edit before generating CV)",
        value=st.session_state['extracted_themes'],
        height=100,
        key="themes_textbox",
        help="AI will suggest themes to highlight from the project experience. You can edit these before generating the final answer."
    )

    # NEW: Generate CV button (will be enabled after themes are available)
    generate_cv_btn = st.button("Generate CV", type="primary", use_container_width=True, key="generate_cv_btn")

    show_sections = False
    use_gpt_processing = True

    show_advanced_settings = False 

    if show_advanced_settings:
        with st.expander("🔧 Advanced Settings"):
            st.info("CV analysis uses OpenAI GPT-4.1-mini for intelligent project matching and qualification generation.")
            show_sections = st.checkbox(
                "Show CV Section Breakdown", 
                value=False,
                help="Display how the CV was separated into bio and projects sections"
            )
            use_gpt_processing = st.checkbox(
                "Enable GPT Processing", 
                value=True,
                help="Enable AI processing with GPT-4.1-mini. If disabled, will only show document sections without AI analysis."
            )

    # NEW: Extract themes from requirements logic
    if extract_themes_btn and work_description:
        with st.spinner("🔍 Extracting themes from requirements..."):
            def extract_themes_from_requirements(work_description, use_gpt=True):
                if not use_gpt:
                    return []
                if not openai_client:
                    st.error("OpenAI client not initialized. Please check your API key.")
                    return []
                try:
                    prompt = f"""
                    You are an expert CV analyst. I will provide you with a job/work description.\n\nYour task is to identify the main themes, skills, sectors, technologies, and types of experience that would be most relevant for this position.\n\nJOB/WORK DESCRIPTION:\n{work_description}\n\nPlease return a concise comma-separated list of themes (e.g., 'solar power, feasibility studies, project management, Africa, financial modeling, regulatory analysis').\nFocus on specific technical skills, sectors, technologies, and experience types that would be valuable for this role.\nDo not include generic words like 'project' or 'experience'.\n"""
                    response = openai_client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert CV analyst specializing in extracting key themes from job requirements."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=200
                    )
                    result = response.choices[0].message.content.strip()
                    # Split by comma and clean
                    themes = [t.strip() for t in result.split(',') if t.strip()]
                    return themes
                except Exception as e:
                    st.error(f"Error calling OpenAI API for theme extraction: {e}")
                    return []

            themes = extract_themes_from_requirements(work_description, use_gpt_processing)
            if themes:
                st.session_state['extracted_themes'] = ', '.join(themes)
                st.success(f"✅ Extracted {len(themes)} themes from requirements!")
                st.rerun()
            else:
                st.error("No themes could be extracted. Please check your requirements description or try again.")

    if uploaded_file and work_description:
        # This section is now handled by the Extract Themes and Generate CV buttons
        pass

    # --- NEW: Generate CV button logic ---
    if generate_cv_btn and uploaded_file and work_description and st.session_state.get('extracted_themes') and st.session_state['extracted_themes'] != "(Click 'Extract Themes' above to get themes from your requirements)":
        # Use the user-edited themes
        user_themes = st.session_state.get('extracted_themes')
        # Use the user-edited prompt
        user_prompt = qual_prompt
        # Use the previously extracted bio_text, individual_projects, etc.
        # For now, re-extract (could be optimized with session state)
        file_content = uploaded_file.read()
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            cv_text = extract_text_from_docx(file_content)
        elif uploaded_file.type == "application/pdf":
            cv_text = extract_text_from_pdf(file_content)
        else:
            st.error("Unsupported file type")
            st.stop()
        if not cv_text:
            st.error("Could not extract text from the uploaded file")
            st.stop()
        bio_text, projects_text = separate_cv_sections(cv_text)
        individual_projects = parse_projects_section(projects_text)
        relevant_projects = find_relevant_projects_with_gpt(projects_text, work_description, use_gpt_processing)
        # Insert the user themes into the prompt if needed
        # For now, just append to the prompt
        final_prompt = user_prompt + f"\n\nHIGHLIGHT THESE THEMES: {user_themes}"
        # Use the custom prompt in GPT call
        def generate_qualification_paragraphs_custom_prompt(bio_text, relevant_projects, user_description, custom_prompt, user_themes, use_gpt=True):
            if not use_gpt:
                return "Qualification generation disabled. Please enable GPT processing to generate tailored qualification paragraphs."
            if not openai_client:
                return "Unable to generate qualifications - OpenAI API not available."
            try:
                projects_combined = "\n\n".join(relevant_projects) if relevant_projects else "No directly relevant projects identified."
                prompt = (
                    custom_prompt
                    .replace("{{bio_text}}", bio_text)
                    .replace("{{relevant_projects}}", projects_combined)
                    .replace("{{user_description}}", user_description)
                    .replace("{{themes}}", user_themes)
                )
                # DEBUG: Show what is being sent to the LLM
                # debug_info = {
                #     "bio_text": bio_text,
                #     "relevant_projects": relevant_projects,
                #     "user_description": user_description,
                #     "user_themes": user_themes,
                #     "prompt": prompt
                # }
                # st.write("DEBUG LLM INPUT", debug_info)
                # st.session_state['cv_debug_llm_input'] = debug_info
                response = openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert proposal writer specializing in highlighting candidate qualifications for infrastructure and consulting projects."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=3000
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Unable to generate qualifications due to API error: {str(e)}"
        qualifications = generate_qualification_paragraphs_custom_prompt(bio_text, relevant_projects, work_description, final_prompt, user_themes, use_gpt_processing)
        
        # Store results in session state to persist across reruns
        st.session_state['cv_qualifications'] = qualifications
        st.session_state['cv_relevant_projects'] = relevant_projects
        st.session_state['cv_generated'] = True
        
        st.success("✅ AI Analysis Complete!")
        st.rerun()

    # --- NEW: Download as Word document button (outside conditional block) ---
    if st.session_state.get('cv_generated', False):
        st.markdown("---")
        st.markdown("## 📄 Generated CV")
        
        # Re-display the results
        st.markdown("### 🎯 Key Qualifications")
        st.text_area("Key Qualification Paragraphs", st.session_state.get('cv_qualifications', ''), height=220, disabled=False, key="qualifications_display")
        
        relevant_projects = st.session_state.get('cv_relevant_projects', [])
        if relevant_projects:
            st.markdown(f"### 📁 Relevant Project Experience")
            for i, project in enumerate(relevant_projects, 1):
                st.text_area(f"Project {i}", project, height=200, disabled=False, key=f"relevant_project_{i}_display")
        else:
            st.info("No directly relevant projects identified.")
        
        # Download button
        from docx import Document
        from docx.shared import Pt
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        from io import BytesIO
        doc = Document()
        # Set default font for the document to Century Gothic
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Century Gothic'
        font.size = Pt(11)
        # For compatibility with some Word versions
        style.element.rPr.rFonts.set(qn('w:eastAsia'), 'Century Gothic')

        # add body text "Key Qualifications"
        doc.add_paragraph('Key Qualifications')
        qualifications = st.session_state.get('cv_qualifications', '')

        # --- FIX: Output each table immediately after its associated paragraphs ---
        import re
        qualifications = st.session_state.get('cv_qualifications', '')
        # Split on the theme markers
        split_sections = re.split(r'===Relevant Project Experience for Theme 2===|===Relevant Project Experience for Theme 3===', qualifications)
        # There can be up to 3 sections: [before Theme 2, between Theme 2 and 3, after Theme 3]
        def extract_projects_and_text(section_text):
            if not section_text:
                return [], []
            # Find all project blocks
            projects = re.findall(r'>>>PROJECT_START<<<(.*?)>>>PROJECT_END<<<', section_text, re.DOTALL)
            # Remove all project blocks from the section
            cleaned = re.sub(r'>>>PROJECT_START<<<.*?>>>PROJECT_END<<<', '', section_text, flags=re.DOTALL)
            # Any non-empty lines left are paragraphs
            paras = [p.strip() for p in cleaned.split('\n') if p.strip()]
            return projects, paras
        # Helper to set cell background color
        def set_cell_background(cell, color_hex):
            from docx.oxml import parse_xml
            from docx.oxml.ns import nsdecls
            cell._tc.get_or_add_tcPr().append(
                parse_xml(r'<w:shd {} w:fill="{}"/>'.format(nsdecls('w'), color_hex))
            )
        DARK_BLUE = '002060'      # Dark blue, accent 1
        LIGHT_BLUE = 'C6D9F1'     # Blue, accent 1, lighter 80%
        WHITE = 'FFFFFF'
        # Section 0: before Theme 2 (paragraphs 1 and 2)
        if len(split_sections) > 0:
            for para in [p.strip() for p in split_sections[0].split('\n') if p.strip()]:
                p = doc.add_paragraph(para)
                for run in p.runs:
                    run.font.name = 'Century Gothic'
                    run.font.size = Pt(11)
                p.style = doc.styles['Normal']
        # Section 1: Theme 2 table and any text before Theme 3 (projects for theme 2, possible paragraph 3)
        if len(split_sections) > 1:
            theme2_projects, theme2_paras = extract_projects_and_text(split_sections[1])
            if theme2_projects:
                table = doc.add_table(rows=len(theme2_projects)+1, cols=1)
                table.style = 'Table Grid'
                # First row: empty, dark blue
                cell = table.cell(0, 0)
                cell.text = ''
                set_cell_background(cell, DARK_BLUE)
                # Project rows: alternate white and light blue, starting from second project row (i=1)
                for i, proj in enumerate(theme2_projects):
                    cell = table.cell(i+1, 0)
                    cell.text = ''  # Clear cell
                    if ":" in proj:
                        before, after = proj.split(":", 1)
                        run = cell.paragraphs[0].add_run(before + ":")
                        run.bold = True
                        run.font.name = 'Century Gothic'
                        run.font.size = Pt(11)
                        run2 = cell.paragraphs[0].add_run(after)
                        run2.font.name = 'Century Gothic'
                        run2.font.size = Pt(11)
                    else:
                        run = cell.paragraphs[0].add_run(proj.strip())
                        run.font.name = 'Century Gothic'
                        run.font.size = Pt(11)
                    if i == 0:
                        color = WHITE
                    elif i % 2 == 1:
                        color = LIGHT_BLUE
                    else:
                        color = WHITE
                    set_cell_background(cell, color)
                doc.add_paragraph('')  # Add space after table
            for para in theme2_paras:
                p = doc.add_paragraph(para)
                for run in p.runs:
                    run.font.name = 'Century Gothic'
                    run.font.size = Pt(11)
                p.style = doc.styles['Normal']
        # Section 2: Theme 3 table and any text after (projects for theme 3, possible paragraph 4)
        if len(split_sections) > 2:
            theme3_projects, theme3_paras = extract_projects_and_text(split_sections[2])
            if theme3_projects:
                table = doc.add_table(rows=len(theme3_projects)+1, cols=1)
                table.style = 'Table Grid'
                # First row: empty, dark blue
                cell = table.cell(0, 0)
                cell.text = ''
                set_cell_background(cell, DARK_BLUE)
                # Project rows: alternate white and light blue, starting from second project row (i=1)
                for i, proj in enumerate(theme3_projects):
                    cell = table.cell(i+1, 0)
                    cell.text = ''  # Clear cell
                    if ":" in proj:
                        before, after = proj.split(":", 1)
                        run = cell.paragraphs[0].add_run(before + ":")
                        run.bold = True
                        run.font.name = 'Century Gothic'
                        run.font.size = Pt(11)
                        run2 = cell.paragraphs[0].add_run(after)
                        run2.font.name = 'Century Gothic'
                        run2.font.size = Pt(11)
                    else:
                        run = cell.paragraphs[0].add_run(proj.strip())
                        run.font.name = 'Century Gothic'
                        run.font.size = Pt(11)
                    if i == 0:
                        color = WHITE
                    elif i % 2 == 1:
                        color = LIGHT_BLUE
                    else:
                        color = WHITE
                    set_cell_background(cell, color)
                doc.add_paragraph('')  # Add space after table
            for para in theme3_paras:
                p = doc.add_paragraph(para)
                for run in p.runs:
                    run.font.name = 'Century Gothic'
                    run.font.size = Pt(11)
                p.style = doc.styles['Normal']

        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button(
            label="Download as Word Document",
            data=buffer,
            file_name="KM_CV_Qualifications.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    elif uploaded_file:
        st.info("👆 Please describe the work requirements to analyze the CV")
    elif work_description:
        st.info("👆 Please upload a CV file to analyze")


# ---- EXISTING CODE CONTINUES (All the original project search functionality) ----

# Complete list of all countries from your images
KNOWN_COUNTRIES = {
    # A
    'argentina': ['argentina', 'argentinian'],
    'armenia': ['armenia', 'armenian'],
    'aruba': ['aruba'],
    'austria': ['austria', 'austrian'],
    'azerbaijan': ['azerbaijan'],
    
    # B
    'bahamas': ['bahamas', 'the bahamas'],
    'bangladesh': ['bangladesh'],
    'barbados': ['barbados'],
    'belize': ['belize'],
    'bolivia': ['bolivia', 'bolivian'],
    'bosnia herzegovina': ['bosnia herzegovina', 'bosnia', 'herzegovina'],
    'botswana': ['botswana'],
    'brazil': ['brazil', 'brazilian'],
    'bulgaria': ['bulgaria', 'bulgarian'],
    
    # C
    'canada': ['canada', 'canadian'],
    'caribbean': ['caribbean'],
    'cayman islands': ['cayman islands', 'cayman'],
    'chile': ['chile', 'chilean'],
    'china': ['china', 'chinese'],
    'colombia': ['colombia', 'colombian'],
    'costa rica': ['costa rica', 'costa rican'],
    'côte d\'ivoire': ['côte d\'ivoire', 'cote d\'ivoire', 'ivory coast'],
    'curacao': ['curacao', 'curaçao'],
    'czech republic': ['czech republic', 'czech'],
    
    # D
    'dominica': ['dominica'],
    'dominican republic': ['dominican republic', 'dominican', 'dr'],
    
    # E
    'ecuador': ['ecuador', 'ecuadorian'],
    'egypt': ['egypt', 'egyptian'],
    'el salvador': ['el salvador', 'salvador', 'salvadoran'],
    'estonia': ['estonia', 'estonian'],
    
    # G
    'gabon': ['gabon'],
    'germany': ['germany', 'german'],
    'ghana': ['ghana', 'ghanaian'],
    'global': ['global', 'worldwide', 'international'],
    'grenada': ['grenada'],
    'guam': ['guam'],
    'guatemala': ['guatemala', 'guatemalan'],
    'guinea': ['guinea'],
    'guinea-bissau': ['guinea-bissau', 'guinea bissau'],
    'guyana': ['guyana'],
    
    # H
    'haiti': ['haiti', 'haitian'],
    'honduras': ['honduras', 'honduran'],
    'hungary': ['hungary', 'hungarian'],
    
    # I
    'india': ['india', 'indian'],
    'indonesia': ['indonesia', 'indonesian'],
    
    # J
    'jamaica': ['jamaica', 'jamaican'],
    'jordan': ['jordan', 'jordanian'],
    
    # K
    'kenya': ['kenya', 'kenyan'],
    'korea': ['korea', 'korean', 'south korea'],
    'kyrgyz republic': ['kyrgyz republic', 'kyrgyzstan'],
    
    # L
    'laos': ['laos', 'lao'],
    'lebanon': ['lebanon', 'lebanese'],
    'liberia': ['liberia', 'liberian'],
    'lithuania': ['lithuania', 'lithuanian'],
    
    # M
    'madagascar': ['madagascar'],
    'malawi': ['malawi'],
    'malaysia': ['malaysia', 'malaysian'],
    'maldives': ['maldives'],
    'mauritania': ['mauritania', 'mauritanian'],
    'mauritius': ['mauritius'],
    'mexico': ['mexico', 'mexican'],
    'mongolia': ['mongolia', 'mongolian'],
    'morocco': ['morocco', 'moroccan'],
    
    # N
    'namibia': ['namibia', 'namibian'],
    'nepal': ['nepal', 'nepalese'],
    'netherlands': ['netherlands', 'dutch'],
    'nigeria': ['nigeria', 'nigerian'],
    
    # O
    'oman': ['oman'],
    
    # P
    'pakistan': ['pakistan', 'pakistani'],
    'panama': ['panama', 'panamanian'],
    'peru': ['peru', 'peruvian'],
    'philippines': ['philippines', 'philippine', 'filipino'],
    'poland': ['poland', 'polish'],
    'puerto rico': ['puerto rico'],
    
    # R
    'romania': ['romania', 'romanian'],
    'russia': ['russia', 'russian'],
    
    # S
    'saudi arabia': ['saudi arabia', 'saudi'],
    'senegal': ['senegal', 'senegalese'],
    'sierra leone': ['sierra leone'],
    'singapore': ['singapore'],
    'sint maarten': ['sint maarten', 'st maarten'],
    'slovakia': ['slovakia', 'slovak'],
    'south africa': ['south africa', 'south african'],
    'southern africa': ['southern africa'],
    'sri lanka': ['sri lanka', 'sri lankan'],
    'st kitts and nevis': ['st kitts and nevis', 'st. kitts & nevis', 'saint kitts and nevis'],
    'st lucia': ['st lucia', 'st. lucia', 'saint lucia'],
    'st vincent and the grenadines': ['st vincent and the grenadines', 'st. vincent & the grenadines', 'saint vincent and the grenadines'],
    'sub-saharan africa': ['sub-saharan africa', 'sub saharan africa'],
    'suriname': ['suriname'],
    'swaziland': ['swaziland', 'eswatini'],
    
    # T
    'tanzania': ['tanzania', 'tanzanian'],
    'thailand': ['thailand', 'thai'],
    'togo': ['togo'],
    'trinidad and tobago': ['trinidad and tobago', 'trinidad', 'tobago', 'tt'],
    'tunisia': ['tunisia', 'tunisian'],
    'turkey': ['turkey', 'turkish'],
    'turks and caicos': ['turks and caicos', 'turks & caicos', 'tci', 'turks & caicos islands'],
    
    # U
    'uganda': ['uganda', 'ugandan'],
    'united states': ['united states', 'usa', 'us', 'america', 'american'],
    'uruguay': ['uruguay', 'uruguayan'],
    
    # V
    'venezuela': ['venezuela', 'venezuelan'],
    'vietnam': ['vietnam', 'vietnamese'],
    
    # Y
    'yemen': ['yemen', 'yemeni'],
    
    # Z
    'zambia': ['zambia', 'zambian'],
    'zimbabwe': ['zimbabwe', 'zimbabwean']
}

# Enhanced region mapping with proper country associations based on World Bank regions
KNOWN_REGIONS = {
    'east asia and pacific': {
        'variations': ['east asia and pacific', 'east asia', 'pacific', 'eap', 'asia pacific'],
        'countries': ['china', 'indonesia', 'korea', 'laos', 'malaysia', 'mongolia', 'philippines', 
                     'thailand', 'vietnam', 'guam']
    },
    'europe and central asia': {
        'variations': ['europe and central asia', 'europe', 'central asia', 'eca', 'european'],
        'countries': ['austria', 'azerbaijan', 'bosnia herzegovina', 'bulgaria', 'czech republic', 
                     'estonia', 'germany', 'hungary', 'kyrgyz republic', 'lithuania', 'netherlands', 
                     'poland', 'romania', 'russia', 'slovakia', 'turkey']
    },
    'latin america and the caribbean': {
        'variations': ['latin america and the caribbean', 'latin america', 'caribbean', 'lac', 'latam'],
        'countries': ['argentina', 'aruba', 'bahamas', 'barbados', 'belize', 'bolivia', 'brazil', 
                     'chile', 'colombia', 'costa rica', 'curacao', 'dominica', 'dominican republic', 
                     'ecuador', 'el salvador', 'grenada', 'guatemala', 'guyana', 'haiti', 'honduras', 
                     'jamaica', 'mexico', 'panama', 'peru', 'puerto rico', 'sint maarten', 
                     'st kitts and nevis', 'st lucia', 'st vincent and the grenadines', 'suriname', 
                     'trinidad and tobago', 'turks and caicos', 'uruguay', 'venezuela']
    },
    'middle east and north africa': {
        'variations': ['middle east and north africa', 'middle east', 'north africa', 'mena', 'mea'],
        'countries': ['egypt', 'jordan', 'lebanon', 'morocco', 'oman', 'saudi arabia', 'tunisia', 'yemen']
    },
    'north america': {
        'variations': ['north america', 'na', 'north american'],
        'countries': ['canada', 'united states']
    },
    'south asia': {
        'variations': ['south asia', 'sa', 'south asian'],
        'countries': ['bangladesh', 'india', 'maldives', 'nepal', 'pakistan', 'sri lanka']
    },
    'sub saharan africa': {
        'variations': ['sub saharan africa', 'sub-saharan africa', 'africa', 'ssa', 'sub saharan', 'southern africa'],
        'countries': ['botswana', 'côte d\'ivoire', 'gabon', 'ghana', 'guinea', 'guinea-bissau', 
                     'kenya', 'liberia', 'madagascar', 'malawi', 'mauritania', 'mauritius', 
                     'namibia', 'nigeria', 'senegal', 'sierra leone', 'south africa', 'swaziland', 
                     'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe']
    },
    'global': {
        'variations': ['global', 'worldwide', 'international'],
        'countries': ['global']
    }
}

# Enhanced services with more variations and alternative wordings
KNOWN_SERVICES = {
    'due diligence': ['due diligence', 'dd', 'due-diligence', 'technical due diligence', 'commercial due diligence', 
                     'financial due diligence', 'environmental due diligence', 'diligence'],
    'feasibility study': ['feasibility study', 'feasibility', 'fs', 'feas study', 'feasibility analysis', 
                         'technical feasibility', 'economic feasibility', 'pre-feasibility'],
    "lender's engineer": ["lender's engineer", "lenders engineer", 'le', 'lender engineer', 'independent engineer',
                         'lenders technical advisor', 'lta'],
    "owner's engineer": ["owner's engineer", "owners engineer", 'oe', 'owner engineer', 'owners representative',
                        'project management', 'construction supervision'],
    'policy & regulation': ['policy & regulation', 'policy and regulation', 'policy', 'regulation', 'regulatory',
                           'policy analysis', 'regulatory framework', 'compliance'],
    'project development': ['project development', 'development', 'dev', 'project dev', 'business development',
                           'project planning', 'project structuring'],
    'transaction advisory': ['transaction advisory', 'transaction', 'advisory', 'ta', 'financial advisory',
                            'investment advisory', 'deal advisory', 'merger', 'acquisition']
}

# Enhanced sectors with STRICTER matching - only exact sector matches
KNOWN_SECTORS = {
    'renewable energy': ['renewable energy', 'renewable', 'renewables', 're', 'clean energy', 'green energy',
                        'sustainable energy', 'alternative energy'],
    'conventional energy': ['conventional energy', 'conventional', 'traditional energy', 'ce', 'fossil',
                           'fossil fuel', 'thermal power', 'conventional power'],
    'energy storage': ['energy storage', 'storage', 'battery storage', 'es', 'energy storage system',
                      'grid storage', 'utility storage'],
    'hydrogen': ['hydrogen', 'h2', 'green hydrogen', 'blue hydrogen', 'hydrogen energy', 'hydrogen power',
                'hydrogen production', 'hydrogen economy'],
    'lng to power': ['lng to power', 'lng power', 'lng-to-power', 'lng2power', 'lng'], 
    'other energy': ['other energy', 'misc energy', 'miscellaneous energy', 'mixed energy'],
    'water & wastewater': ['water & wastewater', 'water and wastewater', 'water', 'wastewater', 'ww',
                          'water treatment', 'sewage treatment', 'water infrastructure'],
    'other infrastructure': ['other infrastructure', 'infrastructure', 'other infra', 'infra',
                            'civil infrastructure', 'public infrastructure']
}

# Enhanced technologies with STRICTER matching - only exact technology matches
KNOWN_TECHNOLOGIES = {
    'wind': ['wind', 'wind power', 'wind energy', 'onshore wind', 'offshore wind', 'wind turbine',
            'wind farm', 'eolic'],
    'solar': ['solar', 'solar power', 'solar energy', 'pv', 'photovoltaic', 'solar pv', 'solar panel',
             'solar farm', 'solar plant', 'csp', 'concentrated solar power'],
    'hydro': ['hydro', 'hydropower', 'hydroelectric', 'hydro power', 'hydroelectric power',
             'small hydro', 'mini hydro', 'run of river'],
    'ulsd/diesel': ['ulsd/diesel', 'ulsd', 'diesel', 'ultra low sulfur diesel', 'diesel generator',
                   'diesel power', 'diesel engine'],
    'hfo': ['hfo', 'heavy fuel oil', 'fuel oil', 'residual fuel oil'],
    'others': ['others', 'other', 'misc', 'miscellaneous', 'mixed technology'],
    'nuclear': ['nuclear', 'nuclear power', 'nuclear energy', 'atomic power', 'nuclear plant'],
    'natural gas': ['natural gas', 'gas', 'ng', 'nat gas', 'gas turbine', 'gas engine', 'ccgt',
                   'combined cycle', 'gas power', 'liquefied natural gas'],
    'coal': ['coal', 'coal power', 'coal energy', 'coal plant', 'coal fired', 'thermal coal'],
    'green hydrogen': ['green hydrogen', 'green h2', 'renewable hydrogen', 'electrolytic hydrogen'],
    'geothermal': ['geothermal', 'geothermal energy', 'geothermal power', 'geothermal plant'],
    'bess': ['bess', 'battery energy storage system', 'battery storage', 'battery', 'lithium battery',
            'grid battery', 'utility battery'],
    'biomass': ['biomass', 'biomass energy', 'biomass power', 'bio energy', 'biofuel', 'biogas',
               'waste to energy', 'bagasse']
}

# ---- NEW: Client List Generation and Fuzzy Matching ----

def generate_client_list(projects: List[Dict]) -> List[str]:
    """Generate a comprehensive list of unique clients from the project data"""
    clients = set()
    
    for project in projects:
        # Get client from main field
        if project.get('client'):
            clients.add(project['client'].strip())
        
        # Get client from docx data
        if project.get('docx_data', {}).get('docx_client'):
            clients.add(project['docx_data']['docx_client'].strip())
    
    # Clean and filter clients
    cleaned_clients = []
    for client in clients:
        if client and len(client) > 2:  # Filter out very short or empty entries
            cleaned_clients.append(client)
    
    return sorted(list(set(cleaned_clients)))

def fuzzy_match_clients(query_term: str, client_list: List[str], threshold: float = 0.6) -> List[str]:
    """Fuzzy match client names using sequence matching"""
    query_lower = query_term.lower().strip()
    matches = []
    
    for client in client_list:
        client_lower = client.lower()
        
        # Exact match
        if query_lower == client_lower:
            matches.append(client)
            continue
        
        # Partial match
        if query_lower in client_lower or client_lower in query_lower:
            matches.append(client)
            continue
        
        # Fuzzy match using sequence matcher
        similarity = SequenceMatcher(None, query_lower, client_lower).ratio()
        if similarity >= threshold:
            matches.append(client)
            continue
        
        # Word-level fuzzy matching
        query_words = query_lower.split()
        client_words = client_lower.split()
        
        for query_word in query_words:
            if len(query_word) >= 3:  # Skip very short words
                for client_word in client_words:
                    if len(client_word) >= 3:
                        word_similarity = SequenceMatcher(None, query_word, client_word).ratio()
                        if word_similarity >= threshold:
                            matches.append(client)
                            break
                if client in matches:
                    break
    
    return list(set(matches))  # Remove duplicates

# ---- SPECIALIZED SEARCH FUNCTIONS ----

def country_region_search(query_terms: List[str], projects: List[Dict]) -> List[Dict]:
    """Specialized search for country/region queries - searches only in location fields"""
    filtered_projects = []
    
    for project in projects:
        # Build location search fields
        location_fields = []
        
        # Add country field
        if project.get('country'):
            location_fields.append(project['country'])
        
        # Add regions field
        if project.get('regions'):
            if isinstance(project['regions'], list):
                location_fields.extend(project['regions'])
            else:
                location_fields.append(project['regions'])
        
        # Add docx location data
        if project.get('docx_data', {}).get('docx_country'):
            location_fields.append(project['docx_data']['docx_country'])
        
        # Check if any query term matches any location field
        for query_term in query_terms:
            for location_field in location_fields:
                if advanced_fuzzy_match(query_term, location_field):
                    filtered_projects.append(project)
                    break
            if project in filtered_projects:
                break
    
    return filtered_projects

def sector_technology_search(query_terms: List[str], projects: List[Dict]) -> List[Dict]:
    """FIXED: Handle multiple sectors/technologies with strict LNG matching"""
    filtered_projects = []
    
    for project in projects:
        project_matches = False
        
        # For LNG searches, be VERY strict
        if any('lng' in term.lower() for term in query_terms):
            # Check if project has LNG to Power sector specifically
            has_lng_sector = False
            
            if project.get('sectors'):
                sectors = project['sectors'] if isinstance(project['sectors'], list) else [project['sectors']]
                for sector in sectors:
                    if 'lng to power' in sector.lower():
                        has_lng_sector = True
                        break
            
            # Check project name for LNG context
            has_lng_name = False
            if project.get('project_name'):
                project_name = project['project_name'].lower()
                if 'lng' in project_name:
                    has_lng_name = True
            
            # Only include if project explicitly has LNG sector OR LNG in name
            if has_lng_sector or has_lng_name:
                project_matches = True
        
        else:
            # For non-LNG searches, use broader matching
            technical_fields = []
            
            # Add project name
            if project.get('project_name'):
                technical_fields.append(project['project_name'])
            
            # Add all sectors (handle array)
            if project.get('sectors'):
                sectors = project['sectors'] if isinstance(project['sectors'], list) else [project['sectors']]
                technical_fields.extend(sectors)
            
            # Add all services (handle array)
            if project.get('services'):
                services = project['services'] if isinstance(project['services'], list) else [project['services']]
                technical_fields.extend(services)
            
            # Add all technologies (handle complex structure)
            if project.get('technologies'):
                for tech in project['technologies']:
                    if isinstance(tech, dict):
                        if tech.get('type'):
                            technical_fields.append(tech['type'])
                        if tech.get('name'):
                            technical_fields.append(tech['name'])
                    elif isinstance(tech, str):
                        technical_fields.append(tech)
            
            # Check if any query term matches any technical field
            for query_term in query_terms:
                for technical_field in technical_fields:
                    if advanced_fuzzy_match(query_term, technical_field):
                        project_matches = True
                        break
                if project_matches:
                    break
        
        if project_matches:
            filtered_projects.append(project)
    
    return filtered_projects

def client_search(query_terms: List[str], projects: List[Dict], client_list: List[str]) -> List[Dict]:
    """Specialized search for client queries - uses fuzzy matching on client list"""
    # First, find matching clients using fuzzy search
    matched_clients = []
    for query_term in query_terms:
        matched_clients.extend(fuzzy_match_clients(query_term, client_list))
    
    if not matched_clients:
        return []
    
    # Then filter projects by matched clients
    filtered_projects = []
    
    for project in projects:
        # Build client search fields
        client_fields = []
        
        if project.get('client'):
            client_fields.append(project['client'])
        
        if project.get('docx_data', {}).get('docx_client'):
            client_fields.append(project['docx_data']['docx_client'])
        
        # Check if any matched client appears in project client fields
        for matched_client in matched_clients:
            for client_field in client_fields:
                if advanced_fuzzy_match(matched_client, client_field):
                    filtered_projects.append(project)
                    break
            if project in filtered_projects:
                break
    
    return filtered_projects

# ---- CATEGORY DETECTION FUNCTIONS ----

def detect_query_category(query: str) -> Dict[str, List[str]]:
    """
    Enhanced with STRICTER LNG detection
    """
    query_lower = query.lower()
    words = query_lower.split()
    
    detected = {
        'countries': [],
        'regions': [],
        'sectors': [],
        'technologies': [],
        'services': [],
        'clients': [],
        'other_terms': []
    }
    
    # SPECIAL CASE: LNG detection
    if 'lng' in query_lower:
        detected['sectors'].append('lng to power')
        detected['technologies'].append('natural gas')
        # Remove 'lng' from further processing to avoid confusion
        words = [w for w in words if w != 'lng']
    
    # Check for multi-word phrases first (up to 4 words)
    for i in range(len(words)):
        for j in range(i+1, min(i+5, len(words)+1)):
            phrase = ' '.join(words[i:j])
            
            # Country detection - exact matching
            for country_name, variations in KNOWN_COUNTRIES.items():
                if phrase in variations:
                    detected['countries'].append(country_name)
                    break
            
            # Region detection - exact matching
            for region_name, region_data in KNOWN_REGIONS.items():
                if phrase in region_data['variations']:
                    detected['regions'].append(region_name)
                    # Also add all countries in this region
                    detected['countries'].extend(region_data['countries'])
                    break
            
            # Sector detection - strict matching
            for sector_name, variations in KNOWN_SECTORS.items():
                if phrase in variations:
                    detected['sectors'].append(sector_name)
                    break
            
            # Technology detection - strict matching
            for tech_name, variations in KNOWN_TECHNOLOGIES.items():
                if phrase in variations:
                    detected['technologies'].append(tech_name)
                    break
            
            # Service detection
            for service_name, variations in KNOWN_SERVICES.items():
                if phrase in variations:
                    detected['services'].append(service_name)
                    break
    
    # Check individual words
    for word in words:
        if len(word) <= 2:
            continue
        
        # Skip if already found in phrases
        already_found = (word in ' '.join(detected['countries']) or 
                        word in ' '.join(detected['regions']) or
                        word in ' '.join(detected['sectors']) or
                        word in ' '.join(detected['technologies']) or
                        word in ' '.join(detected['services']))
        
        if already_found:
            continue
        
        # Country detection
        for country_name, variations in KNOWN_COUNTRIES.items():
            if word in variations:
                detected['countries'].append(country_name)
                break
        
        # Region detection
        for region_name, region_data in KNOWN_REGIONS.items():
            if word in region_data['variations']:
                detected['regions'].append(region_name)
                detected['countries'].extend(region_data['countries'])
                break
        
        # Sector detection
        for sector_name, variations in KNOWN_SECTORS.items():
            if word in variations:
                detected['sectors'].append(sector_name)
                break
        
        # Technology detection
        for tech_name, variations in KNOWN_TECHNOLOGIES.items():
            if word in variations:
                detected['technologies'].append(tech_name)
                break
        
        # Service detection
        for service_name, variations in KNOWN_SERVICES.items():
            if word in variations:
                detected['services'].append(service_name)
                break
        
        # Client detection (look for uppercase words or known patterns)
        if len(word) > 2 and (word.isupper() or word.istitle()):
            # Common client abbreviations
            client_mappings = {
                'usaid': 'USAID',
                'worldbank': 'World Bank',
                'wb': 'World Bank',
                'ifc': 'IFC',
                'adb': 'Asian Development Bank',
                'afdb': 'African Development Bank',
                'iadb': 'Inter-American Development Bank'
            }
            
            if word.lower() in client_mappings:
                detected['clients'].append(client_mappings[word.lower()])
            elif word not in ['Give', 'Show', 'Find', 'List', 'Projects', 'Me', 'In', 'For', 'The', 'And', 'Of', 'To']:
                detected['clients'].append(word)
    
    # Remove duplicates
    for key in detected:
        detected[key] = list(set(detected[key]))
    
    return detected

# ---- MAIN SEARCH ROUTING FUNCTION ----

def intelligent_project_search_v2(query: str, projects: List[Dict]) -> tuple:
    """
    Enhanced intelligent project search with specialized routing
    """
    # Generate client list
    client_list = generate_client_list(projects)
    
    # Detect query categories
    detected = detect_query_category(query)
    
    # Determine primary search method based on detected categories
    search_results = []
    search_method = "combined"
    
    # Priority: Location > Technical > Client > Combined
    if detected['countries'] or detected['regions']:
        # Location-focused search
        location_terms = detected['countries'] + detected['regions']
        search_results = country_region_search(location_terms, projects)
        search_method = "location"
        
        # If we also have technical terms, further filter by technical criteria
        if detected['sectors'] or detected['technologies'] or detected['services']:
            technical_terms = detected['sectors'] + detected['technologies'] + detected['services']
            technical_results = sector_technology_search(technical_terms, projects)
            # Intersection of location and technical results
            search_results = [p for p in search_results if p in technical_results]
            search_method = "location + technical"
        
        # If we also have client terms, further filter by client
        if detected['clients']:
            client_results = client_search(detected['clients'], projects, client_list)
            search_results = [p for p in search_results if p in client_results]
            search_method = "location + technical + client"
    

    elif detected['sectors'] or detected['technologies'] or detected['services']:
        # Technical-focused search
        technical_terms = detected['sectors'] + detected['technologies'] + detected['services']
        search_results = sector_technology_search(technical_terms, projects)
        search_method = "technical"
        
        # If we also have client terms, further filter by client
        if detected['clients']:
            client_results = client_search(detected['clients'], projects, client_list)
            search_results = [p for p in search_results if p in client_results]
            search_method = "technical + client"
    
    elif detected['clients']:
        # Client-focused search
        search_results = client_search(detected['clients'], projects, client_list)
        search_method = "client"
    
    else:
        # Fallback to original combined search
        search_results = enhanced_project_lookup_function(
            projects=projects,
            country_filter=detected['countries'] if detected['countries'] else None,
            region_filter=detected['regions'] if detected['regions'] else None,
            service_filter=detected['services'] if detected['services'] else None,
            sector_filter=detected['sectors'] if detected['sectors'] else None,
            technology_filter=detected['technologies'] if detected['technologies'] else None,
            client_filter=detected['clients'] if detected['clients'] else None,
            other_terms_filter=detected['other_terms'] if detected['other_terms'] else None
        )
        search_method = "fallback"
    # Combine criteria for display
    all_tech_criteria = detected['services'] + detected['sectors'] + detected['technologies']
    all_location_criteria = detected['countries'] + detected['regions']
    
    return search_results, {
        'locations': all_location_criteria,
        'technical': all_tech_criteria,
        'clients': detected['clients'],
        'other_terms': detected['other_terms'],
        'search_method': search_method,
        'detected_details': detected,
        'available_clients': len(client_list)
    }

# ---- GLOBAL HELPER FUNCTIONS (keeping existing ones) ----

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    return text.lower().strip()

def advanced_fuzzy_match(search_term: str, target_text: str) -> bool:
    """Enhanced fuzzy matching with better logic - MOVED TO GLOBAL SCOPE"""
    search_norm = normalize_text(search_term)
    target_norm = normalize_text(target_text)
    
    if not search_norm or not target_norm:
        return False
    
    # Exact match
    if search_norm == target_norm:
        return True
    
    # Handle abbreviations and common variations
    abbreviation_mappings = {
        'usa': 'united states',
        'us': 'united states',
        'dr': 'dominican republic',
        'tt': 'trinidad and tobago',
        'tci': 'turks and caicos'
    }
    
    if search_norm in abbreviation_mappings:
        search_norm = abbreviation_mappings[search_norm]
    if target_norm in abbreviation_mappings:
        target_norm = abbreviation_mappings[target_norm]
    
    # Partial match with minimum length requirement
    if len(search_norm) >= 3 and len(target_norm) >= 3:
        if search_norm in target_norm or target_norm in search_norm:
            return True
    
    # Word-level match
    search_words = search_norm.split()
    target_words = target_norm.split()
    
    for search_word in search_words:
        if len(search_word) >= 3:  # Skip very short words
            for target_word in target_words:
                if len(target_word) >= 3:
                    if search_word == target_word or search_word in target_word or target_word in search_word:
                        return True
    
    return False

def enhanced_project_lookup_function(
    projects: List[Dict], 
    country_filter: Optional[List[str]] = None,
    region_filter: Optional[List[str]] = None,
    service_filter: Optional[List[str]] = None,
    sector_filter: Optional[List[str]] = None,
    technology_filter: Optional[List[str]] = None,
    client_filter: Optional[List[str]] = None,
    other_terms_filter: Optional[List[str]] = None
) -> List[Dict]:
    """
    Enhanced project lookup with improved fuzzy matching and multiple criteria support
    """
    
    def check_location_match(project: Dict, country_filters: List[str], region_filters: List[str]) -> bool:
        """Enhanced location matching with better coverage"""
        if not country_filters and not region_filters:
            return True
            
        search_fields = []
        
        # Add all location-related fields
        location_fields = ['country', 'regions']
        for field in location_fields:
            if project.get(field):
                if isinstance(project[field], list):
                    search_fields.extend(project[field])
                else:
                    search_fields.append(project[field])
        
        # Add docx location data
        if project.get('docx_data'):
            docx_data = project['docx_data']
            for field in ['docx_client', 'docx_country']:
                if docx_data.get(field):
                    search_fields.append(docx_data[field])
        
        # Check country matches
        if country_filters:
            for country_filter in country_filters:
                for field in search_fields:
                    if advanced_fuzzy_match(country_filter, field):
                        return True
        
        # Check region matches
        if region_filters:
            for region_filter in region_filters:
                for field in search_fields:
                    if advanced_fuzzy_match(region_filter, field):
                        return True
        
        return False
    
    def check_technical_match(project: Dict, service_filters: List[str], sector_filters: List[str], tech_filters: List[str]) -> bool:
        """Enhanced technical criteria matching"""
        if not service_filters and not sector_filters and not tech_filters:
            return True
            
        search_fields = []
        
        # Add project name (often contains technical info)
        if project.get('project_name'):
            search_fields.append(project['project_name'])
        
        # Add technical fields
        technical_fields = ['sectors', 'services']
        for field in technical_fields:
            if project.get(field):
                if isinstance(project[field], list):
                    search_fields.extend(project[field])
                else:
                    search_fields.append(project[field])
        
        # Add technologies (handle complex structure)
        if project.get('technologies'):
            for tech in project['technologies']:
                if isinstance(tech, dict):
                    if tech.get('type'):
                        search_fields.append(tech['type'])
                    if tech.get('name'):
                        search_fields.append(tech['name'])
                elif isinstance(tech, str):
                    search_fields.append(tech)
        
        # Check service matches
        if service_filters:
            for service_filter in service_filters:
                for field in search_fields:
                    if advanced_fuzzy_match(service_filter, field):
                        return True
        
        # Check sector matches
        if sector_filters:
            for sector_filter in sector_filters:
                for field in search_fields:
                    if advanced_fuzzy_match(sector_filter, field):
                        return True
        
        # Check technology matches
        if tech_filters:
            for tech_filter in tech_filters:
                for field in search_fields:
                    if advanced_fuzzy_match(tech_filter, field):
                        return True
        
        return False
    
    def check_client_match(project: Dict, client_filters: List[str]) -> bool:
        """Enhanced client matching"""
        if not client_filters:
            return True
            
        search_fields = []
        
        # Add client-related fields
        if project.get('client'):
            search_fields.append(project['client'])
        
        if project.get('docx_data'):
            docx_data = project['docx_data']
            client_fields = ['docx_client', 'docx_country']
            for field in client_fields:
                if docx_data.get(field):
                    search_fields.append(docx_data[field])
            
            # Add assignment data
            if docx_data.get('assignment'):
                assignment = docx_data['assignment']
                for field in ['name', 'description']:
                    if assignment.get(field):
                        search_fields.append(assignment[field])
        
        # Check for matches
        for client_filter in client_filters:
            for field in search_fields:
                if advanced_fuzzy_match(client_filter, field):
                    return True
        
        return False
    
    def check_other_terms_match(project: Dict, other_filters: List[str]) -> bool:
        """Enhanced general term matching"""
        if not other_filters:
            return True
            
        # Combine all searchable text from project
        all_text = []
        
        # Basic fields
        basic_fields = ['project_name', 'country', 'client']
        for field in basic_fields:
            if project.get(field):
                all_text.append(project[field])
        
        # List fields
        list_fields = ['sectors', 'services', 'regions']
        for field in list_fields:
            if project.get(field):
                if isinstance(project[field], list):
                    all_text.extend(project[field])
                else:
                    all_text.append(project[field])
        
        # Technologies
        if project.get('technologies'):
            for tech in project['technologies']:
                if isinstance(tech, dict):
                    for tech_field in ['type', 'name']:
                        if tech.get(tech_field):
                            all_text.append(tech[tech_field])
                elif isinstance(tech, str):
                    all_text.append(tech)
        
        # DOCX data
        if project.get('docx_data'):
            docx_data = project['docx_data']
            docx_fields = ['docx_client', 'docx_country', 'role']
            for field in docx_fields:
                if docx_data.get(field):
                    all_text.append(docx_data[field])
            
            if docx_data.get('assignment'):
                assignment = docx_data['assignment']
                for field in ['name', 'description']:
                    if assignment.get(field):
                        all_text.append(assignment[field])
        

        combined_text = ' '.join(all_text).lower()
        
        for term in other_filters:
            term_norm = normalize_text(term)
            if len(term_norm) >= 3 and term_norm in combined_text:
                return True
        
        return False
    
    # Apply filters with OR logic within each category, AND logic between categories
    filtered_projects = []
    
    for project in projects:
        matches_location = check_location_match(project, country_filter, region_filter)
        matches_technical = check_technical_match(project, service_filter, sector_filter, technology_filter)
        matches_client = check_client_match(project, client_filter)
        matches_other = check_other_terms_match(project, other_terms_filter)
        
        # Project must match ALL specified criteria categories
        if matches_location and matches_technical and matches_client and matches_other:
            filtered_projects.append(project)
    
    return filtered_projects

# ---- Keep all your existing formatting functions unchanged ----

def format_project_details(project: Dict) -> str:
    """Format project details for clear, visually appealing display"""
    
    # Basic info
    project_id = project.get('project_id') or project.get('job_number', 'N/A')
    project_name = project.get('project_name', 'Unnamed Project')
    
    # Start with a clear header
    details = []
    details.append(f"🏗️ **{project_name}**")
    details.append(f"📋 **ID:** {project_id}")
    details.append("")  # Empty line for spacing
    
    # Core project information in a structured format
    details.append("**📍 PROJECT DETAILS:**")
    
    if project.get('country'):
        details.append(f"   • **Country:** {project['country']}")
        
    if project.get('client'):
        details.append(f"   • **Client:** {project['client']}")
        
    if project.get('year_start') or project.get('year_end'):
        date_range = f"{project.get('year_start', 'N/A')} - {project.get('year_end', 'Present')}"
        details.append(f"   • **Duration:** {date_range}")
    
    if project.get('contract_value'):
        details.append(f"   • **Contract Value:** ${project['contract_value']}")
        
    if project.get('mw_total'):
        details.append(f"   • **Capacity:** {project['mw_total']} MW")
    
    details.append("")  # Empty line
    
    # Technical specifications
    details.append("**⚡ TECHNICAL SPECIFICATIONS:**")
    
    if project.get('sectors'):
        sectors_formatted = ', '.join(project['sectors']) if isinstance(project['sectors'], list) else project['sectors']
        details.append(f"   • **Sectors:** {sectors_formatted}")
        
    if project.get('technologies'):
        tech_list = []
        for tech in project['technologies']:
            if isinstance(tech, dict):
                tech_str = tech.get('type', '')
                if tech.get('capacity'):
                    tech_str += f" ({tech['capacity']})"
                tech_list.append(tech_str)
            else:
                tech_list.append(str(tech))
        if tech_list:
            tech_formatted = ', '.join(tech_list)
            details.append(f"   • **Technologies:** {tech_formatted}")
    
    if project.get('services'):
        services_formatted = ', '.join(project['services']) if isinstance(project['services'], list) else project['services']
        details.append(f"   • **Services:** {services_formatted}")
        
    if project.get('regions'):
        regions_formatted = ', '.join(project['regions']) if isinstance(project['regions'], list) else project['regions']
        details.append(f"   • **Regions:** {regions_formatted}")
    
    details.append("")  # Empty line
    
    # Assignment description from docx (if available)
    if project.get('docx_data', {}).get('assignment'):
        assignment = project['docx_data']['assignment']
        details.append("**📋 ASSIGNMENT DETAILS:**")
        
        if assignment.get('name'):
            # Clean up the assignment name (remove redundancy)
            assignment_name = assignment['name']
            if len(assignment_name) > 150:
                assignment_name = assignment_name[:150] + "..."
            details.append(f"   • **Scope:** {assignment_name}")
        
        if assignment.get('description'):
            description = assignment['description']
            # Format description nicely with proper line breaks
            # if len(description) > 400:
            #     description = description[:400] + "..."
            
            # Split long descriptions into readable chunks
            description_lines = description.replace('. ', '.\n   ').split('\n')
            details.append(f"   • **Description:**")
            for line in description_lines[:3]:  # Limit to first 3 sentences
                if line.strip():
                    details.append(f"     {line.strip()}")

    
    # Additional DOCX info if available
    if project.get('docx_data'):
        docx_data = project['docx_data']
        if docx_data.get('role'):
            details.append(f"   • **K&M Role:** {docx_data['role']}")
        if docx_data.get('duration', {}).get('original'):
            details.append(f"   • **Duration Details:** {docx_data['duration']['original']}")
    
    return "\n".join(details)

def format_multiple_projects(projects: List[Dict]) -> str:
    """Format multiple projects with clear separation and numbering"""
    if not projects:
        return "❌ **No projects found matching your criteria.**"
    
    formatted_projects = []
    formatted_projects.append(f"✅ **Found {len(projects)} matching project(s):**")
    formatted_projects.append("=" * 80)
    formatted_projects.append("")
    
    for i, project in enumerate(projects, 1):
        formatted_projects.append(f"## PROJECT {i}")
        formatted_projects.append(format_project_details(project))
        formatted_projects.append("")
        formatted_projects.append("-" * 80)
        formatted_projects.append("")
    
    return "\n".join(formatted_projects)

def format_search_summary(search_criteria: Dict, num_results: int) -> str:
    """Format search criteria summary with enhanced method display"""
    summary = []
    summary.append("🔍 **SEARCH CRITERIA APPLIED:**")
    summary.append("")
    
    # Show search method used
    if search_criteria.get('search_method'):
        method_icons = {
            'location': '🌍',
            'technical': '⚡',
            'client': '🏢',
            'location + technical': '🌍⚡',
            'location + technical + client': '🌍⚡🏢',
            'technical + client': '⚡🏢',
            'combined': '🔄',
            'fallback': '🔄'
        }
        method = search_criteria['search_method']
        icon = method_icons.get(method, '🔍')
        summary.append(f"   {icon} **Search Method:** {method.title()}")
        summary.append("")
    
    if search_criteria.get('locations'):
        summary.append(f"   🌍 **Locations:** {', '.join(search_criteria['locations'])}")
    else:
        summary.append(f"   🌍 **Locations:** Any")
        
    if search_criteria.get('technical'):
        summary.append(f"   ⚡ **Technical:** {', '.join(search_criteria['technical'])}")
    else:
        summary.append(f"   ⚡ **Technical:** Any")
        
    if search_criteria.get('clients'):
        summary.append(f"   🏢 **Clients:** {', '.join(search_criteria['clients'])}")
    else:
        summary.append(f"   🏢 **Clients:** Any")
    
    if search_criteria.get('other_terms'):
        summary.append(f"   🔤 **Other Terms:** {', '.join(search_criteria['other_terms'])}")
    
    # Show available clients count
    if search_criteria.get('available_clients'):
        summary.append(f"   📊 **Client Database:** {search_criteria['available_clients']} unique clients available")
    
    summary.append("")
    summary.append(f"📊 **Results:** {num_results} project(s) found")
    
    return "\n".join(summary)

# ---- NEW: Website Content Processing and LLM Setup ----

@st.cache_data
def load_website_content():
    """Load and process the website markdown content"""
    try:
        with open(WEBSITE_MD, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        st.error(f"Error loading website content: {e}")
        return None

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
    
def reset_vectorstore():
    """Reset the vectorstore by deleting the existing one"""
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
        print(f"Deleted existing vectorstore at {VECTORSTORE_DIR}")


def setup_vectorstore():
    """Setup vectorstore for website content"""
    if st.session_state.get('vectorstore') is not None:
        return st.session_state['vectorstore']
    
    reset_vectorstore()
    
    website_content = load_website_content()
    if not website_content:
        return None
    
    # Create documents from website content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Split the website content
    chunks = text_splitter.split_text(website_content)
    documents = [LangchainDocument(page_content=chunk) for chunk in chunks]
    
    # Setup embeddings and vectorstore
    if USE_OPENAI:
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    
    st.session_state['vectorstore'] = vectorstore
    return vectorstore

def setup_llm_chain():
    """Setup the LLM chain for general queries"""
    vectorstore = setup_vectorstore()
    if not vectorstore:
        return None
    
    # Setup LLM
    if USE_OPENAI:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
        print("Using OpenAI LLM")
    else:
        llm = ChatOllama(model="llama3", temperature=0)
    
    # Create custom prompt
    custom_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""You are an expert assistant for K&M Advisors LLC, a leading infrastructure advisory firm. 
        Use the following context from K&M's website and your knowledge to answer questions about K&M's services, expertise, and capabilities.

        Context from K&M website:
        {context}

        Chat History:
        {chat_history}

        Question: {question}

        Please provide a comprehensive and professional answer based on the context provided. If the information isn't available in the context, use your general knowledge about infrastructure advisory services, but clearly indicate when you're doing so.

        Answer:"""
    )
    
    # Create the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
    return chain

# ---- UPDATED: Process Query Functions ----

def process_project_query(user_input: str, projects: List[Dict], chat_history: List):
    """Process user query with intelligent project lookup and enhanced formatting"""
    
    # Check if this is a project lookup query
    lookup_keywords = ['projects in', 'projects for', 'show me', 'give me', 'list', 'find projects', 'tell me about projects']
    is_lookup_query = any(keyword in user_input.lower() for keyword in lookup_keywords)
    
    if is_lookup_query:
        # Use NEW intelligent project search with specialized routing
        filtered_projects, search_criteria = intelligent_project_search_v2(user_input, projects)
        
        # Format results with enhanced formatting
        if filtered_projects:
            # Create formatted response
            response_parts = []
            
            # Add search summary
            response_parts.append(format_search_summary(search_criteria, len(filtered_projects)))
            response_parts.append("")
            response_parts.append("")
            
            # Add formatted projects
            response_parts.append(format_multiple_projects(filtered_projects))
            
            formatted_response = "\n".join(response_parts)
        else:
            # No results found
            response_parts = []
            response_parts.append(format_search_summary(search_criteria, 0))
            response_parts.append("")
            response_parts.append("❌ **No projects found matching your criteria.**")
            response_parts.append("")
            response_parts.append("💡 **Suggestions:**")
            response_parts.append("   • Try broader search terms")
            response_parts.append("   • Check spelling of country/client names")
            response_parts.append("   • Use different technology keywords (e.g., 'renewable' instead of 'solar')")
            response_parts.append("   • Try abbreviations (e.g., 'RE' for Renewable Energy, 'DD' for Due Diligence)")
            response_parts.append("")
            response_parts.append("🌍 **Available Countries (Sample):**")
            response_parts.append("   • **Americas:** Argentina, Brazil, Chile, Colombia, Mexico, United States")
            response_parts.append("   • **Caribbean:** Jamaica, Dominican Republic, Trinidad & Tobago, Aruba")
            response_parts.append("   • **Africa:** Kenya, Nigeria, South Africa, Morocco, Ghana")
            response_parts.append("   • **Asia:** China, India, Indonesia, Vietnam, Philippines")
            response_parts.append("   • **Europe:** Germany, Netherlands, Poland, Turkey")
            
            # Show available clients sample if client search was attempted
            if search_criteria.get('clients'):
                client_list = generate_client_list(projects)
                response_parts.append("")
                response_parts.append("🏢 **Available Clients (Sample):**")
                sample_clients = sorted(client_list)[:10]
                for client in sample_clients:
                    response_parts.append(f"   • {client}")
                if len(client_list) > 10:
                    response_parts.append(f"   • ... and {len(client_list) - 10} more clients")
            
            formatted_response = "\n".join(response_parts)
        
        return formatted_response, search_criteria, filtered_projects
    
    else:
        # Not a project lookup query
        return None, None, None

def process_general_query(user_input: str, projects: List[Dict], chat_history: List):
    """Process general queries using project search first, then website content"""
    
    # First try project search
    project_response, criteria, filtered_projects = process_project_query(user_input, projects, chat_history)
    
    if filtered_projects and len(filtered_projects) > 0:
        # Found relevant projects - summarize them instead of listing all details
        summary_parts = []
        summary_parts.append(f"Based on your question, I found {len(filtered_projects)} relevant K&M projects:")
        summary_parts.append("")
        
        # Group projects by key characteristics for summary
        countries = set()
        sectors = set()
        services = set()
        clients = set()
        
        for project in filtered_projects[:10]:  # Limit to first 10 for summary
            if project.get('country'):
                countries.add(project['country'])
            if project.get('sectors'):
                if isinstance(project['sectors'], list):
                    sectors.update(project['sectors'])
                else:
                    sectors.add(project['sectors'])
            if project.get('services'):
                if isinstance(project['services'], list):
                    services.update(project['services'])
                else:
                    services.add(project['services'])
            if project.get('client'):
                clients.add(project['client'])
        
        # Create summary
        if countries:
            summary_parts.append(f"**Countries:** {', '.join(sorted(countries))}")
        if sectors:
            summary_parts.append(f"**Sectors:** {', '.join(sorted(sectors))}")
        if services:
            summary_parts.append(f"**Services:** {', '.join(sorted(services))}")
        if clients:
            summary_parts.append(f"**Key Clients:** {', '.join(sorted(list(clients)[:5]))}")
        
        summary_parts.append("")
        
        # Add website content context
        website_content = load_website_content()
        if website_content:
            summary_parts.append("**Additional Context from K&M's Services:**")
            summary_parts.append("")
            
            # Extract relevant sections from website content based on query
            query_lower = user_input.lower()
            if any(term in query_lower for term in ['feasibility', 'study', 'assessment']):
                summary_parts.append("K&M's **Feasibility Study** services include comprehensive technical and financial analysis, with experience across 90+ countries. We assess technology solutions, conduct site studies, and develop detailed economic models to determine project viability.")
            elif any(term in query_lower for term in ['due diligence', 'dd']):
                summary_parts.append("K&M provides **Due Diligence** services combining engineering expertise with financial analysis to assess infrastructure projects for investors and lenders.")
            elif any(term in query_lower for term in ['lender', 'engineer', 'le']):
                summary_parts.append("As **Lender's Engineer**, K&M provides independent technical oversight and monitoring services throughout project development and construction phases.")
            else:
                summary_parts.append("K&M offers comprehensive infrastructure advisory services including feasibility studies, due diligence, lender's engineer services, and project development support across power, water, and infrastructure sectors.")
        
        return "\n".join(summary_parts)
    
    else:
        # No relevant projects found, use website content with LLM
        chain = setup_llm_chain()
        if chain:
            try:
                # Format chat history for the chain
                current_history = get_current_chat_history()
                formatted_history = [(msg[0], msg[1]) for msg in current_history[-5:]]  # Last 5 exchanges

                result = chain({
                    "question": user_input,
                    "chat_history": formatted_history
                })
                
                return result["answer"]
            except Exception as e:
                return f"I apologize, but I encountered an error processing your question: {str(e)}. Please try rephrasing your question or ask about specific K&M projects."
        else:
            # Fallback to basic response
            return "I can help you with information about K&M's projects and services. Try asking about specific projects, countries, technologies, or services. For example: 'Show me renewable energy projects in Brazil' or 'What services does K&M provide?'"

# ---- Session State and UI ----

if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None

if 'project_data' not in st.session_state:
    st.session_state['project_data'] = load_project_data()

# Initialize separate chat histories for each tab
if 'chat_history_general' not in st.session_state:
    st.session_state['chat_history_general'] = []

if 'chat_history_search' not in st.session_state:
    st.session_state['chat_history_search'] = []

if 'chat_history_cv' not in st.session_state:
    st.session_state['chat_history_cv'] = []

# Keep backward compatibility
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


if 'current_question' not in st.session_state:
    st.session_state['current_question'] = ""

# Enhanced sidebar with mode-specific examples
with st.sidebar:
    if st.session_state['mode'] == 'search':
        st.subheader("💡 Smart Project Search Examples")
        
        # Updated examples showcasing new specialized search methods
        lookup_examples = [
            "Show me renewable energy projects in Brazil",
            "Find DD projects for solar technology in India",  # Location + Technical + Service
            "Give me LNG projects in United States",  # Location + Technical
            "Find solar and natural gas projects for in Brazil",  # Location + Technical
            "Show me feasibility studies in MENA region",  # Region + Service
            "Find owner's engineer projects for BESS in Germany",  # Location + Service + Technical
            "Show me projects in Nigeria or Kenya",  # Multi-location
            "Find solar projects in Morocco",  # Location + Technical
            "List conventional energy projects in Indonesia",  # Location + Sector
            "Show me projects in Caribbean region",  # Region
            "Find HFO and natural gas projects in Turkey",  # Location + Technical (strict)
            "Show me USAID renewable projects"  # Client + Technical
        ]
        
        for example in lookup_examples:
            if st.button(example, key=f"lookup_{hash(example)}", use_container_width=True):
                st.session_state['current_question'] = example
                st.rerun()
    
    elif st.session_state['mode'] == 'general':
        st.subheader("💡 General Query Examples")
        
        general_examples = [
            "What services does K&M provide?",
            "Tell me about K&M's feasibility study process",
            "What is K&M's experience in renewable energy?",
            "How does K&M conduct due diligence?",
            "What sectors does K&M work in?",
            "Describe K&M's lender's engineer services",
            "What is K&M's geographic coverage?",
            "Tell me about K&M's project development services",
            "What makes K&M different from other advisors?",
            "How does K&M assess project risks?",
            "What is K&M's approach to feasibility studies?",
            "Tell me about K&M's water sector experience"
        ]
        
        for example in general_examples:
            if st.button(example, key=f"general_{hash(example)}", use_container_width=True):
                st.session_state['current_question'] = example
                st.rerun()
    
    else:  # CV mode
        st.subheader("💡 CV Analysis Examples")
        cv_examples = [
            "Senior consultant for 100MW solar project feasibility study",
            "Project manager for LNG terminal development in Africa",
            "Due diligence expert for renewable energy investments",
            "Lender's engineer for power plant construction",
            "Financial advisor for infrastructure PPP projects",
            "Technical expert for water treatment facility",
            "Transaction advisor for energy sector M&A",
            "Regulatory consultant for power sector reform"
        ]
        
        st.info("Upload a CV file and use descriptions like:")
        for example in cv_examples:
            st.caption(f"• {example}")
    
    st.divider()
    
    # Show client database info
    if st.session_state['project_data']:
        projects = st.session_state['project_data'].get('projects', [])
        client_list = generate_client_list(projects)
        
        st.subheader("🏢 Client Database")
        st.metric("Available Clients", len(client_list))
        
    # Enhanced category reference guide with complete country list
    st.subheader("📚 Search Categories & Coverage")
    
    with st.expander("🌍 Available Countries by Region"):
        st.markdown("""
        **🌎 Latin America & Caribbean (33 countries):**
        - Argentina, Brazil, Chile, Colombia, Mexico, Peru
        - Jamaica, Dominican Republic, Trinidad & Tobago
        - Costa Rica, Panama, Ecuador, Venezuela, Uruguay
        - Aruba, Curacao, Barbados, Bahamas, Belize
        - And 14 more Caribbean nations...
        
        **🌍 Sub-Saharan Africa (24 countries):**
        - Nigeria, Kenya, South Africa, Ghana, Tanzania
        - Botswana, Namibia, Zambia, Zimbabwe, Uganda
        - Senegal, Mali, Guinea, Liberia, Sierra Leone
        - And 9 more African nations...
        
        **🌏 East Asia & Pacific (10 countries):**
        - China, Indonesia, Philippines, Vietnam, Thailand
        - Malaysia, Korea, Mongolia, Laos, Guam
        
        **🌍 Europe & Central Asia (16 countries):**
        - Germany, Netherlands, Poland, Turkey, Russia
        - Austria, Bulgaria, Czech Republic, Hungary
        - And 7 more European nations...
        
        **🌍 South Asia (6 countries):**
        - India, Pakistan, Bangladesh, Sri Lanka
        - Nepal, Maldives
        
        **🌍 Middle East & North Africa (8 countries):**
        - Saudi Arabia, Egypt, Morocco, Turkey, Yemen
        - Jordan, Lebanon, Oman, Tunisia
        
        **🌎 North America (2 countries):**
        - United States, Canada
        """)
    
    with st.expander("🌍 Regions & Abbreviations"):
        st.markdown("""
        - **EAP**: East Asia and Pacific
        - **ECA**: Europe and Central Asia  
        - **LAC**: Latin America and the Caribbean
        - **MENA**: Middle East and North Africa
        - **NA**: North America
        - **SA**: South Asia
        - **SSA**: Sub Saharan Africa
        - **Global**: Worldwide/International projects
        """)
    
    with st.expander("⚡ Sectors & Technologies"):
        st.markdown("""
        **Energy Sectors:**
        - **Renewable**: Solar, Wind, Hydro, Geothermal, Biomass
        - **Conventional**: Natural Gas, Coal, Nuclear, Oil
        - **Storage**: BESS, Grid Storage, Pumped Hydro
        - **Hydrogen**: Green H2, Blue H2, Fuel Cells
        - **LNG to Power**: Gas-to-Power, LNG Terminals
        - **Infrastructure**: Water, Wastewater, Transport
        
        **Key Technologies:**
        - Solar: PV, CSP, Distributed Solar
        - Wind: Onshore, Offshore, Small Wind
        - Gas: CCGT, OCGT, Cogeneration, LNG
        - Nuclear: Nuclear Power, Atomic Energy
        - Storage: Lithium-ion, Flow Batteries
        - Hydro: Large, Small, Pumped Storage
        """)
    
    with st.expander("📋 Services & Clients"):
        st.markdown("""
        **Services:**
        - **DD**: Due Diligence (Technical, Commercial, Environmental)
        - **FS**: Feasibility Study (Pre-feasibility, Bankability)
        - **LE**: Lender's Engineer (Independent Engineer)
        - **OE**: Owner's Engineer (Project Management)
        - **TA**: Transaction Advisory (M&A, Investment)
        - **Dev**: Project Development & Structuring
        - **Policy**: Regulatory & Policy Analysis
        
        **Client Search Features:**
        - Fuzzy matching for client names
        - Handles abbreviations (WB → World Bank)
        - Dynamic client list from database
        - Case-insensitive search
        """)
    
    st.divider()
    
    # Add comprehensive project statistics
    if st.session_state['project_data']:
        projects = st.session_state['project_data'].get('projects', [])
        
        st.subheader("📊 Database Statistics")
        st.metric("Total Projects", len(projects))
        
        # Count by regions
        region_counts = {}
        for project in projects:
            if project.get('regions'):
                regions = project['regions'] if isinstance(project['regions'], list) else [project['regions']]
                for region in regions:
                    region_counts[region] = region_counts.get(region, 0) + 1
        
        if region_counts:
            st.markdown("**Projects by Region:**")
            sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
            for region, count in sorted_regions:
                st.markdown(f"• {region}: {count}")
        
        # Count by sectors
        sector_counts = {}
        for project in projects:
            if project.get('sectors'):
                sectors = project['sectors'] if isinstance(project['sectors'], list) else [project['sectors']]
                for sector in sectors:
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        if sector_counts:
            st.markdown("**Top Sectors:**")
            sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for sector, count in sorted_sectors:
                st.markdown(f"• {sector}: {count}")
        
        # Count by countries (from the complete list)
        country_counts = {}
        for project in projects:
            country = project.get('country', '')
            if country:
                # Map to known countries
                for known_country in KNOWN_COUNTRIES.keys():
                    if advanced_fuzzy_match(country, known_country):
                        country_counts[known_country.title()] = country_counts.get(known_country.title(), 0) + 1
                        break
        
        if country_counts:
            st.markdown("**Top Countries:**")
            sorted_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            for country, count in sorted_countries:
                st.markdown(f"• {country}: {count}")

# Add input box at the top for General and Search modes
if st.session_state['mode'] in ['general', 'search']:
    st.markdown("---")
    st.subheader("💬 Ask a Question")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        # Dynamic placeholder based on mode
        if st.session_state['mode'] == 'search':
            placeholder_text = "Search K&M projects... (e.g., 'Give me LNG projects in United States')"
        else:  # general mode
            placeholder_text = "Ask about K&M's services... (e.g., 'What services does K&M provide?')"
        
        user_input = st.text_input(
            "", 
            value=st.session_state['current_question'],
            key="kb_input", 
            placeholder=placeholder_text,
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", key="kb_send", use_container_width=True)
    
    st.markdown("---")

# Display chat history - show only current mode's history
current_chat_history = get_current_chat_history()
for i, (user, bot) in enumerate(current_chat_history):
    with st.container():
        st.markdown(f"**👤 User:** {user}")
        st.markdown(f"**🤖 Assistant:** {bot}")
        st.divider()

# Add fixed input at bottom only for CV mode
if st.session_state['mode'] == 'cv':
    # FIXED: ChatGPT-style sticky input at bottom with proper overlay for CV mode
    st.markdown("""
    <style>
    /* Add bottom padding to main content to avoid overlap with fixed input */
    .main .block-container {
        padding-bottom: 120px !important;
    }

    /* Fixed floating input container */
    .chat-input-container {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: white !important;
        border-top: 1px solid #e0e0e0 !important;
        padding: 15px 20px !important;
        z-index: 999 !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1) !important;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .chat-input-container {
            background: #0e1117 !important;
            border-top: 1px solid #262730 !important;
        }
    }

    /* Hide default streamlit input styling */
    .stTextInput > label {
        display: none !important;
    }

    /* Ensure input takes full width in container */
    .chat-input-container .stTextInput {
        margin-bottom: 0 !important;
    }

    /* Style the input field */
    .chat-input-container input {
        border-radius: 25px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
    }

    .chat-input-container input:focus {
        border-color: #ff4b4b !important;
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.2) !important;
    }

    /* Style the send button */
    .chat-input-container .stButton button {
        border-radius: 25px !important;
        height: 48px !important;
        background: #ff4b4b !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }

    .chat-input-container .stButton button:hover {
        background: #ff3333 !important;
        transform: translateY(-1px) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create the fixed floating input container for CV mode
    # st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

    # col1, col2 = st.columns([5, 1])

    # with col1:
    #     placeholder_text = "CV analysis ready - upload file above and describe work requirements"
        
    #     cv_user_input = st.text_input(
    #         "", 
    #         value=st.session_state['current_question'],
    #         key="cv_kb_input", 
    #         placeholder=placeholder_text,
    #         label_visibility="collapsed"
    #     )

    # with col2:
    #     cv_send_button = st.button("Send", key="cv_kb_send", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Process input based on mode
# Handle top input for general and search modes
if 'send_button' in locals() and send_button and user_input.strip():
    if st.session_state['project_data']:
        projects = st.session_state['project_data'].get('projects', [])
        
        if st.session_state['mode'] == 'search':
            # Search mode - use project search only
            with st.spinner("🔍 Searching project database..."):
                formatted_response, criteria, filtered_projects = process_project_query(
                    user_input, projects, st.session_state['chat_history']
                )
                
                if formatted_response is not None:
                    response = formatted_response
                    st.session_state['last_search_criteria'] = criteria
                else:
                    response = "I can help you search for specific projects. Try asking something like 'Show me projects in [country]' or 'Find LNG projects for [client]'."
        
        elif st.session_state['mode'] == 'general':
            # General mode - try project search first, then use website content
            with st.spinner("🤔 Analyzing your question..."):
                response = process_general_query(
                    user_input, projects, st.session_state['chat_history']
                )
        
        add_to_current_chat_history(user_input, response)
        st.session_state['current_question'] = ""
        st.rerun()

# Handle bottom input for CV mode
if 'cv_send_button' in locals() and cv_send_button and cv_user_input.strip():
    if st.session_state['mode'] == 'cv':
        # CV mode - handled above in the CV section
        st.info("Please use the CV analysis section above to upload a file and describe work requirements.")


# Enhanced footer with mode indication
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    K&M Advisors LLC - Global Project Knowledge Assistant<br>
    Current Mode: <strong>{st.session_state['mode'].title()}</strong> | 
    Enhanced with Specialized Search Methods: Location-First | Technical-First | Client-First | Combined | CV Analysis<br>
    CV Mode: Advanced semantic matching with DOCX/PDF support for targeted proposal generation
</div>
""", unsafe_allow_html=True)
