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

VECTORSTORE_DIR = "./chroma_store"
PROJECT_JSON = "./Shared/Marketing/Project Descriptions/combined_project_data.json"
WEBSITE_MD = "./Shared/website.md"
USE_OPENAI = False  # Set to True to use OpenAI, False for local Ollama

st.title("K&M Project Knowledge Assistant")
st.caption("Powered by comprehensive project database and website content")

# ---- Enhanced Project Lookup Functions with Smart Recognition ----

# Define all known categories and their variations/abbreviations
KNOWN_SERVICES = {
    'due diligence': ['due diligence', 'dd', 'due-diligence'],
    'feasibility study': ['feasibility study', 'feasibility', 'fs', 'feas study'],
    "lender's engineer": ["lender's engineer", "lenders engineer", 'le', 'lender engineer'],
    "owner's engineer": ["owner's engineer", "owners engineer", 'oe', 'owner engineer'],
    'policy & regulation': ['policy & regulation', 'policy and regulation', 'policy', 'regulation', 'regulatory'],
    'project development': ['project development', 'development', 'dev', 'project dev'],
    'transaction advisory': ['transaction advisory', 'transaction', 'advisory', 'ta']
}

KNOWN_REGIONS = {
    'east asia and pacific': ['east asia and pacific', 'east asia', 'pacific', 'eap', 'asia pacific'],
    'europe and central asia': ['europe and central asia', 'europe', 'central asia', 'eca', 'european'],
    'latin america and the caribbean': ['latin america and the caribbean', 'latin america', 'caribbean', 'lac', 'latam'],
    'middle east and north africa': ['middle east and north africa', 'middle east', 'north africa', 'mena', 'mea'],
    'north america': ['north america', 'na', 'north american'],
    'south asia': ['south asia', 'sa', 'south asian'],
    'sub saharan africa': ['sub saharan africa', 'sub-saharan africa', 'africa', 'ssa', 'sub saharan'],
    'global': ['global', 'worldwide', 'international']
}

KNOWN_SECTORS = {
    'renewable energy': ['renewable energy', 'renewable', 'renewables', 're', 'clean energy'],
    'conventional energy': ['conventional energy', 'conventional', 'traditional energy', 'ce', 'fossil'],
    'energy storage': ['energy storage', 'storage', 'battery storage', 'es'],
    'hydrogen': ['hydrogen', 'h2', 'green hydrogen', 'blue hydrogen'],
    'lng to power': ['lng to power', 'lng power', 'lng-to-power', 'lng2power', 'liquefied natural gas'],
    'other energy': ['other energy', 'misc energy', 'miscellaneous energy'],
    'water & wastewater': ['water & wastewater', 'water and wastewater', 'water', 'wastewater', 'ww'],
    'other infrastructure': ['other infrastructure', 'infrastructure', 'other infra', 'infra']
}

KNOWN_TECHNOLOGIES = {
    'wind': ['wind', 'wind power', 'wind energy', 'onshore wind', 'offshore wind'],
    'solar': ['solar', 'solar power', 'solar energy', 'pv', 'photovoltaic', 'solar pv'],
    'hydro': ['hydro', 'hydropower', 'hydroelectric', 'hydro power'],
    'ulsd/diesel': ['ulsd/diesel', 'ulsd', 'diesel', 'ultra low sulfur diesel'],
    'hfo': ['hfo', 'heavy fuel oil', 'fuel oil'],
    'others': ['others', 'other', 'misc', 'miscellaneous'],
    'nuclear': ['nuclear', 'nuclear power', 'nuclear energy'],
    'natural gas': ['natural gas', 'gas', 'ng', 'nat gas'],
    'coal': ['coal', 'coal power', 'coal energy'],
    'green hydrogen': ['green hydrogen', 'green h2', 'renewable hydrogen'],
    'geothermal': ['geothermal', 'geothermal energy', 'geothermal power'],
    'bess': ['bess', 'battery energy storage system', 'battery storage', 'battery'],
    'biomass': ['biomass', 'biomass energy', 'biomass power', 'bio energy']
}

def normalize_and_match_categories(query_term: str, category_dict: Dict[str, List[str]]) -> List[str]:
    """
    Match query terms to known categories using fuzzy matching and abbreviations
    """
    query_lower = query_term.lower().strip()
    matches = []
    
    for canonical_name, variations in category_dict.items():
        for variation in variations:
            if query_lower == variation.lower() or query_lower in variation.lower() or variation.lower() in query_lower:
                matches.append(canonical_name)
                break
    
    return list(set(matches))  # Remove duplicates

def smart_extract_keywords_from_query(query: str) -> Dict[str, List[str]]:
    """
    Enhanced keyword extraction that recognizes known categories and abbreviations
    """
    query_lower = query.lower()
    
    extracted = {
        'countries': [],
        'regions': [],
        'services': [],
        'sectors': [],
        'technologies': [],
        'clients': [],
        'other_terms': []
    }
    
    # Split query into words and phrases
    words = query_lower.split()
    
    # Check for multi-word phrases first
    for i in range(len(words)):
        for j in range(i+1, min(i+6, len(words)+1)):  # Check up to 5-word phrases
            phrase = ' '.join(words[i:j])
            
            # Check against known categories
            services = normalize_and_match_categories(phrase, KNOWN_SERVICES)
            if services:
                extracted['services'].extend(services)
                
            regions = normalize_and_match_categories(phrase, KNOWN_REGIONS)
            if regions:
                extracted['regions'].extend(regions)
                
            sectors = normalize_and_match_categories(phrase, KNOWN_SECTORS)
            if sectors:
                extracted['sectors'].extend(sectors)
                
            technologies = normalize_and_match_categories(phrase, KNOWN_TECHNOLOGIES)
            if technologies:
                extracted['technologies'].extend(technologies)
    
    # Check individual words
    for word in words:
        # Services
        services = normalize_and_match_categories(word, KNOWN_SERVICES)
        extracted['services'].extend(services)
        
        # Regions
        regions = normalize_and_match_categories(word, KNOWN_REGIONS)
        extracted['regions'].extend(regions)
        
        # Sectors
        sectors = normalize_and_match_categories(word, KNOWN_SECTORS)
        extracted['sectors'].extend(sectors)
        
        # Technologies
        technologies = normalize_and_match_categories(word, KNOWN_TECHNOLOGIES)
        extracted['technologies'].extend(technologies)
        
        # Countries (keep existing logic but enhance)
        if len(word) > 3 and word.isalpha():
            # Common country name mappings
            country_mappings = {
                'usa': 'United States',
                'us': 'United States',
                'uk': 'United Kingdom',
                'uae': 'United Arab Emirates'
            }
            
            if word in country_mappings:
                extracted['countries'].append(country_mappings[word])
            else:
                extracted['countries'].append(word.capitalize())
        
        # Clients (look for uppercase words or known client patterns)
        if len(word) > 2 and (word.isupper() or word.istitle()):
            extracted['clients'].append(word.upper())
        
        # Other terms for fallback
        if len(word) > 3:
            extracted['other_terms'].append(word)
    
    # Remove duplicates
    for key in extracted:
        extracted[key] = list(set(extracted[key]))
    
    return extracted

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
    Enhanced project lookup with precise filtering based on known categories
    """
    
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        return text.lower().strip()
    
    def fuzzy_match(search_term: str, target_text: str) -> bool:
        """Check if search term matches target text with fuzzy logic"""
        search_norm = normalize_text(search_term)
        target_norm = normalize_text(target_text)
        
        # Exact match
        if search_norm == target_norm:
            return True
        
        # Partial match
        if search_norm in target_norm or target_norm in search_norm:
            return True
        
        # Word-level match
        search_words = search_norm.split()
        target_words = target_norm.split()
        
        for search_word in search_words:
            for target_word in target_words:
                if search_word == target_word or search_word in target_word or target_word in search_word:
                    return True
        
        return False
    
    def check_country_region_match(project: Dict, country_filters: List[str], region_filters: List[str]) -> bool:
        """Check if project matches country/region criteria"""
        if not country_filters and not region_filters:
            return True
            
        search_fields = []
        
        # Add country field
        if project.get('country'):
            search_fields.append(project['country'])
            
        # Add regions
        if project.get('regions'):
            if isinstance(project['regions'], list):
                search_fields.extend(project['regions'])
            else:
                search_fields.append(project['regions'])
            
        # Add docx_client and docx_country
        if project.get('docx_data'):
            docx_data = project['docx_data']
            if docx_data.get('docx_client'):
                search_fields.append(docx_data['docx_client'])
            if docx_data.get('docx_country'):
                search_fields.append(docx_data['docx_country'])
        
        # Check country matches
        if country_filters:
            for country_filter in country_filters:
                for field in search_fields:
                    if fuzzy_match(country_filter, field):
                        return True
        
        # Check region matches
        if region_filters:
            for region_filter in region_filters:
                for field in search_fields:
                    if fuzzy_match(region_filter, field):
                        return True
        
        return False
    
    def check_sector_tech_service_match(project: Dict, sector_filters: List[str], tech_filters: List[str], service_filters: List[str]) -> bool:
        """Check if project matches sector/technology/service criteria"""
        if not sector_filters and not tech_filters and not service_filters:
            return True
            
        search_fields = []
        
        # Add project_name
        if project.get('project_name'):
            search_fields.append(project['project_name'])
            
        # Add sectors
        if project.get('sectors'):
            if isinstance(project['sectors'], list):
                search_fields.extend(project['sectors'])
            else:
                search_fields.append(project['sectors'])
            
        # Add technologies-type
        if project.get('technologies'):
            for tech in project['technologies']:
                if isinstance(tech, dict) and tech.get('type'):
                    search_fields.append(tech['type'])
                elif isinstance(tech, str):
                    search_fields.append(tech)
        
        # Add services
        if project.get('services'):
            if isinstance(project['services'], list):
                search_fields.extend(project['services'])
            else:
                search_fields.append(project['services'])
        
        # Check sector matches
        if sector_filters:
            for sector_filter in sector_filters:
                for field in search_fields:
                    if fuzzy_match(sector_filter, field):
                        return True
        
        # Check technology matches
        if tech_filters:
            for tech_filter in tech_filters:
                for field in search_fields:
                    if fuzzy_match(tech_filter, field):
                        return True
        
        # Check service matches
        if service_filters:
            for service_filter in service_filters:
                for field in search_fields:
                    if fuzzy_match(service_filter, field):
                        return True
        
        return False
    
    def check_client_match(project: Dict, client_filters: List[str]) -> bool:
        """Check if project matches client criteria"""
        if not client_filters:
            return True
            
        search_fields = []
        
        # Add docx_client
        if project.get('docx_data', {}).get('docx_client'):
            search_fields.append(project['docx_data']['docx_client'])
            
        # Add docx_country  
        if project.get('docx_data', {}).get('docx_country'):
            search_fields.append(project['docx_data']['docx_country'])
            
        # Add assignment name and description
        if project.get('docx_data', {}).get('assignment'):
            assignment = project['docx_data']['assignment']
            if assignment.get('name'):
                search_fields.append(assignment['name'])
            if assignment.get('description'):
                search_fields.append(assignment['description'])
        
        # Add main client field
        if project.get('client'):
            search_fields.append(project['client'])
        
        # Check for matches
        for client_filter in client_filters:
            for field in search_fields:
                if fuzzy_match(client_filter, field):
                    return True
        
        return False
    
    def check_other_terms_match(project: Dict, other_filters: List[str]) -> bool:
        """Check if project matches other search terms"""
        if not other_filters:
            return True
            
        # Combine all searchable text from project
        all_text = []
        
        # Basic fields
        for field in ['project_name', 'country', 'client']:
            if project.get(field):
                all_text.append(project[field])
        
        # Lists
        for field in ['sectors', 'services', 'regions']:
            if project.get(field):
                if isinstance(project[field], list):
                    all_text.extend(project[field])
                else:
                    all_text.append(project[field])
        
        # Technologies
        if project.get('technologies'):
            for tech in project['technologies']:
                if isinstance(tech, dict) and tech.get('type'):
                    all_text.append(tech['type'])
                elif isinstance(tech, str):
                    all_text.append(tech)
        
        # DOCX data
        if project.get('docx_data'):
            docx_data = project['docx_data']
            for field in ['docx_client', 'docx_country']:
                if docx_data.get(field):
                    all_text.append(docx_data[field])
            
            if docx_data.get('assignment'):
                assignment = docx_data['assignment']
                for field in ['name', 'description']:
                    if assignment.get(field):
                        all_text.append(assignment[field])
        
        # Check for matches
        combined_text = ' '.join(all_text).lower()
        
        for term in other_filters:
            if normalize_text(term) in combined_text:
                return True
        
        return False
    
    # Apply filters
    filtered_projects = []
    
    for project in projects:
        matches_location = check_country_region_match(project, country_filter, region_filter)
        matches_technical = check_sector_tech_service_match(project, sector_filter, technology_filter, service_filter)
        matches_client = check_client_match(project, client_filter)
        matches_other = check_other_terms_match(project, other_terms_filter)
        
        # Project must match ALL specified criteria
        if matches_location and matches_technical and matches_client and matches_other:
            filtered_projects.append(project)
    
    return filtered_projects

def intelligent_project_search(query: str, projects: List[Dict]) -> tuple:
    """
    Enhanced intelligent project search with smart category recognition
    """
    # Extract keywords using enhanced function
    extracted = smart_extract_keywords_from_query(query)
    
    # Perform lookup with all extracted criteria
    results = enhanced_project_lookup_function(
        projects=projects,
        country_filter=extracted['countries'] if extracted['countries'] else None,
        region_filter=extracted['regions'] if extracted['regions'] else None,
        service_filter=extracted['services'] if extracted['services'] else None,
        sector_filter=extracted['sectors'] if extracted['sectors'] else None,
        technology_filter=extracted['technologies'] if extracted['technologies'] else None,
        client_filter=extracted['clients'] if extracted['clients'] else None,
        other_terms_filter=extracted['other_terms'] if extracted['other_terms'] else None
    )
    
    # Combine all technical criteria for display
    all_tech_criteria = []
    all_tech_criteria.extend(extracted['services'])
    all_tech_criteria.extend(extracted['sectors'])
    all_tech_criteria.extend(extracted['technologies'])
    
    # Combine all location criteria for display
    all_location_criteria = []
    all_location_criteria.extend(extracted['countries'])
    all_location_criteria.extend(extracted['regions'])
    
    return results, {
        'locations': all_location_criteria,
        'technical': all_tech_criteria,
        'clients': extracted['clients'],
        'other_terms': extracted['other_terms']
    }

# ---- Enhanced Formatting Functions ----

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
            if len(description) > 400:
                description = description[:400] + "..."
            
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
    """Format search criteria summary"""
    summary = []
    summary.append("🔍 **SEARCH CRITERIA APPLIED:**")
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
    
    summary.append("")
    summary.append(f"📊 **Results:** {num_results} project(s) found")
    
    return "\n".join(summary)

# ---- Rest of your existing code with modifications ----

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

# Updated process_project_query function
def process_project_query(user_input: str, projects: List[Dict], chat_history: List):
    """Process user query with intelligent project lookup and enhanced formatting"""
    
    # Check if this is a project lookup query
    lookup_keywords = ['projects in', 'projects for', 'show me', 'give me', 'list', 'find projects', 'tell me about projects']
    is_lookup_query = any(keyword in user_input.lower() for keyword in lookup_keywords)
    
    if is_lookup_query:
        # Use intelligent project search
        filtered_projects, search_criteria = intelligent_project_search(user_input, projects)
        
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
            
            formatted_response = "\n".join(response_parts)
        
        return formatted_response, search_criteria, filtered_projects
    
    else:
        # Use existing vectorstore approach for general queries
        return None, None, None

# ---- Session State and UI ----
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None

if 'project_data' not in st.session_state:
    st.session_state['project_data'] = load_project_data()

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'current_question' not in st.session_state:
    st.session_state['current_question'] = ""

# Enhanced chat input processing
user_input = st.text_input("Ask about K&M's projects:", 
                          value=st.session_state['current_question'],
                          key="kb_input", 
                          placeholder="e.g., 'Show me renewable energy projects in Sub Saharan Africa' or 'Find DD projects for solar'")

if st.button("Send", key="kb_send") and user_input.strip():
    if st.session_state['project_data']:
        projects = st.session_state['project_data'].get('projects', [])
        
        with st.spinner("🔍 Searching project database..."):
            # Try project lookup first
            formatted_response, criteria, filtered_projects = process_project_query(
                user_input, projects, st.session_state['chat_history']
            )
            
            if formatted_response is not None:
                # This is a project lookup query - use the formatted response
                response = formatted_response
            else:
                # Fall back to existing vectorstore approach
                response = "I can help you search for specific projects. Try asking something like 'Show me projects in [country]' or 'Find LNG projects for [client]'."
        
        st.session_state['chat_history'].append((user_input, response))
        st.session_state['current_question'] = ""
        st.rerun()

# Display chat history
for i, (user, bot) in enumerate(st.session_state['chat_history']):
    with st.container():
        st.markdown(f"**👤 User:** {user}")
        st.markdown(f"**🤖 Assistant:** {bot}")
        st.divider()

# Enhanced sidebar with smart lookup examples
with st.sidebar:
    st.subheader("💡 Smart Project Lookup Examples")
    
    lookup_examples = [
        "Show me renewable energy projects in Sub Saharan Africa",
        "Find DD projects for solar technology", 
        "Give me LNG projects in North America",
        "List all wind projects we've done",
        "Show me feasibility studies in MENA",
        "Find owner's engineer projects for BESS",
        "Give me hydro projects in Latin America",
        "Show me all projects in EAP region"
    
    ]
    
    for example in lookup_examples:
        if st.button(example, key=f"lookup_{hash(example)}", use_container_width=True):
            st.session_state['current_question'] = example
            st.rerun()
    
    st.divider()
    
    # Add category reference guide
    st.subheader("📚 Search Categories & Abbreviations")
    
    with st.expander("🌍 Regions"):
        st.markdown("""
        - **EAP**: East Asia and Pacific
        - **ECA**: Europe and Central Asia  
        - **LAC**: Latin America and the Caribbean
        - **MENA**: Middle East and North Africa
        - **NA**: North America
        - **SA**: South Asia
        - **SSA**: Sub Saharan Africa
        """)
    
    with st.expander("⚡ Sectors"):
        st.markdown("""
        - **RE**: Renewable Energy
        - **CE**: Conventional Energy
        - **ES**: Energy Storage
        - **H2**: Hydrogen
        - **LNG**: LNG to Power
        - **WW**: Water & Wastewater
        """)
    
    with st.expander("🔧 Technologies"):
        st.markdown("""
        - **PV**: Solar/Photovoltaic
        - **NG**: Natural Gas
        - **BESS**: Battery Energy Storage
        - **ULSD**: Ultra Low Sulfur Diesel
        - **HFO**: Heavy Fuel Oil
        - **H2**: Green Hydrogen
        """)
    
    with st.expander("📋 Services"):
        st.markdown("""
        - **DD**: Due Diligence
        - **FS**: Feasibility Study
        - **LE**: Lender's Engineer
        - **OE**: Owner's Engineer
        - **TA**: Transaction Advisory
        - **Dev**: Project Development
        """)
    
    st.divider()
    
    # Add project statistics if available
    if st.session_state['project_data']:
        projects = st.session_state['project_data'].get('projects', [])
        
        st.subheader("📊 Database Statistics")
        st.metric("Total Projects", len(projects))
        
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
        
        # Count by regions
        region_counts = {}
        for project in projects:
            if project.get('regions'):
                regions = project['regions'] if isinstance(project['regions'], list) else [project['regions']]
                for region in regions:
                    region_counts[region] = region_counts.get(region, 0) + 1
        
        if region_counts:
            st.markdown("**Top Regions:**")
            sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for region, count in sorted_regions:
                st.markdown(f"• {region}: {count}")

# Add some helpful information at the bottom
st.markdown("---")
st.markdown("""
### 🎯 **How to Search Effectively:**

**Use Natural Language:**
- "Show me renewable energy projects in Africa"
- "Find all solar projects we've done for USAID"
- "Give me LNG projects in the Middle East"

**Use Abbreviations:**
- "Find DD projects for BESS technology"
- "Show me FS projects in MENA region"
- "List OE projects for wind power"

**Combine Multiple Criteria:**
- "Find renewable energy feasibility studies in Sub Saharan Africa"
- "Show me owner's engineer projects for solar in North America"
- "Give me all LNG due diligence projects"

**Search Tips:**
- Use region abbreviations (EAP, MENA, SSA, etc.)
- Use service abbreviations (DD, FS, OE, LE, etc.)
- Use technology abbreviations (BESS, PV, NG, etc.)
- Combine location + technology + service for precise results
""")

# Debug information (optional - can be hidden in production)
if st.checkbox("Show Debug Info", value=False):
    if st.session_state['project_data']:
        projects = st.session_state['project_data'].get('projects', [])
        st.subheader("🔍 Debug Information")
        
        # Show sample project structure
        if projects:
            st.markdown("**Sample Project Structure:**")
            sample_project = projects[0]
            st.json({
                "project_id": sample_project.get('project_id'),
                "project_name": sample_project.get('project_name'),
                "country": sample_project.get('country'),
                "sectors": sample_project.get('sectors'),
                "technologies": sample_project.get('technologies'),
                "services": sample_project.get('services'),
                "regions": sample_project.get('regions'),
                "has_docx_data": bool(sample_project.get('docx_data'))
            })
        
        # Show last search criteria if available
        if hasattr(st.session_state, 'last_search_criteria'):
            st.markdown("**Last Search Criteria:**")
            st.json(st.session_state.last_search_criteria)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    K&M Engineering & Consulting Corporation - Project Knowledge Assistant<br>
    Powered by Advanced Project Database Search & AI
</div>
""", unsafe_allow_html=True)
