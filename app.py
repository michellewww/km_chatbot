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

VECTORSTORE_DIR = "./chroma_store"
PROJECT_JSON = "./Shared/Marketing/Project Descriptions/combined_project_data.json"
WEBSITE_MD = "./Shared/website.md"
USE_OPENAI = False  # Set to True to use OpenAI, False for local Ollama

st.title("K&M Project Knowledge Assistant")
st.caption("Powered by comprehensive project database and website content")

# ---- MOVE: Button Mode Selection to sidebar top ----
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'search'  # Default to search
    
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
    
    # Show current mode
    st.caption(f"**Current Mode:** {st.session_state['mode'].title()}")
    st.divider()
    


# ---- Enhanced Project Lookup Functions with Complete Country List ----

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
        
        # Check for matches
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

def setup_vectorstore():
    """Setup vectorstore for website content"""
    if st.session_state.get('vectorstore') is not None:
        return st.session_state['vectorstore']
    
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
        embeddings = OpenAIEmbeddings()
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
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
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
                formatted_history = [(msg[0], msg[1]) for msg in chat_history[-5:]]  # Last 5 exchanges
                
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
        st.subheader("💡 CV Query Examples")
        st.info("CV functionality will be implemented soon.")
    
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

# Display chat history FIRST
# Display chat history FIRST
for i, (user, bot) in enumerate(st.session_state['chat_history']):
    with st.container():
        st.markdown(f"**👤 User:** {user}")
        st.markdown(f"**🤖 Assistant:** {bot}")
        st.divider()

# FIXED: ChatGPT-style sticky input at bottom with proper overlay
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

# Create the fixed floating input container
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    # Dynamic placeholder based on mode
    if st.session_state['mode'] == 'search':
        placeholder_text = "Search K&M projects... (e.g., 'Give me LNG projects in United States')"
    elif st.session_state['mode'] == 'general':
        placeholder_text = "Ask about K&M's services... (e.g., 'What services does K&M provide?')"
    else:  # CV mode
        placeholder_text = "CV queries coming soon..."
    
    user_input = st.text_input(
        "", 
        value=st.session_state['current_question'],
        key="kb_input", 
        placeholder=placeholder_text,
        label_visibility="collapsed",
        disabled=(st.session_state['mode'] == 'cv')  # Disable for CV mode for now
    )

with col2:
    send_button = st.button("Send", key="kb_send", use_container_width=True, 
                           disabled=(st.session_state['mode'] == 'cv'))

st.markdown('</div>', unsafe_allow_html=True)


# Process input based on mode
if send_button and user_input.strip():
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
        
        else:  # CV mode
            response = "CV functionality is not yet implemented. Please switch to Search or General mode."
        
        st.session_state['chat_history'].append((user_input, response))
        st.session_state['current_question'] = ""
        st.rerun()

# Enhanced footer with mode indication
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    K&M Advisors LLC - Global Project Knowledge Assistant<br>
    Current Mode: <strong>{st.session_state['mode'].title()}</strong> | 
    Enhanced with Specialized Search Methods: Location-First | Technical-First | Client-First | Combined<br>
</div>
""", unsafe_allow_html=True)


        