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
    'lng to power': ['lng to power', 'lng power', 'lng-to-power', 'lng2power', 'lng'],  # RESTRICTED: only LNG sector terms
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
    'nuclear': ['nuclear', 'nuclear power', 'nuclear energy', 'atomic power', 'nuclear plant'],  # RESTRICTED: only nuclear technology terms
    'natural gas': ['natural gas', 'gas', 'ng', 'nat gas', 'gas turbine', 'gas engine', 'ccgt',
                   'combined cycle', 'gas power', 'liquefied natural gas'],  # Added LNG as gas technology
    'coal': ['coal', 'coal power', 'coal energy', 'coal plant', 'coal fired', 'thermal coal'],
    'green hydrogen': ['green hydrogen', 'green h2', 'renewable hydrogen', 'electrolytic hydrogen'],
    'geothermal': ['geothermal', 'geothermal energy', 'geothermal power', 'geothermal plant'],
    'bess': ['bess', 'battery energy storage system', 'battery storage', 'battery', 'lithium battery',
            'grid battery', 'utility battery'],
    'biomass': ['biomass', 'biomass energy', 'biomass power', 'bio energy', 'biofuel', 'biogas',
               'waste to energy', 'bagasse']
}

# ---- GLOBAL HELPER FUNCTIONS ----

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

def normalize_and_match_categories_strict(query_term: str, category_dict: Dict[str, List[str]]) -> List[str]:
    """
    STRICTER matching - only exact matches or very close variations
    """
    query_lower = query_term.lower().strip()
    matches = []
    
    for canonical_name, variations in category_dict.items():
        for variation in variations:
            variation_lower = variation.lower()
            # Only exact match or very specific partial matches
            if query_lower == variation_lower:
                matches.append(canonical_name)
                break
            # Very restrictive partial matching - only for longer terms
            elif (len(query_lower) >= 4 and len(variation_lower) >= 4 and 
                  (query_lower in variation_lower or variation_lower in query_lower)):
                # Additional check: ensure it's a meaningful match
                if abs(len(query_lower) - len(variation_lower)) <= 3:
                    matches.append(canonical_name)
                    break
    
    return list(set(matches))  # Remove duplicates

def normalize_and_match_categories(query_term: str, category_dict: Dict[str, List[str]]) -> List[str]:
    """
    Enhanced matching with better fuzzy logic and partial matching
    """
    query_lower = query_term.lower().strip()
    matches = []
    
    for canonical_name, variations in category_dict.items():
        for variation in variations:
            variation_lower = variation.lower()
            # Exact match
            if query_lower == variation_lower:
                matches.append(canonical_name)
                break
            # Partial match (query contains variation or vice versa)
            elif query_lower in variation_lower or variation_lower in query_lower:
                # Avoid very short partial matches unless they're exact
                if len(query_lower) >= 3 and len(variation_lower) >= 3:
                    matches.append(canonical_name)
                    break
    
    return list(set(matches))  # Remove duplicates

def map_location_to_countries_and_regions(location_term: str) -> Dict[str, List[str]]:
    """
    Map a location term to both specific countries and regions with stricter matching
    """
    location_lower = location_term.lower().strip()
    result = {'countries': [], 'regions': []}
    
    # First check if it's a specific country with EXACT matching
    for country_name, variations in KNOWN_COUNTRIES.items():
        for variation in variations:
            variation_lower = variation.lower()
            # Use exact match or very close match only
            if (location_lower == variation_lower or 
                (len(location_lower) > 3 and len(variation_lower) > 3 and 
                 location_lower == variation_lower)):
                result['countries'].append(country_name)
                break
    
    # Only check regions if no specific country was found
    if not result['countries']:
        for region_name, region_data in KNOWN_REGIONS.items():
            region_variations = region_data['variations']
            for variation in region_variations:
                variation_lower = variation.lower()
                # Exact match for regions
                if location_lower == variation_lower:
                    result['regions'].append(region_name)
                    # Add all countries in this region
                    result['countries'].extend(region_data['countries'])
                    break
    
    # Remove duplicates
    result['countries'] = list(set(result['countries']))
    result['regions'] = list(set(result['regions']))
    
    return result

def smart_extract_keywords_from_query(query: str) -> Dict[str, List[str]]:
    """
    Enhanced keyword extraction with more precise location and technical matching
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
    
    # Check for multi-word phrases first (up to 4 words for locations)
    for i in range(len(words)):
        for j in range(i+1, min(i+5, len(words)+1)):
            phrase = ' '.join(words[i:j])
            
            # Location mapping with stricter matching
            location_mapping = map_location_to_countries_and_regions(phrase)
            if location_mapping['countries'] or location_mapping['regions']:
                extracted['countries'].extend(location_mapping['countries'])
                extracted['regions'].extend(location_mapping['regions'])
            
            # Check against other categories with STRICT matching for sectors/technologies
            services = normalize_and_match_categories(phrase, KNOWN_SERVICES)
            extracted['services'].extend(services)
            
            sectors = normalize_and_match_categories_strict(phrase, KNOWN_SECTORS)  # STRICTER
            extracted['sectors'].extend(sectors)
            
            technologies = normalize_and_match_categories_strict(phrase, KNOWN_TECHNOLOGIES)  # STRICTER
            extracted['technologies'].extend(technologies)
    
    # Check individual words only if no multi-word location matches found
    if not extracted['countries'] and not extracted['regions']:
        for word in words:
            if len(word) <= 2:  # Skip very short words
                continue
                
            # Location mapping for individual words
            location_mapping = map_location_to_countries_and_regions(word)
            extracted['countries'].extend(location_mapping['countries'])
            extracted['regions'].extend(location_mapping['regions'])
    
    # Always check individual words for non-location categories
    for word in words:
        if len(word) <= 2:
            continue
            
        # Other categories with STRICT matching for technical terms
        services = normalize_and_match_categories(word, KNOWN_SERVICES)
        extracted['services'].extend(services)
        
        sectors = normalize_and_match_categories_strict(word, KNOWN_SECTORS)  # STRICTER
        extracted['sectors'].extend(sectors)
        
        technologies = normalize_and_match_categories_strict(word, KNOWN_TECHNOLOGIES)  # STRICTER
        extracted['technologies'].extend(technologies)
        
        # Clients (look for uppercase words or known client patterns)
        if len(word) > 2 and (word.isupper() or word.istitle()):
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
                extracted['clients'].append(client_mappings[word.lower()])
            elif word not in ['give', 'show', 'find', 'list', 'projects', 'me', 'in', 'for', 'the', 'and', 'of', 'to']:
                extracted['clients'].append(word.upper())
        
        # Other terms for fallback (exclude common words and already matched terms)
        if (len(word) > 3 and 
            word not in ['give', 'show', 'find', 'list', 'projects', 'project', 'tell', 'about', 'with', 'from', 'that', 'this', 'have', 'been', 'were', 'will', 'would', 'could', 'should'] and
            word not in [country for countries in extracted['countries'] for country in countries.split()] and
            word not in [service for services in extracted['services'] for service in services.split()] and
            word not in [sector for sectors in extracted['sectors'] for sector in sectors.split()] and
            word not in [tech for techs in extracted['technologies'] for tech in techs.split()]):
            extracted['other_terms'].append(word)
    
    # Remove duplicates and empty values
    for key in extracted:
        extracted[key] = list(set([item for item in extracted[key] if item]))
    
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

def intelligent_project_search(query: str, projects: List[Dict]) -> tuple:
    """
    Enhanced intelligent project search with comprehensive mapping
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
    
    # Combine criteria for display
    all_tech_criteria = []
    all_tech_criteria.extend(extracted['services'])
    all_tech_criteria.extend(extracted['sectors'])
    all_tech_criteria.extend(extracted['technologies'])
    
    # Combine location criteria for display
    all_location_criteria = []
    all_location_criteria.extend(extracted['countries'])
    all_location_criteria.extend(extracted['regions'])
    
    return results, {
        'locations': all_location_criteria,
        'technical': all_tech_criteria,
        'clients': extracted['clients'],
        'other_terms': extracted['other_terms'],
        'extracted_details': extracted  # For debugging
    }

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
            response_parts.append("")
            response_parts.append("🌍 **Available Countries (Sample):**")
            response_parts.append("   • **Americas:** Argentina, Brazil, Chile, Colombia, Mexico, United States")
            response_parts.append("   • **Caribbean:** Jamaica, Dominican Republic, Trinidad & Tobago, Aruba")
            response_parts.append("   • **Africa:** Kenya, Nigeria, South Africa, Morocco, Ghana")
            response_parts.append("   • **Asia:** China, India, Indonesia, Vietnam, Philippines")
            response_parts.append("   • **Europe:** Germany, Netherlands, Poland, Turkey")
            
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

# Enhanced sidebar with smart lookup examples
with st.sidebar:
    st.subheader("💡 Smart Project Lookup Examples")
    
    # Updated examples with actual countries from the complete list
    lookup_examples = [
        "Show me renewable energy projects in Brazil",
        "Find DD projects for solar technology in India", 
        "Give me LNG projects in United States",
        "List all wind projects in China",
        "Show me feasibility studies in MENA region",
        "Find owner's engineer projects for BESS in Germany",
        "Give me hydro projects in Philippines",
        "Show me projects in Nigeria or Kenya",
        "Find solar projects in Morocco",
        "List water projects in Indonesia",
        "Show me projects in Caribbean region",
        "Find nuclear projects in Turkey",
        "Give me geothermal projects in Indonesia"
    ]
    
    for example in lookup_examples:
        if st.button(example, key=f"lookup_{hash(example)}", use_container_width=True):
            st.session_state['current_question'] = example
            st.rerun()
    
    st.divider()
    
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
        - **LNG to Power**: Gas-to-Power, LNG Terminals  ⚠️ *Specific sector*
        - **Infrastructure**: Water, Wastewater, Transport
        
        **Key Technologies:**
        - Solar: PV, CSP, Distributed Solar
        - Wind: Onshore, Offshore, Small Wind
        - Gas: CCGT, OCGT, Cogeneration, LNG
        - Nuclear: Nuclear Power, Atomic Energy  ⚠️ *Specific technology*
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
        
        **Common Clients:**
        - World Bank, IFC, ADB, AfDB, IADB
        - USAID, Development Finance Institutions
        - Private Equity, Infrastructure Funds
        - Utilities, IPPs, Government Agencies
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
for i, (user, bot) in enumerate(st.session_state['chat_history']):
    with st.container():
        st.markdown(f"**👤 User:** {user}")
        st.markdown(f"**🤖 Assistant:** {bot}")
        st.divider()

# MOVED TO BOTTOM: Enhanced chat input processing with sticky container
with st.container():
    st.markdown("---")
    st.markdown("### 💬 Ask about K&M's projects:")
    
    # Create columns for input and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input("", 
                              value=st.session_state['current_question'],
                              key="kb_input", 
                              placeholder="e.g., 'Show me renewable energy projects in Brazil' or 'Find DD projects for solar in Asia'",
                              label_visibility="collapsed")
    
    with col2:
        send_button = st.button("Send", key="kb_send", use_container_width=True)

    if send_button and user_input.strip():
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
                    # Store last search criteria for debugging
                    st.session_state['last_search_criteria'] = criteria
                else:
                    # Fall back to existing vectorstore approach
                    response = "I can help you search for specific projects. Try asking something like 'Show me projects in [country]' or 'Find LNG projects for [client]'."
            
            st.session_state['chat_history'].append((user_input, response))
            st.session_state['current_question'] = ""
            st.rerun()

# Add enhanced information section
st.markdown("---")
st.markdown("""
### 🎯 **Complete Search Guide:**

#### **🌍 Global Coverage - 99+ Countries:**
Our database now covers **99+ countries** across all 7 World Bank regions:

**🌎 Americas (35 countries):** From Canada to Argentina, including all Caribbean nations
**🌍 Africa (32 countries):** Comprehensive Sub-Saharan and North African coverage  
**🌏 Asia-Pacific (16 countries):** Major economies from China to Australia
**🌍 Europe (16 countries):** EU and Central Asian markets
**🌍 Middle East (8 countries):** Key MENA energy markets

#### **🔍 Smart Search Examples:**

**Regional Searches:**
- ✅ "renewable projects in Africa" → All SSA countries
- ✅ "solar projects in LAC" → Latin America & Caribbean
- ✅ "wind projects in Asia" → EAP + South Asia regions

**Multi-Country Searches:**
- ✅ "projects in Brazil or Mexico" → Both countries
- ✅ "show me projects in BRICS countries" → Brazil, Russia, India, China
- ✅ "find projects in island nations" → Caribbean, Pacific islands

**Technology + Location:**
- ✅ "offshore wind projects in Europe"
- ✅ "solar + storage projects in Africa"
- ✅ "LNG terminals in Asia Pacific"

**Service + Sector Combinations:**
- ✅ "feasibility studies for renewable energy"
- ✅ "due diligence for LNG projects"
- ✅ "owner's engineer for wind farms"

#### **💡 Pro Search Tips:**

1. **Use Natural Language**: "Show me all hydro projects in South America"
2. **Combine Multiple Filters**: "Find renewable energy DD projects in emerging markets"
3. **Try Regional Abbreviations**: "List FS projects in SSA" or "Show MENA projects"
4. **Use Technology Variations**: "Battery storage" = "BESS" = "Energy storage"
5. **Client-Specific Searches**: "World Bank projects in Africa" or "USAID renewable projects"
6. **⚠️ Precise Technical Terms**: "LNG" → LNG to Power sector only, "Nuclear" → Nuclear technology only

#### **📊 Database Scope:**
- **99+ Countries** with intelligent fuzzy matching
- **8 Major Sectors** from renewable to infrastructure
- **15+ Technologies** including emerging tech like hydrogen
- **7 Service Types** from feasibility to transaction advisory
- **Smart Region Mapping** with automatic country inclusion
- **Stricter Technical Matching** for more precise results
""")

# Enhanced debug information
if st.checkbox("Show Advanced Debug Info", value=False):
    if st.session_state['project_data']:
        projects = st.session_state['project_data'].get('projects', [])
        st.subheader("🔍 Advanced Debug Information")
        
        # Show complete country coverage
        st.markdown("**Complete Country Coverage:**")
        col1, col2, col3 = st.columns(3)
        
        country_list = list(KNOWN_COUNTRIES.keys())
        chunk_size = len(country_list) // 3
        
        with col1:
            st.markdown("**Americas & Europe:**")
            for country in sorted(country_list[:chunk_size]):
                st.markdown(f"• {country.title()}")
        
        with col2:
            st.markdown("**Africa & Middle East:**")
            for country in sorted(country_list[chunk_size:2*chunk_size]):
                st.markdown(f"• {country.title()}")
        
        with col3:
            st.markdown("**Asia & Pacific:**")
            for country in sorted(country_list[2*chunk_size:]):
                st.markdown(f"• {country.title()}")
        
        st.markdown(f"**Total Countries Supported:** {len(KNOWN_COUNTRIES)}")
        
        # Show technical matching examples
        st.markdown("**Technical Search Restrictions:**")
        st.markdown("- **'LNG'** → Only matches 'LNG to Power' sector")
        st.markdown("- **'Nuclear'** → Only matches 'Nuclear' technology")
        st.markdown("- **'Gas'** → Matches both 'Natural Gas' technology and general gas terms")
        st.markdown("- **'Storage'** → Matches 'Energy Storage' sector and 'BESS' technology")
        
        # Show region mapping examples
        st.markdown("**Region → Country Mapping Examples:**")
        test_locations = ["Africa", "LAC", "Asia", "MENA", "Caribbean", "Europe"]
        for location in test_locations:
            mapping = map_location_to_countries_and_regions(location)
            st.markdown(f"**{location}** → {len(mapping['countries'])} countries, {len(mapping['regions'])} regions")
        
        # Show last search criteria if available
        if hasattr(st.session_state, 'last_search_criteria') and st.session_state.last_search_criteria:
            st.markdown("**Last Search Breakdown:**")
            st.json(st.session_state.last_search_criteria)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    K&M Engineering & Consulting Corporation - Global Project Knowledge Assistant<br>
    Enhanced with Complete Country Coverage (99+ Nations) & Precise Technical Matching<br>
    <em>Comprehensive global project database with advanced search capabilities</em>
</div>
""", unsafe_allow_html=True)

