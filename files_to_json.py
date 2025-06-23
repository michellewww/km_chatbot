import pandas as pd
import numpy as np
import json
import os
from docx import Document
import re

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def parse_client_country(client_country_text):
    """Parse client and country from combined text with empty line separation"""
    if not client_country_text:
        return "", ""
    
    # Split by multiple newlines or significant whitespace
    lines = [line.strip() for line in client_country_text.split('\n') if line.strip()]
    
    if not lines:
        return "", ""
    
    # If we have multiple lines, typically:
    # First line(s) = Country
    # Last line(s) = Client/Organization
    
    if len(lines) == 1:
        # Only one piece of info - determine if it's country or client
        text = lines[0]
        # Common client organization keywords
        client_keywords = [
            'corporation', 'company', 'limited', 'ltd', 'inc', 'agency', 
            'usaid', 'ustda', 'bank', 'development', 'energy', 'power',
            'industries', 'group', 'authority', 'ministry', 'department'
        ]
        
        if any(keyword in text.lower() for keyword in client_keywords):
            return text, ""  # It's a client
        else:
            return "", text  # It's a country
    
    elif len(lines) == 2:
        # Two lines - first is usually country, second is client
        return lines[1], lines[0]
    
    else:
        # Multiple lines - first line is country, rest are client
        country = lines[0]
        client = " ".join(lines[1:])
        return client, country

def extract_table_data(table):
    """Extract data from a single table"""
    projects = []
    
    # Skip header row (index 0)
    for row_idx, row in enumerate(table.rows):
        if row_idx == 0:  # Skip header row
            continue
            
        cells = [cell.text for cell in row.cells]
        
        # Make sure we have enough cells
        if len(cells) < 6:
            continue
            
        # Extract data from each column
        job_number_raw = clean_text(cells[0])
        # Convert job number format (e.g., 1065.15 -> 1065-15)
        job_number = format_job_id(job_number_raw)
        
        duration = clean_text(cells[1])
        assignment_description = clean_text(cells[2])
        client_country_raw = cells[3]  # Keep raw text with newlines
        contract_value = clean_text(cells[4])
        role = clean_text(cells[5])
        
        # Skip empty rows
        if not job_number and not assignment_description:
            continue
            
        # Parse assignment name and description
        assignment_parts = assignment_description.split(':', 1)
        if len(assignment_parts) >= 2:
            assignment_name = clean_text(assignment_parts[0])
            description = clean_text(assignment_parts[1])
        else:
            assignment_name = ""
            description = assignment_description
        
        # Parse client and country using the improved function
        client, country = parse_client_country(client_country_raw)
        
        # Parse duration
        start_date = ""
        end_date = ""
        if duration and duration.lower() != 'n/a':
            # Look for date patterns like "Sep. 1987 - Oct. 1987"
            date_match = re.search(r'(\w+\.?\s+\d{4})\s*-\s*(\w+\.?\s+\d{4})', duration)
            if date_match:
                start_date = date_match.group(1)
                end_date = date_match.group(2)
        
        project = {
            "job_number": job_number,
            "duration": {
                "original": duration,
                "start_date": start_date,
                "end_date": end_date
            },
            "assignment": {
                "name": assignment_name,
                "description": description
            },
            "client": client,
            "country": country,
            "contract_value": contract_value,
            "role": role,
            "source": "docx"
        }
        
        projects.append(project)
    
    return projects

def process_docx_file(docx_path):
    """Convert DOCX file with tables to JSON"""
    
    try:
        # Load the document
        doc = Document(docx_path)
        
        all_projects = []
        
        # Process each table in the document
        for table_idx, table in enumerate(doc.tables):
            print(f"Processing DOCX table {table_idx + 1}...")
            
            # Extract projects from this table
            projects = extract_table_data(table)
            all_projects.extend(projects)
        
        print(f"Successfully extracted {len(all_projects)} projects from DOCX")
        return all_projects
        
    except Exception as e:
        print(f"Error processing DOCX file: {str(e)}")
        return []

# Function to safely get column value
def safe_get_column(row, possible_keys, df):
    for key in possible_keys:
        if key in df.columns:
            value = row.get(key)
            if pd.notna(value) and str(value).strip() != '':
                return value
    return ''

# Updated function to format job ID with decimal to dash conversion
def format_job_id(value):
    """Format job ID, converting decimal format to dash format (e.g., 1065.15 -> 1065-15)"""
    if pd.isna(value) or str(value).strip() == '':
        return ''
    
    try:
        # Convert to string first
        value_str = str(value).strip()
        
        # Check if it contains a decimal point
        if '.' in value_str:
            # Split by decimal point
            parts = value_str.split('.')
            if len(parts) == 2:
                # Convert to integer parts and join with dash
                main_part = str(int(float(parts[0])))
                sub_part = parts[1].rstrip('0')  # Remove trailing zeros
                if sub_part:  # Only add dash if there's a meaningful sub-part
                    return f"{main_part}-{sub_part}"
                else:
                    return main_part
        
        # If no decimal, try to convert to integer to remove any .0
        try:
            return str(int(float(value_str)))
        except (ValueError, TypeError):
            return value_str
            
    except (ValueError, TypeError):
        return str(value).strip()

# Function to format project ID as integer (no decimals) - UPDATED
def format_project_id(value):
    """Format project ID, converting decimal format to dash format for Excel data"""
    return format_job_id(value)  # Use the same logic as job ID formatting

# Function to format year values
def format_year(value):
    if pd.isna(value) or str(value).strip() == '':
        return ''
    try:
        if isinstance(value, (int, float)):
            return str(int(value))
        return str(value).strip()
    except (ValueError, TypeError):
        return str(value).strip()

# Function to format numeric values
def format_numeric(value):
    if pd.isna(value) or str(value).strip() == '':
        return ''
    try:
        if isinstance(value, (int, float)):
            if value == int(value):
                return str(int(value))
            else:
                return str(value)
        return str(value).strip()
    except (ValueError, TypeError):
        return str(value).strip()

# Function to process each row into a structured JSON object
def process_excel_row(row, df):
    project = {}
    
    # Basic project information from first columns
    project_id_raw = safe_get_column(row, [
        ('Job # (Full)', ''),
        ('Unnamed: 0_level_0', 'Unnamed: 0_level_1'),
        df.columns[0] if len(df.columns) > 0 else None
    ], df)
    project['project_id'] = format_project_id(project_id_raw)
    
    project['job_status'] = str(safe_get_column(row, [
        ('Job Status', ''),
        ('Unnamed: 1_level_0', 'Unnamed: 1_level_1'),
        df.columns[1] if len(df.columns) > 1 else None
    ], df)).strip()
    
    year_start_raw = safe_get_column(row, [
        ('Start Year', ''),
        ('Unnamed: 2_level_0', 'Unnamed: 2_level_1'),
        df.columns[2] if len(df.columns) > 2 else None
    ], df)
    project['year_start'] = format_year(year_start_raw)
    
    year_end_raw = safe_get_column(row, [
        ('End Year', ''),
        ('Unnamed: 3_level_0', 'Unnamed: 3_level_1'),
        df.columns[3] if len(df.columns) > 3 else None
    ], df)
    project['year_end'] = format_year(year_end_raw)
    
    website_value = safe_get_column(row, [
        ('Website', ''),
        ('Unnamed: 4_level_0', 'Unnamed: 4_level_1'),
        df.columns[4] if len(df.columns) > 4 else None
    ], df)
    project['website'] = 'Yes' if str(website_value).lower() == 'yes' else 'No'
    
    confidential_value = safe_get_column(row, [
        ('Confidential', ''),
        ('Unnamed: 5_level_0', 'Unnamed: 5_level_1'),
        df.columns[5] if len(df.columns) > 5 else None
    ], df)
    project['confidential'] = 'Yes' if str(confidential_value).lower() == 'yes' else 'No'
    
    project_name_raw = safe_get_column(row, [
        ('Project Name', ''),
        ('Unnamed: 6_level_0', 'Unnamed: 6_level_1'),
        df.columns[6] if len(df.columns) > 6 else None
    ], df)
    project['project_name'] = str(project_name_raw).strip() if project_name_raw else ''
    
    # Find the rightmost columns for Country, Contract Value, Client, MW total
    # These should be the last 4 columns based on your image
    total_cols = len(df.columns)
    
    # Country (should be 4th from the end)
    if total_cols >= 4:
        country_col = df.columns[total_cols - 4]
        country_raw = row.get(country_col)
        project['country'] = str(country_raw).strip() if pd.notna(country_raw) else ''
    else:
        project['country'] = ''
    
    # Contract Value (should be 3rd from the end)
    if total_cols >= 3:
        contract_col = df.columns[total_cols - 3]
        contract_raw = row.get(contract_col)
        project['contract_value'] = format_numeric(contract_raw)
    else:
        project['contract_value'] = ''
    
    # Client (should be 2nd from the end)
    if total_cols >= 2:
        client_col = df.columns[total_cols - 2]
        client_raw = row.get(client_col)
        project['client'] = str(client_raw).strip() if pd.notna(client_raw) else ''
    else:
        project['client'] = ''
    
    # MW total (should be the last column)
    if total_cols >= 1:
        mw_col = df.columns[total_cols - 1]
        mw_raw = row.get(mw_col)
        project['mw_total'] = format_numeric(mw_raw)
    else:
        project['mw_total'] = ''
    
    # Extract services - look for columns with "Services" in the first level header
    services = []
    for col in df.columns:
        if 'Services' in str(col[0]) and col not in df.columns[total_cols-4:]:  # Exclude the last 4 columns
            value = row.get(col)
            if pd.notna(value) and (value == 1 or str(value).lower() == 'yes'):
                service_name = col[1] if col[1] and col[1] != '' and 'Unnamed' not in str(col[1]) else str(col[0])
                if service_name not in ['', 'Unnamed', 'Services']:
                    services.append(service_name)
    project['services'] = services
    
    # Extract regions - look for columns with "Region" in the first level header
    regions = []
    for col in df.columns:
        if 'Region' in str(col[0]) and col not in df.columns[total_cols-4:]:  # Exclude the last 4 columns
            value = row.get(col)
            if pd.notna(value) and (value == 1 or str(value).lower() == 'yes'):
                region_name = col[1] if col[1] and col[1] != '' and 'Unnamed' not in str(col[1]) else str(col[0])
                if region_name not in ['', 'Unnamed', 'Region']:
                    regions.append(region_name)
    project['regions'] = regions
    
    # Extract sectors - look for columns with "Sector" in the first level header
    sectors = []
    for col in df.columns:
        if 'Sector' in str(col[0]) and col not in df.columns[total_cols-4:]:  # Exclude the last 4 columns
            value = row.get(col)
            if pd.notna(value) and (value == 1 or str(value).lower() == 'yes'):
                sector_name = col[1] if col[1] and col[1] != '' and 'Unnamed' not in str(col[1]) else str(col[0])
                if sector_name not in ['', 'Unnamed', 'Sector']:
                    sectors.append(sector_name)
    project['sectors'] = sectors
    
    # Extract technologies - look for columns with "Technology" or "Fuel" in the first level header
    # BUT exclude the last 4 columns which are Country, Contract Value, Client, MW total
    technologies = []
    for col in df.columns:
        if ('Technology' in str(col[0]) or 'Fuel' in str(col[0])) and col not in df.columns[total_cols-4:]:
            value = row.get(col)
            if pd.notna(value) and value != 0 and str(value).strip() != '':
                tech_name = col[1] if col[1] and col[1] != '' and 'Unnamed' not in str(col[1]) else str(col[0])
                if tech_name not in ['', 'Unnamed', 'Technology', 'Fuel']:
                    tech_info = {
                        "type": tech_name,
                        "capacity": str(int(value)) + " MW" if isinstance(value, (int, float)) and value != 1 else "Yes"
                    }
                    technologies.append(tech_info)
    project['technologies'] = technologies
    
    # Add source identifier
    project['source'] = 'excel'
    
    return project

def process_excel_file(excel_path, sheet_name):
    """Process Excel file and return project data"""
    try:
        # Read the Excel file with multi-level headers (2 rows)
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=[0, 1])
        
        print("Excel column names:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        
        # Process all rows, starting from row 2 to skip headers
        json_data = []
        for idx, row in df.iterrows():
            if idx < 2:  # Skip header rows
                continue
                
            if not row.isna().all():
                project_json = process_excel_row(row, df)
                if project_json['project_id'] or project_json['project_name']:
                    json_data.append(project_json)
        
        print(f"Successfully extracted {len(json_data)} projects from Excel")
        return json_data
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return []

def merge_project_data(excel_projects, docx_projects):
    """Merge project data from both sources"""
    # Create a dictionary to track projects by job number/ID
    merged_projects = {}
    
    # Add Excel projects first
    for project in excel_projects:
        key = project.get('project_id', '').strip()
        if key:
            merged_projects[key] = project
    
    # Add DOCX projects, checking for matches
    unmatched_docx = []
    matched_count = 0
    
    for project in docx_projects:
        job_number = project.get('job_number', '').strip()
        
        # Try to find matching Excel project
        if job_number and job_number in merged_projects:
            # Merge the data - add DOCX specific fields to Excel project
            excel_project = merged_projects[job_number]
            excel_project['docx_data'] = {
                'duration': project['duration'],
                'assignment': project['assignment'],
                'role': project['role'],
                'docx_client': project['client'],
                'docx_country': project['country'],
                'docx_contract_value': project['contract_value']
            }
            excel_project['source'] = 'both'
            matched_count += 1
            print(f"Matched project ID: {job_number}")
        else:
            # No match found, add as separate project
            unmatched_docx.append(project)
            print(f"Unmatched DOCX project ID: {job_number}")
    
    print(f"Successfully matched {matched_count} projects between Excel and DOCX")
    print(f"Unmatched DOCX projects: {len(unmatched_docx)}")
    
    # Combine all projects
    all_projects = list(merged_projects.values()) + unmatched_docx
    
    return all_projects

def main():
    # Set file paths
    excel_path = os.path.join('.', 'Shared', 'Marketing', 'Project Descriptions', 'Copy of K&M Project List v12 (Updated).xlsx')
    docx_path = os.path.join('.', 'Shared', 'Marketing', 'Project Descriptions', 'K&M TECH-2B Short Form Compendium.docx')
    output_path = os.path.join('.', 'Shared', 'Marketing', 'Project Descriptions', 'combined_project_data.json')
    
    sheet_name = 'Project List'
    
    # Process Excel file
    print("Processing Excel file...")
    excel_projects = []
    if os.path.exists(excel_path):
        excel_projects = process_excel_file(excel_path, sheet_name)
        # Print some sample Excel project IDs for verification
        print("\nSample Excel project IDs:")
        for i, project in enumerate(excel_projects[:5]):
            print(f"  {project.get('project_id', 'N/A')}")
    else:
        print(f"Excel file not found: {excel_path}")
    
    # Process DOCX file
    print("\nProcessing DOCX file...")
    docx_projects = []
    if os.path.exists(docx_path):
        docx_projects = process_docx_file(docx_path)
        # Print some sample DOCX job numbers for verification
        print("\nSample DOCX job numbers:")
        for i, project in enumerate(docx_projects[:5]):
            print(f"  {project.get('job_number', 'N/A')}")
    else:
        print(f"DOCX file not found: {docx_path}")
    
    # Merge the data
    print("\nMerging project data...")
    merged_projects = merge_project_data(excel_projects, docx_projects)
    
    # Create final JSON structure
    result = {
        "document_info": {
            "title": "K&M Combined Project Data",
            "excel_projects": len(excel_projects),
            "docx_projects": len(docx_projects),
            "total_merged_projects": len(merged_projects),
            "extraction_date": "2025-06-23"
        },
        "projects": merged_projects
    }
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nCombined JSON file created at: {output_path}")
    print(f"Total projects: {len(merged_projects)}")
    print(f"- From Excel only: {len([p for p in merged_projects if p['source'] == 'excel'])}")
    print(f"- From DOCX only: {len([p for p in merged_projects if p['source'] == 'docx'])}")
    print(f"- From both sources: {len([p for p in merged_projects if p['source'] == 'both'])}")
    
    # Print sample data for verification
    if merged_projects:
        print("\nSample merged project data:")
        # Find a project that has both sources if available
        sample_project = next((p for p in merged_projects if p['source'] == 'both'), merged_projects[0])
        print(json.dumps(sample_project, indent=2))

if __name__ == "__main__":
    main()
