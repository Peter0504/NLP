import streamlit as st
import os
import time
import re
import json # <--- Added for Jupyter support
from pathlib import Path
from litellm import completion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
st.set_page_config(page_title="SmartDocs Pro", layout="wide", page_icon="🧠")

# --- 1. EXPANDED FILE SUPPORT ---
SUPPORTED_EXTENSIONS = {
    # Core Code
    '.py', '.ipynb', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt',
    # Web & Styles
    '.html', '.css', '.scss', '.vue', '.svelte',
    # Data & Config
    '.sql', '.yaml', '.yml', '.json', '.xml', '.toml', '.ini',
    # Scripts & DevOps
    '.sh', '.bat', '.ps1', 'Dockerfile', 'Makefile'
}

IGNORED_DIRS = {'node_modules', 'venv', 'env', '.git', '__pycache__', 'dist', 'build', '.idea', '.vscode', '.ipynb_checkpoints'}

# --- 2. JUPYTER PARSER (The Fix) ---
def parse_ipynb(file_path):
    """
    Reads a .ipynb file, parses the JSON, and extracts only the CODE cells.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
            
        code_lines = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                # Join the lines of code in this cell
                source = "".join(cell.get('source', []))
                code_lines.append(source)
                
        return "\n\n# --- CELL BREAK ---\n\n".join(code_lines)
    except Exception as e:
        return f"# Error parsing Notebook: {str(e)}"

# --- 3. INTELLIGENT SCANNING ---
def get_universal_skeleton(code_text, extension):
    lines = code_text.split('\n')
    skeleton = []
    
    # Regex patterns
    patterns = {
        '.py': r'^\s*(def|class)\s+',
        '.ipynb': r'^\s*(def|class)\s+', # Treat notebooks like Python
        '.js': r'^\s*(function|class|const\s+.*=.*=>)\s+',
        '.ts': r'^\s*(function|class|interface|type)\s+',
        '.go': r'^\s*(func|type)\s+',
        '.rs': r'^\s*(fn|struct|impl|trait)\s+',
        '.sql': r'^\s*(CREATE|ALTER|DROP)\s+',
    }
    
    pattern = patterns.get(extension, r'^\s*(def|class|function|func|public|private)\s+')
    
    for line in lines:
        if re.search(pattern, line):
            skeleton.append(line.strip().rstrip('{:'))
            
    return "\n".join(skeleton[:50])

def scan_project_structure(source_dir):
    source_path = Path(source_dir).resolve()
    file_map = {}
    skeleton_map = {}
    
    if not source_path.exists():
        return None, None, "Source directory not found."

    for root, dirs, files in os.walk(source_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        
        for f in files:
            file_path = Path(root) / f
            
            if file_path.suffix in SUPPORTED_EXTENSIONS or f in SUPPORTED_EXTENSIONS:
                try:
                    # SPECIAL HANDLING FOR NOTEBOOKS
                    if file_path.suffix == '.ipynb':
                        content = parse_ipynb(file_path)
                    else:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    rel_path = str(file_path.relative_to(source_path))
                    file_map[rel_path] = content
                    skeleton_map[rel_path] = get_universal_skeleton(content, file_path.suffix)
                except Exception:
                    continue
                    
    return file_map, skeleton_map, None

# --- 4. CONTEXT AWARE GENERATION ---
def get_relevant_context(target_file, file_map, skeleton_map):
    if len(file_map) < 10:
        return "\n".join([f"File: {k}\nDefinitions:\n{v}" for k,v in skeleton_map.items() if k != target_file])

    try:
        paths = list(file_map.keys())
        contents = list(file_map.values())
        
        # Use TF-IDF but be safe about empty contents
        valid_contents = [c if c.strip() else "empty" for c in contents]
        
        vec = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vec.fit_transform(valid_contents)
        
        target_idx = paths.index(target_file)
        similarities = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()
        related_indices = similarities.argsort()[-4:-1]
        
        context_str = []
        for idx in related_indices:
            neighbor = paths[idx]
            context_str.append(f"Related File: {neighbor}\nKey Logic:\n{skeleton_map[neighbor]}")
            
        return "\n---\n".join(context_str)
    except:
        return "No specific context available."

def generate_smart_docs(filename, code_content, global_context, api_key, model):
    prompt = f"""
    You are a Senior Developer. Document the file '{filename}'.
    
    --- GLOBAL PROJECT CONTEXT (Cross-File Logic) ---
    {global_context}
    -------------------------------------------------
    
    FILE CONTENT:
    {code_content[:20000]} 
    
    TASK:
    Write a Markdown documentation file.
    1. Explain the Logic: How does this file interact with the project?
    2. Key Definitions: List functions/classes.
    3. Usage: Provide a code example.
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"Error after {max_retries} attempts: {str(e)}"

# --- 5. SESSION STATE UI ---
if "processed_files" not in st.session_state: st.session_state.processed_files = set()
if "doc_results" not in st.session_state: st.session_state.doc_results = {}
if "project_map" not in st.session_state: st.session_state.project_map = None
if "skeleton_map" not in st.session_state: st.session_state.skeleton_map = None

# --- 6. UI LAYOUT ---
st.title("🧠 SmartDocs: Jupyter & Code Analyzer")
st.markdown("Supports `.ipynb`, `.py`, `.js`, and more.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI Key", type="password")
    model = st.selectbox("Model", ["gpt-4o", "gpt-3.5-turbo"])
    # Default to ./src as requested
    src_dir = st.text_input("Source Folder", value="./src")
    dest_dir = st.text_input("Output Folder", value="./docs")

# Step 1: Scan
if st.button("1. Scan Project Structure"):
    if not api_key:
        st.error("Missing API Key")
    else:
        with st.spinner("Indexing Global Logic..."):
            f_map, s_map, err = scan_project_structure(src_dir)
            if err:
                st.error(err)
            elif not f_map:
                st.warning("No supported files found in ./src")
            else:
                st.session_state.project_map = f_map
                st.session_state.skeleton_map = s_map
                st.success(f"Indexed {len(f_map)} files (including Notebooks). Ready to generate.")

# Step 2: Generate
if st.session_state.project_map:
    st.divider()
    st.write(f"**Found:** {len(st.session_state.project_map)} files")
    
    if st.button("2. Generate Context-Aware Docs"):
        out_path = Path(dest_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        
        prog_bar = st.progress(0)
        status = st.empty()
        
        total = len(st.session_state.project_map)
        files = list(st.session_state.project_map.items())
        
        for i, (rel_path, content) in enumerate(files):
            if rel_path in st.session_state.processed_files:
                continue
                
            status.text(f"Analyzing: {rel_path}...")
            
            context = get_relevant_context(rel_path, st.session_state.project_map, st.session_state.skeleton_map)
            doc = generate_smart_docs(rel_path, content, context, api_key, model)
            
            save_p = out_path / Path(rel_path).with_suffix('.md')
            save_p.parent.mkdir(parents=True, exist_ok=True)
            save_p.write_text(doc, encoding='utf-8')
            
            st.session_state.processed_files.add(rel_path)
            st.session_state.doc_results[rel_path] = doc
            
            prog_bar.progress((i + 1) / total)
            
        st.success("Documentation Complete!")

# View Results
if st.session_state.doc_results:
    st.divider()
    selected_file = st.selectbox("View Generated Doc:", list(st.session_state.doc_results.keys()))
    st.markdown(st.session_state.doc_results[selected_file])