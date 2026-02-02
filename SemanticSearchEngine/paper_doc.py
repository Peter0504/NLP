import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import arxiv
import tiktoken

# --- Configuration ---
st.set_page_config(page_title="Semantic Paper Search", layout="wide")

# --- Session State Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# --- Sidebar: Setup & Inputs ---
st.sidebar.title("⚙️ Configuration")

# 1. API Key Input
api_key = st.sidebar.text_input("1. Enter OpenAI API Key", type="password")

# 2. Data Source Selection
data_source = st.sidebar.radio("2. Select Data Source", ["Manual PDF Upload", "ArXiv Search"])

uploaded_files = None
domain_query = ""
num_papers = 50

if data_source == "Manual PDF Upload":
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Papers", 
        type="pdf", 
        accept_multiple_files=True
    )
else:
    domain_query = st.sidebar.text_input("Domain/Topic", value="Computer Vision")
    num_papers = st.sidebar.slider("Number of Papers", 50, 200, 100)

# 3. Manual Start Button
start_button = st.sidebar.button("🚀 Start Processing")


# --- Functions ---

def chunk_text(text, chunk_size=2000, overlap=200):
    """
    Splits text into chunks of 2000 tokens.
    This safely fits into the 8192 limit of text-embedding-3-large.
    """
    try:
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    tokens = encoding.encode(text)
    
    chunks = []
    # Create chunks with overlap so context isn't lost at the cut point
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def get_embedding(text, client, model="text-embedding-3-large"):
    """Generates embedding for a single text string."""
    text = text.replace("\n", " ").strip()
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def parse_pdf(uploaded_file):
    """Extracts text from a simplified PDF object."""
    try:
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return ""

def process_and_embed(raw_docs, client):
    """
    Takes a LIST of raw documents (dict), chunks them, and embeds every chunk.
    """
    chunked_data = []
    progress_bar = st.progress(0)
    status = st.empty()
    
    total_docs = len(raw_docs)
    
    for i, doc in enumerate(raw_docs):
        status.text(f"Processing document {i+1}/{total_docs}: {doc['title']}")
        
        # 1. Split document into chunks
        text_chunks = chunk_text(doc['summary'])
        
        # 2. Embed each chunk individually
        for chunk_idx, chunk_content in enumerate(text_chunks):
            embedding = get_embedding(chunk_content, client)
            chunked_data.append({
                "title": doc['title'],
                "chunk_id": chunk_idx,
                "content": chunk_content,  # This is the split text
                "url": doc['url'],
                "embedding": embedding
            })
            
        progress_bar.progress((i + 1) / total_docs)
    
    status.empty()
    progress_bar.empty()
    return pd.DataFrame(chunked_data)

def search_documents(query, df, client, top_k=5):
    """Computes cosine similarity and returns top K results."""
    query_embedding = get_embedding(query, client)
    
    doc_embeddings = np.array(df['embedding'].tolist())
    query_embedding_np = np.array([query_embedding])
    
    similarities = cosine_similarity(query_embedding_np, doc_embeddings).flatten()
    
    # Ensure we don't request more results than available chunks
    actual_k = min(top_k, len(df))
    top_indices = similarities.argsort()[-actual_k:][::-1]
    
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    return results

def summarize_results(query, results, client):
    """Uses LLM to synthesize an answer."""
    context_text = ""
    for _, row in results.iterrows():
        # Use 'content' (the chunk) NOT 'summary'
        context_text += f"Paper: {row['title']}\nContent: {row['content'][:1500]}...\n\n"

    prompt = f"""
    You are a research assistant. Answer the user's question based ONLY on the following scientific papers. 
    Cite the papers by title when you use their information.
    
    User Question: {query}
    
    Context Papers:
    {context_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content


# --- Main Application Logic ---

st.title("🧬 Semantic Paper Search")

# EXECUTION BLOCK (Only runs when Button is clicked)
if start_button:
    if not api_key:
        st.error("❌ Please enter your OpenAI API Key first.")
    else:
        client = openai.OpenAI(api_key=api_key)
        
        with st.spinner("Processing Documents..."):
            raw_data = [] # This is a LIST of dictionaries
            
            # CASE A: Manual Upload
            if data_source == "Manual PDF Upload":
                if not uploaded_files:
                    st.error("❌ Please upload at least one PDF file.")
                    st.stop()
                
                for pdf_file in uploaded_files:
                    text = parse_pdf(pdf_file)
                    if text:
                        raw_data.append({
                            "title": pdf_file.name,
                            "summary": text,
                            "url": "Local File",
                            "published": "N/A"
                        })
            
            # CASE B: ArXiv Search
            else:
                search = arxiv.Search(
                    query=domain_query,
                    max_results=num_papers,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                for result in search.results():
                    raw_data.append({
                        "title": result.title,
                        "summary": result.summary.replace("\n", " "),
                        "url": result.pdf_url,
                        "published": result.published.strftime("%Y-%m-%d")
                    })
            
            if raw_data:
                st.info(f"Chunking and embedding {len(raw_data)} documents...")
                
                # FIXED: Pass the LIST 'raw_data', not a DataFrame
                df_chunks = process_and_embed(raw_data, client)
                
                # Save to Session State
                st.session_state.df = df_chunks
                st.session_state.processing_complete = True
                st.success(f"✅ Knowledge Base Built! Indexed {len(df_chunks)} chunks.")
            else:
                st.error("No data found or processed.")

# SEARCH INTERFACE (Only shows after processing is complete)
if st.session_state.processing_complete and st.session_state.df is not None:
    st.divider()
    
    # Re-initialize client for the search phase using the key
    if api_key:
        client = openai.OpenAI(api_key=api_key)
        
        user_query = st.text_input("🔎 Ask a question about the papers:", placeholder="e.g., What is the methodology used?")
        
        if user_query:
            with st.spinner("Analyzing..."):
                top_results = search_documents(user_query, st.session_state.df, client, top_k=4)
                summary = summarize_results(user_query, top_results, client)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("💡 Answer")
                    st.markdown(summary)
                
                with col2:
                    st.subheader("📄 Sources used")
                    for _, row in top_results.iterrows():
                        with st.expander(f"{row['title']} ({row['similarity_score']:.2f})"):
                            # Use 'content' for the preview text
                            st.caption(row['content'][:300] + "...")
    else:
        st.warning("Please keep your API Key in the sidebar to search.")

elif not start_button:
    st.info("👈 Enter API Key and click 'Start Processing' in the sidebar to begin.")
