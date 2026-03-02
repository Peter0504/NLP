import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="Production RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Advanced RAG Chatbot")
st.markdown("Features: Multi-Query Retrieval, Persistent Vector DB, Conversation Memory, & Citations")

# --- SIDEBAR: SETTINGS & UPLOADS ---
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
    st.header("Document Processing")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, HTML)", 
        type=['pdf', 'docx', 'html'], 
        accept_multiple_files=True
    )
    process_btn = st.button("Process & Index Documents")

# --- INITIALIZE SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- HELPER FUNCTIONS ---
def process_documents(files):
    """Saves uploaded files temporarily, loads them, and extracts text & metadata."""
    documents = []
    temp_dir = tempfile.mkdtemp()
    
    for file in files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
            
        # Select appropriate loader based on extension
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext == "docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == "html":
            loader = UnstructuredHTMLLoader(file_path)
        else:
            continue
            
        # Load document with metadata (filename, page numbers, etc.)
        docs = loader.load()
        # Add source filename to metadata explicitly just in case
        for doc in docs:
            doc.metadata["source_file"] = file.name
        documents.extend(docs)
        
    return documents

def get_vector_store(documents):
    """Implements smart chunking and creates a persistent Chroma vector DB."""
    # Smart chunking: Recursive tries to split by paragraphs, then sentences, then words.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Initialize Vector DB (Chroma) with persistent storage
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db" # Persistent local storage
    )
    return vector_store

def get_rag_chain(vector_store):
    """Builds the RAG pipeline with Memory, Multi-Query Retrieval, and Citations."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 1. Base Retriever (Top K=4)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # ADVANCED FEATURE: Multi-Query Retriever
    # Generates multiple perspectives of the user query to ensure high recall
    advanced_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    # 2. History-Aware Retriever (handles follow-up questions)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, advanced_retriever, contextualize_q_prompt)
    
    # 3. QA Chain (Generates answer with context)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise and cite your sources based on the context metadata.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 4. Final RAG Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# --- MAIN APP LOGIC ---

# 1. Handle Document Processing
if process_btn and uploaded_files:
    if not openai_api_key:
        st.sidebar.error("Please provide an OpenAI API Key.")
    else:
        with st.spinner("Processing and chunking documents..."):
            raw_docs = process_documents(uploaded_files)
            st.session_state.vector_store = get_vector_store(raw_docs)
            st.sidebar.success(f"Successfully indexed {len(uploaded_files)} document(s)!")

# 2. Re-load existing Vector DB if available and not in session state
if not st.session_state.vector_store and os.path.exists("./chroma_db") and openai_api_key:
    st.session_state.vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=OpenAIEmbeddings()
    )

# 3. Conversational UI
# Display chat history
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
        
    if not st.session_state.vector_store:
        st.warning("Please upload and process documents first, or ensure the persistent DB exists.")
        st.stop()

    # Add user message to UI and history
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Keep only the last 5 exchanges (10 messages) for memory efficiency
    recent_history = st.session_state.chat_history[-10:] if len(st.session_state.chat_history) > 10 else st.session_state.chat_history

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking (and running multi-query retrieval)..."):
            rag_chain = get_rag_chain(st.session_state.vector_store)
            
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": recent_history
            })
            
            answer = response["answer"]
            source_docs = response["context"]
            
            st.markdown(answer)
            
            # Display Source Citations
            if source_docs:
                with st.expander("📚 Source Citations"):
                    for i, doc in enumerate(source_docs):
                        source_file = doc.metadata.get('source_file', 'Unknown Source')
                        page = doc.metadata.get('page', 'N/A')
                        st.markdown(f"**Source {i+1}:** `{source_file}` (Page: {page})")
                        st.caption(f"_{doc.page_content[:200]}..._")
                        st.divider()

    # Update history
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=answer)
    ])
