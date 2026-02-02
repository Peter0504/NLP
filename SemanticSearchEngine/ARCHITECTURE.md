This document explains the "How" — the engineering decisions and data flow inside the script.

Technical Architecture
Overview
This application follows a standard **RAG (Retrieval-Augmented Generation)** architecture. It runs entirely locally (server-side via Streamlit) but relies on OpenAI's API for inference and vector generation.

Data Flow Pipeline

1. Ingestion Layer
Input: Raw PDF files (via `pypdf`) or ArXiv metadata (via `arxiv` API).
Processing: Text is extracted and cleaned (newlines removed) to ensure high-quality embeddings.

2. Chunking Strategy (Crucial)
To bypass the 8,192 token limit of OpenAI's embedding models, we implement Sliding Window Chunking:
Chunk Size: 2,000 tokens (approx. 1.5 pages of text).
Overlap: 200 tokens.
Why?: Overlap ensures that context isn't lost if a sentence is split perfectly in half between two chunks.
Tool: `tiktoken` (cl100k_base encoding) is used for precise token counting.

3. Embedding Layer
Model: text-embedding-3-large.
Dimensions: 3,072.
Process: Every text chunk is sent to the OpenAI API, returning a dense vector representation.
Storage: Vectors are stored in an in-memory Pandas DataFrame (`st.session_state.df`).

4. Retrieval Layer (Search)
Algorithm: Cosine Similarity (`sklearn.metrics.pairwise.cosine_similarity`).
Operation:
1.  User query is embedded into a vector $Q$.
2.  $Q$ is compared against matrix $D$ (all document chunks).
3.  Scores are ranked (0 to 1), and the top $K=5$ chunks are retrieved.

5. Generation Layer (Synthesis)
Model: `gpt-4o`.
Prompt Engineering: The top 5 chunks are injected into a system prompt:
> "Answer the user's question based ONLY on the following context..."
Output: The LLM generates a natural language response citing the provided context.

Tech Stack & Decisions

| Component | Technology | Reasoning |
| Frontend | Streamlit | Rapid prototyping, built-in state management. |
| Vector Store | Pandas (In-Memory) | Sufficient for <10,000 chunks; zero-setup compared to Pinecone/Chroma. |
| Math Engine | Scikit-Learn | Efficient C-optimized implementation of Cosine Similarity. |
| PDF Parser | PyPDF | Lightweight dependency, handles standard text layers well. |

Future Scalability
To move this from a prototype to a large-scale production app, the following changes would be required:
1.  Persistent Storage: Replace Pandas with a Vector Database (Pinecone, Weaviate) to save embeddings permanently.
2.  Async Processing: Use Celery/Redis to handle PDF processing in the background, preventing UI blocking.
3.  Hybrid Search: Combine Semantic Search (Vectors) with Keyword Search (BM25) for better precision on proper nouns (e.g., specific acronyms).