Semantic Paper Search (RAG)

A production-ready Retrieval-Augmented Generation (RAG) application built with Python. This tool allows researchers to perform semantic searches across multiple scientific PDF documents, retrieving exact context and synthesizing answers with citations.

Key Features

Dual Data Sources: 
    Manual Upload: Drag-and-drop local PDF files.
    ArXiv Integration: Auto-fetch papers by domain/topic directly from ArXiv.
Smart Chunking: Implements **Sliding Window Chunking** (2000 tokens) to handle large documents without exceeding token limits.
Semantic Search: Uses **Cosine Similarity** to find concepts, not just keywords (e.g., searching "cost" finds "financial overhead").
Citations: Every answer includes expandable "Reference Chunks" showing the exact source text and similarity score.
Model: Powered by OpenAI's `gpt-4o` for synthesis and `text-embedding-3-large` for high-precision retrieval.

Installation & Setup

Clone the Repository

git clone [https://github.com/your-username/semantic-paper-search.git](https://github.com/your-username/semantic-paper-search.git)
cd semantic-paper-search
pip install -r requirements.txt
streamlit run paper_doc.py or running run_app.bat
Paste your OpenAI API Key.
Select Data Source: Upload PDFs or type a topic (e.g., "Computer Vision").
Click Start Processing.
Once the "Knowledge Base Built" message appears, type questions in the search bar