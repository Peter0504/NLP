NLP Pipeline: Text Classification & Hierarchical Topic Clustering

# Overview
This project implements a fully automated, end-to-end Natural Language Processing (NLP) pipeline evaluating the full 20 Newsgroups dataset (>18,000 documents across 20 classes). It explores two distinct paradigms of feature engineering for classification, followed by an unsupervised, AI-driven clustering approach.

The pipeline executes in three sequential phases:
Part 1: Classic Features (TF-IDF)** — Evaluates traditional sparse bag-of-words vectorization using Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest.
Part 2: Dense Embeddings** — Projects text into a 384-dimensional continuous semantic space using Hugging Face's `SentenceTransformer` (`all-MiniLM-L6-v2`) and benchmarks the same classical classifiers.
Part 3: Hierarchical Topic Clustering** — Computes optimal $K$-Means clusters (< 10) dynamically via the Elbow Method, extracts geometrically representative documents, and uses OpenAI's Large Language Models to generate a human-readable, 2-level hierarchical topic tree.

# Setup & Installation

# 1. Prerequisites
Ensure you have Python 3.9+ installed on your system. 
To enable dynamic LLM topic generation in Part 3, you will need an OpenAI API key. *(Note: If no API key is provided, the script safely defaults to a local keyword-based heuristic fallback).*

# 2. Install Dependencies
Install the required libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn sentence-transformers openai

# API Key Configuration
client = OpenAI(api_key="sk-your-actual-api-key")

# How to run
python "Text Classification & Hierarchical Topic Clustering.py"