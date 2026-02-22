System Architecture & Data Flow


This document outlines the architectural flow and module responsibilities of the end-to-end NLP Pipeline. 

# Data Flow

To optimize execution time and memory overhead on the 10,000+ row dataset, data ingestion and feature extraction are centralized.

1.  Ingestion:** The `main()` orchestrator fetches the training and testing sets.
2.  Global Embedding:** The raw text is passed through `SentenceTransformer` once. 
3.  Branching:** * The raw text arrays are passed to Part 1, which builds internal `TfidfVectorizer` pipelines.
      The pre-computed dense embeddings are passed to Part 2 for classification.
      The same dense embeddings are passed to Part 3 for unsupervised K-Means clustering.

# Module Responsibilities

# Orchestration & Initialization
* `main()`: Acts as the pipeline controller. Fetches the dataset, initializes the `SentenceTransformer` model, computes the global dense embeddings `X_train_emb` and `X_test_emb`, and invokes the downstream task modules.

# Supervised Classification Tracks
* `run_TF_IDF(data_train, data_test)`: Manages the sparse feature track. Wraps vectorization and modeling in `sklearn.pipeline.Pipeline` to strictly prevent data leakage. Triggers evaluation and visualization.
* `run_Embeddings(X_train_emb, X_test_emb, data_train, data_test)`: Manages the dense feature track. Applies a `MinMaxScaler` specifically for Multinomial Naive Bayes to handle negative values in the continuous vector space. Triggers evaluation and visualization.
* `compare_and_analyze_metrics()`: Consolidates the dictionaries returned by Part 1 and Part 2, rendering a Pandas DataFrame for direct side-by-side metric comparison, followed by an automated analytical breakdown.

# Unsupervised Clustering Engine
* `run_Hierarchical_Topic_Clustering(embeddings, data)`: The primary controller for Part 3. Executes the top-level clustering, isolates the two largest clusters, and performs second-level sub-clustering on those specific subsets.
* `find_elbow_point(wcss)`: A mathematical utility that calculates the point of maximum curvature (the elbow) from the Within-Cluster Sum of Squares (WCSS) array, allowing the pipeline to autonomously select optimal $K$ without human intervention.
* `get_representative_docs(...)`: Uses `numpy.linalg.norm` to calculate Euclidean distances within a cluster, isolating the geometric center (centroid) and returning the top $N$ closest documents to ensure high-quality, noise-free prompts.
* `generate_llm_label(docs)`: Handles external Generative AI integrations. Truncates texts to respect context windows and queries the `gpt-3.5-turbo` model for a concise topic label. Features an integrated fallback heuristic if network requests fail or API keys are missing.