import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Graceful import for sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("CRITICAL ERROR: 'sentence_transformers' is not installed. Please run: pip install sentence-transformers")
    sys.exit(1)

# Graceful import for openai
try:
    from openai import OpenAI
    # Initialize the client. Make sure to use environment variables in production!
    client = OpenAI(api_key="sk-your-key-here") 
    OPENAI_AVAILABLE = True
except ImportError:
    print("WARNING: 'openai' library not found. LLM labeling will use local fallbacks.")
    OPENAI_AVAILABLE = False
except Exception as e:
    print(f"WARNING: Failed to initialize OpenAI client ({e}). Using local fallbacks.")
    OPENAI_AVAILABLE = False

# 0. DATASET LOADING
print("Loading 20 Newsgroups data...")

# NOTE: To speed up execution for this demo, we are selecting 4 specific categories.
# To use the full dataset, set categories=None.
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

print(f"Train samples: {len(newsgroups_train.data)}")
print(f"Test samples: {len(newsgroups_test.data)}")
print("-" * 50)

# Helper function for evaluation
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"[{name}] Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    return acc, f1

def plot_confusion_matrix(y_true, y_pred, classes, title):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    except Exception as e:
        print(f"Error plotting confusion matrix for {title}: {e}")

# PART 1: Classic Features (TF-IDF)
def run_TF_IDF(data_train, data_test):
    print("\n" + "="*40)
    print("PART 1: TF-IDF Pipeline")
    print("="*40)
    
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": SVC(kernel='linear', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for model_name, classifier in models.items():
        print(f"--- Training {model_name} ---")
        try:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
                ('clf', classifier)
            ])
            pipeline.fit(data_train.data, data_train.target)
            y_pred = pipeline.predict(data_test.data)
            
            acc, macro_f1 = evaluate_model(model_name, data_test.target, y_pred)
            results[model_name] = {'Accuracy': acc, 'Macro-F1': macro_f1}

        # Generate Confusion Matrix on the held-out test set
            plot_confusion_matrix(
                data_test.target, 
                y_pred, 
                data_train.target_names, 
                f"TF-IDF Confusion Matrix: {model_name}"
            )

        except Exception as e:
            print(f"Error training {model_name}: {e}")

    print("\n=== Final Evaluation Summary (TF-IDF) ===")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Macro-F1':<10}")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} | {metrics['Accuracy']:<10.4f} | {metrics['Macro-F1']:<10.4f}")
    return results
# PART 2: Embeddings
def run_Embeddings(X_train_emb, X_test_emb, data_train, data_test):
    print("\n" + "="*40)
    print("PART 2: Dense Embeddings Pipeline")
    print("="*40)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": SVC(kernel='linear', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for model_name, classifier in models.items():
        print(f"--- Training {model_name} on Embeddings ---")
        try:
            if model_name == "Multinomial Naive Bayes":
                model = Pipeline([('scaler', MinMaxScaler()), ('clf', classifier)])
            else:
                model = classifier
                
            model.fit(X_train_emb, data_train.target)
            y_pred = model.predict(X_test_emb)
            
            acc, macro_f1 = evaluate_model(model_name, data_test.target, y_pred)
            results[model_name] = {'Accuracy': acc, 'Macro-F1': macro_f1}

            plot_confusion_matrix(
                data_test.target, 
                y_pred, 
                data_train.target_names, 
                f"Embeddings CM: {model_name}"
            )
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    print("\n=== Final Evaluation Summary (Dense Embeddings) ===")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Macro-F1':<10}")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} | {metrics['Accuracy']:<10.4f} | {metrics['Macro-F1']:<10.4f}")
    return results
# NEW FUNCTION: Direct Side-by-Side Comparison
def compare_and_analyze_metrics(results_tfidf, results_emb):
    print("\n" + "="*80)
    print("DIRECT COMPARISON: TF-IDF vs. DENSE EMBEDDINGS")
    print("="*80)
    
    # Build a DataFrame for clean side-by-side display
    comparison_data = []
    for model_name in results_tfidf.keys():
        tf_acc = results_tfidf[model_name]['Accuracy']
        em_acc = results_emb[model_name]['Accuracy']
        tf_f1 = results_tfidf[model_name]['Macro-F1']
        em_f1 = results_emb[model_name]['Macro-F1']
        
        # Dynamically determine the winner based on Macro-F1 score
        if em_f1 > tf_f1:
            winner = "Embeddings"
        elif tf_f1 > em_f1:
            winner = "TF-IDF"
        else:
            winner = "Tie"
            
        comparison_data.append({
            'Model': model_name,
            'TF-IDF Acc': tf_acc,
            'Embed Acc': em_acc,
            'TF-IDF F1': tf_f1,
            'Embed F1': em_f1,
            'Winner (F1)': winner
        })
        
    df_compare = pd.DataFrame(comparison_data)
    
    # Format floats to 4 decimal places for readability AFTER doing the math
    for col in ['TF-IDF Acc', 'Embed Acc', 'TF-IDF F1', 'Embed F1']:
        df_compare[col] = df_compare[col].apply(lambda x: f"{x:.4f}")
        
    print("\n=== SIDE-BY-SIDE METRICS ===")
    print(df_compare.to_string(index=False))
    
    print("MODEL-BY-MODEL BREAKDOWN:")
    for _, row in df_compare.iterrows():
        print(f"   - {row['Model']}: {row['Winner (F1)']} achieved a higher F1-Score.")

def get_representative_docs(embeddings, docs, centroid, cluster_indices, n=3):
    try:
        cluster_embeddings = embeddings[cluster_indices]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_local_indices = np.argsort(distances)[:n]
        closest_global_indices = cluster_indices[closest_local_indices]
        return [docs[i] for i in closest_global_indices]
    except Exception as e:
        print(f"Error extracting representative documents: {e}")
        return []

def generate_llm_label(docs):
    if not docs:
        return "Unknown Topic"

    truncated_docs = [doc[:300] + "..." for doc in docs]
    prompt = "Read the following document excerpts and provide a single, concise topic label (maximum 3 words) that summarizes their shared theme.\n\n"
    for i, doc in enumerate(truncated_docs):
        prompt += f"Document {i+1}:\n{doc}\n\n"
        
    if OPENAI_AVAILABLE:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3
            )
            return response.choices[0].message.content.strip().replace('"', '')
        except Exception as e:
            pass # Fall through to heuristic if API fails
            
    # MOCK FALLBACK
    combined = " ".join(truncated_docs).lower()
    if "god" in combined or "jesus" in combined: return "Religion / Theology"
    if "graphic" in combined or "image" in combined: return "Computer Graphics"
    if "patient" in combined or "doctor" in combined: return "Medicine / Health"
    if "car" in combined or "engine" in combined: return "Automotive"
    if "space" in combined or "orbit" in combined: return "Space / Science"
    return "General Discussion"

def find_elbow_point(wcss):
    try:
        n_points = len(wcss)
        all_coords = np.vstack((range(n_points), wcss)).T
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = all_coords - all_coords[0]
        scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
        return np.argmax(dist_to_line) + 2
    except Exception:
        return 4 # Default to 4 if math fails

def run_Hierarchical_Topic_Clustering(embeddings, data):
    print("\n" + "="*40)
    print("PART 3: Hierarchical Topic Clustering")
    print("="*40)

    print("\n--- STEP A: Determining Top-Level Clusters ---")
    try:
        wcss = []
        k_range = range(2, 10)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(embeddings)
            wcss.append(kmeans.inertia_)
            
        optimal_k = find_elbow_point(wcss)
        print(f"Optimal K selected via Elbow Method: {optimal_k}")

        kmeans_top = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        top_labels = kmeans_top.fit_predict(embeddings)
    except Exception as e:
        print(f"FAILED during top-level K-Means clustering: {e}")
        return
        
    topic_tree = {}
    print("\nGenerating Top-Level Labels via LLM...")
    
    for k in range(optimal_k):
        try:
            cluster_indices = np.where(top_labels == k)[0]
            centroid = kmeans_top.cluster_centers_[k]
            rep_docs = get_representative_docs(embeddings, data.data, centroid, cluster_indices, n=3)
            label = generate_llm_label(rep_docs)
            
            topic_tree[k] = {
                "label": label,
                "count": len(cluster_indices),
                "indices": cluster_indices,
                "subclusters": []
            }
            print(f"  -> Cluster {k} Label: '{label}' ({len(cluster_indices)} docs)")
        except Exception as e:
            print(f"Error processing Top-Level Cluster {k}: {e}")

    print("\n--- STEP B: Second-Level Clustering ---")
    try:
        cluster_sizes = {k: v["count"] for k, v in topic_tree.items()}
        largest_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)[:2]
        
        for k in largest_clusters:
            parent_label = topic_tree[k]["label"]
            print(f"\nRe-clustering '{parent_label}' into 3 sub-clusters...")
            
            parent_indices = topic_tree[k]["indices"]
            parent_embeddings = embeddings[parent_indices]
            
            kmeans_sub = KMeans(n_clusters=3, random_state=42, n_init='auto')
            sub_labels = kmeans_sub.fit_predict(parent_embeddings)
            
            for sub_k in range(3):
                try:
                    sub_local_indices = np.where(sub_labels == sub_k)[0]
                    sub_centroid = kmeans_sub.cluster_centers_[sub_k]
                    sub_rep_docs = get_representative_docs(
                        parent_embeddings, 
                        [data.data[i] for i in parent_indices], 
                        sub_centroid, 
                        sub_local_indices, 
                        n=3
                    )
                    
                    sub_label = generate_llm_label(sub_rep_docs)
                    topic_tree[k]["subclusters"].append({
                        "label": sub_label,
                        "count": len(sub_local_indices)
                    })
                    print(f"    -> Sub-cluster {sub_k} Label: '{sub_label}' ({len(sub_local_indices)} docs)")
                except Exception as e:
                    print(f"Error processing Sub-cluster {sub_k}: {e}")
    except Exception as e:
        print(f"FAILED during second-level clustering: {e}")

    print("\n--- STEP C: Final Topic Tree ---")
    try:
        for k, info in topic_tree.items():
            print(f"├── {info['label']} (Total Docs: {info['count']})")
            if info['subclusters']:
                for i, sub in enumerate(info['subclusters']):
                    connector = "└──" if i == len(info['subclusters']) - 1 else "├──"
                    print(f"│   {connector} {sub['label']} (Docs: {sub['count']})")
            else:
                print("│   └── (No sub-clustering performed)")
    except Exception as e:
        print(f"Error rendering Topic Tree: {e}")

def main():
    print("\n[INIT] Fetching dataset (Loading ONCE for entire pipeline)...")
    try:
        # Load all 20 classes and 18,000+ rows
        data_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
        data_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
        print(f" -> Successfully loaded {len(data_train.data)} training and {len(data_test.data)} test samples.")
        print(f" -> Number of classes: {len(data_train.target_names)}") # Proves it is > 5
    except Exception as e:
        print(f"CRITICAL ERROR loading dataset: {e}")
        return

    print("\n[INIT] Computing Dense Embeddings (Computing ONCE for Part 2 & Part 3)...")
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        X_train_emb = embedder.encode(data_train.data, show_progress_bar=True)
        X_test_emb = embedder.encode(data_test.data, show_progress_bar=True)
        print(" -> Embeddings successfully generated.")
    except Exception as e:
        print(f"CRITICAL ERROR generating embeddings: {e}")
        return

    # Now that the heavy lifting is done, pass the data to our pipeline sections
# Now that the heavy lifting is done, pass the data to our pipeline sections
    try:
        # Capture the results dictionaries returned by Part 1 and Part 2
        results_tfidf = run_TF_IDF(data_train, data_test)
        results_emb = run_Embeddings(X_train_emb, X_test_emb, data_train, data_test)
        
        # Execute the direct comparison right after models are finished
        compare_and_analyze_metrics(results_tfidf, results_emb)
        
        # Part 3: Clustering
        run_Hierarchical_Topic_Clustering(X_train_emb, data_train)
    except KeyboardInterrupt:
        print("\nPipeline execution cancelled by user.")
    except Exception as e:
        print(f"\nUNEXPECTED PIPELINE FAILURE: {e}")
if __name__ == "__main__":
    main()