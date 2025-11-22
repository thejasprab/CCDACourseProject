# Limitations

This project showcases a scalable Spark-based recommender system for large-scale arXiv metadata.  
However, several **inherent constraints in data quality, modeling approach, system architecture, and evaluation methodology** limit overall performance.  
Some of these challenges stem from the nature of arXiv metadata, while others reflect trade-offs made to ensure the system can run on a single machine under course constraints.

---

## 1. Dataset Limitations

### **1.1 Metadata-only representation (no full text)**
The Kaggle arXiv dataset includes **titles, abstracts, authors, categories, and submission metadata**, but *not full PDF content*.  
Because of this, the recommendation engine relies entirely on abstract-level descriptions:

- Abstracts may omit methodology, results, or detailed contributions.  
- Many papers—especially math or theory—have short abstracts with limited semantic cues.  
- TF‑IDF representations capture only surface-level word frequency patterns.

As a result, the system cannot fully model deep semantic similarity, which modern embedding models can capture.

### **1.2 Inconsistent or missing metadata**
Throughout ingestion, several issues appeared:

- Missing fields (`doi`, `authors`, `submitted_date`, secondary categories).  
- Non-standard category formatting or multi-label inconsistencies.  
- Abstracts containing LaTeX math environments, broken markup, or special symbols.  

These inconsistencies reduce the reliability of analytics such as category co-occurrence networks or DOI coverage statistics.

### **1.3 Temporal / versioning noise**
arXiv allows multiple revisions per paper, but the dataset does not always distinguish:

- Version histories  
- Original vs updated categories  
- True submission timeline  

Thus, analyses like **category migration**, **yearly trends**, and **author productivity over time** can be skewed.

---

## 2. Methodological Limitations

### **2.1 TF‑IDF + cosine similarity is limited and non-semantic**
Although TF‑IDF is scalable and works well for sparse high-dimensional text, it has fundamental limitations:

- No modeling of synonyms or scientific terminology associations.  
- No contextual understanding (e.g., “graph neural networks” vs “GNNs”).  
- No semantic embeddings or neural representation learning.  
- No ability to detect subtle interdisciplinary relationships.

A purely lexical model can only approximate similarity based on overlapping vocabulary.

### **2.2 Exact Top‑K similarity search is expensive**
The system uses **exact cosine-similarity computation** over millions of sparse vectors:

- Complexity scales linearly with dataset size.  
- No Approximate Nearest Neighbor (ANN) acceleration.  
- Search speed may degrade significantly on full datasets (~2M papers).  

This is suitable for local demonstration but not for real academic-scale deployment.

### **2.3 Simplified document filtering**
Filtering abstracts by minimum length improves quality but is imperfect:

- Short but meaningful abstracts may be discarded.  
- Long but noisy abstracts (with excessive LaTeX) may pass filtering.  
- Some malformed JSON entries survive due to inconsistent formatting.

---

## 3. Pipeline & System Limitations

### **3.1 Running the full pipeline requires significant resources**
Even with optimized Spark configs, running the full dataset pipeline requires:

- 10–16GB Spark driver memory  
- Large disk space for shuffle spill  
- Extended runtime (especially for TF‑IDF training and complex SQL analytics)  

These constraints push the limits of a single-machine setup, especially without distributed cluster support.

### **3.2 Streaming workflow is a controlled simulation**
Structured Streaming is used on **artificial weekly drops generated from the dataset**:

- Not real incremental ingestion from an external data source.  
- Assumes consistent formatting and no schema drift.  
- Serves educational purposes but not production reliability.

### **3.3 Flask web app is intentionally minimal**
The web UI demonstrates the pipeline output but is not production-ready:

- Shared SparkSession limits concurrency.  
- Each search may trigger full vector comparison over millions of papers.  
- No caching layer or ANN index for performance optimization.  
- No load balancing or distributed backend support.

---

## 4. Evaluation Limitations

### **4.1 No formal model evaluation or ground-truth measurement**
The system currently lacks:

- Precision/recall evaluations  
- Benchmark comparisons (e.g., vs SciBERT, SPECTER embeddings)  
- Human-labeled relevance judgments  

Thus, recommendation quality is demonstrated qualitatively through examples rather than through statistical metrics.

### **4.2 Analytics depend heavily on metadata correctness**
Because trends and insights come exclusively from metadata:

- Missing or inconsistent fields lead to biased trends.  
- Author disambiguation (e.g., multiple authors sharing the same name) is not performed.  
- Category distribution analyses assume accurate primary/secondary category labeling.  

---

## 5. AI & ML–Specific Limitations

To highlight the AI-related constraints more explicitly:

### **5.1 No semantic embedding models**
Modern recommender systems use:

- SPECTER / SPECTER2 (scientific embeddings)  
- SciBERT / BioBERT domain models  
- Sentence-BERT / Contriever embeddings  
- Large language model–based embedding spaces  

These provide meaningful similarity even across non-overlapping vocabulary—something TF‑IDF cannot do.

### **5.2 No training of supervised or semi-supervised models**
Due to the dataset structure, the pipeline avoids:

- Classification tasks  
- Contrastive learning  
- Doc2Vec / neural retrieval models  
- Graph neural networks for citation/author graphs  

Thus, the system remains a high-quality but traditional lexical recommender.

### **5.3 No annotation data for quality scoring**
Without human feedback or citation-based relevance, the system cannot:

- Learn from user interactions  
- Adapt to trending research topics  
- Prioritize high-impact papers  
- Detect low-quality or spam submissions  

These limitations keep the system static and non-adaptive.

---

## 6. Future Improvements

To address the above constraints, several future enhancements are possible:

- Integrate embedding models (SPECTER, Sentence-BERT, SciBERT).  
- Build ANN index (FAISS, ScaNN, HNSW) for real-time search.  
- Parse PDF full text for rich semantic representations.  
- Normalize arXiv version history and metadata.  
- Move to distributed Spark cluster or Kubernetes environment.  
- Add evaluation datasets and human relevance scoring.  
- Introduce hybrid recommenders (content + graph-based citations).  

---

These limitations reflect both practical constraints (hardware, dataset) and methodological decisions (scalability, Spark-friendly algorithms).  
Despite these, the system successfully demonstrates an end‑to‑end large-scale recommender pipeline suitable for academic exploration and scalable data engineering workflows.
