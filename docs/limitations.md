# Limitations

This system demonstrates a complete Spark-based recommender pipeline, but several structural limitations remain due to the nature of arXiv metadata and the constraints of scalable, single-machine execution.

---

## 1. Dataset-Level Limitations

### 1.1 Metadata-only text (no full paper content)
The system relies solely on titles, abstracts, and categories because the dataset does not include PDF full text.  
This limits semantic depth and restricts the recommender to high-level lexical similarity rather than full scientific understanding.

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
- Search speed may degrade significantly on full datasets (~3M papers).  

This is suitable for local demonstration but not for real academic-scale deployment.

### **2.3 Simplified document filtering**
Filtering abstracts by minimum length improves quality but is imperfect:

- Short but meaningful abstracts may be discarded.  
- Long but noisy abstracts (with excessive LaTeX) may pass filtering.  
- Some malformed JSON entries survive due to inconsistent formatting.

---

## 2. Pipeline & System Limitations

### 2.1 Single-machine resource constraints
Training TF-IDF and running complex SQL analyses on the full arXiv dataset requires significant memory and disk spill space.  
These limitations reflect hardware constraints rather than Spark’s capability on distributed clusters.

### 2.2 Streaming is a controlled simulation
The streaming workflow processes synthetic weekly batches rather than live external data, which limits realism but satisfies the course requirement to demonstrate Structured Streaming.
