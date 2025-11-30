Overview
========

The New Bottleneck: Role of Advanced Negative Mining in Dense Retrieval
------------------------------------------------------------------------

Historical Context
^^^^^^^^^^^^^^^^^^

The evolution of information retrieval can be understood through distinct eras:

**1960s-1990s: Boolean and TF-IDF Era**

* Boolean retrieval: Exact keyword matching with AND/OR operators
* TF-IDF: Term frequency-inverse document frequency weighting
* Limitation: No semantic understanding, pure lexical matching

**1990s-2010s: Probabilistic IR and BM25 Dominance**

* BM25 (Robertson et al., 1994): Probabilistic relevance framework
* Became the de facto standard for web search engines
* Robust, interpretable, and efficient
* Limitation: Vocabulary mismatch problem (Furnas et al., 1987)

**2010s: Early Neural IR (Limited Success)**

* DSSM (Huang et al., 2013): Deep Structured Semantic Models
* CDSSM (Shen et al., 2014): Convolutional extensions
* Limitation: Shallow architectures, insufficient pre-training

**2018-Present: Dense Retrieval Revolution**

* BERT (Devlin et al., 2018): Pre-trained language model breakthrough
* DPR (Karpukhin et al., 2020): Dense Passage Retrieval establishes paradigm
* ColBERT, ANCE, RocketQA: Rapid innovation in training strategies

The Paradigm Shift from Sparse to Dense Retrieval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The field of information retrieval has undergone a fundamental paradigm shift. For decades, 
retrieval systems were dominated by sparse, lexical-based methods like BM25. These approaches, 
while robust and efficient, are limited by their reliance on exact keyword matching. They 
struggle to capture the underlying semantic intent of a query, failing when users employ 
different terminology (the "vocabulary mismatch" problem).

The advent of pre-trained language models (PLMs) such as BERT introduced the era of 
**dense retrieval**. Instead of sparse vectors of word counts, dense retrievers map queries 
and documents into low-dimensional, continuous-valued vectors (embeddings). These embeddings 
capture semantic relevance, allowing a model to retrieve documents that are contextually 
related to a query, even if they share no keywords.

However, this power comes with a critical dependency. The performance of these dense models 
is not just a function of their architecture (e.g., BERT, Sentence Transformers) but is 
overwhelmingly reliant on the quality of the data used during their contrastive or 
multi-negative training.

The Central Challenge
^^^^^^^^^^^^^^^^^^^^^

The central challenge has shifted from lexical matching to a new, more difficult problem: 
**teaching the model to distinguish between genuine semantic relevance and mere semantic similarity**.

This distinction is subtle but crucial:

* **Semantic Similarity**: Two texts that discuss similar topics or share contextual background
* **Semantic Relevance**: A document that actually answers the query or satisfies the information need

**Illustrative Example:**

For the query *"What is the capital of France?"*:

* **Semantically similar but irrelevant**: "Best tourist attractions in Paris" or "French economy overview"
* **Semantically relevant**: "Paris is the capital and most populous city of France"

Both types may have high cosine similarity in embedding space, but only one answers the query.

When Dense Retrieval Fails
^^^^^^^^^^^^^^^^^^^^^^^^^^

Dense retrieval is not universally superior. Understanding failure modes is critical:

**1. Exact Match Queries**

* Legal search: "42 USC § 1983" must match exact statute
* Code search: Function names require precise matching
* **Recommendation**: Use BM25 or hybrid approach

**2. Low-Resource Languages**

* Pre-trained models lack sufficient training data
* Embeddings may not capture semantic nuances
* **Recommendation**: Fine-tune on domain data or use multilingual models (mBERT, XLM-R)

**3. Frequently Updating Corpora**

* Index refresh overhead can be prohibitive
* Embeddings become stale as corpus changes
* **Recommendation**: Consider BM25 for real-time indexing, dense for periodic batches

**4. Negation and Subtle Semantics**

* "Not recommended" has high similarity to "recommended"
* Dense models struggle with logical operators
* **Recommendation**: Use cross-encoder re-ranking for precision-critical applications

Evaluation Metrics
^^^^^^^^^^^^^^^^^^

Standard metrics for retrieval evaluation:

**Retrieval Metrics (Stage 1)**

* **MRR@k** (Mean Reciprocal Rank): Average of 1/rank of first relevant document
* **Recall@k**: Fraction of relevant documents in top-k
* **nDCG@k** (Normalized Discounted Cumulative Gain): Position-weighted relevance

**Re-ranking Metrics (Stage 2)**

* **P@k** (Precision at k): Fraction of top-k that are relevant
* **MAP** (Mean Average Precision): Average precision across recall levels

.. note::

   **Why MRR@10?** This metric mimics user behavior—users rarely look past the first 
   page of results. Optimizing for MRR@10 directly improves user experience.

**Standard Benchmarks:**

* **MS MARCO**: 8.8M passages, ~500K training queries (Microsoft)
* **Natural Questions**: 300K queries from Google search (Google)
* **BEIR**: 18 diverse datasets for zero-shot evaluation (Thakur et al., 2021)

This is the core bottleneck in modern dense retrieval systems, and the reason why advanced 
negative mining techniques have become non-negotiable for achieving state-of-the-art performance.

Key Takeaways
^^^^^^^^^^^^^

* **Historical progression**: Boolean → TF-IDF → BM25 → Neural → Dense Retrieval
* **Sparse methods (BM25)**: Rely on keyword matching, suffer from vocabulary mismatch
* **Dense retrieval**: Uses embeddings to capture semantic relevance
* **Critical dependency**: Model performance depends on training data quality, not just architecture
* **Core challenge**: Distinguishing semantic relevance from semantic similarity
* **Failure modes**: Exact match, low-resource languages, dynamic corpora, negation
* **Evaluation**: MRR@10, Recall@100, nDCG@10 are standard metrics

Next Steps
^^^^^^^^^^

* See :doc:`hard_negatives` for the detailed analysis of the hard negative problem
* See :doc:`stage1_retrieval/hard_mining` for practical mining strategies
* See :doc:`rag_overview` for the two-stage retrieval-reranking pipeline
* See :doc:`benchmarks_and_datasets` for evaluation metrics and standard benchmarks
