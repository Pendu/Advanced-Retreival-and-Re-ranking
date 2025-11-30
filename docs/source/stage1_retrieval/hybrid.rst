Hybrid Dense-Sparse Methods
============================

Hybrid methods combine the strengths of both sparse (BM25) and dense (neural) retrieval 
to achieve robustness across different query types.

The Complementarity Principle
------------------------------

**BM25 Strengths:**

* Exact keyword matching (entity names, IDs, codes)
* No vocabulary mismatch for technical terms
* Works well for rare terms

**Dense Strengths:**

* Semantic matching (synonyms, paraphrases)
* Handles natural language questions
* Captures context and intent

**Together:** Cover both keyword-heavy and semantic queries.

Hybrid Methods Literature
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index (DENSPI) <https://arxiv.org/abs/1906.05807>`_
     - Seo et al.
     - ACL 2019
     - `Code <https://github.com/uwnlp/denspi>`_
     - **Dense-Sparse Phrase Index**: Combines dense vectors with sparse phrase matching. Real-time phrase-level retrieval. Stores billions of phrase representations for precise extraction.
   * - `Complementing Lexical Retrieval with Semantic Residual Embedding <https://arxiv.org/abs/2004.13969>`_
     - Gao et al.
     - ECIR 2021
     - NA
     - **Learn What BM25 Misses**: Neural model learns semantic "residual"—what BM25 fails to capture. Highly efficient complementarity through orthogonalization objective.
   * - `DensePhrases: Learning Dense Representations of Phrases at Scale <https://arxiv.org/abs/2012.12624>`_
     - Lee et al.
     - ACL 2021
     - `Code <https://github.com/princeton-nlp/DensePhrases>`_
     - **Dense Phrase Embeddings**: Every phrase gets dense vector. Novel negative sampling at phrase level. Can serve as knowledge base for multi-hop reasoning.

Implementation Patterns
-----------------------

Pattern 1: Score Fusion
^^^^^^^^^^^^^^^^^^^^^^^

Simplest approach: retrieve with both, combine scores.

.. code-block:: python

   # Retrieve from both indices
   bm25_results = bm25.search(query, k=100)
   dense_results = bi_encoder.search(query, k=100)
   
   # Normalize scores to [0, 1]
   bm25_scores = normalize(bm25_results.scores)
   dense_scores = normalize(dense_results.scores)
   
   # Weighted fusion
   alpha = 0.5  # Tunable weight
   for doc_id in union(bm25_results, dense_results):
       combined_score = alpha * bm25_scores[doc_id] + (1 - alpha) * dense_scores[doc_id]
   
   # Rank by combined score
   final_results = rank_by_score(combined_score)[:100]

**Tuning alpha**: Use validation set to find optimal weight (typically 0.3-0.7).

Pattern 2: Cascade Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use fast method first, then refine with slow method.

.. code-block:: python

   # Stage 1a: BM25 (very fast, cast wide net)
   bm25_candidates = bm25.search(query, k=10000)
   
   # Stage 1b: Dense re-rank to top-100
   dense_scores = bi_encoder.score(query, bm25_candidates)
   top_100 = rank_by_score(bm25_candidates, dense_scores)[:100]

**Advantage**: BM25 is so fast that retrieving 10K docs costs ~5ms extra but improves recall.

Pattern 3: Semantic Residual
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train neural model to capture what BM25 misses.

.. code-block:: python

   # From Gao et al., ECIR 2021
   # Loss encourages dense model to be orthogonal to BM25
   
   loss = contrastive_loss(query, pos, neg) \\
        + lambda * orthogonality_loss(dense_scores, bm25_scores)
   
   # At inference: dense focuses on semantic gaps

When to Use Hybrid
------------------

.. list-table:: Use Case Recommendations
   :header-rows: 1
   :widths: 30 70

   * - Scenario
     - Recommendation
   * - **Mixed Query Types**
     - Use hybrid (some queries keyword-heavy, some semantic)
   * - **Technical Domains**
     - Use hybrid (entity names need exact match, concepts need semantics)
   * - **Maximum Recall**
     - Use hybrid (BM25 catches what dense misses and vice versa)
   * - **Unknown Query Distribution**
     - Start with hybrid (safer than choosing one)
   * - **Homogeneous Semantic Queries**
     - Dense only (hybrid overhead not worth it)
   * - **Pure Keyword Search**
     - BM25 only (faster, simpler)

Empirical Results
-----------------

Typical improvements from hybrid over single method:

.. code-block:: text

   Dataset: MS MARCO Dev
   BM25 only:        MRR@10 = 0.187
   Dense only:       MRR@10 = 0.311
   Hybrid (α=0.5):   MRR@10 = 0.336  (+8% over dense alone!)
   
   Dataset: Natural Questions
   BM25 only:        Recall@100 = 0.73
   Dense only:       Recall@100 = 0.85
   Hybrid (α=0.4):   Recall@100 = 0.89  (+4.7% over dense alone)

**Pattern**: Hybrid helps most when query distribution is diverse.

Implementation Resources
-------------------------

**Libraries with Hybrid Support**

.. code-block:: python

   # Haystack framework
   from haystack.nodes import BM25Retriever, EmbeddingRetriever
   from haystack.pipelines import Pipeline
   
   bm25 = BM25Retriever(document_store)
   dense = EmbeddingRetriever(document_store, model="BAAI/bge-base-en-v1.5")
   
   # Haystack automatically fuses scores
   
   # LlamaIndex
   from llama_index import VectorStoreIndex, SimpleKeywordTableIndex
   from llama_index.query_engine import RetrieverQueryEngine
   
   # Combines vector and keyword retrieval

Best Practices
--------------

1. **Always tune the fusion weight (alpha)** on validation set
2. **Normalize scores** before combining (different ranges)
3. **Consider query type classification**: route to BM25 or dense based on query
4. **Monitor both components**: ensure neither is degraded
5. **Evaluate on diverse benchmark** (BEIR has 18 datasets)

Next Steps
----------

* See :doc:`sparse` for detailed BM25 explanation
* See :doc:`dense_baselines` for dense retrieval fundamentals
* See :doc:`hard_mining` for improving dense component
* See :doc:`late_interaction` for ColBERT's unified approach

