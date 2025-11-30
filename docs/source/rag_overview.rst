Overview of RAG and the Two-Stage Pipeline
===========================================

Retrieval-Augmented Generation (RAG) systems typically employ a two-stage architecture 
that balances efficiency with accuracy. Understanding this distinction is crucial for 
selecting appropriate methods and architectures.

The Two-Stage Architecture
---------------------------

.. code-block:: text

   ┌─────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌─────────┐
   │  Query  │ ───► │  Stage 1         │ ───► │  Stage 2         │ ───► │   LLM   │
   │         │      │  (Top-1000)      │      │  (Top-10)        │      │         │
   └─────────┘      │  Fast Retrieval  │      │  Precision       │      └─────────┘
                    └──────────────────┘      │  Re-ranking      │
                                              └──────────────────┘

*Conceptual flow: Query → Fast Retrieval (Stage 1) → Precision Re-ranking (Stage 2) → LLM Generation*

Stage 1: Retrieval (Candidate Selection)
-----------------------------------------

**Goal**: Efficiently fetch a small set of candidate documents (e.g., top-100 to top-1000) 
from a massive collection (millions to billions of documents).

**Key Requirement**: **Speed** - Must process millions of documents in milliseconds.

**Trade-off**: Sacrifices some accuracy for speed by using simpler similarity computations.

Approaches
^^^^^^^^^^

**Sparse Retrieval (BM25)**

* Uses keyword matching and term frequency statistics
* Extremely fast (inverted index lookup)
* Limited by vocabulary mismatch problem
* Best for: Keyword-heavy queries, legal/medical domains

**Dense Retrieval (Bi-Encoders)**

* Encodes queries and documents into fixed-size vectors independently
* Similarity computed via dot product or cosine similarity
* Can pre-compute and index all document vectors
* Captures semantic meaning beyond keywords
* Examples: DPR, ANCE, BGE, E5

**Late Interaction (ColBERT)**

* Hybrid approach: stores multiple vectors per document (one per token)
* More expressive than single-vector but still indexable
* Bridges gap between Stage 1 and Stage 2
* Can serve both roles depending on implementation

**Hybrid Methods**

* Combines sparse (BM25) and dense retrieval
* Leverages complementary strengths
* Examples: SPLADE, DENSPI, Semantic Residual

Key Characteristics
"""""""""""""""""""

.. list-table:: Stage 1 Requirements
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Description
   * - **Indexing**
     - Pre-computes document representations offline
   * - **Retrieval Speed**
     - Sub-second for millions of documents
   * - **Architecture**
     - Dual-encoder (query and doc encoded separately)
   * - **Similarity**
     - Simple operations (dot product, cosine)
   * - **Recall Focus**
     - Prioritizes not missing relevant documents

Stage 2: Re-ranking (Precision Scoring)
----------------------------------------

**Goal**: Precisely score the small set of candidates (e.g., 100 documents) retrieved 
in Stage 1 to produce a final ranked list (e.g., top-10).

**Key Requirement**: **Accuracy** - Must identify the truly relevant documents with high precision.

**Trade-off**: More computational cost is acceptable since candidate set is small.

Approaches
^^^^^^^^^^

**Cross-Encoders**

* Concatenates query and document: ``[CLS] query [SEP] document [SEP]``
* Full self-attention between all query-document token pairs
* Most accurate but slowest
* Must score each (query, doc) pair independently
* Examples: BERT re-ranker, MonoT5, RankLlama

**Poly-Encoders**

* Middle ground between bi-encoders and cross-encoders
* Uses learned "attention codes" to aggregate information
* Faster than cross-encoders, more accurate than bi-encoders
* Can cache some computation

**Late Interaction (ColBERT as Re-ranker)**

* Performs MaxSim operation over token pairs
* More fine-grained than single-vector similarity
* Can be used for both retrieval and re-ranking
* Efficient enough for top-100 candidates

Key Characteristics
"""""""""""""""""""

.. list-table:: Stage 2 Requirements
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Description
   * - **Indexing**
     - Not needed (online scoring)
   * - **Scoring Speed**
     - Can be slower (only 10-1000 candidates)
   * - **Architecture**
     - Cross-encoder (joint encoding of query-doc)
   * - **Interaction**
     - Full attention between query and document
   * - **Precision Focus**
     - Prioritizes ranking truly relevant docs at top

The Efficiency-Accuracy Trade-off
----------------------------------

.. code-block:: text

   Accuracy ↑
      │
      │                                    ┌─── Cross-Encoder (Stage 2)
      │                                ┌───┤
      │                            ┌───┘   └─── Poly-Encoder
      │                        ┌───┘
      │                    ┌───┘ ColBERT (Can do both!)
      │                ┌───┘
      │            ┌───┘ Dense Bi-Encoder (Stage 1)
      │        ┌───┘
      │    ┌───┘ BM25 (Sparse)
      │┌───┘
      └─────────────────────────────────────────────► Speed

Why Two Stages?
^^^^^^^^^^^^^^^

**Computational Reality**

* Running a cross-encoder on 10 million documents × 1 query = 10 million forward passes
* At ~50ms per forward pass: 500,000 seconds = 139 hours per query ❌
* Dense retrieval on 10 million documents: ~100ms per query ✅
* Then cross-encoder on top-100 candidates: ~5 seconds per query ✅
* **Total: ~5 seconds vs 139 hours**

**The Pipeline Math**

.. math::

   \text{Total Time} = \underbrace{O(N \cdot \log N)}_{\text{Stage 1: ANN Search}} + \underbrace{O(k \cdot C)}_{\text{Stage 2: Rerank top-k}}

Where:
- N = corpus size (millions)
- k = candidates to re-rank (100-1000)
- C = cross-encoder cost per pair

Since k ≪ N, this is vastly more efficient than O(N · C).

Common Configurations
---------------------

Configuration 1: Standard Two-Stage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Stage 1: Dense Retrieval
   bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
   candidates = bi_encoder.retrieve(query, corpus, top_k=100)
   
   # Stage 2: Cross-Encoder Re-ranking
   cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
   scores = cross_encoder.predict([(query, cand) for cand in candidates])
   final_results = rank_by_score(candidates, scores)[:10]

**Use case**: Maximum accuracy, acceptable latency (1-5 seconds)

Configuration 2: Hybrid Retrieval + Cross-Encoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Stage 1: Hybrid (Sparse + Dense)
   bm25_results = bm25.search(query, top_k=100)
   dense_results = bi_encoder.search(query, top_k=100)
   candidates = merge_and_dedupe(bm25_results, dense_results)  # ~150 docs
   
   # Stage 2: Cross-Encoder
   final_results = cross_encoder.rerank(query, candidates)[:10]

**Use case**: Robust to both keyword and semantic queries

Configuration 3: ColBERT-only (Single Stage)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # ColBERT does both retrieval and fine-grained matching
   colbert = ColBERT("colbert-ir/colbertv2.0")
   results = colbert.search(query, corpus, top_k=10)

**Use case**: When you want late interaction quality with single-stage simplicity

Configuration 4: Three-Stage (Sparse → Dense → Cross)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Stage 1a: BM25 (very fast, 10K candidates)
   bm25_candidates = bm25.search(query, top_k=10000)
   
   # Stage 1b: Dense retrieval (re-rank to 100)
   dense_scores = bi_encoder.score(query, bm25_candidates)
   top_100 = rank_by_score(bm25_candidates, dense_scores)[:100]
   
   # Stage 2: Cross-Encoder (final top-10)
   final_results = cross_encoder.rerank(query, top_100)[:10]

**Use case**: Maximum recall (BM25 cast wide net) + maximum precision (cross-encoder)

When to Use What
----------------

.. list-table:: Method Selection Guide
   :header-rows: 1
   :widths: 20 25 25 30

   * - Method
     - Best For
     - Latency
     - Use Case
   * - **BM25 Only**
     - Keyword queries
     - < 50ms
     - Legal search, exact match
   * - **Bi-Encoder Only**
     - Semantic search
     - 50-200ms
     - FAQ matching, chatbots
   * - **Bi-Encoder + Cross**
     - Accuracy critical
     - 1-5s
     - Question answering, RAG
   * - **Hybrid + Cross**
     - Robustness critical
     - 2-10s
     - Enterprise search
   * - **ColBERT**
     - Balance of both
     - 200-500ms
     - Research, precision needs

Key Research Questions
----------------------

The field continues to evolve around several key questions:

**For Stage 1 (Retrieval)**

1. How to mine hard negatives that improve discrimination?
2. How to handle false negatives in training data?
3. How to make models robust to domain shift?
4. How to efficiently update indexes as models improve?

**For Stage 2 (Re-ranking)**

1. How to train cross-encoders with limited labeled data?
2. How to distill cross-encoder knowledge into faster models?
3. How to handle position bias in training data?
4. How to make re-rankers calibrated (output meaningful probabilities)?

**Cross-Cutting**

1. Can we learn a single model that does both stages well?
2. How to leverage LLMs as zero-shot re-rankers?
3. How to optimize end-to-end (retrieval + re-ranking + generation)?

Organization of This Documentation
-----------------------------------

This documentation is organized to reflect the two-stage architecture:

**Stage 1: Retrieval** (:doc:`stage1_retrieval/index`)

* Sparse methods (BM25)
* Dense baselines (DPR, RepBERT)
* **Hard negative mining** - The core focus of dense retrieval research
* Late interaction (ColBERT)
* Hybrid approaches

**Stage 2: Re-ranking** (:doc:`stage2_reranking/index`)

* Cross-encoders
* Poly-encoders
* LLM-based re-rankers

Each section provides:

* Paper surveys with links to code
* Detailed explanations of key innovations
* Implementation guidance
* Trade-off analysis

Next Steps
----------

* Start with :doc:`stage1_retrieval/index` to understand dense retrieval fundamentals
* Focus on :doc:`stage1_retrieval/hard_mining` for the critical bottleneck in training
* See :doc:`stage2_reranking/index` for precision re-ranking methods
* Refer to :doc:`contributing` to add papers or improve documentation

