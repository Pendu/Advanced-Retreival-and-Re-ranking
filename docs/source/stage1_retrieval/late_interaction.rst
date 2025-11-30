Late Interaction (ColBERT)
===========================

**Do they do Stage-1 and Stage-2 together?**

Short Answer
------------

**Yes, late interaction models like ColBERT bridge the gap between Stage 1 and Stage 2.**

They can serve as:

* **High-quality Stage 1**: Retrieval from millions of documents (with optimized indexing)
* **Fine-grained Stage 2**: Token-level matching that mimics cross-encoder precision

Long Answer: Why Late Interaction is Special
---------------------------------------------

**Standard Dense Retrieval (Stage 1)**

* Compresses entire document into single vector (768-d)
* Fast but loses nuance
* Can't capture fine-grained matches

**Cross-Encoders (Stage 2)**

* Full attention between every query-document token pair
* Extremely accurate but prohibitively slow
* Can't pre-compute (needs query at inference time)

**Late Interaction (ColBERT) - The Bridge**

* Stores vector for *every token* in document
* Performs interaction *after* retrieval ("late")
* Fast enough for Stage 1, accurate enough for Stage 2

The ColBERT Architecture
-------------------------

.. code-block:: text

   Standard Bi-Encoder:
   ┌─────────┐                    ┌──────────┐
   │  Query  │ → BERT → [768-d] → │          │
   │         │                    │ Dot Prod │ → Score
   │Document │ → BERT → [768-d] → │          │
   └─────────┘                    └──────────┘
   
   ColBERT (Late Interaction):
   ┌─────────┐                    ┌──────────┐
   │  Query  │ → BERT → [32 x 128-d] →│      │
   │ (32 tok)│                         │MaxSim│ → Score
   │Document │ → BERT → [200 x 128-d]→│      │
   │(200 tok)│                         └──────┘
   └─────────┘
   
   MaxSim Operation:
   For each query token, find max similarity with any document token,
   then sum across all query tokens.

ColBERT Literature
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT <https://arxiv.org/abs/2004.12832>`_
     - Khattab & Zaharia
     - SIGIR 2020
     - `Code <https://github.com/stanford-futuredata/ColBERT>`_
     - **Late Interaction**: Retains token-level embeddings and computes MaxSim *after* retrieval. Captures fine details like cross-encoders but indexable like bi-encoders.
   * - `ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction <https://arxiv.org/abs/2112.01488>`_
     - Santhanam et al.
     - NAACL 2022
     - `Code <https://github.com/stanford-futuredata/ColBERT>`_
     - **Compression + Denoising**: Residual compression reduces index size by 6-10x. Denoised supervision from cross-encoder improves quality. Enables billion-scale retrieval.

Other Multi-Vector Methods
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring <https://arxiv.org/abs/1905.01969>`_
     - Humeau et al.
     - ICLR 2020
     - `Code <https://github.com/facebookresearch/ParlAI>`_
     - **Attention over Codes**: Learned codes represent the candidate document. Query attends to these codes. Balance of speed/accuracy; versatile for dialogue systems.
   * - `ME-BERT: Multi-Vector Encoding for Document Retrieval <https://arxiv.org/abs/2009.13013>`_
     - Luan et al.
     - arXiv 2020
     - NA
     - **Multi-Vector per Doc**: Multiple vectors per document to capture diverse topics. Each vector represents different aspect/sub-topic.

How ColBERT Does Both Stages
-----------------------------

ColBERT v2 with PLAID
^^^^^^^^^^^^^^^^^^^^^

The key innovation is the **PLAID (Performance-optimized Late Interaction Driver)** index:

**Stage 1 Capability: Fast Retrieval**

1. **Centroid-based pruning**: Groups similar token embeddings into centroids
2. **Early termination**: Stops scoring if document clearly won't be in top-k
3. **Quantization**: Compresses embeddings from 128-d float to 2-bit integers
4. **Result**: Can search 10M documents in ~50-100ms

**Stage 2 Capability: Fine-grained Matching**

1. **Token-level MaxSim**: Each query token finds best matching document token
2. **Captures phrases**: "capital of France" can match non-contiguous tokens
3. **Position awareness**: Different tokens can match different parts
4. **Result**: Accuracy approaching cross-encoders

Performance Comparison
----------------------

.. list-table:: Speed vs Accuracy Trade-off
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Method
     - Latency
     - Accuracy
     - Index Size
     - Use Case
   * - BM25
     - ~1ms
     - Baseline
     - Small (GB)
     - Keywords
   * - Bi-Encoder
     - ~10ms
     - +15%
     - Medium (10GB)
     - Semantic
   * - ColBERT
     - ~50ms
     - +25%
     - Large (100GB)
     - Both stages
   * - Cross-Encoder
     - ~1000ms/doc
     - +30%
     - None
     - Stage 2 only

**Key Insight**: ColBERT is 20x slower than bi-encoder but 20x faster than cross-encoder.

When to Use ColBERT
-------------------

✅ **Use ColBERT As Primary Retriever When:**

* Accuracy is critical (medical, legal, research)
* You can afford the larger index (100-300GB for 10M docs)
* Latency budget allows 50-200ms
* Want single-stage solution (no separate re-ranker needed)

✅ **Use ColBERT As Re-ranker When:**

* Bi-encoder retrieves top-1000
* ColBERT MaxSim re-ranks to top-100
* Cross-encoder (optional) produces final top-10
* Best of all worlds: fast initial retrieval, precise final ranking

❌ **Don't Use ColBERT When:**

* Index size is constrained (edge devices, mobile)
* Need sub-10ms latency (real-time autocomplete)
* Corpus is small enough for cross-encoder on everything (<10K docs)

Implementation Example
-----------------------

**Basic ColBERT Retrieval**

.. code-block:: python

   from colbert import Searcher
   from colbert.infra import ColBERTConfig
   
   # Initialize
   config = ColBERTConfig(root="experiments/")
   searcher = Searcher(index="my_index", config=config)
   
   # Search
   query = "What is the capital of France?"
   results = searcher.search(query, k=10)
   
   for passage_id, rank, score in results:
       print(f"Rank {rank}: {passages[passage_id]} (score: {score:.2f})")

**ColBERT as Re-ranker**

.. code-block:: python

   # Stage 1: Bi-encoder retrieves candidates
   bi_encoder = SentenceTransformer('BAAI/bge-base-en-v1.5')
   candidates = bi_encoder.search(query, corpus, top_k=1000)
   
   # Stage 2: ColBERT re-ranks
   colbert_scores = []
   for doc in candidates:
       score = colbert.maxsim(query, doc)  # Token-level matching
       colbert_scores.append(score)
   
   # Top-100 after ColBERT re-ranking
   top_100 = rank_by_score(candidates, colbert_scores)[:100]
   
   # Stage 3 (optional): Cross-encoder for final top-10
   cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
   final_top_10 = cross_encoder.rerank(query, top_100)[:10]

The Index Size Problem
-----------------------

**Why ColBERT Indexes Are Large**

Standard bi-encoder:

.. math::

   \text{Index Size} = N \times d \times 4 \text{ bytes}
   
   \text{For 10M docs: } 10M \times 768 \times 4 = 30\text{GB}

ColBERT:

.. math::

   \text{Index Size} = N \times \text{avg\_tokens} \times d \times 4
   
   \text{For 10M docs: } 10M \times 200 \times 128 \times 4 = 1\text{TB}

**ColBERTv2 Solutions**:

1. **Compression**: Quantization reduces to ~100GB (10x smaller)
2. **Pruning**: Remove low-importance tokens
3. **Residual encoding**: Store deltas from centroids

Advanced Topic: MaxSim Operation
---------------------------------

The MaxSim operation is what gives ColBERT its power:

.. code-block:: python

   def maxsim(query_embeddings, doc_embeddings):
       """
       query_embeddings: (num_query_tokens, 128)
       doc_embeddings: (num_doc_tokens, 128)
       """
       # Compute all pairwise similarities
       similarities = query_embeddings @ doc_embeddings.T  # (Q, D)
       
       # For each query token, find max similarity with any doc token
       max_per_query_token = similarities.max(dim=1).values  # (Q,)
       
       # Sum across all query tokens
       score = max_per_query_token.sum()
       
       return score

**Why This Works**:

* Query token "capital" matches doc token "capital" (exact)
* Query token "France" matches doc tokens "French", "Paris" (semantic)
* Flexible matching while maintaining efficiency

Comparison with Cross-Encoders
-------------------------------

.. list-table:: ColBERT vs Cross-Encoder
   :header-rows: 1
   :widths: 30 35 35

   * - Dimension
     - ColBERT
     - Cross-Encoder
   * - **Encoding**
     - Independent (query, doc separate)
     - Joint ([CLS] query [SEP] doc [SEP])
   * - **Interaction**
     - Late (after encoding)
     - Early (full self-attention)
   * - **Pre-computation**
     - Yes (doc embeddings offline)
     - No (must encode each pair)
   * - **Speed for 100 docs**
     - ~50ms (MaxSim is cheap)
     - ~5000ms (100 forward passes)
   * - **Accuracy**
     - 90-95% of cross-encoder
     - Best possible
   * - **Best Use**
     - Stage 1 or Stage 2
     - Stage 2 only

Poly-Encoders: Another Middle Ground
-------------------------------------

Poly-encoders offer a different trade-off:

**Architecture**:

1. Document → BERT → Multiple "code" vectors (e.g., 64 codes)
2. Query → BERT → Single query vector
3. Query attends to document codes
4. Weighted sum of codes based on attention

**Advantage**: More flexible than ColBERT's MaxSim

**Disadvantage**: More complex, less interpretable

**Use Case**: Dialogue systems, where documents are short and interaction patterns complex

Research Directions
-------------------

Current research on late interaction focuses on:

1. **Reducing Index Size**: Can we get ColBERT quality with bi-encoder size?
2. **Dynamic Pruning**: Adaptively decide which tokens to keep
3. **Learned Aggregation**: Learn better operations than MaxSim
4. **Multi-modal**: Extend late interaction to images, video
5. **Long Documents**: Handle documents with thousands of tokens

Next Steps
----------

* See :doc:`hard_mining` for how ColBERT trains with hard negatives
* See :doc:`dense_baselines` for comparison with standard bi-encoders
* See :doc:`hybrid` for combining ColBERT with BM25
* See :doc:`../stage2_reranking/cross_encoders` for true Stage 2 methods

