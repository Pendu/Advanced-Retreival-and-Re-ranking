Dense Baselines & Fixed Embeddings
====================================

This section covers the foundational papers that established dense retrieval as a viable 
alternative to sparse methods like BM25.

The Dense Retrieval Revolution
-------------------------------

Before 2020, information retrieval was dominated by BM25. The key innovation of dense 
retrieval was to use pre-trained language models (BERT) to create semantic embeddings 
that could match queries and documents by meaning, not just keywords.

Dense Passage Retrieval (DPR)
------------------------------

**The Foundation Paper**

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Dense Passage Retrieval for Open-Domain Question Answering <https://arxiv.org/abs/2004.04906>`_
     - Karpukhin et al.
     - EMNLP 2020
     - `Code <https://github.com/facebookresearch/DPR>`_
     - **In-batch + BM25 Static**: The standard dual-encoder baseline using in-batch negatives and static BM25 hard negatives. Established that dense retrieval can outperform BM25.

**Key Components**

1. **Architecture**: Dual-encoder (separate BERT for query and passage)
2. **Training**: In-batch negatives + BM25-mined hard negatives
3. **Similarity**: Dot product of embeddings
4. **Index**: FAISS for fast approximate nearest neighbor search

**Why It Worked**

* Pre-trained BERT captures semantic meaning
* Hard negatives (from BM25) force discrimination
* Efficient indexing makes it practical

**Code Example**

.. code-block:: python

   from transformers import DPRQuestionEncoder, DPRContextEncoder
   import torch
   
   # Load DPR models
   q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
   ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
   
   # Encode
   query = "What is the capital of France?"
   query_emb = q_encoder(**tokenizer(query, return_tensors="pt")).pooler_output
   
   # Search (using pre-computed passage embeddings + FAISS)
   scores, indices = index.search(query_emb.numpy(), k=100)

Fixed Embeddings: RepBERT
--------------------------

**Extreme Efficiency**

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `RepBERT: Contextualized Text Embeddings for First-Stage Retrieval <https://arxiv.org/abs/2006.15498>`_
     - Zhan et al.
     - arXiv 2020
     - `Code <https://github.com/jingtaozhan/RepBERT-Index>`_
     - **Contextualized Fixed-Length**: Fixed-length embeddings with contextualization. Achieves efficiency comparable to bag-of-words while maintaining semantic understanding.

**Key Innovation**

RepBERT showed that you could get dense retrieval quality with near-BM25 speed by:

1. Pre-computing all passage embeddings offline
2. Using highly optimized indexing (quantization, compression)
3. Simple dot product similarity (no expensive operations)

**Performance vs Speed**

.. code-block:: text

   Speed →
   BM25: ████████████████████ (fastest, ~1ms)
   RepBERT: ███████████████ (fast, ~5ms)
   DPR: ██████████ (medium, ~10ms)
   ColBERT: ████ (slower, ~50ms)
   Cross-Encoder: █ (slowest, ~1000ms per doc)

Comparison: DPR vs RepBERT
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Dimension
     - DPR
     - RepBERT
   * - **Architecture**
     - Dual BERT encoders
     - Single shared BERT
   * - **Embedding Size**
     - 768-d (BERT hidden)
     - 768-d (BERT hidden)
   * - **Training**
     - In-batch + BM25 negatives
     - In-batch only
   * - **Speed**
     - ~10ms per query
     - ~5ms per query
   * - **Index Size**
     - Standard (4 bytes/dim)
     - Can be quantized heavily
   * - **Best For**
     - Accuracy
     - Speed/efficiency

When to Use Each
----------------

**Use DPR When:**

* Accuracy is more important than speed
* You have good hard negative mining
* Standard FAISS index is acceptable
* Following best practices from literature

**Use RepBERT When:**

* Speed is critical (near-BM25 performance needed)
* Index size must be minimal (e.g., edge deployment)
* Don't have resources for hard negative mining
* Want simplest possible dense retrieval

Modern Successors
-----------------

Both DPR and RepBERT have been superseded by more advanced methods, but they remain 
important baselines. Modern alternatives include:

**Contriever** (Facebook AI, 2022)
  Unsupervised dense retrieval with contrastive learning. No labels needed!

**BGE** (BAAI, 2023)
  State-of-the-art dense retriever with advanced hard negative mining.

**E5** (Microsoft, 2023)
  Multi-stage pre-training with massive scale (hundreds of millions of pairs).

**Nomic-Embed** (Nomic AI, 2024)
  Open-source, high-quality embeddings with permissive license.

Implementation Recommendations
-------------------------------

**For New Projects in 2024**

Don't implement DPR/RepBERT from scratch. Instead:

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   
   # Use modern pre-trained model
   model = SentenceTransformer('BAAI/bge-base-en-v1.5')  # Better than DPR
   
   # Or for speed
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Better than RepBERT

**Why?**

* Pre-trained on larger datasets
* Better hard negative mining during training
* Optimized inference
* Active maintenance

**But Still Study DPR/RepBERT Because:**

* Understand foundational architecture
* Baseline for your own research
* Many papers compare against them
* Core concepts still apply

Next Steps
----------

* See :doc:`hard_mining` for how modern methods improve over DPR's static negatives
* See :doc:`late_interaction` for ColBERT's approach to more expressive representations
* See :doc:`pretraining` for methods that improve the base encoders

