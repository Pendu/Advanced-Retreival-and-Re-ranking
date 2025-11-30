Stage 2: Re-ranking Methods
============================

This section covers methods for the second stage of the RAG pipeline: precisely scoring 
the candidate documents retrieved in Stage 1.

.. toctree::
   :maxdepth: 2
   :caption: Stage 2 Topics:

   cross_encoders
   llm_rerankers

Overview
--------

Stage 2 re-ranking focuses on **precision over speed**. Since the candidate set is small 
(typically 10-1000 documents), we can afford more expensive computations to get highly 
accurate relevance scores.

Why Re-ranking is Needed
-------------------------

**The Stage 1 Limitation**

Bi-encoders (Stage 1) encode query and document *independently*:

* No interaction between query and document tokens
* Can't perform complex reasoning about relevance
* Limited to similarity in embedding space

**The Stage 2 Solution**

Re-rankers encode query and document *jointly*:

* Full attention between all query-document token pairs
* Can perform complex relevance reasoning
* Much higher accuracy at cost of speed

The Accuracy Gain
^^^^^^^^^^^^^^^^^

Typical improvements when adding Stage 2:

.. code-block:: text

   Dataset: MS MARCO Dev (1000 candidates from Stage 1)
   
   Bi-encoder only:        MRR@10 = 0.311
   + Cross-encoder:        MRR@10 = 0.389  (+25% improvement!)
   
   Dataset: Natural Questions
   Bi-encoder only:        Top-10 Accuracy = 0.68
   + Cross-encoder:        Top-10 Accuracy = 0.81  (+19% improvement!)

**Key Insight**: Re-ranking the top-100 with cross-encoder provides massive gains for 
just 100 forward passes (~5 seconds).

Architecture Types
------------------

Cross-Encoders
^^^^^^^^^^^^^^

**Most Common**: BERT-based cross-encoder

* Concatenates: ``[CLS] query [SEP] document [SEP]``
* Self-attention across all tokens
* Classification head predicts relevance
* Highest accuracy

**Variants:**

* MonoBERT: BERT cross-encoder for binary classification
* MonoT5: T5 model generates "true"/"false" token
* RankT5: T5 generates relevance score directly
* RankLlama: Large language model fine-tuned for ranking

Poly-Encoders
^^^^^^^^^^^^^

**Middle Ground**: Faster than cross-encoder, better than bi-encoder

* Document → Multiple learned "codes" (e.g., 64 codes)
* Query attends to codes
* Much faster than cross-encoder (can pre-compute codes)

LLM Re-rankers
^^^^^^^^^^^^^^

**Latest Trend**: Zero-shot re-ranking with instruction-tuned LLMs

* Prompt LLM: "Is this passage relevant to this query?"
* No training needed
* Can provide explanations
* Expensive but highly effective

Organization of This Section
-----------------------------

**Cross-Encoders** (:doc:`cross_encoders`)

* Traditional BERT-based re-rankers
* MonoT5 and RankT5
* Training strategies
* Implementation guide

**LLM Re-rankers** (:doc:`llm_rerankers`)

* Zero-shot prompting approaches
* RankGPT, RankLlama
* Listwise vs pointwise ranking
* Cost-performance trade-offs

When to Use Stage 2
-------------------

✅ **You Need Stage 2 When:**

* Top-10 accuracy is critical (user sees only first page)
* False positives are costly (e.g., medical, legal)
* You can afford 1-10 second latency
* Final answer quality >> speed

❌ **You Can Skip Stage 2 When:**

* Latency must be < 100ms (real-time autocomplete)
* Top-100 recall is all that matters (no precision needed)
* Very simple queries (BM25 or bi-encoder sufficient)
* Candidates from Stage 1 are already very precise

The Two-Stage Pipeline
-----------------------

**Standard Configuration:**

.. code-block:: python

   # Stage 1: Fast retrieval (100-1000 candidates)
   bi_encoder = SentenceTransformer('BAAI/bge-base-en-v1.5')
   candidates = bi_encoder.search(query, corpus, top_k=100)
   
   # Stage 2: Precise re-ranking (top-10)
   cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
   reranked = cross_encoder.rerank(query, candidates, top_k=10)

**Cost Analysis:**

* Stage 1: 10M docs × 0.00001s = 0.1s (with FAISS)
* Stage 2: 100 docs × 0.05s = 5s (cross-encoder)
* **Total: ~5s** (vs ~140 hours if cross-encoder on full corpus!)

Next Steps
----------

* See :doc:`cross_encoders` for traditional BERT-based re-rankers
* See :doc:`llm_rerankers` for modern LLM-based approaches
* See :doc:`../stage1_retrieval/late_interaction` for ColBERT (can replace both stages)

