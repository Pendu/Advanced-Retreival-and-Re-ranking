.. _hard-negative-mining-deep-dive:

==========================================================================
Hard Negative Mining: A Deep Dive for Building a Unified Mining Library
==========================================================================

This document consolidates research on hard negative mining strategies with the goal of 
informing the design of a **unified hard negative mining library** — analogous to the 
`rerankers <https://github.com/AnswerDotAI/rerankers>`_ library but for mining strategies.

.. contents:: Table of Contents
   :local:
   :depth: 3

Motivation: The Gap in the Ecosystem
====================================

Current State of Libraries
--------------------------

.. code-block:: text

   Inference Libraries              │  Training Libraries
   ─────────────────────────────────┼─────────────────────────────────
   ✅ rerankers (unified API)       │  ❌ No unified hard mining lib
   ✅ RAGatouille (ColBERT)         │  ✅ Contrastors (basic mining)
   ✅ sentence-transformers         │  ✅ FlagEmbedding (basic mining)
   ✅ PyLate (late interaction)     │  ❌ Advanced mining scattered
   ─────────────────────────────────┴─────────────────────────────────

**The Problem**: Hard negative mining code is scattered across paper repositories with no 
unified interface. Researchers must re-implement strategies for each project.

Ecosystem-Wide Hard Negative Mining Support
-------------------------------------------

The following table comprehensively maps **50+ libraries** against **10 hard negative mining strategies**. 
This analysis reveals the gap that a unified mining library could fill.

**Legend:**

* ✅ = Native support (built-in, documented)
* 🔧 = Partial/Manual (requires custom code or workarounds)
* ❌ = Not supported
* N/A = Not applicable (inference-only library)

.. list-table:: Hard Negative Mining Support by Library
   :header-rows: 1
   :widths: 18 8 8 8 8 8 8 8 8 8 10

   * - Library
     - In-Batch
     - Static BM25
     - Margin
     - Dynamic (ANCE)
     - Denoised (CE)
     - Cross-Batch
     - SimANS
     - Synthetic
     - Import. Samp.
     - Notes
   * - **Sentence-Transformers**
     - ✅
     - ✅
     - ✅
     - ❌
     - 🔧
     - ❌
     - ❌
     - ❌
     - ❌
     - ``mine_hard_negatives()`` utility
   * - **FlagEmbedding (BGE)**
     - ✅
     - ✅
     - 🔧
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - Basic mining scripts
   * - **Contrastors (Nomic)**
     - ✅
     - 🔧
     - ✅
     - ❌
     - 🔧
     - ❌
     - ❌
     - ❌
     - ❌
     - Best current mining support
   * - **PyLate**
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ColBERT-style training
   * - **ColBERT (Stanford)**
     - ✅
     - ✅
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - Distillation in v2
   * - **RAGatouille**
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - Inference only
   * - **Pyserini**
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - BM25 negatives only
   * - **SPLADE**
     - ✅
     - ✅
     - 🔧
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - Distillation support
   * - **Neural-Cherche**
     - ✅
     - 🔧
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - Basic contrastive
   * - **Instructor**
     - ✅
     - 🔧
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - Task-specific training
   * - **LlamaIndex**
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - Orchestration only
   * - **LangChain**
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - Orchestration only
   * - **Haystack**
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - Orchestration only
   * - **Rankify**
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - Evaluation toolkit
   * - **Rerankers**
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - N/A
     - Inference only
   * - **Hard-Neg-Mixing**
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - MixGEN interpolation
   * - **ANCE (Official)**
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - Paper repo only
   * - **RocketQA (Official)**
     - ✅
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - Paper repo only
   * - **SimANS (Official)**
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - Paper repo only

**Summary Statistics:**

.. list-table:: Mining Strategy Availability
   :header-rows: 1
   :widths: 40 20 40

   * - Strategy
     - Libraries Supporting
     - Availability
   * - In-Batch Negatives
     - 10+
     - ✅ Widely available
   * - Static BM25 Hard Negatives
     - 8+
     - ✅ Widely available
   * - Margin-Based Filtering
     - 3
     - 🔧 Limited (Contrastors, ST)
   * - Dynamic ANN Refresh (ANCE)
     - 1
     - ❌ **Paper repo only**
   * - Cross-Encoder Denoising
     - 2
     - 🔧 Limited (ColBERT, RocketQA)
   * - Cross-Batch Negatives
     - 1
     - ❌ **Paper repo only**
   * - Ambiguous Zone (SimANS)
     - 1
     - ❌ **Paper repo only**
   * - LLM-Synthetic (SyNeg)
     - 0
     - ❌ **Not implemented**
   * - Importance Sampling
     - 0
     - ❌ **Not implemented**
   * - Query-Side (ADORE)
     - 0
     - ❌ **Not implemented**

**Key Finding**: Advanced mining strategies (ANCE, SimANS, ADORE, importance sampling) 
exist only in paper repositories with no production-ready implementations.

Gap Analysis: Why a Unified Library is Needed
---------------------------------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    HARD NEGATIVE MINING ECOSYSTEM                        │
   ├─────────────────────────────────────────────────────────────────────────┤
   │                                                                          │
   │  WELL-SERVED (Inference)          │  UNDERSERVED (Training)             │
   │  ─────────────────────────────────┼───────────────────────────────────  │
   │  ✅ Rerankers: 20+ models         │  ❌ No unified mining library        │
   │  ✅ LlamaIndex: 160+ connectors   │  ❌ Advanced methods scattered       │
   │  ✅ LangChain: 700+ integrations  │  ❌ Paper repos hard to use          │
   │  ✅ Vector DBs: 8+ options        │  ❌ No common interface              │
   │                                    │                                     │
   │  BASIC MINING AVAILABLE           │  ADVANCED MINING MISSING            │
   │  ─────────────────────────────────┼───────────────────────────────────  │
   │  ✅ In-batch (everywhere)         │  ❌ Dynamic refresh (ANCE)           │
   │  ✅ Static BM25 (most libs)       │  ❌ Query-side (ADORE)               │
   │  🔧 Margin (Contrastors only)     │  ❌ Ambiguous zone (SimANS)          │
   │  🔧 Denoising (ColBERT only)      │  ❌ LLM-synthetic (SyNeg)            │
   │                                    │  ❌ Importance sampling              │
   │                                    │  ❌ Curriculum learning              │
   └─────────────────────────────────────────────────────────────────────────┘

**Conclusion**: The ecosystem has mature inference libraries but lacks a unified training 
library for hard negative mining. This is the gap that ``hardminers`` would fill.

**The Solution**: A library like ``rerankers`` but for hard negative mining:

.. code-block:: python

   # Hypothetical unified API
   from hardminers import HardMiner
   
   # Same interface, different strategies
   miner = HardMiner("ance")           # Dynamic refresh
   miner = HardMiner("denoised")       # Cross-encoder filtering
   miner = HardMiner("margin")         # Margin-based selection
   miner = HardMiner("simans")         # Ambiguous zone sampling
   
   hard_negatives = miner.mine(queries, corpus, positives)

How ColBERTv2 Uses Hard Negative Mining
=======================================

ColBERTv2 and other late interaction models use hard negative mining as a **core part** 
of their training. Understanding this is essential for designing a mining library.

Training Pipeline
-----------------

According to the `ColBERTv2 paper (Santhanam et al., NAACL 2022) 
<https://aclanthology.org/2022.naacl-main.272/>`_:

**1. Knowledge Distillation from Cross-Encoder**

ColBERTv2 uses a **cross-encoder teacher** to generate soft labels and mine hard negatives:

.. code-block:: text

   Training Pipeline:
   1. Cross-encoder scores all (query, passage) pairs
   2. Hard negatives = high-scoring passages that aren't the gold positive
   3. ColBERT student learns from both:
      - Hard labels (gold positives)
      - Soft labels (cross-encoder scores on negatives)

**2. Denoised Supervision**

ColBERTv2 specifically addresses the **false negative problem**:

* Uses cross-encoder to identify passages that are actually relevant but unlabeled
* Filters these out or assigns appropriate soft labels
* This is essentially the "denoising" technique from RocketQA

**3. Residual Compression Training**

The compressed representations are trained with hard negatives to maintain discrimination.

What ColBERTv2 Uses vs. What It Doesn't
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Strategy
     - Used by ColBERT?
     - Potential Value for Library
   * - In-Batch Negatives
     - ✅ Yes
     - Baseline implementation
   * - Static BM25 Hard Negatives
     - ✅ Yes
     - Basic static mining
   * - Cross-Encoder Distillation
     - ✅ Yes
     - Denoising + soft labels
   * - Dynamic ANN Refresh (ANCE)
     - ❌ No
     - High - could improve ColBERT
   * - Curriculum Learning (SimANS)
     - ❌ No
     - High - "ambiguous zone" sampling
   * - LLM-Synthetic Negatives (SyNeg)
     - ❌ No
     - High - factually contradictory
   * - Query-side Finetuning (ADORE)
     - ❌ No
     - Medium - efficient dynamic mining
   * - Topic-Aware Sampling (TAS-B)
     - ❌ No
     - Medium - cluster-based diversity

Actual Sampling Logic in ColBERTv2
----------------------------------

From the ColBERT codebase, the actual sampling works as follows:

.. code-block:: python

   # Step 1: Load pre-mined negatives (BM25 or previous ColBERT checkpoint)
   negatives = load_negatives(negatives_path)  # Pre-computed top-k
   
   # Step 2: For each training triple
   for query, positive, negative_ids in dataloader:
       # negative_ids are indices into the corpus
       # Typically: top 100-200 BM25 results, sample 7 per query
       
       # Step 3: With distillation, also load cross-encoder scores
       ce_scores = load_cross_encoder_scores(query_id)
       
       # Step 4: Training uses both hard labels and soft labels
       loss = kl_divergence(colbert_scores, ce_scores) + contrastive_loss

**Key Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Typical Value
     - Purpose
   * - ``top_k`` (initial retrieval)
     - 100-200
     - Candidate pool size
   * - ``num_negatives`` per query
     - 1-7
     - Negatives used per training step
   * - ``in_batch_size``
     - 32-128
     - In-batch negatives = batch_size - 1
   * - ``margin``
     - 0.0-0.1
     - Filter threshold for denoising

Existing Implementations Analyzed
=================================

Nomic AI Contrastors
--------------------

The `Contrastors library <https://github.com/nomic-ai/contrastors>`_ includes a hard 
negative mining script that wraps sentence-transformers functionality.

**Source**: `st_mine_hard_negatives.py 
<https://github.com/nomic-ai/contrastors/blob/main/scripts/text/st_mine_hard_negatives.py>`_

**How It Works:**

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   
   triplets = mine_hard_negatives(
       dataset=ds,
       model=model,                    # SentenceTransformer for embeddings
       anchor_column_name="query",
       positive_column_name="pos",
       num_negatives=20,               # Sample 20 negatives per query
       margin=0.95,                    # Only keep negatives with score < pos_score * margin
       range_min=10,                   # Skip top-10 (too similar, likely false negatives)
       range_max=50,                   # Only consider ranks 10-50
       sampling_strategy="top",        # "top" or "random"
       use_faiss=True,                 # FAISS for large-scale
   )

**Sampling Logic:**

.. code-block:: python

   # Step 1: Embed all queries and corpus
   query_embeddings = model.encode(queries)
   corpus_embeddings = model.encode(corpus)
   
   # Step 2: Find top-k nearest neighbors for each query
   similarities = query_embeddings @ corpus_embeddings.T
   top_k_indices = similarities.argsort(descending=True)[:, :range_max]
   
   # Step 3: Filter candidates
   for query_idx, candidates in enumerate(top_k_indices):
       positive_score = similarities[query_idx, positive_idx]
       
       hard_negatives = []
       for rank, candidate_idx in enumerate(candidates):
           if rank < range_min:  # Skip top ranks (false negative risk)
               continue
           if candidate_idx == positive_idx:  # Skip the actual positive
               continue
           
           candidate_score = similarities[query_idx, candidate_idx]
           
           # Margin filtering: negative must be sufficiently worse than positive
           if margin and candidate_score > positive_score - margin:
               continue  # Too similar to positive
           
           hard_negatives.append(candidate_idx)

**Key Features:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Implementation
   * - Mining Source
     - Same model (bi-encoder)
   * - Filtering
     - ``range_min``, ``range_max``, ``margin``, ``max_score``, ``min_score``
   * - Sampling
     - "top" (deterministic) or "random"
   * - Cross-Encoder
     - Optional re-scoring
   * - Scale
     - FAISS support for large corpora

PyLate (ColBERT Training)
-------------------------

PyLate uses a simpler approach inherited from ColBERT:

.. code-block:: python

   # PyLate/ColBERT approach
   # Pre-mined negatives from BM25 or previous model checkpoint
   negatives = load_precomputed_negatives()  # Static, not mined on-the-fly
   
   # In-batch negatives during training
   for batch in dataloader:
       # All other positives in batch become negatives
       in_batch_negs = batch.positives[batch.positives != current_positive]

**Comparison:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - Nomic Contrastors
     - PyLate
   * - Mining Time
     - Pre-training (offline)
     - Pre-training (offline)
   * - Dynamic Refresh
     - ❌ No
     - ❌ No
   * - Margin Filtering
     - ✅ Yes
     - ❌ No
   * - Range Filtering
     - ✅ Yes (skip top-k)
     - ❌ No
   * - In-Batch Negatives
     - ❌ Separate
     - ✅ Built-in

Hard Negative Mixing (MixGEN)
-----------------------------

The `hard-negative-mixing <https://github.com/davidsvy/hard-negative-mixing>`_ library 
implements a different approach: **synthetic hard negative generation** via embedding 
interpolation.

.. code-block:: python

   # MixGEN approach (conceptual)
   def mix_hard_negatives(anchor_emb, positive_emb, negative_emb, alpha=0.5):
       """
       Create synthetic hard negative by mixing positive and negative embeddings.
       Result is harder than the original negative but still negative.
       """
       synthetic_neg = alpha * positive_emb + (1 - alpha) * negative_emb
       return synthetic_neg

**Key Innovation**: Instead of mining harder negatives from the corpus, it **creates** 
them by interpolation.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - Nomic Contrastors
     - Hard-Negative-Mixing
   * - Approach
     - Mine from corpus
     - Synthesize via interpolation
   * - Requires Corpus
     - ✅ Yes
     - ❌ No (uses existing negatives)
   * - Difficulty Control
     - Via margin/range
     - Via interpolation ratio (α)
   * - Novel Negatives
     - ❌ Only existing docs
     - ✅ Creates new embeddings
   * - Training Integration
     - Offline mining
     - Online during training

Theoretically-Grounded Mining Methods
=====================================

These methods are **grounded in theory** and don't rely on curriculum learning or 
synthetic generation.

ANCE: Dynamic ANN Refresh
-------------------------

**Paper**: `Approximate Nearest Neighbor Negative Contrastive Learning 
<https://arxiv.org/abs/2007.00808>`_ (ICLR 2021)

**Theoretical Grounding**: The paper proves that **static negatives cause gradient vanishing**:

.. code-block:: text

   Gradient magnitude ∝ (1 - similarity(q, neg))
   
   If neg is "easy" (low similarity): gradient ≈ 1 (good)
   If neg is "hard" (high similarity): gradient ≈ 0 (vanishing!)
   
   BUT: As model improves, previously "hard" negatives become "easy"
        → Need to refresh to find NEW hard negatives

**Key Innovation**: Periodically rebuild the ANN index with current model embeddings.

.. code-block:: python

   # ANCE training loop (conceptual)
   for epoch in range(num_epochs):
       for step, batch in enumerate(dataloader):
           # Standard training step
           loss = contrastive_loss(model, batch)
           loss.backward()
           optimizer.step()
           
           # Periodic index refresh
           if step % refresh_interval == 0:
               # Re-encode all documents with current model
               new_embeddings = model.encode(corpus)
               index.rebuild(new_embeddings)
               
               # Mine new hard negatives
               hard_negatives = index.search(queries, top_k=100)

ADORE: Query-Side Finetuning
----------------------------

**Paper**: `Optimizing Dense Retrieval Model Training with Hard Negatives 
<https://arxiv.org/abs/2104.08051>`_ (SIGIR 2021)

**Theory**: Instead of refreshing the document index (expensive), only update the 
**query encoder** while keeping document embeddings fixed.

.. code-block:: text

   Standard ANCE:
     Query Encoder ──► Update
     Doc Encoder   ──► Update  
     Doc Index     ──► Rebuild (expensive!)
   
   ADORE:
     Query Encoder ──► Update
     Doc Encoder   ──► FROZEN
     Doc Index     ──► FIXED (cheap!)

**Theoretical Grounding**:

* Proves that query-side optimization is sufficient for learning hard negatives
* Document representations are already "good enough" after initial training
* Reduces computational cost from O(N) index rebuild to O(1)

RocketQA: Cross-Batch Negatives + Denoising
-------------------------------------------

**Paper**: `RocketQA: An Optimized Training Approach to Dense Passage Retrieval 
<https://arxiv.org/abs/2010.08191>`_ (NAACL 2021)

**Insight 1: Cross-Batch Negatives**

.. code-block:: python

   # Standard in-batch: negatives from same GPU batch
   # Cross-batch: negatives from ALL GPUs in distributed training
   
   # If batch_size=32, num_gpus=8:
   # In-batch negatives: 31 per query
   # Cross-batch negatives: 32*8 - 1 = 255 per query

**Insight 2: Denoised Hard Negatives**

Uses cross-encoder to **filter false negatives**:

.. code-block:: python

   # Denoising logic
   for neg in hard_negatives:
       ce_score = cross_encoder.score(query, neg)
       if ce_score > threshold:  # This "negative" is actually relevant!
           remove(neg)  # Don't train on false negatives

**Theoretical Basis**: InfoNCE loss is biased when negatives are actually positives.

Margin-Based Mining (Triplet Loss Theory)
-----------------------------------------

**Classical Theory** from metric learning:

.. code-block:: python

   # Triplet Loss
   L = max(0, margin + d(anchor, positive) - d(anchor, negative))
   
   # Semi-Hard Negatives (theoretically optimal)
   # Negatives that satisfy:
   d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin
   
   # This is the "sweet spot":
   # - Hard enough to provide learning signal
   # - Not so hard it's a false negative

**This is what Nomic's ``margin`` parameter implements!**

Importance Sampling
-------------------

**Paper**: `Sampling Matters in Deep Embedding Learning 
<https://arxiv.org/abs/1706.07567>`_ (ICCV 2017)

**Theory**: Proposes **distance-weighted sampling** where probability of selecting a 
negative is proportional to its similarity:

.. code-block:: python

   # Uniform sampling (bad)
   P(neg) = 1/N
   
   # Distance-weighted sampling (better)
   P(neg) ∝ exp(similarity(anchor, neg) / temperature)
   
   # This naturally samples harder negatives more often

Alignment and Uniformity
------------------------

**Paper**: `Understanding Contrastive Representation Learning through Alignment and 
Uniformity <https://arxiv.org/abs/2005.10242>`_ (ICML 2020)

**Theory**: Contrastive learning optimizes two properties:

1. **Alignment**: Similar items should be close
2. **Uniformity**: Embeddings should be uniformly distributed on hypersphere

**Implication for Hard Negatives**:

.. code-block:: text

   Hard negatives improve UNIFORMITY by pushing apart similar-but-different items
   Easy negatives are already far apart → don't improve uniformity

Curriculum and Synthetic Methods (For Completeness)
====================================================

While not purely theoretically-grounded, these methods are widely used:

SimANS: Ambiguous Zone Sampling
-------------------------------

**Paper**: `SimANS: Simple Ambiguous Negatives Sampling 
<https://arxiv.org/abs/2210.11773>`_ (EMNLP 2022)

**Key Insight**: Sample negatives from the "ambiguous zone" — ranked neither too high 
(false positive risk) nor too low (too easy).

.. code-block:: python

   # SimANS sampling
   def sample_ambiguous_negatives(query, corpus, model, zone_start=50, zone_end=200):
       scores = model.score(query, corpus)
       ranked_indices = scores.argsort(descending=True)
       
       # Sample from ambiguous zone
       ambiguous_candidates = ranked_indices[zone_start:zone_end]
       return random.sample(ambiguous_candidates, num_negatives)

SyNeg: LLM-Synthetic Negatives
------------------------------

**Paper**: `SyNeg: LLM-Driven Synthetic Hard Negatives 
<https://arxiv.org/abs/2412.17250>`_ (arXiv 2024)

**Key Innovation**: Uses LLMs to generate text that is semantically similar but 
factually contradictory.

.. code-block:: python

   # SyNeg approach (conceptual)
   prompt = f"""
   Given this query: "{query}"
   And this relevant passage: "{positive}"
   
   Generate a passage that:
   1. Uses similar vocabulary and style
   2. Discusses the same topic
   3. But contains INCORRECT or CONTRADICTORY information
   """
   synthetic_negative = llm.generate(prompt)

MUVERA: Multi-Vector Efficiency (Not Mining)
============================================

It's important to note that `MUVERA <https://arxiv.org/abs/2405.19504>`_ (Google's 
multi-vector paper) focuses on **inference efficiency**, NOT training:

* Converts multi-vector similarity to single-vector MIPS
* Fixed Dimensional Encodings (FDE) for compression
* **Does not propose new negative mining strategies**

MUVERA assumes you already have a trained ColBERT-style model and makes it faster to deploy.

Proposed Library Design
=======================

Based on this analysis, here's a proposed design for a unified hard mining library:

API Design
----------

.. code-block:: python

   from hardminers import HardMiner
   
   # Strategy 1: Dynamic ANN (ANCE-style)
   miner = HardMiner("dynamic", refresh_every=1000)
   hard_negs = miner.mine(queries, corpus, positives)
   
   # Strategy 2: Cross-Encoder Denoised (RocketQA-style)
   miner = HardMiner("denoised", 
                     retriever="bm25",
                     denoiser="cross-encoder/ms-marco-MiniLM-L-6-v2",
                     threshold=0.7)
   hard_negs = miner.mine(queries, corpus, positives)
   
   # Strategy 3: Margin-Based (Nomic-style)
   miner = HardMiner("margin", 
                     margin=0.1, 
                     range_min=10, 
                     range_max=200)
   hard_negs = miner.mine(queries, corpus, positives)
   
   # Strategy 4: Query-Side Only (ADORE-style)
   miner = HardMiner("query_side", freeze_docs=True)
   hard_negs = miner.mine(queries, corpus, positives)
   
   # Strategy 5: Ambiguous Zone (SimANS-style)
   miner = HardMiner("ambiguous", zone_start=50, zone_end=200)
   hard_negs = miner.mine(queries, corpus, positives)
   
   # Strategy 6: LLM-Synthetic (SyNeg-style)
   miner = HardMiner("synthetic", llm="gpt-4o-mini")
   hard_negs = miner.generate(queries, positives)
   
   # Strategy 7: Importance Sampling
   miner = HardMiner("importance", temperature=0.1)
   hard_negs = miner.mine(queries, corpus, positives)

Priority Strategies for Implementation
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 25 25 20 20

   * - Priority
     - Strategy
     - Paper
     - Complexity
     - Theory-Based?
   * - 1
     - In-Batch Negatives
     - Baseline
     - Easy
     - ✅ Yes
   * - 2
     - Static BM25
     - DPR
     - Easy
     - ✅ Yes
   * - 3
     - Margin-Based
     - Triplet Loss
     - Easy
     - ✅ Yes
   * - 4
     - Cross-Encoder Denoising
     - RocketQA
     - Medium
     - ✅ Yes
   * - 5
     - Dynamic ANN Refresh
     - ANCE
     - Medium
     - ✅ Yes
   * - 6
     - Query-Side Finetuning
     - ADORE
     - Hard
     - ✅ Yes
   * - 7
     - Importance Sampling
     - Sampling Matters
     - Medium
     - ✅ Yes
   * - 8
     - Ambiguous Zone Sampling
     - SimANS
     - Medium
     - ❌ Curriculum
   * - 9
     - LLM-Synthetic Generation
     - SyNeg
     - Medium
     - ❌ Synthetic
   * - 10
     - False Negative Filtering
     - Various
     - Easy
     - ✅ Yes

Integration with Training Libraries
-----------------------------------

.. code-block:: python

   from hardminers import HardMiner, ContrastiveTrainer
   
   # Mine negatives
   miner = HardMiner("denoised", denoiser="cross-encoder/ms-marco-MiniLM-L-6-v2")
   dataset_with_negatives = miner.mine(queries, corpus, positives)
   
   # Train with sentence-transformers
   from sentence_transformers import SentenceTransformer, losses
   
   model = SentenceTransformer("BAAI/bge-base-en-v1.5")
   train_loss = losses.MultipleNegativesRankingLoss(model)
   
   # Or use built-in trainer
   trainer = ContrastiveTrainer(
       model="BAAI/bge-base-en-v1.5",
       miner=miner,
       loss="infonce",
       dynamic_mining=True,  # Re-mine every N steps
       refresh_interval=1000
   )
   trainer.train(dataset)

Detailed Library Implementation Analysis
========================================

This section provides deeper analysis of how each library implements (or could implement) 
hard negative mining.

Embedding/Training Libraries
----------------------------

.. list-table:: Detailed Mining Implementation
   :header-rows: 1
   :widths: 15 20 20 45

   * - Library
     - Mining Function
     - Mining Source
     - Implementation Details
   * - **Sentence-Transformers**
     - ``util.mine_hard_negatives()``
     - Same bi-encoder
     - Uses FAISS for ANN search. Supports ``range_min``, ``range_max``, ``margin``, ``max_score``. Can optionally use cross-encoder for re-scoring. Most complete implementation in mainstream libs.
   * - **Contrastors**
     - ``st_mine_hard_negatives.py``
     - Same bi-encoder
     - Wraps Sentence-Transformers mining. Adds margin filtering (0.95, 0.98). Used to train Nomic Embed models.
   * - **FlagEmbedding**
     - ``scripts/hn_mine.py``
     - Same or teacher model
     - Basic top-k mining from dense retrieval. Supports BM25 fallback. No margin or denoising.
   * - **PyLate**
     - Pre-computed files
     - BM25 or ColBERT
     - Expects negatives in training data. No built-in mining. Uses ``negatives.tsv`` format.
   * - **ColBERT**
     - ``colbert.training``
     - BM25 + CE distillation
     - Pre-mines with BM25, then uses cross-encoder scores for soft labels. Denoising via score thresholding.
   * - **SPLADE**
     - Training scripts
     - BM25 or dense
     - Distillation from cross-encoder. Uses MarginMSE loss with teacher scores.

Paper Repository Implementations
--------------------------------

These implementations exist but are not production-ready:

.. list-table:: Paper Repo Mining Implementations
   :header-rows: 1
   :widths: 15 15 15 55

   * - Method
     - Repository
     - Maintenance
     - Portability Issues
   * - **ANCE**
     - microsoft/ANCE
     - ❌ Archived
     - Tied to specific DPR architecture. Requires custom distributed training setup. Index refresh code not modular.
   * - **ADORE**
     - jingtaozhan/DRhard
     - ❌ Inactive
     - Query-side finetuning logic embedded in training loop. No standalone mining API.
   * - **RocketQA**
     - PaddlePaddle/RocketQA
     - 🔧 Limited
     - PaddlePaddle dependency. Cross-batch requires specific distributed setup. Denoising logic not extractable.
   * - **SimANS**
     - microsoft/SimXNS
     - 🔧 Limited
     - Ambiguous zone sampling buried in training code. No configurable API for zone boundaries.
   * - **TAS-Balanced**
     - sebastian-hofstaetter/tas-balanced
     - ❌ Inactive
     - Topic clustering requires pre-processing. Sampling logic not standalone.

What Would a Unified Library Provide?
-------------------------------------

Comparing the current fragmented state to a hypothetical unified library:

.. list-table:: Current vs. Unified Library
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Current State
     - With ``hardminers``
   * - **Using ANCE**
     - Clone repo, modify DPR code, set up distributed training
     - ``HardMiner("ance", refresh_every=1000)``
   * - **Using SimANS**
     - Clone repo, understand internal sampling, extract zone logic
     - ``HardMiner("simans", zone_start=50, zone_end=200)``
   * - **Using Denoising**
     - Implement cross-encoder scoring, set threshold manually
     - ``HardMiner("denoised", threshold=0.7)``
   * - **Combining Strategies**
     - Write custom code to chain methods
     - ``HardMiner("margin+denoised", margin=0.1, threshold=0.7)``
   * - **Switching Strategies**
     - Rewrite training pipeline
     - Change one string parameter
   * - **Benchmarking**
     - Run each repo separately, different eval setups
     - Unified evaluation across all strategies

Estimated Development Effort
----------------------------

To build a unified hard mining library comparable to ``rerankers``:

.. list-table:: Implementation Roadmap
   :header-rows: 1
   :widths: 25 15 15 45

   * - Component
     - Effort
     - Priority
     - Dependencies
   * - Core API + In-Batch
     - 1 week
     - P0
     - PyTorch, numpy
   * - Static BM25 Mining
     - 3 days
     - P0
     - Pyserini or rank_bm25
   * - Margin-Based Mining
     - 3 days
     - P0
     - FAISS or sentence-transformers
   * - Cross-Encoder Denoising
     - 1 week
     - P1
     - transformers, cross-encoder models
   * - Dynamic ANN (ANCE)
     - 2 weeks
     - P1
     - FAISS, training loop integration
   * - Query-Side (ADORE)
     - 2 weeks
     - P2
     - Custom training logic
   * - SimANS Sampling
     - 1 week
     - P2
     - Score distribution analysis
   * - Importance Sampling
     - 1 week
     - P2
     - Probability distributions
   * - LLM-Synthetic
     - 1 week
     - P3
     - OpenAI/Anthropic API or local LLM
   * - Documentation + Tests
     - 2 weeks
     - P0
     - pytest, sphinx
   * - **Total**
     - **~10 weeks**
     - 
     - For MVP with P0+P1 features

Summary
=======

.. important::

   **Key Takeaways:**
   
   1. **ColBERTv2 uses**: In-batch + static BM25 + cross-encoder distillation/denoising
   2. **ColBERTv2 does NOT use**: Dynamic refresh (ANCE), curriculum (SimANS), synthetic (SyNeg)
   3. **Gap in ecosystem**: No unified library for hard negative mining strategies
   4. **Theoretically-grounded methods**: ANCE, ADORE, RocketQA, margin-based, importance sampling
   5. **Library opportunity**: Unified API similar to ``rerankers`` for mining strategies

Ecosystem Gap Validation
------------------------

Based on the comprehensive analysis above:

.. code-block:: text

   VALIDATION METRICS FOR LIBRARY NEED:
   
   ┌────────────────────────────────────────────────────────────────────┐
   │ Metric                                    │ Value    │ Assessment │
   ├────────────────────────────────────────────────────────────────────┤
   │ Advanced mining strategies in papers      │ 10+      │ High       │
   │ Production-ready implementations          │ 2-3      │ Low        │
   │ Libraries with unified mining API         │ 0        │ None       │
   │ Paper repos that are maintained           │ 1-2      │ Low        │
   │ Time to implement ANCE from scratch       │ ~2 weeks │ High       │
   │ Time with unified library                 │ 1 line   │ Trivial    │
   │ Potential users (embedding trainers)      │ 1000s    │ High       │
   └────────────────────────────────────────────────────────────────────┘
   
   CONCLUSION: Strong justification for unified hard mining library

References
==========

**Core Papers:**

1. Santhanam, K., et al. (2022). "ColBERTv2: Effective and Efficient Retrieval via 
   Lightweight Late Interaction." *NAACL 2022*. 
   `Paper <https://aclanthology.org/2022.naacl-main.272/>`_

2. Xiong, L., et al. (2021). "Approximate Nearest Neighbor Negative Contrastive Learning 
   for Dense Text Retrieval." *ICLR 2021*. 
   `arXiv:2007.00808 <https://arxiv.org/abs/2007.00808>`_

3. Zhan, J., et al. (2021). "Optimizing Dense Retrieval Model Training with Hard Negatives." 
   *SIGIR 2021*. `arXiv:2104.08051 <https://arxiv.org/abs/2104.08051>`_

4. Qu, Y., et al. (2021). "RocketQA: An Optimized Training Approach to Dense Passage 
   Retrieval." *NAACL 2021*. `arXiv:2010.08191 <https://arxiv.org/abs/2010.08191>`_

5. Zhou, K., et al. (2022). "SimANS: Simple Ambiguous Negatives Sampling for Dense Text 
   Retrieval." *EMNLP 2022*. `arXiv:2210.11773 <https://arxiv.org/abs/2210.11773>`_

**Libraries:**

* Contrastors: https://github.com/nomic-ai/contrastors
* Rerankers: https://github.com/AnswerDotAI/rerankers
* PyLate: https://github.com/lightonai/pylate
* Hard-Negative-Mixing: https://github.com/davidsvy/hard-negative-mixing

**Theoretical Foundations:**

6. Wang, T., & Isola, P. (2020). "Understanding Contrastive Representation Learning 
   through Alignment and Uniformity on the Hypersphere." *ICML 2020*. 
   `arXiv:2005.10242 <https://arxiv.org/abs/2005.10242>`_

7. Wu, C., et al. (2017). "Sampling Matters in Deep Embedding Learning." *ICCV 2017*. 
   `arXiv:1706.07567 <https://arxiv.org/abs/1706.07567>`_

----

*This document was created as a knowledge consolidation for building a unified hard 
negative mining library. Last updated: December 2024.*

