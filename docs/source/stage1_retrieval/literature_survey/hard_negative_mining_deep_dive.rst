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

The Problem Statement
---------------------

Hard negative mining is a critical component of training dense retrieval models, yet:

1. **No unified library exists** for hard negative mining (unlike inference where ``rerankers`` unifies 20+ models)
2. **Advanced strategies are scattered** across paper repositories with incompatible interfaces
3. **Production implementations are limited** to basic methods (in-batch, static BM25)

This document analyzes the ecosystem to validate the need for a unified mining library.

Formal Definition: What is a Hard Negative?
-------------------------------------------

Before analyzing implementations, we establish precise definitions:

**Definition (Negative Sample)**: Given a query \(q\) and its relevant document \(d^+\), a negative 
sample \(d^-\) is any document where \(\text{rel}(q, d^-) = 0\).

**Definition (Hard Negative)**: A hard negative \(d^-_{\text{hard}}\) satisfies:

.. math::

   \text{sim}(q, d^-_{\text{hard}}) > \tau \quad \text{where} \quad \text{rel}(q, d^-_{\text{hard}}) = 0

where \(\tau\) is a similarity threshold (typically top-k retrieval rank or score cutoff).

**Definition (False Negative)**: A sample labeled as negative but actually relevant:

.. math::

   d^-_{\text{false}}: \quad \text{label}(q, d^-_{\text{false}}) = 0 \quad \text{but} \quad \text{rel}(q, d^-_{\text{false}}) = 1

**The Goldilocks Principle**: Optimal negatives are "just right" — hard enough to provide 
learning signal, but not so hard they're likely false negatives.

**The Critical Tension**: The hardest negatives (highest similarity to query) are also 
the most likely to be **false negatives** (actually relevant but unlabeled). This tension 
is the central challenge that advanced mining strategies address:

.. code-block:: text

   Similarity Score to Query
   0.0                    0.5                    0.8        1.0
   ├──────────────────────┼──────────────────────┼──────────┤
   │                      │                      │          │
   │   EASY NEGATIVES     │   HARD NEGATIVES     │  FALSE   │
   │                      │   (Goldilocks Zone)  │NEGATIVES │
   │                      │                      │          │
   │  Score ≈ 0.0-0.3     │  Score ≈ 0.3-0.7     │ Score >  │
   │  Gradient ≈ 0        │  Gradient is HIGH    │ 0.7-0.8  │
   │  (no learning)       │  (optimal learning)  │ (wrong   │
   │                      │                      │ gradient)│
   └──────────────────────┴──────────────────────┴──────────┘
   
   Key: The "Goldilocks Zone" provides high gradient (learning signal) 
        while minimizing false negative risk.

Ecosystem-Wide Hard Negative Mining Support
-------------------------------------------

The following table maps **19 training-relevant libraries** against **10 hard negative mining strategies**. 
This analysis reveals the gap that a unified mining library could fill.

**Legend:**

* ✅ = Native support (built-in, documented)
* 🔧 = Partial/Manual (requires custom code or workarounds)
* ❌ = Not supported
* N/A = Not applicable (inference-only library)

.. list-table:: Hard Negative Mining Support by Library (Training Time)
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
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ``CachedMNRL`` + CE filtering
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
     - 🔧
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - Score filtering in v2 distillation
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

How ColBERTv2 Uses Hard Negative Mining
=======================================

ColBERTv2 and other late interaction models use hard negative mining as a **core part** 
of their training. Understanding this is essential for designing a mining library.

Training Pipeline
-----------------

According to the `ColBERTv2 paper (Santhanam et al., NAACL 2022) 
<https://aclanthology.org/2022.naacl-main.272/>`_:

.. important::

   ColBERTv2 uses a **two-stage training approach**, not simultaneous distillation:
   
   1. **Stage 1**: Train ColBERT with standard BM25 hard negatives + in-batch negatives
   2. **Stage 2**: Apply cross-encoder distillation for denoising in a second training phase
   
   The paper states: *"we start with a ColBERT model trained with triples as in Khattab 
   and Zaharia (2020)"* — the distillation happens after initial training.

**Stage 1: Initial Training with Hard Negatives**

.. code-block:: text

   Initial Training:
   1. Mine hard negatives using BM25 (top-k passages that don't contain answer)
   2. Combine with in-batch negatives during training
   3. Train ColBERT with standard contrastive loss

**Stage 2: Cross-Encoder Distillation (Denoising)**

.. code-block:: text

   Distillation Pipeline:
   1. Cross-encoder teacher scores all (query, passage) pairs
   2. High-scoring "negatives" are identified as potential false negatives
   3. ColBERT student learns from:
      - Hard labels (gold positives)
      - Soft labels (cross-encoder scores) — this denoises false negatives

**Why Two Stages?** The initial BM25-trained model provides a reasonable starting point. 
Cross-encoder distillation then refines the model by addressing false negatives that 
BM25 mining inevitably introduces.

**Residual Compression Training**

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

This section analyzes existing implementations, separated into **production-ready libraries** 
(maintained, documented, pip-installable) and **research implementations** (paper repos, 
often unmaintained).

Production-Ready Libraries
--------------------------

Contrastors (Nomic AI)
^^^^^^^^^^^^^^^^^^^^^^

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
       margin=0.05,                    # Negative score must be < positive_score - margin
       range_min=10,                   # Skip top-10 (too similar, likely false negatives)
       range_max=50,                   # Only consider ranks 10-50
       sampling_strategy="top",        # "top" or "random"
       use_faiss=True,                 # FAISS for large-scale
   )

.. note::

   The ``margin`` parameter in Sentence-Transformers is **additive**: a negative is kept 
   only if ``score(q, neg) < score(q, pos) - margin``. Contrastors uses multiplicative 
   margins (e.g., 0.95 means ``score(q, neg) < 0.95 * score(q, pos)``). Both achieve 
   similar filtering but with different parameterizations.

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

PyLate
^^^^^^

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

Research Implementations (Paper Repos)
--------------------------------------

These implementations exist in paper repositories but are **not production-ready**: 
often unmaintained, tied to specific architectures, or lacking documentation.

Hard Negative Mixing (MixGEN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

These methods are **grounded in mathematical theory** with formal guarantees or principled 
derivations. We present them with their theoretical foundations.

Why Hard Negatives Matter: The InfoNCE Perspective
--------------------------------------------------

The theoretical foundation for hard negative mining comes from the **InfoNCE loss** 
(Oord et al., 2018), which most dense retrieval models optimize:

.. math::

   \mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{i=1}^{K} \exp(\text{sim}(q, d^-_i) / \tau)}

**Gradient Analysis** (key insight for understanding mining strategies):

The gradient with respect to a negative sample \(d^-_i\) is:

.. math::

   \frac{\partial \mathcal{L}}{\partial \text{sim}(q, d^-_i)} = \frac{\exp(\text{sim}(q, d^-_i) / \tau)}{\sum_j \exp(\text{sim}(q, d^-_j) / \tau)} = p(d^-_i | q)

**Key Insight**: The gradient is proportional to the softmax probability of the negative. 
This means:

* **Easy negatives** (low similarity): Small gradient → minimal learning signal
* **Hard negatives** (high similarity): Large gradient → strong learning signal
* **False negatives** (actually relevant): Large gradient in WRONG direction → damages model

This explains why hard negative mining is critical: easy negatives waste compute, while 
hard negatives provide the signal needed for learning fine-grained distinctions.

ANCE: Dynamic ANN Refresh
-------------------------

**Paper**: `Approximate Nearest Neighbor Negative Contrastive Learning 
<https://arxiv.org/abs/2007.00808>`_ (ICLR 2021)

**Theoretical Grounding**: ANCE addresses the **staleness problem** of static negatives:

.. code-block:: text

   Training Dynamics:
   
   Step 0:    Model M₀, Negatives N₀ (hard for M₀)
   Step 1000: Model M₁ has improved → N₀ now "easy" for M₁
   Step 2000: Model M₂ even better → N₀ provides almost no gradient
   
   Problem: Static negatives become uninformative as model improves
   Solution: Periodically refresh negatives using current model

**Formal Claim** (Theorem 1 in paper): Under certain conditions, ANCE converges to a 
stationary point of the true contrastive objective, while static negative sampling may not.

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

ADORE: Query-Encoder Finetuning
--------------------------------

**Paper**: `Optimizing Dense Retrieval Model Training with Hard Negatives 
<https://arxiv.org/abs/2104.08051>`_ (SIGIR 2021)

**Theory**: Instead of refreshing the document index (expensive), only finetune the 
**query encoder** while keeping the document encoder and its embeddings fixed.

.. note::

   "Query-side finetuning" means finetuning the **query encoder during training**, 
   NOT query-time optimization. The document encoder remains frozen after initial training.

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

**Why Freezing Helps Beyond Efficiency**:

Freezing the document encoder provides a **stability benefit**, not just computational savings:

.. code-block:: text

   ANCE Problem: "Chasing a Moving Target"
   ───────────────────────────────────────
   Step 1000: Query learns to find hard negatives in Doc Space v1
   Step 2000: Doc Space changes to v2 → previous learning partially invalidated
   Step 3000: Query adapts to v2, but Doc Space is now v3...
   
   ADORE Solution: "Fixed Target"
   ──────────────────────────────
   Doc Space is FIXED after initial training
   Query encoder learns to navigate this stable space
   → Allows harder negatives without instability
   → More aggressive mining is safe

This stability allows ADORE to use **harder negatives** than standard ANCE without 
the risk of training divergence.

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

**Theoretical Justification**: More negatives → better approximation of the full softmax 
partition function → lower variance gradient estimates.

**Insight 2: Denoised Hard Negatives**

Uses cross-encoder to **filter false negatives**:

.. code-block:: python

   # Denoising logic
   for neg in hard_negatives:
       ce_score = cross_encoder.score(query, neg)
       if ce_score > threshold:  # This "negative" is actually relevant!
           remove(neg)  # Don't train on false negatives

**Theoretical Basis**: False negatives cause the gradient to push apart actually-relevant 
pairs, directly contradicting the training objective:

.. math::

   \text{If } d^-_{\text{false}} \text{ is relevant: } \frac{\partial \mathcal{L}}{\partial d^-_{\text{false}}} \text{ pushes } d^-_{\text{false}} \text{ away from } q

This corrupts the embedding space by creating "holes" where relevant documents should be close.

**Empirical Finding**: RocketQA found ~30% of top-retrieved "negatives" were actually 
relevant on MS MARCO, and filtering these improved MRR@10 by 2-3 points.

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

.. math::

   P(d^-_i) = \frac{\exp(\text{sim}(q, d^-_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(q, d^-_j) / \tau)}

.. code-block:: python

   # Uniform sampling (uninformative)
   P(neg) = 1/N
   
   # Distance-weighted sampling (informative)
   similarities = model.score(query, all_negatives)
   weights = softmax(similarities / temperature)
   sampled_neg = np.random.choice(all_negatives, p=weights)

**Theoretical Justification**: This sampling distribution matches the gradient weighting 
in InfoNCE, meaning we sample negatives proportionally to their contribution to the loss.

**Temperature Effect**:

* \(\tau \to 0\): Always sample hardest negative (greedy)
* \(\tau \to \infty\): Uniform sampling (random)
* \(\tau \approx 0.1\): Balance between hard and diverse

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

TAS-Balanced: Topic-Aware Sampling
-----------------------------------

**Paper**: `Balanced Topic-Aware Sampling for Effective Dense Retriever Training 
<https://arxiv.org/abs/2104.06967>`_ (SIGIR 2021)

**Theory**: Cluster queries by topic, then sample negatives that are topically related 
but not relevant. This ensures negatives are semantically challenging.

.. code-block:: python

   # TAS-Balanced approach
   # 1. Cluster queries by topic (using query embeddings)
   clusters = kmeans(query_embeddings, n_clusters=100)
   
   # 2. For each query, sample negatives from same topic cluster
   for query in queries:
       topic = clusters.predict(query)
       topic_negatives = get_negatives_from_cluster(topic)
       # These are topically related but not relevant

**Theoretical Justification**: Topic-aware sampling ensures negatives share semantic 
features with the query, forcing the model to learn fine-grained distinctions rather 
than coarse topic classification.

Principled Sampling Methods
===========================

These methods use principled (though not purely theoretical) approaches to negative selection.

SimANS: Ambiguous Zone Sampling
-------------------------------

**Paper**: `SimANS: Simple Ambiguous Negatives Sampling 
<https://arxiv.org/abs/2210.11773>`_ (EMNLP 2022)

**Key Insight**: The optimal negatives lie in an "ambiguous zone" — ranked high enough 
to be informative, but not so high they're likely false negatives.

**Probabilistic Formulation**:

.. math::

   P(\text{false negative} | \text{rank} = r) \approx \begin{cases}
   \text{high} & \text{if } r < 50 \\
   \text{medium} & \text{if } 50 \leq r < 200 \\
   \text{low} & \text{if } r \geq 200
   \end{cases}

.. code-block:: python

   # SimANS sampling
   def sample_ambiguous_negatives(query, corpus, model, zone_start=50, zone_end=200):
       scores = model.score(query, corpus)
       ranked_indices = scores.argsort(descending=True)
       
       # Sample from ambiguous zone (the "Goldilocks" region)
       ambiguous_candidates = ranked_indices[zone_start:zone_end]
       return random.sample(ambiguous_candidates, num_negatives)

**Why This Works**: By avoiding top-ranked candidates (high false negative risk) and 
bottom-ranked candidates (too easy), SimANS balances informativeness with label reliability.

Synthetic Generation Methods
============================

.. warning::

   **Experimental / High-Compute Methods**
   
   The methods in this section require significant computational resources (LLM inference) 
   and are not yet production-ready. They represent cutting-edge research directions.

SyNeg: LLM-Synthetic Negatives
------------------------------

**Paper**: `SyNeg: LLM-Driven Synthetic Hard Negatives 
<https://arxiv.org/abs/2412.17250>`_ (arXiv 2024)

**Status**: 🧪 **Experimental** — Very recent research (2024), high compute cost.

**Key Innovation**: Uses LLMs to generate text that is semantically similar but 
factually contradictory — creating "maximally hard" negatives that cannot be false negatives.

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

**Advantage over Corpus Mining**: Synthetic negatives are guaranteed to be:

1. **Semantically similar** (same topic, vocabulary)
2. **Definitely not relevant** (factually contradictory)
3. **Novel** (not limited to corpus documents)

**Cost Analysis**:

.. code-block:: text

   MS MARCO scale (500K queries):
   
   LLM API Cost (GPT-4o-mini @ $0.15/1M tokens):
   - ~500 tokens per generation
   - 500K queries × 500 tokens = 250M tokens
   - Cost: ~$37.50 per negative per query
   - For 10 negatives: ~$375
   
   Local LLM (Llama-3-8B):
   - ~50 tokens/sec on A100
   - 500K × 500 tokens ÷ 50 = 139 hours
   - Much cheaper but still significant

**Recommendation**: Use SyNeg for **high-value, small-scale** applications (e.g., 
domain-specific fine-tuning with <10K queries) rather than large-scale pretraining. 
For most use cases, **BM25 + denoising** is more practical.

Quantitative Impact of Mining Strategies
=========================================

To justify the need for a unified library, we summarize published benchmark results 
showing the impact of different mining strategies.

Understanding MRR@10
--------------------

**MRR@10 (Mean Reciprocal Rank at 10)** is the official MS MARCO passage ranking metric:

.. math::

   \text{MRR@10} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i} \quad \text{where } \text{rank}_i \leq 10

* Measures the average inverse rank of the **first relevant passage** in the top-10 results
* Scores range from 0 to 1 (higher is better)
* **Typical SOTA performance**: 0.38-0.42 on MS MARCO Dev
* A score of 0.40 means, on average, the first relevant result appears at rank 2.5

MS MARCO Passage Ranking (MRR@10)
---------------------------------

.. list-table:: Impact of Mining Strategies on MS MARCO Dev
   :header-rows: 1
   :widths: 30 25 15 15 15

   * - Method
     - Mining Strategy
     - MRR@10
     - Δ Absolute
     - Δ Relative
   * - BM25 (baseline)
     - Lexical only
     - 0.187
     - —
     - —
   * - DPR
     - Random + BM25 static
     - 0.314
     - +12.7
     - +68%
   * - DPR + In-Batch
     - In-batch negatives
     - 0.326
     - +1.2
     - +4%
   * - ANCE
     - Dynamic ANN refresh
     - 0.330
     - +1.6
     - +5%
   * - SimANS
     - Ambiguous zone
     - 0.341
     - +2.7
     - +9%
   * - RocketQA
     - Cross-batch + denoising
     - 0.370
     - +5.6
     - +18%
   * - ColBERTv2
     - BM25 + CE distillation
     - 0.397
     - +8.3
     - +26%

*Sources: Original papers (Karpukhin 2020, Xiong 2021, Zhou 2022, Qu 2021, Santhanam 2022). 
Results may vary with implementation details and hyperparameters.*

**Key Observations**: 

1. Mining strategy choice accounts for **+1.6 to +8.3 absolute MRR@10 improvement**
2. This is often **larger than architectural changes** (e.g., BERT-base vs BERT-large)
3. **Denoising** (RocketQA, ColBERTv2) provides the largest gains by addressing false negatives
4. **Dynamic mining** (ANCE) shows modest but consistent improvement over static methods

Computational Trade-offs
------------------------

.. list-table:: Computational Cost of Mining Strategies
   :header-rows: 1
   :widths: 20 18 18 18 26

   * - Strategy
     - Pre-training Cost
     - Per-Step Cost
     - Memory
     - Notes
   * - In-Batch
     - None
     - O(B²)
     - O(B × d)
     - B = batch size, d = dim
   * - Static BM25
     - O(N log N)
     - O(1)
     - O(K × N)
     - One-time; K negs/query
   * - Dynamic ANN (ANCE)
     - O(N) per refresh
     - O(log N)
     - O(N × d)
     - Refresh every ~1K steps
   * - Cross-Encoder Denoising
     - O(Q × K × T)
     - O(1)
     - O(K)
     - T = CE inference time
   * - Cross-Batch
     - None
     - O(G × B²)
     - O(G × B × d)
     - G = num GPUs
   * - SimANS
     - O(N log N)
     - O(K)
     - O(K)
     - Score sorting only
   * - LLM-Synthetic
     - O(Q × L)
     - O(1)
     - O(1)
     - L = LLM cost per query

**Practical Estimates** (MS MARCO scale: 8.8M passages, 500K queries):

.. list-table:: Real-World Time Estimates
   :header-rows: 1
   :widths: 30 25 25 20

   * - Strategy
     - Preprocessing Time
     - Training Overhead
     - GPU Memory
   * - In-Batch only
     - 0
     - Baseline
     - ~8GB
   * - Static BM25
     - ~2 hours
     - +0%
     - ~8GB
   * - ANCE (refresh/5K steps)
     - 0
     - +50-100%
     - ~16GB (index)
   * - CE Denoising
     - ~24 hours
     - +0%
     - ~8GB
   * - Cross-Batch (8 GPUs)
     - 0
     - +10%
     - ~12GB/GPU

**Trade-off Summary**: More sophisticated mining generally improves quality but increases 
computational cost. A unified library should make this trade-off explicit and configurable:

* **Best quality/cost ratio**: Static BM25 + denoising (one-time preprocessing, no training overhead)
* **Best quality**: ANCE + denoising (but 2x training time)
* **Fastest**: In-batch only (but lowest quality)

.. important::

   **Why Denoising is the Industry Standard**
   
   Dynamic ANCE mining (re-indexing 8.8M passages every 5K steps) is operationally 
   expensive — it essentially pauses training for significant periods. This is why 
   **static mining + denoising** (ColBERTv2/RocketQA style) dominates production:
   
   * Achieves ~80% of dynamic mining gains with ~10% of the engineering complexity
   * One-time preprocessing cost, zero training overhead
   * No "chasing a moving target" instability
   
   **This insight shapes our library design**: The highest-value contribution is making 
   sophisticated *static* filtering (denoising) accessible, not dynamic ANCE.

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

Based on this analysis, here's a proposed design for a unified hard mining library.

.. _critical-architecture:

.. important::

   **🚨 Critical Architectural Insight: Why This is Harder Than ``rerankers``**
   
   The comparison to ``rerankers`` is tempting but **architecturally misleading**:
   
   * ``rerankers`` is an **inference** library — stateless, sits outside the loop
   * ``hardminers`` is a **training utility** — stateful, coupled to the loop
   
   This changes the physics of the problem entirely. The following sections explain 
   the specific challenges and our architectural response.

Why ``rerankers`` is Easy, ``hardminers`` is Hard
-------------------------------------------------

.. list-table:: Architectural Comparison
   :header-rows: 1
   :widths: 20 40 40

   * - Dimension
     - ``rerankers`` (Inference)
     - ``hardminers`` (Training)
   * - **Statefulness**
     - Stateless: pass query + docs, get scores
     - **Stateful**: needs entire corpus (8.8M passages)
   * - **Loop Coupling**
     - Outside the loop: retrieval → reranking (linear)
     - **Inside the loop**: model ↔ index ↔ negatives (circular)
   * - **Distributed**
     - 8 GPUs = 8 independent copies
     - **Cross-batch needs GPU communication** (DDP nightmare)
   * - **Compute**
     - Re-ranks 50-100 docs in milliseconds
     - **Re-encodes millions of docs** (hours/days)
   * - **User Data**
     - Doesn't care how you store data
     - **Must handle JSONL, Parquet, SQL, Arrow, Vector DBs**

The Four Engineering Cliffs
---------------------------

.. warning::

   **Cliff #1: The Statefulness Trap**
   
   To mine negatives, you need access to the **entire corpus**. Your library cannot 
   just be a function call — it needs to manage a massive index (FAISS, HNSW).
   
   *Challenge*: "How do I load the user's 10GB corpus without crashing their RAM?"

.. warning::

   **Cliff #2: The Loop Coupling Problem**
   
   Dynamic mining (ANCE) creates a **circular dependency**:
   
   1. Trainer updates Model
   2. Miner needs *current* Model to re-encode corpus
   3. Miner updates Index
   4. Miner gives new negatives to Trainer → goto 1
   
   *Challenge*: How do you inject into HuggingFace Trainer, PyTorch Lightning, 
   Accelerate, or raw PyTorch? You need a **Callback System**, not a function.

.. warning::

   **Cliff #3: The Distributed Nightmare (DDP)**
   
   Cross-batch negatives require gathering embeddings across all GPUs. ANCE refresh 
   requires coordinating index updates across workers.
   
   *Challenge*: You will spend 50% of dev time debugging ``torch.distributed`` errors.

.. warning::

   **Cliff #4: The Compute Cost Trap**
   
   If your library defaults to "best practices" (ANCE), users will run it on their 
   laptop and it will hang forever.
   
   *Challenge*: Need safety warnings: *"Dynamic mining on 10M docs without GPU index 
   will take 4 days. Switch to approximate mode?"*

The Correct Architecture: Two Distinct Tools
--------------------------------------------

.. important::

   **Key Insight: Don't try to be ``rerankers`` (a simple wrapper). Be modular components.**
   
   Instead of one giant ``miner.mine()`` black box, split into two distinct tool types:
   
   * **Offline Miners** (preprocessing, decoupled from training)
   * **Online Callbacks** (integrated into training frameworks)

**Phase 1: Offline Miners (The Realistic MVP)**

This is the ``rerankers``-equivalent — high value, achievable, fits "pip install and run":

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                    OFFLINE MINING (Phase 1)                     │
   ├─────────────────────────────────────────────────────────────────┤
   │                                                                 │
   │  Input:  Corpus + Queries + Positives                          │
   │          (JSONL, Parquet, Arrow, HuggingFace Dataset)          │
   │                                                                 │
   │  Process: BM25 → SimANS → Cross-Encoder Denoising              │
   │                                                                 │
   │  Output: .jsonl or .arrow file with:                           │
   │          (query, positive, hard_neg_1, hard_neg_2, ...)        │
   │                                                                 │
   │  ✅ Decoupled from training loop                                │
   │  ✅ Works with ANY trainer                                      │
   │  ✅ Solves fragmentation without coupling headache              │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘

**Phase 2: Online Callbacks (The Advanced Feature)**

For ANCE and dynamic denoising — requires deep integration:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                   ONLINE MINING (Phase 2)                       │
   ├─────────────────────────────────────────────────────────────────┤
   │                                                                 │
   │  NOT a standalone miner — INTEGRATIONS for:                    │
   │                                                                 │
   │  • HuggingFace Trainer: HardMiningCallback                     │
   │  • PyTorch Lightning: HardMiningCallback                       │
   │  • Accelerate: HardMiningPlugin                                │
   │                                                                 │
   │  Meets users where they are — hooks into their loop            │
   │                                                                 │
   │  ⚠️ Requires deep framework knowledge                          │
   │  ⚠️ DDP complexity cannot be fully hidden                      │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘

**The Correct Mental Model:**

.. code-block:: text

   ❌ WRONG (what rerankers does):
      output = lib(input)
   
   ✅ RIGHT (what hardminers must do):
      dataset = lib.preprocess(dataset)     # Phase 1: Offline
      trainer = lib.attach(trainer)         # Phase 2: Online

Recommended Recipe: Where to Start
----------------------------------

.. tip::

   **Default Recommendation for Practitioners**
   
   If you are starting today, **don't implement ANCE**. Start with:
   
   **BM25 Static Mining + Cross-Encoder Denoising**
   
   This offers ~80% of the gains of dynamic mining with ~10% of the engineering complexity.

**Why This Recipe?**

1. **BM25 mining** is fast, well-understood, and catches lexically similar negatives
2. **Cross-encoder denoising** filters false negatives (the main source of training damage)
3. **One-time preprocessing** — no training loop modifications needed
4. **Works with any trainer** — outputs standard triplets

Phase 1 API: Offline Mining (The MVP)
-------------------------------------

This is what you should build first — the realistic, high-value MVP:

Dataset In → Dataset Out
^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   **Philosophy: Dataset In, Dataset Out**
   
   * Input:  Dataset with ``query`` + ``positive`` columns (JSONL, Parquet, HF Dataset)
   * Corpus: Text collection (list, dict, datasets.Dataset, Pyserini index, FAISS index, Vector DB)
   * Output: Same dataset with a new ``hard_negatives`` column
   * ✅ No training loop coupling
   * ✅ Works with SentenceTransformers, ColBERT, PyTorch Lightning, Tevatron, etc.

Hello World: Static BM25
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from hardminers import OfflineMiner
   
   # 1. Initialize the miner (uses rank_bm25 under the hood for zero deps)
   miner = OfflineMiner(
       strategy="bm25",
       corpus=corpus_dataset,     # dict[id -> text], HF Dataset, or list[str]
       num_negatives=10
   )
   
   # 2. Run once (offline). Adds "hard_negatives" column.
   training_data = miner.mine(
       dataset=query_dataset,
       query_column="query",
       positive_column="positive",
       batch_size=32
   )
   
   print(training_data[0]["hard_negatives"][0])
   # "Data mining is a process of..."  <-- Hard negative (lexical overlap)

Power User: BM25 + Cross-Encoder Denoising
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from hardminers import OfflineMiner
   from hardminers.filters import CrossEncoderDenoiser
   
   denoiser = CrossEncoderDenoiser(
       model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
       threshold=0.1,        # If CE score > 0.1 → false negative → discard
       keep_top_k=50         # Only rescore top 50 BM25 hits
   )
   
   miner = OfflineMiner(
       strategy="bm25",
       corpus=corpus_dataset,
       filters=[denoiser]
   )
   
   clean_data = miner.mine(query_dataset)

This abstracts away the 200+ lines of custom script normally required for RocketQA/ColBERTv2 denoising.

Researcher Mode: SimANS / Ambiguous Zone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from hardminers import OfflineMiner
   from hardminers.samplers import AmbiguousZoneSampler
   
   sampler = AmbiguousZoneSampler(
       start_rank=50,
       end_rank=200,
       strategy="uniform"
   )
   
   miner = OfflineMiner(
       strategy="dense",                     # Dense retriever (SentenceTransformers)
       model_name="sentence-transformers/all-MiniLM-L6-v2",
       sampler=sampler
   )
   
   dataset = miner.mine(query_dataset)

Backend Abstraction: Handling Scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Option A: In-memory (small corpora)
   miner = OfflineMiner(strategy="bm25", corpus=list_of_docs)
   
   # Option B: Pyserini (prebuilt MS MARCO index on disk)
   miner = OfflineMiner(
       strategy="bm25",
       backend="pyserini",
       index_path="indexes/msmarco-passage"
   )
   
   # Option C: FAISS (pre-computed dense vectors)
   miner = OfflineMiner(
       strategy="dense",
       backend="faiss",
       index_path="indexes/my_faiss.index"
   )

Outputs & Interoperability
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``output_format="arrow"`` → HuggingFace Dataset
* ``output_format="jsonl"`` → Tevatron / PyTorch Lightning
* ``output_format="parquet"`` → Spark / data warehouses

Users can inspect negatives before training (debuggability) and train with any framework (interoperability).

MVP Checklist
^^^^^^^^^^^^^

1. ``OfflineMiner`` core class (Dataset In → Dataset Out)
2. ``BM25`` retriever (rank_bm25; optional Pyserini backend)
3. ``CrossEncoderDenoiser`` filter (RocketQA/ColBERTv2 killer feature)
4. Outputs in HF Dataset / JSONL / Parquet formats

Phase 2 API: Online Callbacks (Advanced)
----------------------------------------

For users who need dynamic mining — framework-specific integrations:

.. code-block:: python

   # HuggingFace Trainer Integration
   from hardminers.integrations.hf import ANCECallback, DenoisingCallback
   from transformers import Trainer
   
   # ANCE-style dynamic refresh
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset,
       callbacks=[
           ANCECallback(
               refresh_every=5000,      # Refresh index every 5K steps
               top_k=100,
               index_type="faiss_gpu"   # Use GPU index for speed
           )
       ]
   )
   
   # PyTorch Lightning Integration
   from hardminers.integrations.lightning import ANCECallback
   
   trainer = pl.Trainer(
       callbacks=[ANCECallback(refresh_every=5000)]
   )

**Safety Warnings (Built-in):**

.. code-block:: python

   # The library should warn users about compute costs
   miner = OfflineMiner(strategy="ance_offline", corpus_size=10_000_000)
   # WARNING: ANCE on 10M documents without GPU index will take ~4 days.
   # Recommendations:
   #   1. Use 'faiss_gpu' index (requires GPU with 16GB+ VRAM)
   #   2. Use 'approximate' mode (faster, slightly lower quality)
   #   3. Use 'denoised' strategy instead (similar quality, 10x faster)
   # Continue anyway? [y/N]

Vector Database Integration
---------------------------

Modern deployments often store embeddings in vector databases. The library should 
interface with these systems:

.. code-block:: python

   # Mine from existing vector DB (killer feature over academic repos)
   miner = OfflineMiner(strategy="margin", range_min=10, range_max=200)
   
   # From Qdrant
   dataset_with_negatives = miner.mine(
       dataset,
       corpus_source="qdrant://localhost:6333/my_collection"
   )
   
   # From Milvus
   dataset_with_negatives = miner.mine(
       dataset,
       corpus_source="milvus://localhost:19530/embeddings"
   )
   
   # From local FAISS index (skip re-encoding)
   dataset_with_negatives = miner.mine(
       dataset,
       corpus_source="faiss://./my_index.faiss",
       corpus_texts="./corpus.jsonl"  # For text lookup
   )

**Why This Matters**: If users already have data in Qdrant/Milvus/Weaviate, they shouldn't 
need to export to flat files or rebuild local FAISS indices.

Legacy API Design (Reference)
-----------------------------

The original simplified API design is preserved here for reference, but the **Phase 1/2 
architecture above is the recommended approach**.

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

Prioritized by **impact/effort ratio**, not just theoretical elegance:

.. list-table::
   :header-rows: 1
   :widths: 8 22 20 15 15 20

   * - Priority
     - Strategy
     - Paper
     - Complexity
     - Value
     - Notes
   * - **1**
     - **Cross-Encoder Denoising**
     - RocketQA
     - Medium
     - ⭐⭐⭐
     - **Start here!** Highest impact, no lib has this
   * - 2
     - Static BM25
     - DPR
     - Easy
     - ⭐⭐
     - Foundation for denoising
   * - 3
     - Margin-Based Filtering
     - Triplet Loss
     - Easy
     - ⭐⭐
     - Simple, effective
   * - 4
     - In-Batch Negatives
     - Baseline
     - Easy
     - ⭐
     - Already everywhere
   * - 5
     - Ambiguous Zone (SimANS)
     - SimANS
     - Medium
     - ⭐⭐
     - Good alternative to margin
   * - 6
     - Dynamic ANN (ANCE)
     - ANCE
     - Hard
     - ⭐⭐
     - High complexity, moderate gain
   * - 7
     - Query-Side (ADORE)
     - ADORE
     - Hard
     - ⭐⭐
     - Efficient dynamic mining
   * - 8
     - Importance Sampling
     - Sampling Matters
     - Medium
     - ⭐
     - Theoretical, less practical
   * - 9
     - Cross-Batch
     - RocketQA
     - Hard
     - ⭐
     - Requires distributed setup
   * - 10
     - LLM-Synthetic
     - SyNeg
     - Medium
     - ⭐
     - 🧪 Experimental, high cost

**Strategic Insight**: Cross-Encoder Denoising is the **highest-value contribution** because:

1. **No library has it** as a standardized function
2. **Immediate value** — delivers ColBERTv2/RocketQA gains
3. **No training loop coupling** — works as standalone preprocessing
4. **Low engineering complexity** — just BM25 → CE scoring → filtering

Integration with Training Libraries
-----------------------------------

**Example 1: Static Mining + Sentence-Transformers**

.. code-block:: python

   from hardminers import HardMiner
   from sentence_transformers import SentenceTransformer, InputExample
   from sentence_transformers.losses import MultipleNegativesRankingLoss
   from torch.utils.data import DataLoader
   
   # Step 1: Mine hard negatives (one-time preprocessing)
   miner = HardMiner("denoised", 
                     retriever="bm25",
                     denoiser="cross-encoder/ms-marco-MiniLM-L-6-v2",
                     threshold=0.7)
   triplets = miner.mine(queries, corpus, positives)  # Returns (query, pos, neg) triplets
   
   # Step 2: Convert to training format
   train_examples = [
       InputExample(texts=[q, p, n]) for q, p, n in triplets
   ]
   train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True)
   
   # Step 3: Train with standard sentence-transformers
   model = SentenceTransformer("BAAI/bge-base-en-v1.5")
   train_loss = MultipleNegativesRankingLoss(model)
   model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)

**Example 2: Dynamic Mining (ANCE-style) with ColBERT**

.. code-block:: python

   from hardminers import ANCEMiner
   from pylate import ColBERT, Trainer
   
   # Initialize model and dynamic miner
   model = ColBERT("bert-base-uncased")
   miner = ANCEMiner(
       model=model,
       refresh_interval=5000,   # Refresh ANN index every 5K steps
       top_k=200,               # Mine from top-200 candidates
       filter_margin=0.05       # Remove candidates within 5% of positive score
   )
   
   # Training loop with automatic negative refresh
   trainer = Trainer(
       model=model,
       negative_miner=miner,    # Miner integrated into training loop
       train_dataset=dataset,
       per_device_train_batch_size=32
   )
   trainer.train()

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

   **🎯 Key Takeaways**
   
   **The Ecosystem Gap (Validated):**
   
   1. **ColBERTv2 uses**: In-batch + static BM25 + cross-encoder distillation/denoising
   2. **ColBERTv2 does NOT use**: Dynamic refresh (ANCE), curriculum (SimANS), synthetic (SyNeg)
   3. **Gap confirmed**: No unified library for hard negative mining strategies
   4. **Theoretically-grounded methods**: ANCE, ADORE, RocketQA, margin-based, importance sampling

.. warning::

   **🚨 Critical Architectural Lesson**
   
   **The ``rerankers`` analogy is misleading!**
   
   * ``rerankers`` = inference library (stateless, outside the loop)
   * ``hardminers`` = training utility (stateful, coupled to the loop)
   
   **You cannot build a simple ``miner.mine()`` wrapper like ``rerankers``.**
   
   Instead, build **two distinct tools**:
   
   1. **Offline Miners** (Phase 1 MVP): Preprocessing, decoupled, works with any trainer
   2. **Online Callbacks** (Phase 2): Framework-specific integrations for dynamic mining

.. tip::

   **🏆 The Winning Strategy**
   
   **Start with Phase 1: Offline Mining + Cross-Encoder Denoising**
   
   This delivers:
   
   * ~80% of dynamic mining gains
   * ~10% of the engineering complexity
   * No training loop coupling
   * No DDP nightmares
   * Immediate, shippable value
   
   **This is the "low-hanging fruit" that no library currently provides.**

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

Gap Analysis: Visual Summary
----------------------------

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
   │  🔧 Margin (Contrastors, ST)      │  ❌ Ambiguous zone (SimANS)          │
   │  🔧 Denoising (ColBERT only)      │  ❌ LLM-synthetic (SyNeg)            │
   │                                    │  ❌ Importance sampling              │
   │                                    │  ❌ Topic-aware (TAS-Balanced)       │
   └─────────────────────────────────────────────────────────────────────────┘

**The Correct Solution** (NOT a simple ``rerankers`` clone):

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    HARDMINERS ARCHITECTURE                               │
   ├─────────────────────────────────────────────────────────────────────────┤
   │                                                                          │
   │  PHASE 1: OFFLINE MINERS (MVP)    │  PHASE 2: ONLINE CALLBACKS          │
   │  ─────────────────────────────────┼───────────────────────────────────  │
   │  ✅ BM25 Static Mining            │  🔧 HuggingFace Trainer Callback    │
   │  ✅ Cross-Encoder Denoising       │  🔧 PyTorch Lightning Callback      │
   │  ✅ SimANS (Ambiguous Zone)       │  🔧 Accelerate Plugin               │
   │  ✅ Margin-Based Filtering        │  🔧 ANCE Dynamic Refresh            │
   │  ✅ LLM-Synthetic (SyNeg)         │  🔧 Cross-Batch (DDP)               │
   │                                    │                                     │
   │  → Decoupled from training        │  → Coupled to training loop         │
   │  → Works with ANY trainer         │  → Framework-specific               │
   │  → "pip install and run"          │  → Requires deep integration        │
   │  → HIGH VALUE, LOW COMPLEXITY     │  → HIGH VALUE, HIGH COMPLEXITY      │
   └─────────────────────────────────────────────────────────────────────────┘

.. code-block:: python

   # Phase 1 API (The MVP - Start Here!)
   from hardminers import OfflineMiner
   
   miner = OfflineMiner("denoised")    # BM25 + Cross-Encoder filtering
   dataset = miner.mine(dataset, corpus=corpus)
   # → Ready for training with ANY framework
   
   # Phase 2 API (Advanced - Later)
   from hardminers.integrations.hf import ANCECallback
   trainer = Trainer(model, callbacks=[ANCECallback(refresh_every=5000)])

**CONCLUSION**: 

* ✅ **Gap validated**: No unified library exists for hard negative mining
* ⚠️ **Architectural lesson**: Cannot be a simple ``rerankers`` clone
* 🎯 **Winning strategy**: Phase 1 (Offline Denoising) delivers 80% value with 10% complexity
* 🚀 **Start here**: Cross-Encoder Denoising is the highest-value, lowest-complexity contribution

References
==========

**Core Mining Papers:**

1. Santhanam, K., et al. (2022). "ColBERTv2: Effective and Efficient Retrieval via 
   Lightweight Late Interaction." *NAACL 2022*. 
   `Paper <https://aclanthology.org/2022.naacl-main.272/>`_

2. Xiong, L., et al. (2021). "Approximate Nearest Neighbor Negative Contrastive Learning 
   for Dense Text Retrieval." *ICLR 2021*. 
   `arXiv:2007.00808 <https://arxiv.org/abs/2007.00808>`_

3. Zhan, J., et al. (2021). "Optimizing Dense Retrieval Model Training with Hard Negatives 
   (ADORE)." *SIGIR 2021*. `arXiv:2104.08051 <https://arxiv.org/abs/2104.08051>`_

4. Qu, Y., et al. (2021). "RocketQA: An Optimized Training Approach to Dense Passage 
   Retrieval." *NAACL 2021*. `arXiv:2010.08191 <https://arxiv.org/abs/2010.08191>`_

5. Zhou, K., et al. (2022). "SimANS: Simple Ambiguous Negatives Sampling for Dense Text 
   Retrieval." *EMNLP 2022*. `arXiv:2210.11773 <https://arxiv.org/abs/2210.11773>`_

6. Hofstätter, S., et al. (2021). "Efficiently Teaching an Effective Dense Retriever with 
   Balanced Topic Aware Sampling (TAS-Balanced)." *SIGIR 2021*. 
   `arXiv:2104.06967 <https://arxiv.org/abs/2104.06967>`_

7. Zhang, Y., et al. (2024). "SyNeg: LLM-Driven Synthetic Hard Negatives for Dense Retrieval." 
   *arXiv preprint*. `arXiv:2412.17250 <https://arxiv.org/abs/2412.17250>`_

**Theoretical Foundations:**

8. Oord, A., et al. (2018). "Representation Learning with Contrastive Predictive Coding 
   (InfoNCE)." *arXiv preprint*. `arXiv:1807.03748 <https://arxiv.org/abs/1807.03748>`_

9. Wang, T., & Isola, P. (2020). "Understanding Contrastive Representation Learning 
   through Alignment and Uniformity on the Hypersphere." *ICML 2020*. 
   `arXiv:2005.10242 <https://arxiv.org/abs/2005.10242>`_

10. Wu, C., et al. (2017). "Sampling Matters in Deep Embedding Learning." *ICCV 2017*. 
    `arXiv:1706.07567 <https://arxiv.org/abs/1706.07567>`_

**Baseline Papers:**

11. Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question 
    Answering (DPR)." *EMNLP 2020*. `arXiv:2004.04906 <https://arxiv.org/abs/2004.04906>`_

**Additional Relevant Work:**

12. Tabassum, A., et al. (2022). "Hard Negative Sampling Strategies for Contrastive 
    Representation Learning (UnReMix)." *arXiv preprint*. 
    `arXiv:2206.01197 <https://arxiv.org/abs/2206.01197>`_

13. Hofstätter, S., et al. (2021). "Improving Efficient Neural Ranking Models with 
    Cross-Architecture Knowledge Distillation." *SIGIR 2021*. 
    `arXiv:2010.02666 <https://arxiv.org/abs/2010.02666>`_

**Libraries:**

* Sentence-Transformers: https://github.com/huggingface/sentence-transformers
* Contrastors: https://github.com/nomic-ai/contrastors
* FlagEmbedding: https://github.com/FlagOpen/FlagEmbedding
* Rerankers: https://github.com/AnswerDotAI/rerankers
* PyLate: https://github.com/lightonai/pylate
* Hard-Negative-Mixing: https://github.com/davidsvy/hard-negative-mixing
* ColBERT: https://github.com/stanford-futuredata/ColBERT

----

*This document was created as a knowledge consolidation for building a unified hard 
negative mining library. Last updated: December 2024.*

