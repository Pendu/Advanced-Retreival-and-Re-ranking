Literature Overview
===================

This section provides access to all research papers covered in this documentation, 
organized by their role in the retrieval and re-ranking pipeline.

.. note::

   **Papers are now organized by topic!** Instead of one large table, papers are 
   distributed across focused sections below. This makes it easier to find papers 
   relevant to your specific interest.

Papers by Stage
---------------

Stage 1: Retrieval
^^^^^^^^^^^^^^^^^^

Papers focused on efficiently retrieving candidates from large corpora.

**By Topic:**

* **Sparse Methods** (:doc:`stage1_retrieval/sparse`)
  
  * BM25 and traditional IR

* **Dense Baselines** (:doc:`stage1_retrieval/dense_baselines`)
  
  * DPR (Karpukhin et al., EMNLP 2020)
  * RepBERT (Zhan et al., arXiv 2020)

* **Hard Negative Mining** (:doc:`stage1_retrieval/hard_mining`)
  
  * ANCE (Xiong et al., ICLR 2021) - Dynamic index refresh
  * RocketQA (Qu et al., NAACL 2021) - Cross-batch denoising
  * ADORE (Zhan et al., SIGIR 2021) - Query-side finetuning
  * TAS-Balanced (Hofstätter et al., SIGIR 2021) - Topic-aware sampling
  * SimANS (Zhou et al., EMNLP 2022) - Ambiguous negatives
  * GradCache (Gao et al., RepL4NLP 2021) - Memory-efficient training
  * CL-DRD (Zeng et al., SIGIR 2022) - Curriculum learning
  * SyNeg (arXiv 2024) - LLM-driven synthetic negatives
  * And many more...

* **Late Interaction** (:doc:`stage1_retrieval/late_interaction`)
  
  * ColBERT (Khattab & Zaharia, SIGIR 2020)
  * ColBERTv2 (Santhanam et al., NAACL 2022)
  * Poly-encoders (Humeau et al., ICLR 2020)

* **Hybrid Methods** (:doc:`stage1_retrieval/hybrid`)
  
  * DENSPI (Seo et al., ACL 2019)
  * Semantic Residual (Gao et al., ECIR 2021)
  * DensePhrases (Lee et al., ACL 2021)

* **Pre-training** (:doc:`stage1_retrieval/pretraining`)
  
  * ORQA/ICT (Lee et al., ACL 2019)
  * REALM (Guu et al., ICML 2020)
  * Condenser (Gao & Callan, EMNLP 2021)
  * coCondenser (Gao & Callan, ACL 2022)
  * Contriever (Izacard et al., TMLR 2022)

* **Joint Learning** (:doc:`stage1_retrieval/joint_learning`)
  
  * JPQ (Zhan et al., CIKM 2021)
  * EHI/Poeem (arXiv 2023)

Stage 2: Re-ranking
^^^^^^^^^^^^^^^^^^^

Papers focused on precision scoring of candidates.

**By Topic:**

* **Cross-Encoders** (:doc:`stage2_reranking/cross_encoders`)
  
  * BERT Cross-Encoder
  * MonoT5 / RankT5
  * Training strategies

* **LLM Re-rankers** (:doc:`stage2_reranking/llm_rerankers`)
  
  * RankGPT
  * RankLlama
  * Zero-shot prompting approaches

Papers by Research Theme
------------------------

By Key Innovation
^^^^^^^^^^^^^^^^^

**Hard Negative Mining**

The core bottleneck in dense retrieval—see :doc:`stage1_retrieval/hard_mining` for:

* Dynamic mining (ANCE)
* Cross-encoder denoising (RocketQA)
* Score-based sampling (SimANS)
* Curriculum learning (CL-DRD)
* LLM synthesis (SyNeg)

**False Negative Handling**

Methods that address the damaging effects of false negatives:

* RocketQA: Cross-encoder filtering (~70% detection rate)
* TAS-Balanced: Balanced margin reduces noise
* Noisy Pair Corrector: Perplexity-based detection
* CCR: Confidence regularization
* TriSampler: Triangular relationship modeling

**Training Efficiency**

Methods that reduce computational cost:

* GradCache: Memory-efficient large batches
* Negative Cache: Amortized hard negative mining
* TAS-Balanced: Single GPU training (<48h)
* ADORE: Fixed document encoder
* JPQ: Joint query-index optimization

**Knowledge Distillation**

Using strong teachers to train fast students:

* RocketQA: Cross-encoder teacher
* PAIR: Passage-centric similarity
* TAS-Balanced: Dual-teacher (pairwise + in-batch)
* ColBERTv2: Denoised supervision
* CL-DRD: Curriculum distillation

By Dataset/Domain
^^^^^^^^^^^^^^^^^

Papers organized by evaluation dataset:

* **MS MARCO**: Most papers (standard benchmark)
* **Natural Questions**: DPR, REALM, ORQA
* **BEIR** (zero-shot): Contriever, coCondenser, BGE
* **Domain-specific**: Legal, medical, code search

Complete Chronological Timeline
--------------------------------

**2019**

* ORQA (Lee et al., ACL)
* DENSPI (Seo et al., ACL)
* Poly-encoders (Humeau et al., ICLR)

**2020**

* DPR (Karpukhin et al., EMNLP) - *The foundation*
* RepBERT (Zhan et al., arXiv)
* REALM (Guu et al., ICML)
* ANCE (Xiong et al., ICLR 2021, arXiv 2020)
* RocketQA (Qu et al., NAACL 2021, arXiv 2020)
* ColBERT (Khattab & Zaharia, SIGIR)

**2021**

* TAS-Balanced (Hofstätter et al., SIGIR)
* ADORE (Zhan et al., SIGIR)
* PAIR (Ren et al., ACL Findings)
* GradCache (Gao et al., RepL4NLP)
* Negative Cache (Lindgren et al., NeurIPS)
* Condenser (Gao & Callan, EMNLP)
* DensePhrases (Lee et al., ACL)
* JPQ (Zhan et al., CIKM)

**2022**

* SimANS (Zhou et al., EMNLP)
* CL-DRD (Zeng et al., SIGIR)
* ColBERTv2 (Santhanam et al., NAACL)
* coCondenser (Gao & Callan, ACL)
* Contriever (Izacard et al., TMLR)

**2023**

* Noisy Pair Corrector (EMNLP Findings)
* EHI/Poeem (arXiv)
* `BGE <https://arxiv.org/abs/2309.07597>`_ (BAAI) - State-of-the-art embedding models
* `E5-Mistral <https://arxiv.org/abs/2401.00368>`_ (Microsoft) - LLM-based embeddings

**2024**

* CCR (arXiv)
* TriSampler (arXiv)
* SyNeg (arXiv)
* `LLM2Vec <https://arxiv.org/abs/2404.05961>`_ (McGill) - Converting LLMs to text encoders
* `BGE-M3 <https://arxiv.org/abs/2402.03216>`_ (BAAI) - Multi-lingual, multi-granularity embeddings
* `Jina Embeddings v3 <https://arxiv.org/abs/2409.10173>`_ (Jina AI) - 8K context window embeddings
* `NV-Embed <https://arxiv.org/abs/2405.17428>`_ (NVIDIA) - Generalist embedding model

Quick Navigation
----------------

**I want to improve my retrieval model's accuracy:**

→ Start with :doc:`stage1_retrieval/hard_mining`

**I'm building a system from scratch:**

→ Read :doc:`rag_overview` then :doc:`stage1_retrieval/index`

**I need faster inference:**

→ See :doc:`stage1_retrieval/sparse` (BM25) or :doc:`stage1_retrieval/joint_learning` (compression)

**I want better re-ranking:**

→ See :doc:`stage2_reranking/cross_encoders` or :doc:`stage2_reranking/llm_rerankers`

**I'm doing research on training techniques:**

→ Deep dive into :doc:`stage1_retrieval/hard_mining` and :doc:`overview` (theory)

Contributing New Papers
------------------------

See :doc:`contributing` for how to add new papers to this collection.

When adding papers, please categorize them appropriately:

* Stage 1 or Stage 2?
* What's the key innovation?
* Which section does it best fit?
