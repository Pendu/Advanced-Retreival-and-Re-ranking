Stage 1: Retrieval Methods
===========================

This section covers methods for the first stage of the RAG pipeline: efficiently retrieving 
candidate documents from large corpora.

.. toctree::
   :maxdepth: 2
   :caption: Stage 1 Topics:

   sparse
   dense_baselines
   hard_mining
   late_interaction
   hybrid
   pretraining
   joint_learning
   literature_survey/index

Overview
--------

Stage 1 retrieval must balance two competing demands:

* **Speed**: Process millions of documents in milliseconds
* **Recall**: Don't miss relevant documents

The solution is to use architectures that allow pre-computation and efficient similarity search.

Evolution of Stage 1 Methods
-----------------------------

.. list-table:: Historical Evolution
   :header-rows: 1
   :widths: 15 25 30 30

   * - Era
     - Method Type
     - Key Innovation
     - Representative Papers
   * - Pre-2020
     - Sparse (BM25)
     - Inverted index, TF-IDF
     - Traditional IR
   * - 2020
     - Dense Baselines
     - Dual-encoder with BERT
     - DPR, RepBERT
   * - 2021
     - Hard Negatives
     - Dynamic mining, denoising
     - ANCE, RocketQA, ADORE
   * - 2021-2022
     - Late Interaction
     - Token-level representations
     - ColBERT, ColBERTv2
   * - 2022-2023
     - Sampling Strategies
     - Curriculum, score-based
     - TAS-Balanced, SimANS
   * - 2023-2024
     - LLM Integration
     - Synthetic negatives, prompting
     - SyNeg, LLM embeddings

Key Dimensions
--------------

When evaluating Stage 1 methods, consider:

**Architecture**

* **Dual-Encoder**: Independent encoding (fastest)
* **Late Interaction**: Token-level matching (more accurate)
* **Hybrid**: Combines sparse and dense

**Training Strategy**

* **Negative Mining**: How to find informative negatives?
* **Knowledge Distillation**: Learn from cross-encoder teachers
* **Curriculum Learning**: Progressive difficulty

**Index Structure**

* **Dense Vector**: Single vector per document
* **Multi-Vector**: Multiple vectors (e.g., ColBERT)
* **Learned Index**: End-to-end optimized structures

Quick Navigation
----------------

* :doc:`sparse` - BM25 and traditional IR methods
* :doc:`dense_baselines` - DPR, RepBERT (foundational papers)
* :doc:`hard_mining` - **Core focus**: Advanced negative mining strategies
* :doc:`late_interaction` - ColBERT and token-level methods
* :doc:`hybrid` - Combining sparse and dense
* :doc:`pretraining` - Pre-training strategies for dense retrievers
* :doc:`joint_learning` - Jointly optimizing retrieval and indexing

The Central Challenge: Hard Negative Mining
--------------------------------------------

The quality of Stage 1 retrievers depends critically on the negative examples used during 
training. This is explored in depth in :doc:`hard_mining`, which covers:

* The hard negative problem
* Evolution from static to dynamic mining
* Denoising strategies
* Curriculum learning
* LLM-based generation

This is the **primary bottleneck** in dense retrieval research today.

