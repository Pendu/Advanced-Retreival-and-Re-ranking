.. _zelo-elo-inspired-training:

===================================================================
zELO: ELO-inspired Training Method for Rerankers and Embedding Models
===================================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Paper Information
=================

- **Title**: zELO: ELO-inspired Training Method for Rerankers and Embedding Models
- **Authors**: Nicholas Pipitone, Ghita Houir Alami, Advaith Avadhanam, Anton Kaminskyi, Ashley Khoo (ZeroEntropy)
- **Publication**: arXiv:2509.12541v1 [cs.AI], September 16, 2025
- **Models Released**: zerank-1 (Qwen3-4B based), zerank-1-small (Qwen3-1.7B based)
- **Code/Benchmark**: https://github.com/zeroentropy-ai/zbench

Abstract
========

zELO introduces a novel training methodology that optimizes retrieval performance by recognizing that ranking tasks are statistically equivalent to a **Thurstone model**. The method uses unsupervised data to train state-of-the-art open-weight reranker models (zerank-1 and zerank-1-small) that achieve the highest retrieval scores across multiple domains including finance, legal, code, and STEM—outperforming closed-source proprietary rerankers on both NDCG@10 and Recall.

Key training statistics:

- 112,000 queries with 100 documents per query
- Over 5 million query-zELO pairs
- Trained end-to-end in less than 10,000 H100-hours
- No human annotations required

Motivation: The Laffer Curve Problem
====================================

The Fundamental Constraint on Hard Negative Mining
--------------------------------------------------

The authors identify a critical limitation in existing SOTA hard negative mining techniques. Experimentally, they found that making the "hard negative miner" increasingly intelligent eventually **degrades** model performance rather than improving it.

**The Core Problem**: Manual inspection revealed that hard negatives were, on average, *legitimately more relevant* than human-annotated positives. This occurs because:

1. Humans cannot exhaustively scan an entire corpus
2. SOTA methods like LLM-ensemble rerankers can reason on a much larger knowledge base than even expert annotators
3. These methods can identify relevant documents at scale that human annotators would miss

The Laffer Curve Analogy
------------------------

The relationship between hard negative miner intelligence and student model performance follows a Laffer curve pattern:

.. code-block:: text

                        Student Model Performance
                                  ^
                                  |
                                  |        *  Optimal Point
                                  |       /\
                                  |      /  \
                                  |     /    \
                                  |    /      \
                                  |   /        \
                                  |  /          \
                                  | /            \
                                  |/              \
                                  +-----------------> Miner Intelligence

As ensemble-generated hard negatives approach and exceed the quality of human-positive annotations, the marginal benefit from distillation diminishes and eventually becomes **negative**.

**Key Insight**: The highest possible pointwise reranker performance is NOT that which corresponds to the optimal point on this Laffer curve. Hard negative mining is fundamentally flawed, and its accuracy is capped by the training algorithm itself.

The Intractability of False Negatives
-------------------------------------

While one could human-annotate triplets :math:`(q, d^-, d^+)` to confirm :math:`d^-` as a true negative relative to the positive, this is inherently a **pairwise comparison**. For pointwise models, absolute scoring via InfoNCE requires in-batch negatives, which necessitates an unsupervised negative sampling strategy.

This motivates the zELO approach: using **pairwise annotations** from LLM ensembles and converting them to absolute relevance scores via the Elo/Thurstone framework.

The zELO Method
===============

Core Definitions
----------------

**Pointwise Reranker**: A function :math:`R_{point}: Q \times D \rightarrow [0, 1]` such that for a query :math:`q` and corpus :math:`C = \{d_1, \ldots, d_n\}`, if :math:`i_1, \ldots, i_n` is the relevance ranking:

.. math::

   R_{point}(q, d_{i_1}) > R_{point}(q, d_{i_2}) > \ldots > R_{point}(q, d_{i_n})

**Pairwise Reranker**: A function :math:`R_{pair}: Q \times D \times D \rightarrow [0, 1]` where:

.. math::

   p_{ij} := R_{pair}(q, d_i, d_j)

represents the probability that document :math:`d_i` is preferred over document :math:`d_j` for query :math:`q`.

Multi-Stage Training Pipeline
-----------------------------

The zELO method consists of four main stages:

**Stage 1: Candidate Generation**
   Generate candidate documents using a first-stage retriever (e.g., hybrid search combining embeddings with BM25). Top-k = 100 documents are retrieved per query.

**Stage 2: Pairwise Preference Collection**
   Gather sparse pairwise preferences from an ensemble of LLMs (|P| = 3 frontier models). For efficiency, a pairwise SLM is distilled from the LLM ensemble.

**Stage 3: Elo Score Computation**
   Convert pairwise preferences to absolute relevance scores using the Thurstone statistical model via the Bradley-Terry framework.

**Stage 4: Pointwise Reranker Training**
   Fine-tune pointwise rerankers on the computed zELO scores using MSE loss.

Bradley-Terry Model Connection
------------------------------

The Bradley-Terry model assumes that for documents :math:`d_i` and :math:`d_j` with latent abilities :math:`\pi_i` and :math:`\pi_j`:

.. math::

   P(d_i \succ d_j) = \frac{\pi_i}{\pi_i + \pi_j}

In the Elo formulation, parameterizing :math:`\pi_i = e^{Elo_i}`:

.. math::

   P(d_i \succ d_j) = \frac{e^{Elo_i}}{e^{Elo_i} + e^{Elo_j}} = \frac{1}{1 + e^{-(Elo_i - Elo_j)}} = \sigma(Elo_i - Elo_j)

The Elo scores are fit via gradient descent using negative log-likelihood loss:

.. math::

   \mathcal{L} = \sum_{i,j} w_{ij} \log(1 + e^{Elo_j - Elo_i})

Subject to the constraint :math:`\sum_i Elo_i = 0` for normalization.

Thurstone Model Extension
-------------------------

The Thurstone model provides a better fit by assuming document noise follows a normal distribution (rather than Gumbel):

.. math::

   w_{ij} = \frac{1 + \text{erf}(Elo_i - Elo_j)}{2}

This is justified via the central limit theorem, given that document comparisons are subject to multiple sources of noise.

Sparse Matrix Subsampling
=========================

Graph Construction for Efficient Elo Estimation
-----------------------------------------------

Dense :math:`n \times n` pairwise inference is prohibitively expensive. The method uses sparse sampling with :math:`O(n)` pairwise comparisons while maintaining accurate Elo estimates.

**Three Key Properties for Graph G**:

1. **Connectivity**: The graph must be connected to establish relative Elo relationships between all documents.

2. **Minimum Degree**: No nodes should have very low degree (Var[:math:`e'_i`] :math:`\propto 1/\text{deg}(d_i)`).

3. **Low Diameter**: Maximum separation should be small (Var[:math:`e'_i - e'_j`] :math:`\propto \text{dist}_G(d_i, d_j)`).

Random k-Regular Graph via Cycle Splicing
-----------------------------------------

The optimal solution uses :math:`k/2` random n-cycles with their edge sets unioned:

.. code-block:: text

   Step 1: Generate k/2 random cycles over the vertex set
   Step 2: Overlay the cycles to create a k-regular graph
   
   Result: k-connected graph with N = kn/2 edges
           Low diameter: O(log_{k-1}(n))
           Uniform degree distribution

For a random k-regular graph G:

.. math::

   \text{diam}(G) \leq \log_{k-1}(n) + \log_{k-1}(\log(n)) + \log_{k-1}\left(\frac{5}{2}k(k-1)\right)

with probability asymptotically 1 (Bollobás 2001).

**Final Configuration**: N = 400 inferences (0.4% of full 100×100 matrix) with k = 8 (4 random cycles).

Training Details
================

Dataset
-------

- **Queries**: 112,000 publicly available queries across finance, law, medicine, code, and STEM
- **Documents**: >100M publicly available web-scale documents
- **Initial Retrieval**: Qwen3-Embedding-4B embeddings combined via RRF with lexical BM25
- **Top-k**: 100 documents per query

Ensemble Annotation
-------------------

An ensemble of |P| = 3 frontier LLMs generates pairwise preferences:

- Each LLM outputs chain-of-thought justification and preference score on [-1, 1]
- Scores are clamped to {-1, 0, 1} and averaged
- Document order is randomized to mitigate position bias

**Key Finding**: Ensembles of LLMs via zELO generate **higher quality data** than equivalent human annotators on average.

Binary Cross-Entropy Loss for Pairwise Training
-----------------------------------------------

.. math::

   \mathcal{L} = \sum_{i,j \text{ sampled over } q} \text{BCE}(p_{ij}, p'_{ij})

Where:

.. math::

   \text{BCE}(p, q) := -(p \log(q) + (1-p) \log(1-q))

Pointwise Reranker Training
---------------------------

Standard MSE loss for supervised fine-tuning:

.. math::

   \mathcal{L}_{SFT} = \frac{1}{|D_{train}|} \sum_{(q,d,y) \in D_{train}} (R_{point}(q, d) - y)^2

Where :math:`y` values are the computed zELO scores.

RLHF Refinement Stage
---------------------

A second training pass adds data based on pointwise reranker failures:

1. For each query, identify :math:`d_{human}` (highest human-annotated document)
2. Let :math:`r_{human}` = rank of this document by the trained pointwise reranker
3. If :math:`r_{human} > t` (threshold), this is a failure
4. Inference the pairwise ensemble on :math:`(d_{human}, d')` where :math:`d'` is ranked at position :math:`r_{human} - 1`
5. Add this comparison to training data and retrain

This recaptures signal that pure LLM-ensemble distillation missed while using high-SNR human annotations.

Results
=======

Public Benchmark Performance (NDCG@10)
--------------------------------------

+------------------+-------------------+---------------------+------------------------+-----------------+--------------+
| Task             | Default(emb)      | Cohere rerank-v3.5  | Salesforce/Llama-rank  | zerank-1-small  | zerank-1     |
+==================+===================+=====================+========================+=================+==============+
| Code             | 0.678             | 0.724               | 0.694                  | 0.730           | **0.754**    |
+------------------+-------------------+---------------------+------------------------+-----------------+--------------+
| Conversational   | 0.250             | 0.571               | 0.484                  | 0.556           | **0.596**    |
+------------------+-------------------+---------------------+------------------------+-----------------+--------------+
| Finance          | 0.839             | 0.824               | 0.828                  | 0.861           | **0.894**    |
+------------------+-------------------+---------------------+------------------------+-----------------+--------------+
| Legal            | 0.703             | 0.804               | 0.767                  | 0.817           | **0.821**    |
+------------------+-------------------+---------------------+------------------------+-----------------+--------------+
| Medical          | 0.619             | 0.750               | 0.719                  | 0.773           | **0.796**    |
+------------------+-------------------+---------------------+------------------------+-----------------+--------------+
| STEM             | 0.401             | 0.510               | 0.595                  | 0.680           | **0.694**    |
+------------------+-------------------+---------------------+------------------------+-----------------+--------------+

Private Dataset Performance (NDCG@10)
-------------------------------------

+-------------------+---------------------+------------------------+--------------------+-----------------+--------------+
| Task              | Cohere rerank-v3.5  | Salesforce/Llama-rank  | VoyageAI/rerank-2  | zerank-1-small  | zerank-1     |
+===================+=====================+========================+====================+=================+==============+
| Legal             | 0.718               | 0.766                  | 0.746              | 0.799           | **0.854**    |
+-------------------+---------------------+------------------------+--------------------+-----------------+--------------+
| Enterprise Search | 0.674               | 0.629                  | 0.735              | 0.765           | **0.799**    |
+-------------------+---------------------+------------------------+--------------------+-----------------+--------------+
| Conversational    | 0.727               | 0.653                  | 0.727              | 0.747           | **0.787**    |
+-------------------+---------------------+------------------------+--------------------+-----------------+--------------+
| Healthcare        | 0.706               | 0.756                  | 0.749              | 0.885           | **0.898**    |
+-------------------+---------------------+------------------------+--------------------+-----------------+--------------+

**Key Observation**: Margins improve on private datasets, indicating high generalization and low overfitting to public benchmarks.

Latency Comparison
------------------

+-------------+------------+---------------------+----------------------+
| Model       | NDCG@10    | Latency (12 KB)     | Latency (150 KB)     |
+=============+============+=====================+======================+
| Jina m0     | 0.7279     | 547.14 ± 66.84 ms   | 2,543.8 ± 2,984.9 ms |
+-------------+------------+---------------------+----------------------+
| Cohere 3.5  | 0.7091     | 171.5 ± 106.8 ms    | 459.2 ± 87.9 ms      |
+-------------+------------+---------------------+----------------------+
| zerank-1    | **0.7683** | **149.7 ± 53.1 ms** | **314.4 ± 94.6 ms**  |
+-------------+------------+---------------------+----------------------+

Key Contributions
=================

1. **Novel Theoretical Framework**: Identification of the Laffer curve limitation in hard negative mining, explaining why increasingly sophisticated miners eventually degrade performance.

2. **zELO Training Pipeline**: A multi-stage approach that bypasses the Laffer curve by using pairwise comparisons and Elo-based scoring rather than pointwise annotations with hard negatives.

3. **Unsupervised Data Generation**: Demonstration that LLM ensembles via zELO generate higher quality training data than human annotators.

4. **Efficient Sparse Sampling**: k-regular graph construction via cycle splicing achieves accurate Elo estimation with only 0.4% of full pairwise comparisons.

5. **State-of-the-Art Models**: zerank-1 and zerank-1-small achieve best-in-class performance across multiple domains while maintaining competitive latency.

Practical Applications
======================

- **Automated Benchmarking**: zELO can benchmark internal private documents without human annotation
- **Domain-Specific Fine-tuning**: Generate domain-specific training data automatically
- **Live Production Evaluation**: Randomly sample live query logs, annotate via zELO, and discover/fix retrieval issues
- **Continuous Improvement**: Use zELO annotations to fine-tune rerankers on production data

Model Availability
==================

- **zerank-1**: https://huggingface.co/zeroentropy/zerank-1 (Open-weight, ZeroEntropy license)
- **zerank-1-small**: https://huggingface.co/zeroentropy/zerank-1-small (Apache 2.0 License)
- **Evaluation Pipeline**: https://github.com/zeroentropy-ai/zbench

References
==========

.. [1] Pipitone, N., Alami, G.H., Avadhanam, A., Kaminskyi, A., & Khoo, A. (2025). zELO: ELO-inspired Training Method for Rerankers and Embedding Models. arXiv:2509.12541.

.. [2] Bradley, R.A., & Terry, M.E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika, 39(3-4), 324-345.

.. [3] Zermelo, E. (1929). Die Berechnung der Turnier-Ergebnisse als ein Maximumproblem der Wahrscheinlichkeitsrechnung. Mathematische Zeitschrift, 29(1), 436-460.

.. [4] Bollobás, B. (2001). Random graphs (2nd ed.). Cambridge University Press.

.. [5] Yang, A., et al. (2025). Qwen3 technical report. arXiv:2505.09388.

----

