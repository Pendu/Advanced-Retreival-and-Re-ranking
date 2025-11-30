Hard Negative Mining
====================

**The quality of negative samples is the primary determinant of dense retrieval performance.**

This section covers strategies for mining "hard" negatives—documents that are semantically 
similar to the query but not relevant. This is the **core bottleneck** in dense retrieval 
research and the main focus of modern improvements.

.. important::

   **The Arms Race Metaphor**: Advanced negative mining is a co-evolutionary arms race 
   between the retriever (student model being trained) and the sampler (mechanism for 
   finding negatives). As the model improves, the negatives must get harder.

The Hard Negative Problem
--------------------------

See the detailed explanation in the main documentation: :doc:`../hard_negatives`

**Quick Summary:**

* **Easy Negatives**: Random, unrelated documents → Model learns too quickly, no challenge
* **Hard Negatives**: Semantically similar but irrelevant → Forces fine-grained discrimination
* **False Negatives**: Actually relevant but mislabeled → Actively damages training

The goal is to mine negatives in the "Goldilocks zone": hard enough to be challenging, 
but not so hard that they're actually false positives.

Evolution of Hard Negative Strategies
--------------------------------------

.. list-table:: Six Generations of Negative Mining
   :header-rows: 1
   :widths: 15 25 25 20 15

   * - Generation
     - Strategy
     - Example Papers
     - Pros
     - Cons
   * - **1st Gen**
     - Random / In-batch
     - DPR, RepBERT
     - Simple, fast
     - Easy negatives
   * - **2nd Gen**
     - Static BM25
     - DPR (enhanced)
     - Lexically hard
     - Stale quickly
   * - **3rd Gen**
     - Dynamic ANN Refresh
     - ANCE
     - Always fresh
     - Expensive
   * - **4th Gen**
     - Cross-encoder Denoised
     - RocketQA, PAIR
     - Filters false negs
     - Needs cross-encoder
   * - **5th Gen**
     - Curriculum / Smart Sampling
     - TAS-Balanced, SimANS
     - Efficient
     - Complex design
   * - **6th Gen**
     - LLM-Synthetic
     - SyNeg
     - Perfect calibration
     - LLM cost

Hard Negative Mining Papers
----------------------------

Core Dynamic Mining Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval (ANCE) <https://arxiv.org/abs/2007.00808>`_
     - Xiong et al.
     - ICLR 2021
     - `Code <https://github.com/microsoft/ANCE>`_
     - **Dynamic Index Refresh**: Asynchronously refreshes ANN index using latest model checkpoint to find negatives hard for the *current* model state, not just initial model.
   * - `RocketQA: An Optimized Training Approach to Dense Passage Retrieval <https://arxiv.org/abs/2010.08191>`_
     - Qu et al.
     - NAACL 2021
     - `Code <https://github.com/PaddlePaddle/RocketQA>`_
     - **Cross-Batch & Denoising**: Uses cross-batch negatives (share across GPUs) and filters ~70% false negatives using cross-encoder teacher. Huge negative pool improves discrimination.
   * - `Optimizing Dense Retrieval Model Training with Hard Negatives (ADORE) <https://arxiv.org/abs/2104.08051>`_
     - Zhan et al.
     - SIGIR 2021
     - `Code <https://github.com/jingtaozhan/DRhard>`_
     - **Query-side Finetuning**: ADORE optimizes query encoder against fixed document index to generate dynamic negatives efficiently. STAR adds stabilization with random negatives.
   * - `Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently <https://arxiv.org/abs/2010.10469>`_
     - Zhan et al.
     - arXiv 2020
     - NA
     - **Comprehensive Analysis**: Systematic study of training strategies including negative sampling, loss functions, and architecture choices. Essential reading for understanding trade-offs.
   * - `Neural Passage Retrieval with Improved Negative Contrast <https://arxiv.org/abs/2010.12523>`_
     - Lu et al.
     - arXiv 2020
     - NA
     - **Improved Contrast**: Enhanced negative contrast mechanism for better discrimination between relevant and irrelevant passages.
   * - `Learning Robust Dense Retrieval Models from Incomplete Relevance Labels <https://arxiv.org/abs/2104.07662>`_
     - Prakash et al.
     - SIGIR 2021
     - `Code <https://github.com/thakur-nandan/income>`_
     - **Incomplete Labels**: Robust training methodology that handles incomplete and noisy relevance labels, common in real-world datasets.

Smart Sampling Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling (TAS-Balanced) <https://arxiv.org/abs/2104.06967>`_
     - Hofstätter et al.
     - SIGIR 2021
     - `Code <https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval>`_
     - **Topic Aware Sampling**: Clusters queries into topics and samples negatives that are topologically related but distinct. Dual-teacher distillation (pairwise + in-batch). Trains on single GPU in <48h!
   * - `PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval <https://arxiv.org/abs/2108.06027>`_
     - Ren et al.
     - ACL Findings 2021
     - NA
     - **Passage-Centric Loss**: Enforces structure among passages themselves (not just query-passage). Uses cross-encoder teacher with confidence thresholds to denoise.
   * - `SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval <https://arxiv.org/abs/2210.11773>`_
     - Zhou et al.
     - EMNLP 2022
     - `Code <https://github.com/microsoft/SimXNS/tree/main/SimANS>`_
     - **Ambiguous Zone Sampling**: Samples negatives from "ambiguous zone"—ranked neither too high (false positive risk) nor too low (too easy). Avoids both extremes.
   * - `Curriculum Learning for Dense Retrieval Distillation (CL-DRD) <https://arxiv.org/abs/2204.13679>`_
     - Zeng et al.
     - SIGIR 2022
     - NA
     - **Curriculum-based Distillation**: Trains on progressively harder negatives (easy → medium → hard). Progressive difficulty controls noise and stabilizes training. Cross-encoder teacher guidance.

Efficiency and Scalability
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Scaling Deep Contrastive Learning Batch Size with Almost Constant Peak Memory Usage (GradCache) <https://arxiv.org/abs/2101.06983>`_
     - Gao et al.
     - RepL4NLP 2021
     - `Code <https://github.com/luyug/GradCache>`_
     - **Memory-Efficient In-Batch**: Gradient caching enables very large batch sizes (thousands) with constant memory. More in-batch negatives = harder negatives for free.
   * - `Efficient Training of Retrieval Models Using Negative Cache <https://papers.nips.cc/paper/2021/hash/2175f8c5cd9604f6b1e576b252d4c86e-Abstract.html>`_
     - Lindgren et al.
     - NeurIPS 2021
     - `Code <https://github.com/google-research/google-research/tree/master/negative_cache>`_
     - **Negative Cache**: Caches negatives from previous batches for reuse. Amortizes cost of hard negative mining across steps. Mitigates stale negative problem through cache rotation.
   * - `Multi-stage Training with Improved Negative Contrast for Neural Passage Retrieval <https://aclanthology.org/2021.emnlp-main.492/>`_
     - Lu et al.
     - EMNLP 2021
     - NA
     - **Multi-Stage Training**: Synthetic pre-train → fine-tune → negative sampling. Progressive hardness across stages. Uses synthetic data and multi-stage refinement.

Advanced False Negative Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Noisy Pair Corrector for Dense Retrieval <https://aclanthology.org/2023.findings-emnlp.765/>`_
     - EMNLP Authors
     - EMNLP Findings 2023
     - NA
     - **Automatic Detection & Correction**: Uses perplexity-based noise detection to identify false negatives. EMA (exponential moving average) model provides correction signal.
   * - `Mitigating the Impact of False Negatives in Dense Retrieval with Contrastive Confidence Regularization (CCR) <https://arxiv.org/abs/2401.00165>`_
     - arXiv Authors
     - arXiv 2024
     - NA
     - **Confidence Regularization**: Adds regularizer to NCE (noise contrastive estimation) loss that adjusts based on model's confidence in negative labels. Softens penalties for uncertain negatives.
   * - `TriSampler: A Better Negative Sampling Principle for Dense Retrieval <https://arxiv.org/abs/2402.11855>`_
     - arXiv Authors
     - arXiv 2024
     - NA
     - **Quasi-Triangular Principle**: Models the triangular relationship among query, positive, and negative. Ensures negatives are far from positive but informative relative to query.

LLM-Enhanced Methods (Latest Frontier)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval <https://arxiv.org/abs/2412.17250>`_
     - arXiv Authors
     - arXiv 2024
     - NA
     - **LLM-Synthetic Generation**: Uses LLMs to generate text that is semantically similar to positive but factually contradictory. Creates "perfect" hard negatives that may not exist in corpus. Risk: adversarial gap (model learns LLM detector).

Key Implementation Strategies
------------------------------

For detailed implementation guidance, see the full documentation section on each strategy.

Strategy 1: ANCE-Style Dynamic Mining
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Architecture**: Two asynchronous processes

1. **Training Process**: Reads hard negatives, trains model, saves checkpoints
2. **Generation Process**: Loads latest checkpoint, re-encodes corpus, mines new negatives

**When to Use**: Large-scale, SOTA performance critical, have distributed infrastructure

**Challenge**: Catastrophic forgetting if refresh too frequent

Strategy 2: RocketQA-Style Cross-Batch Denoising
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Architecture**: Share negatives across GPUs + cross-encoder filtering

1. Collect negatives from all GPUs in cluster (huge pool)
2. Use cross-encoder to score and filter false negatives
3. Train bi-encoder on denoised set

**When to Use**: Multi-GPU setup, false negatives are major concern

**Advantage**: ~70% false negative detection improves quality significantly

Strategy 3: TAS-Balanced-Style Topic Clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Architecture**: Cluster queries → sample from same cluster

1. Cluster queries by topic (e.g., k-means on embeddings)
2. For each query, sample negatives from same cluster
3. Use balanced margin loss to handle noise

**When to Use**: Want efficiency (single GPU), have query distribution

**Advantage**: Very efficient, matches ANCE performance in less time

Strategy 4: SimANS-Style Ambiguous Sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Architecture**: Score-based filtering around positive

1. Retrieve top-k candidates with student model
2. Filter to "ambiguous zone": sim ∈ [positive_score - margin, positive_score + margin]
3. These are hard but not false negatives

**When to Use**: Simple to implement, works well in practice

**Advantage**: Avoids extremes (too easy, too hard)

Bird's Eye View: Which Strategy to Choose?
-------------------------------------------

.. list-table:: Strategy Comparison
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Strategy
     - Complexity
     - Performance
     - Speed
     - False Neg Rate
     - Best For
   * - In-Batch (Baseline)
     - Very Low
     - Baseline
     - Very Fast
     - High (15-25%)
     - Prototyping
   * - Static BM25
     - Low
     - +5-8%
     - Fast
     - Medium (8-15%)
     - First iteration
   * - ANCE (Dynamic)
     - Very High
     - SOTA
     - Slow
     - Low (3-6%)
     - Research, Scale
   * - RocketQA (Denoised)
     - High
     - SOTA -2%
     - Medium
     - Very Low (<2%)
     - Multi-GPU
   * - TAS-Balanced
     - Medium
     - SOTA -5%
     - Fast
     - Low (3-5%)
     - **Recommended**
   * - SimANS
     - Low
     - SOTA -7%
     - Fast
     - Low (2-4%)
     - **Recommended**
   * - SyNeg (LLM)
     - Medium
     - SOTA -3%
     - Medium
     - Very Low (<1%)
     - Domain-specific

Recommended Path for Practitioners
-----------------------------------

**Phase 1: Baseline (Week 1)**

Start with in-batch negatives to establish baseline:

.. code-block:: python

   from sentence_transformers import SentenceTransformer, losses
   
   model = SentenceTransformer('bert-base-uncased')
   loss = losses.MultipleNegativesRankingLoss(model)

**Phase 2: Static Hard Negatives (Week 2)**

Add BM25-mined negatives for 5-8% improvement:

.. code-block:: python

   # Mine negatives offline
   bm25_negatives = bm25.search(query, k=100)
   hard_negs = [neg for neg in bm25_negatives if not is_positive(neg)]

**Phase 3: Cross-Encoder Denoising (Week 3)**

Implement 2-step pipeline for 10-15% improvement:

.. code-block:: python

   # Filter BM25 negatives with cross-encoder
   cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
   scores = cross_encoder.predict([(q, neg) for neg in bm25_negatives])
   denoised = [neg for neg, score in zip(bm25_negatives, scores) if score < 0.5]

**Phase 4: Smart Sampling (Week 4)**

Add score-based filtering (SimANS-style) for another 5-10%:

.. code-block:: python

   # Mine in "ambiguous zone"
   pos_score = model.similarity(query, positive)
   margin = 0.1
   ambiguous_negatives = [
       neg for neg in candidates 
       if pos_score - margin < model.similarity(query, neg) < pos_score + margin
   ]

**Phase 5: Curriculum (Optional, Week 5)**

Wrap in curriculum for stability:

.. code-block:: python

   # Stage 1: Easy (in-batch)
   train(model, in_batch_data, epochs=2)
   
   # Stage 2: Medium (BM25)
   train(model, bm25_data, epochs=3)
   
   # Stage 3: Hard (denoised)
   train(model, denoised_data, epochs=5)

This progression gives you **85-90% of SOTA performance** with manageable complexity.

Common Pitfalls
---------------

❌ **Pitfall 1**: Not measuring false negative rate

* Assume 10-20% false negative rate in naive mining
* Validate on subset with exhaustive labels

❌ **Pitfall 2**: Using only the hardest negatives

* Balance hard and medium negatives
* Too-hard negatives are likely false negatives

❌ **Pitfall 3**: Ignoring computational cost

* ANCE requires 3-5x more compute than 2-step
* Factor in infrastructure cost, not just metrics

❌ **Pitfall 4**: Not using curriculum

* Curriculum is low-cost, high-reward
* Almost always improves by 3-5%

❌ **Pitfall 5**: Treating all queries equally

* Query difficulty varies widely
* Consider query-specific thresholds

Next Steps
----------

* See :doc:`../overview` for the full explanation of the hard negative problem
* See :doc:`../hard_negatives` for detailed analysis of why baselines fail
* See :doc:`dense_baselines` for the foundational papers (DPR, RepBERT)
* See :doc:`late_interaction` for ColBERT's approach to negatives

