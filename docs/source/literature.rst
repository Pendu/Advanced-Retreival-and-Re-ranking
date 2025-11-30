Literature
==========

Dense Retrieval and Negative Selection Methods
-----------------------------------------------

This section provides a comprehensive list of research papers that advance the state of 
the art in dense retrieval and negative sampling techniques. The table includes detailed 
information about each paper's approach to hard negative mining, training efficiency, 
and key innovations.

Comprehensive Research Papers Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :header-rows: 1
   :widths: 20 12 8 8 52

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Dense Passage Retrieval for Open-Domain Question Answering <https://arxiv.org/abs/2004.04906>`_
     - Karpukhin et al.
     - EMNLP 2020
     - `Python <https://github.com/facebookresearch/DPR>`_
     - **Strategy**: In-batch + BM25 static | **Index Refresh**: No (static BM25) | **Architecture**: Dual-encoder (BERT-base) | In-batch negatives + BM25 hard negatives baseline
   * - `RepBERT: Contextualized Text Embeddings for First-Stage Retrieval <https://arxiv.org/abs/2006.15498>`_
     - Zhan et al.
     - arXiv 2020
     - `Python <https://github.com/jingtaozhan/RepBERT-Index>`_
     - **Strategy**: In-batch only | **Index Refresh**: No | **Architecture**: Shared encoder (BERT-base) | First effective dense retriever with contextualized embeddings
   * - `Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval <https://arxiv.org/abs/2007.00808>`_
     - Xiong et al.
     - ICLR 2021
     - `Python <https://github.com/microsoft/ANCE>`_
     - **Strategy**: Dynamic via ANN index refresh | **Index Refresh**: Yes (periodic async) | **Architecture**: Dual-encoder (RoBERTa-base) | Asynchronous ANN index refresh during training
   * - `Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently <https://arxiv.org/abs/2010.10469>`_
     - Zhan et al.
     - arXiv 2020
     - NA
     - **Strategy**: Multi-strategy analysis | **Architecture**: Various | Comprehensive training strategy analysis
   * - `RocketQA: An Optimized Training Approach to Dense Passage Retrieval <https://arxiv.org/abs/2010.08191>`_
     - Qu et al.
     - NAACL 2021
     - `Python <https://github.com/PaddlePaddle/RocketQA>`_
     - **Strategy**: Cross-batch + denoised hard negs | **Distillation**: Yes (cross-encoder teacher) | **False Neg Handling**: Cross-encoder denoising (~70% false neg detected) | **Architecture**: Dual-encoder (ERNIE-base) | Cross-batch negatives + cross-encoder denoising + data augmentation
   * - `Neural Passage Retrieval with Improved Negative Contrast <https://arxiv.org/abs/2010.12523>`_
     - Lu et al.
     - arXiv 2020
     - NA
     - **Strategy**: Improved negative contrast | **Architecture**: Dual-encoder | Improved negative contrast mechanism
   * - `Scaling Deep Contrastive Learning Batch Size with Almost Constant Peak Memory Usage <https://arxiv.org/abs/2101.06983>`_
     - Gao et al.
     - RepL4NLP 2021
     - `Python <https://github.com/luyug/GradCache>`_
     - **Strategy**: Memory-efficient in-batch | **Training Efficiency**: Very High (memory efficient) | **Architecture**: Any | GradCache for memory-efficient large batch training
   * - `Optimizing Dense Retrieval Model Training with Hard Negatives <https://arxiv.org/abs/2104.08051>`_
     - Zhan et al.
     - SIGIR 2021
     - `Python <https://github.com/jingtaozhan/DRhard>`_
     - **Strategy**: STAR: Static+random; ADORE: Dynamic | **Index Refresh**: STAR: No; ADORE: Fixed index | **Training Efficiency**: High (fixed doc encoder) | **Architecture**: Dual-encoder (RoBERTa-base) | ADORE: Query-side finetuning with LambdaLoss; STAR: Stabilized training
   * - `Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling <https://arxiv.org/abs/2104.06967>`_
     - Hofstätter et al.
     - SIGIR 2021
     - `Python <https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval>`_
     - **Strategy**: Topic-aware clustering + balanced margin | **Training Efficiency**: Very High (single GPU, <48h) | **Distillation**: Yes (dual-teacher: pairwise + in-batch) | **Architecture**: Dual-encoder (DistilBERT 6-layer) | Query clustering + balanced margin + dual-teacher distillation
   * - `PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval <https://arxiv.org/abs/2108.06027>`_
     - Ren et al.
     - ACL Findings 2021
     - `Python <https://github.com/DaoD/PAIR>`_
     - **Strategy**: Query + passage-centric similarity | **Distillation**: Yes (cross-encoder teacher) | **Architecture**: Shared encoder (ERNIE-base) | Passage-centric similarity relation for better discrimination
   * - `Learning Robust Dense Retrieval Models from Incomplete Relevance Labels <https://dl.acm.org/doi/10.1145/3404835.3463106>`_
     - Prakash et al.
     - SIGIR 2021
     - `Python <https://github.com/thakur-nandan/income>`_
     - **Strategy**: Robust to incomplete labels | **False Neg Handling**: Robust training with incomplete labels | **Architecture**: Dual-encoder | Training with incomplete/noisy labels
   * - `Multi-stage Training with Improved Negative Contrast for Neural Passage Retrieval <https://aclanthology.org/2021.emnlp-main.492/>`_
     - Lu et al.
     - EMNLP 2021
     - NA
     - **Strategy**: Multi-stage progressive | **Distillation**: Yes (synthetic data) | **Architecture**: Dual-encoder | Multi-stage training: synthetic pre-train → fine-tune → negative sampling
   * - `Efficient Training of Retrieval Models Using Negative Cache <https://papers.nips.cc/paper/2021/hash/2175f8c5cd9604f6b1e576b252d4c86e-Abstract.html>`_
     - Lindgren et al.
     - NeurIPS 2021
     - `Python <https://github.com/google-research/google-research/tree/master/negative_cache>`_
     - **Strategy**: Cached negatives from previous batches | **Training Efficiency**: High (amortized refresh) | **Architecture**: Dual-encoder | Negative cache mechanism for efficient hard negative reuse
   * - `CODER: An Efficient Framework for Improving Retrieval through Contextual Document Embedding Reranking <https://arxiv.org/abs/2112.08766>`_
     - Zerveas et al.
     - arXiv 2021
     - NA
     - **Strategy**: List-wise with many negatives | **Training Efficiency**: High | **Architecture**: Dual-encoder | Contextual document embedding re-ranking with list-wise loss
   * - `Curriculum Learning for Dense Retrieval Distillation <https://arxiv.org/abs/2204.13679>`_
     - Zeng et al.
     - SIGIR 2022
     - `Python <https://github.com/hansizeng/CPR>`_
     - **Strategy**: Curriculum-based progressive hardness | **Training Efficiency**: High | **Distillation**: Yes (cross-encoder teacher) | **Architecture**: Dual-encoder | CL-DRD: Coarse-to-fine curriculum distillation
   * - `SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval <https://arxiv.org/abs/2210.11773>`_
     - Zhou et al.
     - EMNLP 2022
     - `Python <https://github.com/RUCAIBox/SimANS>`_
     - **Strategy**: Ambiguous negatives (middle-ranked) | **False Neg Handling**: Avoids too-hard (false neg) and too-easy | **Architecture**: Dual-encoder | Sample negatives ranked around positives (ambiguous zone)
   * - `Noisy Pair Corrector for Dense Retrieval <https://aclanthology.org/2023.findings-emnlp.765/>`_
     - EMNLP Authors
     - EMNLP Findings 2023
     - NA
     - **Strategy**: Detection + correction modules | **Distillation**: Yes (EMA model) | **False Neg Handling**: Perplexity-based noise detection + EMA correction | **Architecture**: Dual-encoder | Automatic noisy pair detection and correction
   * - `Mitigating the Impact of False Negatives in Dense Retrieval with Contrastive Confidence Regularization <https://arxiv.org/abs/2401.00165>`_
     - arXiv Authors
     - arXiv 2024
     - NA
     - **Strategy**: Confidence-regularized NCE | **False Neg Handling**: Contrastive confidence regularization | **Architecture**: Dual-encoder | Regularizer for NCE loss to handle false negatives
   * - `TriSampler: A Better Negative Sampling Principle for Dense Retrieval <https://arxiv.org/abs/2402.11855>`_
     - arXiv Authors
     - arXiv 2024
     - NA
     - **Strategy**: Quasi-triangular principle | **False Neg Handling**: Triangular relationship modeling | **Architecture**: Dual-encoder | Quasi-triangular principle: query-positive-negative interplay
   * - `SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval <https://arxiv.org/abs/2412.17250>`_
     - arXiv Authors
     - arXiv 2024
     - NA
     - **Strategy**: LLM-generated synthetic hard negatives | **False Neg Handling**: LLM ensures semantic similarity without relevance | **Architecture**: Dual-encoder | LLM-driven synthetic hard negative generation

---

Summary of Key Strategies
--------------------------

Hard Negative Mining Evolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The field has evolved through multiple generations of increasingly sophisticated techniques:

.. list-table:: Evolution of Hard Negative Mining
   :header-rows: 1
   :widths: 15 25 25 17 18

   * - Generation
     - Strategy
     - Example Papers
     - Pros
     - Cons
   * - **1st Gen**
     - Random / In-batch
     - DPR, RepBERT
     - Simple, efficient
     - Easy negatives, false negatives
   * - **2nd Gen**
     - Static BM25
     - DPR (enhanced)
     - Lexically hard
     - Stale after training
   * - **3rd Gen**
     - Dynamic ANN Refresh
     - ANCE
     - Always fresh negatives
     - Computationally expensive
   * - **4th Gen**
     - Cross-encoder Denoised
     - RocketQA, PAIR
     - Filters false negatives
     - Requires cross-encoder
   * - **5th Gen**
     - Curriculum / Sampling
     - TAS-Balanced, SimANS, CL-DRD
     - Efficient, controlled difficulty
     - Requires careful design
   * - **6th Gen**
     - LLM-Synthetic
     - SyNeg
     - High-quality, controllable
     - LLM inference cost

Key Dimensions of Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When evaluating dense retrieval papers, consider these critical dimensions:

**Hard Negative Strategy**

* **In-batch**: Uses other positives in batch as negatives (baseline)
* **Static**: Pre-mined hard negatives (e.g., BM25)
* **Dynamic**: Refreshes negatives during training (e.g., ANCE)
* **Cross-batch**: Shares negatives across batches (e.g., RocketQA)
* **Curriculum**: Progressive difficulty (e.g., TAS-Balanced, CL-DRD)
* **Synthetic**: LLM-generated (e.g., SyNeg)

**Index Refresh**

* **None**: Static negatives throughout training
* **Periodic**: Asynchronous updates (e.g., ANCE)
* **Cache-based**: Maintains rolling cache (e.g., Negative Cache)

**Training Efficiency**

* **Standard**: Normal training time and memory
* **High**: Optimized for efficiency (e.g., GradCache, TAS-Balanced)
* **Low**: Expensive operations (e.g., frequent index refresh)

**False Negative Handling**

* **None**: No explicit handling
* **Denoising**: Cross-encoder filtering (e.g., RocketQA)
* **Regularization**: Loss-based mitigation (e.g., CCR)
* **Detection**: Automatic identification (e.g., Noisy Pair Corrector)
* **Avoidance**: Strategic sampling (e.g., SimANS)

**Knowledge Distillation**

* **Cross-encoder teacher**: Learns from powerful re-ranker
* **Ensemble teacher**: Learns from multiple models (e.g., TAS-Balanced)
* **Self-distillation**: Learns from own predictions

---

Paper Categories
----------------

By Technique
^^^^^^^^^^^^

**Dynamic Index Refresh**

* **ANCE** (Xiong et al., ICLR 2021) - Refreshes the ANN index during training to maintain hard negatives
* **ADORE** (Zhan et al., SIGIR 2021) - Query-side finetuning with fixed document index

**Cross-Batch and Memory Augmentation**

* **RocketQA** (Qu et al., NAACL 2021) - Uses cross-batch negatives and cross-encoder denoising
* **GradCache** (Gao et al., RepL4NLP 2021) - Enables larger batch sizes through gradient caching
* **Negative Cache** (Lindgren et al., NeurIPS 2021) - Maintains cache of hard negatives

**Query-Side Optimization**

* **ADORE&STAR** (Zhan et al., SIGIR 2021) - Fine-tunes query encoder with frozen document encoder

**Curriculum and Distillation**

* **TAS-Balanced** (Hofstätter et al., SIGIR 2021) - Topic-aware sampling with BERT ensemble distillation
* **CL-DRD** (Zeng et al., SIGIR 2022) - Curriculum learning for retrieval distillation
* **Multi-stage** (Lu et al., EMNLP 2021) - Progressive training pipeline

**False Negative Mitigation**

* **RocketQA** (Qu et al., NAACL 2021) - Cross-encoder denoising
* **SimANS** (Zhou et al., EMNLP 2022) - Ambiguous negatives sampling
* **Noisy Pair Corrector** (EMNLP Findings 2023) - Automatic detection and correction
* **CCR** (arXiv 2024) - Contrastive confidence regularization
* **TriSampler** (arXiv 2024) - Quasi-triangular sampling principle

**LLM-Enhanced Methods**

* **SyNeg** (arXiv 2024) - LLM-driven synthetic hard negative generation

By Year
^^^^^^^

**2020**

* DPR (EMNLP 2020) - Foundation of dense retrieval
* RepBERT (arXiv 2020) - Contextualized embeddings for retrieval
* Learning To Retrieve (arXiv 2020) - Comprehensive training guide
* Neural Passage Retrieval (arXiv 2020) - Improved negative contrast

**2021**

* ANCE (ICLR 2021) - Dynamic index refresh
* RocketQA (NAACL 2021) - Cross-batch negatives and denoising
* GradCache (RepL4NLP 2021) - Memory-efficient training
* ADORE&STAR (SIGIR 2021) - Query-side finetuning
* TAS-Balanced (SIGIR 2021) - Topic-aware sampling
* PAIR (ACL Findings 2021) - Passage-centric similarity
* Learning from incomplete labels (SIGIR 2021) - Robust training
* Multi-stage training (EMNLP 2021) - Progressive hardness
* Negative Cache (NeurIPS 2021) - Efficient negative storage
* CODER (arXiv 2021) - Contextual re-ranking

**2022**

* CL-DRD (SIGIR 2022) - Curriculum distillation
* SimANS (EMNLP 2022) - Ambiguous negatives

**2023**

* Noisy Pair Corrector (EMNLP Findings 2023) - Automatic noise handling

**2024**

* CCR (arXiv 2024) - Contrastive confidence regularization
* TriSampler (arXiv 2024) - Triangular sampling principle
* SyNeg (arXiv 2024) - LLM-driven synthetic negatives

Implementation Resources
-------------------------

All papers with available code implementations are linked in the table above. Most 
implementations are in Python and leverage popular frameworks like:

* **PyTorch** - Primary deep learning framework
* **Hugging Face Transformers** - Pre-trained language models
* **Sentence Transformers** - Dense retrieval library
* **FAISS** - Efficient similarity search and ANN indexing

Recommended Starting Points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For getting started with dense retrieval research and implementation:

**Beginners**

1. **DPR** (Karpukhin et al., EMNLP 2020) - The foundational paper with clean implementation
2. **RepBERT** (Zhan et al., arXiv 2020) - Simple yet effective architecture

**Intermediate**

3. **ANCE** (Xiong et al., ICLR 2021) - For understanding dynamic negative mining
4. **GradCache** (Gao et al., RepL4NLP 2021) - For practical large-batch training on limited hardware
5. **SimANS** (Zhou et al., EMNLP 2022) - Simple but effective negative sampling strategy

**Advanced**

6. **RocketQA** (Qu et al., NAACL 2021) - Comprehensive approach with denoising
7. **TAS-Balanced** (Hofstätter et al., SIGIR 2021) - Efficient training with distillation
8. **CL-DRD** (Zeng et al., SIGIR 2022) - Curriculum learning approach

**Latest Innovations**

9. **SyNeg** (arXiv 2024) - LLM-driven synthetic negative generation
10. **TriSampler** (arXiv 2024) - Novel sampling principle

---

Research Trends
---------------

The field is moving toward:

1. **Efficiency**: Methods like GradCache and TAS-Balanced enable training on single GPUs
2. **False Negative Handling**: Increasing focus on detecting and mitigating false negatives
3. **Curriculum Learning**: Progressive difficulty for stable and effective training
4. **LLM Integration**: Using large language models to generate high-quality synthetic data
5. **Practical Deployment**: Focus on methods that work well in real-world settings
