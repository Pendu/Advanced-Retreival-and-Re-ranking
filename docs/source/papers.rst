Papers
======

Dense Retrieval and Negative Selection Methods
-----------------------------------------------

This section provides a comprehensive list of research papers that advance the state of 
the art in dense retrieval and negative sampling techniques.

.. list-table:: Research Papers
   :header-rows: 1
   :widths: 30 15 10 10 35

   * - Paper
     - Author
     - Venue
     - Code
     - Key Contribution
   * - `Dense Passage Retrieval for Open-Domain Question Answering <https://arxiv.org/abs/2004.04906>`_
     - Vladimir Karpukhin et al.
     - EMNLP 2020
     - `Python <https://github.com/facebookresearch/DPR>`_
     - DPR with in-batch negatives
   * - `RepBERT: Contextualized Text Embeddings for First-Stage Retrieval <https://arxiv.org/abs/2006.15498>`_
     - Jingtao Zhan et al.
     - Arxiv 2020
     - `Python <https://github.com/jingtaozhan/RepBERT-Index>`_
     - RepBERT architecture
   * - `Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval <https://arxiv.org/abs/2007.00808>`_
     - Lee Xiong et al.
     - ICLR 2021
     - `Python <https://github.com/microsoft/ANCE>`_
     - ANCE - refresh index during training
   * - `Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently <https://arxiv.org/abs/2010.10469>`_
     - Jingtao Zhan et al.
     - Arxiv 2020
     - NA
     - Comprehensive training strategies
   * - `RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering <https://arxiv.org/abs/2010.08191>`_
     - Yingqi Qu et al.
     - NAACL 2021
     - `Python <https://github.com/PaddlePaddle/RocketQA>`_
     - Cross-batch negatives, denoise hard negatives and data augmentation
   * - `Neural Passage Retrieval with Improved Negative Contrast <https://arxiv.org/abs/2010.12523>`_
     - Jing Lu et al.
     - Arxiv 2020
     - NA
     - Improved negative contrast
   * - `Scaling deep contrastive learning batch size under memory limited setup <https://arxiv.org/abs/2101.06983>`_
     - Luyu Gao et al.
     - RepL4NLP 2021
     - `Python <https://github.com/luyug/GradCache>`_
     - GradCache for memory-efficient training
   * - `Optimizing Dense Retrieval Model Training with Hard Negatives <https://arxiv.org/abs/2104.08051>`_
     - Jingtao Zhan et al.
     - SIGIR 2021
     - `Python <https://github.com/jingtaozhan/DRhard>`_
     - ADORE&STAR - query-side finetuning on pretrained document encoders
   * - `Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling <https://arxiv.org/abs/2104.06967>`_
     - Sebastian Hofstätter et al.
     - SIGIR 2021
     - `Python <https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval>`_
     - TAS-Balanced - sample from query cluster and distill from BERT ensemble
   * - `PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval <https://arxiv.org/abs/2108.06027>`_
     - Ruiyang Ren et al.
     - EMNLP Findings 2021
     - `Python <https://github.com/DaoD/PAIR>`_
     - Passage-centric similarity relations
   * - `Learning robust dense retrieval models from incomplete relevance labels <https://dl.acm.org/doi/10.1145/3404835.3463106>`_
     - Prafull Prakash et al.
     - SIGIR 2021
     - `Python <https://github.com/thakur-nandan/income>`_
     - Training with incomplete labels
   * - `Multi-stage training with improved negative contrast for neural passage retrieval <https://aclanthology.org/2021.emnlp-main.492/>`_
     - Jing Lu et al.
     - EMNLP 2021
     - NA
     - Multi-stage training approach
   * - `Efficient Training of Retrieval Models Using Negative Cache <https://papers.nips.cc/paper/2021/hash/2175f8c5cd9604f6b1e576b252d4c86e-Abstract.html>`_
     - Erik M. Lindgren et al.
     - NeurIPS 2021
     - `Python <https://github.com/google-research/google-research/tree/master/negative_cache>`_
     - Negative cache mechanism
   * - `CODER: An efficient framework for improving retrieval through COntextual Document Embedding Reranking <https://arxiv.org/abs/2112.08766>`_
     - George Zerveas et al.
     - Arxiv 2021
     - NA
     - Contextual document embedding re-ranking
   * - `Curriculum Learning for Dense Retrieval Distillation <https://arxiv.org/abs/2204.13679>`_
     - Hansi Zeng et al.
     - SIGIR 2022
     - `Python <https://github.com/hansizeng/CPR>`_
     - Curriculum-based distillation
   * - `SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval <https://arxiv.org/abs/2210.11773>`_
     - Kun Zhou et al.
     - EMNLP 2022
     - `Python <https://github.com/RUCAIBox/SimANS>`_
     - Ambiguous negatives sampling

Paper Categories
----------------

By Technique
^^^^^^^^^^^^

**Dynamic Index Refresh**

* ANCE (Lee Xiong et al., ICLR 2021) - Refreshes the ANN index during training to maintain hard negatives

**Cross-Batch and Memory Augmentation**

* RocketQA (Yingqi Qu et al., NAACL 2021) - Uses cross-batch negatives and denoising
* GradCache (Luyu Gao et al., RepL4NLP 2021) - Enables larger batch sizes through gradient caching

**Query-Side Optimization**

* ADORE&STAR (Jingtao Zhan et al., SIGIR 2021) - Fine-tunes query encoder with frozen document encoder

**Curriculum and Distillation**

* TAS-Balanced (Sebastian Hofstätter et al., SIGIR 2021) - Topic-aware sampling with BERT ensemble distillation
* CPR (Hansi Zeng et al., SIGIR 2022) - Curriculum learning for retrieval distillation

**Negative Caching**

* Negative Cache (Erik M. Lindgren et al., NeurIPS 2021) - Maintains a cache of hard negatives

**Multi-Stage Training**

* Multi-stage training (Jing Lu et al., EMNLP 2021) - Progressive training with increasingly hard negatives

By Year
^^^^^^^

**2020**

* DPR (EMNLP 2020) - Foundation of dense retrieval
* RepBERT (Arxiv 2020) - Contextualized embeddings for retrieval
* Learning To Retrieve (Arxiv 2020) - Comprehensive training guide
* Neural Passage Retrieval (Arxiv 2020) - Improved negative contrast

**2021**

* ANCE (ICLR 2021) - Dynamic index refresh
* RocketQA (NAACL 2021) - Cross-batch negatives and denoising
* GradCache (RepL4NLP 2021) - Memory-efficient training
* ADORE&STAR (SIGIR 2021) - Query-side finetuning
* TAS-Balanced (SIGIR 2021) - Topic-aware sampling
* PAIR (EMNLP Findings 2021) - Passage-centric similarity
* Learning from incomplete labels (SIGIR 2021) - Robust training
* Multi-stage training (EMNLP 2021) - Progressive hardness
* Negative Cache (NeurIPS 2021) - Efficient negative storage
* CODER (Arxiv 2021) - Contextual re-ranking

**2022**

* CPR (SIGIR 2022) - Curriculum distillation
* SimANS (EMNLP 2022) - Ambiguous negatives

Implementation Resources
-------------------------

All papers with available code implementations are linked in the table above. Most 
implementations are in Python and leverage popular frameworks like:

* PyTorch
* Hugging Face Transformers
* Sentence Transformers
* FAISS (for ANN search)

For getting started, we recommend beginning with:

1. **DPR** - The foundational paper with clean implementation
2. **ANCE** - For understanding dynamic negative mining
3. **GradCache** - For practical large-batch training on limited hardware

