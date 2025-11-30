# Advanced Retrieval and Re-ranking

A curated collection of research papers on dense retrieval, negative sampling strategies, and re-ranking techniques for information retrieval and question answering systems.

## Table of Contents
- [Overview](#overview)
- [The Hard Negative Problem](#the-hard-negative-problem)
- [Papers](#papers)
- [Contributing](#contributing)

---

## Overview

### The New Bottleneck: Why Advanced Negative Mining is Non-Negotiable in Dense Retrieval

#### The Paradigm Shift from Sparse to Dense Retrieval

The field of information retrieval has undergone a fundamental paradigm shift. For decades, retrieval systems were dominated by sparse, lexical-based methods like BM25. These approaches, while robust and efficient, are limited by their reliance on exact keyword matching. They struggle to capture the underlying semantic intent of a query, failing when users employ different terminology (the "vocabulary mismatch" problem).

The advent of pre-trained language models (PLMs) such as BERT introduced the era of dense retrieval. Instead of sparse vectors of word counts, dense retrievers map queries and documents into low-dimensional, continuous-valued vectors (embeddings). These embeddings capture semantic relevance, allowing a model to retrieve documents that are contextually related to a query, even if they share no keywords.

However, this power comes with a critical dependency. The performance of these dense models is not just a function of their architecture (e.g., BERT, Sentence Transformers) but is overwhelmingly reliant on the quality of the data used during their contrastive or multi-negative training. The central challenge has shifted from lexical matching to a new, more difficult problem: **teaching the model to distinguish between genuine semantic relevance and mere semantic similarity**.

---

## The Hard Negative Problem

### Defining the "Hard Negative" Challenge

In the context of dense retrieval, training data is typically structured as triplets: a query (anchor), a relevant document (positive), and an irrelevant document (negative). The model's goal, often optimized via a contrastive loss, is to pull the (query, positive) pair together in the embedding space while pushing the (query, negative) pair apart.

The choice of this negative sample is arguably the most critical factor in a model's final performance. We can categorize negatives as follows:

- **Easy Negatives**: These are random documents from the corpus. They are typically semantically and lexically unrelated to the query. Models learn to distinguish these very quickly, and they provide diminishing gradients early in training.

- **Hard Negatives**: These are the critical samples. A hard negative is a document that is semantically similar to the query—and thus likely to be highly-ranked by an untrained model—but is contextually or factually irrelevant. For example, for the query "What is the capital of France?", a hard negative might be "Best tourist attractions in Paris" or "The economy of France."

- **False Negatives**: This is a damaging artifact of the mining process. A false negative is a document that is relevant to the query but was not present in the original (query, positive)-pair labels. If the model is trained to push this "negative" away from the query, it is actively penalized for making a correct semantic connection, leading to a confused and suboptimal embedding space.

**The central thesis of modern dense retrieval research** is that the quality, difficulty, and "cleanness" (i.e., a lack of false negatives) of the negative training set is the primary determinant of model performance.

### The Failure of Baseline Strategies

Initial forays into dense retrieval training exposed the inadequacy of simple negative sampling strategies, which now serve as baselines against which advanced techniques are measured.

#### In-Batch Negatives

The most common and efficient baseline is "in-batch" negative sampling, famously implemented in the sentence-transformers library as `MultipleNegativesRankingLoss` (MNRL). In this strategy, a batch of (query, positive) pairs is processed. For a given query q<sub>i</sub>, its corresponding p<sub>i</sub> is the positive, and all other positive documents p<sub>j</sub> (where j ≠ i) within the same batch are used as negatives.

This method, while computationally convenient, suffers from two critical flaws:
1. It has a high probability of introducing false negatives. If a batch contains two semantically related queries, one query's positive document will be used as a hard false negative for the other, actively teaching the model the wrong signal.
2. In a sufficiently large and diverse corpus, the vast majority of in-batch negatives are "easy" and uninformative. This leads to diminishing gradient norms, slow convergence, and a model that is not challenged to learn the fine-grained distinctions necessary for high-quality retrieval.

#### Static Negatives (e.g., BM25)

The original Dense Passage Retrieval (DPR) paper proposed using a static, pre-mined set of hard negatives. These were generated by retrieving the top-k documents using BM25 that did not contain the answer. This was an improvement, as it forced the model to learn beyond simple lexical overlap. However, this set is static. As the neural model trains, it quickly learns to defeat these "stale" BM25-mined negatives. The negatives are no longer "hard" for the current model state, and training plateaus.

### The Arms Race: Co-Evolution of Retriever and Sampler

This failure of baseline methods reveals a deeper pattern: **the entire field of advanced negative mining can be understood as a co-evolutionary "arms race" between the retriever (the student model being trained) and the sampler (the mechanism for finding negatives).**

A model trained on random negatives is easily fooled by BM25 negatives. A model trained on BM25 negatives is then easily fooled by semantically similar negatives. This necessitates a "harder" sampler, such as the model's own previous checkpoint. This iterative process, where the sampler and retriever continuously sharpen each other, defines the frontier of the field.

The following papers explore various implementations and strategies in this ongoing arms race.

---

## Papers

### Dense Retrieval and Negative Selection Methods

| Paper | Author | Venue | Code | Key Contribution |
|-------|--------|-------|------|------------------|
| [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) | Vladimir Karpukhin et al. | EMNLP 2020 | [Python](https://github.com/facebookresearch/DPR) | DPR with in-batch negatives |
| [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval](https://arxiv.org/abs/2006.15498) | Jingtao Zhan et al. | Arxiv 2020 | [Python](https://github.com/jingtaozhan/RepBERT-Index) | RepBERT architecture |
| [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/abs/2007.00808) | Lee Xiong et al. | ICLR 2021 | [Python](https://github.com/microsoft/ANCE) | ANCE - refresh index during training |
| [Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently](https://arxiv.org/abs/2010.10469) | Jingtao Zhan et al. | Arxiv 2020 | NA | Comprehensive training strategies |
| [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2010.08191) | Yingqi Qu et al. | NAACL 2021 | [Python](https://github.com/PaddlePaddle/RocketQA) | Cross-batch negatives, denoise hard negatives and data augmentation |
| [Neural Passage Retrieval with Improved Negative Contrast](https://arxiv.org/abs/2010.12523) | Jing Lu et al. | Arxiv 2020 | NA | Improved negative contrast |
| [Scaling deep contrastive learning batch size under memory limited setup](https://arxiv.org/abs/2101.06983) | Luyu Gao et al. | RepL4NLP 2021 | [Python](https://github.com/luyug/GradCache) | GradCache for memory-efficient training |
| [Optimizing Dense Retrieval Model Training with Hard Negatives](https://arxiv.org/abs/2104.08051) | Jingtao Zhan et al. | SIGIR 2021 | [Python](https://github.com/jingtaozhan/DRhard) | ADORE&STAR - query-side finetuning on pretrained document encoders |
| [Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling](https://arxiv.org/abs/2104.06967) | Sebastian Hofstätter et al. | SIGIR 2021 | [Python](https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval) | TAS-Balanced - sample from query cluster and distill from BERT ensemble |
| [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval](https://arxiv.org/abs/2108.06027) | Ruiyang Ren et al. | EMNLP Findings 2021 | [Python](https://github.com/DaoD/PAIR) | Passage-centric similarity relations |
| [Learning robust dense retrieval models from incomplete relevance labels](https://dl.acm.org/doi/10.1145/3404835.3463106) | Prafull Prakash et al. | SIGIR 2021 | [Python](https://github.com/thakur-nandan/income) | Training with incomplete labels |
| [Multi-stage training with improved negative contrast for neural passage retrieval](https://aclanthology.org/2021.emnlp-main.492/) | Jing Lu et al. | EMNLP 2021 | NA | Multi-stage training approach |
| [Efficient Training of Retrieval Models Using Negative Cache](https://papers.nips.cc/paper/2021/hash/2175f8c5cd9604f6b1e576b252d4c86e-Abstract.html) | Erik M. Lindgren et al. | NeurIPS 2021 | [Python](https://github.com/google-research/google-research/tree/master/negative_cache) | Negative cache mechanism |
| [CODER: An efficient framework for improving retrieval through COntextual Document Embedding Reranking](https://arxiv.org/abs/2112.08766) | George Zerveas et al. | Arxiv 2021 | NA | Contextual document embedding re-ranking |
| [Curriculum Learning for Dense Retrieval Distillation](https://arxiv.org/abs/2204.13679) | Hansi Zeng et al. | SIGIR 2022 | [Python](https://github.com/hansizeng/CPR) | Curriculum-based distillation |
| [SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval](https://arxiv.org/abs/2210.11773) | Kun Zhou et al. | EMNLP 2022 | [Python](https://github.com/RUCAIBox/SimANS) | Ambiguous negatives sampling |

---

## Contributing

Feel free to open issues or submit pull requests if you'd like to add more papers or suggest improvements to this repository.

## License

This repository is for educational and research purposes.

