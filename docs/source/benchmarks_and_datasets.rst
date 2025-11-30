Benchmarks and Datasets for Retrieval and Re-ranking
=====================================================

This section provides a comprehensive overview of evaluation benchmarks, datasets, and metrics 
used in dense retrieval and re-ranking research. Understanding these resources is essential for 
rigorous experimental design and fair comparison of methods.

.. contents:: Table of Contents
   :local:
   :depth: 2

1. Evaluation Paradigms
-----------------------

1.1 In-Domain vs. Zero-Shot Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**In-Domain Evaluation**

Models are trained and evaluated on the same dataset (with train/dev/test splits). This measures 
how well a model learns the specific characteristics of a dataset.

* **Advantage**: High performance achievable through dataset-specific optimization
* **Limitation**: Does not measure generalization; models may overfit to annotation artifacts
* **Example**: Training on MS MARCO train set, evaluating on MS MARCO dev set

**Zero-Shot Evaluation**

Models are trained on one dataset and evaluated on completely different datasets without any 
fine-tuning. This measures true generalization capability.

* **Advantage**: Tests whether models learn transferable semantic representations
* **Limitation**: Performance ceiling is lower; domain mismatch can be severe
* **Example**: Training on MS MARCO, evaluating on BEIR (18 diverse datasets)

.. important::

   **For hard negative mining research**, zero-shot evaluation is critical. A mining strategy 
   that improves in-domain performance but hurts zero-shot performance has likely introduced 
   dataset-specific biases rather than improving semantic understanding.

1.2 Retrieval vs. Re-ranking Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Retrieval Evaluation (Stage 1)**

Measures the ability to find relevant documents from a large corpus.

* **Task**: Given query q, retrieve top-k documents from corpus C (|C| = millions)
* **Key Metrics**: Recall@k, MRR@k, nDCG@k
* **Focus**: High recall (don't miss relevant documents)

**Re-ranking Evaluation (Stage 2)**

Measures the ability to precisely order a small candidate set.

* **Task**: Given query q and candidates D (|D| = 100-1000), produce optimal ranking
* **Key Metrics**: nDCG@10, MAP, P@k
* **Focus**: High precision at top positions

2. Primary Benchmarks
---------------------

2.1 MS MARCO (Microsoft Machine Reading Comprehension)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Overview**

The de facto standard benchmark for passage retrieval, derived from Bing search logs.

.. list-table:: MS MARCO Statistics
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Value
   * - Corpus Size
     - 8,841,823 passages
   * - Training Queries
     - 502,939 (with ~1 relevant passage each)
   * - Dev Queries
     - 6,980 (official small dev set)
   * - Avg. Query Length
     - 5.96 words
   * - Avg. Passage Length
     - 55.98 words
   * - Annotation Type
     - Sparse (typically 1 positive per query)

**Characteristics**

* **Sparse Annotations**: Most queries have only 1 labeled positive passage, despite multiple 
  relevant passages existing in the corpus. This creates a significant **false negative problem**.
* **Web Domain**: Queries reflect real user search behavior (navigational, informational, transactional)
* **Answer Extraction**: Original task was extractive QA; passages contain answer spans

**Evaluation Protocol**

* **MRR@10**: Primary metric (Mean Reciprocal Rank of first relevant result in top-10)
* **Recall@1000**: Secondary metric for retrieval (measures candidate generation quality)

**Known Limitations**

1. **Incomplete Labels**: Estimated 30-50% of top-retrieved passages are unlabeled positives
2. **Position Bias**: Annotators saw BM25-ranked results, biasing labels toward lexical matches
3. **Single Positive**: Contrastive learning with only 1 positive limits training signal

**Implications for Hard Negative Mining**

The sparse annotation makes MS MARCO particularly challenging for hard negative mining:

* Mining from top BM25/dense results has high false negative risk
* Cross-encoder denoising is essential (RocketQA detected ~70% false negatives)
* Models trained on MS MARCO may learn to avoid semantically similar passages

2.2 Natural Questions (NQ)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Overview**

Google's open-domain question answering dataset based on real search queries.

.. list-table:: Natural Questions Statistics
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Value
   * - Corpus
     - Wikipedia (21M passages in DPR split)
   * - Training Queries
     - 79,168
   * - Dev Queries
     - 8,757
   * - Test Queries
     - 3,610
   * - Avg. Query Length
     - 9.2 words
   * - Annotation Type
     - Dense (multiple annotators, long/short answers)

**Characteristics**

* **Natural Queries**: Real Google search queries (more natural than synthetic)
* **Wikipedia Corpus**: Well-structured, factual content
* **Multiple Answer Types**: Short answer spans + long answer paragraphs
* **Higher Quality Labels**: Multiple annotators reduce noise

**Evaluation Protocol**

* **Top-k Accuracy**: Whether any of top-k retrieved passages contains the answer
* **Recall@k**: Standard retrieval recall
* **Exact Match (EM)**: For end-to-end QA evaluation

2.3 BEIR (Benchmarking IR)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Overview**

The gold standard for zero-shot retrieval evaluation, comprising 18 diverse datasets.

.. list-table:: BEIR Dataset Summary
   :header-rows: 1
   :widths: 20 15 15 50

   * - Dataset
     - Domain
     - Corpus Size
     - Task Description
   * - MS MARCO
     - Web
     - 8.8M
     - Passage retrieval from Bing
   * - TREC-COVID
     - Biomedical
     - 171K
     - COVID-19 scientific literature
   * - NFCorpus
     - Biomedical
     - 3.6K
     - Nutrition and medical
   * - NQ
     - Wikipedia
     - 2.7M
     - Open-domain QA
   * - HotpotQA
     - Wikipedia
     - 5.2M
     - Multi-hop reasoning
   * - FiQA
     - Finance
     - 57K
     - Financial QA
   * - ArguAna
     - Misc
     - 8.7K
     - Argument retrieval
   * - Touché-2020
     - Misc
     - 382K
     - Argument retrieval (web)
   * - CQADupStack
     - StackExchange
     - 457K
     - Duplicate question detection
   * - Quora
     - Web
     - 523K
     - Duplicate question pairs
   * - DBPedia
     - Wikipedia
     - 4.6M
     - Entity retrieval
   * - SCIDOCS
     - Scientific
     - 25K
     - Citation prediction
   * - FEVER
     - Wikipedia
     - 5.4M
     - Fact verification
   * - Climate-FEVER
     - Scientific
     - 5.4M
     - Climate fact-checking
   * - SciFact
     - Scientific
     - 5.2K
     - Scientific claim verification

**Evaluation Protocol**

* **Primary Metric**: nDCG@10 (position-weighted relevance)
* **Aggregation**: Average nDCG@10 across all 18 datasets
* **No Fine-tuning**: Models must be evaluated without dataset-specific training

**Why BEIR Matters for Hard Negative Mining**

BEIR tests whether negative mining strategies produce generalizable representations:

1. **Domain Diversity**: Finance, biomedical, legal, scientific, web
2. **Task Diversity**: QA, fact-checking, argument retrieval, duplicate detection
3. **Corpus Size Variation**: 3.6K to 8.8M documents
4. **Query Style Variation**: Keywords, questions, claims, arguments

.. note::

   **Key Finding**: Models with aggressive hard negative mining (e.g., ANCE with frequent refresh) 
   sometimes show strong MS MARCO performance but weaker BEIR generalization. This suggests 
   overfitting to MS MARCO's specific negative distribution.

2.4 MTEB (Massive Text Embedding Benchmark)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Overview**

Comprehensive benchmark covering 8 embedding tasks across 58+ datasets.

.. list-table:: MTEB Task Categories
   :header-rows: 1
   :widths: 25 25 50

   * - Task
     - Metric
     - Description
   * - Retrieval
     - nDCG@10
     - Information retrieval (includes BEIR)
   * - Reranking
     - MAP
     - Re-ordering candidate documents
   * - Classification
     - Accuracy
     - Text categorization
   * - Clustering
     - V-measure
     - Document grouping
   * - Pair Classification
     - AP
     - Binary relationship prediction
   * - STS
     - Spearman ρ
     - Semantic textual similarity
   * - Summarization
     - Spearman ρ
     - Summary quality scoring
   * - Bitext Mining
     - F1
     - Cross-lingual alignment

**Relevance to Dense Retrieval Research**

MTEB's retrieval subset includes 15 BEIR datasets, making it the standard for embedding model 
comparison. The leaderboard at HuggingFace provides:

* Standardized evaluation protocols
* Fair comparison across models
* Task-specific and aggregate rankings

2.5 RTEB (Retrieval Text Embedding Benchmark)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Overview**

A newer benchmark specifically designed for retrieval, with private test sets to prevent 
data contamination.

**Key Differentiators from MTEB**

.. list-table:: MTEB vs. RTEB Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - MTEB
     - RTEB
   * - Test Sets
     - Public (risk of contamination)
     - Private + Public
   * - Evaluation
     - Local scripts
     - Server-side submission
   * - Focus
     - 8 embedding tasks
     - Retrieval only
   * - Metric
     - Task-specific averages
     - nDCG@10
   * - Leakage Risk
     - High (test sets in training data)
     - Low (private evaluation)

**Implications for Research**

* **For Development**: Use MTEB/BEIR for ablation studies and hyperparameter tuning
* **For Publication**: Submit to RTEB for credible zero-shot generalization claims
* **For Hard Negative Mining**: RTEB's private sets better test true generalization

3. Evaluation Metrics
---------------------

3.1 Retrieval Metrics (Stage 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Recall@k**

Measures the fraction of relevant documents retrieved in the top-k results.

.. math::

   \text{Recall@k} = \frac{|\text{Relevant} \cap \text{Retrieved@k}|}{|\text{Relevant}|}

* **Use Case**: Evaluating candidate generation for re-ranking pipeline
* **Typical Values**: Recall@100 = 0.85-0.95 for strong retrievers on MS MARCO
* **Interpretation**: Higher is better; measures "not missing" relevant documents

**MRR@k (Mean Reciprocal Rank)**

Average of reciprocal ranks of the first relevant document.

.. math::

   \text{MRR@k} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}

where rank_i is the position of the first relevant document for query i (0 if not in top-k).

* **Use Case**: When users care about the first relevant result
* **Typical Values**: MRR@10 = 0.35-0.42 for strong retrievers on MS MARCO
* **Interpretation**: Emphasizes top positions; penalizes late appearances

**nDCG@k (Normalized Discounted Cumulative Gain)**

Position-weighted relevance score, normalized by ideal ranking.

.. math::

   \text{DCG@k} = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}

.. math::

   \text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}

* **Use Case**: When relevance is graded (not binary) or position matters
* **Typical Values**: nDCG@10 = 0.45-0.55 on BEIR average
* **Interpretation**: Accounts for both relevance grade and position

3.2 Re-ranking Metrics (Stage 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**MAP (Mean Average Precision)**

Average precision across all recall levels.

.. math::

   \text{AP} = \frac{1}{|\text{Relevant}|} \sum_{k=1}^{n} P(k) \cdot \text{rel}(k)

where P(k) is precision at position k, and rel(k) is 1 if document at k is relevant.

* **Use Case**: Evaluating full ranking quality
* **Typical Values**: MAP = 0.30-0.45 for strong re-rankers

**P@k (Precision at k)**

Fraction of top-k results that are relevant.

.. math::

   \text{P@k} = \frac{|\text{Relevant} \cap \text{Top-k}|}{k}

* **Use Case**: When only top-k results are shown to users
* **Typical Values**: P@10 = 0.60-0.80 for strong re-rankers

3.3 Metric Selection Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: When to Use Each Metric
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Best For
     - Limitations
   * - Recall@k
     - Stage 1 retrieval, candidate generation
     - Ignores ranking quality within top-k
   * - MRR@k
     - Single-answer tasks (QA, fact-checking)
     - Only considers first relevant result
   * - nDCG@k
     - Graded relevance, general ranking
     - Requires relevance grades; complex
   * - MAP
     - Full ranking evaluation
     - Sensitive to number of relevant docs
   * - P@k
     - User-facing top-k display
     - Ignores ranking within top-k

4. Dataset Selection for Research
---------------------------------

4.1 For Hard Negative Mining Research
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Recommended Primary Dataset**: MS MARCO

* Large scale enables meaningful negative mining
* Sparse annotations make false negative handling critical
* Standard benchmark allows comparison with prior work

**Recommended Zero-Shot Evaluation**: BEIR (full 18 datasets)

* Tests generalization across domains
* Reveals overfitting to MS MARCO negatives
* Standard for publication

**Recommended Ablation Datasets**:

* **NQ**: Higher quality labels, different domain
* **FiQA**: Domain shift (finance)
* **SciFact**: Fact verification (different task)

4.2 For Re-ranking Research
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Recommended Datasets**:

* **MS MARCO Passage Re-ranking**: Standard benchmark
* **TREC Deep Learning Track**: Higher quality judgments (graded relevance)
* **BEIR Re-ranking Subset**: Zero-shot evaluation

4.3 Dataset Pitfalls to Avoid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Training on Test Data**: Ensure no overlap between training corpus and evaluation sets
2. **Annotation Artifacts**: Be aware of biases in how datasets were annotated
3. **Corpus Contamination**: Check if your pre-trained model was trained on evaluation corpora
4. **Metric Gaming**: Don't optimize for a single metric at the expense of others

5. Practical Recommendations
----------------------------

5.1 Experimental Design
^^^^^^^^^^^^^^^^^^^^^^^

**For a Hard Negative Mining Paper**:

1. **In-Domain Baseline**: Report MS MARCO MRR@10 and Recall@1000
2. **Zero-Shot Generalization**: Report BEIR nDCG@10 (average and per-dataset)
3. **Ablation Studies**: Use NQ or FiQA for faster iteration
4. **Statistical Significance**: Report variance across multiple runs (3-5 seeds)

**For a Re-ranking Paper**:

1. **Primary Metrics**: nDCG@10, MAP, MRR@10
2. **Latency Analysis**: Report inference time per query
3. **Candidate Set Size**: Evaluate across k ∈ {10, 50, 100, 200}

5.2 Reporting Standards
^^^^^^^^^^^^^^^^^^^^^^^

Following community standards (from BEIR, MTEB papers):

* Report mean and standard deviation across runs
* Include per-dataset results for BEIR (not just average)
* Specify model size, training data, and compute budget
* Provide code and model checkpoints for reproducibility

References
----------

1. Thakur et al. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." *NeurIPS 2021*. `arXiv:2104.08663 <https://arxiv.org/abs/2104.08663>`_

2. Muennighoff et al. "MTEB: Massive Text Embedding Benchmark." *EACL 2023*. `arXiv:2210.07316 <https://arxiv.org/abs/2210.07316>`_

3. Nguyen et al. "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset." *NeurIPS 2016 Workshop*. `Paper <https://arxiv.org/abs/1611.09268>`_

4. Kwiatkowski et al. "Natural Questions: A Benchmark for Question Answering Research." *TACL 2019*. `Paper <https://aclanthology.org/Q19-1026/>`_

5. Craswell et al. "Overview of the TREC 2019 Deep Learning Track." *TREC 2019*. `Paper <https://trec.nist.gov/pubs/trec28/papers/OVERVIEW.DL.pdf>`_

6. HuggingFace MTEB Leaderboard. `Link <https://huggingface.co/spaces/mteb/leaderboard>`_
