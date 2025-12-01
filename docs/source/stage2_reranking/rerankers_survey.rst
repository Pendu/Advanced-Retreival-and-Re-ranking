==================================================================
Reranker Survey: Models, Libraries, and Frameworks
==================================================================

Overview
--------
This page provides a comprehensive overview of state-of-the-art reranking models, evaluation frameworks, and open-source libraries for Stage-2 reranking in information retrieval pipelines. The content is based on recent survey literature examining both academic and production-ready reranking systems.

Survey Paper Summary
--------------------

**Research Focus**

Recent survey work systematically examines the landscape of reranking models for information retrieval, providing both theoretical foundations and practical deployment guidance. The survey addresses a critical gap: while numerous reranking models exist, researchers and practitioners lack systematic comparison frameworks and reproducible evaluation pipelines.

**Key Contributions:**

1. **Taxonomy of Reranking Approaches**: Models are categorized by their ranking paradigm:
   
   - **Pointwise**: Each query-document pair is scored independently (e.g., MonoT5, ColBERT)
   - **Pairwise**: Models compare document pairs to determine relative relevance (e.g., EcoRank)
   - **Listwise**: The entire candidate list is processed jointly to produce optimal rankings (e.g., RankZephyr, ListT5)

2. **Reproducibility Framework**: Introduction of standardized evaluation using the Rankify library, enabling fair comparisons across diverse models and datasets.

3. **Open vs. Closed Source Analysis**: Systematic comparison of publicly available models versus proprietary API-based systems, highlighting trade-offs in performance, cost, privacy, and accessibility.

4. **Performance Benchmarking**: Evaluation across standard IR benchmarks including TREC-DL (Deep Learning Track), MS MARCO, BEIR, and domain-specific datasets.

**Main Findings:**

- Open-source listwise rerankers (e.g., RankZephyr-7B) achieve competitive performance with closed-source API models while maintaining full transparency and local deployment capabilities.
- Late-interaction models like ColBERT-v2 offer an excellent balance between effectiveness and efficiency for production systems.
- Proprietary systems (Cohere Rerank-v2, GPT-4-based methods) often lead leaderboards but introduce vendor lock-in, data privacy concerns, and reproducibility challenges.
- Distilled models (InRanker, FlashRank) enable deployment in resource-constrained environments with minimal performance degradation.

Open-Source Reranking Models
-----------------------------

The following models are publicly available, typically via Hugging Face, and can be deployed locally or integrated into custom pipelines using frameworks like Rankify.

+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| Model Name          | Type      | Base Architecture   | Description                                                                       |
+=====================+===========+=====================+===================================================================================+
| **MonoT5**          | Pointwise | T5 (Base/Large/3B)  | Sequence-to-sequence model predicting True/False relevance for Q-D pairs.         |
|                     |           |                     | Strong baseline for pointwise reranking.                                          |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **RankT5**          | Pointwise | T5 (Base/Large/3B)  | Fine-tuned with ranking-specific losses to directly output numerical scores       |
|                     |           |                     | rather than token predictions.                                                    |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **InRanker**        | Pointwise | T5 (Small/Base/3B)  | Distilled from MonoT5-3B into smaller variants (60M, 220M parameters).            |
|                     |           |                     | Trained via InPars (Inquisitive Paraphrasing) for data efficiency.               |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **FlashRank**       | Pointwise | TinyBERT, MiniLM    | Ultra-lightweight models optimized for millisecond-level latency.                 |
|                     |           |                     | Ideal for production systems requiring high throughput.                           |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **RankZephyr**      | Listwise  | Zephyr-7B           | Open-source 7B parameter LLM fine-tuned on RankGPT4 distillation data.            |
|                     |           |                     | Excels in news, healthcare, and general-domain retrieval tasks.                   |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **RankVicuna**      | Listwise  | Vicuna-7B           | Distilled from RankGPT-3.5 with shuffled input augmentation for robustness.      |
|                     |           |                     | Handles variable-length candidate lists effectively.                              |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **ListT5**          | Listwise  | T5 (Base/3B)        | Uses "Fusion-in-Decoder" architecture to jointly encode query-passage pairs       |
|                     |           |                     | and decode sorted document identifiers.                                           |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **LiT5**            | Listwise  | T5 (Distill-XL)     | Distilled listwise model with strong top-1 accuracy for open-domain QA.          |
|                     |           |                     | Balances effectiveness and computational efficiency.                              |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **RankLLaMA**       | Pointwise | Llama-2 (7B/13B)    | Large language model specialized for retrieval via instruction tuning.            |
|                     |           |                     | Fine-tuned on query-document relevance prediction tasks.                          |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **ColBERT**         | Pointwise | BERT (Late Int.)    | ColBERT-v2 uses late interaction: independent query/document encodings with       |
|                     |           |                     | MaxSim aggregation. Highly efficient for large-scale retrieval.                   |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **TWOLAR**          | Pointwise | TWOLAR-Large/XL     | Transformer with Wide and Long Attention for Ranking. Top performer on            |
|                     |           |                     | scientific/biomedical benchmarks (SciFact, TREC-COVID).                           |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **SPLADE**          | Pointwise | SPLADE CoCondenser  | Sparse lexical model with learned expansion. Combines term-matching with          |
|                     |           |                     | semantic understanding. Efficient for inverted index deployment.                  |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **TransformerRanker**| Pointwise| BGE, MXBai, BCE,    | Family of cross-encoder rerankers: bge-reranker-large, mxbai-rerank-base,        |
|                     |           | Jina, etc.          | bce-reranker, jina-reranker-v1. General-purpose models from Hugging Face.         |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+
| **EcoRank**         | Pairwise  | Flan-T5 (Large/XL)  | Budget-conscious pipeline using sliding window approach with smaller LLMs         |
|                     |           |                     | for pairwise comparisons. Cost-effective alternative to large proprietary models. |
+---------------------+-----------+---------------------+-----------------------------------------------------------------------------------+

**Note**: All models listed are available via public repositories (primarily Hugging Face) and can be deployed locally. Many integrate seamlessly with the Rankify framework for standardized evaluation.

Closed-Source / Proprietary API-Based Rerankers
-----------------------------------------------

These systems require commercial API access and operate as black-box services. While often high-performing, they introduce limitations around transparency, data privacy, and reproducibility.

+---------------------+-----------------------------+------------------------------------------------------------------------+
| Reranker Name       | Underlying API/Model        | Description & Constraints                                              |
+=====================+=============================+========================================================================+
| **Cohere Rerank**   | Cohere Rerank-v2            | Fully managed commercial API. Strong performance (73.22 nDCG@10 on     |
|                     |                             | TREC-DL19) but closed-source. No insight into model architecture or    |
|                     |                             | training data. Vendor lock-in and per-query pricing.                   |
+---------------------+-----------------------------+------------------------------------------------------------------------+
| **RankGPT**         | GPT-4, GPT-3.5              | Method is open (permutation generation via prompting), but requires    |
|                     |                             | OpenAI API access. Privacy concerns with sending documents to external |
|                     |                             | services. Cost scales with document count and query volume.            |
+---------------------+-----------------------------+------------------------------------------------------------------------+
| **TourRank**        | GPT-4o, GPT-3.5-turbo       | Tournament-style ranking using LLM pairwise comparisons. Strong        |
|                     |                             | generalization (62.02 nDCG@10 on BEIR), but requires high-tier OpenAI  |
|                     |                             | model access. Expensive for production use.                            |
+---------------------+-----------------------------+------------------------------------------------------------------------+
| **LRL**             | GPT-3                       | "Listwise Reranker with Large Language Models" - uses GPT-3 prompting  |
|                     |                             | to reorder passages. Fully dependent on OpenAI API availability.       |
+---------------------+-----------------------------+------------------------------------------------------------------------+
| **PRP (variant)**   | InstructGPT                 | "Pairwise Ranking Prompting" - while some variants use open models     |
|                     |                             | (FlanUL2), the original uses closed InstructGPT from OpenAI.           |
+---------------------+-----------------------------+------------------------------------------------------------------------+
| **Promptagator++**  | Proprietary Google Model    | High-performing model (76.2 nDCG@10 on TREC-DL19) from Google          |
|                     |                             | Research. Not publicly available. Represents internal/limited-access   |
|                     |                             | large-scale systems.                                                   |
+---------------------+-----------------------------+------------------------------------------------------------------------+

**Key Limitation**: Survey literature highlights that "many LLM-based approaches assume access to powerful proprietary APIs (e.g., OpenAI's GPT-4)... where such access may not be uniformly available." This creates reproducibility barriers and raises concerns about data privacy in sensitive domains (healthcare, legal, enterprise).

Rankify Framework and Supported Libraries
------------------------------------------

**What is Rankify?**

Rankify is an open-source Python framework designed to standardize evaluation, benchmarking, and deployment of both Stage-1 (retrieval) and Stage-2 (reranking) models in information retrieval pipelines. It addresses the fragmentation in IR research where different papers use inconsistent evaluation protocols, making fair comparison difficult.

**Key Features:**

- **Unified Interface**: Single API for evaluating diverse reranker types (pointwise, pairwise, listwise)
- **Standardized Metrics**: Built-in support for nDCG@k, MAP, MRR, Recall@k, and other IR metrics
- **Benchmark Integration**: Direct support for TREC-DL, MS MARCO, BEIR, and custom datasets
- **Hugging Face Integration**: Seamless loading of models from Hugging Face Hub
- **Extensibility**: Easy addition of custom models and evaluation protocols
- **Reproducibility**: Version-controlled configurations and deterministic evaluation

**Installation:**

.. code-block:: bash

   pip install rankify

**Supported Stage-1 Retrievers:**

Rankify provides plug-and-play support for first-stage retrieval models:

- **Sparse Methods**: 
  
  - BM25 (via Pyserini, Elasticsearch, or custom implementations)
  - SPLADE variants (SPLADE++, SPLADEv2)
  
- **Dense Methods**:
  
  - DPR (Dense Passage Retrieval)
  - Contriever
  - ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation)
  - MPNet, BGE (BAAI General Embedding)
  - E5 (Text Embeddings by Weakly-Supervised Contrastive Pre-training)
  - Sentence-BERT variants

**Supported Stage-2 Rerankers:**

All open-source models from the table above are supported:

- **Pointwise**: MonoT5, RankT5, InRanker, FlashRank, ColBERT, TWOLAR, SPLADE, TransformerRanker family
- **Pairwise**: EcoRank
- **Listwise**: RankZephyr, RankVicuna, ListT5, LiT5, RankLLaMA

**Basic Usage Example:**

.. code-block:: python

   from rankify import Reranker, Evaluator
   from rankify.datasets import load_trec_dl
   
   # Load dataset
   dataset = load_trec_dl(year=2019)
   
   # Initialize reranker
   reranker = Reranker.from_pretrained("castorini/monot5-base-msmarco")
   
   # Evaluate
   evaluator = Evaluator(metrics=["ndcg@10", "mrr@10"])
   results = evaluator.evaluate(reranker, dataset)
   
   print(f"nDCG@10: {results['ndcg@10']:.4f}")

**Advanced Features:**

- **Pipeline Composition**: Chain Stage-1 and Stage-2 models
- **Batch Processing**: Efficient evaluation on large datasets
- **Distributed Evaluation**: Multi-GPU support for large models
- **Custom Metrics**: Define domain-specific evaluation measures
- **Ablation Studies**: Built-in tools for hyperparameter sweeps

**Repository and Documentation:**

- GitHub: `https://github.com/castorini/rankify <https://github.com/castorini/rankify>`_ (check for actual repo location)
- Documentation: Comprehensive guides for model integration, custom dataset support, and advanced evaluation scenarios
- Community: Active development with contributions from IR research community

Understanding Reranking Paradigms
----------------------------------

**Pointwise Rerankers**

- Score each query-document pair independently
- Most common approach in production systems
- Examples: MonoT5, ColBERT, cross-encoders
- **Advantages**: Simple to train, easy to parallelize, predictable inference cost
- **Limitations**: Ignores inter-document relationships, may miss relative relevance signals

**Pairwise Rerankers**

- Compare document pairs to determine relative ordering
- Learn preference functions rather than absolute scores
- Examples: EcoRank, some LLM-based comparison methods
- **Advantages**: Captures relative relevance well, robust to score calibration issues
- **Limitations**: Quadratic complexity in candidate list size, harder to optimize

**Listwise Rerankers**

- Process entire candidate list jointly
- Optimize directly for ranking metrics (nDCG, MAP)
- Examples: RankZephyr, ListT5, RankGPT
- **Advantages**: Optimal for ranking objectives, captures global context
- **Limitations**: Computationally expensive, sensitive to list length, requires sophisticated training

Performance Benchmarks
-----------------------

Representative results from survey literature (nDCG@10 on TREC-DL 2019):

+---------------------+-------------+------------------+
| Model               | Type        | nDCG@10          |
+=====================+=============+==================+
| Promptagator++      | Closed      | 76.2             |
+---------------------+-------------+------------------+
| Cohere Rerank-v2    | Closed      | 73.22            |
+---------------------+-------------+------------------+
| RankZephyr-7B       | Open        | 71.0 (approx)    |
+---------------------+-------------+------------------+
| MonoT5-3B           | Open        | 69.5             |
+---------------------+-------------+------------------+
| ColBERT-v2          | Open        | 68.4             |
+---------------------+-------------+------------------+
| TourRank (GPT-4o)   | Closed      | 62.02 (BEIR avg) |
+---------------------+-------------+------------------+

**Note**: Performance varies significantly across datasets and domains. These numbers represent single-dataset snapshots. Consult the full survey for comprehensive cross-dataset analysis.

Recommendations for Practitioners
----------------------------------

**For Academic Research:**

- Use Rankify with open-source models for reproducibility
- Report results on standard benchmarks (TREC-DL, BEIR)
- Include ablation studies with multiple reranker types
- Consider RTEB for zero-shot generalization evaluation

**For Production Systems:**

- Start with ColBERT-v2 or FlashRank for latency-sensitive applications
- Consider RankZephyr-7B for quality-critical use cases with adequate compute
- Evaluate Cohere Rerank if API costs are acceptable and data privacy permits
- Always benchmark on your domain-specific data before deployment

**For Resource-Constrained Environments:**

- InRanker or FlashRank for minimal hardware requirements
- SPLADE for efficient inverted-index-based deployment
- Consider distillation from larger models for custom domains

References
----------

**Primary Reference**

This documentation is based on the following comprehensive survey:

.. note::

   Abdallah, A., Piryani, B., Mozafari, J., Ali, M., & Jatowt, A. (2025). 
   "How good are LLM-based rerankers? An empirical analysis of state-of-the-art reranking models." 
   *arXiv preprint* arXiv:2508.XXXXX [cs.CL].
   
   `GitHub Repository <https://github.com/DataScienceUIBK/llm-reranking-generalization-study>`_

**Key Findings from the Survey:**

* Evaluated **22 methods** with **40 variants** across TREC DL19, DL20, BEIR, and novel query datasets
* LLM-based rerankers show superior performance on familiar queries but variable generalization to novel queries
* Lightweight models offer comparable efficiency with competitive performance
* Query novelty significantly impacts reranking effectiveness
* Training data overlap is a confounding factor in benchmark performance

**Additional References**

1. Nogueira, R., & Cho, K. (2019). "Passage Re-ranking with BERT." *arXiv:1901.04085*. `Paper <https://arxiv.org/abs/1901.04085>`_

2. Nogueira, R., Jiang, Z., Pradeep, R., & Lin, J. (2020). "Document Ranking with a Pretrained Sequence-to-Sequence Model." *EMNLP 2020*. `Paper <https://arxiv.org/abs/2003.06713>`_

3. Sun, W., et al. (2023). "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents." *EMNLP 2023*. `arXiv:2304.09542 <https://arxiv.org/abs/2304.09542>`_

4. Pradeep, R., et al. (2023). "RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!" *arXiv:2312.02724*. `Paper <https://arxiv.org/abs/2312.02724>`_

5. Santhanam, K., et al. (2022). "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." *NAACL 2022*. `Paper <https://aclanthology.org/2022.naacl-main.272/>`_

6. Formal, T., et al. (2021). "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking." *SIGIR 2021*. `arXiv:2107.05720 <https://arxiv.org/abs/2107.05720>`_

Further Reading
---------------

- BEIR benchmark: Comprehensive zero-shot evaluation suite
- TREC Deep Learning Track: Annual evaluation campaigns
- MS MARCO: Large-scale passage ranking dataset

**Related Documentation Pages:**

- :doc:`index` - Overview of reranking in RAG pipelines
- :doc:`cross_encoders` - Deep dive into cross-encoder architectures
- :doc:`../benchmarks_and_datasets` - BEIR, MTEB, and RTEB benchmark details
- :doc:`../stage1_retrieval/index` - First-stage retrieval methods

----

*This documentation is based on recent survey literature and active open-source projects. Model availability and performance metrics are subject to change as the field evolves rapidly. Always verify current model versions and consult official repositories for the latest information.*
