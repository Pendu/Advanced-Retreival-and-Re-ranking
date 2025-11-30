

Benchmarks and Open Source Datasets for Retrieval and Re-ranking
==============================================================

1. BEIR and MTEB Benchmark Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**BEIR (Benchmarking Information Retrieval)**

BEIR is a heterogeneous zero-shot evaluation benchmark for information retrieval. The benchmark comprises 18 diverse datasets spanning multiple domains and tasks:

**Key Datasets:**
- MS MARCO: Large-scale web passage ranking (8.8M passages)
- Natural Questions (NQ): Google search queries with Wikipedia annotations
- FiQA-2018: Financial question answering (57,638 documents)
- SciFact: Scientific fact verification (5,183 documents)
- TREC-COVID: Biomedical retrieval for COVID-19 research (171,332 documents)
- HotpotQA: Multi-hop question answering (5.2M documents)
- Climate-FEVER: Climate change fact-checking
- NFCorpus: Biomedical information retrieval
- DBPedia: Entity retrieval from Wikipedia (4.6M documents)
- ArguAna: Argument retrieval (8,674 documents)
- Quora: Duplicate question detection
- FEVER: Fact verification (5.4M documents)
- Touché-2020: Argument retrieval (528,155 documents)
- CQADupStack: Community question answering across multiple domains

**Tasks Evaluated**
- Retrieval: Finding relevant documents for queries
- Fact-checking: Verifying claims against evidence
- Question answering: Open-domain and domain-specific QA
- Argument retrieval: Finding argumentative passages
- Duplicate detection: Identifying semantically similar queries
- Citation prediction: Scientific document retrieval
- Entity retrieval: Finding entity-related documents
- Biomedical/scientific retrieval: Domain-specific search

**Evaluation Metrics:**
BEIR primarily uses nDCG@10 as the main metric, along with Recall@k, MAP, and MRR.

**MTEB (Massive Text Embedding Benchmark)**

MTEB provides comprehensive evaluation across 8 embedding tasks covering 58 datasets and 112+ languages:

**Core Tasks:**
- Classification: Text categorization with accuracy/F1 metrics
- Clustering: Document grouping using v-measure
- Pair Classification: Binary relationship prediction with Average Precision
- Reranking: Document reordering using MAP/MRR@k
- Retrieval: Information retrieval with nDCG@10 (includes 15 BEIR datasets)
- Semantic Textual Similarity (STS): Correlation-based similarity scoring
- Summarization: Summary quality evaluation
- Bitext Mining: Cross-lingual sentence pair identification

**Key Features:**
- Standardized evaluation protocols across diverse tasks
- Zero-shot evaluation capabilities
- Multilingual support (1,000+ languages in MMTEB extension)
- Public leaderboard for model comparison
- Integration with 15 BEIR retrieval datasets

2. Is the RTEB Leaderboard for Dense Retrieval and Hard Negative Mining Methods?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes and No. 
While the RTEB (Retrieval Embedding Benchmark) is a highly relevant leaderboard for dense retrieval research, it does not specifically benchmark hard negative mining "methods" but rather the models that result from these techniques.

**1. Is it for Dense Retrieval?**

Yes. RTEB is designed specifically to evaluate dense embedding models. If your research produces a dense embedding model (e.g., a BERT-based bi-encoder), this is where such models are ranked. The metric used is nDCG@10, which is the gold standard for dense retrieval evaluation.

**2. Is it for Hard Negative Mining?**

Indirectly, but crucially. There is no "Hard Negative Mining Leaderboard" because hard negative mining is a training technique, not a downstream task. However, RTEB is arguably the best current benchmark to prove the efficacy of a new hard negative mining strategy because of its focus on Generalization.
- **Limitation of Old Benchmarks (MTEB/BEIR):** Standard benchmarks use public datasets (like MS MARCO). Models trained with standard negative mining can "overfit" to the specific negatives in those datasets.
- **The RTEB Advantage:** RTEB uses private, unseen datasets. High performance on RTEB indicates strong generalization and conceptual learning—key for hard negative mining.

**Summary Comparison Table**

+-----------------------------+---------------------+-------------------------+------------------------------------------+
| Feature                     | MTEB (Standard)     | RTEB (Beta)            | Relevance to Research                    |
+=============================+=====================+=========================+==========================================+
| Datasets                    | Public (MS MARCO,   | Private & Public        | RTEB is better for proving method works  |
|                             | NQ, etc.)           |                         | on unseen data.                          |
+-----------------------------+---------------------+-------------------------+------------------------------------------+
| Focus                       | Broad (Classification, Clustering, Retrieval)| Retrieval Only | RTEB is strictly focused on retrieval.    |
+-----------------------------+---------------------+-------------------------+------------------------------------------+
| Evaluation                  | Offline scripts     | Server-side (submit)    | RTEB prevents test-set leakage, ensures  |
|                             | (run locally)       |                         | credibility.                             |
+-----------------------------+---------------------+-------------------------+------------------------------------------+
| Main Metric                 | Avg of 56 tasks     | nDCG@10                 | RTEB aligns with standard retrieval      |
|                             |                     |                         | metrics.                                 |
+-----------------------------+---------------------+-------------------------+------------------------------------------+

**Recommendation**
- Use BEIR/MTEB for Training/Validation: Leverage standard public datasets for development/ablation studies.
- Use RTEB for Leaderboard Submission: Submit your final model to RTEB for robust, credible zero-shot generalization results.

**References:**
- [HuggingFace MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard?benchmark_name=RTEB%28beta%29)
