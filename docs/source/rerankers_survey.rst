Reranker Survey: Models, Libraries, and Frameworks
=================================================

Survey Paper Summary
--------------------
The reviewed survey systematically examines state-of-the-art reranking models for information retrieval, highlighting both open-source and proprietary systems, as well as the evaluation pipelines and resources available to researchers.

The survey focuses on practical deployment of rerankers in modern retrieval pipelines, categorizing models by architecture (pointwise, pairwise, listwise), and benchmarking their performance using reproducible methodologies (notably using the Rankify framework). Key points include:
- Comprehensive listing and comparison of retriever and reranker models (both open-source and closed-source/APIs).
- In-depth analysis of evaluation strategies, with a focus on public reproducibility.
- Rankify as an extensible open-source framework enabling quick evaluation, comparison, and deployment of both Stage-1 (retrievers) and Stage-2 (rerankers) models.

Open-Source Reranking Models
----------------------------

+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| Model Name        | Type      | Base           | Description                                                                     |
+===================+===========+================+=================================================================================+
| MonoT5            | Pointwise | T5 (B/L/3B)    | Seq2seq T5 model, predicts True/False relevance for Q-D pairs                    |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| RankT5            | Pointwise | T5 (B/L/3B)    | T5, fine-tuned to output numerical relevance scores                              |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| InRanker          | Pointwise | T5 (S/B/3B)    | Distilled from MonoT5, smaller, efficient model via InPars                       |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| FlashRank         | Pointwise | TinyBERT,MiniLM| Lightweight, low-latency rerankers                                               |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| RankZephyr        | Listwise  | Zephyr-7B      | 7B LLM, fine-tuned for listwise, excels in news/healthcare                       |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| RankVicuna        | Listwise  | Vicuna-7B      | Distilled from RankGPT-3.5 on shuffled lists for robustness                      |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| ListT5            | Listwise  | T5 (B/3B)      | "Fusion in Decoder" listwise reranker                                            |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| LiT5              | Listwise  | T5 (Distill-XL)| Distilled listwise, strong top-1 accuracy for open-domain QA                    |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| RankLLaMA         | Pointwise | Llama-2 (7/13B)| LLM reranker, fine-tuned for IR tasks                                            |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| ColBERT           | Pointwise | BERT (Late Int)| Colbert-v2, uses late interaction technique                                      |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| TWOLAR            | Pointwise | TWOLAR-Large/XL| Very strong in science/medical domains                                           |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| SPLADE            | Pointwise | SPLADE Cocon.  | Sparse lexical+expansion model for reranking                                     |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| TransformerRanker | Pointwise | BGE, MXBai, etc| General transformer rerankers: e.g. bge-reranker, mxbai, BCE, Jina               |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+
| EcoRank           | Pairwise  | Flan-T5 (L/XL) | Budget pipeline; pairwise via window & small LLMs                                |
+-------------------+-----------+----------------+---------------------------------------------------------------------------------+

Note: All open-source models are publicly available (often via Hugging Face) and can be run with the Rankify framework. Proprietary systems (e.g., OpenAI, Cohere) are excluded.

Closed-Source / Proprietary API-Based Rerankers
-----------------------------------------------

+-------------------+-----------------------------+------------------------------------------------------------------------+
| Reranker Name     | Underlying API/Model        | Description & Constraints                                               |
+===================+=============================+========================================================================+
| Cohere Rerank     | Cohere Rerank-v2            | Fully managed API, high performer, but closed; black-box system         |
+-------------------+-----------------------------+------------------------------------------------------------------------+
| RankGPT           | GPT-4, GPT-3.5              | Method open, relies on OpenAI APIs, privacy & reproducibility limits    |
+-------------------+-----------------------------+------------------------------------------------------------------------+
| TourRank          | GPT-4o, GPT-3.5-turbo       | Tournament-style, needs OpenAI access, excels at generalization         |
+-------------------+-----------------------------+------------------------------------------------------------------------+
| LRL               | GPT-3                       | Listwise reranker by prompting, but closed API                          |
+-------------------+-----------------------------+------------------------------------------------------------------------+
| PRP (variant)     | InstructGPT                 | Pairwise ranking prompting; OpenAI models closed                        |
+-------------------+-----------------------------+------------------------------------------------------------------------+
| Promptagator++    | Proprietary Google Model    | Google internal/limited model, strong on benchmarks, closed to public   |
+-------------------+-----------------------------+------------------------------------------------------------------------+

Note: These models require access to proprietary commercial APIs, limiting transparency and modifiability for many users.

Rankify Framework and Supported Libraries
-----------------------------------------
Rankify is an open-source Python library for quick evaluation, benchmarking, and deployment of both Stage-1 (retrievers) and Stage-2 (rerankers) models in the information retrieval pipeline. The framework offers:
- Plug-and-play interface for integrating open-source rerankers and retrievers
- Standardized evaluation pipelines and metrics (e.g., nDCG@10)
- Support for both pointwise and listwise aggregation
- Compatibility with Hugging Face models and datasets

**Stage-1 retrievers available:**
  - BM25 (sparse well-supported)
  - DPR, Contriever, MPNet, BGE, SPLADE, and others (dense)

**Stage-2 rerankers available:**
  - All open-source models above (MonoT5, RankT5, FlashRank, ColBERT, etc.)
  - Listwise rerankers such as RankZephyr, ListT5, LiT5

The coherent integration of open and closed-source models within Rankify allows for fair benchmarking and ablation studies across a wide research landscape.

References:
 - Survey paper analyzed and Rankify documentation
