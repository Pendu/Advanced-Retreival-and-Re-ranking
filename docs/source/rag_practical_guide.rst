Building RAG Pipelines: A Practical Guide
==========================================

This guide presents practical patterns for building Retrieval-Augmented Generation (RAG) 
pipelines, progressing from minimal viable implementations to production-ready systems. 
The content is inspired by Ben Clavié's `"Beyond the Basics of RAG" 
<https://parlance-labs.com/education/rag/ben.html>`_ talk at the Mastering LLMs Conference 
(`video <https://youtu.be/0nA5QG3087g>`_) and reflects real-world best practices.

.. contents:: Table of Contents
   :local:
   :depth: 2

.. note::

   **Key Insight from Ben Clavié (Answer.AI):**
   
   *"RAG is not a new paradigm, a framework, or an end-to-end system. RAG is the act of 
   stitching together Retrieval and Generation to ground the latter. Good RAG is made 
   up of good components: good retrieval pipeline, good generative model, good way of 
   linking them up."*

Video: Beyond the Basics of RAG
-------------------------------

Watch Ben Clavié's full talk from the Mastering LLMs Conference:

.. raw:: html

   <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 20px;">
     <iframe 
       src="https://www.youtube.com/embed/0nA5QG3087g" 
       style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
       frameborder="0" 
       allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
       allowfullscreen>
     </iframe>
   </div>

Slides
------

Download the presentation slides: :download:`Beyond the Basics of RAG (PDF) <ben_claive_beyond_basics_of_rag_talk.pdf>`

.. raw:: html

   <div style="margin-bottom: 20px;">
     <iframe 
       src="_static/ben_claive_beyond_basics_of_rag_talk.pdf" 
       width="100%" 
       height="600px" 
       style="border: 1px solid #ccc;">
     </iframe>
     <p style="font-size: 0.9em; color: #666;">
       If the PDF doesn't display, <a href="_static/ben_claive_beyond_basics_of_rag_talk.pdf" target="_blank">click here to open it directly</a>.
     </p>
   </div>

The Compact MVP: Start Simple
-----------------------------

The most minimal deep retrieval pipeline is surprisingly simple. Before reaching for 
complex architectures, start here.

Minimal Implementation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   import numpy as np
   
   # Load embedding model
   model = SentenceTransformer("BAAI/bge-base-en-v1.5")
   
   # Embed your documents (do this once, store the results)
   documents = ["Document 1 text...", "Document 2 text...", ...]
   doc_embeddings = model.encode(documents, normalize_embeddings=True)
   
   # At query time: embed query and find similar documents
   query = "What is the capital of France?"
   query_embedding = model.encode(query, normalize_embeddings=True)
   
   # Compute similarities (this IS your "vector database" at small scale)
   similarities = np.dot(doc_embeddings, query_embedding.T)
   top_k_indices = np.argsort(similarities)[-3:][::-1]
   
   results = [documents[i] for i in top_k_indices]

**That's it.** This works for thousands of documents on any modern CPU.

When Do You Need a Vector Database?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important::

   **You don't need a vector database for small-scale search.**
   
   A numpy array IS your vector database at small scale. Any modern CPU can search 
   through hundreds of vectors in milliseconds.

Vector databases (FAISS, Milvus, Pinecone, etc.) become necessary when:

* **Scale**: > 100K documents (need approximate nearest neighbor search)
* **Persistence**: Need to store and reload indexes
* **Filtering**: Need metadata-based pre-filtering
* **Updates**: Frequent document additions/deletions
* **Distribution**: Multi-node deployment

.. list-table:: When to Use What
   :header-rows: 1
   :widths: 25 25 50

   * - Scale
     - Solution
     - Rationale
   * - < 10K docs
     - NumPy array
     - Brute force is fast enough (~10ms)
   * - 10K - 100K docs
     - FAISS (flat or IVF)
     - Need some optimization
   * - 100K - 10M docs
     - FAISS HNSW / Milvus
     - Need ANN for sub-second latency
   * - > 10M docs
     - Distributed (Milvus, Pinecone)
     - Need sharding and replication

Why Bi-Encoders Work (and When They Don't)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bi-encoders encode queries and documents **entirely separately**. They are unaware 
of each other until the similarity computation.

**Advantages:**

* Pre-compute all document embeddings (offline)
* Only encode the query at inference time
* Extremely fast retrieval via ANN indexes

**Limitations:**

* Compressing hundreds of tokens to a single vector loses information
* Training data never fully represents your domain
* Humans use keywords that embeddings may not capture well

This is why we need the next component: **reranking**.

Adding Reranking: The Power of Cross-Encoders
---------------------------------------------

Cross-encoders fix the "query-document unawareness" problem by processing them together.

How Cross-Encoders Work
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Bi-Encoder (Stage 1):
   ┌─────────┐     ┌─────────┐
   │  Query  │     │  Doc    │
   │ Encoder │     │ Encoder │
   └────┬────┘     └────┬────┘
        │               │
        ▼               ▼
      [768]           [768]      → Dot Product → Score
   
   Cross-Encoder (Stage 2):
   ┌─────────────────────────────────┐
   │  [CLS] Query [SEP] Document [SEP] │
   │         Joint Encoder            │
   └───────────────┬─────────────────┘
                   │
                   ▼
                 Score

**The key difference:** Cross-encoders see the full query-document interaction through 
self-attention. This is much more powerful but computationally expensive.

Adding Reranking to the Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sentence_transformers import SentenceTransformer, CrossEncoder
   import numpy as np
   
   # Stage 1: Fast retrieval with bi-encoder
   bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
   doc_embeddings = bi_encoder.encode(documents, normalize_embeddings=True)
   
   query = "What was Studio Ghibli's first film?"
   query_embedding = bi_encoder.encode(query, normalize_embeddings=True)
   
   # Get top-100 candidates (fast, ~10ms)
   similarities = np.dot(doc_embeddings, query_embedding.T)
   top_100_indices = np.argsort(similarities)[-100:][::-1]
   candidates = [documents[i] for i in top_100_indices]
   
   # Stage 2: Precise reranking with cross-encoder
   cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
   pairs = [[query, doc] for doc in candidates]
   scores = cross_encoder.predict(pairs)
   
   # Get final top-10 (slower but much more accurate, ~2-5s)
   top_10_indices = np.argsort(scores)[-10:][::-1]
   final_results = [candidates[i] for i in top_10_indices]

The World of Rerankers
^^^^^^^^^^^^^^^^^^^^^^

Beyond basic cross-encoders, there are many reranking approaches:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Type
     - Examples
     - Trade-off
   * - **Cross-Encoders**
     - MiniLM, BGE-reranker
     - Best accuracy, moderate speed
   * - **T5-based**
     - MonoT5, RankT5
     - Good accuracy, slower
   * - **LLM-based**
     - RankGPT, RankZephyr
     - Excellent zero-shot, expensive
   * - **API-based**
     - Cohere, Jina, Voyage
     - Easy to use, cost per query

.. tip::

   **Using the rerankers library** (maintained by Ben Clavié):
   
   .. code-block:: python
   
      from rerankers import Reranker
      
      # Local cross-encoder
      ranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
      
      # Or API-based (Cohere)
      ranker = Reranker("cohere", api_key="...")
      
      # Same interface for all!
      results = ranker.rank(query="...", docs=[...])

Keyword Search: The Old Legend Lives On
---------------------------------------

One of the most overlooked components in modern RAG systems is good old BM25.

Why BM25 Still Matters
^^^^^^^^^^^^^^^^^^^^^^

.. important::

   **"An ongoing joke is that information retrieval has progressed slowly because 
   BM25 is too strong a baseline."** — Ben Clavié

Semantic search via embeddings is powerful, but compressing hundreds of tokens to a 
single vector **inevitably loses information**:

* Embeddings learn to represent information useful to their **training queries**
* Training data is **never fully representative** of your domain
* **Humans love keywords**: acronyms, domain-specific terms, product codes

BEIR Benchmark Evidence
^^^^^^^^^^^^^^^^^^^^^^^

From the BEIR benchmark (Thakur et al., 2021), BM25 outperforms many dense models 
on several datasets:

.. code-block:: text

   Dataset         │ BM25   │ DPR    │ ANCE   │ ColBERT
   ────────────────┼────────┼────────┼────────┼────────
   TREC-COVID      │ 0.656  │ 0.332  │ 0.654  │ 0.677
   NFCorpus        │ 0.325  │ 0.189  │ 0.237  │ 0.319
   Touché-2020     │ 0.367  │ 0.131  │ 0.240  │ 0.162
   Robust04        │ 0.408  │ 0.252  │ 0.392  │ 0.427
   
   Avg vs BM25     │   —    │ -47.7% │ -7.4%  │ -2.8%

BM25 is especially powerful for:

* **Longer documents** (more term statistics to leverage)
* **Domain-specific jargon** (medical, legal, technical)
* **Exact match requirements** (product codes, statute numbers)

And its inference overhead is **virtually unnoticeable** — a near free-lunch addition.

Hybrid Search: Best of Both Worlds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combine BM25 and dense retrieval for robustness:

.. code-block:: python

   from rank_bm25 import BM25Okapi
   from sentence_transformers import SentenceTransformer
   import numpy as np
   
   # Prepare BM25
   tokenized_docs = [doc.lower().split() for doc in documents]
   bm25 = BM25Okapi(tokenized_docs)
   
   # Prepare dense
   bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
   doc_embeddings = bi_encoder.encode(documents, normalize_embeddings=True)
   
   def hybrid_search(query, top_k=100, alpha=0.5):
       """Combine BM25 and dense scores with weight alpha."""
       # BM25 scores
       tokenized_query = query.lower().split()
       bm25_scores = bm25.get_scores(tokenized_query)
       bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
       
       # Dense scores
       query_emb = bi_encoder.encode(query, normalize_embeddings=True)
       dense_scores = np.dot(doc_embeddings, query_emb.T).flatten()
       dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-6)
       
       # Combine
       hybrid_scores = alpha * dense_scores + (1 - alpha) * bm25_scores
       top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
       
       return [documents[i] for i in top_indices]

Metadata Filtering: Don't Search What You Don't Need
----------------------------------------------------

Outside of academic benchmarks, documents don't exist in a vacuum. Metadata filtering 
is crucial for production systems.

The Problem
^^^^^^^^^^^

Consider this query:

   *"Get me the cruise division financial report for Q4 2022"*

Vector search can fail here because:

1. The model must accurately represent "financial report" + "cruise division" + "Q4" + "2022" 
   in a single vector
2. If top-k is too high, you'll pass irrelevant financial reports to your LLM

The Solution: Pre-filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use entity extraction to identify filterable attributes:

.. code-block:: text

   Query: "Get me the cruise division financial report for Q4 2022"
   
   Extracted entities:
   - DEPARTMENT: "cruise division"
   - DOCUMENT_TYPE: "financial report"  
   - TIME_PERIOD: "Q4 2022"

Then filter **before** vector search:

.. code-block:: python

   # Instead of searching all documents...
   results = vector_search(query, all_documents, top_k=100)
   
   # Pre-filter to relevant subset
   filtered_docs = [d for d in all_documents 
                    if d.department == "cruise" 
                    and d.doc_type == "financial_report"
                    and d.period == "Q4_2022"]
   results = vector_search(query, filtered_docs, top_k=10)

Entity Extraction with GliNER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from gliner import GLiNER
   
   model = GLiNER.from_pretrained("urchade/gliner_base")
   
   query = "Get me the cruise division financial report for Q4 2022"
   labels = ["department", "document_type", "time_period"]
   
   entities = model.predict_entities(query, labels)
   # [{'text': 'cruise division', 'label': 'department'},
   #  {'text': 'financial report', 'label': 'document_type'},
   #  {'text': 'Q4 2022', 'label': 'time_period'}]

The Final MVP++: Putting It All Together
----------------------------------------

Here's the complete production-ready pipeline in ~30 lines:

.. code-block:: python

   import lancedb
   from lancedb.pydantic import LanceModel, Vector
   from lancedb.embeddings import get_registry
   from lancedb.rerankers import CohereReranker
   
   # Initialize embedding model
   model = get_registry().get("sentence-transformers").create(
       name="BAAI/bge-small-en-v1.5"
   )
   
   # Define document schema with metadata
   class Document(LanceModel):
       text: str = model.SourceField()
       vector: Vector(384) = model.VectorField()
       category: str  # Metadata for filtering
   
   # Create database and table
   db = lancedb.connect(".my_db")
   tbl = db.create_table("my_table", schema=Document)
   
   # Add documents (embedding happens automatically)
   tbl.add(docs)  # docs = [{"text": "...", "category": "..."}, ...]
   
   # Create full-text search index for hybrid search
   tbl.create_fts_index("text")
   
   # Initialize reranker
   reranker = CohereReranker()
   
   # Query with all components
   query = "What is Chihiro's new name given to her by the witch?"
   results = (
       tbl.search(query, query_type="hybrid")  # Hybrid = BM25 + dense
       .where("category = 'film'", prefilter=True)  # Metadata filter
       .limit(100)  # First-pass retrieval
       .rerank(reranker=reranker)  # Cross-encoder reranking
   )

Pipeline Architecture Summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────┐
   │                        MVP++ RAG Pipeline                           │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                     │
   │  ┌─────────┐     ┌──────────────┐     ┌──────────────────────────┐ │
   │  │  Query  │ ──► │   Entity     │ ──► │   Metadata Filtering     │ │
   │  │         │     │  Extraction  │     │   (Pre-filter corpus)    │ │
   │  └─────────┘     └──────────────┘     └────────────┬─────────────┘ │
   │                                                     │               │
   │                                                     ▼               │
   │                                       ┌──────────────────────────┐ │
   │                                       │   Hybrid Retrieval       │ │
   │                                       │   (BM25 + Dense)         │ │
   │                                       │   → Top-100 candidates   │ │
   │                                       └────────────┬─────────────┘ │
   │                                                     │               │
   │                                                     ▼               │
   │                                       ┌──────────────────────────┐ │
   │                                       │   Cross-Encoder          │ │
   │                                       │   Reranking              │ │
   │                                       │   → Top-10 final         │ │
   │                                       └────────────┬─────────────┘ │
   │                                                     │               │
   │                                                     ▼               │
   │                                       ┌──────────────────────────┐ │
   │                                       │   LLM Generation         │ │
   │                                       │   (with retrieved docs)  │ │
   │                                       └──────────────────────────┘ │
   │                                                                     │
   └─────────────────────────────────────────────────────────────────────┘

Component Checklist
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Component
     - Priority
     - Notes
   * - **Bi-encoder retrieval**
     - Required
     - Start with BGE or E5 models
   * - **Cross-encoder reranking**
     - Highly recommended
     - 10-30% accuracy improvement typical
   * - **BM25 / Hybrid search**
     - Recommended
     - Near-zero overhead, helps with keywords
   * - **Metadata filtering**
     - Situational
     - Essential when documents have clear attributes
   * - **Entity extraction**
     - Optional
     - Automates metadata filtering from queries

What's Next?
------------

This guide covers the "compact MVP++" — the foundation every RAG system should have. 
More advanced topics include:

**Beyond Single Vectors:**

* **ColBERT / Late Interaction**: Multiple vectors per document for fine-grained matching
* **SPLADE**: Learned sparse representations combining neural + keyword matching

**Training and Optimization:**

* **Hard negative mining**: Improving retrieval with better training data
* **Knowledge distillation**: Making cross-encoders faster
* **Domain adaptation**: Fine-tuning for your specific use case

**Evaluation:**

* Systematic evaluation is critical but too important to cover briefly
* See :doc:`benchmarks_and_datasets` for evaluation metrics and datasets

References
----------

1. Clavié, B. (2024). "Beyond Explaining the Basics of Retrieval (Augmented Generation)." 
   Talk at Mastering LLMs Conference.
   
   * **Video**: `YouTube <https://youtu.be/0nA5QG3087g>`_
   * **Slides & Transcript**: `Parlance Labs <https://parlance-labs.com/education/rag/ben.html>`_

2. Thakur, N., et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation 
   of Information Retrieval Models." NeurIPS 2021.

3. RAGatouille library: https://github.com/AnswerDotAI/RAGatouille

4. Rerankers library: https://github.com/AnswerDotAI/rerankers

5. LanceDB documentation: https://lancedb.github.io/lancedb/

6. GLiNER: Generalist Model for Named Entity Recognition (arXiv:2311.08526)

Related Documentation
---------------------

* :doc:`rag_overview` - Conceptual two-stage architecture
* :doc:`rag_complexity` - Mathematical complexity analysis
* :doc:`stage1_retrieval/index` - Deep dive into retrieval methods
* :doc:`stage2_reranking/index` - Reranking architectures
* :doc:`retrieval-reranking-libraries-comparison` - Library comparison

