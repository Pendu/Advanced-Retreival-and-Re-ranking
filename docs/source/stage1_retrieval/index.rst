Stage 1: Retrieval Methods
===========================

This section covers methods for the first stage of the RAG pipeline: efficiently retrieving 
candidate documents from large corpora.

.. toctree::
   :maxdepth: 2
   :caption: Stage 1 Topics:

   sparse
   dense_baselines
   hard_mining
   late_interaction
   hybrid
   pretraining
   joint_learning
   literature_survey/index

Introduction: From Lexical to Semantic Matching
-----------------------------------------------

Text retrieval aims to find relevant information resources (e.g., documents or passages) in response to a user's natural language query. As a fundamental technique for overcoming information overload, its methodology has evolved through several distinct paradigms, as outlined in foundational surveys of the field.

1. The Era of Sparse Retrieval (Lexical Matching)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For decades, the field was dominated by the **Vector Space Model** and the "bag-of-words" assumption.
*   **Mechanism**: Queries and documents are represented as sparse vectors where dimensions correspond to explicit terms (words) from the vocabulary.
*   **Algorithms**: **TF-IDF** and **BM25** became the gold standard for estimating relevance based on lexical overlap (exact word matches).
*   **Infrastructure**: These methods are efficiently supported by **Inverted Indexes**, allowing for lightning-fast lookup.

While effective and explainable, these methods struggle with the **vocabulary mismatch** problem—failing to retrieve relevant documents that use synonyms or different phrasing than the query.

2. Learning to Rank and Early Neural IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To move beyond simple heuristics, researchers adopted **Learning to Rank (LTR)**, using supervised learning with hand-crafted features (e.g., query term proximity, page rank) to train ranking functions.

Subsequently, early **Neural IR** approaches began using shallow neural networks (e.g., word2vec) to learn low-dimensional embeddings. Unlike sparse vectors, these **dense vectors** aim to capture latent semantics, allowing matching based on meaning rather than just surface forms.

3. The Rise of PLM-based Dense Retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The advent of Pretrained Language Models (PLMs) like **BERT** marked a revolutionary paradigm shift.
*   **Deep Understanding**: PLMs, pretrained on massive text corpora, encode rich semantic knowledge and context sensitivity.
*   **The "Pretrain-then-Finetune" Paradigm**: Models are first pretrained on general text, then fine-tuned on retrieval datasets (like **MS MARCO** or **Natural Questions**).
*   **Semantic Matching**: Relevance is measured by the similarity (e.g., dot product or cosine) between the dense vector representations of the query and document.

This shift enables systems to answer complex queries (e.g., *"average salary for dental hygienist in nebraska"*) where the answer depends on understanding intent and semantic relationships, not just keyword matching. This survey and documentation focus on this modern era of **PLM-based Dense Retrieval**.

Core Aspects of Modern Retrieval
--------------------------------

We organize the study of Stage 1 retrieval into four major aspects:

**1. Architecture**
How to design the neural networks that encode text.
*   **Dual-Encoders**: Independent encoding of query and document into single vectors (fastest, standard for dense retrieval).
*   **Late Interaction**: Preserving token-level embeddings for richer, fine-grained interaction (e.g., **ColBERT**).
*   **Hybrid**: Architectures that explicitly combine sparse (lexical) and dense (semantic) signals.

**2. Training Strategies**
How to optimize the retriever effectively.
*   **Hard Negative Mining**: The critical process of identifying challenging negatives to teach the model fine-grained distinctions (see :doc:`hard_mining`).
*   **Knowledge Distillation**: Learning from more powerful cross-encoder teachers.
*   **Pre-training**: Tailoring the underlying PLM specifically for retrieval tasks before fine-tuning.

**3. Indexing and Efficiency**
How to search millions of dense vectors in milliseconds.
*   **ANN Search**: Approximate Nearest Neighbor algorithms (e.g., HNSW, FAISS) used to query the dense vector space.
*   **Learned Indexes**: Optimizing the index structure end-to-end with the model.

**4. Integration**
Building the complete retrieval pipeline, including combining multiple retrievers and optimizing the retrieval depth.

Quick Navigation
----------------

* :doc:`sparse` - The foundation: BM25 and Inverted Indexes.
* :doc:`dense_baselines` - The shift to DPR and BERT-based retrieval.
* :doc:`hard_mining` - **Deep Dive**: The most critical training component for dense retrievers.
* :doc:`late_interaction` - Architectures like ColBERT that trade some speed for higher precision.
* :doc:`hybrid` - Best of both worlds: Combining BM25 and Dense Retrieval.
* :doc:`pretraining` - Methods to pretrain models specifically for retrieval (e.g., RetroMAE).
