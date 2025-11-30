Cross-Encoders for Re-ranking
==============================

Cross-encoders are the most accurate re-ranking models, processing query and document 
jointly through a single transformer to produce precise relevance scores.

Architecture Overview
---------------------

**How Cross-Encoders Work**

Unlike bi-encoders that encode query and document separately, cross-encoders concatenate 
them and process together:

.. code-block:: text

   Bi-Encoder (Stage 1):
   Query    → BERT → embedding_q ┐
                                  ├→ dot_product(emb_q, emb_d) → score
   Document → BERT → embedding_d ┘
   
   Cross-Encoder (Stage 2):
   [CLS] Query [SEP] Document [SEP] → BERT → [CLS] token → Linear → score

**The Key Difference**:

* **Bi-encoder**: Similarity in embedding space (fast, pre-computable)
* **Cross-encoder**: Full self-attention between query-document tokens (slow, accurate)

Why Cross-Encoders Are More Accurate
-------------------------------------

**Token-Level Interactions**

The transformer's self-attention allows every query token to attend to every document token:

* Query "capital" can attend to doc "capital", "city", "largest", etc.
* Can perform multi-hop reasoning across tokens
* Captures semantic composition (not just bag-of-words similarity)

**Example:**

Query: *"Who invented the telephone?"*

Document: *"Alexander Graham Bell patented the telephone in 1876"*

**Bi-encoder sees:**
- High similarity (both contain "telephone", "Bell", etc.)
- But can't connect "invented" → "patented" or "who" → "Alexander Graham Bell"

**Cross-encoder sees:**
- "who" attends to "Alexander Graham Bell" → Answer to question
- "invented" attends to "patented" → Semantic equivalence
- Full reasoning chain: This doc answers the query

Implementation
--------------

**Using Sentence-Transformers**

.. code-block:: python

   from sentence_transformers import CrossEncoder
   
   # Load pre-trained cross-encoder
   model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
   
   # Score query-document pairs
   pairs = [
       ("What is the capital of France?", "Paris is the capital of France"),
       ("What is the capital of France?", "France is in Europe"),
       ("What is the capital of France?", "Best restaurants in Paris")
   ]
   
   scores = model.predict(pairs)
   # scores: [0.98, 0.12, 0.35] - clearly ranks correct answer first

**Training Your Own**

.. code-block:: python

   from sentence_transformers import CrossEncoder, InputExample
   
   # Prepare training data
   train_samples = [
       InputExample(texts=["query1", "relevant_doc"], label=1.0),
       InputExample(texts=["query1", "irrelevant_doc"], label=0.0),
       # ... more pairs
   ]
   
   # Initialize from pre-trained BERT
   model = CrossEncoder('bert-base-uncased', num_labels=1)
   
   # Train
   model.fit(
       train_dataloader=train_samples,
       epochs=3,
       warmup_steps=100
   )

Variants and Improvements
--------------------------

MonoT5
^^^^^^

Instead of BERT, uses T5 (text-to-text transformer):

.. code-block:: python

   # Input to T5
   input_text = f"Query: {query} Document: {document} Relevant:"
   
   # T5 generates
   output = model.generate(input_text)  # "true" or "false"
   
   # Score = probability of generating "true"

**Advantage**: T5's generative nature may capture relevance better than classification head.

RankT5
^^^^^^

T5 that directly generates relevance scores:

.. code-block:: python

   input_text = f"Query: {query} Document: {document} Score:"
   output = model.generate(input_text)  # "0", "1", "2", ... "9"
   
   # 10-way classification via generation

duoT5
^^^^^

Pairwise ranking with T5:

.. code-block:: python

   input_text = f"Query: {query} Document1: {doc1} Document2: {doc2} More relevant:"
   output = model.generate(input_text)  # "Document1" or "Document2"

**Advantage**: More stable than absolute scores (easier for model to judge relative relevance).

Training Cross-Encoders with Hard Negatives
--------------------------------------------

**The Same Hard Negative Problem Applies!**

Cross-encoders also benefit from hard negative training:

.. code-block:: python

   # Bad: Random negatives
   train_data = [(query, positive, random_doc) for ...]
   
   # Better: BM25 negatives
   train_data = [(query, positive, bm25_hard_neg) for ...]
   
   # Best: Bi-encoder mined negatives
   # These are docs that bi-encoder ranked high but are actually irrelevant
   bi_encoder_errors = bi_encoder.search(query, k=100)
   hard_negs = [doc for doc in bi_encoder_errors if not is_relevant(doc)]
   train_data = [(query, positive, hard_neg) for hard_neg in hard_negs]

**Why This Works**:

Cross-encoder learns to correct bi-encoder's mistakes. Training it on bi-encoder's 
hardest errors makes it the perfect "teacher" for Stage 2.

Performance Benchmarks
-----------------------

.. list-table:: Cross-Encoder Performance (MS MARCO)
   :header-rows: 1
   :widths: 30 20 20 30

   * - Model
     - MRR@10
     - Latency (100 docs)
     - Size
   * - Bi-encoder only
     - 0.311
     - ~10ms
     - 400MB
   * - + MiniLM-L6 Cross-encoder
     - 0.389
     - ~3s
     - 90MB
   * - + MiniLM-L12 Cross-encoder
     - 0.402
     - ~5s
     - 130MB
   * - + BERT-base Cross-encoder
     - 0.416
     - ~8s
     - 420MB
   * - + BERT-large Cross-encoder
     - 0.428
     - ~15s
     - 1.3GB

**Trade-off**: Larger models = better accuracy but slower.

Cost-Effective Choices
-----------------------

**For Production (Recommended)**

.. code-block:: python

   # MiniLM-L6: 85% of BERT-large performance, 10% of latency
   model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

**For Research/Maximum Accuracy**

.. code-block:: python

   # BERT-large or T5-large
   model = CrossEncoder('cross-encoder/ms-marco-electra-base')  # Faster than BERT

**For Budget Constrained**

.. code-block:: python

   # TinyBERT cross-encoder (custom trained)
   # Or use ColBERT as re-ranker (better speed-accuracy than small cross-encoder)

Deployment Considerations
--------------------------

**Batching**

.. code-block:: python

   # Don't score one-by-one
   for doc in candidates:
       score = model.predict([(query, doc)])  # ❌ Slow!
   
   # Batch all pairs together
   pairs = [(query, doc) for doc in candidates]
   scores = model.predict(pairs)  # ✅ Fast! (GPU batching)

**GPU vs CPU**

* GPU: ~50-100 pairs/second
* CPU: ~10-20 pairs/second
* For 100 candidates: 1-2s on GPU, 5-10s on CPU

**Caching**

For frequently-seen documents, cache scores:

.. code-block:: python

   cache = {}  # {(query_hash, doc_hash): score}
   
   if (query_hash, doc_hash) in cache:
       score = cache[(query_hash, doc_hash)]
   else:
       score = model.predict([(query, doc)])[0]
       cache[(query_hash, doc_hash)] = score

Next Steps
----------

* See :doc:`llm_rerankers` for using large language models as re-rankers
* See :doc:`../stage1_retrieval/late_interaction` for ColBERT as alternative
* See :doc:`../stage1_retrieval/hard_mining` for training data quality

