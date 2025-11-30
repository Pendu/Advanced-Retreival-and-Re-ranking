LLM-Based Re-rankers
====================

Large Language Models (LLMs) can perform re-ranking through prompting, offering zero-shot 
capability and explainability at the cost of higher latency and compute.

The LLM Re-ranking Paradigm
----------------------------

**Traditional Cross-Encoder:**

* Requires training on labeled (query, doc, relevance) data
* Fixed to specific task/domain
* Fast inference (~50ms per pair)
* No explanation

**LLM Re-ranker:**

* Zero-shot via prompting (no training needed)
* Generalizes across tasks
* Slower inference (~500-2000ms per pair)
* Can provide reasoning

Zero-Shot Prompting Approaches
-------------------------------

Pointwise Relevance
^^^^^^^^^^^^^^^^^^^

**Method**: Ask LLM to judge each document independently.

.. code-block:: python

   prompt = f"""
   Given the query: "{query}"
   And the document: "{document}"
   
   Is this document relevant to the query?
   Answer with only "Yes" or "No".
   """
   
   response = llm.generate(prompt)
   score = 1.0 if response == "Yes" else 0.0

**Problem**: No relative comparison, binary scores limit ranking.

Pairwise Comparison
^^^^^^^^^^^^^^^^^^^

**Method**: Ask LLM to compare pairs of documents.

.. code-block:: python

   prompt = f"""
   Query: "{query}"
   
   Document A: "{doc_a}"
   Document B: "{doc_b}"
   
   Which document is more relevant to the query?
   Answer with "A" or "B".
   """
   
   response = llm.generate(prompt)
   # Build ranking via pairwise comparisons (like bubble sort)

**Advantage**: Relative judgments are easier than absolute scores.

**Problem**: Requires O(n²) comparisons for n documents.

Listwise Ranking
^^^^^^^^^^^^^^^^

**Method**: Ask LLM to rank entire list at once.

.. code-block:: python

   prompt = f"""
   Query: "{query}"
   
   Rank the following documents by relevance:
   [1] {doc_1}
   [2] {doc_2}
   [3] {doc_3}
   ...
   [10] {doc_10}
   
   Provide the ranking as a list of numbers (e.g., [3, 1, 7, ...]).
   """
   
   response = llm.generate(prompt)
   # Parse: [3, 1, 7, ...] means doc 3 is most relevant

**Advantage**: Single LLM call, considers all documents together.

**Problem**: Performance degrades with >20 documents (context length, attention issues).

Sliding Window Approach
^^^^^^^^^^^^^^^^^^^^^^^

**Method**: For large candidate sets, use sliding window.

.. code-block:: python

   # RankGPT approach
   window_size = 20
   sorted_docs = candidates.copy()
   
   for i in range(0, len(candidates), window_size):
       window = sorted_docs[i:i+window_size]
       reranked_window = llm.listwise_rank(query, window)
       sorted_docs[i:i+window_size] = reranked_window
   
   # Refine with multiple passes
   for pass_num in range(3):
       sorted_docs = sliding_window_rank(sorted_docs, window_size)

Cost vs Performance
-------------------

.. list-table:: LLM Re-ranker Trade-offs
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Model
     - Latency (100 docs)
     - Cost (100 docs)
     - Accuracy
     - Zero-shot?
   * - MiniLM Cross-encoder
     - ~2s
     - $0.000
     - Good
     - No (needs training)
   * - GPT-3.5 (pointwise)
     - ~60s
     - $0.20
     - Better
     - Yes
   * - GPT-4 (pointwise)
     - ~120s
     - $2.00
     - Best
     - Yes
   * - GPT-3.5 (listwise)
     - ~10s
     - $0.05
     - Better
     - Yes
   * - GPT-4 (listwise)
     - ~20s
     - $0.50
     - Best
     - Yes
   * - RankLlama (self-hosted)
     - ~30s
     - $0.000
     - Good
     - Yes

**Key Insight**: LLMs are 10-100x more expensive than trained cross-encoders but offer 
zero-shot capability.

When to Use LLM Re-rankers
--------------------------

✅ **Use LLM Re-rankers When:**

* No training data available (pure zero-shot)
* Need explainability (LLM can explain why doc is relevant)
* Domain shifts frequently (no time to retrain)
* Budget allows ($0.10-$1.00 per query acceptable)
* Accuracy is paramount (research, high-value queries)

❌ **Don't Use When:**

* Serving millions of queries (cost prohibitive)
* Latency < 5s required (LLMs too slow)
* Have good training data (cross-encoder is better value)
* Queries are simple (overkill)

Practical Implementation
-------------------------

**Cost Optimization: Staged LLM Re-ranking**

.. code-block:: python

   # Stage 1: Bi-encoder (10M → 1000)
   candidates_1k = bi_encoder.search(query, corpus, top_k=1000)
   
   # Stage 2: Fast cross-encoder (1000 → 100)
   candidates_100 = cross_encoder.rerank(query, candidates_1k, top_k=100)
   
   # Stage 3: LLM re-ranking (100 → 10)
   # Only use expensive LLM on final 100
   final_10 = llm_reranker.rerank(query, candidates_100, top_k=10)

**Cost**: 100 docs × $0.0005 = $0.05 per query (vs $5.00 if LLM on all 10K)

**Caching for Repeated Queries**

.. code-block:: python

   import hashlib
   
   def get_cache_key(query, doc):
       return hashlib.md5(f"{query}::{doc}".encode()).hexdigest()
   
   cache = {}  # Persistent cache (Redis, DynamoDB, etc.)
   
   if cache_key in cache:
       score = cache[cache_key]  # Free!
   else:
       score = llm.rank(query, doc)  # Expensive
       cache[cache_key] = score

**Batch Processing for Cost**

.. code-block:: python

   # Instead of real-time, batch queries
   queries_batch = collect_queries_for_10_minutes()
   
   # Send all to LLM in one request (cheaper bulk pricing)
   all_pairs = [(q, d) for q in queries_batch for d in candidates[q]]
   all_scores = llm.batch_predict(all_pairs)  # Bulk API rates

Best Practices
--------------

**Prompt Engineering**

.. code-block:: text

   # ❌ Bad prompt (ambiguous)
   "Is this relevant?"
   
   # ✅ Good prompt (clear instructions)
   "You are a search quality evaluator. Given the query and document below, 
   determine if the document directly answers the query or provides the 
   information the user is looking for. Consider factual accuracy and 
   completeness. Respond with only 'Relevant' or 'Not Relevant'.
   
   Query: {query}
   Document: {document}
   
   Judgment:"

**Few-Shot Examples**

Include examples in prompt for better calibration:

.. code-block:: text

   Here are examples of relevant and irrelevant documents:
   
   Example 1:
   Query: "What is photosynthesis?"
   Document: "Photosynthesis is the process by which plants convert sunlight to energy."
   Judgment: Relevant
   
   Example 2:
   Query: "What is photosynthesis?"
   Document: "Plants are green and grow in soil."
   Judgment: Not Relevant
   
   Now judge this pair:
   Query: "{query}"
   Document: "{document}"
   Judgment:

Future Directions
-----------------

**Active Research Areas:**

1. **Distillation**: Train small cross-encoder to mimic LLM judgments
2. **Efficient Prompting**: Compress documents before passing to LLM
3. **Hybrid Scoring**: Combine LLM with traditional cross-encoder
4. **Explanation Generation**: Use LLM to explain ranking decisions
5. **Multi-modal**: LLM re-rankers for images, videos

**Emerging Models:**

* RankLlama: LLama-2 fine-tuned specifically for ranking
* RankGPT: GPT-based with specialized prompting
* PRP: Pairwise Ranking Prompting
* Self-Consistency: Multiple LLM calls then vote

Next Steps
----------

* See :doc:`cross_encoders` for traditional trained re-rankers
* See :doc:`../stage1_retrieval/hard_mining` for improving training data
* See :doc:`../stage1_retrieval/late_interaction` for ColBERT alternative

