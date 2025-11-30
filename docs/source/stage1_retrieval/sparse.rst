Sparse Retrieval Methods
=========================

Sparse retrieval methods dominated information retrieval for decades before the advent 
of dense neural methods. They remain relevant today as fast, interpretable baselines and 
as components of hybrid systems.

BM25: The Classic Baseline
---------------------------

**Best Matching 25** (BM25) is the most widely used sparse retrieval algorithm.

Key Characteristics
^^^^^^^^^^^^^^^^^^^

* **Bag-of-Words**: Ignores word order, focuses on term frequency
* **Inverted Index**: Pre-computed data structure for fast lookup
* **TF-IDF Based**: Balances term frequency with document frequency
* **Deterministic**: No training required, pure statistics

The BM25 Formula
^^^^^^^^^^^^^^^^

.. math::

   \text{score}(D, Q) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}

Where:
- :math:`f(q_i, D)` = frequency of term q\ :sub:`i` in document D
- :math:`|D|` = length of document D
- :math:`avgdl` = average document length in collection
- :math:`k_1` = term frequency saturation parameter (typically 1.2-2.0)
- :math:`b` = length normalization parameter (typically 0.75)

Strengths
^^^^^^^^^

✅ **Very Fast**: Index lookup is O(log N) with inverted index
✅ **No Training**: Works out-of-the-box on any text
✅ **Interpretable**: Can explain scores via term matching
✅ **Robust**: Works well for keyword-heavy queries
✅ **Memory Efficient**: Only stores term statistics

Weaknesses
^^^^^^^^^^

❌ **Vocabulary Mismatch**: Fails when query and document use different words
❌ **No Semantics**: Can't match "car" with "automobile"
❌ **Keyword Dependent**: Poor for natural language questions
❌ **No Context**: Treats all occurrences of a word identically

When to Use BM25
----------------

BM25 is still the **best choice** for:

**Legal and Medical Search**
  Documents and queries use precise technical terminology.
  Example: "42 USC 1983" should match exact statute.

**Code Search**
  Identifiers and function names should match exactly.
  Example: "numpy.array" shouldn't match "array list"

**Hybrid Stage 1**
  BM25 casts a wide net (10K candidates), then dense retrieval re-ranks.

**Evaluation Baseline**
  Always compare dense methods against BM25 to show semantic improvement.

Implementation
--------------

**Python (Pyserini/Lucene)**

.. code-block:: python

   from pyserini.search import SimpleSearcher
   
   # Index your corpus
   searcher = SimpleSearcher('indexes/my-corpus')
   searcher.set_bm25(k1=0.9, b=0.4)  # Tune hyperparameters
   
   # Search
   hits = searcher.search('what is the capital of France', k=100)
   
   for hit in hits:
       print(f"Doc {hit.docid}: {hit.score:.4f}")

**Python (Elasticsearch)**

.. code-block:: python

   from elasticsearch import Elasticsearch
   
   es = Elasticsearch(['localhost:9200'])
   
   query = {
       "query": {
           "match": {
               "content": {
                   "query": "capital of France",
                   "operator": "or"
               }
           }
       }
   }
   
   results = es.search(index="my-corpus", body=query, size=100)

Modern Sparse Methods
----------------------

Recent research has developed **learned sparse** methods that retain BM25's efficiency 
while adding semantic capability.

SPLADE
^^^^^^

**Sparse Lexical and Expansion** model learns to:

* Up-weight important terms (like BM25)
* Add related terms not in the original text (expansion)
* All while maintaining sparse representations for efficient indexing

.. code-block:: python

   from splade import Splade
   
   model = Splade('naver/splade-cocondenser-ensembledistil')
   
   # Produces sparse vector with learned weights
   query_vec = model.encode("what is the capital of France")
   # Non-zero entries might include: "capital", "France", "Paris" (expansion!)

**Advantages over BM25:**
- Handles synonyms and related terms
- Still uses inverted index
- Can leverage neural pre-training

Recommendations
---------------

**For New Projects**

1. **Start with BM25** as baseline (always measure improvement over it)
2. **Add Dense Retrieval** (bi-encoder) for semantic matching
3. **Consider Hybrid** (BM25 + Dense) for robustness
4. **SPLADE** if you want learned sparse (best of both worlds)

**Hyperparameter Tuning**

BM25 has two key parameters:

* **k1** (term frequency saturation): Try 0.9, 1.2, 2.0
* **b** (length normalization): Try 0.4, 0.75, 1.0

Optimal values depend on your corpus:
- Short documents (e.g., tweets): lower b (0.3-0.5)
- Long documents (e.g., articles): higher b (0.7-0.9)
- Use grid search on validation set

Resources
---------

**Libraries**

* `Pyserini <https://github.com/castorini/pyserini>`_ - Python wrapper for Lucene (BM25)
* `Elasticsearch <https://www.elastic.co/elasticsearch/>`_ - Distributed search engine
* `SPLADE <https://github.com/naver/splade>`_ - Learned sparse retrieval

**Papers**

* Robertson & Zaragoza (2009) - "The Probabilistic Relevance Framework: BM25 and Beyond"
* Formal et al. (2021) - "SPLADE: Sparse Lexical and Expansion Model"

**Datasets for Evaluation**

* MS MARCO Passage Ranking
* Natural Questions (NQ)
* BEIR (for zero-shot evaluation)

Next Steps
----------

* Proceed to :doc:`dense_baselines` to see how neural methods improve over BM25
* See :doc:`hybrid` for combining sparse and dense methods

