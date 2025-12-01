Computational Complexity Analysis of RAG Systems
=================================================

This section provides a rigorous analysis of the time and space complexity of each component 
in a Retrieval-Augmented Generation (RAG) pipeline. While :doc:`rag_overview` covers the 
conceptual architecture, this page dives into the mathematical foundations that determine 
system performance at scale.

.. contents:: Table of Contents
   :local:
   :depth: 2

Why Complexity Analysis Matters
-------------------------------

Understanding computational complexity is essential for:

1. **Capacity Planning**: Predicting infrastructure requirements as corpus size grows
2. **Bottleneck Identification**: Knowing where optimization efforts yield the highest returns
3. **Architecture Selection**: Choosing between HNSW vs. IVF, cross-encoder vs. bi-encoder
4. **Cost Estimation**: Translating Big-O to actual latency and compute costs

.. note::

   **Latency Distribution in Production RAG** (empirical data from Milvus, 2024):
   
   * Retrieval: 41-47% of total latency
   * LLM Generation: 50-60% of total latency
   * Preprocessing/Postprocessing: <5%

Document Processing and Chunking
--------------------------------

Complexity Analysis
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Operation
     - Time Complexity
     - Space Complexity
   * - Text tokenization
     - :math:`O(n)`
     - :math:`O(n)`
   * - Fixed-size chunking
     - :math:`O(n)`
     - :math:`O(n/c)` chunks
   * - Semantic chunking
     - :math:`O(n \cdot d^2)` (embedding-based)
     - :math:`O(n)`
   * - Overlap handling
     - :math:`O(n \cdot o/c)`
     - :math:`O(n \cdot (1 + o/c))`

Where:

* :math:`n` = document length in tokens
* :math:`c` = chunk size
* :math:`o` = overlap size
* :math:`d` = embedding dimension

Chunk Size Trade-offs
^^^^^^^^^^^^^^^^^^^^^

The choice of chunk size :math:`c` affects both retrieval quality and storage:

.. math::

   \text{Storage} = N_{chunks} \times d_{emb} \times 4 \text{ bytes} = \frac{n}{c - o} \times d_{emb} \times 4

**Empirical Guidelines** (from NVIDIA, 2025):

* **256-512 tokens**: Best for factoid QA, precise retrieval
* **512-1024 tokens**: Balanced for general RAG
* **1024+ tokens**: Better context but noisier embeddings

Embedding Generation
--------------------

Transformer Encoder Complexity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a transformer encoder with :math:`L` layers, :math:`d` hidden dimension, and sequence 
length :math:`n`:

**Self-Attention Complexity:**

.. math::

   \text{Time}_{attention} = O(n^2 \cdot d)

**Feed-Forward Complexity:**

.. math::

   \text{Time}_{FFN} = O(n \cdot d^2)

**Total Per Layer:**

.. math::

   \text{Time}_{layer} = O(n^2 \cdot d + n \cdot d^2)

For typical embedding models where :math:`n < d` (e.g., n=512, d=768):

.. math::

   \text{Time}_{total} \approx O(L \cdot n \cdot d^2)

Practical Latency Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Embedding Model Latencies (CPU, 100 tokens)
   :header-rows: 1
   :widths: 30 20 20 30

   * - Model
     - Dimension
     - Latency
     - Storage (1M docs)
   * - all-MiniLM-L6-v2
     - 384
     - ~2ms
     - ~1.5 GB
   * - BGE-base-en-v1.5
     - 768
     - ~10ms
     - ~3.0 GB
   * - E5-large-v2
     - 1024
     - ~50ms
     - ~4.0 GB
   * - BGE-M3
     - 1024
     - ~100ms
     - ~4.0 GB

Vector Indexing Algorithms
--------------------------

This is where the "magic" of sub-linear retrieval happens. Understanding these algorithms 
is crucial for production systems.

HNSW (Hierarchical Navigable Small World)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Algorithm Overview:**

HNSW constructs a multi-layer graph where:

* Layer 0 contains all vectors
* Higher layers contain exponentially fewer vectors (skip-list structure)
* Each vector connects to :math:`M` neighbors per layer

**Complexity:**

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Operation
     - Time
     - Space
   * - Index Construction
     - :math:`O(N \cdot \log N \cdot M)`
     - :math:`O(N \cdot d + N \cdot M \cdot L_{max})`
   * - Query (Search)
     - :math:`O(\log N \cdot M \cdot ef)`
     - :math:`O(ef)`
   * - Insert (Single)
     - :math:`O(\log N \cdot M)`
     - :math:`O(d + M \cdot L_{max})`

Where:

* :math:`N` = number of vectors
* :math:`M` = connections per node (typically 16-64)
* :math:`ef` = search beam width (controls recall/speed trade-off)
* :math:`L_{max}` = maximum layer (typically :math:`\log N`)

**Memory Formula:**

.. math::

   \text{Memory}_{HNSW} = N \times (d \times 4 + M \times L_{avg} \times 8) \text{ bytes}

For 1M vectors with d=768, M=32:

.. math::

   \text{Memory} = 10^6 \times (768 \times 4 + 32 \times 4 \times 8) \approx 4.1 \text{ GB}

IVF (Inverted File Index)
^^^^^^^^^^^^^^^^^^^^^^^^^

**Algorithm Overview:**

IVF partitions the vector space into :math:`k` clusters via k-means, then searches only 
the :math:`nprobe` nearest clusters.

**Complexity:**

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Operation
     - Time
     - Space
   * - Index Construction
     - :math:`O(N \cdot d \cdot k \cdot I)`
     - :math:`O(N \cdot d + k \cdot d)`
   * - Query (Search)
     - :math:`O(k \cdot d + nprobe \cdot N/k \cdot d)`
     - :math:`O(d)`
   * - Optimal nprobe
     - :math:`O(\sqrt{N} \cdot d)`
     - —

Where:

* :math:`k` = number of clusters (typically :math:`\sqrt{N}` to :math:`4\sqrt{N}`)
* :math:`I` = k-means iterations
* :math:`nprobe` = clusters to search

**With Product Quantization (PQ):**

PQ compresses vectors by splitting into :math:`m` subvectors and quantizing each to 
:math:`2^{nbits}` centroids:

.. math::

   \text{Compression Ratio} = \frac{d \times 32}{m \times nbits}

For d=768, m=96, nbits=8:

.. math::

   \text{Compression} = \frac{768 \times 32}{96 \times 8} = 32\times

HNSW vs. IVF Trade-offs
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Metric
     - HNSW
     - IVF
     - IVF+PQ
     - Brute Force
   * - Query Time
     - :math:`O(\log N)`
     - :math:`O(\sqrt{N})`
     - :math:`O(\sqrt{N})`
     - :math:`O(N)`
   * - Recall@10
     - 95-99%
     - 85-95%
     - 75-90%
     - 100%
   * - Memory (1M, d=768)
     - ~4.1 GB
     - ~3.1 GB
     - ~0.1 GB
     - ~3.0 GB
   * - Build Time
     - Slow
     - Medium
     - Medium
     - None
   * - Dynamic Insert
     - Good
     - Poor
     - Poor
     - N/A

Cross-Encoder Reranking
-----------------------

Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^

A cross-encoder computes:

.. math::

   s(q, d) = \text{MLP}(\text{BERT}([CLS] \oplus q \oplus [SEP] \oplus d \oplus [SEP]))

**Attention Complexity:**

For input length :math:`L = |q| + |d| + 3` (including special tokens):

.. math::

   \text{Time}_{cross-encoder} = O(L^2 \cdot d + L \cdot d^2)

**Comparison with Bi-Encoder:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Architecture
     - Scoring Complexity
     - Total for k candidates
   * - Bi-Encoder
     - :math:`O(d)` (dot product)
     - :math:`O(L_q \cdot d^2 + k \cdot d)`
   * - Cross-Encoder
     - :math:`O(L^2 \cdot d)` (full attention)
     - :math:`O(k \cdot L^2 \cdot d)`

For k=100 candidates, L=256:

* **Bi-Encoder**: ~10ms (1 query encoding + 100 dot products)
* **Cross-Encoder**: ~5,000ms (100 full forward passes)

**Speedup Ratio:**

.. math::

   \frac{\text{Time}_{cross}}{\text{Time}_{bi}} = \frac{k \cdot L^2}{L_q + k} \approx k \cdot L

For k=100, L=256: **~25,000× slower** (theoretical), ~500× in practice due to batching.

LLM Generation Complexity
-------------------------

This section analyzes the generation phase, which dominates RAG latency.

Transformer Decoder Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Prefill Phase (Prompt Processing):**

Processes the entire prompt in parallel:

.. math::

   \text{Time}_{prefill} = O(n_{layers} \cdot L_{prompt}^2 \cdot d)

**Decode Phase (Token Generation):**

Generates tokens autoregressively:

.. math::

   \text{Time}_{decode} = O(n_{layers} \cdot n_{tokens} \cdot L_{total} \cdot d)

With KV caching, each new token only attends to cached keys/values:

.. math::

   \text{Time}_{decode}^{cached} = O(n_{layers} \cdot n_{tokens} \cdot d)

KV Cache Memory Analysis
^^^^^^^^^^^^^^^^^^^^^^^^

**Memory Formula:**

.. math::

   \text{Memory}_{KV} = 2 \times n_{layers} \times n_{heads} \times L \times d_{head} \times \text{precision}

For Llama-2-7B (32 layers, 32 heads, d_head=128, fp16):

.. math::

   \text{Memory}_{KV} = 2 \times 32 \times 32 \times L \times 128 \times 2 = 524,288 \times L \text{ bytes}

* L=2048: **1.07 GB** per sequence
* L=8192: **4.29 GB** per sequence
* L=32768: **17.18 GB** per sequence

**KV Cache Trade-off:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Strategy
     - Time (200 tokens)
     - Memory
   * - No Cache
     - :math:`O(n \cdot L^2)` → ~40s
     - :math:`O(L \cdot d)`
   * - Full Cache
     - :math:`O(n \cdot L)` → ~4s
     - :math:`O(n_{layers} \cdot L \cdot d)`
   * - Sliding Window
     - :math:`O(n \cdot W)` → ~2s
     - :math:`O(n_{layers} \cdot W \cdot d)`

End-to-End Latency Model
------------------------

Putting it all together:

.. math::

   T_{total} = T_{embed} + T_{search} + T_{rerank} + T_{prefill} + T_{decode}

**Typical Values (MS MARCO scale, V100 GPU):**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Component
     - Complexity
     - Latency
     - % of Total
   * - Query Embedding
     - :math:`O(L_q \cdot d^2)`
     - 10-20ms
     - 1-2%
   * - Vector Search (HNSW)
     - :math:`O(\log N)`
     - 5-50ms
     - 3-5%
   * - Document Fetch
     - :math:`O(k)`
     - 10-30ms
     - 2-3%
   * - Reranking (optional)
     - :math:`O(k \cdot L^2 \cdot d)`
     - 500-2000ms
     - 30-40%
   * - LLM Prefill
     - :math:`O(L^2 \cdot d)`
     - 100-300ms
     - 10-15%
   * - LLM Decode
     - :math:`O(n \cdot L \cdot d)`
     - 2000-8000ms
     - 40-50%

**Total**: 2.5-10 seconds depending on configuration.

Optimization Strategies
-----------------------

Based on the complexity analysis, here are high-impact optimizations:

**For Retrieval (41-47% of latency):**

1. **HNSW Parameter Tuning**: Increase ef_search for recall, decrease for speed
2. **Quantization**: IVF+PQ for memory-constrained deployments
3. **Caching**: Cache frequent queries (30% latency reduction)

**For Reranking (30-40% when used):**

1. **Distillation**: Train smaller cross-encoders (MiniLM-L6 vs BERT-base)
2. **Early Exit**: Stop reranking when confidence is high
3. **Batching**: Process multiple candidates in parallel

**For Generation (40-50% of latency):**

1. **KV Caching**: Essential for any production system
2. **Speculative Decoding**: 2-3× speedup with draft models
3. **Quantization**: INT8/INT4 for 2-4× memory reduction
4. **Context Compression**: Reduce prompt length with summarization

Scaling Laws
------------

How does performance scale with corpus size?

**Retrieval Scaling:**

.. math::

   T_{retrieval}(N) = c_1 \cdot \log N + c_2

For HNSW, doubling corpus size adds ~constant time (logarithmic scaling).

**Memory Scaling:**

.. math::

   M(N) = N \cdot (d \cdot 4 + \text{index overhead})

Linear scaling—10× corpus = 10× memory.

**Throughput Scaling:**

With batching, throughput scales sub-linearly due to memory bandwidth limits:

.. math::

   \text{Throughput}(batch) = \frac{batch}{\alpha + \beta \cdot batch}

Where :math:`\alpha` = fixed overhead, :math:`\beta` = per-query cost.

Summary
-------

.. important::

   **Key Takeaways:**
   
   1. **Retrieval is O(log N)** with HNSW—corpus size matters less than you think
   2. **Reranking is O(k × L²)**—limit candidate count k, not corpus size
   3. **LLM generation dominates**—optimize here for biggest gains
   4. **KV caching is mandatory**—10× speedup, but watch memory
   5. **Trade-offs are unavoidable**—HNSW vs IVF, accuracy vs speed

References
----------

1. Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and Robust Approximate Nearest Neighbor 
   Search Using Hierarchical Navigable Small World Graphs." *IEEE TPAMI*.

2. Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale Similarity Search with GPUs." 
   *IEEE TBD*.

3. Pope, R., et al. (2023). "Efficiently Scaling Transformer Inference." *MLSys 2023*.

4. Milvus Documentation. "RAG Pipeline Latency Analysis." 2024.

5. Gao, L., et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." 
   *arXiv:2312.10997*.

Next Steps
----------

* See :doc:`rag_overview` for conceptual architecture and practical configurations
* See :doc:`benchmarks_and_datasets` for evaluation metrics and benchmarks
* See :doc:`stage1_retrieval/index` for retrieval method details
* See :doc:`stage2_reranking/index` for reranking architectures
