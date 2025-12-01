MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings
================================================================

:Authors: Laxman Dhulipala, Majid Hadian, Rajesh Jayaram, Jason Lee, Vahab Mirrokni
:Affiliation: Google Research and Google DeepMind
:Publication: NeurIPS 2024
:arXiv: 2405.19504
:Date: May 29, 2024

Overview
--------

MUVERA (Multi-Vector Retrieval Algorithm) introduces a principled approach to reduce multi-vector similarity search to single-vector Maximum Inner Product Search (MIPS), enabling the use of highly optimized off-the-shelf MIPS solvers for multi-vector retrieval tasks. The method achieves **10% improved recall** with **90% lower latency** compared to prior state-of-the-art implementations across BEIR benchmarks.

Problem Statement
-----------------

Traditional IR systems use single-vector embeddings :math:`x \in \mathbb{R}^d`, enabling fast retrieval via optimized MIPS algorithms. Multi-vector models like **ColBERT** produce sets of embeddings per data point (one per token), achieving superior semantic matching through late interaction mechanisms. However, this comes at significant computational cost:

* **Storage overhead**: Vectors must be stored for every token in the corpus
* **Retrieval complexity**: Multi-vector similarity requires exhaustive token-level comparisons
* **Scoring inefficiency**: MaxSim operator :math:`\sum_{q \in Q} \max_{p \in P} \langle q, p \rangle` is computationally expensive

**Chamfer Similarity**: The multi-vector similarity metric used by ColBERT and related models, defined as:

.. math::

   \text{Chamfer}(Q, P) = \sum_{q \in Q} \max_{p \in P} \langle q, p \rangle

where :math:`Q` and :math:`P` are sets of token embeddings for query and document respectively.

Core Innovation: Fixed Dimensional Encodings (FDEs)
----------------------------------------------------

MUVERA's breakthrough lies in asymmetrically transforming multi-vector representations into single vectors whose inner product approximates the original Chamfer similarity. The transformation produces **Fixed Dimensional Encodings** where:

.. math::

   \langle F_q(Q), F_{doc}(P) \rangle \approx \text{Chamfer}(Q, P)

**Key insight**: If we knew the optimal matching :math:`\pi: Q \to P` (which query token matches which document token), we could concatenate matched pairs and compute their similarities. Since we don't know :math:`\pi` a priori, MUVERA partitions the embedding space into regions and computes local aggregations within each region.

FDE Construction Algorithm
---------------------------

The FDE generation process consists of four stages:

1. Space Partitioning via Locality Sensitive Hashing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MUVERA employs **SimHash** to partition the :math:`d`-dimensional embedding space into :math:`2^b` buckets in a data-oblivious manner:

* Sample :math:`b` random Gaussian vectors :math:`h_1, \ldots, h_b \sim \mathcal{N}(0, I_d)`
* For vector :math:`v`, compute bucket assignment: :math:`\varphi(v) = \text{binary}(\text{sign}(\langle v, h_1 \rangle), \ldots, \text{sign}(\langle v, h_b \rangle))`
* This creates :math:`2^b` regions where similar vectors (by angular distance) likely land in the same bucket

**Rationale**: LSH ensures that for each :math:`q \in Q`, its closest match :math:`p \in P` lands in the same cluster with high probability, enabling decomposition of Chamfer similarity into cluster-wise computations.

2. Dimensionality Reduction via Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each bucket :math:`i \in [2^b]`:

* Collect all vectors from the multi-vector set that hash to bucket :math:`i`
* Apply random projection matrix :math:`R_i \in \mathbb{R}^{d_{sub} \times d}` to reduce dimensionality from :math:`d` to :math:`d_{sub}`
* Aggregate (typically via summation) to create sub-vector :math:`v_i \in \mathbb{R}^{d_{sub}}`

**Handling empty buckets**: Project zero vector for empty buckets, ensuring every partition contributes to the final encoding.

3. Repetition for Improved Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To improve approximation quality, repeat the partitioning process :math:`r` times with different random hash functions :math:`\varphi_1, \ldots, \varphi_r`. This is analogous to using multiple hash tables in standard LSH for improved recall.

4. Concatenation
~~~~~~~~~~~~~~~~

Concatenate all sub-vectors from all repetitions:

.. math::

   F(X) = [v_{1,1}, \ldots, v_{1,2^b}, v_{2,1}, \ldots, v_{r,2^b}] \in \mathbb{R}^{r \cdot 2^b \cdot d_{sub}}

The final FDE dimensionality is :math:`d_{FDE} = r \cdot 2^b \cdot d_{sub}`, which is **fixed** regardless of document/query length.

Theoretical Guarantees
----------------------

**Theorem 2.1** (Simplified): For unit vectors, given error tolerance :math:`\epsilon, \delta > 0`, and query size :math:`m = |Q|`, there exists a choice of parameters :math:`(b, r, d_{sub})` such that with probability at least :math:`1-\delta`:

.. math::

   |\langle F_q(Q), F_{doc}(P) \rangle - \text{Chamfer}(Q, P)| \leq \epsilon \cdot m

**Key properties**:

* **Additive approximation**: Error scales linearly with query size, not quadratically
* **Tunable precision**: Increasing :math:`b`, :math:`r`, or :math:`d_{sub}` improves approximation at the cost of higher dimensionality
* **First theoretical guarantee**: MUVERA provides the first single-vector proxy for multi-vector similarity with provable approximation bounds

Asymmetric Encoding Strategy
-----------------------------

MUVERA uses **different encoding strategies** for queries vs documents:

* **Documents**: Use more aggressive compression (lower :math:`r`, :math:`b`, or :math:`d_{sub}`) since they're indexed once
* **Queries**: Can afford slightly higher dimensionality since encoding happens at query time
* This asymmetry optimizes the storage-latency tradeoff

Two-Stage Retrieval Pipeline
-----------------------------

1. **Stage 1 - Approximate retrieval**:
   
   * Transform query into FDE: :math:`F_q(Q)`
   * Use off-the-shelf MIPS solver (e.g., DiskANN, HNSW) to retrieve top-k candidates based on :math:`\langle F_q(Q), F_{doc}(P) \rangle`
   * Achieves 2-5× fewer candidate retrievals compared to prior heuristics

2. **Stage 2 - Exact reranking**:
   
   * For retrieved candidates, compute exact Chamfer similarity using original token embeddings
   * Re-rank by exact scores
   * Return final ranked list

This two-stage approach combines the efficiency of single-vector search with the accuracy of multi-vector scoring.

Empirical Results
-----------------

Evaluated on BEIR benchmark suite:

* **Recall improvement**: +10% average recall compared to PLAID (prior SOTA)
* **Latency reduction**: 90% lower latency than PLAID
* **Candidate efficiency**: FDEs retrieve 2-5× fewer candidates while maintaining same recall as single-vector heuristics
* **Variance**: FDEs show lower variance across different parameter settings compared to single-vector pooling baselines
* **QPS vs Recall**: Consistently superior Pareto frontier across datasets

Implementation Details
----------------------

* **Base embeddings**: ColBERTv2 token embeddings (dimension :math:`d=128`)
* **SimHash parameters**: Typically :math:`b \in [4, 8]` bits (16-256 buckets)
* **Repetitions**: :math:`r \in [2, 8]`
* **Sub-vector dimension**: :math:`d_{sub} \in [16, 64]`
* **Final FDE dimension**: Ranges from 512 to 4096 depending on configuration
* **MIPS backend**: DiskANN with graph-based indexing
* **Compression**: Optional Product Quantization (PQ) for further storage reduction

Connections to Existing Work
-----------------------------

**Relationship to ColBERT**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* MUVERA directly addresses ColBERT's computational bottleneck
* Preserves ColBERT's semantic richness through approximation-theoretic approach
* Enables deployment of ColBERT-quality retrieval at production scale

**Comparison with ColBERTv2 and PLAID**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **ColBERTv2**: Uses centroid-based compression with residuals; MUVERA avoids centroid training and uses data-oblivious LSH
* **PLAID**: Uses centroid interaction mechanism and pruning; MUVERA reduces problem to standard MIPS, enabling use of any MIPS solver
* **Storage**: MUVERA FDEs require similar storage to compressed ColBERTv2 but with simpler indexing

**Alternative Multi-Vector Approaches**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **ConstBERT**: Learns fixed pooling to reduce token embeddings; MUVERA is training-free and works with any multi-vector model
* **Token Pooling**: Clusters similar tokens at indexing; MUVERA uses LSH-based bucketing with theoretical guarantees
* **Static Pruning**: Removes low-impact token embeddings; MUVERA compresses all tokens into fixed dimension

**Late Interaction Models**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MUVERA generalizes to other late interaction architectures:

* **ColPali**: Multi-vector embeddings for multimodal (text + image) retrieval
* **ColQwen**: Multilingual late interaction model
* Any model using MaxSim operator can benefit from MUVERA's FDE transformation

Integration into RAG Pipelines
-------------------------------

MUVERA fits naturally into modern Retrieval-Augmented Generation systems:

**Stage 1: Initial Retrieval**
  * Convert corpus documents to FDEs offline
  * Store FDEs in MIPS-optimized vector database (FAISS, Weaviate, Qdrant, Milvus)
  * At query time, encode query to FDE and retrieve top-k candidates via MIPS

**Stage 2: Re-ranking** (optional but recommended)
  * Retrieve original token embeddings for candidates
  * Compute exact Chamfer similarity
  * Re-rank by exact scores
  * Alternatively, use cross-encoder for final re-ranking

**Stage 3: Generation**
  * Pass top-ranked documents to LLM for response generation

**Advantages over traditional RAG**:

* Token-level semantic matching vs sentence-level
* Better handling of long documents (each token contributes independently)
* Improved recall for complex queries with multiple concepts
* Faster than multi-stage re-ranking with cross-encoders alone

Practical Considerations
-------------------------

**When to use MUVERA**:

* Large-scale retrieval systems (millions+ documents)
* Latency-critical applications
* When semantic precision matters (e.g., question answering, fact verification)
* Systems already using ColBERT but facing scaling challenges

**Trade-offs**:

* **Index size**: FDE dimension (512-4096) is larger than single-vector embeddings (128-1024) but much smaller than full multi-vector storage
* **Two-stage pipeline**: Requires storing both FDEs and original token embeddings for exact reranking
* **Parameter tuning**: Requires experimentation to find optimal :math:`(b, r, d_{sub})` for specific use case

**Optimization opportunities**:

* **Hybrid retrieval**: Combine FDE-based retrieval with sparse retrieval (BM25) for robustness
* **Cascading**: Use FDE for first-stage, coarse-grained reranking for second stage, cross-encoder for final stage
* **Adaptive parameters**: Different :math:`(b, r, d_{sub})` for different document types or query complexities

Open Questions and Future Directions
-------------------------------------

1. **Learned partitioning**: Can we learn data-dependent hash functions that improve over SimHash?

2. **Dynamic FDEs**: Adapting FDE parameters at query time based on query characteristics

3. **Multimodal extension**: Applying FDE transformation to multimodal embeddings (text + image + audio)

4. **Streaming updates**: Efficient incremental index updates for dynamic document collections

5. **Cross-encoder distillation**: Training cross-encoders that operate directly on FDEs

6. **Compression-aware training**: Fine-tuning ColBERT models to be more amenable to FDE compression

Code and Resources
-------------------

* **Official paper**: https://arxiv.org/abs/2405.19504
* **NeurIPS 2024 proceedings**: https://dl.acm.org/doi/10.5555/3737916.3741120
* **Blog post**: https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/

**Relevant implementations**:

* **Weaviate**: Native multi-vector support with MUVERA-inspired optimizations
* **Qdrant**: Multi-vector indexing with MaxSim operators
* **LanceDB**: Multivector search implementation
* **Milvus**: ColPali and ColBERT integration

Key Takeaways
-------------

1. **Principled compression**: MUVERA provides the first theoretically-grounded approach to compress multi-vector embeddings into fixed-dimensional single vectors

2. **Efficiency gains**: 90% latency reduction and 10% recall improvement demonstrate the practical impact of theory-driven design

3. **Algorithmic versatility**: By reducing multi-vector search to MIPS, MUVERA enables the use of decades of MIPS optimization research

4. **Production readiness**: The method is simple enough to implement and efficient enough to deploy at scale

5. **RAG enhancement**: Multi-vector retrieval via MUVERA offers a compelling upgrade path for RAG systems seeking better semantic matching without sacrificing latency

Citation
--------

.. code-block:: bibtex

   @inproceedings{dhulipala2024muvera,
     title={MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings},
     author={Dhulipala, Laxman and Hadian, Majid and Jayaram, Rajesh and Lee, Jason and Mirrokni, Vahab},
     booktitle={Advances in Neural Information Processing Systems},
     year={2024},
     volume={37},
     url={https://arxiv.org/abs/2405.19504}
   }

Related Papers in This Repository
----------------------------------

* **ColBERT** - Foundational late interaction model that MUVERA optimizes
* **ColBERTv2** - Centroid-based compression approach; compare with MUVERA's LSH-based method
* **Cross-encoder re-ranking** - Can be used in Stage 2 after MUVERA retrieval
* **Dense retrieval models** (DPR, ANCE, etc.) - Single-vector approaches that MUVERA outperforms
* **Hybrid retrieval** - MUVERA can be combined with sparse retrieval for enhanced robustness
* **ConstBERT** - Alternative fixed-vector approach for multi-vector models
* **PLAID** - Prior SOTA for ColBERT efficiency; MUVERA supersedes this approach

Connections to Advanced Retrieval Concepts
-------------------------------------------

**Retrieval Stages**:
  * MUVERA operates at **Stage 1** (candidate generation)
  * Naturally integrates with **Stage 2** re-ranking (cross-encoders, etc.)
  * Complements metadata filtering and query rewriting techniques

**Embedding Models**:
  * **Single-vector models**: MUVERA supersedes by leveraging multi-vector richness
  * **Multi-vector models**: MUVERA makes practical at scale
  * **Hybrid models**: MUVERA can work alongside sparse methods (BM25, SPLADE)

**Indexing Strategies**:
  * Leverages HNSW, DiskANN, or other MIPS-optimized indexes
  * Supports Product Quantization for further compression
  * Enables sharding and distributed retrieval

**Similarity Metrics**:
  * Approximates **Chamfer similarity** (MaxSim aggregation)
  * Related to **Hausdorff distance** in metric space theory
  * Connects to **optimal transport** theory (approximate matching)
