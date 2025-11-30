Joint Learning of Retrieval and Indexing
=========================================

Traditional approach: train model, then build index separately. Joint learning optimizes 
both simultaneously for better end-to-end performance.

The Motivation
--------------

**Traditional Pipeline:**

1. Train query encoder
2. Train document encoder
3. Encode all documents
4. Build index (e.g., Product Quantization, LSH)
5. **Problem**: Index is oblivious to encoder; encoder is oblivious to index compression

**Joint Learning:**

1. Train encoder *with awareness* of how it will be indexed
2. Or: train index *with awareness* of encoder's embedding distribution
3. **Result**: Better quality after compression

Joint Learning Literature
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance (JPQ) <https://arxiv.org/abs/2108.00644>`_
     - Zhan et al.
     - CIKM 2021
     - `Code <https://github.com/jingtaozhan/JPQ>`_
     - **Joint Query-Index Optimization**: Optimizes query encoder jointly with product quantization index. 30x compression with minimal accuracy loss. Query learns to work with compressed index.
   * - `End-to-End Learning of Hierarchical Index for Efficient Dense Retrieval (Poeem/EHI) <https://arxiv.org/abs/2310.08891>`_
     - arXiv Authors
     - arXiv 2023
     - NA
     - **Differentiable Quantization**: Makes index structure differentiable so gradients flow from retrieval loss to index parameters. End-to-end learning of hierarchical index structure.

Why This Matters
-----------------

**The Index Compression Problem:**

* Dense vectors are large: 10M docs × 768-d × 4 bytes = 30GB
* Product Quantization (PQ) compresses 30x: ~1GB
* But compression loses information → accuracy drops

**Joint Learning Solution:**

Train the encoder to produce embeddings that:

* Are robust to PQ compression
* Maintain discriminative power after quantization
* Align with index structure

**Result**: 30x compression with 2-3% accuracy loss (vs 10-15% without joint learning)

JPQ Implementation Strategy
----------------------------

**Architecture:**

.. code-block:: python

   # Conceptual JPQ training
   
   # 1. Encode query
   query_emb = query_encoder(query)
   
   # 2. Encode document
   doc_emb = doc_encoder(document)
   
   # 3. Quantize document (PQ compression)
   doc_quantized = product_quantize(doc_emb)
   
   # 4. Compute similarity with quantized version
   score = dot_product(query_emb, doc_quantized)
   
   # 5. Loss includes reconstruction error
   loss = retrieval_loss(score, labels) \\
        + lambda * reconstruction_loss(doc_emb, doc_quantized)
   
   # Query encoder learns to work with compressed docs

**Key Insight**: Query encoder adapts to index imperfections.

When to Use Joint Learning
---------------------------

✅ **Use When:**

* Index size is critical (edge devices, cost optimization)
* Serving from memory (want 10-100x compression)
* Have engineering resources for custom indexing
* Accuracy loss from compression is significant

❌ **Don't Use When:**

* Disk storage is cheap (just use larger index)
* Accuracy is paramount (compression always loses some quality)
* Using standard FAISS (works well without joint learning)
* Quick iteration more important than optimization

The Future: Learned Indexes
----------------------------

Research is moving toward **fully learned index structures**:

* Neural networks that predict document locations
* Differentiable routing in hierarchical indexes
* Learned quantization codebooks optimized for retrieval

This is still early-stage research but promises to close the gap between dense vector 
search and traditional inverted indexes.

Next Steps
----------

* See :doc:`dense_baselines` for standard (non-joint) training
* See :doc:`late_interaction` for ColBERT's indexing approach
* See :doc:`hard_mining` for improving training data quality

