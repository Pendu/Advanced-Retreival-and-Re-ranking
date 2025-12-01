Literature Survey: Retrieval Methods
=====================================

This section contains detailed surveys and analyses of individual papers related to 
Stage 1 retrieval methods, including dense retrieval, sparse retrieval, hybrid 
approaches, and hard negative mining strategies.

.. toctree::
   :maxdepth: 2
   :caption: Papers:

Overview
--------

The papers in this section provide in-depth technical analysis of key contributions 
to the retrieval literature. Each survey includes:

* **Problem formulation** and mathematical foundations
* **Algorithmic innovations** with theoretical guarantees
* **Empirical results** on standard benchmarks
* **Practical considerations** for deployment
* **Connections** to other methods in the retrieval-reranking pipeline

Topics Covered
--------------

* **Dense Retrieval**: DPR, ANCE, Contriever, and embedding-based methods
* **Sparse Retrieval**: BM25, SPLADE, and learned sparse representations
* **Hard Negative Mining**: Dynamic mining, curriculum learning, false negative handling
* **Hybrid Methods**: Combining dense and sparse retrieval for robustness
* **Pre-training**: ICT, contrastive pre-training, and domain adaptation

Contributing
------------

To add a new paper survey to this section:

1. Create a new ``.rst`` file following the structure of existing surveys
2. Include: problem statement, core innovation, theoretical analysis, empirical results
3. Add the file to the toctree above
4. Ensure proper citations and links to related papers

