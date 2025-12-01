Literature Survey: Re-ranking Methods
======================================

This section contains detailed surveys and analyses of individual papers related to 
Stage 2 re-ranking methods, including cross-encoders, late interaction models, and 
LLM-based rerankers.

.. toctree::
   :maxdepth: 2
   :caption: Papers:

   muvera-multi-vector-retrieval

Overview
--------

The papers in this section provide in-depth technical analysis of key contributions 
to the re-ranking literature. Each survey includes:

* **Problem formulation** and mathematical foundations
* **Algorithmic innovations** with theoretical guarantees
* **Empirical results** on standard benchmarks
* **Practical considerations** for deployment
* **Connections** to other methods in the retrieval-reranking pipeline

Featured Papers
---------------

**MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings** (NeurIPS 2024)
    A principled approach to reduce multi-vector similarity search to single-vector 
    MIPS, achieving 10% improved recall with 90% lower latency compared to prior 
    state-of-the-art. Enables ColBERT-quality retrieval at production scale.

Contributing
------------

To add a new paper survey to this section:

1. Create a new ``.rst`` file following the structure of existing surveys
2. Include: problem statement, core innovation, theoretical analysis, empirical results
3. Add the file to the toctree above
4. Ensure proper citations and links to related papers

