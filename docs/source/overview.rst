Overview
========

The New Bottleneck: Role of Advanced Negative Mining in Dense Retrieval Evaluation
--------------------------------------------------------------------------------------

The Paradigm Shift from Sparse to Dense Retrieval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The field of information retrieval has undergone a fundamental paradigm shift. For decades, 
retrieval systems were dominated by sparse, lexical-based methods like BM25. These approaches, 
while robust and efficient, are limited by their reliance on exact keyword matching. They 
struggle to capture the underlying semantic intent of a query, failing when users employ 
different terminology (the "vocabulary mismatch" problem).

The advent of pre-trained language models (PLMs) such as BERT introduced the era of 
**dense retrieval**. Instead of sparse vectors of word counts, dense retrievers map queries 
and documents into low-dimensional, continuous-valued vectors (embeddings). These embeddings 
capture semantic relevance, allowing a model to retrieve documents that are contextually 
related to a query, even if they share no keywords.

However, this power comes with a critical dependency. The performance of these dense models 
is not just a function of their architecture (e.g., BERT, Sentence Transformers) but is 
overwhelmingly reliant on the quality of the data used during their contrastive or 
multi-negative training. 

The Central Challenge
^^^^^^^^^^^^^^^^^^^^^

The central challenge has shifted from lexical matching to a new, more difficult problem: 
**teaching the model to distinguish between genuine semantic relevance and mere semantic similarity**.

This is the core bottleneck in modern dense retrieval systems, and the reason why advanced 
negative mining techniques have become non-negotiable for achieving state-of-the-art performance.

Key Takeaways
^^^^^^^^^^^^^

* Sparse methods (BM25) rely on keyword matching and suffer from vocabulary mismatch
* Dense retrieval uses embeddings to capture semantic relevance
* Model performance depends critically on training data quality
* The main challenge is distinguishing semantic relevance from semantic similarity

