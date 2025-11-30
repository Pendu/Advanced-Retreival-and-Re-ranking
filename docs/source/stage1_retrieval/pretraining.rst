Pre-training Methods for Dense Retrievers
==========================================

Standard dense retrievers start from general pre-trained models (BERT, RoBERTa) and 
fine-tune on retrieval tasks. However, specialized pre-training strategies can create 
better initialization, leading to stronger final performance.

Why Pre-training Matters for Retrieval
---------------------------------------

**The Problem with Standard BERT**

BERT was pre-trained for masked language modeling and next sentence prediction—tasks 
quite different from "is this passage relevant to this query?"

**The Solution**

Pre-train specifically for retrieval using:

* Unsupervised contrastive learning on documents
* Inverse tasks (generate query from passage)
* Retrieval-augmented objectives
* Corpus-aware objectives

Pre-training Methods Literature
--------------------------------

Unsupervised Pre-training
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Latent Retrieval for Weakly Supervised Open Domain QA (ORQA) <https://arxiv.org/abs/1906.00300>`_
     - Lee et al.
     - ACL 2019
     - `Code <https://github.com/google-research/language/tree/master/language/orqa>`_
     - **Inverse Cloze Task (ICT)**: Pre-trains by predicting which passage a sentence came from. Generates pseudo-queries automatically. Unsupervised data generation at scale.
   * - `Unsupervised Dense Information Retrieval with Contrastive Learning (Contriever) <https://arxiv.org/abs/2112.09118>`_
     - Izacard et al.
     - TMLR 2022
     - `Code <https://github.com/facebookresearch/contriever>`_
     - **Contrastive + Augmentation**: Robust unsupervised features via contrastive learning and aggressive data augmentation. State-of-the-art zero-shot retrieval. No labels needed!

Supervised Retrieval-Augmented Pre-training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `REALM: Retrieval-Augmented Language Model Pre-Training <https://arxiv.org/abs/2002.08909>`_
     - Guu et al.
     - ICML 2020
     - `Code <https://github.com/google-research/language/tree/master/language/realm>`_
     - **End-to-End Retrieval Pre-training**: Jointly pre-trains retriever and language model. Index refreshed during pre-training. Computationally heavy but powerful for end-to-end QA.

Architecture-Aware Pre-training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `Condenser: a Pre-training Architecture for Dense Retrieval <https://arxiv.org/abs/2104.08253>`_
     - Gao & Callan
     - EMNLP 2021
     - `Code <https://github.com/luyug/Condenser>`_
     - **Skip-Connection Head**: Architectural modification that forces global information into CLS token. Makes CLS better suited for representing entire document.
   * - `Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval (coCondenser) <https://arxiv.org/abs/2108.05540>`_
     - Gao & Callan
     - ACL 2022
     - `Code <https://github.com/luyug/Condenser>`_
     - **Corpus-Aware Contrastive**: Unsupervised contrastive learning at corpus level. Aligns document spans without labels. Strong zero-shot performance.

Multi-View Pre-training
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 12 8 10 45

   * - Paper
     - Author
     - Venue
     - Code
     - Key Innovation
   * - `MVR: Multi-View Representation for Dense Retrieval <https://arxiv.org/abs/2104.07652>`_
     - Zhang et al.
     - arXiv 2021
     - NA
     - **Multi-View Generation**: Learns explicit views for different retrieval intents. Anti-collapse regularization prevents views from becoming identical. Handles diverse information needs.

Comparison: Fine-tuning vs Pre-training
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Dimension
     - Standard Fine-tuning
     - Specialized Pre-training
   * - **Starting Point**
     - General BERT/RoBERTa
     - Retrieval-optimized model
   * - **CLS Token**
     - Optimized for MLM
     - Optimized for full doc representation
   * - **Zero-shot**
     - Poor (~0.3 MRR)
     - Good (~0.45 MRR)
   * - **Fine-tuning Data**
     - Needs more data
     - Needs less data
   * - **Training Cost**
     - Lower (fine-tune only)
     - Higher (pre-train + fine-tune)
   * - **Final Performance**
     - Good
     - Better (+3-7%)

When to Use Pre-trained Models
-------------------------------

✅ **Use Retrieval-Specific Pre-training When:**

* Zero-shot performance matters (new domains)
* Limited fine-tuning data available
* Want best possible final performance
* Have computational budget for pre-training

✅ **Use Pre-trained Models (from others):**

Most practical approach: use existing pre-trained models:

.. code-block:: python

   # Modern pre-trained retrievers (recommended)
   model = SentenceTransformer('BAAI/bge-base-en-v1.5')  # Used coCondenser
   model = SentenceTransformer('intfloat/e5-base-v2')     # Multi-stage pre-training
   model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')  # Contriever-based

These are all pre-trained with retrieval-specific objectives and ready to fine-tune.

❌ **Don't Pre-train From Scratch Unless:**

* You have unique domain with massive unlabeled data
* You're doing research on pre-training itself
* Existing models fail completely on your domain

Recommended Practice
--------------------

**For Most Projects:**

1. Start with **pre-trained retrieval model** (e.g., BGE, E5)
2. Fine-tune on your specific task with good hard negatives
3. Result: 95% of custom pre-training performance, 10% of cost

**For Research/Scale:**

1. Use **Contriever** or **coCondenser** approach
2. Pre-train on your domain corpus (unsupervised)
3. Then fine-tune on labeled data
4. Result: Best possible performance, significant cost

Implementation Example
-----------------------

**Using Pre-trained Contriever**

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   
   # Load pre-trained with contrastive objective
   model = SentenceTransformer('facebook/contriever')
   
   # Already good zero-shot performance
   results = model.search(query, corpus, top_k=10)
   
   # Fine-tune on your data for even better performance
   from sentence_transformers import losses, InputExample
   
   train_examples = [
       InputExample(texts=[query, positive_passage])
       for query, positive_passage in train_data
   ]
   
   loss = losses.MultipleNegativesRankingLoss(model)
   model.fit(train_examples, epochs=3)

Next Steps
----------

* See :doc:`dense_baselines` for standard fine-tuning approaches
* See :doc:`hard_mining` for improving fine-tuning with better negatives
* See :doc:`joint_learning` for jointly optimizing pre-training and indexing

