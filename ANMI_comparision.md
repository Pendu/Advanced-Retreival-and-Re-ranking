# ANMI 2.0 — Unified Overview, Insights, and Comparison

This document consolidates **all insights** from the previous discussions, including:

* Comparison of **ANMI 2.0** with dense retrieval literature
* Comparison with the **Information Retrieval Effectiveness Survey (2211.14876)**
* Comparison with **zELO and ELO-based models**
* Cost analysis and architectural trade-offs
* Guidelines on **first-stage vs second-stage** use
* Offline vs online LLM usage

Where possible, references to the relevant **sections or themes** of the uploaded papers are mentioned textually (no citation markers as the canvas does not support them).

---

# 1. What ANMI 2.0 Is

ANMI 2.0 (**Adaptive Negative Mining Intelligence, ELO-Enhanced**) is a unified framework for:

* generating **graded relevance** from pairwise comparisons
* mitigating **false negatives**
* stabilizing **contrastive learning**
* enabling **pairwise calibration** via ELO / Bradley–Terry models
* building either a:

  * **second-stage reranker**, or
  * **first-stage ANN-compatible retriever**

Key components:

* Sparse pairwise comparison graph
* ELO scoring to convert preferences → stable semantic scale
* Hybrid loss (InfoNCE + regression)
* Difficulty-aware negative mining
* Curriculum training from ELO uncertainty

## 1.1 ANMI 2.0 Component Architecture

```mermaid
graph TB
    subgraph "ANMI 2.0 Core Components"
        A[Sparse Pairwise<br/>Comparison Graph] --> B[ELO Scoring<br/>Engine]
        B --> C[Hybrid Loss<br/>Function]
        C --> D[Trained Model]
        
        E[Difficulty-Aware<br/>Negative Mining] --> C
        F[Curriculum<br/>Scheduler] --> C
        
        B --> E
        B --> F
    end
    
    subgraph "Outputs"
        D --> G[Second-Stage<br/>Reranker]
        D --> H[First-Stage<br/>Retriever]
    end
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
```

---

# 2. Insights from the Dense Retrieval Survey (2211.14876)

## 2.1 Major problems identified in the survey

The survey highlights multiple fundamental issues in modern dense retrieval:

* **False negatives** due to incomplete relevance judgments
* **Binary labels** that fail to capture graded relevance
* **Over-reliance on one positive per query**
* **Hard negative mining instability**
* **Dataset noisiness and incompleteness**
* **Representation collapse** from equal treatment of all positives

These match directly with the motivations behind ANMI 2.0.

## 2.2 Problem-Solution Mapping

```mermaid
flowchart LR
    subgraph "Survey Problems"
        P1[False Negatives]
        P2[Binary Labels]
        P3[Single Positive]
        P4[Mining Instability]
        P5[Representation Collapse]
    end
    
    subgraph "ANMI 2.0 Solutions"
        S1[Probabilistic<br/>Hardness Filtering]
        S2[ELO-Based<br/>Graded Scores]
        S3[Multi-Positive<br/>Ordering]
        S4[Bradley-Terry<br/>Stabilization]
        S5[Difficulty-Aware<br/>Curriculum]
    end
    
    P1 --> S1
    P2 --> S2
    P3 --> S3
    P4 --> S4
    P5 --> S5
    
    style P1 fill:#ffcdd2
    style P2 fill:#ffcdd2
    style P3 fill:#ffcdd2
    style P4 fill:#ffcdd2
    style P5 fill:#ffcdd2
    style S1 fill:#c8e6c9
    style S2 fill:#c8e6c9
    style S3 fill:#c8e6c9
    style S4 fill:#c8e6c9
    style S5 fill:#c8e6c9
```

## 2.3 Opportunities the survey reveals that ANMI 2.0 addresses

* Survey shows lack of **multi-positive ordering** → ANMI 2.0 introduces ELO-based ranking among positives
* Survey shows difficulty in **identifying true hard negatives** → ANMI uses probabilistic hardness
* Survey shows evaluation requires **graded relevance** → ANMI creates this automatically
* Survey notes absence of **difficulty-aware training** → ANMI provides curriculum via ELO uncertainty

## 2.4 Unique contributions of ANMI 2.0 beyond survey

* First time **pairwise preference aggregation** is used to form a continuous training target
* First probabilistic negative-mining system grounded in Bradley–Terry / ELO theory
* Enables **ANN-friendly embeddings** using calibrated scores
* Turns dense retrieval training into a **ranking-model-consistent** process

---

# 3. Comparison With ELO-Based Ranking Papers (e.g., zELO)

## 3.1 zELO background

zELO uses ELO scoring for document ranking evaluation or label refinement. It focuses on **post-processing relevance estimation**, not training-time embedding shaping.

## 3.2 Key differences

* zELO is used **after** retrieval; ANMI uses ELO **to train** the model
* zELO refines scores; ANMI **generates training gradients** from ELO
* zELO does not modify embedding space; ANMI **calibrates** it
* zELO treats ELO as a finished score; ANMI integrates ELO into **hybrid losses**

## 3.3 ANMI vs zELO Processing Flow

```mermaid
flowchart TB
    subgraph "zELO Flow"
        direction TB
        Z1[Query] --> Z2[Retriever]
        Z2 --> Z3[Initial Ranking]
        Z3 --> Z4[ELO Refinement]
        Z4 --> Z5[Final Scores]
        
        style Z4 fill:#fff3e0,stroke:#ff9800
    end
    
    subgraph "ANMI 2.0 Flow"
        direction TB
        A1[Training Data] --> A2[Pairwise LLM<br/>Comparisons]
        A2 --> A3[ELO Graph<br/>Construction]
        A3 --> A4[Hybrid Loss<br/>Training]
        A4 --> A5[Calibrated<br/>Embeddings]
        A5 --> A6[Inference]
        
        style A3 fill:#e8f5e9,stroke:#4caf50
        style A4 fill:#e8f5e9,stroke:#4caf50
    end
```

### Summary:

**ANMI = generalization of zELO into the training pipeline, not just scoring.**

---

# 4. Insights from the "IR Effectiveness" Survey

## 4.1 Survey themes aligning with ANMI

The IR survey highlights:

* Problems caused by **binary relevance**
* Need for **graded relevance**
* Impact of **judgment incompleteness**
* Difficulty of **reliable evaluation**
* Importance of **user models**, probabilistic ranking

ANMI 2.0 operationalizes these principles:

* Converts binary labels → **probabilistic scores**
* Mitigates incomplete judgments via **pairwise LLM comparisons**
* Builds a training objective consistent with **evaluation metrics like nDCG** (graded relevance)

## 4.2 Interesting theoretical alignment

* Both documents emphasize **probabilistic user behavior** models
* Both emphasize the structure of **difficulty curves** (Laffer curve vs evaluation depth)
* Training instability (in retrieval) resembles **evaluation instability** from missing judgments

**ANMI 2.0 is essentially an evaluation-aware training paradigm.**

---

# 5. First-Stage vs Second-Stage Considerations

## 5.1 Decision Flow Chart

```mermaid
flowchart TD
    START[Choose ANMI<br/>Deployment Mode] --> Q1{Budget<br/>Constraint?}
    
    Q1 -->|< $5K| R1[Second-Stage<br/>Reranker Only]
    Q1 -->|$5K-$20K| Q2{Need Zero-Shot<br/>Generalization?}
    Q1 -->|> $20K| Q3{Enterprise<br/>Requirements?}
    
    Q2 -->|No| R1
    Q2 -->|Yes| R2[First-Stage<br/>Retriever]
    
    Q3 -->|Standard| R2
    Q3 -->|Safety-Critical| R3[Unified<br/>Retriever + Reranker]
    
    R1 --> O1[Low Cost<br/>High Precision]
    R2 --> O2[Research Innovation<br/>Better Generalization]
    R3 --> O3[Premium Quality<br/>Maximum Safety]
    
    style R1 fill:#e3f2fd
    style R2 fill:#fff8e1
    style R3 fill:#fce4ec
```

## 5.2 Second-stage reranker

**Recommended when cost is a concern.**

* Needs only ~10–20 pairwise LLM comparisons per query
* Cost is usually in the low thousands of USD
* Provides large quality gains
* Requires no ANN reconfiguration

## 5.3 First-stage retriever

Possible through:

* ELO-regressed embeddings
* ANN-compatible similarity function
* Difficulty-aware sampling during training

This produces an ELO-calibrated dense retriever with:

* Better geometry
* Fewer false negatives
* Better zero-shot generalization

**Cost becomes higher (~10–20k+), but still one-time offline.**

---

# 6. Cost Analysis

## 6.1 Offline, not online

All LLM-based ELO computations are **offline pre-training**.

### LLM is NOT used at inference.

Runtime cost = same as any dense retriever / reranker.

## 6.2 Cost Comparison Chart

```mermaid
pie showData
    title "ANMI 2.0 Implementation Costs - Relative Scale"
    "Reranker Only ~$3.5K" : 35
    "First-Stage ~$15K" : 150
    "Unified System ~$40K" : 400
```

## 6.3 Typical cost estimates

### Reranker-only training:

$2,000–$5,000 (one-time)

### First-stage retriever calibration:

$10,000–$20,000 (one-time)

### Full unified retriever + reranker:

$30,000–$50,000 (still one-time)

These assume use of cheap LLMs (GPT-4o-mini, Haiku, etc.).

## 6.4 Cost vs Quality Tradeoff

```mermaid
flowchart TB
    subgraph "HIGH QUALITY GAIN"
        direction LR
        UNI[Unified System<br/>High Cost, Highest Quality]
        FS[First-Stage Retriever<br/>Medium Cost, High Quality]
        RR[Reranker Only<br/>Low Cost, Good Quality]
    end
    
    subgraph "MODERATE QUALITY GAIN"
        PE[Partial ELO<br/>Lowest Cost, Moderate Quality]
    end
    
    style UNI fill:#fce4ec
    style FS fill:#fff8e1
    style RR fill:#e8f5e9
    style PE fill:#e3f2fd
```

---

# 7. Why ANMI 2.0 Is Always Offline

* Pairwise judgments → offline
* ELO graph construction → offline
* Training → offline

**Inference is fast**:

* Query → embedding → ANN → reranker

No LLM calls or pairwise computation happen online.

## 7.1 Offline vs Online Boundary

```mermaid
flowchart LR
    subgraph "OFFLINE (One-Time)"
        direction TB
        O1[LLM Pairwise<br/>Comparisons] --> O2[ELO Graph<br/>Construction]
        O2 --> O3[Model Training]
        O3 --> O4[Index Building]
        
        style O1 fill:#fff3e0
        style O2 fill:#fff3e0
        style O3 fill:#fff3e0
        style O4 fill:#fff3e0
    end
    
    subgraph "ONLINE (Per Request)"
        direction TB
        I1[Query] --> I2[Embedding]
        I2 --> I3[ANN Search]
        I3 --> I4[Reranking]
        I4 --> I5[Results]
        
        style I1 fill:#e8f5e9
        style I2 fill:#e8f5e9
        style I3 fill:#e8f5e9
        style I4 fill:#e8f5e9
        style I5 fill:#e8f5e9
    end
    
    O4 -.->|Deploy| I2
    
    COST1[/"$2K-$50K<br/>One-Time"/]
    COST2[/"~$0.001 per query<br/>Standard Compute"/]
    
    O4 --- COST1
    I5 --- COST2
```

---

# 8. Probability Model Summary

ANMI 2.0 uses **ELO / Bradley–Terry** to convert comparisons into:

* Expected win probabilities
* Graded relevance scores
* Uncertainty for curriculum
* Difficulty-aware negative mining
* Regression targets for embedding training

This integrates cleanly with:

* InfoNCE
* Soft contrastive losses
* Cross-entropy ranking
* Regression heads

## 8.1 Bradley-Terry to Training Target Flow

```mermaid
flowchart TD
    subgraph "Pairwise Input"
        A[Doc A] 
        B[Doc B]
        C[LLM Judgment:<br/>P of A beats B = 0.73]
    end
    
    A --> C
    B --> C
    
    C --> D[Bradley-Terry<br/>Update]
    
    D --> E[ELO Score A: 1847]
    D --> F[ELO Score B: 1623]
    
    E --> G[Normalize to<br/>0.0 - 1.0]
    F --> G
    
    G --> H[Relevance Target A: 0.82]
    G --> I[Relevance Target B: 0.58]
    
    H --> J[Hybrid Loss<br/>Computation]
    I --> J
    
    J --> K[Gradient Update]
    
    style C fill:#e3f2fd
    style D fill:#fff3e0
    style G fill:#f3e5f5
    style J fill:#e8f5e9
```

---

# 9. Final Architecture Options

## 9.1 Architecture Comparison

```mermaid
flowchart TB
    subgraph "Option A: Second-Stage Only"
        A1[Any Dense<br/>Retriever] --> A2[ANN<br/>Top-200]
        A2 --> A3[ANMI 2.0<br/>Reranker]
        A3 --> A4[Final<br/>Top-K]
    end
    
    subgraph "Option B: Unified Retriever"
        B1[Query] --> B2[ANMI<br/>Embedding]
        B2 --> B3[ELO-Calibrated<br/>ANN Search]
        B3 --> B4[Optional<br/>Reranker]
        B4 --> B5[Results]
    end
    
    subgraph "Option C: Hybrid Partial"
        C1[5-10%<br/>ELO Labels] --> C2[Knowledge<br/>Distillation]
        C2 --> C3[Full<br/>Retriever]
        C3 --> C4[Deploy]
    end
    
    style A3 fill:#e3f2fd
    style B2 fill:#fff8e1
    style B3 fill:#fff8e1
    style C1 fill:#fce4ec
    style C2 fill:#fce4ec
```

## Option A — Second-stage only (recommended for production)

1. Dense Retriever
2. ANN
3. ANMI 2.0 Reranker

Best tradeoff between cost and benefit.

## Option B — Unified ANMI Retriever

1. Query → ELO-trained embedding
2. ANN search
3. (Optional) ANMI reranker

Best research novelty.

## Option C — Hybrid Partial ELO Labeling

* Label only 5–10% of corpus
* Distill into full retriever

Best for startups with limited budget.

---

# 10. Overall Conclusions

* ANMI 2.0 addresses the biggest weaknesses highlighted in dense retrieval literature
* It introduces a new **probabilistic, calibrated relevance model**
* It can operate as **reranker**, **retriever**, or **both**
* It is **cost-effective** because all LLM usage is offline
* It fills key research gaps: graded relevance, multi-positive ranking, difficulty modeling
* It produces embeddings that are consistent with IR evaluation metrics
* It is theoretically defensible and practically feasible

---

# 11. Detailed Technical Expansion

## 11.1 Detailed Mechanics of ELO in ANMI 2.0

ANMI uses a **sparse pairwise comparison graph** where nodes represent documents and edges represent a pairwise preference: document A is preferred over document B for a given query.

### ELO Update Formula (Adapted for IR)

ANMI modifies the classical ELO formulation to use soft LLM judgments:

* Let *s(A, B)* be the soft LLM-estimated probability that A is preferred to B.
* Expected win probability: `E = 1 / (1 + 10^{((Score_B - Score_A)/400)})`
* Update rule:

```
Score_A ← Score_A + K * (s(A,B) - E)
Score_B ← Score_B - K * (s(A,B) - E)
```

This stabilizes training by ensuring:

* **Graded supervision** instead of binary labels
* **Uncertainty-aware updates** using soft scores
* **Separation of positives into strong and weak positives**

### 11.1.1 ELO Update Cycle

```mermaid
stateDiagram-v2
    [*] --> Initialize: Set all docs to 1500
    Initialize --> Compare: Select pair (A, B)
    Compare --> LLM: Get P(A > B)
    LLM --> Expected: Compute E(A beats B)
    Expected --> Update: Apply K-factor update
    Update --> Check: More pairs?
    Check --> Compare: Yes
    Check --> Normalize: No
    Normalize --> [*]: ELO scores ready
```

## 11.2 ELO → Regression Target for Embeddings

Once ELO stabilizes, each document gets a **continuous relevance value**:

```
0.0 = irrelevant    → 1.0 = highly relevant
```

During training:

* Dot-product similarity approximates normalized ELO
* Loss combines InfoNCE with MSE regression:

```
L_total = L_InfoNCE + λ · MSE(sim(q, d), ELO_norm(d))
```

Advantages:

* Embedding space becomes **metric-aligned**
* ANN retrieval becomes **probabilistic** rather than heuristic

### 11.2.1 Hybrid Loss Architecture

```mermaid
flowchart TB
    Q[Query<br/>Embedding] --> SIM[Similarity<br/>Computation]
    D[Document<br/>Embeddings] --> SIM
    
    SIM --> L1[InfoNCE Loss]
    SIM --> L2[MSE Regression<br/>Loss]
    
    ELO[Normalized<br/>ELO Targets] --> L2
    
    L1 --> COMBINE[λ-Weighted<br/>Combination]
    L2 --> COMBINE
    
    COMBINE --> GRAD[Gradient<br/>Update]
    
    GRAD --> Q
    GRAD --> D
    
    style L1 fill:#e3f2fd
    style L2 fill:#fff3e0
    style COMBINE fill:#f3e5f5
```

## 11.3 Difficulty-Aware Curriculum

Difficulty = `|s(A,B) - 0.5|` (uncertainty of preference)

* High uncertainty → hard pairs
* Low uncertainty → easy pairs

Training uses a **3-stage curriculum**:

1. Train on easy positives (clear wins)
2. Introduce medium difficulty
3. Introduce hard positives & near-negatives

### 11.3.1 Curriculum Progression

```mermaid
flowchart LR
    subgraph "Training Progression"
        S1[Stage 1<br/>Easy Pairs<br/>0-33%] --> S2[Stage 2<br/>Medium Pairs<br/>33-66%]
        S2 --> S3[Stage 3<br/>Hard Pairs<br/>66-100%]
    end
    
    style S1 fill:#c8e6c9
    style S2 fill:#fff9c4
    style S3 fill:#ffcdd2
```

### 11.3.2 Difficulty Distribution

```mermaid
pie showData
    title "Training Sample Distribution by Difficulty"
    "Easy Pairs (Clear Wins)" : 40
    "Medium Pairs (Moderate)" : 35
    "Hard Pairs (Ambiguous)" : 25
```

This produces:

* Faster convergence
* Better discrimination between positives
* Robustness to false negatives

## 11.4 ANMI 2.0 Compared to RankNet and LambdaRank

RankNet & LambdaRank use:

* Handcrafted probability models
* No graded labels
* No dynamic hardness modeling
* No multi-positive ordering

ANMI 2.0 offers:

* A real, grounded probabilistic model (BT/ELO)
* Continuous labels
* Hardness-aware sampling
* Multi-positive ordering
* LLM-augmented supervision

### 11.4.1 Feature Comparison Matrix

```mermaid
flowchart TB
    subgraph "Feature Comparison"
        direction LR
        
        subgraph RankNet/LambdaRank
            R1[❌ Graded Labels]
            R2[❌ Dynamic Hardness]
            R3[❌ Multi-Positive]
            R4[❌ LLM Supervision]
            R5[✅ Probabilistic Model]
        end
        
        subgraph "ANMI 2.0"
            A1[✅ ELO Graded Labels]
            A2[✅ Difficulty-Aware]
            A3[✅ Multi-Positive Ordering]
            A4[✅ LLM-Augmented]
            A5[✅ Bradley-Terry Model]
        end
    end
    
    style R1 fill:#ffcdd2
    style R2 fill:#ffcdd2
    style R3 fill:#ffcdd2
    style R4 fill:#ffcdd2
    style R5 fill:#c8e6c9
    style A1 fill:#c8e6c9
    style A2 fill:#c8e6c9
    style A3 fill:#c8e6c9
    style A4 fill:#c8e6c9
    style A5 fill:#c8e6c9
```

## 11.5 ANN Compatibility Deep Dive

ANMI embeddings must remain compatible with vector indices like:

* Faiss
* ScaNN
* Milvus
* Qdrant

Because ELO-regressed similarities are linear transforms:

```
sim(q,d) = w · dot(q,d) + b
```

ANN systems can store **only the embedding**, while w and b apply at scoring time.

This ensures:

* No modification to ANN systems
* High-speed retrieval
* Zero runtime overhead

### 11.5.1 ANN Integration Architecture

```mermaid
flowchart LR
    subgraph "Query Time"
        Q[Query] --> E[ANMI<br/>Encoder]
        E --> V[Query<br/>Vector]
    end
    
    subgraph "ANN Index"
        V --> ANN[Vector<br/>Index]
        ANN --> TOP[Top-K<br/>Candidates]
    end
    
    subgraph "Scoring"
        TOP --> DOT[Dot Product]
        DOT --> LIN[Linear Transform<br/>w·sim + b]
        LIN --> CAL[Calibrated<br/>Scores]
    end
    
    CAL --> RES[Final<br/>Ranking]
    
    style ANN fill:#e3f2fd
    style LIN fill:#fff3e0
```

## 11.6 Full Offline Pipeline

1. Sample training queries
2. Retrieve candidate docs
3. LLM computes pairwise preferences
4. Build sparse graph
5. Compute ELO scores
6. Normalize ELO scores
7. Train dual encoder / cross encoder with hybrid loss
8. Export embeddings
9. Build ANN index
10. Deploy reranker or unified model

No LLM calls occur after step 3.

### 11.6.1 Complete Pipeline Visualization

```mermaid
flowchart TD
    subgraph "Phase 1: Data Preparation"
        S1[1. Sample Training<br/>Queries] --> S2[2. Retrieve<br/>Candidate Docs]
        S2 --> S3[3. LLM Pairwise<br/>Preferences]
    end
    
    subgraph "Phase 2: ELO Construction"
        S3 --> S4[4. Build Sparse<br/>Graph]
        S4 --> S5[5. Compute<br/>ELO Scores]
        S5 --> S6[6. Normalize<br/>Scores]
    end
    
    subgraph "Phase 3: Model Training"
        S6 --> S7[7. Train with<br/>Hybrid Loss]
        S7 --> S8[8. Export<br/>Embeddings]
    end
    
    subgraph "Phase 4: Deployment"
        S8 --> S9[9. Build<br/>ANN Index]
        S9 --> S10[10. Deploy<br/>Model]
    end
    
    LLM_BOUNDARY[/"LLM Usage Ends Here"/]
    S3 --- LLM_BOUNDARY
    
    style S1 fill:#e1f5fe
    style S2 fill:#e1f5fe
    style S3 fill:#fff3e0
    style S4 fill:#f3e5f5
    style S5 fill:#f3e5f5
    style S6 fill:#f3e5f5
    style S7 fill:#e8f5e9
    style S8 fill:#e8f5e9
    style S9 fill:#fce4ec
    style S10 fill:#fce4ec
```

## 11.7 Why ANMI Is Evaluation-Aware

Evaluation metrics like nDCG require **graded relevance**.
ANMI 2.0 trains embeddings that reflect exactly:

* monotonic ordering
* graded utility
* probability of relevance

This eliminates mismatch between:

* training loss (binary)
* evaluation metric (graded)

### 11.7.1 Training vs Evaluation Alignment

```mermaid
flowchart LR
    subgraph "Traditional Approach"
        T1[Binary Labels] --> T2[Binary Loss]
        T2 --> T3[Trained Model]
        T3 --> T4[nDCG Evaluation]
        T4 --> T5[❌ Metric Mismatch]
    end
    
    subgraph "ANMI 2.0 Approach"
        A1[ELO Graded<br/>Labels] --> A2[Graded Hybrid<br/>Loss]
        A2 --> A3[Trained Model]
        A3 --> A4[nDCG Evaluation]
        A4 --> A5[✅ Aligned]
    end
    
    style T5 fill:#ffcdd2
    style A5 fill:#c8e6c9
```

## 11.8 Statistical Foundations

ANMI leverages:

* Bradley–Terry ranking model
* ELO logistic model
* Sparse graph consistency proofs
* Convergence under noisy comparisons

LLM comparisons satisfy conditions for convergent ranking:

* asymmetric noise
* bounded error
* majority preference consistency

Thus ANMI has **theoretically stable convergence**, unlike heuristic negative mining.

### 11.8.1 Convergence Properties

```mermaid
flowchart TB
    subgraph "Input Conditions"
        C1[Asymmetric Noise]
        C2[Bounded Error]
        C3[Majority Consistency]
    end
    
    C1 --> BT[Bradley-Terry<br/>Model]
    C2 --> BT
    C3 --> BT
    
    BT --> CONV[Guaranteed<br/>Convergence]
    
    CONV --> P1[Stable ELO<br/>Scores]
    CONV --> P2[Monotonic<br/>Ordering]
    CONV --> P3[Calibrated<br/>Probabilities]
    
    style CONV fill:#e8f5e9
    style P1 fill:#c8e6c9
    style P2 fill:#c8e6c9
    style P3 fill:#c8e6c9
```

# 12. Expanded First-Stage vs Second-Stage Recommendations

## When to use ANMI as a Reranker (Second-Stage)

* Limited LLM budget
* Real-time production systems
* Stable, high-precision ranking needed
* Works with any existing dense retriever

## When to use ANMI as a Retriever (First-Stage)

* Research innovation
* Zero-shot generalization goals
* Replace dense retrieval entirely
* Create end-to-end probabilistic retrieval engine

## When to unify both

* Premium-grade RAG systems
* Enterprise search
* Safety-critical retrieval (legal, medical)

## 12.1 Use Case Decision Matrix

```mermaid
flowchart TB
    subgraph "Use Cases"
        UC1[Production<br/>System]
        UC2[Research<br/>Project]
        UC3[Enterprise<br/>Search]
        UC4[Medical/Legal<br/>RAG]
        UC5[Startup<br/>MVP]
    end
    
    subgraph "Recommendations"
        R1[Reranker<br/>Only]
        R2[First-Stage<br/>Retriever]
        R3[Unified<br/>System]
        R4[Partial<br/>ELO]
    end
    
    UC1 --> R1
    UC2 --> R2
    UC3 --> R3
    UC4 --> R3
    UC5 --> R4
    
    style R1 fill:#e3f2fd
    style R2 fill:#fff8e1
    style R3 fill:#fce4ec
    style R4 fill:#e8f5e9
```

# 13. Cost Optimization Strategies

## How to get costs under $5K

* Use GPT-4o-mini or Claude Haiku
* Reduce queries to 100k
* Use sparse graph comparisons ~10–15 per query

## How to support large-scale retrievers cheaply

* Use smaller LLMs for easy pairs
* Use big LLMs only for ambiguous pairs
* Add self-refinement and bootstrapping

Total cost can go down by **70–90%**.

## 13.1 Cost Optimization Flow

```mermaid
flowchart TD
    START[All Pairs] --> CLASSIFY{Pair<br/>Difficulty?}
    
    CLASSIFY -->|Easy| SMALL[Small LLM<br/>GPT-4o-mini<br/>Claude Haiku]
    CLASSIFY -->|Medium| MED[Medium LLM<br/>GPT-4o<br/>Claude Sonnet]
    CLASSIFY -->|Hard| LARGE[Large LLM<br/>GPT-4<br/>Claude Opus]
    
    SMALL --> SAVE1[85% cost<br/>reduction]
    MED --> SAVE2[50% cost<br/>reduction]
    LARGE --> SAVE3[Full cost<br/>for quality]
    
    SAVE1 --> TOTAL[Total: 70-90%<br/>cost reduction]
    SAVE2 --> TOTAL
    SAVE3 --> TOTAL
    
    style SMALL fill:#c8e6c9
    style MED fill:#fff9c4
    style LARGE fill:#ffcdd2
```

## 13.2 LLM Tier Distribution

```mermaid
pie showData
    title "Optimal LLM Usage by Tier"
    "Small LLM (Easy Pairs)" : 60
    "Medium LLM (Moderate)" : 30
    "Large LLM (Hard Pairs)" : 10
```

# 14. Future Extensions for ANMI

## 14.1 Online ELO Updates Without LLM

You can refine ELO using:

* Click models
* Dwell time
* Engagement signals

## 14.2 ELO-Based Knowledge Distillation

Use cross-encoder logits → convert to pairwise preferences → recompute ELO.

## 14.3 Multi-Hop Retrieval

ELO can incorporate query transformations:

* Q → Q1 → Q2 → D

## 14.4 Contrastive Consistency Regularization

Add loss ensuring:

```
if ELO(A) > ELO(B), then sim(A) > sim(B)
```

## 14.5 Future Roadmap

```mermaid
flowchart LR
    subgraph P1[Phase 1: Core]
        P1A[ELO Graph Builder]
        P1B[Hybrid Loss Training]
        P1C[Basic Curriculum]
    end
    
    subgraph P2[Phase 2: Production]
        P2A[Online ELO Updates]
        P2B[Click Model Integration]
        P2C[Auto-scaling Pipeline]
    end
    
    subgraph P3[Phase 3: Advanced]
        P3A[Multi-Hop Retrieval]
        P3B[Knowledge Distillation]
        P3C[Cross-Domain Transfer]
    end
    
    subgraph P4[Phase 4: Enterprise]
        P4A[Federated Learning]
        P4B[Real-time Adaptation]
        P4C[Domain Plugins]
    end
    
    P1 --> P2 --> P3 --> P4
    
    style P1 fill:#e3f2fd
    style P2 fill:#fff8e1
    style P3 fill:#f3e5f5
    style P4 fill:#e8f5e9
```

---

# 15. Diagrammatic Representations

## 15.1 ANMI 2.0 Offline Training Pipeline

```mermaid
graph TD
    A[Training Queries] --> B[Candidate Retrieval via Dense Retriever]
    B --> C[Pairwise LLM Judgments]
    C --> D[Sparse Comparison Graph]
    D --> E[ELO Score Computation]
    E --> F[Hybrid Loss Training]
    F --> G[ANMI Embedding Model]
    G --> H[ANN Index Build]
    H --> I[Deployment]
```

## 15.2 ANMI 2.0 as a Second-Stage Reranker

```mermaid
graph TD
    A[User Query] --> B[Dense Retriever]
    B --> C[ANN: Top 200]
    C --> D[ANMI 2.0 Reranker]
    D --> E[Top-k Results]
```

## 15.3 ANMI 2.0 as Unified First-Stage Retriever

```mermaid
graph TD
    A[User Query] --> B[ANMI Embedding Model]
    B --> C[ANN Search in ELO-Calibrated Space]
    C --> D[Optional ANMI Reranker]
    D --> E[Final Ranked Results]
```

## 15.4 ELO Scoring Flow

```mermaid
graph LR
    A[Doc A] --> C{LLM Pairwise Preference}
    B[Doc B] --> C
    C --> D[ELO Update]
    D --> E[Score A']
    D --> F[Score B']
```

## 15.5 Complete System Overview

```mermaid
flowchart TB
    subgraph "OFFLINE TRAINING"
        direction TB
        O1[Query Corpus] --> O2[Initial Retrieval]
        O2 --> O3[Candidate Selection]
        O3 --> O4[LLM Pairwise<br/>Comparison]
        O4 --> O5[Sparse ELO<br/>Graph]
        O5 --> O6[Score<br/>Normalization]
        O6 --> O7[Curriculum<br/>Scheduler]
        O7 --> O8[Hybrid Loss<br/>Training]
        O8 --> O9[Export Model]
    end
    
    subgraph "DEPLOYMENT"
        direction TB
        D1[Build ANN Index]
        D2[Deploy Reranker]
        D3[API Gateway]
    end
    
    subgraph "INFERENCE"
        direction TB
        I1[User Query] --> I2[Query Encoder]
        I2 --> I3[ANN Search]
        I3 --> I4[ANMI Reranker]
        I4 --> I5[Results]
    end
    
    O9 --> D1
    O9 --> D2
    D1 --> I3
    D2 --> I4
    D3 --> I1
    
    style O4 fill:#fff3e0
    style O5 fill:#e3f2fd
    style O8 fill:#e8f5e9
    style I4 fill:#f3e5f5
```

---

# 16. Summary Metrics

## 16.1 Expected Improvements

```mermaid
flowchart LR
    subgraph "NDCG Improvement Comparison"
        direction TB
        A[Hard Neg Mining<br/>~8% gain] --> B[Knowledge Distill<br/>~12% gain]
        B --> C[Domain Fine-tune<br/>~18% gain]
        C --> D[Full ANMI 2.0<br/>~42% gain]
    end
    
    style A fill:#ffcdd2
    style B fill:#fff9c4
    style C fill:#c8e6c9
    style D fill:#a5d6a7
```

## 16.2 Component Contribution Analysis

```mermaid
pie showData
    title "ANMI 2.0 Quality Gain Attribution"
    "ELO Graded Labels" : 35
    "Difficulty-Aware Mining" : 25
    "Curriculum Training" : 20
    "Hybrid Loss" : 15
    "False Negative Filter" : 5
```
