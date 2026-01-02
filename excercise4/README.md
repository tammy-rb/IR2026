# Temporal Failure Analysis — Baseline RAG (Time-Blind)

## Overview

In this experiment, we evaluated a baseline Retrieval-Augmented Generation (RAG) system **without any temporal awareness**.  
The goal was to demonstrate how a standard RAG pipeline fails when questions depend on time, even though the correct information exists in the corpus.

We **did not modify**:
- the retriever,
- the ranking mechanism,
- or the LLM prompting logic.

The system relies purely on semantic similarity.

---

## Temporal Query Set

We created a dedicated temporal evaluation file:  
[text](queries/temporal_queries.json)

This file contains factual questions that:
- explicitly mention a year or time period,
- are intentionally **not overly specific**,
- require the system to rely on the **correct temporal documents** rather than general semantic similarity.

The queries focus on:
- roles and affiliations in a specific year (e.g., 2025),
- historical events limited to a specific month/year (e.g., December 2023).

---

## Execution and Outputs

The temporal query set was executed using the existing RAG pipeline across:
- 4 pipelines (fixed/semantic × BM25/dense),
- K ∈ {3, 5, 10}.

The results (answers + retrieved references) were saved to:  
[text](outputs/stage1_baseline_runs/rag_temporal_queries_4pipelines_k3-5-10_20251223_225311.json)

This file is used below to extract concrete temporal failures.

---

## Failure Analysis (Concrete Temporal Errors)

### ❌ Failure 1 — Wrong-Year Event Used as Answer (Battle of the Bulge)

**Query**  
> *What event was being commemorated in December 2023 according to the congressional records?*

**Pipeline / K**  
- **Pipeline:** `semantic + dense`  
- **K:** `3`

**Model Answer**  
> “The event being commemorated in December 2023 was the 80th Anniversary of the Battle of the Bulge, hosted by the American Battle Monuments Commission (ABMC).”

**Retrieved Source (Cited by the Model)**  
- `2024-12-19.txt`

**Expected Temporal Source**  
- `2023-12-22.txt`

**Why This Is a Failure**  
The question explicitly asks about **December 2023**, but the model:
- retrieves a document from **December 2024**,
- presents the event as if it occurred in 2023.

This is a clear temporal error:  
a semantically relevant but **chronologically incorrect** document overrides the correct time constraint.

---

### ❌ Failure 2 — Another Wrong-Year Event Used as Answer (Bayou Classic)

**Query**  
> *What event was being commemorated in December 2023 according to the congressional records?*

**Pipeline / K**  
- **Pipeline:** `semantic + bm25`  
- **K:** `3`

**Model Answer**  
> “The event being commemorated in December 2023 was the 50th Anniversary of the Bayou Classic, which celebrates the excellence, pride, and unity of Historically Black Colleges and Universities.”

**Retrieved Source (Cited by the Model)**  
- `2024-06-28.txt`

**Expected Temporal Source**  
- `2023-12-22.txt`

**Why This Is a Failure**  
Although the query is restricted to **December 2023**, the model:
- retrieves an event described in a **June 2024** document,
- assigns it to December 2023 without any temporal verification.

This demonstrates a second, independent temporal failure:  
a different **future-year event** is incorrectly mapped onto the requested time period due to semantic similarity alone.

---

### ❌ Failure 3 — Wrong-Year Committee Schedule Used as Answer

**Query**  
> *Which Senate committees were scheduled to meet in December 2023?*

**Pipeline / K**  
- **Pipeline:** `fixed + dense`  
- **K:** `3`

**Model Answer**  
> “The Senate committees scheduled to meet included the Committee on Armed Services and the Committee on Homeland Security.”

**Retrieved Source (Cited by the Model)**  
- `2025-10-22.txt`

**Expected Temporal Sources**  
- `2023-12-18.txt`  
- `2023-12-22.txt`

**Why This Is a Failure**  
The question explicitly targets **December 2023**, yet the model:
- retrieves and cites a document from **October 2025**,
- lists committees scheduled according to the **2025 congressional calendar**,
- presents the information as if it applies to 2023.

This is a classic temporal failure in which:
- structural similarity (committee schedules)
- overrides the explicit time constraint imposed by the query.

---

## Summary

Across all three failures, the baseline RAG system demonstrates that:
- semantic similarity consistently dominates over temporal correctness,
- temporal errors occur across different retrieval pipelines and K values,
- and the system lacks any explicit mechanism for enforcing or prioritizing time constraints during retrieval or answer generation.

As a result, documents from incorrect years may override time-relevant evidence, leading to confident but temporally incorrect answers.  
These failures motivate the need for temporal filtering, re-ranking, or explicit time-aware reasoning mechanisms in later stages.

---

## Table of Contents
- [Stage 1 — Baseline (Time-Blind) Failure Analysis](#temporal-failure-analysis--baseline-rag-time-blind)
- [Stage 2 — Temporal Indexing (Data Engineering)](#stage-2--temporal-indexing-data-engineering)
- [Stage 3 — Temporal Query Resolution (Duckling + LLM)](#stage-3--temporal-query-resolution-duckling--llm)

---

## Stage 2 — Temporal Indexing (Data Engineering)

### Goal

Upgrade the static RAG artifacts from “just text + vectors” into **structured, time-aware chunk records**.

Concretely, instead of indexing only:
- $vector
- $text

we index:
- $vector
- $text
- $timestamp (temporal metadata)

This stage is **offline** (done once). The goal is to pay the cost of time normalization during indexing, so later retrieval does not need to repeatedly parse/convert dates.

---

### Temporal Normalization (Two Formats)

Each chunk stores the document date in **two normalized representations**:
- `doc_date_iso`: ISO 8601 date string (`YYYY-MM-DD`) — convenient for logging/visualization/debug.
- `doc_timestamp`: Unix timestamp in **UTC seconds** — convenient for numeric filtering and comparisons during retrieval.

Date extraction is based on the debate filename/path containing a date in the form `YYYY-MM-DD`.

---

### Chunk Schema (What Is Stored Per Chunk)

The uniform chunk record is defined in [models/chunk.py](models/chunk.py).


| Field | Meaning |
|------|---------|
| `doc_id` | Stable document identifier (filename without extension). |
| `source_path` | Full path to the raw debate `.txt` file (used as provenance pointer). |
| `corpus` | Corpus label derived from the folder name (e.g., `us` / `british`). |
| `chunking_method` | Chunking strategy: `fixed` or `semantic`. |
| `chunk_index` | Chunk ordinal within the document (0-based). |
| `start_char`, `end_char` | Character offsets into the original document text (slice boundaries). |
| `text` | The chunk content. |
| `num_words` | Word count (used to enforce the max chunk size constraint). |
| `doc_date_iso` | Normalized document date (ISO 8601). |
| `doc_timestamp` | Normalized document date (Unix timestamp, UTC). |

This schema is reused by all downstream indexing methods so both sparse and dense retrievers can share the same metadata.

**Implementation note (OOP consistency):** throughout Exercise 4, all chunkers, scripts, embedders, and the retriever read/validate/serialize chunks via the shared `Chunk` class in [models/chunk.py](models/chunk.py), rather than passing around ad-hoc dictionaries. This keeps the chunk schema consistent end-to-end.

---

### Chunking (Build Chunk Corpora)

Chunking is implemented under the [chunckers/](chunckers/) folder:
- `FixedChunker` (fixed-size, sentence overlap)
- `SemanticChunker` (semantic segmentation based on sentence embeddings)

Run chunking:

```bash
python s_01_chuncking.py
```

Outputs:
- `outputs/chunks/chunks_fixed.jsonl`
- `outputs/chunks/chunks_semantic.jsonl`

Each line in these JSONL files is a `Chunk` dict including `doc_date_iso` and `doc_timestamp`.

Optional repair utility (if you already have chunk files and want to ensure timestamps are present):

```bash
python s_01_fix_timestamps.py
```

---

### Time Distribution Artifact (Histogram by Year)

To validate that temporal metadata was extracted and stored correctly, we generate a **time distribution report** (chunks per year).

Run:

```bash
python s_01_time_histogram.py
```

Outputs:
- `outputs/reports/time_histograms/fixed_chunks_by_year.png`
- `outputs/reports/time_histograms/fixed_chunks_by_year.json`
- `outputs/reports/time_histograms/semantic_chunks_by_year.png`
- `outputs/reports/time_histograms/semantic_chunks_by_year.json`

Example JSON structure:

```json
{
  "name": "fixed",
  "total_chunks_seen": 12096,
  "total_with_year": 12096,
  "counts_by_year": {
    "2023": 2313,
    "2024": 5276,
    "2025": 4507
  }
}
```

![alt text](outputs/reports/time_histograms/fixed_chunks_by_year.png)

```json
{
  "name": "semantic",
  "total_chunks_seen": 87860,
  "total_with_year": 87860,
  "counts_by_year": {
    "2023": 17204,
    "2024": 38451,
    "2025": 32205
  }
}
```

![alt text](outputs/reports/time_histograms/semantic_chunks_by_year.png)

---

### Embedding + Temporal Index Rebuild

This stage rebuilds the vector indexes **while preserving the chunk metadata** (including time).

Implemented embedders are under [embedders/](embedders/):
- `BM25Embedder` (sparse retrieval)
- `OpenAIEmbedder` (dense retrieval via OpenAI embeddings + FAISS)

Run:

```bash
python s_02_embedding.py
```

Outputs:
- `outputs/embeddings/bm25/fixed/`
- `outputs/embeddings/bm25/semantic/`
- `outputs/embeddings/openai/fixed_faiss/`
- `outputs/embeddings/openai/semantic_faiss/`

**What is stored for each chunk/vector?**
- In BM25: the sparse matrix + vocabulary, and a `meta.json` that contains (among other fields) `doc_date_iso` and `doc_timestamp` for each chunk. The linkage is positional: row `i` in `bm25_okapi.npz` corresponds to `meta.json[i]`.
- In OpenAI+FAISS: vectors are stored in FAISS, and the corresponding texts + metadata are stored in the FAISS docstore (saved by LangChain in `index.pkl`).

At the end of Stage 2, both sparse and dense indexes can support time-aware retrieval in later stages (filtering / re-ranking / temporal reasoning).


## Stage 3 — Temporal Query Resolution (Duckling + LLM)

### Time-Aware Retrieval Strategies

### Overview

This stage implements and compares two **time-aware retrieval strategies** that balance
**semantic relevance** (topic similarity) with **temporal relevance** (time correctness).

The key design principle is that **time constraints are interpreted differently depending on how explicitly they appear in the query**.  
Accordingly, we apply different retrieval strategies.

---

### Strategy Selection Policy

We define **three query categories** based on the output of the `RAG_retriever` module:

#### 1. Explicit Time Range → Hard Filtering

**Examples**
- “What was the official position **in 2024**?”
- “Was the policy supportive **in the last quarter of 2023**?”
- “What was the budget **between 2018 and 2020**?”

**Detection**
- `granularity ∈ {year, quarter, month, range}`
- Explicit `start` and `end` dates extracted

**Strategy**
> **Hard Filtering**

Documents (or chunks) whose timestamps fall **outside the specified range are removed before retrieval**.

**Rationale**
- Documents outside the range are *inherently irrelevant*
- Soft weighting may incorrectly surface semantically similar but temporally incorrect content
- This strategy prevents classic *temporal hallucinations*

---

#### 2. Current / Recent Time → Soft Decay (Recency Weighting)

**Examples**
- “What is the **current** official position?”
- “What is the policy **today**?”

**Detection**
- `mode ∈ {current, recent}`
- Time window inferred (e.g., last 90 days) when applicable

**Strategy**
> **Soft Decay (Recency Weighting)**

No documents are filtered out.  
Instead, newer documents receive a higher score.

**Scoring Function**
\[
Score = (1 - \alpha)\cdot Sim + \alpha \cdot \frac{1}{1 + \Delta t \cdot \lambda}
\]

Where:
- `Sim` = semantic similarity score
- `Δt` = distance between document time and reference time
- `t_ref = now`
- `α` controls semantic vs temporal importance
- `λ` controls decay speed

**Rationale**
- “Current” does not imply a strict boundary
- Older documents may still provide valid background or context
- Soft decay prioritizes recency without discarding valuable information

---

#### 3. No Time Mentioned → Soft Decay (Mild)

**Examples**
- “What is the official position regarding Gaza?”
- “What is the policy on security?”

**Detection**
- `mode = none`
- No time expressions detected

**Strategy**
> **Soft Decay with mild parameters**

- `t_ref = now`
- Small `α` (e.g., 0.1–0.2)
- Small `λ`

**Rationale**
- The query is not explicitly temporal
- However, if two documents are equally relevant semantically, **newer information is generally preferable**
- This reflects realistic user expectations while avoiding over-penalization of older documents

---

### Summary of Strategy Mapping

| Query Type                      | Granularity            | Strategy |
|---------------------------------|------------------------|----------|
| Explicit year / quarter / range | year / quarter / month | Hard Filtering |
| “Current”, “today”, “now”       | relative               | Soft Decay |
| “Recent”, “recently”, “past decade/years” | relative / none          | Soft Decay |
| No time mentioned               | none                   | Soft Decay (mild) |
| Comparative                     | compare                | Separate retrieval per range |

---

### Time Detection: Duckling vs LLM

### Primary Tool: Duckling

We use **Duckling** as the primary component for temporal expression extraction.

Duckling provides a **deterministic, rule-based** solution that reliably identifies explicit
temporal references such as:
- Years (e.g., “2024”)
- Quarters (e.g., “last quarter of 2023”)
- Date ranges (e.g., “between 2018 and 2020”)
- Relative expressions (e.g., “today”, “now”, “current”)

For the scope of this assignment, Duckling is sufficient to:
- Classify queries into `explicit / current / none`
- Generate concrete ISO date ranges required for hard filtering or temporal weighting

Its predictable behavior makes it well-suited for controlled experiments and fair comparison
between retrieval strategies.

### Duckling: Temporal Extraction Overview

Duckling is a rule-based library for extracting structured temporal information from natural
language text. It typically runs as an external service and returns normalized time expressions
in a machine-readable format.

#### Duckling Output Format

When parsing a query, Duckling returns a list of detected entities. For temporal expressions,
each entity follows a structure like:

```json
{
  "dim": "time",
  "body": "tomorrow at 8pm",
  "start": 0,
  "end": 15,
  "latent": false,
  "value": {
    "type": "value",
    "value": "2020-09-28T20:00:00.000-07:00",
    "grain": "hour"
  }
}
```

**Key fields**
- `dim`: detected entity type (for time expressions: `time`)
- `body`: exact text span matched by Duckling
- `start`, `end`: character offsets in the original query
- `value.type`:
  - `value` → single point in time
  - `interval` → bounded time range
- `value.value` / `value.from` / `value.to`: ISO-8601 timestamps
- `value.grain`: temporal resolution (e.g., `year`, `month`, `day`, `hour`)

#### Deriving Temporal Granularity

Duckling does not explicitly return a single “granularity” field. Instead, granularity is derived
in our pipeline based on Duckling’s structured output:
- Use `value.type` to decide point vs. interval
- Use `value.grain` to determine the resolution

Examples:
- `grain = year` → granularity = `year`
- `type = interval` → granularity = `range`
- no detected time entity → granularity = `none`

This separation keeps Duckling as a pure extraction component, while temporal semantics and
retrieval-strategy decisions are handled by the retrieval pipeline.

#### Duckling in a RAG-Based Architecture

Within our RAG pipeline, Duckling is used only for temporal signal extraction. Higher-level
decisions—such as selecting between **Hard Filtering** and **Soft Decay**—are made afterward,
based on the derived granularity.

Duckling’s deterministic behavior makes it well-suited for reproducible experiments and
controlled evaluation of time-aware retrieval strategies.

**Further reading**
- *Using Duckling to Extract Dates and Times in Your Rasa Chatbot*  
  http://medium.com/@adboio/using-duckling-to-extract-dates-and-times-in-your-rasa-chatbot-7687f4fde2e0

---

#### Temporal Signal Extraction (Duckling)
`RAG_retriever/duckling_time_analysis.py`

##### Goal

Deterministically extract and normalize **temporal signals** from user queries in order to support
**time-aware retrieval decisions** in later stages of the pipeline.

This module is responsible **only** for temporal understanding.  
It does **not** decide whether to apply hard filtering or soft decay — it provides structured signals for downstream logic.

---

#### What the Module Does

1. Sends the query to a local **Duckling** server (`POST /parse`) to detect time expressions.
2. Normalizes Duckling’s output into a unified internal representation (`TimeRange`).
3. Derives a **high-level temporal intent mode** for the query.

---

#### Normalized Time Representation (`TimeRange`)

Each detected temporal expression is converted into a `TimeRange` object with the following properties:

- `start` / `end` as ISO-8601 dates (`YYYY-MM-DD`)
- `duckling_type`: `value` or `interval`
- `duckling_grain`: Duckling’s native resolution (`year`, `month`, `day`, etc.)
- `granularity`: derived system-level granularity
- inclusive end-date handling for Duckling intervals
- full support for **open-ended intervals in both directions**

##### Supported Interval Types

| Type | Representation |
|-----|---------------|
| Point | `start == end` |
| Bounded interval | `start != end`, both defined |
| Open-end interval | `start != null`, `end == null` (e.g. *since 2019*) |
| Open-start interval | `start == null`, `end != null` (e.g. *until Jan 1, 2020*) |

**Note:**  
Sub-day resolutions (`hour`, `minute`, `second`) are intentionally normalized to **day-level granularity** to avoid false temporal precision in document retrieval.

---

#### Temporal Modes

Each query is classified into **one primary temporal mode**:

| Mode | Meaning |
|-----|--------|
| `explicit` | A concrete time constraint was detected (point or interval) |
| `current` | Query explicitly targets the present (`now`, `today`, `current`)|
| `recent` | Fuzzy recency intent without explicit temporal bounds |
| `none` | No temporal intent detected |

##### Priority Rules

`explicit` (intervals) > `current` (when only point values exist) > `recent` > `none`

**Rationale**
- Any detected **interval** (bounded or open) is treated as the strongest signal and forces `explicit` mode.
- Present-time intent (`now`, `today`) may override historical *point* mentions but **never** override intervals.
- Vague recency expressions are handled separately as `recent`.

---

#### Returned Output Schema

```json
{
  "query": "<original query>",
  "now_iso": "YYYY-MM-DD",
  "mode": "explicit | current | recent | none",
  "ranges": [
    {
      "id": "t1",
      "source": "duckling",
      "text": "<matched span>",
      "start": "YYYY-MM-DD | null",
      "end": "YYYY-MM-DD | null",
      "open_ended": true | false,
      "duckling_type": "value | interval",
      "duckling_grain": "year | month | week | day | ...",
      "granularity": "year | quarter | month | week | day | range | unknown",
      "kind": "point | bounded_range | open_range"
    }
  ],
  "duckling_raw": [ "...minimal raw Duckling payload (debug)..." ]
}
```

#### Observed Test Cases (CLI Output)

> Reference date during this run: `2025-12-30` (affects `now/today` resolution).

| Query | Mode | Ranges (summary) | Explanation |
|---|---|---|---|
| `what happened in april in 2022?` | `explicit` | `2022-04-01` (month point) | Explicit calendar reference |
| `give me the reports from march 15, 2023 until now` | `explicit` | `2023-03-15 → 2025-12-29` | Interval dominates even with “now” |
| `show me reports until january 1, 2020` | `explicit` | `null → 2019-12-31` (open-start) | Upper-bounded interval |
| `who is the prime minister now?` | `current` | `2025-12-30` (day point) | Explicit present-time intent |
| `what is the biggest news today?` | `current` | `2025-12-30` (day point) | Explicit present-time intent |
| `how has the economy changed recently?` | `recent` | `none` | Vague recency |
| `over the past decade, what changed?` | `recent` | `none` | Fuzzy long-term recency |
| `since 2019, what has changed?` | `explicit` | `2019-01-01 → null` (open-end) | Open-ended lower bound |
| `who is the hero in superball?` | `none` | `none` | No temporal signal |

#### Note on Vague Temporal Expressions

Queries containing vague or implicit temporal expressions such as:
- `recently`
- `in recent years`
- `over the past decade`

are classified as `recent` **only** when they match explicit recency patterns. Otherwise, the query remains `none`.

`recent` mode is intended for **Soft Decay (recency weighting)**:
- no hard filtering
- newer documents are preferred
- older but semantically relevant documents remain eligible

#### Design Rationale

This design:
- cleanly separates temporal signal extraction from retrieval policy,
- prevents temporal leakage and false precision,
- correctly handles mixed-intent queries,
- supports fair and controlled comparison between hard temporal filtering vs. soft recency-based ranking.


### Stage 3.1 — Recency Prior & Soft Decay Reranking
Theoretical Grounding
This stage implements the findings of Solving Freshness in RAG: A Simple Recency Prior and the Limits of Heuristic Trend Detection. The paper demonstrates that standard RAG systems are "temporally blind" because vector embeddings capture semantic similarity but ignore temporal dynamics.

To solve this, we implement a lightweight temporal memory layer that fuses content similarity with a half-life recency prior.

### Soft Decay Scoring Formula

We use a fused score to re-rank candidates:

`Sim(q,d,t) = α * Sim(q,d) + (1-α) * 0.5^(Δt_days/h)`

- `Sim(q, d)`: cosine similarity (dense) or log-normalized BM25 score (sparse).  
- `Δt_days`: document age in days vs. the query reference date, clamped to zero for future documents.  
- `α`: semantic weight.  
- `h`: temporal half-life in days.

### Parameter Policy (build_retrieval_plan)

The parameters are chosen based on the sensitivity analysis reported in the paper, combined with empirical testing on the local parliamentary and congressional corpus (results stored under outputs\rag_runs)

| Query Mode | α (Alpha) | h (Half-life / days) | 
|------------|-----------|---------------|
| Current    | 0.6       | 180            | 
| Recent     | 0.65       | 865           | 
| None       | 0.7       | 730          | 


**Rationale.**  
The parameterization is guided directly by the empirical findings reported in Grofsky (2025).

### Choice of α (Semantic Weight)

The paper reports a sensitivity analysis showing that the recency prior is effective across a broad and stable range of α values, approximately **0.4–0.7**. Within this range, the temporal signal consistently corrects freshness failures, while values above this range (α ≥ 0.9) cause the model to revert to semantic-only behavior and reintroduce temporal blindness.

Based on this finding, we deliberately select α values toward the **upper boundary of the stable range**. This reflects a design choice appropriate for parliamentary and congressional debates, where semantic relevance remains the primary signal and temporal information serves as a corrective rather than a dominant factor.

Empirical testing on the local corpus further supports this choice. We observed that when α is set near the upper bound of the stable range, the system still prioritizes the most recent relevant chunks when appropriate, while avoiding excessive dominance of recency that would suppress substantively important but slightly older debates. These results indicate that the chosen α values are sufficient to correct temporal blindness without destabilizing semantic ranking.

As temporal intent weakens (from *current* to *none*), α is increased accordingly, ensuring that recency influences ranking while preserving substantive content as the main retrieval signal.

---

### Choice of Temporal Half-life (h)

The temporal half-life `h` defines the time scale over which documents lose relevance. While the paper uses a short half-life (14 days) appropriate for fast-moving cybersecurity logs, it explicitly notes that this parameter should be extended for slower-moving domains.

Parliamentary and congressional debates exhibit long-lived relevance, with legislative discussions remaining informative across months or even years. We therefore adopt substantially longer half-lives:
- **Current** queries emphasize recent legislative activity using a half-life of approximately six months.
- **Recent** queries align with annual legislative cycles.
- **None** applies a very mild recency bias, where time functions primarily as a tie-breaker while preserving historical context.

This policy corrects the temporal blindness identified in the paper while respecting the slower temporal dynamics of institutional political discourse.

### Implementation Notes

- Temporal logic is implemented in `temporal_policy.py` and applied via `RAGRetriever.get_topk_timeaware`.
- Candidate sets are **oversampled** prior to temporal re-ranking (`K × 10`, minimum 50), following the post-processing design recommended in the paper.
- BM25 scores are log-normalized to align their scale with the `[0,1]` temporal component before fusion.
- Multiple temporal constraints extracted by Duckling are combined using **intersection semantics**, ensuring strict adherence to explicit time bounds.
- Open-ended expressions (e.g., “since 2022”, “until 2020”) are represented as semi-infinite intervals in the retrieval plan.
- Parameter sweeps and evaluation artifacts are stored under `outputs/rag_runs/` for reproducibility and analysis.

---

### stage 3 deliverable:
























## Comprehensive Temporal Evaluation

### Expanded Query Set

To thoroughly evaluate the temporal RAG system, we expanded the initial query set from 5 to **20 queries**, covering both US Congressional and British Parliamentary debates across four temporal query categories:

**Query Distribution:**
- **Point-in-time queries** (8 queries): Explicit year references requiring hard temporal filtering
  - Budget/defence allocation in 2024 (US + British)
  - Healthcare legislation discussions in 2024 (Congress + Parliament)
  
- **Recency queries** (8 queries): Current position/latest discussion queries requiring soft decay
  - Official positions on Israel (US Congress + British Parliament)
  - Official positions on Hamas/Gaza (US Congress + British Parliament)
  - Latest discussions on immigration reform/policy (Congress + Parliament)
  - Current stance on climate change legislation (Congress + Parliament)

- **Explicit range queries** (6 queries): Specific time period constraints (Q4 2023, Q3 2024)
  - **Note:** Q4 = Quarter 4 (October-December), Q3 = Quarter 3 (July-September)
  - Official positions in Q4 2023 on Israel (US + British)
  - Official positions in Q4 2023 on Hamas/Gaza (US + British)
  - Energy policy debates in Q3 2024 (Congress + Parliament)

- **Comparison/Evolution queries** (4 queries): Cross-temporal analysis
  - Position changes from Q4 2023 to Q4 2025 (US + British)
  - Economic policy evolution 2023-2025 (Congress + Parliament)

All queries include corpus-specific language markers ("US Congress", "British Parliament", "Congress", "Parliament") to enable corpus-targeted retrieval.

**Query file:** `queries/given_temporal_queries.json`

---

### Experimental Configuration

**Evaluation script:** `s_03_temporal_analysis.py`

**Parameters:**
- **K values:** 3, 5, 10 (retrieval depth)
- **TopN:** 5 (comparison window for delta analysis)
- **Pipelines:** All 4 configurations
  - fixed/bm25
  - fixed/dense
  - semantic/bm25
  - semantic/dense

**Execution:**
```bash
python s_03_temporal_analysis.py --queries_json queries/given_temporal_queries.json --k 3 5 10
```

**Total comparisons:** 240 rows (20 queries × 4 pipelines × 3 k values)

**Output artifacts:**
- Raw comparison data: `outputs/rag_runs/stage3_temporal_analysis/stage3_given_temporal_queries_20260102_094323.json`
- Aggregated summary: `outputs/reports/stage3_summaries/stage3_given_temporal_queries_20260102_094323__summary.json`

---

### Aggregated Results Overview

The comprehensive evaluation produced 240 baseline vs time-aware retrieval comparisons. Summary statistics were generated using:

```bash
python s_04_stage3_summary.py --in_json outputs/rag_runs/stage3_temporal_analysis/stage3_given_temporal_queries_20260102_094323.json
```

#### Overall Statistics by Pipeline and K

Each pipeline/k combination processed 20 queries across 4 temporal buckets. Key aggregate metrics:

| Pipeline | K | Total Entered | Total Left | Avg Jaccard | Avg Churn | Interpretation |
|----------|---|---------------|------------|-------------|-----------|----------------|
| fixed/bm25 | 3 | 53 | 53 | 0.27 | 0.53 | Moderate retrieval change |
| fixed/bm25 | 5 | 56 | 56 | 0.25 | 0.56 | Higher churn at k=5 |
| fixed/bm25 | 10 | 56 | 56 | 0.25 | 0.56 | Stable across k=5,10 |
| fixed/dense | 3 | 61 | 61 | 0.19 | 0.61 | High sensitivity to temporal signal |
| fixed/dense | 5 | 65 | 65 | 0.17 | 0.65 | Highest churn across all configs |
| fixed/dense | 10 | 67 | 67 | 0.16 | 0.67 | Dense shows strongest temporal bias |
| semantic/bm25 | 3 | 54 | 54 | 0.24 | 0.54 | Similar to fixed/bm25 |
| semantic/bm25 | 5 | 55 | 55 | 0.24 | 0.55 | Chunking has minimal impact on BM25 |
| semantic/bm25 | 10 | 55 | 55 | 0.24 | 0.55 | Very stable across k values |
| semantic/dense | 3 | 60 | 60 | 0.19 | 0.60 | Dense embeddings dominate |
| semantic/dense | 5 | 63 | 63 | 0.17 | 0.63 | Semantic chunking slightly reduces churn |
| semantic/dense | 10 | 63 | 63 | 0.17 | 0.63 | Stable at k=5,10 |

**Key Observations:**
- **Dense representations** consistently show higher churn (0.60-0.67) compared to BM25 (0.53-0.56)
- **Lower Jaccard similarity** in dense pipelines (0.16-0.19) vs BM25 (0.24-0.27) indicates greater temporal correction needed
- **Chunking method** has minimal impact on temporal sensitivity (fixed vs semantic show similar patterns within same representation)
- **K value impact** is minimal beyond k=5, suggesting temporal re-ranking converges quickly

#### Year Distribution Shifts

Across all 240 comparisons, temporal awareness produced systematic year distribution changes:

**Baseline Retrieval (Time-Blind):**
- Mixed year distribution reflecting semantic relevance
- BM25 baselines: Relatively balanced across 2023-2025
- Dense baselines: Strong bias toward 2024 (most content-rich year)

**Time-Aware Retrieval:**
- **Point-in-time queries:** 100% compliance with target year (2024)
- **Recency queries:** Strong shift to 2025 (most recent year)
- **Explicit range queries:** 100% compliance with specified periods (Q4 2023, Q3 2024)
- **Comparison queries:** Appropriate distribution across compared periods

**Example (fixed/dense, k=5):**
- Baseline: {2023: 19, 2024: 57, 2025: 24}
- Time-aware: {2023: 32, 2024: 25, 2025: 43}
- Shift indicates strong temporal re-ranking toward query-appropriate years

---

### Query Group Performance Breakdown

#### Point-in-Time Queries (8 queries, explicit year constraint)

**Temporal mode:** `explicit`  
**Strategy:** Hard filtering (only target year eligible)

**Aggregate statistics (across all pipelines/k):**
- **Avg entered:** 1.94 - 2.83 chunks per query
- **Avg left:** 1.94 - 2.83 chunks per query
- **Avg Jaccard:** 0.25 - 0.44 (moderate to high overlap)
- **Avg churn:** 0.39 - 0.57 (moderate change)

**Year compliance:**
- All time-aware retrievals achieved **100% target year compliance** (2024 chunks only)
- Baseline retrievals showed mixed years (2023/2024/2025)

**Pipeline differences:**
- **BM25 pipelines** showed better baseline alignment (higher Jaccard 0.35-0.44)
- **Dense pipelines** required more correction (lower Jaccard 0.25-0.33), especially with fixed chunking

---

#### Recency Queries (8 queries, current position/latest discussion)

**Temporal mode:** `current`  
**Strategy:** Soft decay (α=0.6, h=180 days)

**Aggregate statistics:**
- **Avg entered:** 3.75 - 4.38 chunks per query
- **Avg left:** 3.75 - 4.38 chunks per query
- **Avg Jaccard:** 0.06 - 0.13 (very low overlap)
- **Avg churn:** 0.75 - 0.88 (very high change)

**Year distribution shift:**
- **Baseline:** Mixed 2023-2025 (semantic relevance dominated)
- **Time-aware:** **90-100% shift to 2025** (most recent documents)

**Key finding:**
- Recency queries exhibited the **most dramatic temporal correction**
- Near-complete ranking replacement confirms baseline RAG temporal blindness for "current" intent
- All pipelines benefited equally from soft decay strategy

---

#### Explicit Range Queries (6 queries, specific time windows)

**Temporal mode:** `explicit`  
**Strategy:** Hard filtering (Q4 2023, Q3 2024 constraints)

**Aggregate statistics:**
- **Avg entered:** 2.50 - 3.83 chunks per query
- **Avg left:** 2.50 - 3.83 chunks per query
- **Avg Jaccard:** 0.08 - 0.38 (wide variation)
- **Avg churn:** 0.50 - 0.83 (high change)

**Year compliance:**
- **100% compliance** with specified time ranges
- Q4 2023 queries: Only 2023 chunks retrieved
- Q3 2024 queries: Only mid-2024 chunks retrieved

**Pipeline differences:**
- **Dense/fixed** showed **zero overlap** for some queries (complete replacement)
- **BM25** maintained partial overlap, suggesting some lexical temporal alignment

---

#### Comparison/Evolution Queries (4 queries, cross-temporal analysis)

**Temporal mode:** `comparison` / `evolution`  
**Strategy:** Multi-range retrieval with balanced sampling

**Aggregate statistics:**
- **Avg entered:** 3.50 - 4.25 chunks per query
- **Avg left:** 3.50 - 4.25 chunks per query
- **Avg Jaccard:** 0.08 - 0.17 (low overlap)
- **Avg churn:** 0.70 - 0.85 (high change)

**Year distribution:**
- Time-aware retrieval successfully retrieved documents from **both comparison periods**
- Example (Q4 2023 vs Q4 2025): Balanced distribution across 2023 and 2025
- Baseline showed arbitrary temporal distribution

---

### Technical Implementation Details

**LLM Selection:**
- Model: **GPT-4o-mini** (OpenAI)
- Rationale: Cost-effective, sufficient reasoning capability for temporal query understanding, fast response times suitable for batch evaluation

**Embedding Model:**
- Model: **text-embedding-3-large** (OpenAI)
- Dimensionality: 3072 (full)
- Rationale: State-of-the-art semantic representation, proven performance on retrieval benchmarks, consistent with dense pipeline requirements

**K Value Selection:**
- **K=3:** Minimal retrieval for high-precision scenarios
- **K=5:** Standard topN for RAG applications (matches answer generation window)
- **K=10:** Extended context for comparison queries and redundancy

**Rationale:** Incremental k values (3→5→10) enable analysis of how temporal re-ranking stability changes with retrieval depth. Results show diminishing returns beyond k=5 for temporal correction.

**Chunking Methods:**
- **Fixed:** 400-word chunks with sentence-boundary preservation, 50-word overlap
- **Semantic:** LLM-based semantic segmentation preserving topical coherence

**Representation Methods:**
- **BM25:** Okapi BM25 with standard parameters (k1=1.5, b=0.75), log-normalization for score fusion
- **Dense:** OpenAI text-embedding-3-large with FAISS cosine similarity indexing

---

