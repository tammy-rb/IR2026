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
python stage2_indexing/s_01_chuncking.py
```

Outputs:
- `outputs/chunks/chunks_fixed.jsonl`
- `outputs/chunks/chunks_semantic.jsonl`

Each line in these JSONL files is a `Chunk` dict including `doc_date_iso` and `doc_timestamp`.

Optional repair utility (if you already have chunk files and want to ensure timestamps are present):

```bash
python stage2_indexing/s_01_fix_timestamps.py
```

---

### Time Distribution Artifact (Histogram by Year)

To validate that temporal metadata was extracted and stored correctly, we generate a **time distribution report** (chunks per year).

Run:

```bash
python stage2_indexing/s_01_time_histogram.py
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
python stage2_indexing/s_02_embedding.py
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

**Note:** The temporal analysis and comparison scripts are located in `stage3_retrieval/`

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


### RAG Runner Script

The main RAG query runner is located at `stage3_retrieval/s_03_RAG_llm_runner.py`

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
| Current    | 0.6       | 365           | 
| Recent     | 0.65       | 365          | 
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
- **Current, recent** queries emphasize recent legislative activity using a half-life of approximately 1 year. align with annual legislative cycles
- **None** applies a very mild recency bias, where time functions primarily as a tie-breaker while preserving historical context.

This policy corrects the temporal blindness identified in the paper while respecting the slower temporal dynamics of institutional political discourse.

### Implementation Notes

- Temporal logic is implemented in `RAG_retriever/temporal_policy.py` and applied via `RAGRetriever.get_topk_timeaware`.
- Candidate sets are **oversampled** prior to temporal re-ranking (`K × 10`, minimum 50), following the post-processing design recommended in the paper.
- BM25 scores are log-normalized to align their scale with the `[0,1]` temporal component before fusion.
- Multiple temporal constraints extracted by Duckling are combined using **intersection semantics**, ensuring strict adherence to explicit time bounds.
- Open-ended expressions (e.g., “since 2022”, “until 2020”) are represented as semi-infinite intervals in the retrieval plan.
- Parameter sweeps and evaluation artifacts are stored under `outputs/rag_runs/` for reproducibility and analysis.

---

## Stage 3 — Temporal RAG Evaluation

### Overview

This report presents a **retrieval-level comparison** between a baseline
(time-blind) RAG system and a **time-aware RAG** system.

For each query, we show:
- Top-5 retrieved chunks (Baseline vs Time-Aware)
- Entry / exit changes in Top-5
- Baseline score vs Time-Aware score per chunk (when available)

No LLM calls are involved.

---

### Experimental Setup

**Query Set:**  
[queries/given_temporal_queries.json](queries/given_temporal_queries.json)

This file contains 20 temporal queries across four categories:
- **Point-in-time queries** (8 queries): Explicit year references (e.g., "in 2024")
- **Recency queries** (8 queries): Current position/latest discussion queries
- **Explicit range queries** (6 queries): Specific time periods (Q4 2023, Q3 2024)
- **Comparison queries** (4 queries): Cross-temporal analysis

**Evaluation Script:**  
`stage3_retrieval/compare_baseline_vs_timeaware.py`

**Execution:**
```bash
python stage3_retrieval/compare_baseline_vs_timeaware.py --queries_json queries/given_temporal_queries.json --k 3 5 10
```

**Output Files:**
- Raw comparison data: `outputs/rag_runs/stage3_temporal_analysis/stage3_given_temporal_queries_20260102_112311.json`
- Aggregated summary: `outputs/rag_runs/stage3_summaries/stage3_given_temporal_queries_20260102_112311__summary.json`

**Total comparisons:** 240 (20 queries × 4 pipelines × 3 k values)

---

### Illustrative Examples

The following examples demonstrate specific temporal corrections on individual queries.

## Query 1 — Recency (US Congress)

**Query:**  
*What is the current official position of the US Congress regarding the State of Israel?*

**Query Group:** `recency`  
**Pipeline:** `fixed / dense`  
**k = 10 , topn = 5`

### Top-5 Comparison

| Rank | Baseline (Date) | Time-Aware (Date) |
|----|------------------|-------------------|
| 1 | us:2023-10-25 | debates2025-10-29 |
| 2 | us:2025-05-05 | us:2025-10-06 |
| 3 | us:2023-10-20 | debates2025-09-10 |
| 4 | debates2024-03-12 | us:2025-09-09 |
| 5 | us:2024-07-25 | debates2025-09-10 |

### Delta Summary

- **Overlap:** 0 / 5  
- **Entered:** 5 (all 2025 documents)  
- **Left:** 5 (2023–2025 mixed baseline)

### Delta with Scores

**Entered**
| Chunk | Time-Aware Score |
|------|------------------|
| debates2025-10-29 | 0.622 |
| us:2025-10-06 | 0.610 |
| debates2025-09-10 | 0.599 |
| us:2025-09-09 | 0.594 |
| debates2025-09-10 | 0.593 |

**Left**
| Chunk | Baseline Score |
|------|----------------|
| us:2023-10-25 | 0.493 |
| us:2025-05-05 | 0.479 |
| us:2023-10-20 | 0.475 |
| debates2024-03-12 | 0.467 |
| us:2024-07-25 | 0.467 |

---

## Query 2 — Recency (British Parliament)

**Query:**  
*What is the current official position of the British Parliament regarding the State of Israel?*

**Query Group:** `recency`  
**Pipeline:** `fixed / dense`  
**k = 10 , topn = 5`

### Top-5 Comparison

| Rank | Baseline (Date) | Time-Aware (Date) |
|----|------------------|-------------------|
| 1 | debates2024-03-12 | debates2025-10-29 |
| 2 | debates2025-09-10 | debates2025-10-28 |
| 3 | debates2024-10-07 | debates2025-10-29 |
| 4 | debates2025-05-14 | debates2025-10-29 |
| 5 | debates2024-01-30 | debates2025-09-10 |

### Delta Summary

- **Overlap:** 1 / 5  
- **Entered:** 4 (new 2025 chunks)  
- **Left:** 4 (2024–early 2025 chunks)

### Delta with Scores

**Overlap**
| Chunk | Baseline | Time-Aware |
|------|----------|------------|
| debates2025-09-10 | 0.550 | 0.652 |

**Entered**
| Chunk | Time-Aware Score |
|------|------------------|
| debates2025-10-29 | 0.662 |
| debates2025-10-28 | 0.662 |
| debates2025-10-29 | 0.661 |
| debates2025-10-29 | 0.659 |

**Left**
| Chunk | Baseline Score |
|------|----------------|
| debates2024-01-30 | 0.543 |
| debates2024-03-12 | 0.561 |
| debates2024-10-07 | 0.547 |
| debates2025-05-14 | 0.546 |

---

## Query 3 — Recency / Topic Drift

**Query:**  
*What are the latest debates in Parliament about immigration policy?*

**Query Group:** `recency`  
**Pipeline:** `semantic / dense`  
**k = 10 , topn = 5`

### Top-5 Comparison

| Rank | Baseline (Date) | Time-Aware (Date) |
|----|------------------|-------------------|
| 1 | debates2025-07-07 | debates2025-10-13 |
| 2 | debates2024-05-14 | debates2025-07-07 |
| 3 | debates2024-05-14 | debates2025-09-01 |
| 4 | debates2024-02-26 | debates2025-09-15 |
| 5 | debates2024-05-14 | debates2025-07-21 |

### Delta Summary

- **Overlap:** 1 / 5  
- **Entered:** 4 (recent 2025 debates)  
- **Left:** 4 (2024 debates)

### Delta with Scores

**Overlap**
| Chunk | Baseline | Time-Aware |
|------|----------|------------|
| debates2025-07-07 | 0.534 | 0.627 |

**Entered**
| Chunk | Time-Aware Score |
|------|------------------|
| debates2025-10-13 | 0.637 |
| debates2025-09-01 | 0.621 |
| debates2025-09-15 | 0.617 |
| debates2025-07-21 | 0.616 |

**Left**
| Chunk | Baseline Score |
|------|----------------|
| debates2024-02-26 | 0.523 |
| debates2024-05-14 | 0.523 |
| debates2024-05-14 | 0.531 |
| debates2024-05-14 | 0.525 |

---

## Summary

- Mixed-year baselines are corrected into **temporally coherent Top-5 sets**
- Score deltas clearly show **temporal promotion**, not semantic noise
- Overlap remains when a chunk is both recent and semantically strong

---






















