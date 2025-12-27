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


### stage3
Stage 3 — Temporal Query Resolution (Duckling + LLM)

In Stage 3, we implement a time-aware query resolution pipeline whose sole responsibility is to determine whether a user query contains temporal constraints and, if so, to extract and normalize them.

This stage does not handle evolutionary (change-over-time) queries.
Evolutionary reasoning is implemented separately in a later stage.

Overview

Each query is processed using a hard-then-soft resolution strategy:

Duckling-based extraction (primary, deterministic)

LLM-based fallback (only when necessary)

The pipeline always produces a structured temporal resolution output, even when the query is determined to be time-independent.

Step 1 — Duckling (Primary Resolution)

The query is first passed to Duckling, which attempts to extract explicit temporal expressions, including:

Explicit dates and years
(e.g., “2018”, “March 2024”)

Standard date ranges
(e.g., “between 2019 and 2021”)

Relative expressions anchored to query time
(e.g., “last month”, “this year”)

If Duckling successfully extracts temporal information:

The extracted expressions are normalized into absolute time ranges

The LLM is not invoked

The result is treated as high-confidence

Step 2 — LLM Fallback (Implicit Temporal Intent)

If Duckling fails to detect any temporal expressions, an LLM is invoked as a fallback mechanism.

The LLM is used strictly to:

Detect implicit temporal intent
(e.g., “current”, “latest”, “recent”)

Resolve ambiguous time references
(e.g., “last quarter”, “the most recent period”)

Decide whether the query is:

time-constrained, or

time-independent

The LLM returns a strictly structured JSON output describing:

Temporal intent

Optional inferred time ranges

A confidence score

Free-form interpretation is not allowed.

Step 3 — No-Time Classification

If neither Duckling nor the LLM identifies a temporal constraint:

The query is explicitly classified as time-independent

An empty list of time ranges is returned

Downstream retrieval proceeds without temporal filtering or time-based re-ranking

This explicit classification prevents unintended temporal bias in non-temporal queries.

Output Contract

Stage 3 always returns a uniform temporal resolution object containing:

Temporal intent

Zero or more normalized time ranges

Anchor time (e.g., query time or corpus end)

Confidence metadata

This output is consumed by:

Hard temporal filtering

Soft time-decay re-ranking

Key Properties

Deterministic handling of explicit temporal expressions

Minimal and controlled LLM usage

Clear separation between temporal and non-temporal queries

Fully testable and reproducible behavior

first we will do only duckling.
in hard: if no date - no filter
in soft - no date - use t as reference point

for evolution
use duckling to fine the 2 ranges.

we will not implemtn llm fallabck yet.
it is optional.

