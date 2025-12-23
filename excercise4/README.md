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
