# Exercise 5: RAG System for News and Parliamentary Debates

## Preprocessing: BBC/NBC Corpus Cleaning (Phase 1–2)

### Problem
While collecting BBC and NBC articles as `.txt` files, we found that a substantial portion of the raw corpus is **not usable as news content**. The main issues were:

1. **NBC pages dominated by “Cookie Notice” boilerplate**, sometimes leaving only a real headline in the first line  
2. **BBC technical or error pages**, such as geo-blocking messages (“outside the UK”), short closed live pages, and topic/index pages  
3. **Non-news pages**, including podcasts, radio programs, and TV show listings  
4. **Many duplicates**, where different filenames contained identical content on the same day  

Without explicit preprocessing, these issues introduce noise and bias into retrieval and downstream analysis.

---

### Preprocessing Pipeline Overview
We implemented a **two-phase preprocessing pipeline** consisting of analysis and controlled cleanup.

---

### Phase 1: Analyzer (`analyze_corpus.py`)
The analyzer scans every `.txt` file and assigns one of three labels:

- **VALID** — usable news article  
- **JUNK** — technical/error pages, unusable content, or content that is too short  
- **NON_NEWS** — podcasts, programs, and other non-news pages (detected conservatively)

**Key steps:**

- **NBC cookie notice removal (in-place)**  
  If the string `"This Cookie Notice"` is detected, the file is truncated to keep only the text appearing before the cookie notice.  
  This operation overwrites the file and typically leaves only a headline when present.

- **Minimal-words threshold (applied after cleaning)**  
  Files below a source-specific threshold are labeled **JUNK**:
  - **BBC**: 80 words  
  - **NBC**: 30 words  

  Headline-only stubs are intentionally not retained, as they provide insufficient semantic signal for retrieval.

- **Same-day duplicate detection only**  
  The day is extracted from the filename timestamp.  
  Exact duplicates are detected **within the same day only**, using a normalized-content SHA256 hash.

**Outputs:**
- `data/cleanup_summary.json` — global statistics  
- `data/cleanup_manifest.json` — per-file metadata (category, reason, word count) and duplicate groups  

---

### Phase 2: Cleanup (`cleanup_corpus.py`)
The cleanup script consumes the manifest and performs controlled removal:

- Files can be either:
  - **moved** to a structured quarantine directory (recommended first), or  
  - **deleted** permanently
- Categories to remove can be selected (`junk`, `non_news`)
- Duplicate handling is optional:
  - Keeps **one file per same-day duplicate group**
  - Removes the remaining copies
- A **dry-run mode** is always available to preview actions

---

### Filtering Logic

#### NON_NEWS Detection
NON_NEWS pages are detected using **strong marker combinations** to minimize false positives.

Examples:
- **BBC**: podcast/radio pages require multiple markers such as  
  *“BBC Sounds”*, *“podcast”*, *“episode”*, *“duration”*  
- **BBC iPlayer/program pages** require iPlayer markers plus episode/program indicators  
- **NBC**: TV/show pages require multiple signals such as  
  *“full episodes”*, *“watch live”*, *“tv listings”*, *“cast”*, *“peacock”*

A file is labeled NON_NEWS only when several strong signals co-occur.

#### Too Short / Too Few Words → JUNK
Files with too few words after cleaning are labeled **JUNK**.  
This is a deliberate design choice:

- We do **not** maintain a separate category for short news stubs  
- Very short documents behave like noise in embedding-based retrieval  
- Excluding them improves semantic consistency and retrieval quality  

---

### Results (7,331 files)

#### Overall
- **VALID**: 4,650  
- **JUNK**: 2,277  
- **NON_NEWS**: 404  
- **Same-day duplicate groups**: 1,091 (2,208 files involved)

#### BBC (4,484 files)
- VALID: 3,408  
- JUNK: 677  
- NON_NEWS: 399  
- Same-day duplicate groups: 606 (1,221 files)

#### NBC (2,847 files)
- VALID: 1,242  
- JUNK: 1,600  
- NON_NEWS: 5  
- **Cookie cleanup (in-place)**:  
  1,601 files cleaned, 84,854 lines removed  
- Same-day duplicate groups: 485 (987 files)

---

### Indexing Assumptions (Document Granularity)

#### News Corpora (BBC/NBC)
Each file is treated as **one document (one vector)** and **no chunking** is applied.

**Rationale:**
- Each file typically corresponds to a single article and topic  
- Semantic coherence within a file is high  
- Chunking would add overhead and may fragment context without clear retrieval benefits  

#### Debate Corpora (Congress & UK/US Parliamentary Debates)
Chunking **is applied**, and each chunk is treated as one document.

**Rationale:**
- A single debate file often contains:
  - multiple speakers  
  - multiple sub-topics  
  - procedural and administrative sections  
- Embedding an entire debate as one vector mixes unrelated content  
- Chunking preserves topical locality and improves retrieval precision  

---

### Assumptions and Limitations (Living Section)

- **Cookie notice pages** do not represent crawler failures.  
  They are the result of **conditional content delivery** by news websites (e.g., consent or JavaScript requirements).  
  As a result, the corpus represents the subset of news articles that were accessible via static HTTP requests at crawl time.

- Missing articles therefore reflect **systematic accessibility constraints**, not random download errors.

- Same-day duplicate detection assumes that cross-day duplicates may correspond to legitimate updates or revisions and are therefore preserved.

This section documents preprocessing decisions and assumptions.  
Any future changes to thresholds, detection patterns, or document granularity rules will be recorded here.

---

*Last updated: January 28, 2026*
