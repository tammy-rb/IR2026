# Exercise 5: RAG System for News and Parliamentary Debates

## Preprocessing: BBC/NBC corpus cleaning (Phase 1–2)

### Problem
While collecting BBC and NBC articles as `.txt` files, we found that a large portion of the corpus is **not usable news content**. The main issues were:  
1) **NBC pages dominated by “Cookie Notice” boilerplate** (sometimes with only a real headline at the top),  
2) **BBC technical/error pages** (geo-blocking, “outside the UK”, short closed live pages, topic/index pages),  
3) **non-news pages** (podcasts/programs), and  
4) **many duplicates**: different filenames but identical content on the same day.

### What we did
We implemented a two-phase pipeline:

1) **Analyzer (Phase 1 — `analyze_corpus.py`)**
   - Scans every `.txt` file and labels it as:
     - **VALID** (news content)
     - **JUNK** (technical/error/unusable/too-short)
     - **NON_NEWS** (podcasts/programs; conservative detection)
   - **NBC cookie removal (in-place):** if `"This Cookie Notice"` appears, we truncate the file to keep only the text before the cookie notice (often leaving only the headline).
   - **Minimal-words threshold (after cleaning):**
     - BBC: **80** words minimum
     - NBC: **30** words minimum  
     Files below threshold are marked **JUNK** (headline-only stubs are intentionally not kept).
   - **Same-day duplicates only:** we extract the day from the filename timestamp and group **exact duplicates** within the same day using a normalized-content SHA256 hash.

   Outputs:
   - `data/cleanup_summary.json` (counts and stats)
   - `data/cleanup_manifest.json` (per-file category/reason + duplicate groups)

2) **Cleanup (Phase 2 — `cleanup_corpus.py`)**
   - Reads the manifest and either:
     - **moves** selected files to quarantine (recommended first), or
     - **deletes** them permanently.
   - Supports selecting categories (`junk`, `non_news`) and optionally removing **duplicate copies** (keeps one file per same-day duplicate group).
   - Supports **dry-run** to preview actions.

### Filtering logic
- **NON_NEWS** is detected with **strong marker combinations** (low false positives).  
  Example signals: BBC Sounds/podcast markers; BBC iPlayer + episode/program markers; NBC show/listing markers.  
  We require multiple markers before labeling a file as NON_NEWS.

- **Too short / too few words → JUNK** (after NBC cookie removal).  
  This is intentional: we avoid indexing “headline-only” or low-signal pages because they behave like noise in retrieval, and we are not maintaining a separate category for short news stubs.

### Results (run on 7,331 files)
- **Overall (7,331)**
  - VALID: **4,650**
  - JUNK: **2,277**
  - NON_NEWS: **404**
  - Same-day duplicate groups: **1,091** (2,208 files involved)

- **BBC (4,484)**
  - VALID: **3,408**
  - JUNK: **677**
  - NON_NEWS: **399**
  - Same-day duplicate groups: **606** (1,221 files involved)

- **NBC (2,847)**
  - VALID: **1,242**
  - JUNK: **1,600**
  - NON_NEWS: **5**
  - Cookie cleanup (in-place): **1,601 files cleaned**, **84,854 lines removed**
  - Same-day duplicate groups: **485** (987 files involved)

### Indexing assumption (document granularity)
- **BBC/NBC news:** 1 file = 1 document (1 vector), **no chunking**.  
  Each file typically represents one coherent article/topic, so chunking would add overhead and may reduce semantic coherence.

- **Debate corpora (Congress + UK/US parliamentary debates):** **chunking enabled**.  
  A single debate file often contains multiple speakers and sub-topics; embedding the entire file as one vector mixes unrelated content and reduces retrieval precision. Each chunk is indexed as one document (one vector).

### Notes (living section)
This section documents preprocessing decisions. If we change thresholds, add strong patterns, or adjust document granularity rules, we update this section accordingly.

