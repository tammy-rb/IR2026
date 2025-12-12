# Exercise 3: 

### stage 01: Document Chunking

## Overview

This stage implements two document chunking strategies for parliamentary debates:
- **Fixed Chunking**: Size-based segmentation with sentence overlap
- **Semantic Chunking**: Embedding-based segmentation that respects meaning boundaries

## Semantic Chunking (Embedding-Based Segmentation)

### Method

In this project we implemented semantic chunking, an embedding-based segmentation method that aims to split long documents into coherent chunks according to meaning, rather than by a fixed size alone.

The process works as follows:

1. **Sentence Splitting**: Each document is split into full sentences while preserving their original character offsets
2. **Embedding Generation**: Each sentence is encoded into a dense semantic vector using a Sentence-Transformers (SBERT) model (`all-MiniLM-L6-v2`), which captures the contextual meaning in a high-dimensional embedding space
3. **Similarity Computation**: For every pair of adjacent sentences, we compute their cosine similarity
4. **Boundary Detection**: A semantic boundary is detected whenever the similarity between two consecutive sentences falls below a predefined threshold, indicating a meaningful topic or discourse shift
5. **Chunk Formation**: Chunks are formed by grouping consecutive sentences until a semantic boundary is reached, while also enforcing constraints:
   - Chunks contain only full sentences
   - Maximum 660 words per chunk (unless a single sentence is longer)
   - Minimum number of sentences per chunk to avoid very small chunks

### Similarity Threshold Selection

The similarity threshold was set to **0.62** based on empirical considerations and prior observations of cosine similarity distributions produced by SBERT embeddings.

**Rationale**:
- Values **above 0.62** typically indicate strong semantic continuity (e.g., elaboration or clarification of the same idea)
- Values **below 0.6** often correspond to topic changes, speaker shifts, or transitions between different arguments
- This threshold provides a balanced trade-off:
  - Conservative enough to prevent over-fragmentation into overly small chunks
  - Sensitive enough to capture genuine semantic shifts in parliamentary debates, which often evolve gradually rather than abruptly

This threshold supports the creation of chunks that are both semantically coherent and well-suited for downstream tasks such as retrieval, clustering, and classification.

## Configuration

### Fixed Chunking
- Max words per chunk: 660
- Overlap: 3 sentences between consecutive chunks

### Semantic Chunking
- Max words per chunk: 660
- Similarity threshold: 0.62
- Min sentences per chunk: 4
- Overlap: 0 sentences (optional)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

## Output

The script generates two JSONL files in `outputs/chunks/`:
- `chunks_fixed.jsonl`: Fixed-size chunks with sentence overlap
- `chunks_semantic.jsonl`: Semantically coherent chunks

Each chunk contains:
- Document ID and source path
- Corpus label (british/us)
- Chunking method
- Character offsets (start_char, end_char) (for retrieval)
- Chunk text
- Word count

### Note on Storage Format

Each chunk stores character-level offsets (`start_char`, `end_char`) pointing to its location in the original document. This pointer-based representation enables efficient reconstruction of the chunk text directly from the source file without re-running sentence splitting or chunking. 

The actual chunk text is also stored for convenience, debugging, and direct indexing in retrieval systems; however, it is optional and can be omitted in memory- or storage-constrained settings where only offsets are required.

## Usage

```bash
python s_01_chuncking.py
```