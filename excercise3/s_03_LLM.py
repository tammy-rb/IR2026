# s_03_LLM.py
# Stage 03: RAG evaluation (NO hybrid)
# Pipelines:
# (1) fixed + BM25
# (2) semantic + BM25
# (3) fixed + FAISS (OpenAI embeddings)
# (4) semantic + FAISS (OpenAI embeddings)
#
# Run:
#   python s_03_LLM.py --queries_json queries/queries.json --k1 3 --k2 5 --k3 10

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# Paths / Configuration
# ============================================================

BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))

BM25_DIR: str = os.path.join(BASE_DIR, "outputs", "bm25")
DENSE_DIR: str = os.path.join(BASE_DIR, "outputs", "embeddings_openai")
RAG_OUT_DIR: str = os.path.join(BASE_DIR, "outputs", "rag_runs")

BM25_FIXED_DIR: str = os.path.join(BM25_DIR, "fixed")
BM25_SEM_DIR: str = os.path.join(BM25_DIR, "semantic")

FAISS_FIXED_DIR: str = os.path.join(DENSE_DIR, "fixed_faiss")
FAISS_SEM_DIR: str = os.path.join(DENSE_DIR, "semantic_faiss")

OPENAI_EMBED_MODEL: str = "text-embedding-3-large"
STOP_WORDS: str = "english"
DEFAULT_CHAT_MODEL: str = "gpt-4o-mini"


# ============================================================
# Types
# ============================================================

RetrievedChunk = Tuple[Dict[str, Any], float, str]
QueryObj = Dict[str, Any]  # {"query": str, "expected_source": List[str]}


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    """
    Create a directory (and parents) if it doesn't already exist.

    Args:
        path: Directory path.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)


def extract_text(file_path: str) -> str:
    """
    Read a UTF-8 text file from disk.

    Args:
        file_path: Path to the source .txt document.

    Returns:
        The full file contents as a single string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def materialize_from_offsets(source_path: str, start_char: int, end_char: int) -> str:
    """
    Reconstruct chunk text from the original file using character offsets.

    Args:
        source_path: Path to the original source file.
        start_char: Inclusive start offset in the file text.
        end_char: Exclusive end offset in the file text.

    Returns:
        The substring source_text[start_char:end_char].
    """
    text: str = extract_text(source_path)
    return text[start_char:end_char]


def short_source_id(meta: Dict[str, Any]) -> str:
    """
    Create a compact citation id for a chunk (filename + offsets).

    Args:
        meta: Chunk metadata dictionary (must include source_path/start_char/end_char).

    Returns:
        A string like: "file.txt [123,456]"
    """
    base: str = os.path.basename(meta.get("source_path", ""))
    return f"{base} [{meta.get('start_char')},{meta.get('end_char')}]"


def assert_required_files_exist(paths: List[str]) -> None:
    """
    Validate that required artifact/index files exist.

    Args:
        paths: List of filesystem paths that must exist.

    Returns:
        None

    Raises:
        FileNotFoundError: if any path is missing.
    """
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


# ============================================================
# BM25 Loader + Search
# ============================================================

@dataclass
class BM25Index:
    """
    Holds BM25 artifacts required for retrieval.

    Attributes:
        bm25_matrix: Sparse BM25-weighted document-term matrix (N x V).
        vectorizer: CountVectorizer configured with the saved vocabulary.
        meta: Per-document metadata aligned to bm25_matrix row indices.
    """
    bm25_matrix: sparse.csr_matrix
    vectorizer: CountVectorizer
    meta: List[Dict[str, Any]]


def load_bm25_index(index_dir: str) -> BM25Index:
    """
    Load BM25 artifacts created by s_02_bm25_chunks.py.

    Expected files inside index_dir:
        - bm25_okapi.npz
        - vocabulary.json
        - meta.json

    Args:
        index_dir: Path to BM25 artifacts directory (fixed/semantic).

    Returns:
        BM25Index object.
    """
    assert_required_files_exist([
        os.path.join(index_dir, "bm25_okapi.npz"),
        os.path.join(index_dir, "vocabulary.json"),
        os.path.join(index_dir, "meta.json"),
    ])

    bm25 = sparse.load_npz(os.path.join(index_dir, "bm25_okapi.npz"))

    with open(os.path.join(index_dir, "vocabulary.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vectorizer = CountVectorizer(
        input="content",
        analyzer="word",
        stop_words=STOP_WORDS,
        vocabulary=vocab,
    )

    with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return BM25Index(bm25_matrix=bm25.tocsr(), vectorizer=vectorizer, meta=meta)


def bm25_search(bm25_index: BM25Index, query: str, k: int) -> List[RetrievedChunk]:
    """
    Retrieve top-k chunks using BM25 scoring.

    Args:
        bm25_index: Loaded BM25Index.
        query: User query string.
        k: Number of chunks to return.

    Returns:
        List of tuples (metadata, score, chunk_text) for the top-k results.
    """
    q_tf = bm25_index.vectorizer.transform([query]).tocsr().astype(float)
    scores = bm25_index.bm25_matrix.dot(q_tf.T).toarray().ravel()

    top_idx = np.argsort(-scores)[:k]
    results: List[RetrievedChunk] = []

    for i in top_idx:
        meta = bm25_index.meta[int(i)]
        text = materialize_from_offsets(meta["source_path"], meta["start_char"], meta["end_char"])
        results.append((meta, float(scores[int(i)]), text))

    return results


# ============================================================
# FAISS Loader + Search
# ============================================================

def load_faiss(index_dir: str) -> FAISS:
    """
    Load a FAISS vectorstore previously saved with FAISS.save_local().

    Args:
        index_dir: Path to FAISS directory (fixed_faiss/semantic_faiss).

    Returns:
        A LangChain FAISS vectorstore instance.
    """
    # FAISS.save_local creates multiple files; we just check the folder exists.
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"FAISS directory not found: {index_dir}")

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def dense_search(vs: FAISS, query: str, k: int) -> List[RetrievedChunk]:
    """
    Retrieve top-k chunks using dense semantic search (FAISS).

    Args:
        vs: Loaded LangChain FAISS vectorstore.
        query: User query string.
        k: Number of chunks to return.

    Returns:
        List of tuples (metadata, score, chunk_text) for the top-k results.
        Note: score is returned by LangChain/FAISS (often distance; lower may be better).
    """
    pairs = vs.similarity_search_with_score(query, k=k)
    results: List[RetrievedChunk] = []

    for doc, score in pairs:
        meta = dict(doc.metadata)

        # We prefer reconstructing from offsets so the "citations" are consistent
        # and so we rely on original source text.
        if meta.get("source_path") is not None and meta.get("start_char") is not None and meta.get("end_char") is not None:
            text = materialize_from_offsets(meta["source_path"], int(meta["start_char"]), int(meta["end_char"]))
        else:
            text = doc.page_content

        results.append((meta, float(score), text))

    return results


# ============================================================
# RAG Prompting
# ============================================================

def build_context_block(chunks: List[RetrievedChunk]) -> str:
    """
    Convert retrieved chunks into one context block for the LLM.

    Each chunk is prefixed with a citation marker like:
        [file.txt [123,456]]

    Args:
        chunks: Retrieved chunk tuples.

    Returns:
        A single string containing all chunks (with citations) separated by blank lines.
    """
    lines: List[str] = []
    for meta, _score, text in chunks:
        lines.append(f"[{short_source_id(meta)}]")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def answer_with_llm(llm: ChatOpenAI, query: str, context: str) -> str:
    """
    Ask the LLM to answer using ONLY the retrieved context.

    Args:
        llm: LangChain ChatOpenAI instance.
        query: User query string.
        context: Retrieved context block (citations + chunk text).

    Returns:
        The model response text (must include citations).
    """
    system = (
        "You are a question-answering assistant for a RAG system. "
        "Answer ONLY using the provided context. "
        "If the answer is not supported by the context, say: "
        "\"I don't know based on the retrieved chunks.\" "
        "Always cite sources in square brackets."
    )

    user = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    msg = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    return msg.content


# ============================================================
# Queries loader (supports strings OR {query, expected_source})
# ============================================================

def load_queries(path: str) -> Dict[str, List[QueryObj]]:
    """
    Load queries file.

    Supported formats:
    1) Strings:
        {"factual": ["...","...","...","..."], "conceptual": ["...","...","...","..."]}
    2) Objects:
        {"factual": [{"query":"...","expected_source":["file.txt"]}, ...], "conceptual": [...]}

    Args:
        path: Path to JSON file.

    Returns:
        Dict with keys 'factual' and 'conceptual'. Each value is a list of 4 objects:
            {"query": str, "expected_source": List[str]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "factual" not in data or "conceptual" not in data:
        raise ValueError("queries_json must be a dict with keys: 'factual' and 'conceptual'.")

    def normalize(lst: Any, name: str) -> List[QueryObj]:
        if not isinstance(lst, list) or len(lst) != 4:
            raise ValueError(f"{name} must contain exactly 4 queries.")
        out: List[QueryObj] = []
        for item in lst:
            if isinstance(item, str):
                out.append({"query": item, "expected_source": []})
            elif isinstance(item, dict) and "query" in item:
                out.append({
                    "query": str(item["query"]),
                    "expected_source": item.get("expected_source", []),
                })
            else:
                raise ValueError(f"Invalid query entry in {name}: {item}")
        return out

    return {
        "factual": normalize(data["factual"], "factual"),
        "conceptual": normalize(data["conceptual"], "conceptual"),
    }


# ============================================================
# Pipeline loading (load each resource once, reuse many times)
# ============================================================

@dataclass
class PipelineResources:
    """
    Resources for one pipeline.

    Only one of bm25_index or faiss_vs is used, depending on representation.
    """
    chunking: str                  # "fixed" / "semantic"
    representation: str            # "bm25" / "dense"
    bm25_index: Optional[BM25Index]
    faiss_vs: Optional[FAISS]


def load_pipeline_resources(chunking: str, representation: str) -> PipelineResources:
    """
    Load artifacts for a specific pipeline.

    Args:
        chunking: "fixed" or "semantic"
        representation: "bm25" or "dense"

    Returns:
        PipelineResources with loaded index (BM25 or FAISS).

    Raises:
        ValueError / FileNotFoundError if inputs are invalid or artifacts missing.
    """
    if chunking not in ("fixed", "semantic"):
        raise ValueError("chunking must be one of: fixed, semantic")
    if representation not in ("bm25", "dense"):
        raise ValueError("representation must be one of: bm25, dense")

    bm25_index: Optional[BM25Index] = None
    faiss_vs: Optional[FAISS] = None

    if representation == "bm25":
        index_dir = BM25_FIXED_DIR if chunking == "fixed" else BM25_SEM_DIR
        bm25_index = load_bm25_index(index_dir)

    if representation == "dense":
        index_dir = FAISS_FIXED_DIR if chunking == "fixed" else FAISS_SEM_DIR
        faiss_vs = load_faiss(index_dir)

    return PipelineResources(
        chunking=chunking,
        representation=representation,
        bm25_index=bm25_index,
        faiss_vs=faiss_vs,
    )


def run_retrieval(resources: PipelineResources, query: str, k: int) -> List[RetrievedChunk]:
    """
    Run retrieval for the given pipeline resources.

    Args:
        resources: PipelineResources for one pipeline.
        query: Query string.
        k: Top-k to retrieve.

    Returns:
        Retrieved chunks (metadata, score, text).
    """
    if resources.representation == "bm25":
        assert resources.bm25_index is not None
        return bm25_search(resources.bm25_index, query, k)

    if resources.representation == "dense":
        assert resources.faiss_vs is not None
        return dense_search(resources.faiss_vs, query, k)

    raise ValueError("Invalid pipeline representation.")


# ============================================================
# Main runner
# ============================================================

def main() -> None:
    """
    Run evaluation for:
      - 2 chunking methods (fixed, semantic)
      - 2 representations (bm25, dense)
      - 3 K values (k1, k2, k3)
      - 8 queries total (4 factual, 4 conceptual)

    Saves results to outputs/rag_runs/ as JSON.
    """
    parser = argparse.ArgumentParser(description="Stage 03: RAG evaluation for 4 pipelines (fixed/semantic × BM25/dense).")
    parser.add_argument("--queries_json", required=True, help="Path to queries.json")
    parser.add_argument("--k1", type=int, default=3, help="First K value")
    parser.add_argument("--k2", type=int, default=5, help="Second K value")
    parser.add_argument("--k3", type=int, default=10, help="Third K value")
    parser.add_argument("--llm_model", default=DEFAULT_CHAT_MODEL, help="Chat model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    args = parser.parse_args()

    ensure_dir(RAG_OUT_DIR)

    queries_by_type = load_queries(args.queries_json)
    llm = ChatOpenAI(model=args.llm_model, temperature=args.temperature)

    ks: List[int] = [args.k1, args.k2, args.k3]

    # Load resources once per pipeline (much faster and more stable).
    pipelines: List[PipelineResources] = [
        load_pipeline_resources("fixed", "bm25"),
        load_pipeline_resources("semantic", "bm25"),
        load_pipeline_resources("fixed", "dense"),
        load_pipeline_resources("semantic", "dense"),
    ]

    all_results: List[Dict[str, Any]] = []

    for qtype, qlist in queries_by_type.items():
        for qobj in qlist:
            query: str = qobj["query"]
            expected: List[str] = qobj.get("expected_source", [])

            for pipe in pipelines:
                for k in ks:
                    retrieved = run_retrieval(pipe, query, k)

                    context = build_context_block(retrieved)
                    answer = answer_with_llm(llm, query, context)

                    refs = [
                        {
                            "source_path": meta.get("source_path"),
                            "start_char": meta.get("start_char"),
                            "end_char": meta.get("end_char"),
                            "chunk_index": meta.get("chunk_index"),
                        }
                        for (meta, _score, _text) in retrieved
                    ]

                    # Console output for debugging
                    print("\n" + "=" * 90)
                    print(f"QueryType: {qtype}")
                    print(f"Query: {query}")
                    print(f"Expected source(s): {expected}")
                    print(f"Pipeline: chunking={pipe.chunking} | repr={pipe.representation} | K={k}")
                    print("-" * 90)
                    print("Top-K references:")
                    for r in refs:
                        base = os.path.basename(r["source_path"] or "")
                        print(f"- {base} | chunk={r.get('chunk_index')} | offsets=({r.get('start_char')},{r.get('end_char')})")
                    print("-" * 90)
                    print("Answer:")
                    print(answer)

                    all_results.append({
                        "query_type": qtype,
                        "query": query,
                        "expected_source": expected,
                        "pipeline": {
                            "chunking": pipe.chunking,
                            "representation": pipe.representation,
                        },
                        "k": k,
                        "references": refs,
                        "answer": answer,
                    })

    out_path = os.path.join(RAG_OUT_DIR, f"rag_4pipelines_k{args.k1}-{args.k2}-{args.k3}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ RAG evaluation saved to: {out_path}")


if __name__ == "__main__":
    main()
