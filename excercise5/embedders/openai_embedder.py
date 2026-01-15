"""
embedders/openai_embedder.py

OpenAI + Qdrant dense embedding strategy (streaming + resumable + reproducible).

What this file does
- Reads Chunk objects from a JSONL file (streaming, memory-safe).
- Embeds each chunk text using OpenAI embeddings.
- Upserts each embedding into a Qdrant collection as a "point":
    point.id      = deterministic UUID derived from chunk_uid (stable across runs)
    point.vector  = embedding vector
    point.payload = chunk metadata + chunk_uid + chunk text

Why Qdrant
- Qdrant stores vectors AND metadata together.
- At query time you can filter by metadata (e.g., 2-week windows via doc_timestamp range)
  before running vector similarity search.

Artifacts written locally (output_dir)
- qdrant_config_{run_id}.json: collection + model + vector_size + distance
- qdrant_run_meta_pre_{run_id}.json: run context + payload example (preview) + intended indexes
- qdrant_run_meta_post_{run_id}.json: run stats (seen/skipped/upserted) + timings

Requirements
- OPENAI_API_KEY available in environment
- qdrant-client installed and a Qdrant server running (local Docker or remote)
"""

from __future__ import annotations

import gc
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, TypeVar

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from embedders.base import BaseEmbedder
from models.chunk import Chunk

OPENAI_EMBED_MODEL = "text-embedding-3-large"

T = TypeVar("T")


def chunk_uid_to_uuid(chunk_uid: str) -> str:
    """
    Convert a chunk_uid string into a deterministic UUID (stable across runs).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_uid))


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        output_dir: Path,
        *,
        model: str = OPENAI_EMBED_MODEL,
        docs_batch_size: int = 256,
        embed_batch_size: int = 64,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "chunks_openai",
        distance: str = "cosine",  # "cosine" | "dot" | "euclid"
    ) -> None:
        super().__init__(output_dir)
        self.model = model
        self.docs_batch_size = docs_batch_size
        self.embed_batch_size = embed_batch_size
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.distance = distance.lower()

    @property
    def name(self) -> str:
        return "openai_qdrant"

    # ----------------------------
    # Small utilities
    # ----------------------------

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _run_id(self) -> str:
        """
        Timestamp run id used for filenames.
        Format: YYYYMMDD_HHMMSS
        """
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _write_json(self, filename: str, obj: Dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _payload_preview(self, payload: Dict[str, Any], *, text_preview_chars: int = 300) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in payload.items():
            if k == "text" and isinstance(v, str):
                out[k] = v[:text_preview_chars] + ("…" if len(v) > text_preview_chars else "")
            else:
                out[k] = v
        return out

    @staticmethod
    def batched(it: Iterator[Chunk], n: int) -> Iterator[List[Chunk]]:
        batch: List[Chunk] = []
        for x in it:
            batch.append(x)
            if len(batch) >= n:
                yield batch
                batch = []
        if batch:
            yield batch

    # ----------------------------
    # Qdrant schema helpers
    # ----------------------------

    def _distance_enum(self) -> qmodels.Distance:
        if self.distance in ("cosine", "cos"):
            return qmodels.Distance.COSINE
        if self.distance in ("dot", "ip", "inner"):
            return qmodels.Distance.DOT
        if self.distance in ("euclid", "l2"):
            return qmodels.Distance.EUCLID
        raise ValueError(f"Unsupported distance: {self.distance}")

    def _ensure_collection(self, client: QdrantClient, vector_size: int) -> None:
        existing = self._with_retry(lambda: client.get_collections()).collections
        exists = any(c.name == self.collection_name for c in existing)

        if exists:
            info = self._with_retry(lambda: client.get_collection(self.collection_name))
            size_ok = int(info.config.params.vectors.size) == int(vector_size)
            dist_ok = info.config.params.vectors.distance == self._distance_enum()
            if not (size_ok and dist_ok):
                raise ValueError(
                    f"Existing collection '{self.collection_name}' has "
                    f"size={info.config.params.vectors.size}, distance={info.config.params.vectors.distance} "
                    f"but embedder expects size={vector_size}, distance={self._distance_enum()}."
                )
        else:
            self._with_retry(
                lambda: client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(size=int(vector_size), distance=self._distance_enum()),
                )
            )

        # Always ensure indexes (new OR existing)
        self._ensure_payload_indexes(client)

    def _ensure_payload_indexes(self, client: QdrantClient) -> None:
        def safe_create_index(field_name: str, field_schema: qmodels.PayloadSchemaType) -> None:
            try:
                self._with_retry(
                    lambda: client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
                )
            except Exception as e:
                msg = str(e).lower()
                benign_markers = ("already", "exists", "duplicate", "conflict")
                if any(m in msg for m in benign_markers):
                    return
                raise

        safe_create_index("doc_timestamp", qmodels.PayloadSchemaType.INTEGER)
        for field in ("corpus", "doc_id"):
            safe_create_index(field, qmodels.PayloadSchemaType.KEYWORD)

    # ----------------------------
    # Payload + resumability
    # ----------------------------

    def _existing_ids(self, client: QdrantClient, ids: List[str]) -> Set[str]:
        if not ids:
            return set()
        try:
            pts = client.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_payload=False,
                with_vectors=False,
            )
            return {str(p.id) for p in pts}
        except Exception:
            return set()

    def _chunk_payload(self, c: Chunk) -> Dict[str, Any]:
        meta = self.extract_metadata(c)
        meta["chunk_uid"] = c.chunk_uid
        meta["text"] = c.text  # store chunk text in payload
        return meta

    # ----------------------------
    # Main ingestion
    # ----------------------------

    def embed_chunks(self, chunks_jsonl: Path) -> None:
        run_started_iso = self._now_iso()
        run_id = self._run_id()
        t0 = time.time()

        embeddings = OpenAIEmbeddings(model=self.model)
        client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

        chunks_iter = self.read_chunks_stream(chunks_jsonl)
        try:
            first_chunk = next(chunks_iter)
        except StopIteration:
            raise ValueError(f"No chunks found in: {chunks_jsonl}")

        first_vec = self._with_retry(lambda: embeddings.embed_query(first_chunk.text))
        vector_size = len(first_vec)

        self._ensure_collection(client, vector_size=vector_size)

        first_payload = self._chunk_payload(first_chunk)

        # --- Pre-run artifact (timestamped) ---
        self._write_json(
            f"qdrant_run_meta_pre_{run_id}.json",
            {
                "run_id": run_id,
                "run_started_utc": run_started_iso,
                "input_jsonl": str(chunks_jsonl),
                "embedder_name": self.name,
                "collection_name": self.collection_name,
                "qdrant_host": self.qdrant_host,
                "qdrant_port": self.qdrant_port,
                "model": self.model,
                "vector_size": vector_size,
                "distance": self.distance,
                "docs_batch_size": self.docs_batch_size,
                "embed_batch_size": self.embed_batch_size,
                "payload_keys_example": sorted(list(first_payload.keys())),
                "payload_example_preview": self._payload_preview(first_payload, text_preview_chars=300),
                "intended_payload_indexes": {
                    "doc_timestamp": "integer",
                    "corpus": "keyword",
                    "doc_id": "keyword",
                },
            },
        )

        # Config artifact (timestamped)
        self._write_json(
            f"qdrant_config_{run_id}.json",
            {
                "run_id": run_id,
                "collection_name": self.collection_name,
                "qdrant_host": self.qdrant_host,
                "qdrant_port": self.qdrant_port,
                "distance": self.distance,
                "model": self.model,
                "vector_size": vector_size,
            },
        )

        def chain_first() -> Iterator[Chunk]:
            yield first_chunk
            yield from chunks_iter

        total_seen = 0
        total_batches = 0
        total_skipped_existing = 0
        total_new_upserted = 0

        for batch_idx, batch in enumerate(self.batched(chain_first(), self.docs_batch_size), 1):
            total_batches += 1
            total_seen += len(batch)

            batch_ids = [chunk_uid_to_uuid(c.chunk_uid) for c in batch]
            existing = self._existing_ids(client, batch_ids)
            to_proc_idx = [i for i, pid in enumerate(batch_ids) if pid not in existing]

            skipped_here = len(batch) - len(to_proc_idx)
            total_skipped_existing += skipped_here

            if not to_proc_idx:
                print(f"  • batch {batch_idx} | skipped (all {len(batch)} already present)")
                continue

            texts = [batch[i].text for i in to_proc_idx]

            vectors: List[List[float]] = []
            for i in range(0, len(texts), self.embed_batch_size):
                part = texts[i : i + self.embed_batch_size]
                part_vecs = self._with_retry(lambda p=part: embeddings.embed_documents(p))
                vectors.extend(part_vecs)

            if len(vectors) != len(texts):
                raise RuntimeError(f"Embedding count mismatch: got {len(vectors)} vectors for {len(texts)} texts")

            points: List[qmodels.PointStruct] = []
            for local_i, v in enumerate(vectors):
                idx = to_proc_idx[local_i]
                c = batch[idx]
                pid = batch_ids[idx]
                points.append(qmodels.PointStruct(id=pid, vector=v, payload=self._chunk_payload(c)))

            self._with_retry(lambda: client.upsert(collection_name=self.collection_name, points=points))

            total_new_upserted += len(points)
            print(f"  ✓ batch {batch_idx} | upserted {len(points)} | total new: {total_new_upserted}")

            try:
                del vectors
                del points
                del texts
            except Exception:
                pass
            gc.collect()

        elapsed_s = time.time() - t0
        run_finished_iso = self._now_iso()

        collection_points_count = None
        try:
            info = self._with_retry(lambda: client.get_collection(self.collection_name))
            collection_points_count = getattr(info, "points_count", None)
        except Exception:
            pass

        # --- Post-run artifact (timestamped) ---
        self._write_json(
            f"qdrant_run_meta_post_{run_id}.json",
            {
                "run_id": run_id,
                "run_started_utc": run_started_iso,
                "run_finished_utc": run_finished_iso,
                "elapsed_seconds": elapsed_s,
                "input_jsonl": str(chunks_jsonl),
                "collection_name": self.collection_name,
                "model": self.model,
                "vector_size": vector_size,
                "distance": self.distance,
                "total_chunks_seen": total_seen,
                "total_batches": total_batches,
                "total_skipped_existing": total_skipped_existing,
                "total_new_upserted": total_new_upserted,
                "collection_points_count_best_effort": collection_points_count,
                "notes": "new_upserted counts only points not already present (based on deterministic IDs).",
            },
        )

        print(f"✅ Qdrant upsert complete: collection='{self.collection_name}', total_new={total_new_upserted}")

    # ----------------------------
    # Retry helper
    # ----------------------------

    def _with_retry(self, fn: Callable[[], T], *, max_retries: int = 6, base_sleep_s: float = 2.0) -> T:
        last_err: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    raise
                sleep_s = min(base_sleep_s * (2**attempt), 60.0)
                print(f"  ⚠️  call failed ({e.__class__.__name__}: {e}). Retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)

        assert last_err is not None
        raise last_err
