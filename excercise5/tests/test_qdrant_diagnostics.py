"""
Diagnostic script that prints Qdrant connectivity and storage issues.

Run this script when the retriever fails to reach Qdrant. It will:
  1. Check that the TCP port is reachable.
  2. Ask Qdrant for its collection list and report protocol failures.
  3. Inspect the local Docker volume for collection folders and probe each one.

The goal is to highlight which collection is causing the service to drop the
connection so you can decide which on-disk shard needs to be rebuilt.
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path
from typing import Iterable, List

# Allow imports from excercise5 package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import QDRANT_HOST, QDRANT_PORT
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException


def _print_banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _check_tcp(host: str, port: int, timeout_s: float = 5.0) -> bool:
    print("1) Checking TCP connectivity...")
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            print(f"   OK: established TCP connection to {host}:{port}.")
            return True
    except Exception as exc:  # noqa: BLE001 - surfaced for diagnostics
        print(f"   FAIL: unable to reach {host}:{port} ({exc}).")
        print("   Hint: ensure the Docker container is running: 'docker compose up -d'.")
        return False


def _list_collections(host: str, port: int) -> bool:
    print("2) Listing collections via API...")
    client = QdrantClient(host=host, port=port, timeout=20.0)
    try:
        response = client.get_collections()
    except ResponseHandlingException as exc:
        root = exc.__cause__ or exc
        print("   FAIL: Qdrant closed the connection while listing collections.")
        print(f"   Root cause: {root.__class__.__name__}: {root}")
        print("   Hint: check 'docker logs qdrant' for panic messages.")
        return False
    except Exception as exc:  # noqa: BLE001
        print("   FAIL: unexpected exception while calling get_collections().")
        print(f"   Exception: {exc.__class__.__name__}: {exc}")
        return False

    names = sorted(col.name for col in response.collections)
    if names:
        print(f"   OK: collections found -> {', '.join(names)}")
    else:
        print("   OK: no collections present.")
    return True


def _storage_collections() -> List[str]:
    repo_root = Path(__file__).resolve().parents[2]
    storage_dir = repo_root / "servers" / "qdrant" / "qdrant_data" / "collections"
    print("3) Inspecting local storage volume...")
    if not storage_dir.exists():
        print(f"   WARN: storage directory does not exist at {storage_dir}.")
        return []

    collection_dirs = sorted(p.name for p in storage_dir.iterdir() if p.is_dir())
    if collection_dirs:
        print(f"   Found on-disk collections -> {', '.join(collection_dirs)}")
    else:
        print("   No on-disk collections detected under the Docker volume.")
    return collection_dirs


def _probe_each_collection(host: str, port: int, names: Iterable[str]) -> List[str]:
    suspects: List[str] = []
    for name in names:
        print(f"   Probing collection '{name}'...")
        time.sleep(1.0)
        client = QdrantClient(host=host, port=port, timeout=20.0)
        try:
            client.get_collection(name)
            print("     OK: responded without protocol errors.")
        except ResponseHandlingException as exc:
            root = exc.__cause__ or exc
            print("     FAIL: connection dropped while loading this collection.")
            print(f"     Root cause: {root.__class__.__name__}: {root}")
            suspects.append(name)
            print("     Hint: this shard likely needs to be deleted and re-ingested.")
            time.sleep(2.0)
        except Exception as exc:  # noqa: BLE001
            print("     FAIL: unexpected exception.")
            print(f"     Exception: {exc.__class__.__name__}: {exc}")
            suspects.append(name)
            time.sleep(2.0)
    return suspects


def main() -> None:
    _print_banner("Qdrant Diagnostics")

    host = QDRANT_HOST
    port = QDRANT_PORT
    print(f"Target endpoint: {host}:{port}")

    if not _check_tcp(host, port):
        return

    healthy = _list_collections(host, port)
    collections = _storage_collections()

    if healthy:
        print("\nQdrant responded to /collections. No protocol issues detected.")
        return

    if not collections:
        print("\nNo local collections to inspect. Focus on container logs and network state.")
        return

    suspects = _probe_each_collection(host, port, collections)

    if suspects:
        print("\nPotentially corrupted collections: " + ", ".join(suspects))
        print("Suggested fix steps:")
        print("  1. Stop Qdrant: cd servers/qdrant && docker compose down")
        print("  2. Back up then delete the folders for the listed collections inside")
        print("     servers/qdrant/qdrant_data/collections/")
        print("  3. Restart Qdrant: docker compose up -d")
        print("  4. Re-run the embedding scripts to rebuild those collections.")
    else:
        print("\nAll collections responded individually. Investigate network or auth issues.")


if __name__ == "__main__":
    main()
