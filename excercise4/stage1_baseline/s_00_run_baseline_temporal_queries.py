"""
s_00_run_baseline_temporal_queries.py

Exercise 4 - Stage 1 (Baseline failures):
Run the existing Exercise 3 LLM runner *without any temporal mechanism*,
using temporal evaluation queries, and write results directly into
Exercise 4 outputs for later manual failure analysis.

This script does NOT change retrieval, indexing, or ranking logic.
"""

from __future__ import annotations

import subprocess
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from paths import (
    EXERCISE3_DIR,
    TEMPORAL_QUERIES_JSON,
    STAGE1_DIR,
    ensure_dirs,
)

# Path to Exercise 3 runner script
EX3_RUNNER_PATH = EXERCISE3_DIR / "s_03_RAG_llm_runner.py"

DEFAULT_KS = (3, 5, 10)
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0


def run_ex3_runner(
    runner_path: str,
    queries_json: str,
    out_dir: str,
    k1: int,
    k2: int,
    k3: int,
    llm_model: str,
    temperature: float,
) -> None:
    """
    Run Exercise 3 runner via subprocess.

    Assumes Exercise 3 runner supports:
      --out_dir <path>
    """
    cmd = [
        "python",
        runner_path,
        "--queries_json",
        queries_json,
        "--k1",
        str(k1),
        "--k2",
        str(k2),
        "--k3",
        str(k3),
        "--llm_model",
        llm_model,
        "--temperature",
        str(temperature),
        "--out_dir",
        out_dir,
    ]

    print("▶ Running:", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    if completed.stdout.strip():
        print(completed.stdout)
    if completed.stderr.strip():
        print("⚠ STDERR:\n", completed.stderr)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    ensure_dirs()

    if not EX3_RUNNER_PATH.is_file():
        raise FileNotFoundError(f"Exercise 3 runner not found: {EX3_RUNNER_PATH}")

    if not TEMPORAL_QUERIES_JSON.is_file():
        raise FileNotFoundError(f"Temporal queries not found: {TEMPORAL_QUERIES_JSON}")

    k1, k2, k3 = DEFAULT_KS

    run_ex3_runner(
        runner_path=str(EX3_RUNNER_PATH),
        queries_json=str(TEMPORAL_QUERIES_JSON),
        out_dir=str(STAGE1_DIR),
        k1=k1,
        k2=k2,
        k3=k3,
        llm_model=DEFAULT_LLM_MODEL,
        temperature=DEFAULT_TEMPERATURE,
    )

    print("✅ Baseline temporal run completed.")
    print("📁 Results written to:", STAGE1_DIR)


if __name__ == "__main__":
    main()
