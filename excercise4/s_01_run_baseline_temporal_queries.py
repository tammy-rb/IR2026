"""
s_01_run_baseline_temporal_queries.py

Exercise 4 - Stage 1 (Baseline failures):
Run the existing Exercise 3 LLM runner *without any temporal mechanism*,
using temporal evaluation queries, and write results directly into
Exercise 4 outputs for later manual failure analysis.

This script does NOT change retrieval, indexing, or ranking logic.
"""

from __future__ import annotations

import os
import subprocess

# -----------------------------
# Adjust these paths if needed
# -----------------------------

# Path to Exercise 3 runner
EX3_RUNNER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "excercise3", "s_03_RAG_llm_runner.py")
)

# Path to the temporal queries.json (inside Exercise 4)
QUERIES_JSON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "queries", "temporal_queries.json")
)

# Where Exercise 4 will keep baseline outputs (Stage 1 artifacts)
EX4_OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "outputs", "stage1_baseline_runs")
)

DEFAULT_KS = (3, 5, 10)
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


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

    # Force UTF-8 in the child process (prevents emoji/unicode console issues on Windows)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    # Print logs for traceability
    if completed.stdout.strip():
        print(completed.stdout)
    if completed.stderr.strip():
        print("⚠ STDERR:\n", completed.stderr)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    ensure_dir(EX4_OUT_DIR)

    if not os.path.isfile(EX3_RUNNER_PATH):
        raise FileNotFoundError(f"Exercise 3 runner not found: {EX3_RUNNER_PATH}")

    if not os.path.isfile(QUERIES_JSON_PATH):
        raise FileNotFoundError(f"Temporal queries not found: {QUERIES_JSON_PATH}")

    k1, k2, k3 = DEFAULT_KS

    run_ex3_runner(
        runner_path=EX3_RUNNER_PATH,
        queries_json=QUERIES_JSON_PATH,
        out_dir=EX4_OUT_DIR,
        k1=k1,
        k2=k2,
        k3=k3,
        llm_model=DEFAULT_LLM_MODEL,
        temperature=DEFAULT_TEMPERATURE,
    )

    print("✅ Baseline temporal run completed.")
    print("📁 Results written to:", EX4_OUT_DIR)


if __name__ == "__main__":
    main()
