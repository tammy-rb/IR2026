"""
BBC/NBC Cleanup Analyzer + Optional Deleter (with duplicate deletion)

PHASE 1 (SAFE): analyze_all()
- Scans BBC & NBC .txt files
- Classifies each file into: VALID / JUNK / NON_NEWS (conservative, low false-positive)
- Detects DUPLICATES: different filenames but identical content (after light normalization) on the same day
- Writes TWO JSON files:
    1) data/cleanup_summary.json
    2) data/cleanup_manifest.json  (includes filenames + per-file reason + duplicates groups)

PHASE 2 (OPTIONAL): delete_files(...)
- Deletes based on the previously written cleanup_manifest.json
- You choose:
    mode="junk_only" or mode="non_news_and_junk"
- Optionally also deletes duplicates (even VALID duplicates), keeping one per duplicate group.

Important:
- Analysis does NOT delete anything.
- Deletion is not by specific reasons (only by category + duplicate groups).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Literal, Optional, Set
import re
import json
import logging
from datetime import datetime
from collections import Counter, defaultdict
import hashlib


# -----------------------------
# Paths / Outputs
# -----------------------------
from paths import BBC_NEWS_DIR, NBC_NEWS_DIR, PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "data"
SUMMARY_JSON = OUT_DIR / "cleanup_summary.json"
MANIFEST_JSON = OUT_DIR / "cleanup_manifest.json"
LOG_FILE = OUT_DIR / "cleanup_analyze.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

Category = Literal["VALID", "JUNK", "NON_NEWS"]
DeleteMode = Literal["junk_only", "non_news_and_junk"]


# -----------------------------
# Filename day parsing (BBC/NBC scrape format)
# -----------------------------
BBC_NBC_TS_RE = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)_(\d{2})_([A-Za-z]{3})_(\d{4})_"
    r"(\d{2})_(\d{2})_(\d{2})_GMT\.txt$"
)

MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def extract_day_key_from_filename(filename: str) -> Optional[str]:
    """Return YYYY-MM-DD if filename matches BBC/NBC scrape timestamp format, else None."""
    m = BBC_NBC_TS_RE.match(filename)
    if not m:
        return None
    _dow, dd, mon, yyyy, *_ = m.groups()
    mm = MONTH_MAP.get(mon)
    if not mm:
        return None
    return f"{yyyy}-{mm}-{dd}"


# -----------------------------
# Classification (conservative NON_NEWS)
# -----------------------------
def classify_text(lines: List[str], source: str) -> Tuple[Category, str]:
    """
    Classify file content into:
      - JUNK: technical/no-content pages
      - NON_NEWS: strongly identifiable non-news pages (conservative)
      - VALID: assumed news content
    Returns: (category, reason)
    """
    if not lines:
        return "JUNK", "Empty file"

    num_lines = len(lines)
    first_line = lines[0].strip() if lines else ""
    first_5_lines = "".join(lines[:5])
    full_text = "".join(lines)
    full_lc = full_text.lower()

    # --- JUNK ---
    if num_lines <= 5:
        return "JUNK", f"Too short ({num_lines} lines)"

    if source == "NBC":
        # Cookie pages - only mark as JUNK if it's MOSTLY cookie policy
        # Strategy: if file contains cookie notice, check if most content is cookie boilerplate
        if "This Cookie Notice" in first_5_lines:
            # Count how much of the text is cookie policy boilerplate
            cookie_markers = [
                "This Cookie Notice", "NBCUniversal", "First-party Cookies",
                "Third-party Cookies", "Types of Cookies", "Strictly Necessary Cookies",
                "Measurement and Analytics", "Personalization Cookies", "COOKIE MANAGEMENT",
                "Browser Controls", "Analytics Provider Opt-Outs"
            ]
            cookie_marker_count = sum(1 for marker in cookie_markers if marker in full_text)
            
            # If file has >=7 cookie markers and is short, it's pure cookie policy
            if cookie_marker_count >= 7 and num_lines <= 60:
                return "JUNK", f"NBC Cookie Policy (mostly boilerplate, {num_lines} lines)"
            
            # If very short with cookie notice, also likely junk
            if num_lines <= 54:
                return "JUNK", f"NBC Cookie Policy ({num_lines} lines)"

    if source == "BBC":
        # Geo-blocking
        if "BBC iPlayer isn't available" in full_text or "outside of the UK" in full_text:
            return "JUNK", "BBC geo-blocking error"
        if "Mrs Brown's Boys isn't currently available" in full_text:
            return "JUNK", "BBC geo-blocking error"

        # Live blog closure (only if short)
        if (("That's all for now" in full_text) or ("That's all from the live page" in full_text)) and num_lines < 20:
            return "JUNK", "BBC live blog closure (short)"

        # Topic index page heuristic (narrow)
        if first_line.endswith("- BBC"):
            time_patterns = re.findall(r"\d+\s*(hrs?|days?|mins?)\s*ago", full_text)
            if len(time_patterns) > 5 and num_lines < 30:
                return "JUNK", "BBC topic index page"

    # --- NON_NEWS (low false-positive: require multiple strong signals) ---
    if source == "BBC":
        # podcasts/radio pages: require >=2 strong markers
        bbc_podcast_markers = [
            "bbc sounds", "bbc radio", "podcast", "more episodes",
            "episode:", "episode ", "broadcast", "duration:"
        ]
        hits = sum(1 for m in bbc_podcast_markers if m in full_lc)
        if hits >= 2 and num_lines < 300:
            return "NON_NEWS", "BBC podcast/radio page"

        # iPlayer/program/clip pages: require iPlayer + one additional strong program signal
        has_iplayer = "bbc iplayer" in full_lc
        program_signals = [
            "this clip is from", "more clips from", "to watch this film",
            "series", "episodes", "episode:", "episode "
        ]
        if has_iplayer and any(s in full_lc for s in program_signals) and num_lines < 300:
            return "NON_NEWS", "BBC program/iplayer/clip page"

        # education pages: strict
        if ("learning english" in full_lc) and ("grammar" in full_lc) and num_lines < 350:
            return "NON_NEWS", "BBC learning/education page"

    if source == "NBC":
        # show/listing pages: require >=3 markers
        nbc_show_markers = [
            "full episodes", "watch live", "tv listings", "cast",
            "season", "episodes", "shows", "peacock"
        ]
        hits = sum(1 for m in nbc_show_markers if m in full_lc)
        if hits >= 3 and num_lines < 250:
            return "NON_NEWS", "NBC program/show/listing page"

    return "VALID", "Valid content"


# -----------------------------
# Duplicate detection
# -----------------------------
def normalize_for_exact_match(lines: List[str]) -> str:
    """
    Conservative normalization for "exact" duplicates:
    - strip trailing spaces
    - collapse multiple blank lines
    - keep everything else
    """
    cleaned = [ln.rstrip() for ln in lines]
    out: List[str] = []
    blank_run = 0
    for ln in cleaned:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)
    return "\n".join(out).strip()


def content_hash(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------
# Analysis per source
# -----------------------------
def analyze_source(source_name: str, directory: Path) -> Dict:
    """
    Analyze one source directory, producing:
      - summary: counts, reasons, duplicates stats
      - manifest: per-file metadata, by-category lists, duplicates groups
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing {source_name}: {directory}")
    logger.info(f"{'='*70}")

    files = sorted(directory.glob("*.txt"))
    logger.info(f"Total files: {len(files)}")

    by_category: Dict[Category, List[str]] = {"VALID": [], "JUNK": [], "NON_NEWS": []}
    reasons: Dict[str, Counter] = {"JUNK": Counter(), "NON_NEWS": Counter(), "VALID": Counter()}
    per_file: Dict[str, Dict] = {}

    # day -> hash -> [filenames]
    dup_map: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for i, fp in enumerate(files):
        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i+1}/{len(files)} files...")

        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            cat, reason = classify_text(lines, source=source_name)
        except Exception as e:
            lines = []
            cat, reason = "JUNK", f"Error reading file: {e}"

        day_key = extract_day_key_from_filename(fp.name)
        by_category[cat].append(fp.name)
        reasons[cat][reason] += 1
        per_file[fp.name] = {"category": cat, "reason": reason, "day": day_key}

        if day_key:
            norm = normalize_for_exact_match(lines)
            h = content_hash(norm)
            dup_map[day_key][h].append(fp.name)

    # Build duplicates groups (only size >= 2)
    duplicates_groups: List[Dict] = []
    for day, hashes in dup_map.items():
        for h, names in hashes.items():
            if len(names) >= 2:
                duplicates_groups.append(
                    {"day": day, "hash": h, "files": sorted(names), "count": len(names)}
                )

    total_dup_files = sum(g["count"] for g in duplicates_groups)
    total_dup_groups = len(duplicates_groups)

    summary = {
        "total": len(files),
        "counts": {
            "VALID": len(by_category["VALID"]),
            "JUNK": len(by_category["JUNK"]),
            "NON_NEWS": len(by_category["NON_NEWS"]),
        },
        "reasons": {
            "JUNK": dict(reasons["JUNK"]),
            "NON_NEWS": dict(reasons["NON_NEWS"]),
        },
        "duplicates": {
            "groups": total_dup_groups,
            "files_in_duplicate_groups": total_dup_files,
        },
    }

    manifest = {
        "files": per_file,
        "by_category": by_category,
        "duplicates_groups": duplicates_groups,
    }

    logger.info(f"VALID: {summary['counts']['VALID']}")
    logger.info(f"JUNK: {summary['counts']['JUNK']}")
    logger.info(f"NON_NEWS: {summary['counts']['NON_NEWS']}")
    logger.info(f"Duplicates groups: {total_dup_groups}, files involved: {total_dup_files}")

    return {"summary": summary, "manifest": manifest}


def analyze_all() -> Dict:
    """
    Analyze BBC and NBC and write:
      - cleanup_summary.json
      - cleanup_manifest.json
    Returns in-memory results.
    """
    if not BBC_NEWS_DIR.exists():
        raise FileNotFoundError(f"BBC directory not found: {BBC_NEWS_DIR}")
    if not NBC_NEWS_DIR.exists():
        raise FileNotFoundError(f"NBC directory not found: {NBC_NEWS_DIR}")

    results = {"timestamp": datetime.now().isoformat(), "sources": {}}

    bbc = analyze_source("BBC", BBC_NEWS_DIR)
    nbc = analyze_source("NBC", NBC_NEWS_DIR)

    results["sources"]["BBC"] = bbc
    results["sources"]["NBC"] = nbc

    # Overall
    bbc_counts = bbc["summary"]["counts"]
    nbc_counts = nbc["summary"]["counts"]
    results["overall"] = {
        "total": bbc["summary"]["total"] + nbc["summary"]["total"],
        "counts": {
            "VALID": bbc_counts["VALID"] + nbc_counts["VALID"],
            "JUNK": bbc_counts["JUNK"] + nbc_counts["JUNK"],
            "NON_NEWS": bbc_counts["NON_NEWS"] + nbc_counts["NON_NEWS"],
        },
        "duplicates": {
            "groups": bbc["summary"]["duplicates"]["groups"] + nbc["summary"]["duplicates"]["groups"],
            "files_in_duplicate_groups": (
                bbc["summary"]["duplicates"]["files_in_duplicate_groups"]
                + nbc["summary"]["duplicates"]["files_in_duplicate_groups"]
            ),
        },
    }

    # Write summary JSON
    summary_out = {
        "timestamp": results["timestamp"],
        "overall": results["overall"],
        "BBC": results["sources"]["BBC"]["summary"],
        "NBC": results["sources"]["NBC"]["summary"],
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)

    # Write manifest JSON (includes filenames + duplicates groups)
    manifest_out = {
        "timestamp": results["timestamp"],
        "BBC": results["sources"]["BBC"]["manifest"],
        "NBC": results["sources"]["NBC"]["manifest"],
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, indent=2, ensure_ascii=False)

    logger.info(f"\n📄 Wrote summary JSON:  {SUMMARY_JSON}")
    logger.info(f"📄 Wrote manifest JSON: {MANIFEST_JSON}")
    logger.info(f"📋 Log file:            {LOG_FILE}")

    return results


# -----------------------------
# Optional deletion (call later)
# -----------------------------
def delete_files(
    manifest_json_path: Path,
    mode: DeleteMode,
    dry_run: bool = True,
    delete_duplicates: bool = True,
    keep_one_per_duplicate_group: bool = True,
) -> Dict:
    """
    Delete files based on a previously-created manifest JSON.

    Deletion policy:
    - Category deletion:
        mode="junk_only"         -> delete JUNK
        mode="non_news_and_junk" -> delete JUNK + NON_NEWS
    - Duplicate deletion (optional):
        delete_duplicates=True -> for each duplicates_group keep ONE (smallest filename), delete the rest.
        This applies to VALID duplicates too.

    Returns deletion stats (including missing/failed).
    """
    with open(manifest_json_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    def base_dir_for(source: str) -> Path:
        if source == "BBC":
            return BBC_NEWS_DIR
        if source == "NBC":
            return NBC_NEWS_DIR
        raise ValueError(f"Unknown source: {source}")

    def files_for_category(source: str, category: str) -> List[Path]:
        names = manifest[source]["by_category"].get(category, [])
        base = base_dir_for(source)
        return [base / n for n in names]

    targets: Set[Path] = set()

    # 1) Category-based targets
    for source in ["BBC", "NBC"]:
        targets.update(files_for_category(source, "JUNK"))
        if mode == "non_news_and_junk":
            targets.update(files_for_category(source, "NON_NEWS"))

    # 2) Duplicate-based targets
    duplicates_plan = {"groups_seen": 0, "groups_actioned": 0, "kept": 0, "to_delete": 0}

    if delete_duplicates:
        for source in ["BBC", "NBC"]:
            dup_groups = manifest[source].get("duplicates_groups", [])
            duplicates_plan["groups_seen"] += len(dup_groups)
            base = base_dir_for(source)

            for g in dup_groups:
                files = sorted(g.get("files", []))
                if len(files) < 2:
                    continue

                duplicates_plan["groups_actioned"] += 1

                if keep_one_per_duplicate_group:
                    # Keep smallest filename deterministically
                    keep_name = files[0]
                    duplicates_plan["kept"] += 1
                    delete_names = files[1:]
                else:
                    keep_name = None
                    delete_names = files[:]  # delete all (not recommended)

                for name in delete_names:
                    targets.add(base / name)
                    duplicates_plan["to_delete"] += 1

    # Execute
    targets_list = sorted(targets, key=lambda p: str(p))
    logger.info(f"\nDeletion mode: {mode}")
    logger.info(f"delete_duplicates={delete_duplicates}, keep_one_per_duplicate_group={keep_one_per_duplicate_group}")
    logger.info(f"Total target files (unique): {len(targets_list)}")
    if delete_duplicates:
        logger.info(f"Duplicates plan: {duplicates_plan}")

    deleted = 0
    failed = 0
    missing = 0

    if dry_run:
        logger.info("\n⚠️ DRY RUN: would delete the following files (sample up to 40):")
        for p in targets_list[:40]:
            logger.info(f"  WOULD DELETE: {p}")
        if len(targets_list) > 40:
            logger.info(f"  ... and {len(targets_list) - 40} more")
        return {
            "mode": mode,
            "dry_run": True,
            "delete_duplicates": delete_duplicates,
            "keep_one_per_duplicate_group": keep_one_per_duplicate_group,
            "target_unique": len(targets_list),
            "deleted": 0,
            "failed": 0,
            "missing": 0,
            "duplicates_plan": duplicates_plan,
        }

    logger.info("\n🗑️ Deleting files...")
    for i, p in enumerate(targets_list):
        try:
            if not p.exists():
                missing += 1
                continue
            p.unlink()
            deleted += 1
            if (i + 1) % 500 == 0:
                logger.info(f"  Deleted {i+1}/{len(targets_list)}...")
        except Exception as e:
            failed += 1
            logger.error(f"  ❌ Failed deleting {p}: {e}")

    return {
        "mode": mode,
        "dry_run": False,
        "delete_duplicates": delete_duplicates,
        "keep_one_per_duplicate_group": keep_one_per_duplicate_group,
        "target_unique": len(targets_list),
        "deleted": deleted,
        "failed": failed,
        "missing": missing,
        "duplicates_plan": duplicates_plan,
    }


if __name__ == "__main__":
    # SAFE analysis run:
    analyze_all()

    # Optional deletion examples (call manually when you want):
    # stats = delete_files(MANIFEST_JSON, mode="junk_only", dry_run=True, delete_duplicates=True)
    # stats = delete_files(MANIFEST_JSON, mode="non_news_and_junk", dry_run=False, delete_duplicates=True)
