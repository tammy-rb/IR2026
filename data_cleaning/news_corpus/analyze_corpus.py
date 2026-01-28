# ===========================
# analyze_corpus.py
# ===========================
"""
BBC/NBC Corpus Analyzer (Phase 1)

Scans BBC & NBC .txt files and classifies each into:
  - VALID: news content
  - JUNK: technical pages, errors, cookie policies, too-short content
  - NON_NEWS: podcasts, programs, entertainment (conservative detection)

Also detects DUPLICATES: identical content on the same day (from filename date).

Outputs TWO JSON files:
  1) data/cleanup_summary.json   - statistics and counts
  2) data/cleanup_manifest.json  - per-file metadata and duplicate groups

NOTES :
- NBC cookie policy is removed IN-PLACE (file overwrite) when detected.
- duplicates are only detected if files are on the SAME DAY.
- adds minimal-words threshold (after NBC cookie removal).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Literal, Optional
import re
import json
import logging
from datetime import datetime
from collections import Counter, defaultdict
import hashlib
import sys

# Add excercise5 directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "excercise5"))

from paths import BBC_NEWS_DIR, NBC_NEWS_DIR, PROJECT_ROOT

# -----------------------------
# Config
# -----------------------------
OUT_DIR = PROJECT_ROOT / "data"
SUMMARY_JSON = OUT_DIR / "cleanup_summary.json"
MANIFEST_JSON = OUT_DIR / "cleanup_manifest.json"
LOG_FILE = OUT_DIR / "cleanup_analyze.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimal words (after any cleaning)
MIN_WORDS = {
    "BBC": 80,   # BBC articles usually have content; <80 words is often junk/non-usable for you
    "NBC": 30,   # after cookie removal many files become 1-line headline; you want to treat as junk
}

# Optional: if you want to keep 1-line NBC headlines as VALID, set this True.
# You said you are NOT going to deal with short news; so default is False.
KEEP_NBC_HEADLINE_ONLY = False


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
# Helpers
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", re.UNICODE)


def word_count(text: str) -> int:
    """Approx word count for English-ish text."""
    return len(_WORD_RE.findall(text))


def remove_cookie_policy(lines: List[str]) -> Tuple[List[str], bool, int]:
    """
    Remove NBC cookie policy text from content (IN-PLACE overwrite will be done by caller).
    If file starts with headline followed by cookie text, keep only headline part.

    Returns:
      (cleaned_lines, changed?, cookie_start_idx)
    """
    if not lines:
        return lines, False, -1

    full_text = "".join(lines)
    if "This Cookie Notice" not in full_text:
        return lines, False, -1

    cookie_start_idx = -1
    for i, line in enumerate(lines):
        if "This Cookie Notice" in line:
            cookie_start_idx = i
            break

    if cookie_start_idx == -1:
        return lines, False, -1

    cleaned_lines = lines[:cookie_start_idx]
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    changed = len(cleaned_lines) != len(lines)
    return cleaned_lines, changed, cookie_start_idx


# -----------------------------
# Classification (conservative NON_NEWS)
# -----------------------------
def classify_text(lines: List[str], source: str) -> Tuple[Category, str, Dict]:
    """
    Returns: (category, reason, extras)
      extras includes:
        - num_lines
        - words
        - headline_only (for NBC)
    """
    if not lines:
        return "JUNK", "Empty file", {"num_lines": 0, "words": 0, "headline_only": False}

    num_lines = len(lines)
    first_line = lines[0].strip()
    full_text = "".join(lines)
    full_lc = full_text.lower()

    wc = word_count(full_text)

    # Heuristic: NBC headline-only files often end with " - NBC News"
    headline_only = (source == "NBC" and num_lines == 1 and first_line.endswith(" - NBC News"))

    # --- JUNK: minimal words (your request) ---
    if source in MIN_WORDS and wc < MIN_WORDS[source]:
        # If you REALLY want to keep NBC headline-only as valid, allow it optionally
        if source == "NBC" and headline_only and KEEP_NBC_HEADLINE_ONLY:
            pass
        else:
            return "JUNK", f"Too few words ({wc} < {MIN_WORDS[source]})", {
                "num_lines": num_lines, "words": wc, "headline_only": headline_only
            }

    # --- JUNK: very short lines (extra guard) ---
    if source == "BBC" and num_lines <= 5:
        return "JUNK", f"Too short ({num_lines} lines)", {"num_lines": num_lines, "words": wc, "headline_only": False}

    if source == "NBC":
        # if it's extremely short and not headline-like
        if num_lines <= 2 and not headline_only:
            return "JUNK", f"Too short ({num_lines} lines)", {"num_lines": num_lines, "words": wc, "headline_only": False}

    # --- BBC specific JUNK patterns ---
    if source == "BBC":
        if "BBC iPlayer isn't available" in full_text or "outside of the UK" in full_text:
            return "JUNK", "BBC geo-blocking error", {"num_lines": num_lines, "words": wc, "headline_only": False}
        if "Mrs Brown's Boys isn't currently available" in full_text:
            return "JUNK", "BBC geo-blocking error", {"num_lines": num_lines, "words": wc, "headline_only": False}

        if (("That's all for now" in full_text) or ("That's all from the live page" in full_text)) and num_lines < 20:
            return "JUNK", "BBC live blog closure (short)", {"num_lines": num_lines, "words": wc, "headline_only": False}

        if first_line.endswith("- BBC"):
            time_patterns = re.findall(r"\d+\s*(hrs?|days?|mins?)\s*ago", full_text)
            if len(time_patterns) > 5 and num_lines < 30:
                return "JUNK", "BBC topic index page", {"num_lines": num_lines, "words": wc, "headline_only": False}

    # --- NON_NEWS (low false-positive: require multiple strong signals) ---
    if source == "BBC":
        bbc_podcast_markers = [
            "bbc sounds", "bbc radio", "podcast", "more episodes",
            "episode:", "episode ", "broadcast", "duration:"
        ]
        hits = sum(1 for m in bbc_podcast_markers if m in full_lc)
        if hits >= 2 and num_lines < 350:
            return "NON_NEWS", "BBC podcast/radio page", {"num_lines": num_lines, "words": wc, "headline_only": False}

        has_iplayer = "bbc iplayer" in full_lc
        program_signals = [
            "this clip is from", "more clips from", "to watch this film",
            "series", "episodes", "episode:", "episode "
        ]
        if has_iplayer and any(s in full_lc for s in program_signals) and num_lines < 350:
            return "NON_NEWS", "BBC program/iplayer/clip page", {"num_lines": num_lines, "words": wc, "headline_only": False}

        if ("learning english" in full_lc) and ("grammar" in full_lc) and num_lines < 400:
            return "NON_NEWS", "BBC learning/education page", {"num_lines": num_lines, "words": wc, "headline_only": False}

    if source == "NBC":
        nbc_show_markers = [
            "full episodes", "watch live", "tv listings", "cast",
            "season", "episodes", "shows", "peacock"
        ]
        hits = sum(1 for m in nbc_show_markers if m in full_lc)
        if hits >= 3 and num_lines < 300:
            return "NON_NEWS", "NBC program/show/listing page", {"num_lines": num_lines, "words": wc, "headline_only": False}

    return "VALID", "Valid content", {"num_lines": num_lines, "words": wc, "headline_only": headline_only}


# -----------------------------
# Duplicate detection (same day only)
# -----------------------------
def normalize_for_exact_match(lines: List[str]) -> str:
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

    cookie_cleaned_files = 0
    cookie_cleaned_lines_removed_total = 0

    for i, fp in enumerate(files):
        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i+1}/{len(files)} files...")

        lines: List[str] = []
        cat: Category
        reason: str
        extras: Dict

        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # IN-PLACE cookie cleanup for NBC (per your request)
            if source_name == "NBC":
                original_line_count = len(lines)
                cleaned_lines, changed, cookie_start_idx = remove_cookie_policy(lines)
                if changed and cleaned_lines:
                    # overwrite file with cleaned version
                    with open(fp, "w", encoding="utf-8") as f:
                        f.writelines(cleaned_lines)
                    cookie_cleaned_files += 1
                    cookie_cleaned_lines_removed_total += max(0, original_line_count - len(cleaned_lines))
                    lines = cleaned_lines
                elif changed and not cleaned_lines:
                    # cookie policy consumed everything -> keep as empty for classification
                    lines = []

            cat, reason, extras = classify_text(lines, source=source_name)

        except Exception as e:
            lines = []
            cat, reason, extras = "JUNK", f"Error reading file: {e}", {"num_lines": 0, "words": 0, "headline_only": False}

        day_key = extract_day_key_from_filename(fp.name)

        by_category[cat].append(fp.name)
        reasons[cat][reason] += 1

        per_file[fp.name] = {
            "category": cat,
            "reason": reason,
            "day": day_key,
            "num_lines": extras.get("num_lines"),
            "words": extras.get("words"),
            "headline_only": extras.get("headline_only", False),
        }

        # duplicates: ONLY if same-day exists (per your request)
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
        "cookie_cleanup": {
            "enabled": source_name == "NBC",
            "files_cleaned": cookie_cleaned_files,
            "lines_removed_total": cookie_cleaned_lines_removed_total,
        },
        "min_words": {
            "threshold": MIN_WORDS.get(source_name),
            "keep_nbc_headline_only": KEEP_NBC_HEADLINE_ONLY if source_name == "NBC" else None,
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
    if source_name == "NBC":
        logger.info(f"Cookie cleaned files: {cookie_cleaned_files}, lines removed total: {cookie_cleaned_lines_removed_total}")
        logger.info(f"Min words threshold NBC: {MIN_WORDS.get('NBC')} (headline-only keep={KEEP_NBC_HEADLINE_ONLY})")
    else:
        logger.info(f"Min words threshold BBC: {MIN_WORDS.get('BBC')}")

    return {"summary": summary, "manifest": manifest}


def analyze_all() -> Dict:
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

    summary_out = {
        "timestamp": results["timestamp"],
        "overall": results["overall"],
        "BBC": results["sources"]["BBC"]["summary"],
        "NBC": results["sources"]["NBC"]["summary"],
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)

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
    logger.info("\n✅ Analysis complete!")

    return results


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("BBC/NBC Corpus Analyzer - Phase 1: Analysis")
    logger.info("="*70)
    analyze_all()
