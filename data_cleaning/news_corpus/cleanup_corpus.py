
"""
BBC/NBC Corpus Cleanup (Phase 2: Delete or Move)

Reads the manifest JSON created by analyze_corpus.py and:
  - Deletes OR moves files to quarantine folder
  - Organizes quarantine by source (BBC/NBC) and category (junk/non_news/duplicates)
  - Supports dry-run mode to preview actions

Duplicates:
- ONLY same-day duplicates exist in the manifest (because analyzer only records day_key-based dup groups).
- We keep ONE file per duplicate group (the first in sorted order) and remove the rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Tuple
import json
import logging
import shutil
import argparse
from datetime import datetime
import sys

# Add excercise5 directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "excercise5"))

from paths import BBC_NEWS_DIR, NBC_NEWS_DIR, PROJECT_ROOT

# -----------------------------
# Paths
# -----------------------------
OUT_DIR = PROJECT_ROOT / "data"
MANIFEST_JSON = OUT_DIR / "cleanup_manifest.json"
QUARANTINE_DIR = OUT_DIR / "quarantine"
LOG_FILE = OUT_DIR / "cleanup_action.log"

# Quarantine structure
DEST_MAP = {
    ("BBC", "junk"): QUARANTINE_DIR / "BBC" / "junk",
    ("BBC", "non_news"): QUARANTINE_DIR / "BBC" / "non_news",
    ("BBC", "duplicates"): QUARANTINE_DIR / "BBC" / "duplicates",
    ("NBC", "junk"): QUARANTINE_DIR / "NBC" / "junk",
    ("NBC", "non_news"): QUARANTINE_DIR / "NBC" / "non_news",
    ("NBC", "duplicates"): QUARANTINE_DIR / "NBC" / "duplicates",
}

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def base_dir_for(source: str) -> Path:
    if source == "BBC":
        return BBC_NEWS_DIR
    if source == "NBC":
        return NBC_NEWS_DIR
    raise ValueError(f"Unknown source: {source}")


def ensure_quarantine_dirs():
    for p in DEST_MAP.values():
        p.mkdir(parents=True, exist_ok=True)


def safe_move(src: Path, dest_dir: Path) -> Path:
    """
    Move src into dest_dir.
    If name conflict exists, append _dupN before suffix.
    Returns final dest path.
    """
    dest_path = dest_dir / src.name
    if not dest_path.exists():
        shutil.move(str(src), str(dest_path))
        return dest_path

    counter = 1
    stem = src.stem
    suffix = src.suffix
    while True:
        candidate = dest_dir / f"{stem}_dup{counter}{suffix}"
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return candidate
        counter += 1


def cleanup_corpus(
    manifest_path: Path,
    action: Literal["delete", "move"],
    categories: List[str],
    include_duplicates: bool,
    dry_run: bool,
) -> Dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if action == "move":
        ensure_quarantine_dirs()

    # Build targets: file_path -> (source, category, reason)
    targets: Dict[Path, Tuple[str, str, str]] = {}

    # 1) Category-based targets
    for source in ["BBC", "NBC"]:
        base = base_dir_for(source)

        if "junk" in categories:
            for filename in manifest[source]["by_category"].get("JUNK", []):
                fp = base / filename
                reason = manifest[source]["files"][filename]["reason"]
                targets[fp] = (source, "junk", reason)

        if "non_news" in categories:
            for filename in manifest[source]["by_category"].get("NON_NEWS", []):
                fp = base / filename
                reason = manifest[source]["files"][filename]["reason"]
                targets[fp] = (source, "non_news", reason)

    # 2) Duplicate-based targets (keep one per same-day group)
    duplicates_stats = {"groups": 0, "files_kept": 0, "files_removed": 0}

    if include_duplicates:
        for source in ["BBC", "NBC"]:
            base = base_dir_for(source)
            dup_groups = manifest[source].get("duplicates_groups", [])

            for group in dup_groups:
                files = sorted(group.get("files", []))
                if len(files) < 2:
                    continue

                duplicates_stats["groups"] += 1
                duplicates_stats["files_kept"] += 1
                kept = files[0]

                # remove rest
                for filename in files[1:]:
                    fp = base / filename
                    targets[fp] = (source, "duplicates", f"Duplicate of {kept}")
                    duplicates_stats["files_removed"] += 1

    targets_list = sorted(targets.keys(), key=lambda p: str(p))

    logger.info(f"\n{'='*70}")
    logger.info(f"Cleanup Action: {action.upper()}")
    logger.info(f"{'='*70}")
    logger.info(f"Categories: {', '.join(categories)}")
    logger.info(f"Include duplicates: {include_duplicates}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Total target files: {len(targets_list)}")
    if include_duplicates:
        logger.info(
            f"Duplicates: {duplicates_stats['groups']} groups, "
            f"{duplicates_stats['files_kept']} kept, {duplicates_stats['files_removed']} to remove"
        )

    stats = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "dry_run": dry_run,
        "categories": categories,
        "include_duplicates": include_duplicates,
        "total_targets": len(targets_list),
        "processed": 0,
        "missing": 0,
        "failed": 0,
        "duplicates_stats": duplicates_stats,
        "sample": [],  # small action sample for quick inspection
    }

    if dry_run:
        logger.info(f"\n⚠️  DRY RUN - Preview (first 80 files):")
        for fp in targets_list[:80]:
            source, cat, reason = targets[fp]
            if action == "move":
                dest_dir = DEST_MAP[(source, cat)]
                logger.info(f"  [MOVE] {source}/{fp.name} ({reason}) -> {dest_dir / fp.name}")
            else:
                logger.info(f"  [DELETE] {source}/{fp.name} ({reason})")

        if len(targets_list) > 80:
            logger.info(f"  ... and {len(targets_list) - 80} more files")

        stats["sample"] = [str(p) for p in targets_list[:20]]
        return stats

    # Execute
    logger.info(f"\n🔄 Processing {len(targets_list)} files...")
    for i, fp in enumerate(targets_list, start=1):
        source, cat, reason = targets[fp]
        try:
            if not fp.exists():
                stats["missing"] += 1
                continue

            if action == "delete":
                fp.unlink()
                stats["processed"] += 1

            elif action == "move":
                dest_dir = DEST_MAP[(source, cat)]
                final_dest = safe_move(fp, dest_dir)
                stats["processed"] += 1

            if len(stats["sample"]) < 25:
                stats["sample"].append(
                    {"file": fp.name, "source": source, "category": cat, "reason": reason, "action": action}
                )

            if i % 500 == 0:
                logger.info(f"  Processed {i}/{len(targets_list)}...")

        except Exception as e:
            stats["failed"] += 1
            logger.error(f"  ❌ Failed {action} {source}/{fp.name}: {e}")

    logger.info(f"\n{'='*70}")
    logger.info("✅ Cleanup complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Missing:   {stats['missing']}")
    logger.info(f"Failed:    {stats['failed']}")
    if action == "move":
        logger.info(f"Quarantine dir: {QUARANTINE_DIR}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Delete or move BBC/NBC corpus files based on analysis manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be moved (SAFE)
  python cleanup_corpus.py --dry-run --action move --categories junk

  # Move junk files to quarantine
  python cleanup_corpus.py --action move --categories junk

  # Delete junk files permanently
  python cleanup_corpus.py --action delete --categories junk

  # Move junk + non_news + duplicates (duplicates are same-day only)
  python cleanup_corpus.py --action move --categories junk non_news --duplicates

  # Delete everything except valid files
  python cleanup_corpus.py --action delete --categories junk non_news --duplicates
        """,
    )

    parser.add_argument(
        "--action",
        choices=["delete", "move"],
        required=True,
        help="Action to perform: delete permanently or move to quarantine",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["junk", "non_news"],
        required=True,
        help="Categories to process (can specify multiple)",
    )

    parser.add_argument(
        "--duplicates",
        action="store_true",
        help="Also remove duplicate files (same-day groups only; keeps one per group)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing (SAFE)",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_JSON,
        help=f"Path to manifest JSON (default: {MANIFEST_JSON})",
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("BBC/NBC Corpus Cleanup - Phase 2: Delete or Move")
    logger.info("="*70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    stats = cleanup_corpus(
        manifest_path=args.manifest,
        action=args.action,
        categories=args.categories,
        include_duplicates=args.duplicates,
        dry_run=args.dry_run,
    )

    stats_file = OUT_DIR / f"cleanup_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"\n📊 Stats saved to: {stats_file}")
    logger.info(f"📋 Log file:      {LOG_FILE}")


if __name__ == "__main__":
    main()
