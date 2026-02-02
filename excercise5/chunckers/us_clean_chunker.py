import re
import json
from typing import List, Tuple
from chunckers.base import BaseChunker, Range


class USCleanerChunker(BaseChunker):
    """
    US Congressional Record cleaner chunker (TEXT-CHUNK MODE).

    - Splits by <pre>...</pre>
    - Cuts intro so chunk starts after the DATE line
    - Removes the speaker label (Ms./Mr./Mrs. NAME.)
    - Removes header noise: [Page ...], [Extensions of Remarks], GPO lines, HTML tags, separators
    - Returns FINAL cleaned chunk texts.
    - Offsets are set to NULL by BaseChunker (0,0).
    """

    method_name = "us_clean"

    def __init__(self, min_length: int = 100):
        self.min_length = min_length

    # ---- Tell BaseChunker to use text-chunk mode ----
    def supports_raw_text_chunks(self) -> bool:
        return True

    # ---- abstract method (won't be used in text-chunk mode) ----
    def _make_ranges(self, sentence_spans: List[Tuple[str, int, int]]) -> List[Range]:
        return []

    # -------------------------
    # Cleaning helpers
    # -------------------------
    def _strip_html_tags(self, text: str) -> str:
        return re.sub(r"<.*?>", "", text)

    def _normalize_whitespace(self, text: str) -> str:
        # In text-chunk mode it's fine to normalize, offsets are NULL anyway.
        return re.sub(r"\s+", " ", text).strip()

    def _strip_noise(self, t: str) -> str:
        # Remove bracket headers like [Extensions of Remarks], [Daily Digest], etc.
        t = re.sub(r"(?m)^\s*\[[^\]]+\]\s*$", "", t)

        # Remove [Page E635] lines
        t = re.sub(r"(?m)^\s*\[Page[^\]]+\]\s*$", "", t)

        # Remove [[Page E636]] anywhere
        t = re.sub(r"\[\[Page.*?\]\]", "", t)

        # Remove GPO header lines
        t = re.sub(r"(?im)^\s*From the Congressional Record Online.*?\n", "", t)
        t = re.sub(r"(?im)^\s*.*Government Publishing Office.*\n", "", t)
        t = re.sub(r"(?im)^\s*.*\bgpo\.gov\b.*\n", "", t)

        # Remove visual separators (underscores/equals runs)
        t = re.sub(r"[_=]{3,}", " ", t)

        # Remove NOTE markers
        t = re.sub(r"(?im)\bEND NOTE\b", " ", t)
        t = re.sub(r"(?im)\bNOTE\b", " ", t)

        # Strip HTML
        t = self._strip_html_tags(t)

        return t

    def _cut_after_date_and_strip_speaker(self, content: str) -> str:
        """
        Cut after the intro date line and remove the speaker label (Ms./Mr./Mrs. NAME.)
        while keeping the speech opening like "Mr. Speaker, ..."
        """

        # Date line: allow optional dot after year
        date_pat = re.search(
            r"(?im)\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*,\s*"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
            r"\d{1,2},\s*\d{4}\.?\s*",
            content,
        )
        if not date_pat:
            return content

        rel = date_pat.end()

        # consume whitespace after date
        ws = re.match(r"(?s)\s*", content[rel:])
        if ws:
            rel += ws.end()

        tail = content[rel:]

        # speaker label
        speaker = re.match(
            r"(?ims)"
            r"(?:\(?\s*)?"
            r"(Mr\.|Ms\.|Mrs\.)\s+"
            r"[A-Z][A-Za-z'\-]*"
            r"(?:\s+[A-Z][A-Za-z'\-]*)*"
            r"(?:\s+of\s+[A-Za-z\s]+)?"
            r"\.\s*",
            tail,
        )
        if speaker:
            rel += speaker.end()

        return content[rel:]

    # -------------------------
    # Text-chunk mode entrypoint
    # -------------------------
    def _make_text_chunks(self, text: str, sentence_spans: List[Tuple[str, int, int]]) -> List[str]:
        # Use original text directly; sentence_spans not required here
        pre_matches = list(re.finditer(r"<pre>(.*?)</pre>", text, re.DOTALL | re.IGNORECASE))
        out: List[str] = []

        for m in pre_matches:
            content = m.group(1)

            # Skip administrative noise
            if ("SENATE COMMITTEE MEETINGS" in content) or ("MEETINGS SCHEDULED" in content):
                continue

            speech = self._cut_after_date_and_strip_speaker(content)
            speech = self._strip_noise(speech)
            speech = self._normalize_whitespace(speech)

            if len(speech) < self.min_length:
                continue

            out.append(speech)

        return out

    def save_to_jsonl(self, materialized_chunks: list, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in materialized_chunks:
                record = {
                    "text": chunk.text,
                    "metadata": chunk.metadata if hasattr(chunk, "metadata") else {},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
