# s_00_clean_classification_words.py

"""
Cleaning helpers to remove classification-leak information:

1. Remove all country tokens "US" / "U.S." / "UK" / "U.K." as standalone words
   (case-insensitive) from ALL texts.

2. For US Congressional debate files:
   - Many files contain multiple records.
   - Each record is wrapped in a <pre>...</pre> block.
   - We want to DROP all surrounding and internal boilerplate/header lines
     and KEEP only the actual body text for EACH <pre> block.

   Strategy for US files:
   - Find every <pre ...> ... </pre> block.
   - Inside each block, skip initial lines that are boilerplate:
       * empty lines
       * [Extensions of Remarks], [Page E###], etc.  (lines starting with '[' and ending with ']')
       * the "From the Congressional Record Online through the Government Publishing Office ..." line
   - Keep everything after that as the body.
   - Concatenate all bodies from all <pre> blocks with a blank line between them.
"""

import re

# Standalone US/UK tokens (with/without dots), case-insensitive
COUNTRY_PATTERNS = [
    r"(?<!\w)(?:u\.?s\.?|us)(?!\w)",
    r"(?<!\w)(?:u\.?k\.?|uk)(?!\w)",
]

PRE_OPEN_RE = re.compile(r"<pre[^>]*>", re.IGNORECASE)
PRE_CLOSE_RE = re.compile(r"</pre>", re.IGNORECASE)


def remove_country_tokens(text: str) -> str:
    """
    Replace standalone occurrences of 'US' / 'U.S.' / 'UK' / 'U.K.'
    (case-insensitive) with a space.
    """
    out = text
    for pat in COUNTRY_PATTERNS:
        out = re.sub(pat, " ", out, flags=re.IGNORECASE)
    return out


def _extract_us_bodies_from_pre_blocks(text: str) -> str:
    """
    Extract all bodies from <pre>...</pre> blocks in a US Congressional file.

    For each <pre> block:
    - Ignore everything before <pre> and after </pre>.
    - Inside the block, skip initial boilerplate lines:
        * blank lines
        * lines like "[Extensions of Remarks]", "[Page E642]" etc.
        * the line starting with "From the Congressional Record Online
          through the Government Publishing Office"
    - Keep all remaining lines as the body for that block.

    Return all bodies joined with a blank line between them.
    If no <pre> is found, fall back to the original text.
    """
    bodies = []
    pos = 0
    n_text = len(text)

    while True:
        # Find next <pre> block
        m_open = PRE_OPEN_RE.search(text, pos)
        if not m_open:
            break
        start_content = m_open.end()
        m_close = PRE_CLOSE_RE.search(text, start_content)
        # Find </pre> or end of text
        if m_close:
            end_content = m_close.start()
            pos = m_close.end()
        else:
            # No closing tag found; take until end of text
            end_content = n_text
            pos = n_text

        block = text[start_content:end_content]
        lines = block.splitlines()

        cleaned_lines = []
        skipping = True

        for ln in lines:
            stripped = ln.strip()

            if skipping:
                # Skip boilerplate at top of <pre> block
                if (
                    stripped == ""
                    or (stripped.startswith("[") and stripped.endswith("]"))
                    or stripped.startswith(
                        "From the Congressional Record Online through the Government Publishing Office"
                    )
                ):
                    continue
                # First non-boilerplate line – from here on we keep everything
                skipping = False

            cleaned_lines.append(ln)

        cleaned_block = "\n".join(cleaned_lines).strip()
        if cleaned_block:
            bodies.append(cleaned_block)

    if bodies:
        # Join all speech bodies from the file
        return "\n\n".join(bodies)
    else:
        # Fallback: nothing matched, return original
        return text


def remove_classification_words(text: str, country: str) -> str:
    """
    Apply all cleaning steps that remove trivial US/UK classification signals.

    - Always remove US/UK country tokens.
    - For US files, also strip Congressional headers and keep only bodies
      from <pre> blocks.
    """
    cleaned = remove_country_tokens(text)

    if country and country.lower() == "us":
        cleaned = _extract_us_bodies_from_pre_blocks(cleaned)

    return cleaned
