# s_00_clean_classification_words.py

"""
Cleaning helpers to remove classification-leak information:

1. Remove all country tokens "US" / "U.S." / "UK" / "U.K." as standalone words
   (case-insensitive) from ALL texts.

2. For US Congressional debate files:
   - Extract text from each <pre>...</pre> block
   - Remove the 3 specific header lines:
     * [Extensions of Remarks]
     * [Page E###]
     * From the Congressional Record Online through the Government Publishing Office [<a href...>]
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
    1. Extract everything between <pre> and </pre>
    2. Remove the 3 specific header lines:
       - Lines starting with [Extensions of Remarks]
       - Lines starting with [Page E###] or similar
       - The GPO line: "From the Congressional Record Online..."
    
    Return all cleaned bodies joined with blank lines.
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
        
        if m_close:
            end_content = m_close.start()
            pos = m_close.end()
        else:
            end_content = n_text
            pos = n_text

        block = text[start_content:end_content]
        
        # Remove the 3 specific header lines
        # Line 1: [Extensions of Remarks]
        block = re.sub(r'^\s*\[Extensions of Remarks\]\s*$', '', block, flags=re.MULTILINE | re.IGNORECASE)
        
        # Line 2: [Page E###] or [Pages E###-E###] or similar page references
        block = re.sub(r'^\s*\[Pages? [^\]]+\]\s*$', '', block, flags=re.MULTILINE | re.IGNORECASE)
        
        # Line 3: From the Congressional Record Online... (with or without the link)
        block = re.sub(r'^.*From the Congressional Record Online through the Government Publishing Office.*$', '', block, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up multiple blank lines and strip
        block = re.sub(r'\n\s*\n\s*\n+', '\n\n', block)
        block = block.strip()
        
        if block:
            bodies.append(block)

    if bodies:
        return "\n\n".join(bodies)
    else:
        # Fallback: return original if no <pre> blocks found
        return text


def remove_classification_words(text: str, country: str) -> str:
    """
    Apply all cleaning steps that remove trivial US/UK classification signals.

    - Always remove US/UK country tokens.
    - For US files, extract <pre> blocks and remove the 3 header lines.
    """
    cleaned = remove_country_tokens(text)

    if country and country.lower() == "us":
        cleaned = _extract_us_bodies_from_pre_blocks(cleaned)

    return cleaned