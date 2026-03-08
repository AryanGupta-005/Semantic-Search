"""
preprocessing.py

Cleans raw Usenet messages from the 20 Newsgroups dataset.

Design decisions documented here (as required by assignment):
- Headers removed: routing metadata (From, Path, Newsgroups, Xref, etc.) carry
  zero semantic content. Including them would bias embeddings toward metadata
  tokens rather than actual discussion meaning.
- Newsgroups/Xref fields specifically excluded: these contain the category label
  (e.g. "alt.atheism"). Keeping them would cause data leakage — clusters would
  form around label words, not semantic themes.
- Subject line kept: it is natural language written by the author and often
  summarizes the post content. High signal-to-noise ratio.
- Quoted reply lines removed (lines starting with >): they duplicate content
  from other documents and bias embeddings toward repeated phrases.
- Signatures removed (content after '--'): author metadata, not discussion.
- Stopwords NOT removed: sentence-transformers are trained on natural language
  including function words. Removing stopwords breaks sentence structure and
  degrades embedding quality.
- Stemming NOT applied: the embedding model handles morphological variation
  natively. Stemming would reduce text to incomplete tokens the model may not
  recognize.
- Minimum length threshold of 50 characters: posts shorter than this are
  typically noise (acknowledgements, single-line replies) with no semantic value.
"""

import re
from typing import Optional


# Headers that contain routing/metadata only — no semantic value
_METADATA_HEADERS = {
    "from", "path", "newsgroups", "xref", "organization",
    "nntp-posting-host", "distribution", "reply-to", "lines",
    "message-id", "date", "references", "in-reply-to",
    "mime-version", "content-type", "content-transfer-encoding",
    "x-newsreader", "x-mailer", "return-path", "received"
}

# Minimum cleaned document length to be considered usable
MIN_DOCUMENT_LENGTH = 50


def remove_headers(text: str) -> str:
    """
    Removes email/Usenet header block from the top of the message.

    The header block ends at the first blank line. Everything before
    that blank line that looks like 'Field: value' is stripped.
    The Subject field is an exception — it is extracted and prepended
    to the body because it contains author-written summary content.
    """
    lines = text.split('\n')
    subject = ""
    body_start = 0

    for i, line in enumerate(lines):
        # Blank line marks end of header block
        if line.strip() == "":
            body_start = i + 1
            break

        # Check if this line is a known metadata header to discard
        if ':' in line:
            field = line.split(':', 1)[0].lower().strip()
            if field in _METADATA_HEADERS:
                continue
            # Keep subject line content
            if field == "subject":
                subject = line.split(':', 1)[1].strip() if ':' in line else ""
                # Remove common reply prefixes like Re: Re: Re:
                subject = re.sub(r'^(Re:\s*)+', '', subject, flags=re.IGNORECASE).strip()

    body = '\n'.join(lines[body_start:])

    # Prepend subject to body so it contributes to the embedding
    if subject:
        return subject + '\n' + body
    return body


def remove_quotes(text: str) -> str:
    """
    Removes quoted reply lines (lines starting with '>').

    Quoted lines are copies of other documents already in the corpus.
    Including them inflates similarity between documents that share
    reply chains rather than shared topics.
    """
    lines = text.split('\n')
    filtered = [line for line in lines if not line.strip().startswith('>')]
    return '\n'.join(filtered)


def remove_signatures(text: str) -> str:
    """
    Removes email signatures (content after '-- ' on its own line).

    Signatures contain author names, contact info, and quotes — none
    of which reflects the document's topic.
    """
    # Standard signature delimiter is '-- ' (dash dash space) on its own line
    sig_pattern = re.compile(r'\n--\s*\n', re.MULTILINE)
    match = sig_pattern.search(text)
    if match:
        return text[:match.start()]
    return text


def normalize_text(text: str) -> str:
    """
    Applies light normalization to cleaned text.

    Lowercase: standardizes token form without destroying meaning.
    URL removal: URLs carry no semantic meaning for topic clustering.
    Email removal: same reasoning.
    Punctuation normalization: collapses repeated punctuation.
    Whitespace normalization: removes artifact whitespace from header removal.
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove lines that are purely punctuation or numbers (artifacts)
    lines = text.split('\n')
    lines = [l for l in lines if re.search(r'[a-z]{3,}', l)]

    text = '\n'.join(lines)

    # Collapse multiple whitespace/newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def clean_document(raw_text: str) -> Optional[str]:
    """
    Full cleaning pipeline for a single raw Usenet post.

    Returns None if the document is too short to be useful after cleaning,
    allowing the loader to discard it entirely rather than embed noise.
    """
    text = remove_headers(raw_text)
    text = remove_quotes(text)
    text = remove_signatures(text)
    text = normalize_text(text)

    if len(text) < MIN_DOCUMENT_LENGTH:
        return None

    return text
