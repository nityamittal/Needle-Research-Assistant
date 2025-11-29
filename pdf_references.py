"""
Helpers for extracting cited identifiers from a PDF and annotating search results.

Intended usage:
    refs = extract_references_from_pdf("/path/to/upload.pdf")
    annotated = annotate_results(search_results, refs)
"""

import re
import string
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Set

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - descriptive guard
    raise ImportError(
        "PyMuPDF is required. Install with 'pip install PyMuPDF' (and uninstall the stray 'fitz' package if present)."
    ) from exc

# If the wrong 'fitz' package is installed, it won't have fitz.open.
if not hasattr(fitz, "open"):  # pragma: no cover - descriptive guard
    raise ImportError(
        "It looks like the wrong 'fitz' package is installed. "
        "Uninstall it ('pip uninstall fitz') and install PyMuPDF ('pip install PyMuPDF')."
    )


ReferenceMap = Dict[str, Set[str]]

_DOI_PATTERN = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_ARXIV_PATTERN = re.compile(r"arxiv:\s*\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


def _extract_text(pdf_path: str) -> str:
    """Extract raw text from a PDF using PyMuPDF."""
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def _normalize_identifier(value: str) -> str:
    """Trim punctuation and lowercase for loose matching."""
    if not value:
        return ""
    return value.strip().strip(string.punctuation).lower()


def _normalize_for_title_match(value: str) -> str:
    """Looser normalization for title substring matching."""
    if not value:
        return ""
    # Remove common punctuation, collapse whitespace, lowercase.
    cleaned = re.sub(r"[\r\n]+", " ", value)
    cleaned = re.sub(r"-\s+", "", cleaned)  # join hyphenated line breaks
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _find_dois(text: str) -> Set[str]:
    # Basic DOI pattern: 10.<4-9 digits>/<suffix>
    return {_normalize_identifier(m.group()) for m in _DOI_PATTERN.finditer(text)}


def _find_arxiv_ids(text: str) -> Set[str]:
    # Matches arXiv:YYYY.NNNNN[vN]
    results = set()
    for m in _ARXIV_PATTERN.finditer(text):
        val = m.group()
        val = re.sub(r"arxiv:\s*", "", val, flags=re.IGNORECASE)
        results.add(_normalize_identifier(val))
    return results


def _find_urls(text: str) -> Set[str]:
    return {_normalize_identifier(m.group()) for m in _URL_PATTERN.finditer(text)}


def extract_references_from_text(text: str) -> ReferenceMap:
    """
    Extract referenced identifiers (DOIs, arXiv IDs, URLs) from raw text.

    Returns: {"doi": set[str], "arxiv": set[str], "url": set[str]}
    """
    if not text:
        return {"doi": set(), "arxiv": set(), "url": set()}

    return {
        "doi": _find_dois(text),
        "arxiv": _find_arxiv_ids(text),
        "url": _find_urls(text),
    }


def extract_references_from_pdf(pdf_path: str) -> ReferenceMap:
    """
    Extract a best-effort set of referenced identifiers (DOIs, arXiv IDs, URLs).

    Returns: {"doi": set[str], "arxiv": set[str], "url": set[str]}
    """
    raw_text = _extract_text(pdf_path)
    return extract_references_from_text(raw_text)


def _normalize_meta_field(meta: Mapping[str, Any], *keys: str) -> str:
    """Pull the first present metadata key and normalize it for comparison."""
    for key in keys:
        if key in meta and meta[key]:
            return _normalize_identifier(str(meta[key]))
    return ""


def annotate_results(results: Iterable[Mapping[str, Any]], references: ReferenceMap) -> List[MutableMapping[str, Any]]:
    """
    Mark search results that were explicitly referenced in the source PDF.

    results: iterable of search result dicts containing optional metadata keys:
             "doi", "arxiv_id" (or "arxiv"), or "link"/"url".
    references: output of extract_references_from_pdf

    Returns: new list with boolean flag "linked_in_pdf" added.
    """
    references = references or {}
    ref_doi = references.get("doi", set())
    ref_arxiv = references.get("arxiv", set())
    ref_url = references.get("url", set())

    annotated = []
    for item in results:
        meta_raw = item.get("metadata", {}) if isinstance(item, Mapping) else {}
        meta: Mapping[str, Any] = meta_raw if isinstance(meta_raw, Mapping) else {}

        doi = _normalize_meta_field(meta, "doi")
        arxiv_id = _normalize_meta_field(meta, "arxiv_id", "arxiv")
        link = _normalize_meta_field(meta, "link", "url")

        linked = False
        if doi and doi in ref_doi:
            linked = True
        elif arxiv_id and arxiv_id in ref_arxiv:
            linked = True
        elif link and (link in ref_url or any(link.endswith(u) for u in ref_url)):
            linked = True

        new_item = dict(item) if isinstance(item, dict) else {"value": item}
        new_item["linked_in_pdf"] = linked
        annotated.append(new_item)

    return annotated


def test_pdf_references(pdf_path: str, paper_titles: List[str]) -> List[Dict]:
    """
    Lightweight tester: give it a PDF path and a list of paper titles; it returns a list
    with a boolean flag showing whether each title looks referenced.

    Matching heuristic:
    - Identifier-based: uses annotate_results with extracted DOIs/arXiv IDs/URLs.
    - Title substring: case-insensitive substring search in the PDF text.
    """
    raw_text = _extract_text(pdf_path)
    refs = extract_references_from_text(raw_text)
    normalized_text = _normalize_for_title_match(raw_text)

    # Build dummy result objects so annotate_results can try identifier matches.
    dummy_results = [
        {"id": f"test-{i}", "metadata": {"title": title, "doi": "", "arxiv_id": "", "link": ""}}
        for i, title in enumerate(paper_titles)
    ]
    annotated = annotate_results(dummy_results, refs)

    labeled = []
    for item in annotated:
        title = item.get("metadata", {}).get("title", "")
        title_norm = _normalize_for_title_match(title)
        title_hit = title_norm and title_norm in normalized_text
        labeled.append(
            {
                "title": title,
                "linked_in_pdf": bool(item.get("linked_in_pdf")) or bool(title_hit),
                "matched_by_identifier": bool(item.get("linked_in_pdf")),
                "matched_by_title_substring": bool(title_hit),
            }
        )
    return labeled
