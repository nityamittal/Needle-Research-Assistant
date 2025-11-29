import os
from typing import List, Optional, Tuple
from urllib.parse import quote

import requests

OPENCITATIONS_BASE = "https://api.opencitations.net/index/v2"
CROSSREF_BASE = "https://api.crossref.org/works/"


def _extract_doi_from_citing_field(citing: str) -> Optional[str]:
    """
    OpenCitations 'citing' looks like:
      "omid:br/06101801781 doi:10.7717/peerj-cs.421 pmid:33817056"

    Grab the first doi:... token and return just the DOI.
    """
    if not citing:
        return None

    idx = citing.find("doi:")
    if idx == -1:
        return None

    doi_part = citing[idx + 4 :]  # after 'doi:'
    end = len(doi_part)
    for ch in (" ", ";"):
        pos = doi_part.find(ch)
        if pos != -1:
            end = min(end, pos)

    doi = doi_part[:end].strip()
    return doi or None


def _fetch_opencitations_citations(
    doi: str,
    oc_token: Optional[str] = None,
) -> List[dict]:
    """
    Hit /citations/{id} on OpenCitations Index v2. :contentReference[oaicite:0]{index=0}

    id must be 'doi:<DOI>'.
    """
    import requests

def _fetch_opencitations_citations(doi: str, oc_token: Optional[str] = None):
    id_param = f"doi:{doi}"
    url = f"{OPENCITATIONS_BASE}/citations/{id_param}"

    headers = {
        # be a good citizen, add a UA with contact
        "User-Agent": f"needle-research-assistant (mailto:{os.getenv('CROSSREF_MAILTO', 'noreply@example.com')})"
    }
    if oc_token:
        # OpenCitations expects this header name
        headers["access-token"] = oc_token

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 403:
            # Graceful degrade – no citations instead of exploding
            return []
        resp.raise_for_status()
    except requests.RequestException as e:
        # Log if you want, but don't kill the app
        print(f"[WARN] OpenCitations request failed: {e}")
        return []

    data = resp.json()
    if not isinstance(data, list):
        print(f"[WARN] Unexpected OpenCitations response: {data!r}")
        return []

    return data



def _get_citation_year_from_opencitations(row: dict) -> Optional[int]:
    """
    Use the 'creation' field from OpenCitations, which is defined as the
    publication date (YYYY-MM-DD) of the *citing* entity. :contentReference[oaicite:1]{index=1}

    Values can be prefixed like "[index] => 2021-03-10" and separated by ';',
    so we strip that politely.
    """
    raw = row.get("creation")
    if not raw:
        return None

    # Take first segment if multiple indexes
    first_segment = raw.split(";")[0].strip()
    if "=>" in first_segment:
        _, first_segment = first_segment.split("=>", 1)
    date_str = first_segment.strip()

    if len(date_str) < 4:
        return None

    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return None


def _get_year_from_crossref(
    doi: str,
    mailto: Optional[str] = None,
) -> Optional[int]:
    """
    Get publication year from Crossref /works/{doi}. :contentReference[oaicite:2]{index=2}
    Tries several date fields in order and returns the first year it finds.
    """
    url = CROSSREF_BASE + quote(doi)
    headers = {}

    # Crossref wants a helpful User-Agent with an email. :contentReference[oaicite:3]{index=3}
    if mailto:
        headers["User-Agent"] = f"research-assistant-citations (mailto:{mailto})"
    else:
        headers["User-Agent"] = "research-assistant-citations"

    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    msg = resp.json().get("message", {})

    for key in ("published-print", "published-online", "published", "issued", "created"):
        date_info = msg.get(key)
        if not date_info:
            continue

        parts = date_info.get("date-parts")
        if not parts or not parts[0]:
            continue

        year = parts[0][0]
        if isinstance(year, int):
            return year
        try:
            return int(year)
        except (TypeError, ValueError):
            continue

    return None


def citation_count_for_year(
    doi: str,
    year: int,
    use_crossref: bool = False,
    oc_token: Optional[str] = None,
    crossref_mailto: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """
    Main entrypoint.

    Returns (count, list_of_citing_dois) for citations TO <doi> in a given year.

    - doi: target DOI (without 'doi:' prefix)
    - year: target year (e.g. 2020)
    - use_crossref:
        False = use OpenCitations 'creation' date only (fast)
        True  = get year from Crossref if possible (slower, more network)
    """
    if oc_token is None:
        oc_token = os.getenv("OPENCITATIONS_TOKEN")

    if crossref_mailto is None:
        crossref_mailto = os.getenv("CROSSREF_MAILTO")

    rows = _fetch_opencitations_citations(doi, oc_token=oc_token)
    matches: List[str] = []

    for row in rows:
        citing_doi = _extract_doi_from_citing_field(row.get("citing", ""))
        if not citing_doi:
            continue

        if use_crossref:
            pub_year = _get_year_from_crossref(citing_doi, mailto=crossref_mailto)
            if pub_year is None:
                pub_year = _get_citation_year_from_opencitations(row)
        else:
            pub_year = _get_citation_year_from_opencitations(row)

        if pub_year == year:
            matches.append(citing_doi)

    return len(matches), matches

def citation_count_all_years(
    doi: str,
    oc_token: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """
    Return (count, list_of_citing_dois) for *all* citations to <doi> across all years.

    - doi: target DOI (without 'doi:' prefix)
    - oc_token: optional OpenCitations token; if omitted, we use OPENCITATIONS_TOKEN env var.
    """
    if oc_token is None:
        oc_token = os.getenv("OPENCITATIONS_TOKEN")

    rows = _fetch_opencitations_citations(doi, oc_token=oc_token)
    matches: List[str] = []

    for row in rows:
        citing_doi = _extract_doi_from_citing_field(row.get("citing", ""))
        if citing_doi:
            matches.append(citing_doi)

    # Deduplicate while preserving order
    unique = list(dict.fromkeys(matches))
    return len(unique), unique

