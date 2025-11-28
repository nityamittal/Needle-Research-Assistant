"""
Filtering module for arXiv metadata.

Provides utilities to filter search results by:
- arXiv categories (e.g., cs.AI, cs.LG, stat.ML)
- publication year range
- keywords in title or abstract
- author name substring matching
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


# Common arXiv categories (subset of CS + stats). Extend as needed.
COMMON_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision",
    "cs.NLP": "Natural Language Processing",
    "cs.CL": "Computation and Language",
    "cs.IR": "Information Retrieval",
    "stat.ML": "Machine Learning (Statistics)",
    "cs.DB": "Databases",
    "cs.SE": "Software Engineering",
    "cs.PL": "Programming Languages",
    "cs.AR": "Architecture",
    "cs.DC": "Distributed Computing",
    "math.LO": "Logic",
    "math.ST": "Statistics Theory",
}


class FilterConfig:
    """Configuration for filtering arXiv search results."""

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        title_keywords: Optional[List[str]] = None,
        abstract_keywords: Optional[List[str]] = None,
        author_name: Optional[str] = None,
    ):
        """
        Initialize filter configuration.

        Args:
            categories: list of arXiv category codes (e.g., ["cs.LG", "stat.ML"])
            year_min: earliest publication year (inclusive)
            year_max: latest publication year (inclusive)
            title_keywords: keywords that must appear in title (AND logic)
            abstract_keywords: keywords that must appear in abstract (AND logic)
            author_name: substring to match in author names (case-insensitive)
        """
        self.categories = categories or []
        self.year_min = year_min
        self.year_max = year_max
        self.title_keywords = [kw.lower() for kw in (title_keywords or [])]
        self.abstract_keywords = [kw.lower() for kw in (abstract_keywords or [])]
        self.author_name = (author_name or "").lower()

    def matches(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if a single result's metadata matches all active filters.

        Args:
            metadata: metadata dict from vector search result
                      (expected keys: categories, latest_creation_date, title, abstract, authors)

        Returns:
            True if the result passes all filters; False otherwise.
        """
        # Category filter
        if self.categories:
            meta_cats = (metadata.get("categories") or "").split()
            if not any(cat in meta_cats for cat in self.categories):
                return False

        # Year filter
        if self.year_min or self.year_max:
            date_str = metadata.get("latest_creation_date") or ""
            if date_str:
                try:
                    year = int(date_str[:4])
                except (ValueError, IndexError):
                    year = None

                if year is None:
                    return False

                if self.year_min and year < self.year_min:
                    return False
                if self.year_max and year > self.year_max:
                    return False

        # Title keywords filter (all must match)
        if self.title_keywords:
            title_lower = (metadata.get("title") or "").lower()
            if not all(kw in title_lower for kw in self.title_keywords):
                return False

        # Abstract keywords filter (all must match)
        if self.abstract_keywords:
            abstract_lower = (metadata.get("abstract") or "").lower()
            if not all(kw in abstract_lower for kw in self.abstract_keywords):
                return False

        # Author name filter
        if self.author_name:
            authors_lower = (metadata.get("authors") or "").lower()
            if self.author_name not in authors_lower:
                return False

        return True

    def apply_to_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of results by applying all active filters.

        Args:
            results: list of result dicts from vector search
                     (each with "metadata" key)

        Returns:
            list of results that pass all filters, in original order.
        """
        filtered = []
        for result in results:
            metadata = result.get("metadata") or {}
            if self.matches(metadata):
                filtered.append(result)
        return filtered

    def is_empty(self) -> bool:
        """Check if any filters are active."""
        return (
            not self.categories
            and self.year_min is None
            and self.year_max is None
            and not self.title_keywords
            and not self.abstract_keywords
            and not self.author_name
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return filter config as a dict (for serialization or logging)."""
        return {
            "categories": self.categories,
            "year_min": self.year_min,
            "year_max": self.year_max,
            "title_keywords": self.title_keywords,
            "abstract_keywords": self.abstract_keywords,
            "author_name": self.author_name,
        }
