# vertex_vs_client.py
import os
from typing import List, Dict, Any

from google.cloud import aiplatform

OFFLINE_MODE = os.getenv("OFFLINE_MODE", "0") == "1"

if not OFFLINE_MODE:
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not PROJECT_ID:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT")

    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    PAPERS_ENDPOINT_NAME = os.getenv("VS_PAPERS_ENDPOINT_NAME")  # full resource name
    PAPERS_DEPLOYED_INDEX_ID = os.getenv("VS_PAPERS_DEPLOYED_INDEX_ID")

    KB_ENDPOINT_NAME = os.getenv("VS_KB_ENDPOINT_NAME")
    KB_DEPLOYED_INDEX_ID = os.getenv("VS_KB_DEPLOYED_INDEX_ID")

    if not all(
        [PAPERS_ENDPOINT_NAME, PAPERS_DEPLOYED_INDEX_ID, KB_ENDPOINT_NAME, KB_DEPLOYED_INDEX_ID]
    ):
        raise RuntimeError("Vertex Vector Search env vars not set correctly")

    papers_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=PAPERS_ENDPOINT_NAME
    )
    kb_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=KB_ENDPOINT_NAME
    )
else:
    PAPERS_ENDPOINT_NAME = None
    PAPERS_DEPLOYED_INDEX_ID = None
    KB_ENDPOINT_NAME = None
    KB_DEPLOYED_INDEX_ID = None
    papers_endpoint = None
    kb_endpoint = None


def query_papers(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Query vs-papers-index with a single embedding."""
    if OFFLINE_MODE:
        return []

    resp = papers_endpoint.find_neighbors(
        deployed_index_id=PAPERS_DEPLOYED_INDEX_ID,
        queries=[embedding],
        num_neighbors=top_k,
        return_full_datapoint=True,  # needed for metadata
    )

    neighbors = resp[0]
    results = []
    for n in neighbors:
        # structure depends on SDK version, but you get id, distance, and datapoint metadata
        meta = getattr(n, "datapoint", None)
        metadata = getattr(meta, "embedding_metadata", {}) if meta else {}
        results.append(
            {
                "id": n.id,
                "score": n.distance,
                "metadata": metadata,
            }
        )
    return results


def query_kb(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Query vs-kb-index with a single embedding."""
    if OFFLINE_MODE:
        return []

    resp = kb_endpoint.find_neighbors(
        deployed_index_id=KB_DEPLOYED_INDEX_ID,
        queries=[embedding],
        num_neighbors=top_k,
        return_full_datapoint=True,
    )

    neighbors = resp[0]
    results = []
    for n in neighbors:
        meta = getattr(n, "datapoint", None)
        metadata = getattr(meta, "embedding_metadata", {}) if meta else {}
        results.append(
            {
                "id": n.id,
                "score": n.distance,
                "metadata": metadata,
            }
        )
    return results
