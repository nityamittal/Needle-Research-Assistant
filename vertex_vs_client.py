import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from google.cloud import aiplatform

from metadata_store import get_papers_metadata, get_kb_chunks_metadata

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

if not PROJECT_ID:
    raise RuntimeError("Set GOOGLE_CLOUD_PROJECT")

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Vertex Vector Search endpoints
VS_PAPERS_ENDPOINT_NAME = os.getenv("VS_PAPERS_ENDPOINT_NAME")
VS_PAPERS_DEPLOYED_INDEX_ID = os.getenv("VS_PAPERS_DEPLOYED_INDEX_ID")

VS_KB_ENDPOINT_NAME = os.getenv("VS_KB_ENDPOINT_NAME")
VS_KB_DEPLOYED_INDEX_ID = os.getenv("VS_KB_DEPLOYED_INDEX_ID")

if not all([VS_PAPERS_ENDPOINT_NAME, VS_PAPERS_DEPLOYED_INDEX_ID]):
    raise RuntimeError("VS_PAPERS_* endpoint env vars missing")

_papers_endpoint = aiplatform.MatchingEngineIndexEndpoint(
    index_endpoint_name=VS_PAPERS_ENDPOINT_NAME
)

_kb_endpoint = None
if VS_KB_ENDPOINT_NAME and VS_KB_DEPLOYED_INDEX_ID:
    _kb_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=VS_KB_ENDPOINT_NAME
    )


def _match(
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
    deployed_index_id: str,
    query_vector: List[float],
    top_k: int,
):
    """
    Use find_neighbors (managed Vertex API) instead of low-level .match()
    so we don't hit the ':10000' gRPC private endpoint nonsense.
    """
    resp = endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_vector],
        num_neighbors=top_k,
        return_full_datapoint=False,  # we get metadata from Firestore, not Vertex
    )
    return resp[0] if resp else []


def query_papers(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the 'papers' index and hydrate metadata from Firestore.
    Returns: [{id, score, metadata}, ...]
    """
    neighbors = _match(_papers_endpoint, VS_PAPERS_DEPLOYED_INDEX_ID, query_vector, top_k)
    ids = [n.id for n in neighbors]

    meta_by_id = get_papers_metadata(ids)

    results: List[Dict[str, Any]] = []
    for n in neighbors:
        pid = n.id
        results.append(
            {
                "id": pid,
                "score": n.distance,
                "metadata": meta_by_id.get(pid, {}),
            }
        )
    return results


def query_kb(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Same pattern for the Library index (Chat with Research).
    """
    if _kb_endpoint is None:
        return []

    neighbors = _match(_kb_endpoint, VS_KB_DEPLOYED_INDEX_ID, query_vector, top_k)
    ids = [n.id for n in neighbors]

    meta_by_id = get_kb_chunks_metadata(ids)

    results: List[Dict[str, Any]] = []
    for n in neighbors:
        cid = n.id
        results.append(
            {
                "id": cid,
                "score": n.distance,
                "metadata": meta_by_id.get(cid, {}),
            }
        )
    return results
