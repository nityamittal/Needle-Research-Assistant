# vs_upsert.py
import os
from typing import List, Dict, Any

from google.cloud import aiplatform_v1

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
INDEX_ID_PAPERS = os.getenv("VS_PAPERS_INDEX_ID")
INDEX_ID_KB = os.getenv("VS_KB_INDEX_ID")

if not all([PROJECT_ID, LOCATION, INDEX_ID_PAPERS, INDEX_ID_KB]):
    raise RuntimeError("VS_* index env vars missing")

client = aiplatform_v1.IndexServiceClient()

PARENT_PAPERS = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexes/{INDEX_ID_PAPERS}"
PARENT_KB = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexes/{INDEX_ID_KB}"


def upsert_datapoints(index_parent: str, items: List[Dict[str, Any]]) -> None:
    """Upsert a batch of datapoints into a Vertex Vector Search index.

    Each item in `items` must be:
      {
        "id": <str>,
        "vector": <list/np.ndarray>,
        "metadata": <dict>,   # currently ignored at index level
      }
    """
    datapoints: List[aiplatform_v1.IndexDatapoint] = []

    for item in items:
        vec = item["vector"]

        # ensure plain list[float]
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        else:
            vec = list(vec)

        dp = aiplatform_v1.IndexDatapoint(
            datapoint_id=str(item["id"]),
            feature_vector=[float(x) for x in vec],
            # DO NOT ADD embedding_metadata HERE – this client version does not support it
        )
        datapoints.append(dp)

    req = aiplatform_v1.UpsertDatapointsRequest(
        index=index_parent,
        datapoints=datapoints,
    )
    client.upsert_datapoints(request=req)


def upsert_papers(items: List[Dict[str, Any]]) -> None:
    upsert_datapoints(PARENT_PAPERS, items)


def upsert_kb(items: List[Dict[str, Any]]) -> None:
    upsert_datapoints(PARENT_KB, items)
