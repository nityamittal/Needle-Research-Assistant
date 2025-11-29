# vs_upsert.py

from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, Dict, Any

from google.cloud import aiplatform_v1

# --- Config ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or "585221635563"
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
INDEX_ID_PAPERS = os.getenv("VS_PAPERS_INDEX_ID")
INDEX_ID_KB = os.getenv("VS_KB_INDEX_ID")

if not all([INDEX_ID_PAPERS, INDEX_ID_KB]):
    raise RuntimeError("VS_PAPERS_INDEX_ID and VS_KB_INDEX_ID must be set")

# Full index resource names
PARENT_PAPERS = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexes/{INDEX_ID_PAPERS}"
PARENT_KB = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexes/{INDEX_ID_KB}"

# Use *regional* endpoint for Vertex Vector Search
client_options = {
    "api_endpoint": f"{LOCATION}-aiplatform.googleapis.com",
}

client = aiplatform_v1.IndexServiceClient(client_options=client_options)

print(f"[vs_upsert] PROJECT_ID = {PROJECT_ID}")
print(f"[vs_upsert] LOCATION   = {LOCATION}")
print(f"[vs_upsert] PARENT_PAPERS = {PARENT_PAPERS}")
print(f"[vs_upsert] PARENT_KB     = {PARENT_KB}")

# Optional sanity: check the index exists + state
try:
    idx = client.get_index(name=PARENT_PAPERS)
    print(
        "[vs_upsert] Papers index state=",
        getattr(idx, "state", None),
        "update_method=",
        getattr(idx, "index_update_method", None),
    )
except Exception as e:
    print(f"[vs_upsert] get_index FAILED for papers index: {e}")


def _to_float_list(vec) -> list[float]:
    if hasattr(vec, "tolist"):
        return [float(x) for x in vec.tolist()]
    return [float(x) for x in vec]


def upsert_datapoints(index_parent: str, items: List[Dict[str, Any]]) -> None:
    """
    items: [{"id": str, "vector": list-like embedding}, ...]
    """
    datapoints = []
    for item in items:
        vid = str(item["id"])
        emb = _to_float_list(item["vector"])
        dp = aiplatform_v1.IndexDatapoint(
            datapoint_id=vid,
            feature_vector=emb,
        )
        datapoints.append(dp)

    if not datapoints:
        return

    req = aiplatform_v1.UpsertDatapointsRequest(
        index=index_parent,
        datapoints=datapoints,
    )
    print(f"[vs_upsert] Upserting {len(datapoints)} datapoints into {index_parent}")
    client.upsert_datapoints(request=req)


def upsert_papers(items: List[Dict[str, Any]]) -> None:
    upsert_datapoints(PARENT_PAPERS, items)


def upsert_kb(items: List[Dict[str, Any]]) -> None:
    upsert_datapoints(PARENT_KB, items)
