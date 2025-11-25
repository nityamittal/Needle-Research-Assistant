import os
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

gen_model = GenerativeModel("gemini-2.0-flash-001")
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

def main():
    resp = gen_model.generate_content("Say hi in one short sentence.")
    print("LLM response:", resp.text)

    embeds = embed_model.get_embeddings(["hello world"])
    print("Embedding length:", len(embeds[0].values))

if __name__ == "__main__":
    main()
