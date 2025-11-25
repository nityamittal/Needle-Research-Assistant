# Research Assistant

An LLM powered research assistant

The following is the deployment link: https://project-group-06-genai.streamlit.app/

The instructions below are how to run the project locally

## Usage

1. Install Requirements

```bash
pip install -r requirements.txt
```

2. Set up Environment Variables

```bash
PINECONE_API_KEY=<key>
GEMINI_API_KEY=<your_gemini_api_key>
GEMINI_TEXT_MODEL=gemini-2.0-flash-001     
GEMINI_EMBED_MODEL=gemini-embedding-001   
ABSTRACTS_PINECONE_API_KEY=<key>
```

3. Run the App

```bash
streamlit run app.py
```
