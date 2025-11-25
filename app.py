from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import pandas as pd
from chatui import llm_chat
from chatpdf import add_paper_to_kb, clear_kb
from pdf2pdf import extract_text, generate_embeddings, query_pinecone, prompt_to_query
from citations import citation_count_for_year


def search_papers():
    with st.form(key='search_form'):
        search_query = st.text_input(
            "Enter search terms or a prompt to find research papers:")
        search_button = st.form_submit_button(label='Search')
        if search_button:
            with st.spinner('Searching for relevant research papers...'):
                search_query = prompt_to_query(search_query)
                embeddings = generate_embeddings(search_query)
                query_results = query_pinecone(embeddings)
                query_matches = query_results[0]["matches"]

                similar_papers = {"DOI": [], "Title": [], "Date": []}
                for match in query_matches:
                    similar_papers["DOI"].append(match["metadata"]["doi"])
                    similar_papers["Title"].append(match["metadata"]["title"])
                    similar_papers["Date"].append(
                        match["metadata"]["latest_creation_date"])
                    similar_papers = pd.DataFrame(similar_papers)
                    similar_papers_sorted = similar_papers.sort_values(
                        by="Date", ascending=False
                    )

                    # Simple "citation count" = how many similar papers we found
                    citation_style_count = len(similar_papers_sorted)
                    st.markdown(
                        f"**Similar papers found (citation-style count):** {citation_style_count}"
                    )

                    st.dataframe(similar_papers_sorted)

                    # Extra: per-year citation count for a selected DOI
                    with st.expander("Get citation count by year for a paper"):
                        target_year = st.number_input(
                            "Year",
                            min_value=1900,
                            max_value=2100,
                            value=2020,
                            step=1,
                        )

                        doi_options = similar_papers_sorted["DOI"].tolist()
                        selected_doi = st.selectbox(
                            "Select a DOI from the results",
                            doi_options,
                        )

                        use_crossref = st.checkbox(
                            "Use Crossref for publication year (slower, more complete)",
                            value=False,
                        )

                        if st.button("Fetch citation count for this DOI and year"):
                            with st.spinner("Looking up citations via OpenCitations..."):
                                try:
                                    count, citing_dois = citation_count_for_year(
                                        selected_doi,
                                        int(target_year),
                                        use_crossref=use_crossref,
                                    )
                                    st.success(f"{count} citations found in {int(target_year)}.")

                                    if citing_dois:
                                        st.markdown("**Citing DOIs:**")
                                        for cdoi in citing_dois:
                                            st.write(cdoi)
                                except Exception as e:
                                    st.error(f"Failed to fetch citation data: {e}")



def upload_pdf():
    with st.form(key='upload_form'):
        uploaded_file = st.file_uploader(
            "Upload a PDF file of a Research Paper, to find a Similar Research Paper", type=['pdf'])
        upload_button = st.form_submit_button(label='Upload')
        if upload_button and uploaded_file:
            with st.spinner('Processing your PDF...'):
                # if PDFs folder does not exist, create it
                if not os.path.exists("PDFs"):
                    os.makedirs("PDFs")

                file_path = os.path.join("PDFs", uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                data = extract_text("PDFs/" + uploaded_file.name)
                if len(data) > 5:
                    embeddings = generate_embeddings(data)
                    query_results = query_pinecone(embeddings)
                    query_matches = query_results[0]["matches"]

                    similar_papers = {"DOI": [], "Title": [], "Date": []}
                    for match in query_matches:
                        similar_papers["DOI"].append(match["metadata"]["doi"])
                        similar_papers["Title"].append(
                            match["metadata"]["title"])
                        similar_papers["Date"].append(
                            match["metadata"]["latest_creation_date"])
                        similar_papers = pd.DataFrame(similar_papers)
                        similar_papers_sorted = similar_papers.sort_values(
                            by="Date", ascending=False
                        )

                        citation_style_count = len(similar_papers_sorted)
                        st.markdown(
                            f"**Similar papers found (citation-style count):** {citation_style_count}"
                        )

                        st.dataframe(similar_papers_sorted)

                        with st.expander("Get citation count by year for a paper"):
                            target_year = st.number_input(
                                "Year",
                                min_value=1900,
                                max_value=2100,
                                value=2020,
                                step=1,
                            )

                            doi_options = similar_papers_sorted["DOI"].tolist()
                            selected_doi = st.selectbox(
                                "Select a DOI from the results",
                                doi_options,
                            )

                            use_crossref = st.checkbox(
                                "Use Crossref for publication year (slower, more complete)",
                                value=False,
                            )

                            if st.button("Fetch citation count for this DOI and year"):
                                with st.spinner("Looking up citations via OpenCitations..."):
                                    try:
                                        count, citing_dois = citation_count_for_year(
                                            selected_doi,
                                            int(target_year),
                                            use_crossref=use_crossref,
                                        )
                                        st.success(f"{count} citations found in {int(target_year)}.")

                                        if citing_dois:
                                            st.markdown("**Citing DOIs:**")
                                            for cdoi in citing_dois:
                                                st.write(cdoi)
                                    except Exception as e:
                                        st.error(f"Failed to fetch citation data: {e}")



def update_knowledge_base():
    st.title("Update Knowledge Base")
    arxiv_id = st.text_input("Enter the arXiv ID of the paper to add:")
    submit_button = st.button("Add Paper")
    clear_kb_button = st.button("Clear Knowledge Base")

    if submit_button:
        with st.spinner("Adding paper to the knowledge base..."):
            try:
                add_paper_to_kb(arxiv_id)
                st.success("Paper successfully added to the knowledge base!")
            except Exception as e:
                st.error(f"Failed to add paper: {str(e)}")

    if clear_kb_button:
        with st.spinner("Clearing the knowledge base..."):
            try:
                clear_kb()
                st.success("Knowledge base cleared successfully!")
            except Exception as e:
                st.error(f"Failed to clear the knowledge base: {str(e)}")


def main():
    st.set_page_config(page_title="Research Assistant", layout="wide")

    # Sidebar for selecting functionality
    st.sidebar.title("Features")
    app_mode = st.sidebar.selectbox("Choose a feature",
                                    ["Chat with Research", "Update Knowledge Base", "Prompt to Paper", "PDF to Paper", ])

    if app_mode == "Prompt to Paper":
        search_papers()
    elif app_mode == "PDF to Paper":
        upload_pdf()
    elif app_mode == "Chat with Research":
        llm_chat()
    elif app_mode == "Update Knowledge Base":
        update_knowledge_base()


if __name__ == "__main__":
    main()
