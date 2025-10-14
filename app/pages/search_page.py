import streamlit as st
import pandas as pd
from pages.detail_page import display_cluster_details


def display_global_search(df: pd.DataFrame):
    if st.session_state.get('search_viewing_cluster', False):
        cluster_id = st.session_state.get('search_selected_cluster')

        if st.button("← Back to Search Results", key="back_to_search_from_cluster"):
            st.session_state.search_viewing_cluster = False
            st.session_state.search_selected_cluster = None
            st.rerun()

        st.session_state.in_search_context = True
        display_cluster_details(df, cluster_id)
        st.session_state.in_search_context = False
        return

    st.markdown("## Search All Documents")

    if 'saved_search_query' not in st.session_state:
        st.session_state.saved_search_query = ""
    if 'saved_min_confidence' not in st.session_state:
        st.session_state.saved_min_confidence = 0.0

    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        global_search = st.text_input(
            "Search across all documents:",
            placeholder="Enter keywords to find documents across all clusters...",
            value=st.session_state.saved_search_query,
            key="global_search"
        )
    with search_col2:
        confidence_options = [0.0, 0.3, 0.6, 0.8]
        saved_index = confidence_options.index(
            st.session_state.saved_min_confidence) if st.session_state.saved_min_confidence in confidence_options else 0

        min_confidence = st.selectbox(
            "Minimum confidence:",
            options=confidence_options,
            format_func=lambda x: f"{x:.0%}+",
            index=saved_index
        )

    st.session_state.saved_search_query = global_search
    st.session_state.saved_min_confidence = min_confidence
    if global_search.strip():
        search_terms = [term.strip() for term in global_search.split() if term.strip()]
        mask = df['text'].str.contains(global_search.strip(), case=False, na=False)
        confidence_mask = df['certainty'] >= min_confidence
        results = df[mask & confidence_mask].copy()
        results = results.sort_values('certainty', ascending=False)
        if len(results) > 0:
            st.success(
                f"Found {len(results)} documents matching '{global_search}' with {min_confidence:.0%}+ confidence")
            result_clusters = results.groupby('cluster_id')
            for cluster_id, group in result_clusters:
                st.markdown(f"#### From Cluster {cluster_id} ({len(group)} documents)")
                for idx, (_, doc) in enumerate(group.head(5).iterrows()):
                    doc_id = doc.get('id', doc['original_index']) if 'id' in doc else doc['original_index']
                    st.markdown(f"**Result {idx + 1}:**")
                    st.markdown(f"**{doc_id}:** {doc['text']}")
                if len(group) > 5:
                    button_key = f"search_goto_cluster_{cluster_id}_{abs(hash(global_search)) % 10000}"
                    if st.button(
                            f"View all {len(group)} results from Cluster {cluster_id}",
                            key=button_key
                    ):
                        # Set flags to show cluster details in this tab
                        st.session_state.search_viewing_cluster = True
                        st.session_state.search_selected_cluster = cluster_id
                        st.rerun()

        else:
            st.warning(f"No documents found matching '{global_search}' with {min_confidence:.0%}+ confidence")