
import streamlit as st
import pandas as pd


def display_global_search(df: pd.DataFrame):
    st.markdown("## Search All Documents")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        global_search = st.text_input(
            "Search across all documents:",
            placeholder="Enter keywords to find documents across all clusters...",
            key="global_search"
        )
    with search_col2:
        min_confidence = st.selectbox(
            "Minimum confidence:",
            options=[0.0, 0.3, 0.6, 0.8],
            format_func=lambda x: f"{x:.0%}+",
            index=0
        )
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
                    if st.button(f"View all {len(group)} results from Cluster {cluster_id}",
                                 key=f"view_search_cluster_{cluster_id}"):
                        st.session_state.view_mode = 'cluster'
                        st.session_state.selected_cluster = cluster_id
                        st.rerun()
        else:
            st.warning(f"No documents found matching '{global_search}' with {min_confidence:.0%}+ confidence")