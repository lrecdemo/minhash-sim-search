import streamlit as st
import pandas as pd

def get_confidence(certainty: float) -> str:
    return f"Confidence: {certainty:.0%}"

def display_cluster_overview(df: pd.DataFrame):
    cluster_stats = (
        df.groupby('cluster_id')
        .agg(
            size=('cluster_id', 'size'),
            avg_confidence=('certainty', 'mean'),
            text=('text', lambda x: list(x)),
            batches=('batch_id', 'nunique') if 'batch_id' in df.columns else ('cluster_id', lambda x: 1)
        )
        .reset_index()
    )
    cluster_stats['composite_score'] = cluster_stats['size'] * cluster_stats['avg_confidence']
    cluster_stats = cluster_stats.sort_values(by='composite_score', ascending=False)

    with st.expander("Filter & Sort Clusters", expanded=False):
        min_size, max_size = int(cluster_stats['size'].min()), int(cluster_stats['size'].max())
        min_conf, max_conf = float(cluster_stats['avg_confidence'].min()), float(cluster_stats['avg_confidence'].max())
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox(
                "Sort clusters by:",
                options=['composite_score', 'size', 'avg_confidence'],
                index=0,
                format_func=lambda x: {
                    'composite_score': 'Size & Confidence',
                    'size': 'Cluster Size',
                    'avg_confidence': 'Average Confidence'
                }[x],
                key="sort_by"
            )
            sort_order = st.selectbox(
                "Sort order:",
                options=['desc', 'asc'],
                index=0,
                format_func=lambda x: 'Highest First' if x == 'desc' else 'Lowest First',
                key="sort_order"
            )
        with col2:
            selected_min_size = st.number_input(
                "Minimum Cluster Size",
                min_value=min_size,
                max_value=max_size,
                value=min_size,
                step=1,
                key="filter_min_size"
            )
            selected_max_size = st.number_input(
                "Maximum Cluster Size",
                min_value=min_size,
                max_value=max_size,
                value=max_size,
                step=1,
                key="filter_max_size"
            )
        with col3:
            selected_min_conf = st.slider(
                "Minimum Avg Confidence",
                min_value=0.0,
                max_value=1.0,
                value=min_conf,
                step=0.01,
                key="filter_min_conf"
            )
            selected_max_conf = st.slider(
                "Maximum Avg Confidence",
                min_value=0.0,
                max_value=1.0,
                value=max_conf,
                step=0.01,
                key="filter_max_conf"
            )

    filtered_clusters = cluster_stats[
        (cluster_stats['size'] >= selected_min_size) &
        (cluster_stats['size'] <= selected_max_size) &
        (cluster_stats['avg_confidence'] >= selected_min_conf) &
        (cluster_stats['avg_confidence'] <= selected_max_conf)
    ]
    ascending_order = sort_order == 'asc'
    if sort_by == 'composite_score':
        filtered_clusters = filtered_clusters.sort_values('composite_score', ascending=ascending_order)
    elif sort_by == 'size':
        filtered_clusters = filtered_clusters.sort_values(['size', 'avg_confidence'], ascending=[ascending_order, False])
    else:
        filtered_clusters = filtered_clusters.sort_values(['avg_confidence', 'size'], ascending=[ascending_order, False])

    total_clusters = len(filtered_clusters)
    if total_clusters == 0:
        st.warning("No clusters match the filter criteria.")
        return
    st.info(
        f"Showing {total_clusters} clusters, sorted by {sort_by.replace('_', ' ')} ({'highest' if not ascending_order else 'lowest'} first)"
    )
    clusters_per_page = 10
    total_pages = (total_clusters - 1) // clusters_per_page + 1
    current_page = st.selectbox(
        "Navigate pages:",
        options=range(1, total_pages + 1),
        format_func=lambda
            x: f"Page {x} (Clusters {(x - 1) * clusters_per_page + 1}-{min(x * clusters_per_page, total_clusters)})"
    )
    start_idx = (current_page - 1) * clusters_per_page
    end_idx = min(start_idx + clusters_per_page, total_clusters)
    clusters_to_show = filtered_clusters.iloc[start_idx:end_idx]

    for _, row in clusters_to_show.iterrows():
        cluster_id = row['cluster_id']
        cluster_size = row['size']
        avg_confidence = row['avg_confidence']
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div class="cluster-header">
                <h3 style="margin: 0; color: white;">Cluster {cluster_id}</h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9; color: white;">
                    {cluster_size} documents • {avg_confidence:.0%} avg confidence
                </p>
            </div>
            """, unsafe_allow_html=True)

            cluster_docs = df[df['cluster_id'] == cluster_id]
            top_docs = cluster_docs.nlargest(3, 'certainty')
            for idx, (_, doc) in enumerate(top_docs.iterrows()):
                if hasattr(doc, 'to_dict'):
                    doc_dict = doc.to_dict()
                else:
                    doc_dict = doc

                doc_id = doc_dict.get('id', doc_dict['original_index']) if 'id' in doc_dict else doc_dict[
                    'original_index']
                text_preview = str(doc_dict.get('text', ''))[:100]
                certainty = doc_dict.get('certainty', 0)

                st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <strong>Preview {idx + 1}:</strong> {doc_id}: {text_preview}... <span style="color: #667eea;">{get_confidence(certainty)}</span>
                </div>
                """, unsafe_allow_html=True)

            if st.button(f"View All {cluster_size} Documents", key=f"view_cluster_{cluster_id}"):
                st.session_state.view_mode = 'cluster'
                st.session_state.selected_cluster = cluster_id
                st.rerun()

            st.markdown("---")