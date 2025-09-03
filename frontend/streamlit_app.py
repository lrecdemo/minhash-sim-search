import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from typing import List, Dict
import time
import os
import re

API_HOST = os.getenv('API_HOST', 'localhost')
API_PORT = os.getenv('API_PORT', '8002')
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '300'))

# Page config
st.set_page_config(
    page_title="Text Similarity Clustering",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fixed CSS - proper colors and performance optimizations
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .document-container {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        color: #2c3e50 !important;
        font-family: Georgia, serif;
        line-height: 1.6;
    }
    .preview-container {
        background: #ffffff;
        border: 1px solid #dee2e6;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border-radius: 4px;
        color: #495057 !important;
        font-size: 0.9rem;
        border-left: 3px solid #667eea;
        font-family: Georgia, serif;
        line-height: 1.5;
    }
    .cluster-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'clustered_data' not in st.session_state:
    st.session_state.clustered_data = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'overview'
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None


def call_clustering_api(file_content: bytes, filename: str, threshold: float) -> List[Dict]:
    """Call the FastAPI clustering endpoint."""
    try:
        files = {"file": (filename, file_content, "text/csv")}
        data = {"jaccard_threshold": threshold}
        response = requests.post(
            f"http://{API_HOST}:{API_PORT}/api/cluster",
            files=files,
            data=data,
            timeout=API_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return []


def download_results(results: List[Dict]) -> bytes:
    """Download results as CSV via API."""
    try:
        response = requests.post(
            f"http://{API_HOST}:{API_PORT}/api/download",
            json=results,
            timeout=60
        )
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Download error: {response.status_code}")
            return b""
    except requests.exceptions.RequestException as e:
        st.error(f"Download connection error: {str(e)}")
        return b""


def get_confidence_emoji_and_text(certainty: float) -> str:
    """Get emoji and text for confidence level - NO HTML."""
    if certainty >= 0.8:
        return f"🟢 High Confidence ({certainty:.0%})"
    elif certainty >= 0.6:
        return f"🟡 Medium Confidence ({certainty:.0%})"
    else:
        return f"🔴 Low Confidence ({certainty:.0%})"


def get_confidence_level(certainty: float) -> str:
    """Get confidence level label."""
    if certainty >= 0.8:
        return "High"
    elif certainty >= 0.6:
        return "Medium"
    else:
        return "Low"


def display_document_clean(doc: Dict, search_terms: List[str] = None, show_cluster_info: bool = True,
                           is_preview: bool = False):
    """Display document using only Streamlit components with proper styling."""
    search_terms = search_terms or []

    # Show cluster and confidence info
    if show_cluster_info:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**🗂️ Cluster {doc['cluster_id']}**")
        with col2:
            st.markdown(get_confidence_emoji_and_text(doc['certainty']))
    else:
        st.markdown(get_confidence_emoji_and_text(doc['certainty']))

    # Display text with simple highlighting
    display_text = doc['text']

    # For previews, truncate long texts for performance
    if is_preview and len(display_text) > 300:
        display_text = display_text[:300] + "..."

    if search_terms:
        for term in search_terms:
            if term.strip():
                # Simple bold highlighting - no HTML
                display_text = display_text.replace(term, f"**{term}**")
                display_text = display_text.replace(term.lower(), f"**{term.lower()}**")
                display_text = display_text.replace(term.upper(), f"**{term.upper()}**")

    # Use different container styles for previews vs full documents
    container_class = "preview-container" if is_preview else "document-container"

    st.markdown(f"""
    <div class="{container_class}">
        {display_text}
    </div>
    """, unsafe_allow_html=True)


def display_cluster_overview(df: pd.DataFrame):
    """Display overview of all clusters with sample documents."""
    st.markdown("## 📚 Document Clusters Overview")
    st.markdown("*Each cluster contains documents with similar content. Click on a cluster to explore all documents.*")

    # Quick stats (compact)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documents", f"{len(df):,}")
    with col2:
        st.metric("🗂️ Clusters", df['cluster_id'].nunique())
    with col3:
        avg_confidence = df['certainty'].mean()
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.0%}")

    st.markdown("---")

    # Performance optimization: limit clusters shown at once for large corpora
    clusters = df.groupby('cluster_id')
    total_clusters = len(clusters)

    # Show pagination for large numbers of clusters
    if total_clusters > 20:
        st.info(
            f"📊 Large corpus detected ({total_clusters} clusters). Showing clusters in batches for better performance.")

        # Pagination
        clusters_per_page = 10
        total_pages = (total_clusters - 1) // clusters_per_page + 1

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.selectbox(
                "Choose cluster page:",
                options=range(1, total_pages + 1),
                format_func=lambda
                    x: f"Page {x} (Clusters {(x - 1) * clusters_per_page + 1}-{min(x * clusters_per_page, total_clusters)})"
            )

        # Get clusters for current page
        cluster_ids = sorted(df['cluster_id'].unique())
        start_idx = (current_page - 1) * clusters_per_page
        end_idx = min(start_idx + clusters_per_page, total_clusters)
        page_cluster_ids = cluster_ids[start_idx:end_idx]

        clusters_to_show = [(cid, group) for cid, group in clusters if cid in page_cluster_ids]
    else:
        clusters_to_show = list(clusters)

    # Display clusters
    for cluster_id, group in clusters_to_show:
        # Cluster header with better styling
        st.markdown(f"""
        <div class="cluster-header">
            <h3 style="margin: 0; color: white;">🗂️ Cluster {cluster_id}</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; color: white;">
                {len(group)} documents • Average confidence: {group['certainty'].mean():.0%}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Show top 3 most confident documents as preview (performance: limit to 3)
        top_docs = group.nlargest(min(3, len(group)), 'certainty')

        col1, col2 = st.columns([3, 1])

        with col1:
            for idx, (_, doc) in enumerate(top_docs.iterrows()):
                st.markdown(f"**Preview {idx + 1}:**")
                # Use preview mode for performance and better styling
                display_document_clean(doc.to_dict(), show_cluster_info=False, is_preview=True)

            if len(group) > 3:
                st.markdown(f"*... and {len(group) - 3} more documents*")

        with col2:
            st.markdown("")  # spacing
            if st.button(f"📖 View All {len(group)} Documents", key=f"view_cluster_{cluster_id}"):
                st.session_state.view_mode = 'cluster'
                st.session_state.selected_cluster = cluster_id
                st.rerun()

        st.markdown("---")


def display_cluster_details(df: pd.DataFrame, cluster_id: int):
    """Display all documents in a specific cluster."""
    cluster_docs = df[df['cluster_id'] == cluster_id].copy()

    # Header with back button
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"# 📚 Cluster {cluster_id} - All Documents")
        st.markdown(f"**{len(cluster_docs)} documents • Average confidence: {cluster_docs['certainty'].mean():.0%}**")

    with col2:
        if st.button("← Back to Overview", type="secondary"):
            st.session_state.view_mode = 'overview'
            st.rerun()

    # Performance warning for very large clusters
    if len(cluster_docs) > 100:
        st.warning(
            f"⚡ Large cluster detected ({len(cluster_docs)} documents). Consider using search to find specific documents.")

    # Search within cluster
    search_col1, search_col2 = st.columns([3, 1])

    with search_col1:
        search_query = st.text_input(
            "🔍 Search within this cluster:",
            placeholder="Enter words or phrases to find specific documents...",
            key=f"search_cluster_{cluster_id}"
        )

    with search_col2:
        sort_option = st.selectbox(
            "Sort by:",
            options=["confidence", "original", "length"],
            format_func=lambda x: {"confidence": "Confidence", "original": "Original Order", "length": "Text Length"}[
                x],
            key=f"sort_cluster_{cluster_id}"
        )

    # Filter and sort documents
    filtered_docs = cluster_docs.copy()
    search_terms = []

    if search_query.strip():
        search_terms = [term.strip() for term in search_query.split() if term.strip()]
        mask = cluster_docs['text'].str.contains(search_query.strip(), case=False, na=False)
        filtered_docs = cluster_docs[mask]

        if len(filtered_docs) == 0:
            st.warning(f"No documents found containing '{search_query}' in this cluster.")
            filtered_docs = cluster_docs
        else:
            st.success(f"Found {len(filtered_docs)} documents matching '{search_query}'")

    # Sort documents
    if sort_option == "confidence":
        filtered_docs = filtered_docs.sort_values('certainty', ascending=False)
    elif sort_option == "original":
        filtered_docs = filtered_docs.sort_values('original_index')
    elif sort_option == "length":
        filtered_docs['text_length'] = filtered_docs['text'].str.len()
        filtered_docs = filtered_docs.sort_values('text_length', ascending=False)

    # Performance optimization: pagination for large clusters
    docs_per_page = 50 if len(filtered_docs) > 100 else len(filtered_docs)

    if len(filtered_docs) > 50:
        total_pages = (len(filtered_docs) - 1) // docs_per_page + 1

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.selectbox(
                f"Page (showing {docs_per_page} documents per page):",
                options=range(1, total_pages + 1),
                format_func=lambda
                    x: f"Page {x} (Documents {(x - 1) * docs_per_page + 1}-{min(x * docs_per_page, len(filtered_docs))})",
                key=f"page_selector_{cluster_id}"
            )

        # Get documents for current page
        start_idx = (current_page - 1) * docs_per_page
        end_idx = min(start_idx + docs_per_page, len(filtered_docs))
        page_docs = filtered_docs.iloc[start_idx:end_idx]
    else:
        page_docs = filtered_docs
        current_page = 1

    # Display documents
    st.markdown(f"### 📄 Documents ({len(page_docs)} of {len(filtered_docs)} shown)")

    for i, (_, doc) in enumerate(page_docs.iterrows()):
        # For large lists, don't expand all by default (performance)
        expand_default = (i < 5) and (len(page_docs) <= 20)

        with st.expander(f"Document #{doc['original_index']} - {get_confidence_level(doc['certainty'])} Confidence",
                         expanded=expand_default):
            display_document_clean(doc.to_dict(), search_terms, show_cluster_info=False)

            # Additional metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Original Position: #{doc['original_index']}")
            with col2:
                st.caption(f"Confidence: {doc['certainty']:.1%}")
            with col3:
                st.caption(f"Text Length: {len(doc['text'])} chars")


def display_global_search(df: pd.DataFrame):
    """Display global search across all documents."""
    st.markdown("## 🔍 Search All Documents")

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
        # Search across all documents
        search_terms = [term.strip() for term in global_search.split() if term.strip()]
        mask = df['text'].str.contains(global_search.strip(), case=False, na=False)
        confidence_mask = df['certainty'] >= min_confidence

        results = df[mask & confidence_mask].copy()
        results = results.sort_values('certainty', ascending=False)

        if len(results) > 0:
            st.success(
                f"Found {len(results)} documents matching '{global_search}' with {min_confidence:.0%}+ confidence")

            # Group results by cluster
            result_clusters = results.groupby('cluster_id')

            for cluster_id, group in result_clusters:
                st.markdown(f"#### 🗂️ From Cluster {cluster_id} ({len(group)} documents)")

                for idx, (_, doc) in enumerate(group.head(5).iterrows()):  # Show top 5 per cluster
                    st.markdown(f"**Result {idx + 1}:**")
                    display_document_clean(doc.to_dict(), search_terms, is_preview=True)

                if len(group) > 5:
                    if st.button(f"View all {len(group)} results from Cluster {cluster_id}",
                                 key=f"view_search_cluster_{cluster_id}"):
                        st.session_state.view_mode = 'cluster'
                        st.session_state.selected_cluster = cluster_id
                        st.rerun()
        else:
            st.warning(f"No documents found matching '{global_search}' with {min_confidence:.0%}+ confidence")


def main():
    # Cache clear button in sidebar
    with st.sidebar:
        if st.button("🔄 Clear Cache & Refresh"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # Simple header
    st.markdown("# 📚 Text Similarity Analysis")
    st.markdown("*Discover similar documents in your text collection using advanced clustering*")

    # Upload section (always visible at top)
    if st.session_state.clustered_data is None:
        st.markdown("## 📤 Upload Your Documents")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV file with your texts",
                type=['csv'],
                help="Your CSV must have a column named 'text' containing the documents to analyze"
            )

            if uploaded_file is not None:
                # Quick preview
                try:
                    df_preview = pd.read_csv(uploaded_file)

                    # Check for text column
                    text_columns = [col for col in df_preview.columns if col.lower().strip() == 'text']
                    if text_columns:
                        st.success(f"✅ Found {len(df_preview):,} documents in column '{text_columns[0]}'")

                        # Show sample
                        st.markdown("**Sample documents:**")
                        for i in range(min(3, len(df_preview))):
                            st.markdown(f"*{df_preview.iloc[i][text_columns[0]][:150]}...*")

                    else:
                        st.error("❌ No 'text' column found. Please make sure your CSV has a column named 'text'")
                        uploaded_file = None

                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    uploaded_file = None

        with col2:
            st.markdown("**Similarity Settings**")

            similarity_level = st.select_slider(
                "Similarity threshold:",
                options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                value=0.3,
                format_func=lambda x: {
                    0.1: "Very Loose", 0.2: "Loose", 0.3: "Moderate",
                    0.4: "Moderate+", 0.5: "Balanced", 0.6: "Strict",
                    0.7: "Very Strict", 0.8: "Extremely Strict", 0.9: "Nearly Identical"
                }[x]
            )

            st.caption("Higher = stricter similarity requirements")

            if uploaded_file is not None:
                if st.button("🚀 Analyze Documents", type="primary", use_container_width=True):
                    with st.spinner("Analyzing your documents... This may take a few minutes."):
                        progress = st.progress(0)

                        file_content = uploaded_file.read()
                        progress.progress(25)

                        results = call_clustering_api(file_content, uploaded_file.name, similarity_level)
                        progress.progress(75)

                        if results:
                            st.session_state.clustered_data = results
                            st.session_state.view_mode = 'overview'
                            progress.progress(100)
                            time.sleep(0.5)
                            st.rerun()

    else:
        # Main application interface
        df_results = pd.DataFrame(st.session_state.clustered_data)

        # Navigation tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📚 Browse Clusters", "🔍 Search Documents", "📊 Statistics", "💾 Export"])

        with tab1:
            if st.session_state.view_mode == 'overview':
                display_cluster_overview(df_results)
            elif st.session_state.view_mode == 'cluster':
                display_cluster_details(df_results, st.session_state.selected_cluster)

        with tab2:
            display_global_search(df_results)

        with tab3:
            # FIXED: Statistics with proper dropdown
            st.markdown("## 📊 Analysis Statistics")

            col1, col2 = st.columns(2)

            with col1:
                # Basic metrics
                total_docs = len(df_results)
                num_clusters = df_results['cluster_id'].nunique()
                avg_confidence = df_results['certainty'].mean()

                st.markdown("### Overview")
                st.write(f"**Documents analyzed:** {total_docs:,}")
                st.write(f"**Clusters formed:** {num_clusters}")
                st.write(f"**Average confidence:** {avg_confidence:.1%}")

                # FIXED: Cluster sizes with dropdown
                cluster_sizes = df_results['cluster_id'].value_counts().sort_values(ascending=False)

                st.markdown("### Top 5 Largest Clusters")
                for cluster_id, size in cluster_sizes.head(5).items():
                    cluster_confidence = df_results[df_results['cluster_id'] == cluster_id]['certainty'].mean()
                    st.write(f"**Cluster {cluster_id}:** {size} documents ({cluster_confidence:.0%} avg confidence)")

                # FIXED: Dropdown for all clusters
                if len(cluster_sizes) > 5:
                    with st.expander(f"📋 View All {len(cluster_sizes)} Clusters"):
                        st.markdown("**Complete Cluster List:**")

                        # Create a table format
                        cluster_data = []
                        for cluster_id in sorted(cluster_sizes.index):
                            size = cluster_sizes[cluster_id]
                            cluster_docs = df_results[df_results['cluster_id'] == cluster_id]
                            avg_conf = cluster_docs['certainty'].mean()
                            cluster_data.append({
                                'Cluster': f"Cluster {cluster_id}",
                                'Documents': size,
                                'Avg Confidence': f"{avg_conf:.1%}",
                                'Confidence Range': f"{cluster_docs['certainty'].min():.1%} - {cluster_docs['certainty'].max():.1%}"
                            })

                        cluster_df = pd.DataFrame(cluster_data)
                        st.dataframe(cluster_df, hide_index=True, use_container_width=True)

                # Confidence level breakdown
                st.markdown("### Confidence Distribution")
                high_conf = len(df_results[df_results['certainty'] >= 0.8])
                med_conf = len(df_results[(df_results['certainty'] >= 0.6) & (df_results['certainty'] < 0.8)])
                low_conf = len(df_results[df_results['certainty'] < 0.6])

                st.write(f"🟢 **High confidence (≥80%):** {high_conf} documents ({high_conf / total_docs:.1%})")
                st.write(f"🟡 **Medium confidence (60-79%):** {med_conf} documents ({med_conf / total_docs:.1%})")
                st.write(f"🔴 **Low confidence (<60%):** {low_conf} documents ({low_conf / total_docs:.1%})")

            with col2:
                # Confidence distribution chart
                fig = go.Figure(data=[
                    go.Histogram(x=df_results['certainty'], nbinsx=20, marker_color='lightblue')
                ])
                fig.update_layout(
                    title="Confidence Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Number of Documents",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Cluster size distribution
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=[f"C{i}" for i in cluster_sizes.head(10).index],
                        y=cluster_sizes.head(10).values,
                        marker_color='lightcoral'
                    )
                ])
                fig2.update_layout(
                    title="Top 10 Cluster Sizes",
                    xaxis_title="Cluster ID",
                    yaxis_title="Number of Documents",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

        with tab4:
            st.markdown("## 💾 Export Your Results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Download Options")

                if st.button("📥 Download All Results (CSV)", type="primary", use_container_width=True):
                    csv_content = download_results(st.session_state.clustered_data)
                    if csv_content:
                        st.download_button(
                            label="💾 Save CSV File",
                            data=csv_content,
                            file_name="text_clustering_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                # JSON export
                json_data = json.dumps(st.session_state.clustered_data, indent=2)
                st.download_button(
                    label="📄 Download as JSON",
                    data=json_data,
                    file_name="text_clustering_results.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col2:
                st.markdown("### Start New Analysis")
                if st.button("🔄 Analyze Different Documents", use_container_width=True):
                    st.session_state.clustered_data = None
                    st.session_state.view_mode = 'overview'
                    st.rerun()

        # Quick stats always visible at bottom
        with st.container():
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)

            total_docs = len(df_results)
            num_clusters = df_results['cluster_id'].nunique()
            avg_confidence = df_results['certainty'].mean()
            high_confidence = len(df_results[df_results['certainty'] >= 0.8])

            with col1:
                st.metric("📄 Documents", f"{total_docs:,}")
            with col2:
                st.metric("🗂️ Clusters", num_clusters)
            with col3:
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.0%}")
            with col4:
                st.metric("✨ High Confidence", f"{high_confidence} ({high_confidence / total_docs:.0%})")


if __name__ == "__main__":
    main()