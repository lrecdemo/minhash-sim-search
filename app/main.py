from collections import defaultdict
import plotly.graph_objects as go
import numpy as np
from minhash_clustering.cluster_in_mem import MemMinhashLSHClustering, ClusteredDocument
from cluster_logging.performancelogger import PerformanceLogger
from pages.overview_page import display_cluster_overview
from minhash_clustering.cluster_streaming import StreamingMinHashLSHClustering
import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple
import json
import time
from pages.search_page import display_global_search
from pages.detail_page import display_cluster_details
st.set_page_config(
    page_title="Text Similarity Clustering",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    .processing-container {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if 'clustered_data' not in st.session_state:
    st.session_state.clustered_data = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'overview'
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def run_clustering_analysis(texts: List[str], threshold: float, shingle_size: int,
                            progress_placeholder, clustering_method: str = "auto",
                            preprocess_options: Dict[str, bool] = None):
    logger = PerformanceLogger()
    logger.start()
    logger.log("Initializing clustering service")

    if clustering_method == "auto":
        if len(texts) > 10_000:
            clustering_method = "streaming"
        else:
            clustering_method = "optimized"

    if clustering_method == "streaming":
        clustering_service = StreamingMinHashLSHClustering(
            threshold=threshold,
            shingle_size=shingle_size,
            chunk_size=min(5000, max(1000, len(texts) // 20)),
            num_perm=64,
            preprocess_options=preprocess_options,
            progress_interval=10000
        )
        is_streaming = True
        st.session_state.clustering_service = clustering_service
    else:
        clustering_service = MemMinhashLSHClustering(
            threshold=threshold,
            shingle_size=shingle_size,
            preprocess_options=preprocess_options
        )
        is_streaming = False
        st.session_state.clustering_service = clustering_service

    progress_bar = progress_placeholder.progress(0)
    status_text = progress_placeholder.empty()

    optimal_batch_size = calculate_optimal_batch_size(len(texts))
    if not is_streaming:
        clustering_service.batch_size = optimal_batch_size

    def progress_callback(stage, progress, processed, clusters):
        if stage == "Processing documents":
            actual_progress = progress * 0.90
        elif stage == "Clustering":
            actual_progress = 0.90 + (progress * 0.05)
        elif stage == "Building results":
            actual_progress = 0.95 + (progress * 0.05)
        elif stage == "Complete":
            actual_progress = 1.0
        else:
            actual_progress = progress

        progress_bar.progress(min(actual_progress, 1.0))

        if stage == "Processing documents":
            status_text.text(f"Processing {processed:,} / {len(texts):,} documents...")
        elif stage == "Clustering":
            status_text.text("Forming clusters...")
        elif stage == "Building results":
            status_text.text("Finalizing results...")
        elif stage == "Complete":
            status_text.text("Complete!")

        logger.update_peak_memory()
        logger.update_peak_cpu()

    st.session_state.start_time = time.time()

    try:
        logger.log(f"Processing {len(texts):,} documents...")

        if is_streaming:
            cluster_results = clustering_service.cluster_streaming(texts, progress_callback)
            all_similarities = clustering_service.get_all_similarities()
            st.session_state.clustering_similarities = all_similarities

            similarity_dict = {
                tuple(sorted([doc1, doc2])): sim
                for doc1, doc2, sim in all_similarities
            }

            clustered_docs = build_clustered_documents(
                texts, cluster_results, similarity_dict,
                clustering_service.chunk_size, progress_callback
            )
        else:
            clustered_docs = clustering_service.cluster_documents(texts, progress_callback)
            all_similarities = clustering_service.get_all_similarities()
            st.session_state.clustering_similarities = all_similarities

        logger.log(f"Clustering completed for {len(clustered_docs)} documents")

        result = convert_to_result_format(
            clustered_docs, st.session_state.get('file_data'),
            st.session_state.selected_id_column, is_streaming
        )

        if hasattr(clustering_service, 'cleanup'):
            clustering_service.cleanup()

        logger.log(f"Conversion to result format completed")

        progress_callback("Complete", 1.0, len(texts), 0)

        return result, logger

    except Exception as e:
        st.error(f"Clustering failed: {str(e)}")
        return None, logger

def build_clustered_documents(texts: List[str], cluster_results: Dict[int, int],
                              similarity_dict: Dict[Tuple[int, int], float],
                              chunk_size: int, progress_callback=None) -> List[ClusteredDocument]:

    clusters_to_docs = defaultdict(list)
    for doc_id, cluster_id in cluster_results.items():
        clusters_to_docs[cluster_id].append(doc_id)

    certainties = {}
    total_docs = len(texts)
    processed = 0

    for cluster_id, doc_ids in clusters_to_docs.items():
        for doc_id in doc_ids:
            certainty = calculate_certainty_vectorized(
                doc_id, doc_ids, similarity_dict
            )
            certainties[doc_id] = certainty

            processed += 1
            if progress_callback and processed % 5000 == 0:
                build_progress = processed / total_docs
                progress_callback("Building results", build_progress, processed, 0)

    clustered_docs = []
    for i, text in enumerate(texts):
        cluster_id = cluster_results.get(i, 0)
        certainty = certainties.get(i, 0.5)

        clustered_docs.append(ClusteredDocument(
            text=text,
            cluster_id=cluster_id,
            certainty=certainty,
            original_index=i,
            batch_id=i // chunk_size
        ))

    return clustered_docs


def calculate_certainty_vectorized(doc_id: int, cluster_members: List[int],
                                   similarity_dict: Dict[Tuple[int, int], float]) -> float:
    other_members = [m for m in cluster_members if m != doc_id]

    if not other_members:
        return 1.0

    similarities = [
        similarity_dict.get(tuple(sorted([doc_id, member_id])), 0.0)
        for member_id in other_members
    ]

    valid_sims = [s for s in similarities if s > 0]

    if not valid_sims:
        return 0.5

    return sum(valid_sims) / len(valid_sims)


def convert_to_result_format(clustered_docs: List[ClusteredDocument],
                             original_df: pd.DataFrame,
                             selected_id_column: str,
                             is_streaming: bool) -> List[Dict]:
    result = []

    for doc in clustered_docs:
        original_row = original_df.iloc[doc.original_index].to_dict()

        doc_id = original_row.get(selected_id_column, doc.original_index) \
            if selected_id_column else doc.original_index

        result_row = original_row.copy()
        result_row.update({
            "id": doc_id,
            "cluster_id": doc.cluster_id,
            "certainty": round(doc.certainty, 4),
            "original_index": doc.original_index,
            "batch_id": doc.batch_id,
            "clustering_method": "streaming" if is_streaming else "optimized"
        })

        result.append(result_row)

    return result
def calculate_optimal_batch_size(dataset_size: int) -> int:
    if dataset_size <= 5000:
        return 1000
    elif dataset_size <= 20000:
        return 2500
    elif dataset_size <= 100_000:
        return 5000
    else:
        return 10000


def create_export_dataframe(clustered_data: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(clustered_data)

    clustering_cols = ['cluster_id', 'certainty', 'original_index', 'batch_id']
    other_cols = [col for col in df.columns if col not in clustering_cols]

    ordered_cols = other_cols + clustering_cols
    df = df[ordered_cols]

    return df

def main():
    params = st.query_params
    if "view_mode" in params and "selected_cluster" in params:
        if params["view_mode"][0] == "cluster":
            st.session_state.view_mode = "cluster"
            st.session_state.selected_cluster = int(params["selected_cluster"][0])

    with st.sidebar:
        if st.button("🔄 Clear Cache & Reset"):
            if 'clustering_service' in st.session_state:
                if hasattr(st.session_state.clustering_service, 'cleanup'):
                    st.session_state.clustering_service.cleanup()

            st.cache_data.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        if st.session_state.clustered_data is not None:
            st.markdown("### Quick Stats")
            df = pd.DataFrame(st.session_state.clustered_data)
            st.metric("Documents", len(df))
            st.metric("Clusters", df['cluster_id'].nunique())
            st.metric("Avg Confidence", f"{df['certainty'].mean():.0%}")

    st.markdown("# CorpusClues")
    st.markdown("*Discover similar documents in your text collection using advanced MinHash LSH clustering*")

    if st.session_state.clustered_data is None and not st.session_state.processing:
        st.markdown("## Upload Your Documents")

        upload_col1, upload_col2 = st.columns(2)

        with upload_col1:
            st.markdown("### Option 1: Upload Your File")
            uploaded_file = st.file_uploader(
                "Make sure the texts are in a column with header 'text'.",
                type=['csv'],
                help="Your CSV must have a column named 'text' containing the documents to analyze. Maximum 60,000 rows allowed.",
                key="file_uploader"
            )

        with upload_col2:
            st.markdown("### Option 2: Try Demo Data")
            st.markdown("""
            Load a sample dataset of Byzantine book epigrams from the 
            [Database of Byzantine Book Epigrams (DBBE)](https://dbbe.ugent.be), 
            as featured in our LREC 2026 submission.
            """)
            if st.button("Load Demo Dataset", width='stretch'):
                demo_path = "demo_data/paper_verses.csv"
                try:
                    df_demo = pd.read_csv(demo_path)
                    st.session_state.file_data = df_demo
                    st.session_state.file_name = "paper_verses.csv (DBBE)"
                    st.session_state.is_demo = True

                    possible_id_columns = [col for col in df_demo.columns if
                                           col.lower().strip() in ['id', 'doc_id', 'document_id', 'index', 'number']]
                    if possible_id_columns:
                        st.session_state.selected_id_column = possible_id_columns[0]
                    else:
                        st.session_state.selected_id_column = None
                    st.session_state.similarity_threshold = 0.2
                    st.session_state.shingle_size = 4
                    st.session_state.demo_settings_applied = True

                    st.rerun()
                except FileNotFoundError:
                    st.error(f"Demo file not found at {demo_path}")
                except Exception as e:
                    st.error(f"Error loading demo file: {str(e)}")
        if uploaded_file is not None:
            st.session_state.demo_settings_applied = False

            try:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > 100:
                    st.error(f"❌ File too large ({file_size_mb:.1f}MB). Please use a smaller file (max ~100MB).")
                    st.session_state.file_data = None
                else:
                    if 'file_data' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
                        try:
                            df_sample = pd.read_csv(uploaded_file, nrows=1)
                            uploaded_file.seek(0)

                            row_count = sum(1 for _ in uploaded_file) - 1
                            uploaded_file.seek(0)

                            MAX_ROWS = 60_000
                            if row_count > MAX_ROWS:
                                st.error(f"❌ File contains {row_count:,} rows. Maximum allowed is {MAX_ROWS:,} rows.")
                                st.info(f"💡 Consider:")
                                st.info(f"   • Sampling your data randomly")
                                st.info(f"   • Splitting into multiple smaller files")
                                st.info(f"   • Using a more powerful deployment")
                                st.session_state.file_data = None
                            else:
                                df_preview = pd.read_csv(uploaded_file)
                                st.session_state.file_data = df_preview
                                st.session_state.file_name = uploaded_file.name
                                uploaded_file.seek(0)

                        except Exception as read_error:
                            st.error(f"Error reading file: {str(read_error)}")
                            st.session_state.file_data = None
                    else:
                        df_preview = st.session_state.file_data

                    if st.session_state.file_data is not None:
                        text_columns = [col for col in st.session_state.file_data.columns if
                                        col.lower().strip() == 'text']
                        if text_columns:
                            st.success(
                                f"Found {len(st.session_state.file_data):,} documents in column '{text_columns[0]}'")

                            estimated_memory = (len(st.session_state.file_data) * 0.5) / 1024
                            if estimated_memory > 500:
                                st.warning(
                                    f"Large dataset ({estimated_memory:.0f}MB estimated). Processing may take several minutes.")

                            possible_id_columns = [col for col in st.session_state.file_data.columns if
                                                   col.lower().strip() in ['id', 'doc_id', 'document_id', 'index',
                                                                           'number']]
                            if possible_id_columns:
                                if 'selected_id_column' not in st.session_state or st.session_state.get(
                                        'file_name') != uploaded_file.name:
                                    st.session_state.selected_id_column = possible_id_columns[0]
                                selected_id_column = st.selectbox(
                                    "Select ID column (optional):",
                                    options=["(None)"] + possible_id_columns,
                                    index=0 if 'selected_id_column' not in st.session_state else possible_id_columns.index(
                                        st.session_state.selected_id_column) + 1,
                                    key="id_column_select"
                                )
                                st.session_state.selected_id_column = selected_id_column if selected_id_column != "(None)" else None
                            else:
                                st.session_state.selected_id_column = None
                                st.info("ℹ️ No ID column detected. Using document index as ID.")
                        else:
                            st.error("❌ No 'text' column found. Please ensure your CSV has a column named 'text'")
                            st.session_state.file_data = None
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.session_state.file_data = None

        st.markdown("---")
        st.markdown("## Clustering Settings")

        if st.session_state.get('demo_settings_applied', False):
            st.info("**Demo mode active** - Optimized settings applied automatically")

        settings_col1, settings_col2 = st.columns([1, 1])

        with settings_col1:
            default_shingle = st.session_state.get('shingle_size', 4)
            pattern_size = st.selectbox(
                "Text pattern size:",
                options=[2, 3, 4, 5, 6, 7, 8],
                index=[2, 3, 4, 5, 6, 7, 8].index(default_shingle),
                format_func=lambda x: f"{x} characters",
                key="shingle_size_select",
                help="Size of character patterns used for comparison"
            )

            default_threshold = st.session_state.get('similarity_threshold', 0.3)
            similarity_level = st.select_slider(
                "Similarity threshold:",
                options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                value=default_threshold,
                format_func=lambda x: {
                    0.1: "0.1", 0.2: "0.2", 0.3: "0.3",
                    0.4: "0.4+", 0.5: "0.5", 0.6: "0.6",
                    0.7: "0.7", 0.8: "0.8", 0.9: "0.9"
                }[x],
                key="similarity_slider",
                help="Minimum similarity required to group documents"
            )

        with settings_col2:
            st.markdown("**Text preprocessing:**")
            st.caption("Configure how texts are normalized before comparison")

            lowercase = st.checkbox(
                "Convert to lowercase",
                value=True,
                key="preprocess_lowercase",
                help="Makes comparison case-insensitive"
            )

            remove_diacritics = st.checkbox(
                "Remove diacritics/accents",
                value=True,
                key="preprocess_remove_diacritics",
                help="Treats 'café' and 'cafe' as identical (recommended for historical texts)"
            )

            remove_punctuation = st.checkbox(
                "Remove punctuation",
                value=True,
                key="preprocess_remove_punctuation",
                help="Ignores punctuation marks in comparison"
            )


        st.session_state.similarity_threshold = similarity_level
        st.session_state.shingle_size = pattern_size

        if st.session_state.get('demo_settings_applied', False):
            if similarity_level != 0.2 or pattern_size != 4:
                st.session_state.demo_settings_applied = False

        st.markdown("---")
        st.markdown("## Start Cluster Detection")

        has_data = st.session_state.get('file_data') is not None

        if has_data:
            estimated_time = max(5, len(st.session_state.file_data) // 100)
            st.markdown(f"Ready to analyze **{len(st.session_state.file_data):,} documents**")
            st.caption(f"Estimated processing time: ~{estimated_time} seconds")
        else:
            st.markdown("Please upload a file or load demo data first.")

        button_col1, button_col2 = st.columns([1, 1])

        with button_col1:
            if st.button("Start Analysis", type="primary", width='stretch', disabled=not has_data):
                if 'clustering_service' in st.session_state:
                    if hasattr(st.session_state.clustering_service, 'cleanup'):
                        st.session_state.clustering_service.cleanup()
                    del st.session_state.clustering_service

                st.session_state.processing = True
                st.rerun()

    elif st.session_state.processing:
        st.markdown("## Processing Your Documents")

        st.markdown(f"""
        <div class="processing-container">
            <h3>Clustering Analysis in Progress</h3>
            <p>Your documents are being analyzed for similarity patterns using optimized MinHash LSH clustering.</p>
        </div>
        """, unsafe_allow_html=True)

        if 'file_data' not in st.session_state or st.session_state.file_data is None:
            st.error("File data lost. Please re-upload your file.")
            st.session_state.processing = False
            if st.button("Go Back"):
                st.rerun()
            return

        try:
            df = st.session_state.file_data
            text_columns = [col for col in df.columns if col.lower().strip() == 'text']
            texts = df[text_columns[0]].dropna().astype(str).tolist()

            threshold = st.session_state.get('similarity_threshold', 0.3)
            shingle_size = st.session_state.get('shingle_size', 5)

            st.info(
                f"Processing {len(texts):,} documents with {threshold} similarity threshold and {shingle_size}-character shingles...")

            progress_container = st.container()

            if 'clustering_completed' not in st.session_state:
                st.session_state.clustering_started = True

                with st.spinner("Initializing clustering analysis..."):
                    clustering_method = 'auto'
                    preprocess_options = {
                        'lowercase': st.session_state.get('preprocess_lowercase', True),
                        'remove_diacritics': st.session_state.get('preprocess_remove_diacritics', True),
                        'remove_punctuation': st.session_state.get('preprocess_remove_punctuation', True)
                    }
                    results, logger = run_clustering_analysis(
                        texts, threshold, shingle_size, progress_container,
                        clustering_method, preprocess_options
                    )

                if results:
                    st.session_state.clustering_completed = True
                    st.session_state.clustered_data = results
                    st.session_state.view_mode = 'overview'
                    st.session_state.processing = False

                    if 'clustering_started' in st.session_state:
                        del st.session_state.clustering_started

                    df_results = pd.DataFrame(results)
                    logger.store_summary(len(texts), df_results['cluster_id'].nunique())

                    st.success("Analysis complete!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state.processing = False
                    if 'clustering_started' in st.session_state:
                        del st.session_state.clustering_started
                    st.error("Clustering failed. Please try again.")
            else:
                st.session_state.processing = False
                st.session_state.view_mode = 'overview'
                st.rerun()

            st.markdown("---")
            if st.button("Cancel Processing", type="secondary"):
                st.session_state.processing = False
                if 'clustering_started' in st.session_state:
                    del st.session_state.clustering_started
                if 'clustering_completed' in st.session_state:
                    del st.session_state.clustering_completed
                st.rerun()

        except Exception as e:
            st.session_state.processing = False
            if 'clustering_started' in st.session_state:
                del st.session_state.clustering_started
            if 'clustering_completed' in st.session_state:
                del st.session_state.clustering_completed
            st.error(f"Processing error: {str(e)}")
            st.error("Please check your CSV format and try again.")
            if st.button("🔄 Try Again"):
                st.rerun()
    else:
        df_results = pd.DataFrame(st.session_state.clustered_data)

        tab1, tab2, tab3, tab4 = st.tabs(["Browse Clusters", "Search Documents", "Statistics", "Export"])

        with tab1:
            if st.session_state.view_mode == 'overview':
                display_cluster_overview(df_results)

                with st.container():
                    cols = st.columns(4)

                    total_docs = len(df_results)
                    num_clusters = df_results['cluster_id'].nunique()
                    avg_confidence = df_results['certainty'].mean()
                    high_confidence = len(df_results[df_results['certainty'] >= 0.8])

                    with cols[0]:
                        st.metric("Documents", f"{total_docs:,}")
                    with cols[1]:
                        st.metric("Clusters", num_clusters)
                    with cols[2]:
                        st.metric("Avg Confidence", f"{avg_confidence:.0%}")
                    with cols[3]:
                        st.metric("High Confidence", f"{high_confidence} ({high_confidence / total_docs:.0%})")
            elif st.session_state.view_mode == 'cluster':
                display_cluster_details(df_results, st.session_state.selected_cluster)

        with tab2:
            display_global_search(df_results)

        with tab3:
            st.markdown("## Analysis Statistics")
            col1, col2 = st.columns(2)

            with col1:
                total_docs = len(df_results)
                num_clusters = df_results['cluster_id'].nunique()
                avg_confidence = df_results['certainty'].mean()
                st.markdown("### Overview")
                st.write(f"**Documents analyzed:** {total_docs:,}")
                st.write(f"**Clusters formed:** {num_clusters}")
                st.write(f"**Average confidence:** {avg_confidence:.1%}")
                if 'batch_id' in df_results.columns:
                    num_batches = df_results['batch_id'].nunique()
                    st.write(f"**Batches processed:** {num_batches}")

                cluster_sizes = df_results['cluster_id'].value_counts().sort_values(ascending=False)
                st.markdown("### Top 5 Largest Clusters")
                for cluster_id, size in cluster_sizes.head(5).items():
                    cluster_confidence = df_results[df_results['cluster_id'] == cluster_id]['certainty'].mean()
                    st.write(f"**Cluster {cluster_id}:** {size} documents ({cluster_confidence:.0%} confidence)")

                if len(cluster_sizes) > 5:
                    with st.expander(f"View All {len(cluster_sizes)} Clusters"):
                        cluster_data = []
                        for cluster_id in sorted(cluster_sizes.index):
                            size = cluster_sizes[cluster_id]
                            cluster_docs = df_results[df_results['cluster_id'] == cluster_id]
                            avg_conf = cluster_docs['certainty'].mean()
                            cluster_data.append({
                                'Cluster': f"Cluster {cluster_id}",
                                'Documents': size,
                                'Avg Confidence': f"{avg_conf:.1%}"
                            })
                        cluster_df = pd.DataFrame(cluster_data)
                        st.dataframe(cluster_df, hide_index=True, width='stretch')

                st.markdown("### Confidence Distribution")
                high_conf = len(df_results[df_results['certainty'] >= 0.8])
                med_conf = len(df_results[(df_results['certainty'] >= 0.6) & (df_results['certainty'] < 0.8)])
                low_conf = len(df_results[df_results['certainty'] < 0.6])
                st.write(f"**High confidence (≥80%):** {high_conf} documents ({high_conf / total_docs:.1%})")
                st.write(f"**Medium confidence (60-79%):** {med_conf} documents ({med_conf / total_docs:.1%})")
                st.write(f"**Low confidence (<60%):** {low_conf} documents ({low_conf / total_docs:.1%})")

            with col2:
                st.markdown("### Performance Summary")
                if 'performance_summary' in st.session_state:
                    summary = st.session_state.performance_summary
                    st.metric("Total Processing Time", summary["total_processing_time"])
                    st.metric("Peak Memory Usage", summary["peak_memory_usage"])
                    st.metric("Texts Processed", summary["num_texts_processed"])
                    st.metric("Clusters Formed", summary["num_clusters_formed"])
                    st.metric("Peak CPU Usage", summary["peak_cpu_usage"])
                    st.caption(f"System: {summary['system_info']}")
                else:
                    st.info("Performance data will be available after processing.")

                fig = go.Figure(data=[
                    go.Histogram(x=df_results['certainty'], nbinsx=20, marker_color='lightblue')
                ])
                fig.update_layout(
                    title="Confidence Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Number of Documents",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, key="confidence_histogram")

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
                st.plotly_chart(fig, use_container_width=True, key="cluster_sizes_bar")

        with tab4:
            col1, col2 = st.columns(2)

            with col1:
                df_export = create_export_dataframe(st.session_state.clustered_data)

                st.markdown("#### Export Preview")
                st.markdown(f"**{len(df_export)} rows × {len(df_export.columns)} columns**")

                with st.expander("View All Export Columns"):
                    original_cols = []
                    clustering_cols = []

                    for col in df_export.columns:
                        if col in ['cluster_id', 'certainty', 'original_index', 'batch_id']:
                            clustering_cols.append(col)
                        else:
                            original_cols.append(col)

                    st.markdown("**Original columns from your upload:**")
                    st.write(", ".join(original_cols))

                    st.markdown("**Added clustering results:**")
                    st.write(", ".join(clustering_cols))

                st.dataframe(df_export.head(3), width='stretch')

                csv_data = df_export.to_csv(index=False)
                st.download_button(
                    label="Download Complete Results (CSV)",
                    data=csv_data,
                    file_name="text_clustering_results_complete.csv",
                    mime="text/csv",
                    type="primary",
                    width='stretch',
                    help="Downloads all original columns plus clustering results"
                )

                json_data = []
                for _, row in df_export.iterrows():
                    row_dict = {}
                    for k, v in row.items():
                        if pd.isna(v):
                            row_dict[k] = None
                        elif isinstance(v, (np.integer, np.floating)):
                            row_dict[k] = float(v) if isinstance(v, np.floating) else int(v)
                        elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
                            row_dict[k] = str(v)
                        else:
                            row_dict[k] = v
                    json_data.append(row_dict)

                json_str = json.dumps(json_data, indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=json_str,
                    file_name="text_clustering_results_complete.json",
                    mime="application/json",
                    width='stretch',
                    help="Downloads all original columns plus clustering results in JSON format"
                )

            with col2:
                st.markdown("### Export Options")

                clustering_only_df = df_results[
                    ['id', 'text', 'cluster_id', 'certainty', 'original_index', 'batch_id']].copy()
                clustering_csv = clustering_only_df.to_csv(index=False)

                st.download_button(
                    label="Download Clustering Results Only",
                    data=clustering_csv,
                    file_name="clustering_results_only.csv",
                    mime="text/csv",
                    help="Downloads only the text and clustering results (original format)"
                )

                st.markdown("#### Export Specific Clusters")
                selected_clusters = st.multiselect(
                    "Select clusters to export:",
                    options=sorted(df_results['cluster_id'].unique()),
                    key="cluster_export_select"
                )

                if selected_clusters:
                    filtered_df = df_export[df_export['cluster_id'].isin(selected_clusters)]
                    filtered_csv = filtered_df.to_csv(index=False)

                    st.download_button(
                        label=f"📥 Download Selected Clusters ({len(selected_clusters)} clusters, {len(filtered_df)} docs)",
                        data=filtered_csv,
                        file_name=f"selected_clusters_{'_'.join(map(str, selected_clusters))}.csv",
                        mime="text/csv"
                    )

                st.markdown("### Start New Analysis")
                if st.button("Analyze Different Documents", use_container_width=True):
                    if 'clustering_service' in st.session_state:
                        if hasattr(st.session_state.clustering_service, 'cleanup'):
                            st.session_state.clustering_service.cleanup()

                    st.session_state.clustered_data = None
                    st.session_state.view_mode = 'overview'
                    st.session_state.processing = False
                    if 'clustering_completed' in st.session_state:
                        del st.session_state.clustering_completed
                    if 'clustering_started' in st.session_state:
                        del st.session_state.clustering_started
                    st.rerun()

if __name__ == "__main__":
    main()