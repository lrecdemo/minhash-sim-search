import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import io
import json
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datasketch import MinHashLSH, MinHash
import time
import gc
import math
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import psutil
import platform
from datetime import datetime


class PerformanceLogger:
    def __init__(self):
        self.logs = []
        self.start_time = None
        self.peak_memory = 0
        self.peak_cpu = 0

    def start(self):
        self.start_time = time.time()
        self.log("Process started")

    def log(self, message, **kwargs):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": message,
            **kwargs
        }
        self.logs.append(entry)

    def update_peak_memory(self):
        process = psutil.Process()
        mem_info = process.memory_info()
        current_mem = mem_info.rss / (1024 ** 2)  # MB
        if current_mem > self.peak_memory:
            self.peak_memory = current_mem

    def update_peak_cpu(self):
        current_cpu = psutil.cpu_percent(interval=0.1)
        if current_cpu > self.peak_cpu:
            self.peak_cpu = current_cpu


    def get_summary(self, num_texts, num_clusters):
        total_time = time.time() - self.start_time
        return {
            "total_processing_time": f"{total_time:.2f} seconds",
            "peak_memory_usage": f"{self.peak_memory:.2f} MB",
            "num_texts_processed": f"{num_texts:,}",
            "num_clusters_formed": f"{num_clusters:,}",
            "peak_cpu_usage": f"{self.peak_cpu:.1f}%",
            "system_info": f"{platform.system()} {platform.release()}, {psutil.cpu_count()} cores"
        }

    def store_summary(self, num_texts, num_clusters):
        st.session_state.performance_summary = self.get_summary(num_texts, num_clusters)

# Page config
st.set_page_config(
    page_title="Text Similarity Clustering",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS styles
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

# Configuration
BATCH_SIZE = 1000
NUM_PERM = 64  # Reduced for speed
MAX_WORKERS = 4  # Conservative for Streamlit

# Pre-compiled regex patterns
CLEAN_PATTERN = re.compile(r'[^\w\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')


@dataclass
class ClusteredDocument:
    text: str
    cluster_id: int
    certainty: float
    original_index: int
    batch_id: int = 0


class OptimizedMinHashLSHClustering:
    """Optimized clustering implementation for Streamlit deployment."""

    def __init__(self, num_perm: int = NUM_PERM, threshold: float = 0.3):
        self.num_perm = num_perm
        self.threshold = threshold
        self.batch_size = BATCH_SIZE
        self.max_workers = MAX_WORKERS

    def preprocess_text_vectorized(self, texts: List[str]) -> List[str]:
        """Vectorized text preprocessing using pandas."""
        if len(texts) == 1:
            text = texts[0].lower()
            text = CLEAN_PATTERN.sub(' ', text)
            text = WHITESPACE_PATTERN.sub(' ', text)
            return [text.strip()]

        df = pd.DataFrame({'text': texts})
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].str.replace(CLEAN_PATTERN, ' ', regex=True)
        df['text'] = df['text'].str.replace(WHITESPACE_PATTERN, ' ', regex=True)
        df['text'] = df['text'].str.strip()

        return df['text'].tolist()

    @lru_cache(maxsize=10000)
    def generate_shingles_cached(self, text_hash: int, text: str, k: int = 3) -> frozenset:
        """Cached shingle generation."""
        words = text.split()
        if len(words) < k:
            return frozenset(words) if words else frozenset()

        shingles = set()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i + k])
            shingles.add(hash(shingle))
        return frozenset(shingles)

    def generate_shingles_fast(self, text: str, k: int = 3) -> Set[int]:
        """Fast shingle generation with caching."""
        text_hash = hash(text)
        cached_result = self.generate_shingles_cached(text_hash, text, k)
        return set(cached_result)

    def create_minhash_optimized(self, shingles: Set[int]) -> MinHash:
        """Optimized MinHash creation."""
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(str(shingle).encode('utf-8'))
        return minhash

    def process_batch_optimized(self, batch_texts: List[str], batch_start_idx: int,
                                progress_callback=None) -> Tuple[
        List[Tuple[int, Set[int], MinHash]], List[Tuple[int, int, float]]]:
        """Process a batch of texts with progress reporting."""
        batch_size = len(batch_texts)

        # Step 1: Preprocessing
        if progress_callback:
            progress_callback("preprocessing", 0.1)
        cleaned_texts = self.preprocess_text_vectorized(batch_texts)

        # Step 2: Shingle generation
        if progress_callback:
            progress_callback("generating_shingles", 0.3)

        optimal_workers = min(self.max_workers, max(2, batch_size // 10))

        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            shingle_futures = {
                executor.submit(self.generate_shingles_fast, text): i
                for i, text in enumerate(cleaned_texts)
            }

            shingle_sets = [None] * batch_size
            for future in as_completed(shingle_futures):
                i = shingle_futures[future]
                try:
                    shingle_sets[i] = future.result()
                except Exception:
                    shingle_sets[i] = set()

        # Step 3: MinHash creation
        if progress_callback:
            progress_callback("creating_minhashes", 0.5)

        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            minhash_futures = {
                executor.submit(self.create_minhash_optimized, shingles): i
                for i, shingles in enumerate(shingle_sets)
            }

            minhashes = [None] * batch_size
            for future in as_completed(minhash_futures):
                i = minhash_futures[future]
                try:
                    minhashes[i] = future.result()
                except Exception:
                    minhashes[i] = MinHash(num_perm=self.num_perm)

        # Step 4: Build batch data
        if progress_callback:
            progress_callback("building_data", 0.7)

        batch_data = []
        for i, (shingles, minhash) in enumerate(zip(shingle_sets, minhashes)):
            doc_idx = batch_start_idx + i
            batch_data.append((doc_idx, shingles, minhash))

        # Step 5: Similarity detection
        if progress_callback:
            progress_callback("finding_similarities", 0.9)

        similarities = []
        if len(batch_data) > 1:
            temp_lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

            for doc_idx, _, minhash in batch_data:
                temp_lsh.insert(str(doc_idx), minhash)

            processed_pairs = set()
            for doc_idx1, shingles1, minhash1 in batch_data:
                candidates = temp_lsh.query(minhash1)
                for candidate_key in candidates:
                    candidate_idx = int(candidate_key)
                    if candidate_idx != doc_idx1:
                        pair = tuple(sorted([doc_idx1, candidate_idx]))
                        if pair not in processed_pairs:
                            processed_pairs.add(pair)
                            similarity = minhash1.jaccard(batch_data[candidate_idx - batch_start_idx][2])
                            if similarity >= self.threshold:
                                similarities.append((pair[0], pair[1], similarity))

        if progress_callback:
            progress_callback("batch_complete", 1.0)

        return batch_data, similarities

    def cluster_documents_optimized(self, texts: List[str], progress_callback=None) -> List[ClusteredDocument]:
        """Main clustering function with progress reporting."""
        start_time = time.time()
        num_texts = len(texts)

        # Initialize data structures
        all_shingles = {}
        all_similarities = []
        processed_count = 0

        # Calculate batching
        num_batches = math.ceil(num_texts / self.batch_size)

        # Global LSH for cross-batch similarities
        global_lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        global_minhashes = {}

        # Process batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, num_texts)
            batch_texts = texts[batch_start:batch_end]

            def batch_progress_callback(stage, sub_progress):
                overall_progress = (batch_idx + sub_progress) / num_batches
                if progress_callback:
                    progress_callback(
                        f"Processing batch {batch_idx + 1}/{num_batches}: {stage}",
                        overall_progress,
                        processed_count + int(len(batch_texts) * sub_progress),
                        len(all_similarities)
                    )

            # Process batch
            batch_data, batch_similarities = self.process_batch_optimized(
                batch_texts, batch_start, batch_progress_callback
            )

            # Update global structures
            for doc_idx, shingles, minhash in batch_data:
                all_shingles[doc_idx] = shingles
                global_minhashes[doc_idx] = minhash
                global_lsh.insert(str(doc_idx), minhash)

            all_similarities.extend(batch_similarities)

            # Cross-batch similarity detection
            if batch_idx > 0:
                cross_batch_similarities = self._find_cross_batch_similarities_optimized(
                    batch_data, global_lsh, global_minhashes, batch_start
                )
                all_similarities.extend(cross_batch_similarities)

            processed_count += len(batch_texts)

            # Memory management
            if batch_idx % 2 == 0:
                gc.collect()

        # Clustering
        if progress_callback:
            progress_callback("Performing final clustering...", 0.95, processed_count, len(all_similarities))

        clusters = self._union_find_clustering_optimized(num_texts, all_similarities)

        # Create final results
        clustered_docs = self._create_clustered_documents_optimized(
            texts, clusters, all_similarities, all_shingles
        )

        processing_time = time.time() - start_time

        if progress_callback:
            progress_callback("Complete!", 1.0, num_texts, len(set(clusters)))

        return clustered_docs

    def _find_cross_batch_similarities_optimized(self, batch_data, global_lsh, global_minhashes, batch_start):
        """Find similarities across batches."""
        cross_similarities = []
        processed_pairs = set()

        for doc_idx, shingles, minhash in batch_data:
            candidates = global_lsh.query(minhash)
            for candidate_key in candidates:
                candidate_idx = int(candidate_key)
                if candidate_idx < batch_start:
                    pair = tuple(sorted([doc_idx, candidate_idx]))
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        candidate_minhash = global_minhashes[candidate_idx]
                        similarity = minhash.jaccard(candidate_minhash)
                        if similarity >= self.threshold:
                            cross_similarities.append((pair[0], pair[1], similarity))

        return cross_similarities

    def _union_find_clustering_optimized(self, n: int, similarities: List[Tuple[int, int, float]]) -> List[int]:
        """Optimized Union-Find clustering."""
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return

            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1

        for doc1, doc2, _ in similarities:
            union(doc1, doc2)

        root_to_cluster = {}
        clusters = []
        next_cluster_id = 0

        for i in range(n):
            root = find(i)
            if root not in root_to_cluster:
                root_to_cluster[root] = next_cluster_id
                next_cluster_id += 1
            clusters.append(root_to_cluster[root])

        return clusters

    def _create_clustered_documents_optimized(self, texts, clusters, similarities, all_shingles):
        """Create final clustered document objects."""
        similarity_dict = {}
        for doc1, doc2, sim in similarities:
            pair = tuple(sorted([doc1, doc2]))
            similarity_dict[pair] = sim

        clustered_docs = []

        for i, text in enumerate(texts):
            cluster_id = clusters[i]
            certainty = self._calculate_certainty_optimized(i, clusters, similarity_dict)

            clustered_docs.append(ClusteredDocument(
                text=text,
                cluster_id=cluster_id,
                certainty=certainty,
                original_index=i,
                batch_id=i // self.batch_size
            ))

        return clustered_docs

    def _calculate_certainty_optimized(self, doc_idx: int, clusters: List[int],
                                       similarities: Dict[Tuple[int, int], float]) -> float:
        """Calculate certainty score."""
        cluster_id = clusters[doc_idx]
        cluster_members = [i for i, cid in enumerate(clusters) if cid == cluster_id and i != doc_idx]

        if not cluster_members:
            return 1.0

        total_similarity = 0.0
        count = 0

        for member_idx in cluster_members:
            pair = tuple(sorted([doc_idx, member_idx]))
            if pair in similarities:
                total_similarity += similarities[pair]
                count += 1

        return total_similarity / count if count > 0 else 0.5


# Initialize session state
if 'clustered_data' not in st.session_state:
    st.session_state.clustered_data = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'overview'
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
if 'processing' not in st.session_state:
    st.session_state.processing = False


def get_confidence_emoji_and_text(certainty: float) -> str:
    """Get emoji and text for confidence level."""
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
    """Display document with proper styling."""
    search_terms = search_terms or []
    doc_id = doc.get('id', doc['original_index']) if 'id' in doc else doc['original_index']
    if show_cluster_info:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.markdown(f"**🗂️ Cluster {doc['cluster_id']}**")
        with col2:
            st.markdown(get_confidence_emoji_and_text(doc['certainty']))
        with col3:
            if 'batch_id' in doc:
                st.markdown(f"*Batch {doc['batch_id']}*")
    else:
        st.markdown(get_confidence_emoji_and_text(doc['certainty']))
    display_text = doc['text']
    if is_preview and len(display_text) > 300:
        display_text = display_text[:300] + "..."
    if search_terms:
        for term in search_terms:
            if term.strip():
                display_text = display_text.replace(term, f"**{term}**")
                display_text = display_text.replace(term.lower(), f"**{term.lower()}**")
                display_text = display_text.replace(term.upper(), f"**{term.upper()}**")
    container_class = "preview-container" if is_preview else "document-container"
    st.markdown(f"""
    <div class="{container_class}">
        <strong>{doc_id}:</strong> {display_text}
    </div>
    """, unsafe_allow_html=True)


def display_cluster_overview(df: pd.DataFrame):
    # """Display overview of all clusters."""
    # st.markdown("## 📚 Document Clusters Overview")
    # st.markdown("*Each cluster contains documents with similar content. Click on a cluster to explore all documents.*")
    # # Quick stats
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     st.metric("📄 Documents", f"{len(df):,}")
    # with col2:
    #     st.metric("🗂️ Clusters", df['cluster_id'].nunique())
    # with col3:
    #     avg_confidence = df['certainty'].mean()
    #     st.metric("🎯 Avg Confidence", f"{avg_confidence:.0%}")
    # with col4:
    #     if 'batch_id' in df.columns:
    #         batches = df['batch_id'].nunique()
    #         st.metric("📦 Batches", batches)
    # st.markdown("---")
    # Group by cluster
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
    cluster_stats = cluster_stats.sort_values(
        by=['avg_confidence', 'size'],
        ascending=[False, False]
    )
    total_clusters = len(cluster_stats)
    # Pagination for large numbers of clusters
    if total_clusters > 20:
        st.info(
            f"📊 Large corpus detected ({total_clusters} clusters). Showing in pages, sorted by confidence and size.")
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
        start_idx = (current_page - 1) * clusters_per_page
        end_idx = min(start_idx + clusters_per_page, total_clusters)
        clusters_to_show = cluster_stats.iloc[start_idx:end_idx]
    else:
        clusters_to_show = cluster_stats
    # Display clusters
    for _, row in clusters_to_show.iterrows():
        cluster_id = row['cluster_id']
        cluster_size = row['size']
        avg_confidence = row['avg_confidence']
        batch_info = f" • Spans {row['batches']} batch{'es' if row['batches'] != 1 else ''}" if 'batches' in row else ""
        col1, col2 = st.columns([3, 1])
        with col1:
            # Inside your loop for each cluster:
            st.markdown(f"""
            <div class="cluster-header">
                <h3 style="margin: 0; color: white;">🗂️ Cluster {cluster_id}</h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9; color: white;">
                    {cluster_size} documents • {avg_confidence:.0%} avg confidence{batch_info}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Document previews: each preview and confidence on a single row
            cluster_docs = df[df['cluster_id'] == cluster_id]
            top_docs = cluster_docs.nlargest(3, 'certainty')
            for idx, (_, doc) in enumerate(top_docs.iterrows()):
                doc_id = doc.get('id', doc['original_index']) if 'id' in doc else doc['original_index']
                st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <strong>Preview {idx + 1}:</strong> {doc_id}: {doc['text'][:100]}... <span style="color: #667eea;">{get_confidence_emoji_and_text(doc['certainty'])}</span>
                </div>
                """, unsafe_allow_html=True)

            # Button underneath previews
            if st.button(f"📖 View All {cluster_size} Documents", key=f"view_cluster_{cluster_id}"):
                st.session_state.view_mode = 'cluster'
                st.session_state.selected_cluster = cluster_id
                st.rerun()

            st.markdown("---")


def display_cluster_details(df: pd.DataFrame, cluster_id: int):
    """Display all documents in a specific cluster."""
    cluster_docs = df[df['cluster_id'] == cluster_id].copy()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"# 📚 Cluster {cluster_id} - All Documents")
        batch_info = ""
        if 'batch_id' in cluster_docs.columns:
            batches = cluster_docs['batch_id'].nunique()
            batch_info = f" • Spans {batches} batch{'es' if batches != 1 else ''}"
        st.markdown(
            f"**{len(cluster_docs)} documents • {cluster_docs['certainty'].mean():.0%} avg confidence{batch_info}**")
    with col2:
        if st.button("← Back to Overview", type="secondary"):
            st.session_state.view_mode = 'overview'
            st.rerun()
    if len(cluster_docs) > 100:
        st.warning(
            f"⚡ Large cluster detected ({len(cluster_docs)} documents). Consider using search to find specific documents.")
    # Search and sort controls
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
        search_filtered = cluster_docs[mask]
        if len(search_filtered) == 0:
            st.warning(f"No documents found containing '{search_query}' in this cluster.")
        else:
            st.success(f"Found {len(search_filtered)} documents matching '{search_query}'")
            filtered_docs = search_filtered
    # Sort documents
    if sort_option == "confidence":
        filtered_docs = filtered_docs.sort_values('certainty', ascending=False)
    elif sort_option == "original":
        filtered_docs = filtered_docs.sort_values('original_index')
    elif sort_option == "length":
        filtered_docs['text_length'] = filtered_docs['text'].str.len()
        filtered_docs = filtered_docs.sort_values('text_length', ascending=False)
    # Pagination for large clusters
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
        start_idx = (current_page - 1) * docs_per_page
        end_idx = min(start_idx + docs_per_page, len(filtered_docs))
        page_docs = filtered_docs.iloc[start_idx:end_idx]
    else:
        page_docs = filtered_docs
    # Display documents
    st.markdown(f"### 📄 Documents ({len(page_docs)} of {len(filtered_docs)} shown)")
    for i, (_, doc) in enumerate(page_docs.iterrows()):
        expand_default = (i < 5) and (len(page_docs) <= 20)
        doc_id = doc.get('id', doc['original_index']) if 'id' in doc else doc['original_index']
        with st.expander(f"Document #{doc_id} - {get_confidence_level(doc['certainty'])} Confidence",
                         expanded=expand_default):
            st.markdown(f"**{doc_id}:** {doc['text']}")
            cols = st.columns(4) if 'batch_id' in doc else st.columns(3)
            with cols[0]:
                st.caption(f"Original Position: #{doc['original_index']}")
            with cols[1]:
                st.caption(f"Confidence: {doc['certainty']:.1%}")
            with cols[2]:
                st.caption(f"Text Length: {len(doc['text'])} chars")
            if 'batch_id' in doc and len(cols) > 3:
                with cols[3]:
                    st.caption(f"Batch: {doc['batch_id']}")


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
                st.markdown(f"#### 🗂️ From Cluster {cluster_id} ({len(group)} documents)")
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

def run_clustering_analysis(texts: List[str], threshold: float, progress_placeholder):
    logger = PerformanceLogger()
    logger.start()
    logger.log("Initializing clustering service")
    clustering_service = OptimizedMinHashLSHClustering(threshold=threshold)
    progress_bar = progress_placeholder.progress(0)
    status_text = progress_placeholder.empty()
    metrics_container = progress_placeholder.container()

    def progress_callback(stage, progress, processed, clusters):
        progress_bar.progress(progress)
        status_text.text(f"Status: {stage}")
        logger.update_peak_memory()
        logger.update_peak_cpu()
        if processed > 0:
            with metrics_container:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processed", f"{processed:,}/{len(texts):,}")
                with col2:
                    st.metric("Clusters Found", clusters)
                with col3:
                    rate = processed / max(1, (time.time() - st.session_state.start_time))
                    st.metric("Rate", f"{rate:.1f} docs/sec")

    st.session_state.start_time = time.time()
    try:
        logger.log(f"Processing {len(texts):,} documents with threshold {threshold}")
        clustered_docs = clustering_service.cluster_documents_optimized(texts, progress_callback)
        logger.log(f"Clustering completed for {len(clustered_docs)} documents")

        # Convert to dict format
        result = []
        for i, doc in enumerate(clustered_docs):
            doc_id = st.session_state.get('file_data').iloc[doc.original_index].get(st.session_state.selected_id_column,
                                                                                    doc.original_index) \
                if st.session_state.selected_id_column else doc.original_index
            result.append({
                "id": doc_id,
                "text": doc.text,
                "cluster_id": doc.cluster_id,
                "certainty": round(doc.certainty, 4),
                "original_index": doc.original_index,
                "batch_id": doc.batch_id
            })
        logger.log(f"Conversion to result format completed")
        return result, logger
    except Exception as e:
        st.error(f"Clustering failed: {str(e)}")
        return None, logger



def main():
    # Handle URL parameters for cluster navigation
    params = st.query_params
    if "view_mode" in params and "selected_cluster" in params:
        if params["view_mode"][0] == "cluster":
            st.session_state.view_mode = "cluster"
            st.session_state.selected_cluster = int(params["selected_cluster"][0])

    # Sidebar
    with st.sidebar:
        if st.button("🔄 Clear Cache & Reset"):
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

    # Header
    st.markdown("# 📚 Text Similarity Clustering")
    st.markdown("*Discover similar documents in your text collection using advanced MinHash LSH clustering*")

    # Main content
    if st.session_state.clustered_data is None and not st.session_state.processing:
        # Upload section
        st.markdown("## 📤 Upload Your Documents")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV file with your texts",
                type=['csv'],
                help="Your CSV must have a column named 'text' containing the documents to analyze. Optionally, you can select an ID column.",
                key="file_uploader"
            )
            if uploaded_file is not None:
                try:
                    # Store file data in session state immediately
                    if 'file_data' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
                        df_preview = pd.read_csv(uploaded_file)
                        st.session_state.file_data = df_preview
                        st.session_state.file_name = uploaded_file.name
                        uploaded_file.seek(0)  # Reset file pointer
                    else:
                        df_preview = st.session_state.file_data
                    text_columns = [col for col in df_preview.columns if col.lower().strip() == 'text']
                    if text_columns:
                        st.success(f"✅ Found {len(df_preview):,} documents in column '{text_columns[0]}'")
                        # Find possible ID columns
                        possible_id_columns = [col for col in df_preview.columns if
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
                        if len(df_preview) > 50000:
                            st.error("❌ Maximum 50,000 documents supported for Streamlit deployment")
                            st.session_state.file_data = None
                        else:
                            # Show sample
                            st.markdown("**Sample documents:**")
                            for i in range(min(3, len(df_preview))):
                                sample_text = str(df_preview.iloc[i][text_columns[0]])[:150] + "..."
                                sample_id = str(df_preview.iloc[i].get(st.session_state.selected_id_column,
                                                                       i)) if st.session_state.selected_id_column else i
                                st.markdown(f"*{sample_id}: {sample_text}*")
                    else:
                        st.error("❌ No 'text' column found. Please ensure your CSV has a column named 'text'")
                        st.session_state.file_data = None
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.session_state.file_data = None

        with col2:
            st.markdown("**Clustering Settings**")

            similarity_level = st.select_slider(
                "Similarity threshold:",
                options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                value=0.3,
                format_func=lambda x: {
                    0.1: "Very Loose", 0.2: "Loose", 0.3: "Moderate",
                    0.4: "Moderate+", 0.5: "Balanced", 0.6: "Strict",
                    0.7: "Very Strict", 0.8: "Extremely Strict", 0.9: "Nearly Identical"
                }[x],
                key="similarity_slider"
            )

            st.caption("Higher = stricter similarity requirements")

            # Store similarity level in session state
            st.session_state.similarity_threshold = similarity_level

            if st.session_state.get('file_data') is not None:
                estimated_time = max(5, len(st.session_state.file_data) // 100)
                st.caption(f"Estimated processing time: ~{estimated_time} seconds")

                if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                    st.session_state.processing = True
                    st.rerun()

    elif st.session_state.processing:
        # Processing section
        st.markdown("## 🔄 Processing Your Documents")

        st.markdown(f"""
        <div class="processing-container">
            <h3>Clustering Analysis in Progress</h3>
            <p>Your documents are being analyzed for similarity patterns using optimized MinHash LSH clustering.</p>
        </div>
        """, unsafe_allow_html=True)

        # Check if we have file data
        if 'file_data' not in st.session_state or st.session_state.file_data is None:
            st.error("File data lost. Please re-upload your file.")
            st.session_state.processing = False
            if st.button("🔄 Go Back"):
                st.rerun()
            return

        try:
            # Get data from session state
            df = st.session_state.file_data
            text_columns = [col for col in df.columns if col.lower().strip() == 'text']
            texts = df[text_columns[0]].dropna().astype(str).tolist()

            st.info(
                f"Processing {len(texts):,} documents with {st.session_state.similarity_threshold} similarity threshold...")

            # Create a single container for all progress updates
            progress_container = st.container()

            # Run clustering in a separate thread to avoid blocking
            # In the processing section of main():
            # In the processing section of main():
            if 'clustering_started' not in st.session_state:
                st.session_state.clustering_started = True
                threshold = st.session_state.get('similarity_threshold', 0.3)
                with st.spinner("Initializing clustering analysis..."):
                    results, logger = run_clustering_analysis(texts, threshold, progress_container)
                if results:
                    st.session_state.clustered_data = results
                    st.session_state.view_mode = 'overview'
                    st.session_state.processing = False
                    st.success("✅ Analysis complete!")
                    st.balloons()
                    # Store performance summary for later display
                    df_results = pd.DataFrame(results)
                    logger.store_summary(len(texts), df_results['cluster_id'].nunique())
                    time.sleep(1)
                    st.rerun()

            # Show cancel option
            st.markdown("---")
            if st.button("❌ Cancel Processing", type="secondary"):
                st.session_state.processing = False
                if 'clustering_started' in st.session_state:
                    del st.session_state.clustering_started
                st.rerun()

        except Exception as e:
            st.session_state.processing = False
            if 'clustering_started' in st.session_state:
                del st.session_state.clustering_started
            st.error(f"Processing error: {str(e)}")
            st.error("Please check your CSV format and try again.")
            if st.button("🔄 Try Again"):
                st.rerun()

    else:
        # Results section
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
            st.markdown("## 📊 Analysis Statistics")
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

                # Top clusters
                cluster_sizes = df_results['cluster_id'].value_counts().sort_values(ascending=False)
                st.markdown("### Top 5 Largest Clusters")
                for cluster_id, size in cluster_sizes.head(5).items():
                    cluster_confidence = df_results[df_results['cluster_id'] == cluster_id]['certainty'].mean()
                    st.write(f"**Cluster {cluster_id}:** {size} documents ({cluster_confidence:.0%} confidence)")

                if len(cluster_sizes) > 5:
                    with st.expander(f"📋 View All {len(cluster_sizes)} Clusters"):
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
                        st.dataframe(cluster_df, hide_index=True, use_container_width=True)

                # Confidence distribution
                st.markdown("### Confidence Distribution")
                high_conf = len(df_results[df_results['certainty'] >= 0.8])
                med_conf = len(df_results[(df_results['certainty'] >= 0.6) & (df_results['certainty'] < 0.8)])
                low_conf = len(df_results[df_results['certainty'] < 0.6])
                st.write(f"🟢 **High confidence (≥80%):** {high_conf} documents ({high_conf / total_docs:.1%})")
                st.write(f"🟡 **Medium confidence (60-79%):** {med_conf} documents ({med_conf / total_docs:.1%})")
                st.write(f"🔴 **Low confidence (<60%):** {low_conf} documents ({low_conf / total_docs:.1%})")

            with col2:
                # Performance summary
                st.markdown("### 🚀 Performance Summary")
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

                # Charts
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
                # CSV download
                df_export = pd.DataFrame(st.session_state.clustered_data)
                csv_data = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name="text_clustering_results.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
                # JSON download
                json_data = []
                for doc in st.session_state.clustered_data:
                    # Ensure all values are JSON-serializable
                    doc_copy = doc.copy()
                    for k, v in doc_copy.items():
                        if isinstance(v, (np.integer, np.floating)):
                            doc_copy[k] = float(v) if isinstance(v, np.floating) else int(v)
                        elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
                            doc_copy[k] = str(v)
                    json_data.append(doc_copy)
                json_str = json.dumps(json_data, indent=2)
                st.download_button(
                    label="📄 Download as JSON",
                    data=json_str,
                    file_name="text_clustering_results.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col2:
                st.markdown("### Analysis Summary")

                total_docs = len(df_results)
                num_clusters = df_results['cluster_id'].nunique()

                st.write(f"**Total Documents:** {total_docs:,}")
                st.write(f"**Clusters Found:** {num_clusters}")
                st.write(f"**Avg Cluster Size:** {total_docs / num_clusters:.1f} docs/cluster")

                if 'batch_id' in df_results.columns:
                    num_batches = df_results['batch_id'].nunique()
                    st.write(f"**Batches Used:** {num_batches}")

                st.markdown("### Start New Analysis")
                if st.button("🔄 Analyze Different Documents", use_container_width=True):
                    st.session_state.clustered_data = None
                    st.session_state.view_mode = 'overview'
                    st.session_state.processing = False
                    st.rerun()

        # Bottom stats bar
        with st.container():
            cols = st.columns(4)

            total_docs = len(df_results)
            num_clusters = df_results['cluster_id'].nunique()
            avg_confidence = df_results['certainty'].mean()
            high_confidence = len(df_results[df_results['certainty'] >= 0.8])

            with cols[0]:
                st.metric("📄 Documents", f"{total_docs:,}")
            with cols[1]:
                st.metric("🗂️ Clusters", num_clusters)
            with cols[2]:
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.0%}")
            with cols[3]:
                st.metric("✨ High Confidence", f"{high_confidence} ({high_confidence / total_docs:.0%})")


if __name__ == "__main__":
    main()