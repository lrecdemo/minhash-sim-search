import unicodedata
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

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

def calculate_optimal_batch_size(dataset_size: int) -> int:
    """Calculate optimal batch size based on dataset size"""
    if dataset_size <= 5000:
        return 1000  # Small datasets: smaller batches
    elif dataset_size <= 20000:
        return 2500  # Medium datasets
    elif dataset_size <= 100_000:
        return 5000  # Large datasets
    else:
        return 10000  # Very large datasets: biggest jumps
def display_cluster_graph(df: pd.DataFrame, cluster_id: int, threshold: float = 0.3):
    cluster_docs = df[df['cluster_id'] == cluster_id].copy()
    if len(cluster_docs) < 2:
        st.info("Not enough documents to visualize a graph.")
        return

    # Build graph
    G = nx.Graph()
    for _, doc in cluster_docs.iterrows():
        G.add_node(doc['original_index'], label=str(doc['original_index']), title=doc['text'][:200])

    # Add edges based on similarity
    for i, doc1 in cluster_docs.iterrows():
        for j, doc2 in cluster_docs.iterrows():
            if doc1['original_index'] >= doc2['original_index']:
                continue
            # Use certainty or other similarity metric if available
            # For demonstration, connect all docs in the cluster
            G.add_edge(doc1['original_index'], doc2['original_index'])

    # Create PyVis network
    net = Network(height="500px", width="100%", notebook=False, bgcolor="#f8f9fa", font_color="black")
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])

    # Render in Streamlit
    tmp_html = f"/tmp/cluster_{cluster_id}_graph.html"
    net.save_graph(tmp_html)
    with open(tmp_html, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=520)


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

    def __init__(self, num_perm: int = NUM_PERM, threshold: float = 0.3, shingle_size: int = 5):
        self.num_perm = num_perm
        self.threshold = threshold
        self.batch_size = BATCH_SIZE
        self.max_workers = MAX_WORKERS
        self.shingle_size = shingle_size  # Add this parameter

    # Keep all existing methods the same, but update these two:

    @lru_cache(maxsize=10000)
    def generate_character_shingles_cached(self, text_hash: int, text: str, k: int) -> frozenset:
        """Generate character n-grams (true shingles)."""
        if len(text) < k:
            return frozenset([text]) if text else frozenset()

        shingles = set()
        for i in range(len(text) - k + 1):
            shingle = text[i:i + k]  # Character-based
            shingles.add(hash(shingle))
        return frozenset(shingles)

    def generate_shingles_fast(self, text: str) -> Set[int]:
        """Fast character shingle generation with caching."""
        text_hash = hash(text)
        cached_result = self.generate_character_shingles_cached(text_hash, text, self.shingle_size)
        return set(cached_result)


    def preprocess_text_vectorized(self, texts: List[str]) -> List[str]:
        """Vectorized text preprocessing using pandas."""
        if len(texts) == 1:
            text = texts[0].lower()
            text = unicodedata.normalize('NFC', str(text))
            text = CLEAN_PATTERN.sub(' ', text)
            text = WHITESPACE_PATTERN.sub(' ', text)
            return [text.strip()]

        df = pd.DataFrame({'text': texts})
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].str.replace(CLEAN_PATTERN, ' ', regex=True)
        df['text'] = df['text'].str.replace(WHITESPACE_PATTERN, ' ', regex=True)
        df['text'] = df['text'].str.strip()

        return df['text'].tolist()

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


# ===== ADD THESE IMPORTS AT THE TOP (after your existing imports) =====
import tempfile
import shutil
import sqlite3
import pickle
import os


# ===== ADD THIS COMPLETE STREAMING CLASS (after OptimizedMinHashLSHClustering) =====

class StreamingMinHashLSHClustering:
    """
    Streaming clustering for very large datasets (100K+ documents)
    Uses disk storage to handle datasets that don't fit in memory
    """

    def __init__(self, threshold: float = 0.3, shingle_size: int = 4,
                 chunk_size: int = 5000, num_perm: int = 64, temp_dir: str = None):
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.chunk_size = chunk_size
        self.num_perm = num_perm
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "clustering.db")
        self.doc_count = 0
        self.all_minhashes = {}
        self.lsh_index = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for storing signatures"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signatures (
                    doc_id INTEGER PRIMARY KEY,
                    signature BLOB
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Database initialization error: {e}")

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = str(text).lower().strip()
        cleaned = ''.join(c if c.isalnum() else ' ' for c in text)
        return ' '.join(cleaned.split())

    def generate_shingles(self, text: str):
        """Generate character shingles"""
        if len(text) < self.shingle_size:
            return [hash(text)] if text else []
        shingles = []
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i:i + self.shingle_size]
            shingles.append(hash(shingle))
        return shingles

    def create_minhash(self, shingles):
        """Create MinHash from shingles"""
        mh = MinHash(num_perm=self.num_perm)
        if not shingles:
            mh.update(b'empty_document')
        else:
            for sh in shingles:
                mh.update(str(sh).encode('utf-8'))
        return mh

    def process_chunk(self, texts):
        """Process a chunk of texts"""
        chunk_minhashes = {}
        for text in texts:
            try:
                clean_text = self.preprocess_text(text)
                shingles = self.generate_shingles(clean_text)
                mh = self.create_minhash(shingles)

                doc_id = self.doc_count
                self.doc_count += 1

                chunk_minhashes[doc_id] = mh
                self.lsh_index.insert(str(doc_id), mh)
                self.all_minhashes[doc_id] = mh

            except Exception as e:
                st.warning(f"Error processing document {self.doc_count}: {e}")
                self.doc_count += 1
                continue

        return chunk_minhashes

    def cluster_streaming_optimized(self, texts, progress_callback=None):
        """Main streaming clustering function with progress reporting"""
        try:
            chunk = []
            chunk_count = 0
            processed = 0

            for text in texts:
                chunk.append(text)

                if len(chunk) >= self.chunk_size:
                    if progress_callback:
                        progress_callback(
                            f"Processing streaming chunk {chunk_count + 1}",
                            processed / len(texts),
                            processed,
                            0
                        )

                    self.process_chunk(chunk)
                    processed += len(chunk)
                    chunk = []
                    chunk_count += 1

            # Process remaining documents
            if chunk:
                self.process_chunk(chunk)
                processed += len(chunk)

            if progress_callback:
                progress_callback("Performing streaming clustering", 0.9, processed, 0)

            # Perform final clustering
            return self._cluster_all(processed)

        except Exception as e:
            st.error(f"Streaming clustering error: {e}")
            return {i: i for i in range(self.doc_count)}

    def _cluster_all(self, num_docs):
        """Perform Union-Find clustering"""
        try:
            parent = list(range(num_docs))

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[py] = px

            similarity_pairs = 0
            for doc_id in range(num_docs):
                try:
                    if doc_id in self.all_minhashes:
                        mh = self.all_minhashes[doc_id]
                        candidates = self.lsh_index.query(mh)
                        for candidate_str in candidates:
                            candidate_id = int(candidate_str)
                            if candidate_id != doc_id and candidate_id < num_docs:
                                union(doc_id, candidate_id)
                                similarity_pairs += 1
                except Exception:
                    continue

            clusters = {}
            cluster_map = {}
            cluster_idx = 0

            for doc_id in range(num_docs):
                root = find(doc_id)
                if root not in cluster_map:
                    cluster_map[root] = cluster_idx
                    cluster_idx += 1
                clusters[doc_id] = cluster_map[root]

            return clusters

        except Exception as e:
            st.error(f"Final clustering error: {e}")
            return {i: 0 for i in range(num_docs)}

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            st.warning(f"Cleanup error: {e}")



def run_clustering_analysis(texts: List[str], threshold: float, shingle_size: int,
                            progress_placeholder, clustering_method: str = "optimized"):
    logger = PerformanceLogger()
    logger.start()
    logger.log("Initializing clustering service")

    # Choose clustering method based on dataset size and user preference
    if clustering_method == "streaming" or len(texts) > 100_000:
        clustering_service = StreamingMinHashLSHClustering(
            threshold=threshold,
            shingle_size=shingle_size,
            chunk_size=min(5000, max(1000, len(texts) // 20)),
            num_perm=64
        )
        is_streaming = True
        st.info(f"Using streaming clustering for {len(texts):,} documents")
    else:
        clustering_service = OptimizedMinHashLSHClustering(
            threshold=threshold,
            shingle_size=shingle_size
        )
        is_streaming = False
        st.info(f"Using optimized clustering for {len(texts):,} documents")

    progress_bar = progress_placeholder.progress(0)
    status_text = progress_placeholder.empty()
    metrics_container = progress_placeholder.container()

    optimal_batch_size = calculate_optimal_batch_size(len(texts))
    if not is_streaming:
        clustering_service.batch_size = optimal_batch_size
        st.info(f"Using batch size: {optimal_batch_size:,} documents")

    def progress_callback(stage, progress, processed, clusters):
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Status: {stage}")
        logger.update_peak_memory()
        logger.update_peak_cpu()
        if processed > 0:
            with metrics_container:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processed", f"{processed:,}/{len(texts):,}")
                with col2:
                    st.metric("Method", "Streaming" if is_streaming else "Optimized")
                with col3:
                    rate = processed / max(1, (time.time() - st.session_state.start_time))
                    st.metric("Rate", f"{rate:.1f} docs/sec")

    st.session_state.start_time = time.time()
    try:
        logger.log(f"Processing {len(texts):,} documents with threshold {threshold} and shingle size {shingle_size}")

        if is_streaming:
            # For streaming, we get a dictionary of clusters
            cluster_results = clustering_service.cluster_streaming_optimized(texts, progress_callback)

            # Convert to ClusteredDocument format
            clustered_docs = []
            for i, text in enumerate(texts):
                cluster_id = cluster_results.get(i, 0)
                clustered_docs.append(ClusteredDocument(
                    text=text,
                    cluster_id=cluster_id,
                    certainty=0.85,  # Default certainty for streaming
                    original_index=i,
                    batch_id=i // clustering_service.chunk_size
                ))
        else:
            # Original optimized clustering
            clustered_docs = clustering_service.cluster_documents_optimized(texts, progress_callback)

        logger.log(f"Clustering completed for {len(clustered_docs)} documents")

        # Create result with original columns preserved
        result = []
        original_df = st.session_state.get('file_data')

        for i, doc in enumerate(clustered_docs):
            # Get the original row data
            original_row = original_df.iloc[doc.original_index].to_dict()

            # Get document ID
            doc_id = original_row.get(st.session_state.selected_id_column, doc.original_index) \
                if st.session_state.selected_id_column else doc.original_index

            # Create result dict starting with all original columns
            result_row = original_row.copy()

            # Add clustering results
            result_row.update({
                "id": doc_id,
                "cluster_id": doc.cluster_id,
                "certainty": round(doc.certainty, 4),
                "original_index": doc.original_index,
                "batch_id": doc.batch_id,
                "clustering_method": "streaming" if is_streaming else "optimized"
            })

            result.append(result_row)

        # Cleanup streaming resources
        if hasattr(clustering_service, 'cleanup'):
            clustering_service.cleanup()

        logger.log(f"Conversion to result format completed")
        return result, logger

    except Exception as e:
        # Cleanup on error
        if hasattr(clustering_service, 'cleanup'):
            clustering_service.cleanup()
        st.error(f"Clustering failed: {str(e)}")
        return None, logger


# ===== REPLACE THE CLUSTERING SETTINGS SECTION IN main() =====



# ===== ADD TO STATISTICS TAB =====
# In the Statistics tab, after the Overview section, add:

# Add clustering method information
# st.markdown("### Clustering Method")
# clustering_method = "unknown"
# if st.session_state.clustered_data:
#     first_doc = st.session_state.clustered_data[0]
#     clustering_method = first_doc.get('clustering_method', 'optimized')
#
# if clustering_method == "streaming":
#     st.write("🌊 **Streaming clustering** - Used disk storage for large dataset processing")
#     st.write("✅ **Benefits**: Can handle unlimited dataset sizes, constant memory usage")
#     st.write("⚠️ **Trade-offs**: Slower due to disk I/O, but much more scalable")
# else:
#     st.write("⚡ **Optimized clustering** - Used in-memory processing")
#     st.write("✅ **Benefits**: Faster processing, full similarity detection")
#     st.write("⚠️ **Trade-offs**: Limited by available RAM")


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
    # Compute cluster stats
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

    # --- FILTERS in an expandable section ---
    with st.expander("⚙️ Filter Clusters", expanded=False):
        min_size, max_size = int(cluster_stats['size'].min()), int(cluster_stats['size'].max())
        min_conf, max_conf = float(cluster_stats['avg_confidence'].min()), float(cluster_stats['avg_confidence'].max())

        col1, col2 = st.columns(2)
        with col1:
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
        with col2:
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

    # Apply filters
    filtered_clusters = cluster_stats[
        (cluster_stats['size'] >= selected_min_size) &
        (cluster_stats['size'] <= selected_max_size) &
        (cluster_stats['avg_confidence'] >= selected_min_conf) &
        (cluster_stats['avg_confidence'] <= selected_max_conf)
        ]

    total_clusters = len(filtered_clusters)
    if total_clusters == 0:
        st.warning("No clusters match the filter criteria.")
        return

    # Pagination
    clusters_per_page = 10
    total_pages = (total_clusters - 1) // clusters_per_page + 1
    current_page = st.selectbox(
        "",
        options=range(1, total_pages + 1),
        format_func=lambda
            x: f"Page {x} (Clusters {(x - 1) * clusters_per_page + 1}-{min(x * clusters_per_page, total_clusters)})"
    )
    start_idx = (current_page - 1) * clusters_per_page
    end_idx = min(start_idx + clusters_per_page, total_clusters)
    clusters_to_show = filtered_clusters.iloc[start_idx:end_idx]

    # Display clusters
    for _, row in clusters_to_show.iterrows():
        cluster_id = row['cluster_id']
        cluster_size = row['size']
        avg_confidence = row['avg_confidence']
        batch_info = f" • Spans {row['batches']} batch{'es' if row['batches'] != 1 else ''}" if 'batches' in row else ""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div class="cluster-header">
                <h3 style="margin: 0; color: white;">Cluster {cluster_id}</h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9; color: white;">
                    {cluster_size} documents • {avg_confidence:.0%} avg confidence{batch_info}
                </p>
            </div>
            """, unsafe_allow_html=True)

            cluster_docs = df[df['cluster_id'] == cluster_id]
            top_docs = cluster_docs.nlargest(3, 'certainty')
            for idx, (_, doc) in enumerate(top_docs.iterrows()):
                # Convert doc to dictionary if it's not already
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
                    <strong>Preview {idx + 1}:</strong> {doc_id}: {text_preview}... <span style="color: #667eea;">{get_confidence_emoji_and_text(certainty)}</span>
                </div>
                """, unsafe_allow_html=True)

            if st.button(f"View All {cluster_size} Documents", key=f"view_cluster_{cluster_id}"):
                st.session_state.view_mode = 'cluster'
                st.session_state.selected_cluster = cluster_id
                st.rerun()

            st.markdown("---")


import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any


def create_document_graph(docs_df: pd.DataFrame, similarity_threshold: float = 0.3) -> Dict[str, Any]:
    """Create a network graph of documents based on text similarity."""

    # Calculate TF-IDF similarity between documents
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))

    try:
        tfidf_matrix = vectorizer.fit_transform(docs_df['text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
    except:
        # Fallback for very small datasets
        similarity_matrix = np.ones((len(docs_df), len(docs_df)))

    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes (documents)
    for idx, doc in docs_df.iterrows():
        doc_id = doc.get('id', doc['original_index'])
        G.add_node(doc_id,
                   text=doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text'],
                   full_text=doc['text'],
                   confidence=doc['certainty'],
                   original_index=doc['original_index'],
                   batch_id=doc.get('batch_id', 'Unknown'))

    # Add edges based on similarity
    doc_indices = docs_df.index.tolist()
    for i, idx1 in enumerate(doc_indices):
        for j, idx2 in enumerate(doc_indices):
            if i < j and similarity_matrix[i][j] > similarity_threshold:
                doc_id1 = docs_df.loc[idx1].get('id', docs_df.loc[idx1]['original_index'])
                doc_id2 = docs_df.loc[idx2].get('id', docs_df.loc[idx2]['original_index'])
                G.add_edge(doc_id1, doc_id2, weight=similarity_matrix[i][j])

    return G


def create_plotly_network(G: nx.Graph, layout_type: str = "spring") -> go.Figure:
    """Create a Plotly network visualization similar to Neo4j Bloom."""

    # Calculate layout positions
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=3, iterations=50)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G) if len(G.nodes()) > 1 else nx.spring_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Extract node and edge information
    node_x, node_y = [], []
    node_text, node_info, node_colors, node_sizes = [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Node information
        node_data = G.nodes[node]
        confidence = node_data.get('confidence', 0)
        text_preview = node_data.get('text', 'No text')
        batch_id = node_data.get('batch_id', 'Unknown')

        node_text.append(f"Doc #{node}")
        node_info.append(f"Document #{node}<br>"
                         f"Confidence: {confidence:.1%}<br>"
                         f"Batch: {batch_id}<br>"
                         f"Preview: {text_preview}")

        # Color by confidence level
        if confidence >= 0.8:
            node_colors.append('rgba(46, 125, 50, 0.8)')  # Green
        elif confidence >= 0.6:
            node_colors.append('rgba(255, 193, 7, 0.8)')  # Yellow
        else:
            node_colors.append('rgba(244, 67, 54, 0.8)')  # Red

        # Size by number of connections
        node_sizes.append(10 + len(list(G.neighbors(node))) * 5)

    # Create edge traces
    edge_x, edge_y = [], []
    edge_info = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        weight = G.edges[edge].get('weight', 0)
        edge_info.append(f"Similarity: {weight:.2f}")

    # Create the figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                             line=dict(width=1, color='rgba(128, 128, 128, 0.5)'),
                             hoverinfo='none',
                             mode='lines',
                             showlegend=False))

    # Add nodes
    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                             mode='markers+text',
                             marker=dict(size=node_sizes,
                                         color=node_colors,
                                         line=dict(width=2, color='white')),
                             text=node_text,
                             textposition="middle center",
                             textfont=dict(size=10, color='white'),
                             hovertemplate='%{customdata}<extra></extra>',
                             customdata=node_info,
                             showlegend=False))

    # Update layout for Neo4j Bloom-like appearance
    fig.update_layout(
        title="",
        title_x=0.5,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="Node size = number of connections | Color = confidence level",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(20, 20, 30, 0.9)',
        paper_bgcolor='rgba(20, 20, 30, 0.9)',
        font=dict(color='white')
    )

    return fig


def display_cluster_details(df: pd.DataFrame, cluster_id: int):
    """Display all documents in a specific cluster with graph visualization."""
    cluster_docs = df[df['cluster_id'] == cluster_id].copy()

    # Header section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"# Cluster {cluster_id} - All Documents")
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

    # View mode selection
    view_mode = st.radio(
        "",
        ["List View", "Graph View"],
        horizontal=True,
        key=f"view_mode_cluster_{cluster_id}"
    )

    if view_mode == "Graph View":
        # Graph visualization section
        # st.markdown("### Document Relationship Network")

        # Graph controls
        graph_col1, graph_col2, graph_col3 = st.columns([2, 2, 2])

        with graph_col1:
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.1,
                max_value=0.8,
                value=0.3,
                step=0.05,
                help="Higher values show fewer, stronger connections",
                key=f"similarity_threshold_{cluster_id}"
            )

        with graph_col2:
            layout_type = st.selectbox(
                "Layout Algorithm",
                ["spring", "circular", "kamada_kawai"],
                format_func=lambda x: {
                    "spring": "Force-directed",
                    "circular": "Circular",
                    "kamada_kawai": "Kamada-Kawai"
                }[x],
                key=f"layout_type_{cluster_id}"
            )

        with graph_col3:
            if st.button("🔄 Regenerate Graph", key=f"regen_graph_{cluster_id}"):
                st.rerun()

        # Create and display graph
        if len(cluster_docs) > 1:
            try:
                with st.spinner("Creating network graph..."):
                    G = create_document_graph(cluster_docs, similarity_threshold)
                    fig = create_plotly_network(G, layout_type)
                    st.plotly_chart(fig, use_container_width=True, height=600)

                # Graph statistics
                st.markdown("#### 📈 Network Statistics")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                with stats_col1:
                    st.metric("Nodes", len(G.nodes()))
                with stats_col2:
                    st.metric("Edges", len(G.edges()))
                with stats_col3:
                    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                    st.metric("Avg Connections", f"{avg_degree:.1f}")
                with stats_col4:
                    density = nx.density(G) if len(G.nodes()) > 1 else 0
                    st.metric("Network Density", f"{density:.2f}")

            except Exception as e:
                st.error(f"Error creating graph visualization: {str(e)}")
                st.info("Falling back to list view...")
                view_mode = "List View"
        else:
            st.info("Graph view requires at least 2 documents. Showing list view instead.")
            view_mode = "List View"

    if view_mode == "List View":
        # Original list view functionality
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
                format_func=lambda x:
                {"confidence": "Confidence", "original": "Original Order", "length": "Text Length"}[x],
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
        st.markdown(f"### Documents ({len(page_docs)} of {len(filtered_docs)} shown)")
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


def get_confidence_level(certainty: float) -> str:
    """Helper function to get confidence level description."""
    if certainty >= 0.8:
        return "High"
    elif certainty >= 0.6:
        return "Medium"
    else:
        return "Low"


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


def create_export_dataframe(clustered_data: List[Dict]) -> pd.DataFrame:
    """Create a DataFrame for export with all original columns plus clustering results."""
    df = pd.DataFrame(clustered_data)

    # Reorder columns to put clustering results at the end
    clustering_cols = ['cluster_id', 'certainty', 'original_index', 'batch_id']
    other_cols = [col for col in df.columns if col not in clustering_cols]

    # Put clustering columns at the end
    ordered_cols = other_cols + clustering_cols
    df = df[ordered_cols]

    return df


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
    st.markdown("# CorpusClues")
    st.markdown("*Discover similar documents in your text collection using advanced MinHash LSH clustering*")

    # Main content
    if st.session_state.clustered_data is None and not st.session_state.processing:
        # Upload section
        st.markdown("## Upload Your Documents")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Make sure the texts are in a column with header 'text'.",
                type=['csv'],
                help="Your CSV must have a column named 'text' containing the documents to analyze. Optionally, you can select an ID column.",
                key="file_uploader"
            )
            if uploaded_file is not None:
                try:
                    # Store file data in session state immediately
                    if 'file_data' not in st.session_state or st.session_state.get(
                            'file_name') != uploaded_file.name:
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
                        # if len(df_preview) > 50000:
                        #     st.error("❌ Maximum 50,000 documents supported for Streamlit deployment")
                        #     st.session_state.file_data = None
                    else:
                        st.error("❌ No 'text' column found. Please ensure your CSV has a column named 'text'")
                        st.session_state.file_data = None
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.session_state.file_data = None

        with col2:
            with col2:
                st.markdown("**Clustering Settings**")

                # Add character pattern size configuration with user-friendly language
                pattern_size = st.selectbox(
                    "Text pattern size:",
                    options=[2, 3, 4, 5, 6, 7, 8],
                    index=2,  # Default to 4 (was 5 before)
                    format_func=lambda x: f"{x} characters",
                    key="shingle_size_select",
                    help="Size of character patterns used for comparison"
                )

                similarity_level = st.select_slider(
                    "Similarity threshold:",
                    options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    value=0.3,
                    format_func=lambda x: {
                        0.1: "0.1", 0.2: "0.2", 0.3: "0.3",
                        0.4: "0.4+", 0.5: "0.5", 0.6: "0.6",
                        0.7: "0.7", 0.8: "0.8", 0.9: "0.9"
                    }[x],
                    key="similarity_slider",
                    help="Minimum similarity required to group documents"
                )

                st.session_state.similarity_threshold = similarity_level
                st.session_state.shingle_size = pattern_size

                if st.session_state.get('file_data') is not None:
                    estimated_time = max(5, len(st.session_state.file_data) // 100)
                    st.caption(f"Estimated processing time: ~{estimated_time} seconds")

                    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                        st.session_state.processing = True
                        st.rerun()
    elif st.session_state.processing:
        # Processing section
        st.markdown("## Processing Your Documents")

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

            # Get both threshold and shingle size from session state
            threshold = st.session_state.get('similarity_threshold', 0.3)
            shingle_size = st.session_state.get('shingle_size', 5)

            st.info(
                f"Processing {len(texts):,} documents with {threshold} similarity threshold and {shingle_size}-character shingles...")

            # Create a single container for all progress updates
            progress_container = st.container()

            # Run clustering
            if 'clustering_started' not in st.session_state:
                st.session_state.clustering_started = True
                with st.spinner("Initializing clustering analysis..."):
                    clustering_method = st.session_state.get('clustering_method', 'optimized')
                    results, logger = run_clustering_analysis(texts, threshold, shingle_size, progress_container, clustering_method)
                if results:
                    st.session_state.clustered_data = results
                    st.session_state.view_mode = 'overview'
                    st.session_state.processing = False
                    st.success("✅ Analysis complete!")
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

                # MODIFIED: Create export DataFrame with all original columns
                df_export = create_export_dataframe(st.session_state.clustered_data)

                # Show preview of what will be exported
                st.markdown("#### Export Preview")
                st.markdown(f"**{len(df_export)} rows × {len(df_export.columns)} columns**")

                # Show column names
                with st.expander("📋 View All Export Columns"):
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

                # Sample preview
                st.dataframe(df_export.head(3), use_container_width=True)

                # CSV download
                csv_data = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Results (CSV)",
                    data=csv_data,
                    file_name="text_clustering_results_complete.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                    help="Downloads all original columns plus clustering results"
                )

                # JSON download
                json_data = []
                for _, row in df_export.iterrows():
                    # Ensure all values are JSON-serializable
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
                    label="📄 Download as JSON",
                    data=json_str,
                    file_name="text_clustering_results_complete.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Downloads all original columns plus clustering results in JSON format"
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

                st.markdown("### Export Options")

                # Option to export only clustering results
                clustering_only_df = df_results[
                    ['id', 'text', 'cluster_id', 'certainty', 'original_index', 'batch_id']].copy()
                clustering_csv = clustering_only_df.to_csv(index=False)

                st.download_button(
                    label="📊 Download Clustering Results Only",
                    data=clustering_csv,
                    file_name="clustering_results_only.csv",
                    mime="text/csv",
                    help="Downloads only the text and clustering results (original format)"
                )

                # Option to export specific clusters
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