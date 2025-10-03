import tempfile
import shutil
import sqlite3
import pickle
import os
import gc
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datasketch import MinHashLSH, MinHash
from .preprocess_helper import preprocess_text
from .shingle_generator import ShingleGenerator
from .minhash_processor import MinHashProcessor
from .union_find import UnionFind
import streamlit as st


@dataclass
class StreamingDocument:
    doc_id: int
    text: str
    minhash: MinHash
    chunk_id: int


class DataBaseManager:
    def __init__(self, temp_dir: str = None):
        if temp_dir is None:
            temp_dir = os.environ.get('CLUSTERING_TEMP_DIR', tempfile.mkdtemp(prefix='clustering_'))

        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "clustering.db")
        self.conn = None
        self._init_database()

    def _init_database(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=OFF")
            self.conn.execute("PRAGMA synchronous=OFF")
            self.conn.execute("PRAGMA cache_size=-64000")
            self.conn.execute("PRAGMA temp_store=MEMORY")
            self.conn.execute("PRAGMA locking_mode=EXCLUSIVE")
            self.conn.execute("PRAGMA page_size=8192")

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS signatures (
                    doc_id INTEGER PRIMARY KEY,
                    signature BLOB
                ) WITHOUT ROWID
            """)

            self.conn.commit()
        except Exception as e:
            st.error(f"Database initialization error: {e}")

    def load_signatures_batch(self, doc_ids: List[int]) -> Dict[int, MinHash]:
        if not doc_ids:
            return {}

        try:
            placeholders = ','.join('?' * len(doc_ids))
            cursor = self.conn.execute(
                f"SELECT doc_id, signature FROM signatures WHERE doc_id IN ({placeholders})",
                tuple(doc_ids)
            )
            results = {}
            for doc_id, sig_blob in cursor.fetchall():
                results[doc_id] = pickle.loads(sig_blob)
            return results
        except Exception as e:
            st.warning(f"Batch load error: {e}")
            return {}

    def store_signature_batch(self, doc_id: int, minhash: MinHash):
        try:
            signature_blob = pickle.dumps(minhash)
            self.conn.execute(
                "INSERT INTO signatures (doc_id, signature) VALUES (?, ?)",
                (doc_id, signature_blob)
            )
        except Exception as e:
            st.warning(f"Store error: {e}")

    def _flush_signatures(self):
        pass

    def flush_all_batches(self):
        try:
            if self.conn:
                self.conn.commit()
        except Exception as e:
            st.warning(f"Flush error: {e}")

    def load_signature(self, doc_id: int) -> Optional[MinHash]:
        try:
            cursor = self.conn.execute(
                "SELECT signature FROM signatures WHERE doc_id = ?", (doc_id,)
            )
            row = cursor.fetchone()
            if row:
                return pickle.loads(row[0])
            return None
        except Exception as e:
            st.warning(f"Database load error for doc {doc_id}: {e}")
            return None

    def cleanup(self):
        try:
            if self.conn:
                self.conn.close()
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            st.warning(f"Cleanup error: {e}")


class SimilarityFinder:
    def __init__(self, threshold: float, lsh_index: MinHashLSH, database: DataBaseManager):
        self.threshold = threshold
        self.lsh_index = lsh_index
        self.database = database

    def find_similar_documents(self, doc_id: int, minhash: MinHash) -> List[Tuple[int, int, float]]:
        similarities = []

        candidates = self.lsh_index.query(minhash)

        MAX_CANDIDATES = 150
        candidate_list = []
        for c in candidates:
            try:
                cid = int(c)
                if cid >= doc_id:
                    continue
                candidate_list.append(cid)
                if len(candidate_list) >= MAX_CANDIDATES:
                    break
            except (ValueError, TypeError):
                continue

        if not candidate_list:
            return similarities

        loaded = self.database.load_signatures_batch(candidate_list)

        for candidate_id in candidate_list:
            candidate_minhash = loaded.get(candidate_id)
            if not candidate_minhash:
                continue

            similarity = minhash.jaccard(candidate_minhash)

            if similarity >= self.threshold:
                similarities.append((candidate_id, doc_id, similarity))

        loaded.clear()
        del loaded

        return similarities

    def reset_seen_pairs(self):
        pass


class CertaintyCalculator:
    def __init__(self):
        self._certainty_cache = {}
        self._cache_max_size = 1000

    def calculate_certainty(self, doc_id: int, clusters: Dict[int, int],
                            similarities: List[Tuple[int, int, float]]) -> float:
        cache_key = (doc_id, len(similarities))
        if cache_key in self._certainty_cache:
            return self._certainty_cache[cache_key]

        cluster_id = clusters[doc_id]
        cluster_members = [i for i, cid in clusters.items() if cid == cluster_id and i != doc_id]

        if not cluster_members:
            result = 1.0
        else:
            similarity_dict = {}
            for doc1, doc2, sim in similarities:
                pair = tuple(sorted([doc1, doc2]))
                similarity_dict[pair] = sim

            total_similarity = 0.0
            count = 0

            for member_id in cluster_members:
                pair = tuple(sorted([doc_id, member_id]))
                if pair in similarity_dict:
                    total_similarity += similarity_dict[pair]
                    count += 1

            result = total_similarity / count if count > 0 else 0.5

        if len(self._certainty_cache) < self._cache_max_size:
            self._certainty_cache[cache_key] = result

        return result


class StreamingProgressTracker:
    def __init__(self, total_docs: int, progress_callback=None):
        self.total_docs = total_docs
        self.processed = 0
        self.similarities_found = 0
        self.callback = progress_callback

    def update(self, stage: str, processed_count: int, similarities_count: int = 0):
        self.processed = processed_count
        self.similarities_found = similarities_count

        if self.callback:
            progress = min(self.processed / self.total_docs, 1.0)
            self.callback(stage, progress, self.processed, self.similarities_found)


class ChunkProcessor:
    def __init__(self, shingle_generator: ShingleGenerator,
                 minhash_processor: MinHashProcessor,
                 similarity_finder: SimilarityFinder,
                 database: DataBaseManager,
                 preprocess_options: Dict[str, bool] = None):
        self.shingle_generator = shingle_generator
        self.minhash_processor = minhash_processor
        self.similarity_finder = similarity_finder
        self.database = database
        self.doc_count = 0
        self.preprocess_options = preprocess_options


class StreamingMinHashLSHClustering:
    def __init__(self, threshold: float = 0.3, shingle_size: int = 4,
                 chunk_size: int = 5000, num_perm: int = 64, temp_dir: str = None,
                 preprocess_options: Dict[str, bool] = None,
                 use_memory_cache: bool = True, progress_interval: int = 10000):
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.num_perm = num_perm
        self.preprocess_options = preprocess_options
        self.use_memory_cache = use_memory_cache
        self.progress_interval = progress_interval
        self.database = DataBaseManager(temp_dir)
        self.shingle_generator = ShingleGenerator(shingle_size)
        self.minhash_processor = MinHashProcessor(num_perm)
        self.lsh_index = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.similarity_finder = SimilarityFinder(threshold, self.lsh_index, self.database)
        self.chunk_processor = ChunkProcessor(
            self.shingle_generator, self.minhash_processor,
            self.similarity_finder, self.database,
            preprocess_options=self.preprocess_options
        )
        self.certainty_calculator = CertaintyCalculator()
        self._minhash_cache = None
        self._all_similarities = []

    def cluster_streaming(self, texts: List[str], progress_callback=None) -> Dict[int, int]:
        try:
            total_docs = len(texts)
            all_similarities = []
            processed_count = 0
            commit_interval = 10000
            gc_interval = 5000

            for i, text in enumerate(texts):
                clean_text = preprocess_text(text, self.preprocess_options)
                shingles = self.shingle_generator.generate_shingles(clean_text)
                minhash = self.minhash_processor.create_minhash(shingles)

                doc_id = self.chunk_processor.doc_count
                self.chunk_processor.doc_count += 1

                if doc_id > 0:
                    similarities = self.similarity_finder.find_similar_documents(doc_id, minhash)
                    all_similarities.extend(similarities)

                self.lsh_index.insert(str(doc_id), minhash)
                self.database.store_signature_batch(doc_id, minhash)

                del minhash, shingles, clean_text

                processed_count += 1

                if processed_count % commit_interval == 0:
                    self.database.flush_all_batches()

                if processed_count % self.progress_interval == 0:
                    if progress_callback:
                        progress_pct = processed_count / total_docs
                        progress_callback("Processing documents",
                                          progress_pct, processed_count, len(all_similarities))

                if processed_count % gc_interval == 0:
                    gc.collect()

            self.database.flush_all_batches()

            if progress_callback:
                progress_callback("Clustering", 0.0, total_docs, len(all_similarities))

            self._all_similarities = all_similarities

            clusterer = UnionFind(self.chunk_processor.doc_count)
            for doc1, doc2, _ in all_similarities:
                clusterer.union(doc1, doc2)

            cluster_assignments = clusterer.get_cluster_assignments()

            if progress_callback:
                progress_callback("Clustering", 1.0, total_docs, len(all_similarities))

            return cluster_assignments

        except Exception as e:
            st.error(f"Streaming clustering error: {e}")
            return {i: i for i in range(self.chunk_processor.doc_count)}

    def _calculate_certainty(self, doc_id: int, clusters: Dict[int, int],
                             similarities: List[Tuple[int, int, float]]) -> float:
        return self.certainty_calculator.calculate_certainty(doc_id, clusters, similarities)

    def cleanup(self):
        try:
            if hasattr(self, 'lsh_index'):
                self.lsh_index = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            if hasattr(self, 'database'):
                self.database.cleanup()
            if hasattr(self, 'chunk_processor'):
                self.chunk_processor.doc_count = 0
            if hasattr(self, 'similarity_finder'):
                self.similarity_finder.reset_seen_pairs()
            self._all_similarities = []
            gc.collect()
        except Exception as e:
            st.warning(f"Cleanup warning: {e}")

    def get_all_similarities(self) -> List[Tuple[int, int, float]]:
        return self._all_similarities