from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from datasketch import MinHashLSH, MinHash
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from .preprocess_helper import preprocess_text
from .shingle_generator import ShingleGenerator
from .minhash_processor import MinHashProcessor
from .union_find import UnionFind
import multiprocessing
import math
import gc
import os
import random
import numpy as np

os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)


@dataclass
class DocumentData:
    doc_id: int
    text: str
    shingles: Set[int]
    minhash: MinHash
    batch_id: int


@dataclass
class ClusteredDocument:
    text: str
    cluster_id: int
    certainty: float
    original_index: int
    batch_id: int = 0


class ProgressTracker:
    def __init__(self, total_docs: int, progress_callback=None):
        self.total_docs = total_docs
        self.processed = 0
        self.similarities_found = 0
        self.callback = progress_callback

    def update(self, stage: str, processed_count: int, similarities_count: int = 0):
        self.processed = processed_count
        self.similarities_found = similarities_count
        progress = min(self.processed / self.total_docs, 1.0)

        if self.callback:
            self.callback(stage, progress, self.processed, self.similarities_found)


class BatchProcessor:
    def __init__(self, shingle_generator: ShingleGenerator,
                 minhash_processor: MinHashProcessor,
                 max_workers: int = None,
                 preprocess_options: Dict[str, bool] = None):
        self.shingle_generator = shingle_generator
        self.minhash_processor = minhash_processor
        self.preprocess_options = preprocess_options

        if max_workers is None:
            self.max_workers = multiprocessing.cpu_count()
        else:
            self.max_workers = max_workers

    def process_texts(self, texts: List[str], batch_start_idx: int) -> List[DocumentData]:
        batch_size = len(texts)

        if batch_size < 100:
            return self._process_sequential(texts, batch_start_idx)
        elif batch_size < 1000:
            return self._process_with_threads(texts, batch_start_idx, self.max_workers)
        else:
            return self._process_with_processes(texts, batch_start_idx)

    def _process_sequential(self, texts: List[str], batch_start_idx: int) -> List[DocumentData]:
        documents = []
        for i, text in enumerate(texts):
            clean_text = preprocess_text(text, self.preprocess_options)
            shingles = self.shingle_generator.generate_shingles(clean_text)
            minhash = self.minhash_processor.create_minhash(shingles)

            doc_id = batch_start_idx + i
            batch_id = batch_start_idx // 1000

            documents.append(DocumentData(
                doc_id=doc_id,
                text=text,
                shingles=set(shingles),
                minhash=minhash,
                batch_id=batch_id
            ))
        return documents

    def _process_with_threads(self, texts: List[str], batch_start_idx: int,
                              workers: int) -> List[DocumentData]:
        preprocessed = [preprocess_text(text, self.preprocess_options) for text in texts]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            shingle_sets = list(executor.map(
                self.shingle_generator.generate_shingles,
                preprocessed
            ))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            minhashes = list(executor.map(
                self.minhash_processor.create_minhash,
                shingle_sets
            ))

        documents = []
        for i, (text, shingles, minhash) in enumerate(zip(texts, shingle_sets, minhashes)):
            doc_id = batch_start_idx + i
            batch_id = batch_start_idx // 1000

            documents.append(DocumentData(
                doc_id=doc_id,
                text=text,
                shingles=set(shingles),
                minhash=minhash,
                batch_id=batch_id
            ))

        return documents

    def _process_with_processes(self, texts: List[str], batch_start_idx: int) -> List[DocumentData]:
        workers = max(2, self.max_workers // 2)

        preprocessed = [preprocess_text(text, self.preprocess_options) for text in texts]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            shingle_sets = list(executor.map(
                self.shingle_generator.generate_shingles,
                preprocessed,
                chunksize=max(1, len(preprocessed) // workers)
            ))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            minhashes = list(executor.map(
                self.minhash_processor.create_minhash,
                shingle_sets,
                chunksize=max(1, len(shingle_sets) // workers)
            ))

        documents = []
        for i, (text, shingles, minhash) in enumerate(zip(texts, shingle_sets, minhashes)):
            doc_id = batch_start_idx + i
            batch_id = batch_start_idx // 1000

            documents.append(DocumentData(
                doc_id=doc_id,
                text=text,
                shingles=set(shingles),
                minhash=minhash,
                batch_id=batch_id
            ))

        return documents


class SimilarityFinder:
    def __init__(self, threshold: float, num_perm: int, max_candidates: int = 500):
        self.threshold = threshold
        self.num_perm = num_perm
        self.max_candidates = max_candidates

    def find_within_batch(self, documents: List[DocumentData]) -> List[Tuple[int, int, float]]:
        if len(documents) <= 1:
            return []

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for doc in documents:
            lsh.insert(str(doc.doc_id), doc.minhash)

        similarities = []
        processed_pairs = set()

        for doc in documents:
            candidates = lsh.query(doc.minhash)

            candidate_count = 0
            for candidate_key in candidates:
                if candidate_count >= self.max_candidates:
                    break

                try:
                    candidate_id = int(candidate_key)
                    if candidate_id != doc.doc_id:
                        pair = tuple(sorted([doc.doc_id, candidate_id]))
                        if pair not in processed_pairs:
                            processed_pairs.add(pair)
                            candidate_doc = next(d for d in documents if d.doc_id == candidate_id)
                            similarity = doc.minhash.jaccard(candidate_doc.minhash)
                            if similarity >= self.threshold:
                                similarities.append((pair[0], pair[1], similarity))
                            candidate_count += 1
                except (ValueError, TypeError):
                    continue

        return similarities

    def find_cross_batch(self, new_documents: List[DocumentData],
                         global_lsh: MinHashLSH,
                         global_documents: Dict[int, DocumentData]) -> List[Tuple[int, int, float]]:
        similarities = []
        processed_pairs = set()
        min_new_id = min(d.doc_id for d in new_documents)

        for doc in new_documents:
            candidates = global_lsh.query(doc.minhash)

            candidate_count = 0
            for candidate_key in candidates:
                if candidate_count >= self.max_candidates:
                    break

                try:
                    candidate_id = int(candidate_key)
                    if candidate_id < min_new_id:
                        pair = tuple(sorted([doc.doc_id, candidate_id]))
                        if pair not in processed_pairs:
                            processed_pairs.add(pair)
                            candidate_doc = global_documents[candidate_id]
                            similarity = doc.minhash.jaccard(candidate_doc.minhash)
                            if similarity >= self.threshold:
                                similarities.append((pair[0], pair[1], similarity))
                            candidate_count += 1
                except (ValueError, TypeError, KeyError):
                    continue

        return similarities


class MemMinhashLSHClustering:
    def __init__(self, num_perm: int = 64, threshold: float = 0.3,
                 shingle_size: int = 5, batch_size: int = 5000,
                 max_workers: int = None,
                 max_candidates: int = 500,
                 preprocess_options: Dict[str, bool] = None):
        self.threshold = threshold
        self.batch_size = batch_size
        self.preprocess_options = preprocess_options

        self.shingle_generator = ShingleGenerator(shingle_size)
        self.minhash_processor = MinHashProcessor(num_perm)
        self.batch_processor = BatchProcessor(
            self.shingle_generator, self.minhash_processor, max_workers, preprocess_options
        )
        self.similarity_finder = SimilarityFinder(threshold, num_perm, max_candidates)
        self._all_similarities = []

    def cluster_documents(self, texts: List[str],
                          progress_callback=None) -> List[ClusteredDocument]:
        if len(texts) <= 10000:
            return self._cluster_single_batch(texts, progress_callback)
        else:
            return self._cluster_multi_batch(texts, progress_callback)

    def _cluster_single_batch(self, texts: List[str],
                              progress_callback=None) -> List[ClusteredDocument]:
        progress = ProgressTracker(len(texts), progress_callback)

        progress.update("Processing all documents in parallel", 0, 0)

        documents = self.batch_processor.process_texts(texts, 0)

        progress.update("Finding similarities", len(texts) // 2, 0)

        similarities = self.similarity_finder.find_within_batch(documents)

        progress.update("Clustering", len(texts), len(similarities))

        clusterer = UnionFind(len(texts))
        for doc1_id, doc2_id, _ in similarities:
            clusterer.union(doc1_id, doc2_id)

        cluster_assignments_dict = clusterer.get_cluster_assignments()
        cluster_assignments = [cluster_assignments_dict[i] for i in range(len(texts))]

        self._all_similarities = similarities

        clustered_docs = self._create_final_results(
            texts, cluster_assignments, similarities,
            {d.doc_id: d for d in documents}
        )

        progress.update("Complete!", len(texts), len(set(cluster_assignments)))
        return clustered_docs

    def _cluster_multi_batch(self, texts: List[str],
                             progress_callback=None) -> List[ClusteredDocument]:
        progress = ProgressTracker(len(texts), progress_callback)
        all_documents = {}
        all_similarities = []

        global_lsh = MinHashLSH(threshold=self.threshold,
                                num_perm=self.minhash_processor.num_perm)

        num_batches = math.ceil(len(texts) / self.batch_size)
        processed_count = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            progress.update(f"Processing batch {batch_idx + 1}/{num_batches}",
                            processed_count, len(all_similarities))

            batch_documents = self.batch_processor.process_texts(batch_texts, start_idx)

            batch_similarities = self.similarity_finder.find_within_batch(batch_documents)
            all_similarities.extend(batch_similarities)

            for doc in batch_documents:
                all_documents[doc.doc_id] = doc
                global_lsh.insert(str(doc.doc_id), doc.minhash)

            if batch_idx > 0:
                cross_similarities = self.similarity_finder.find_cross_batch(
                    batch_documents, global_lsh, all_documents
                )
                all_similarities.extend(cross_similarities)

            processed_count += len(batch_texts)

            if batch_idx % 3 == 0:
                gc.collect()

        progress.update("Performing final clustering", processed_count, len(all_similarities))

        clusterer = UnionFind(len(texts))
        for doc1_id, doc2_id, _ in all_similarities:
            clusterer.union(doc1_id, doc2_id)

        cluster_assignments_dict = clusterer.get_cluster_assignments()
        cluster_assignments = [cluster_assignments_dict[i] for i in range(len(texts))]

        self._all_similarities = all_similarities

        clustered_docs = self._create_final_results(
            texts, cluster_assignments, all_similarities, all_documents
        )

        progress.update("Complete!", len(texts), len(set(cluster_assignments)))
        return clustered_docs

    def _create_final_results(self, texts: List[str], clusters: List[int],
                              similarities: List[Tuple[int, int, float]],
                              documents: Dict[int, DocumentData]) -> List[ClusteredDocument]:
        similarity_dict = {}
        for doc1, doc2, sim in similarities:
            pair = tuple(sorted([doc1, doc2]))
            similarity_dict[pair] = sim

        clustered_docs = []
        for i, text in enumerate(texts):
            cluster_id = clusters[i]
            certainty = self._calculate_certainty(i, clusters, similarity_dict)
            batch_id = documents[i].batch_id if i in documents else i // self.batch_size

            clustered_docs.append(ClusteredDocument(
                text=text,
                cluster_id=cluster_id,
                certainty=certainty,
                original_index=i,
                batch_id=batch_id
            ))

        return clustered_docs

    def _calculate_certainty(self, doc_idx: int, clusters: List[int],
                             similarities: Dict[Tuple[int, int], float]) -> float:
        cluster_id = clusters[doc_idx]
        cluster_members = [i for i, cid in enumerate(clusters)
                           if cid == cluster_id and i != doc_idx]

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

    def get_all_similarities(self) -> List[Tuple[int, int, float]]:
        return self._all_similarities