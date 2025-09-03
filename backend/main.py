from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import io
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datasketch import MinHashLSH, MinHash
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MinHash LSH Text Clustering", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class ClusteredDocument:
    text: str
    cluster_id: int
    certainty: float
    original_index: int


class MinHashLSHClustering:
    def __init__(self, num_perm: int = 128, threshold: float = 0.3):
        """
        Initialize MinHash LSH clustering.

        Args:
            num_perm: Number of permutation functions for MinHash
            threshold: Jaccard similarity threshold for LSH
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.documents = []
        self.minhashes = []

    def generate_shingles(self, text: str, k: int = 3) -> Set[str]:
        """Generate k-shingles from text."""
        # Clean and normalize text
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()

        if len(words) < k:
            # If text is too short, use individual words
            return set(words) if words else set([''])

        shingles = set()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i + k])
            shingles.add(shingle)

        return shingles

    def create_minhash(self, shingles: Set[str]) -> MinHash:
        """Create MinHash signature from shingles."""
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode('utf8'))
        return minhash

    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def cluster_documents(self, texts: List[str]) -> List[ClusteredDocument]:
        """Cluster documents using MinHash LSH."""
        logger.info(f"Starting clustering of {len(texts)} documents")

        # Reset state
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.documents = texts
        self.minhashes = []

        # Generate shingles and MinHashes
        document_shingles = []
        for i, text in enumerate(texts):
            shingles = self.generate_shingles(text)
            document_shingles.append(shingles)

            minhash = self.create_minhash(shingles)
            self.minhashes.append(minhash)

            # Insert into LSH with document index as key
            self.lsh.insert(str(i), minhash)

        # Find similar document pairs
        similar_pairs = set()
        similarities = {}

        for i in range(len(texts)):
            # Query LSH for similar documents
            candidates = self.lsh.query(self.minhashes[i])

            for candidate_key in candidates:
                candidate_idx = int(candidate_key)
                if candidate_idx != i:
                    # Calculate actual Jaccard similarity
                    jaccard_sim = self.jaccard_similarity(
                        document_shingles[i],
                        document_shingles[candidate_idx]
                    )

                    if jaccard_sim >= self.threshold:
                        pair = tuple(sorted([i, candidate_idx]))
                        similar_pairs.add(pair)
                        similarities[pair] = jaccard_sim

        logger.info(f"Found {len(similar_pairs)} similar pairs")

        # Perform clustering using Union-Find
        clusters = self._union_find_clustering(len(texts), similar_pairs)

        # Create clustered documents with certainty scores
        clustered_docs = []
        for i, text in enumerate(texts):
            cluster_id = clusters[i]
            certainty = self._calculate_certainty(
                i, clusters, similarities, document_shingles
            )

            clustered_docs.append(ClusteredDocument(
                text=text,
                cluster_id=cluster_id,
                certainty=certainty,
                original_index=i
            ))

        logger.info(f"Clustering completed. Created {len(set(clusters))} clusters")
        return clustered_docs

    def _union_find_clustering(self, n: int, similar_pairs: Set[Tuple[int, int]]) -> List[int]:
        """Perform clustering using Union-Find algorithm."""
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

        # Union similar documents
        for doc1, doc2 in similar_pairs:
            union(doc1, doc2)

        # Assign cluster IDs
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

    def _calculate_certainty(self, doc_idx: int, clusters: List[int],
                             similarities: Dict[Tuple[int, int], float],
                             document_shingles: List[Set[str]]) -> float:
        """Calculate certainty score for document's cluster assignment."""
        cluster_id = clusters[doc_idx]
        cluster_members = [i for i, cid in enumerate(clusters) if cid == cluster_id and i != doc_idx]

        if not cluster_members:
            return 1.0  # Singleton cluster, high certainty

        total_similarity = 0.0
        count = 0

        for member_idx in cluster_members:
            pair = tuple(sorted([doc_idx, member_idx]))
            if pair in similarities:
                total_similarity += similarities[pair]
                count += 1
            else:
                # Calculate similarity if not in cache
                sim = self.jaccard_similarity(
                    document_shingles[doc_idx],
                    document_shingles[member_idx]
                )
                total_similarity += sim
                count += 1

        return total_similarity / count if count > 0 else 0.5


# Global clustering instance
clustering_service = MinHashLSHClustering()


@app.get("/")
async def root():
    return {"message": "MinHash LSH Text Clustering API", "version": "1.0.0"}


@app.post("/api/cluster")
async def cluster_texts(
        file: UploadFile = File(...),
        jaccard_threshold: float = Form(...)
):
    """
    Cluster texts from uploaded CSV file.

    Args:
        file: CSV file with 'text' column
        jaccard_threshold: Similarity threshold (0.0-1.0)
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")

        # Validate threshold
        if not 0.0 <= jaccard_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Jaccard threshold must be between 0.0 and 1.0")

        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Find text column
        text_columns = [col for col in df.columns if col.lower().strip() == 'text']
        if not text_columns:
            raise HTTPException(status_code=400, detail="CSV must contain a column named 'text'")

        text_column = text_columns[0]
        texts = df[text_column].dropna().astype(str).tolist()

        if not texts:
            raise HTTPException(status_code=400, detail="No valid texts found in CSV")

        logger.info(f"Processing {len(texts)} texts with threshold {jaccard_threshold}")

        # Update clustering service with new threshold
        global clustering_service
        clustering_service = MinHashLSHClustering(threshold=jaccard_threshold)

        # Perform clustering
        clustered_docs = clustering_service.cluster_documents(texts)

        # Convert to dict format for JSON response
        result = []
        for doc in clustered_docs:
            result.append({
                "text": doc.text,
                "cluster_id": doc.cluster_id,
                "certainty": round(doc.certainty, 4),
                "original_index": doc.original_index
            })

        return result

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logger.error(f"Clustering error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/download")
async def download_results(results: List[Dict]):
    """
    Download clustering results as CSV.

    Args:
        results: List of clustered documents
    """
    try:
        # Create DataFrame from results
        df = pd.DataFrame(results)

        # Reorder columns for better presentation
        if not df.empty:
            column_order = ['text', 'cluster_id', 'certainty', 'original_index']
            df = df[column_order]

        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_data = output.getvalue()

        # Create streaming response
        def iter_csv():
            yield csv_data

        return StreamingResponse(
            iter_csv(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=clustered_results.csv"}
        )

    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating CSV: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "MinHash LSH Clustering"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)