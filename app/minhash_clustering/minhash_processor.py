from datasketch import MinHash
from typing import List


class MinHashProcessor:
    def __init__(self, num_perm: int = 64):
        self.num_perm = num_perm

    def create_minhash(self, shingles: List[int]) -> MinHash:
        mh = MinHash(num_perm=self.num_perm, seed=42)
        if not shingles:
            mh.update(b'empty_document')
        else:
            for shingle in shingles:
                mh.update(str(shingle).encode('utf-8'))
        return mh