from typing import List
from .deterministic_hash import deterministic_hash


class ShingleGenerator:
    def __init__(self, shingle_size: int):
        self.shingle_size = shingle_size

    def generate_shingles(self, text: str) -> List[int]:
        if len(text) < self.shingle_size:
            return [deterministic_hash(text)] if text else []

        num_shingles = len(text) - self.shingle_size + 1
        return [deterministic_hash(text[i:i + self.shingle_size])
                for i in range(num_shingles)]