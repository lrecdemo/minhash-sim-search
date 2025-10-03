import hashlib
import os
import random
import numpy as np

os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)

def deterministic_hash(text: str) -> int:
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)