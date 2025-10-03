from typing import Dict

class UnionFind:
    def __init__(self, num_documents: int):
        self.parent = list(range(num_documents))
        self.rank = [0] * num_documents

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_cluster_assignments(self) -> Dict[int, int]:
        clusters, cluster_map, cluster_idx = {}, {}, 0
        for doc_id in range(len(self.parent)):
            root = self.find(doc_id)
            if root not in cluster_map:
                cluster_map[root] = cluster_idx
                cluster_idx += 1
            clusters[doc_id] = cluster_map[root]
        return clusters