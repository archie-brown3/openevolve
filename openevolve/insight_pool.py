from collections import deque
from typing import Dict, List, Tuple
import numpy as np

class InsightPool:
    def __init__(self, maxlen: int = 30):
        # Use a deque to implement a bounded pool of insights
        self.pool = deque(maxlen=maxlen)
        # Initialize statistics for each insight
        self.stats: Dict[str, Dict[str, float]] = {}
        # Use a dictionary to store insight statistics for efficient lookup
        self.tags: Dict[str, List[str]] = {}

    def add_insight(self, insight: str, used_count: int, effectiveness: float, last_used_generation: int, tags: List[str]):
        # Calculate Jaccard similarity with existing insights to deduplicate
        overlap = self.calculate_overlap(insight)
        if overlap > 0.7:
            return  # Deduplicate if similarity is too high
        # Add insight to the pool and update statistics
        self.pool.append((insight, used_count, effectiveness, last_used_generation, tags))
        self.stats[insight] = {'used_count': used_count, 'effectiveness': effectiveness, 'last_used_generation': last_used_generation}
        # Update tags dictionary
        self.tags[insight] = tags

    def calculate_overlap(self, insight: str) -> float:
        # Calculate Jaccard similarity between the new insight and existing ones
        existing_insights = [i[0] for i in self.pool]
        return self.jaccard_similarity(insight, existing_insights)

    def jaccard_similarity(self, text1: str, text2: List[str]) -> float:
        # Calculate Jaccard similarity between two texts
        set1 = set(text1.split())
        set2 = set(' '.join(text2).split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    def get_insight_stats(self, insight: str):
        # Retrieve statistics for a specific insight
        return self.stats.get(insight, None)