from core.base_rag import BaseRAG
from core.config import pc, ollama_client, DENSE_INDEX_NAME, NAMESPACE, LANGUAGE_MODEL
from typing import List, Dict, Any
import numpy as np

class CacheRAG(BaseRAG):
    """Semantic Cache RAG that stores previously answered queries."""
    
    def __init__(self):
        self.cache = {} # Simple in-memory cache: {embedding: response}
        self.threshold = 0.9 # Similarity threshold for cache hit

    def _get_embedding(self, text: str):
        # Simplified embedding using Ollama
        resp = ollama_client.embeddings(model="nomic-embed-text", prompt=text)
        return np.array(resp['embedding'])

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        # CacheRAG doesn't 'retrieve' docs in the traditional sense, 
        # it checks if the answer is already known.
        # We return a special marker or empty list if it's a cache hit
        query_emb = self._get_embedding(query)
        
        for cached_emb, response in self.cache.items():
            similarity = np.dot(query_emb, cached_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb))
            if similarity > self.threshold:
                return [{"cache_hit": True, "response": response}]
        
        return []

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        # If we found a cache hit, return it directly
        if context and context[0].get("cache_hit"):
            return context[0]["response"]
        
        # Otherwise, perform Normal RAG and store the result
        from strategies.normal_rag import NormalRAG
        normal = NormalRAG()
        docs = normal.retrieve(query)
        response = normal.generate(query, docs)
        
        # Store in cache
        self.cache[tuple(self._get_embedding(query))] = response
        return response
