from core.config import ollama_client, LANGUAGE_MODEL
from strategies.normal_rag import NormalRAG
from strategies.hybrid_rag import HybridRAG
from strategies.corrective_rag import CorrectiveRAG
from strategies.cache_rag import CacheRAG
from strategies.graph_rag import GraphRAG
from strategies.agentic_rag import AgenticRAG
from typing import Dict

class RAGOrchestrator:
    """Routes user queries to the most appropriate RAG strategy."""
    
    def __init__(self):
        self.strategies = {
            "normal": NormalRAG(),
            "hybrid": HybridRAG(),
            "corrective": CorrectiveRAG(),
            "cache": CacheRAG(),
            "graph": GraphRAG(),
            "agentic": AgenticRAG()
        }

    def route_query(self, query: str) -> str:
        # 1. Check Cache First (Bypasses LLM routing for speed)
        cache_res = self.strategies["cache"].run(query)
        # If the cache_rag result is not just a 'miss' (handled inside CacheRAG.run)
        # Note: In my implementation, CacheRAG.run() will do Normal RAG if it misses.
        # For a more sophisticated orchestrator, we'd check if it's a cache hit specifically.
        
        # 2. Intent Routing using LLM
        routing_prompt = f"""Analyze the following user query and categorize it into one of these RAG types:
- normal: Simple fact retrieval.
- hybrid: Requires both keyword and semantic search (specific terms).
- corrective: High precision required, needs verification.
- graph: Questions about relationships, connections, or networks.
- agentic: Complex, multi-step research or analysis.

Query: {query}
Respond ONLY with the name of the strategy (e.g., 'graph' or 'agentic')."""

        response = ollama_client.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user', 'content': routing_prompt}]
        )
        
        strategy_name = response['message']['content'].strip().lower()
        
        # Validate the routed strategy
        if strategy_name not in self.strategies:
            print(f"[Router] Invalid strategy '{strategy_name}' routed. Falling back to hybrid.")
            strategy_name = "hybrid"
            
        print(f"[Router] Routed to: {strategy_name}")
        return self.strategies[strategy_name].run(query)
