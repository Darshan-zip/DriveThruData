from core.base_rag import BaseRAG
from core.config import ollama_client, LANGUAGE_MODEL
from strategies.hybrid_rag import HybridRAG
from typing import List, Dict, Any

class AgenticRAG(BaseRAG):
    """Autonomous agent that can reason and iteratively refine retrieval."""
    
    def __init__(self):
        self.tool_rag = HybridRAG()

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        # The 'retrieval' in Agentic RAG is a reasoning loop.
        # We return an empty list because the 'generate' method handles the loop.
        return []

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        # Reasoning Loop: Think -> Act -> Observe -> Final Answer
        max_iterations = 3
        current_query = query
        collected_context = []
        
        for i in range(max_iterations):
            # 1. Think: What do I need to find?
            reasoning_prompt = f"Given the user query '{query}' and the context gathered so far: {collected_context}, what specific information is still missing? Provide a search query to find the missing info, or say 'FINISH' if you have enough info."
            
            res = ollama_client.chat(model=LANGUAGE_MODEL, messages=[{'role': 'user', 'content': reasoning_prompt}])
            thought = res['message']['content'].strip()
            
            if "FINISH" in thought.upper():
                break
            
            # 2. Act: Use the RAG tool to get more info
            print(f"[Agent] Iteration {i+1}: Searching for {thought}...")
            docs = self.tool_rag.retrieve(thought)
            collected_context.extend(docs)
            
        # 3. Final Synthesis
        context_text = "\n".join([doc.get('document', {}).get('chunk_text', str(doc)) for doc in collected_context])
        final_prompt = f"Based on the following gathered context, answer the user query: {query}\n\nContext:\n{context_text}"
        
        response = ollama_client.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user', 'content': final_prompt}]
        )
        return response['message']['content']
