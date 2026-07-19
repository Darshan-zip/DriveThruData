from core.base_rag import BaseRAG
from core.config import pc, ollama_client, DENSE_INDEX_NAME, SPARSE_INDEX_NAME, NAMESPACE, LANGUAGE_MODEL
from strategies.hybrid_rag import HybridRAG
from typing import List, Dict, Any

class CorrectiveRAG(BaseRAG):
    """RAG with a grading step and fallback to full-text search."""
    
    def __init__(self):
        self.hybrid = HybridRAG()

    def _grade_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        graded_docs = []
        for doc in documents:
            text = doc.get('document', {}).get('chunk_text', str(doc))
            prompt = f"Grade the following document based on its relevance to the query. Respond ONLY with 'Relevant' or 'Irrelevant'.\n\nQuery: {query}\nDocument: {text}"
            
            response = ollama_client.chat(
                model=LANGUAGE_MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )
            grade = response['message']['content'].strip().capitalize()
            if 'Relevant' in grade:
                graded_docs.append(doc)
        return graded_docs

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        # 1. Initial Hybrid Retrieval
        docs = self.hybrid.retrieve(query)
        
        # 2. Grading Step
        relevant_docs = self._grade_documents(query, docs)
        
        if not relevant_docs:
            # 3. Corrective Action: Fallback to basic search/full-text if nothing is relevant
            # In a real scenario, we'd call a web search or a different index
            print("[CRAG] No relevant docs found. Trying fallback retrieval...")
            # Re-trying with a wider search window as a simple fallback
            dense_index = pc.Index(DENSE_INDEX_NAME)
            fallback_docs = dense_index.search(namespace=NAMESPACE, top_k=10, inputs={"text": query})
            return fallback_docs.matches if hasattr(fallback_docs, 'matches') else fallback_docs
        
        return relevant_docs

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        # Use the same generation logic as HybridRAG
        return self.hybrid.generate(query, context)
