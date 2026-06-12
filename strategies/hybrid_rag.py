from core.base_rag import BaseRAG
from core.config import pc, ollama_client, DENSE_INDEX_NAME, SPARSE_INDEX_NAME, NAMESPACE, LANGUAGE_MODEL
from typing import List, Dict, Any

class HybridRAG(BaseRAG):
    """Combines Dense and Sparse retrieval with Reranking."""
    
    def _merge_chunks(self, h1, h2):
        # Simplified merge logic based on the user's existing code
        combined = h1 + h2
        # Deduplicate and sort by a generic 'score' if available, else just return
        deduped = {hit.get('id', i): hit for i, hit in enumerate(combined)}.values()
        return list(deduped)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        dense_index = pc.Index(DENSE_INDEX_NAME)
        sparse_index = pc.Index(SPARSE_INDEX_NAME)
        
        dense_results = dense_index.search(
            namespace=NAMESPACE,
            top_k=20,
            inputs={"text": query}
        )
        
        sparse_results = sparse_index.search(
            namespace=NAMESPACE,
            top_k=20,
            inputs={"text": query}
        )
        
        merged = self._merge_chunks(dense_results, sparse_results)
        
        # Use Pinecone's built-in reranker
        reranked = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=[{"text": doc.get('chunk_text', str(doc))} for doc in merged],
            top_n=10,
            return_documents=True
        )
        
        return reranked.data

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        context_text = "\n".join([doc.get('document', {}).get('chunk_text', str(doc)) for doc in context])
        
        prompt = f'''You are a helpful chatbot.
Use the provided context to answer the query. If the information is not present, say "I don't know".
Context:
{context_text}
'''
        response = ollama_client.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': query},
            ]
        )
        return response['message']['content']
