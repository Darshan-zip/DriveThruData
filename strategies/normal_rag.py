from core.base_rag import BaseRAG
from core.config import pc, ollama_client, DENSE_INDEX_NAME, NAMESPACE, LANGUAGE_MODEL
from typing import List, Dict, Any

class NormalRAG(BaseRAG):
    """Basic Dense Vector RAG using Pinecone."""
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        index = pc.Index(DENSE_INDEX_NAME)
        results = index.search(
            namespace=NAMESPACE,
            top_k=5,
            inputs={"text": query}
        )
        # Pinecone inference search returns a list of hits
        return results if isinstance(results, list) else results.get('hits', [])

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        context_text = "\n".join([doc.get('chunk_text', str(doc)) for doc in context])
        
        prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. If the context is empty or irrelevant, say "I do not have enough information".
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
