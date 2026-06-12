from core.base_rag import BaseRAG
from core.config import driver, ollama_client, LANGUAGE_MODEL
from typing import List, Dict, Any

class GraphRAG(BaseRAG):
    """Knowledge Graph RAG using Neo4j for relationship-based retrieval."""
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        # Extract entities from query (Simplified: just using query as keyword)
        # In a full version, we would use LLM to extract nodes
        with driver.session() as session:
            result = session.run(
                "MATCH (n)-[r]->(m) WHERE n.name CONTAINS $query OR m.name CONTAINS $query "
                "RETURN n.name as source, type(r) as rel, m.name as target", 
                query=query
            )
            records = [rec.data() for rec in result]
            
        return [{"chunk_text": f"{r['source']} {r['rel']} {r['target']}"} for r in records]

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        context_text = "\n".join([doc.get('chunk_text', str(doc)) for doc in context])
        
        prompt = f'''You are a helpful chatbot. 
Use the following knowledge graph triplets to answer the query.
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
