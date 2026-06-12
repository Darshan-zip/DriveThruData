from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRAG(ABC):
    """Abstract Base Class for all RAG strategies."""
    
    @abstractmethod
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on the query."""
        pass

    @abstractmethod
    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate a response using the retrieved context."""
        pass

    def run(self, query: str) -> str:
        """Standard pipeline: Retrieve -> Generate."""
        context = self.retrieve(query)
        return self.generate(query, context)
