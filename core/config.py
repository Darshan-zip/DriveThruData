import os
from dotenv import load_dotenv
from pinecone import Pinecone
import ollama
from neo4j import GraphDatabase

load_dotenv()

# API Keys & Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Shared Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
ollama_client = ollama

# Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Default Index Names
DENSE_INDEX_NAME = "dense-vectors"
SPARSE_INDEX_NAME = "sparse-vectors"
NAMESPACE = "sample-namespace"
LANGUAGE_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"
