from ollama import embed
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
load_dotenv()  
pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

with open("sample.txt", "r") as file:
    dataset = file.readlines()
dataset=" ".join(dataset)
#response = embed(model='nomic-embed-text', input='Hello, world!')
#print(len(response['embeddings'][0]))

index_name = "dense-vectors"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

index = pc.Index(index_name)

records = []
sents=dataset.split(".")
for i,sent in enumerate(sents):
  if not sent:
    continue
  records.append({
    "_id":f"rec{i}",
    "chunk_text":sent
  })

index.upsert_records(records=records, namespace="sample-namespace")
