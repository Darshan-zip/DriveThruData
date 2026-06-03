import ollama
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

dense_index_name = "dense-vectors"

if not pc.has_index(dense_index_name):
    pc.create_index_for_model(
        name=dense_index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

dense_index = pc.Index(dense_index_name)

records = []
sents=dataset.split(".")
for i,sent in enumerate(sents):
  if not sent:
    continue
  records.append({
    "_id":f"rec{i}",
    "chunk_text":sent
  })

dense_index.upsert_records(records=records, namespace="sample-namespace")


sparse_index_name = "sparse-vectors"

if not pc.has_index(sparse_index_name):
    pc.create_index_for_model(
        name=sparse_index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"pinecone-sparse-english-v0",
            "field_map":{"text": "chunk_text"}
        }
    )
sparse_index = pc.Index(sparse_index_name)
sparse_index.upsert_records(records=records, namespace="sample-namespace")

# do full document indexing before this

"""# Search the index and rerank results
reranked_results = dense_index.search(
    namespace="example-namespace",
    query={
        "top_k": 10,
        "inputs": {
            'text': query
        }
    },
    rerank={
        "model": "bge-reranker-v2-m3",
        "top_n": 10,
        "rank_fields": ["chunk_text"]
    }   
)

# Print the reranked results
for hit in reranked_results['result']['hits']:
    print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")
"""


from pinecone.preview import SchemaBuilder

schema = (
    SchemaBuilder()
      .add_string_field(name="title", full_text_search={"language": "en"})
      .add_string_field(name="body", full_text_search={"language": "en", "stemming": True})
      .build()
) 
if not pc.has_index(sparse_index_name):
    index_model = pc.preview.indexes.create(
        name="articles",
        schema=schema,
        read_capacity={"mode": "OnDemand"},
    )
doc_index_model = pc.preview.index(name="articles")
host = doc_index_model.host

NAMESPACE = 'sample-namespace'

docs = [
    {
        "_id": "doc1",
        "title": "Machine learning in 2024",
        "body": "Machine learning models are revolutionizing natural language processing",
        "category": "technology",
        "year": 2024,
    },
    {
        "_id": "doc2",
        "title": "Vector databases",
        "body": "Vector databases enable fast similarity search across embeddings",
        "category": "technology",
        "year": 2023,
    },
    {
        "_id": "doc3",
        "title": "Quantum computing",
        "body": "Quantum computers leverage superposition for faster computation",
        "category": "science",
        "year": 2024,
    },
]

# batch_upsert splits docs into parallel requests — use it for large sets.
# For small batches (≤1000 docs), index.documents.upsert(namespace=..., documents=...) is simpler.
'''doc_index_model.documents.batch_upsert(
    namespace=NAMESPACE,
    documents=docs,
    batch_size=50,
    max_concurrency=4,
    show_progress=True,
)'''

#query = "Machine Learning"



#print(sparse_results)


def merge_chunks(h1, h2):
    """Get the unique hits from two search results and return them as single array of {'_id', 'chunk_text'} dicts, printing each dict on a new line."""
    # Deduplicate by _id
    deduped_hits = {hit['id']: hit for hit in h1['result']['hits'] + h2['result']['hits']}.values()
    # Sort by _score descending
    sorted_hits = sorted(deduped_hits, key=lambda x: x['score'], reverse=True)
    # Transform to format for reranking
    result = [{'id': hit['id'], 'chunk_text': hit['fields']['chunk_text']} for hit in sorted_hits]
    return result






'''This is the working code for full text search retrieval
doc_response=doc_index_model.documents.search(
    namespace=NAMESPACE,
    top_k=5,
    score_by=[
        {
            "type": "text",
            "field": "body",
            "query":query
        }
    ],
    filter={
        "year": {"$gte": 2024},
        "body": {"$match_phrase": "natural language"},
    },
    include_fields=["title", "body", "category", "year"]
    
)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
for match in doc_response.matches:
    print(match._id, match._score, getattr(match, "title", ""))'''
    
    


def retrieve(query):
    dense_results = dense_index.search(
        namespace="sample-namespace",
        top_k=40,
        inputs={
            "text": query
        }
    )


    sparse_results = sparse_index.search(
        namespace="sample-namespace",
        top_k=40,
        inputs={
            "text": query
        }
    )
    merged_results = merge_chunks(sparse_results, dense_results)

    #print('[\n   ' + ',\n   '.join(str(obj) for obj in merged_results) + '\n]')
    
    result = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=merged_results,
        rank_fields=["chunk_text"],
        top_n=10,
        return_documents=True,
        parameters={
            "truncate": "END"
        }
    )

    print("Query", query)
    print('-----')
    '''for row in result.data:
        print(f"{row['document']['id']} - {round(row['score'], 2)} - {row['document']['chunk_text']}")'''
    return '\n'.join([f'- {row["document"]["chunk_text"]}' for row in result.data])


def RAG(input_query):
    retreived_data=retrieve(input_query)

    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information:
    {retreived_data}'''

    LANGUAGE_MODEL='llama3'

    stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
    ],
    stream=True,
    )
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

RAG("tell me about Apples")