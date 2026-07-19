from flask import Flask, render_template, request, jsonify
from CRAG.main import RAG, retrieve, grade_documents, merge_chunks, dense_index, sparse_index, doc_index_model, NAMESPACE
import ollama
import io
from contextlib import redirect_stdout

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    
    # Capture prints from the main.py functions to provide logging (optional)
    f = io.StringIO()
    with redirect_stdout(f):
        # We need a modified version of RAG for the API that returns string
        # instead of just printing
        
        # Temporary internal implementation of RAG logic to return response
        retrieved_data = retrieve(query)
        relevant_docs = grade_documents(query, retrieved_data)
        
        if not relevant_docs:
            doc_response = doc_index_model.documents.search(
                namespace=NAMESPACE,
                top_k=5,
                score_by=[{"type": "text", "field": "body", "query": query}],
                include_fields=["body"]
            )
            relevant_docs = [{"document": {"chunk_text": match.body}} for match in doc_response.matches]
        
        context_text = '\n'.join([f"- {doc['document']['chunk_text']}" for doc in relevant_docs])
        instruction_prompt = f'''You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information. If the given pieces of context does not contain any information related to the user query, then just say that "I do not have enough information to answer this question":
        {context_text}'''

        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ],
        )
        answer = response['message']['content']
    
    return jsonify({
        'response': answer,
        'count': len(relevant_docs)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
