from flask import Flask, render_template, request, jsonify
from core.orchestrator import RAGOrchestrator

app = Flask(__name__)
orchestrator = RAGOrchestrator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat/<strategy>', methods=['POST'])
def chat_strategy(strategy):
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        if strategy not in orchestrator.strategies:
            return jsonify({'error': f'Strategy {strategy} not found'}), 404
        
        response = orchestrator.strategies[strategy].run(query)
        return jsonify({
            'response': response,
            'strategy': strategy
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/strategy-info/<strategy>')
def strategy_info(strategy):
    from core.config import DENSE_INDEX_NAME, SPARSE_INDEX_NAME
    
    info = {
        'normal': {
            'db': 'Pinecone (Dense)',
            'index': DENSE_INDEX_NAME,
            'samples': 'Semantic embeddings of documents'
        },
        'hybrid': {
            'db': 'Pinecone (Dense & Sparse)',
            'index': f'{DENSE_INDEX_NAME}, {SPARSE_INDEX_NAME}',
            'samples': 'Combined keyword and vector search'
        },
        'corrective': {
            'db': 'Pinecone (Dense)',
            'index': DENSE_INDEX_NAME,
            'samples': 'Verified documents'
        },
        'graph': {
            'db': 'Neo4j',
            'index': 'Knowledge Graph',
            'samples': 'Entity relationships (Nodes and Edges)'
        },
        'cache': {
            'db': 'In-Memory Cache',
            'index': 'Python Dict',
            'samples': 'Previously answered queries'
        },
        'agentic': {
            'db': 'Hybrid (Pinecone)',
            'index': f'{DENSE_INDEX_NAME}, {SPARSE_INDEX_NAME}',
            'samples': 'Iterative research data'
        }
    }
    
    data = info.get(strategy, {'db': 'Unknown', 'index': 'N/A', 'samples': 'N/A'})
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
