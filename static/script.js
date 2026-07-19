document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const navBtns = document.querySelectorAll('.nav-btn');
    const titleEl = document.getElementById('current-strategy-title');
    const descEl = document.getElementById('strategy-description');

    const strategyInfo = {
        'normal': { title: 'Normal RAG', desc: 'Simple fact retrieval based on semantic search.' },
        'hybrid': { title: 'Hybrid RAG', desc: 'Combines keyword and semantic search for better precision.' },
        'corrective': { title: 'Corrective RAG', desc: 'RAG with a grading step and fallback to full-text search.' },
        'graph': { title: 'Graph RAG', desc: 'Relationship-based retrieval using Knowledge Graphs.' },
        'cache': { title: 'Cache RAG', desc: 'Fast retrieval using cached responses.' },
        'agentic': { title: 'Agentic RAG', desc: 'Multi-step reasoning and research process.' },
    };

    let currentStrategy = 'normal';

    async function fetchDBInfo(strategy) {
        try {
            const response = await fetch(`/strategy-info/${strategy}`);
            const data = await response.json();
            
            const infoCard = document.createElement('div');
            infoCard.className = 'db-info-card';
            infoCard.innerHTML = `
                <div><strong>Database:</strong> ${data.db}</div>
                <div><strong>Index/Source:</strong> ${data.index}</div>
                <div><strong>Data Sample:</strong> ${data.samples}</div>
            `;
            return infoCard;
        } catch (error) {
            return null;
        }
    }

    async function updateStrategy(strategy) {
        currentStrategy = strategy;
        titleEl.textContent = strategyInfo[strategy].title;
        descEl.textContent = strategyInfo[strategy].desc;
        
        // Update DB Info Card
        const header = document.querySelector('header');
        const oldCard = header.querySelector('.db-info-card');
        if (oldCard) oldCard.remove();
        
        const dbCard = await fetchDBInfo(strategy);
        if (dbCard) header.appendChild(dbCard);

        // Clear chat window when switching strategies
        chatWindow.innerHTML = `<div class="message system">Switched to ${strategyInfo[strategy].title}. Please enter a query.</div>`;
        
        navBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.strategy === strategy);
        });
    }

    async function appendMessage(role, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        const textSpan = document.createElement('span');
        textSpan.textContent = text;
        msgDiv.appendChild(textSpan);
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        appendMessage('user', query);
        userInput.value = '';

        try {
            const response = await fetch(`/chat/${currentStrategy}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            const data = await response.json();

            if (data.error) {
                appendMessage('bot', `Error: ${data.error}`);
            } else {
                appendMessage('bot', data.response);
            }
        } catch (error) {
            appendMessage('bot', 'Error connecting to the server.');
        }
    }

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => updateStrategy(btn.dataset.strategy));
    });

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Initial load
    updateStrategy('normal');
});
