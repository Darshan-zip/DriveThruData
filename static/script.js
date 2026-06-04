async function sendMessage() {
    const inputField = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const message = inputField.value.trim();

    if (!message) return;

    // Append user message
    chatBox.innerHTML += `<div class="message user-message">${message}</div>`;
    inputField.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;

    // Bot typing indicator
    const botMsgDiv = document.createElement('div');
    botMsgDiv.className = 'message bot-message';
    botMsgDiv.innerText = '...';
    chatBox.appendChild(botMsgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: message })
        });
        const data = await response.json();
        
        // Add document count badge
        const countInfo = data.count > 0 ? `<div class="doc-count">Found ${data.count} relevant documents</div>` : '';
        botMsgDiv.innerHTML = `${countInfo}<div>${data.response}</div>`;
    } catch (error) {
        botMsgDiv.innerText = 'Error connecting to server.';
    }
    chatBox.scrollTop = chatBox.scrollHeight;
}

document.getElementById('chat-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    await sendMessage();
});
