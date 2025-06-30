document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const messageForm = document.getElementById('message-form');
    const sessionIdInput = document.getElementById('session-id');
    const pdfFilesInput = document.getElementById('pdf-files');
    const uploadStatus = document.getElementById('upload-status');
    const chatWindow = document.getElementById('chat-window');
    const messageInput = document.getElementById('message-input');
    const sendButton = messageForm.querySelector('button');

    const API_BASE_URL = 'http://127.0.0.1:8000';

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const sessionId = sessionIdInput.value;
        const files = pdfFilesInput.files;

        if (!sessionId || files.length === 0) {
            alert('Please provide a session ID and select at least one PDF file.');
            return;
        }

        const formData = new FormData();
        formData.append('session_id', sessionId);
        for (const file of files) {
            formData.append('files', file);
        }

        uploadStatus.textContent = 'Uploading and processing... This may take a moment.';
        
        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to upload files.');
            }

            const result = await response.json();
            uploadStatus.textContent = `✅ ${result.message} You can now start chatting.`;
            messageInput.disabled = false;
            sendButton.disabled = false;
            addMessage('assistant', `PDFs processed. How can I help you with the content?`);

        } catch (error) {
            uploadStatus.textContent = `❌ Error: ${error.message}`;
            console.error('Upload error:', error);
        }
    });

    messageForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const sessionId = sessionIdInput.value;
        const input = messageInput.value.trim();

        if (!input) return;

        addMessage('user', input);
        messageInput.value = '';
        messageInput.disabled = true;
        sendButton.disabled = true;

        addMessage('assistant', 'Thinking...');

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, input }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to get a response.');
            }

            const result = await response.json();
            
            // Remove the "Thinking..." message before adding the real one
            chatWindow.removeChild(chatWindow.lastChild);
            addMessage('assistant', result.answer);

        } catch (error) {
            chatWindow.removeChild(chatWindow.lastChild);
            addMessage('assistant', `Sorry, an error occurred: ${error.message}`);
            console.error('Chat error:', error);
        } finally {
            messageInput.disabled = false;
            sendButton.disabled = false;
            messageInput.focus();
        }
    });

    function addMessage(sender, text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        
        const p = document.createElement('p');
        p.textContent = text;
        messageElement.appendChild(p);
        
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});