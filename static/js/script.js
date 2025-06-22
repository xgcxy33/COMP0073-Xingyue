function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    
    if (userInput.trim() === "") return; // Don't send empty messages
    
    // Display user's message
    displayMessage(userInput, 'user');
    
    // Clear the input field
    document.getElementById("user-input").value = '';
    
    // Send user message to Flask backend
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        let output = '';

        // Read the stream progressively
        reader.read().then(function processText({ done, value }) {
            if (done) return;

            // Decode and add to the output
            output += decoder.decode(value, { stream: true });
            
            // Update the last message
            updateBotMessage(output);

            // Continue reading the next chunk
            reader.read().then(processText);
        });
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Display user's message in the chatbox
function displayMessage(message, sender) {
    const chatbox = document.getElementById("chatbox");
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender === 'user' ? "user-message" : "bot-message");
    messageDiv.textContent = message;
    chatbox.appendChild(messageDiv);
    
    // Scroll to the bottom of the chatbox
    chatbox.scrollTop = chatbox.scrollHeight;
}

// Update the bot's ongoing message
function updateBotMessage(message) {
    const chatbox = document.getElementById("chatbox");
    
    // Find the last bot message element
    const botMessages = chatbox.getElementsByClassName('bot-message');
    
    if (botMessages.length > 0) {
        const lastBotMessage = botMessages[botMessages.length - 1];
        lastBotMessage.textContent = message; // Update the last bot message
    }
    else {
        // If no bot message exists yet, create one
        displayMessage(message, 'bot');
    }
}
