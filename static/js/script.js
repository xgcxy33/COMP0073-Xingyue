document.getElementById("image").addEventListener("change", function(event) {
    const imageInput = event.target.files[0];  // Get the selected image file

    if (imageInput) {
        // Display the filename of the selected image
        const filename = imageInput.name;  // Get the file name
        document.getElementById("image-filename").textContent = `Selected Image: ${filename}`;
    } else {
        // Clear the filename display if no image is selected
        document.getElementById("image-filename").textContent = "";
    }
});

function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    const imageInput = document.getElementById("image").files[0];  // Get the uploaded image file
    
    if (userInput.trim() === "" && !imageInput) return; // Don't send empty messages or without an image
    
    // Create an object to hold the message data
    const data = { message: userInput };
    
    // If there is an image, convert it to Base64 and add it to the data object
    if (imageInput) {
        const reader = new FileReader();
        
        reader.onloadend = function () {
            // The result contains the Base64-encoded image
            data.image = reader.result.split(',')[1]; // Base64 string
            
            // Now send the data (both message and Base64 image)
            sendDataToBackend(data);
        };
        
        // Read the image as a Data URL (Base64)
        reader.readAsDataURL(imageInput);
    } else {
        // If no image, just send the message
        sendDataToBackend(data);
    }
}

// Function to send data (message + image) to the Flask backend
function sendDataToBackend(data) {
    // Display user's message
    displayMessage(data.message, 'user');
    
    // Clear the input field
    document.getElementById("user-input").value = '';
    document.getElementById("image").value = ''; // Clear the image input
    document.getElementById("image-filename").textContent = ''; // Clear the filename display


    // Send the data to Flask backend
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
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

    // Convert newlines (\n) into <br> tags
    message = message.replace(/\n/g, "<br>");
    
    // Find the last bot message element
    const botMessages = chatbox.getElementsByClassName('bot-message');
    
    if (botMessages.length > 0) {
        const lastBotMessage = botMessages[botMessages.length - 1];
        lastBotMessage.innerHTML = message; // Update the last bot message (use innerHTML to support <br>)
    }
    else {
        // If no bot message exists yet, create one
        displayMessage(message, 'bot');
    }
}
