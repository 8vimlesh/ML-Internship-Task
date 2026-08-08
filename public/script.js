document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('spam-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    
    const resultContainer = document.getElementById('result-container');
    const predictionBadge = document.getElementById('prediction-badge');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const messageInput = document.getElementById('message-input').value;
        if (!messageInput.trim()) return;

        // UI state: loading
        btnText.textContent = 'Analyzing...';
        spinner.classList.remove('hidden');
        submitBtn.disabled = true;
        resultContainer.classList.add('hidden');
        confidenceBar.style.width = '0%';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: messageInput })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Artificial delay for animation effect
            setTimeout(() => {
                displayResult(data.prediction, data.confidence);
            }, 600);

        } catch (error) {
            console.error('Error during prediction:', error);
            alert('Failed to analyze the message. Please ensure the server is running and the model is loaded.');
        } finally {
            // Restore button state (after slight delay if successful, immediate if error)
            setTimeout(() => {
                btnText.textContent = 'Analyze Message';
                spinner.classList.add('hidden');
                submitBtn.disabled = false;
            }, 600);
        }
    });

    function displayResult(prediction, confidence) {
        // Reset classes
        predictionBadge.className = 'badge';
        
        // Update content
        const isSpam = prediction.toLowerCase() === 'spam';
        predictionBadge.textContent = isSpam ? 'Spam Detected' : 'Safe Message';
        predictionBadge.classList.add(isSpam ? 'spam' : 'ham');
        
        const confidencePercentage = (confidence * 100).toFixed(1);
        confidenceText.textContent = `${confidencePercentage}%`;
        
        // Show container
        resultContainer.classList.remove('hidden');
        
        // Animate progress bar
        setTimeout(() => {
            confidenceBar.style.width = `${confidencePercentage}%`;
            // Color is handled via CSS sibling selector based on badge class
        }, 50);
    }
});
