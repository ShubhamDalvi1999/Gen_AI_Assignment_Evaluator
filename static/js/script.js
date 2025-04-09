// Add the UI element for the emphasize_embedding option
document.addEventListener('DOMContentLoaded', function() {
    // Add checkbox for embedding emphasis after the token check button
    const tokenBtnContainer = document.getElementById('tokenCheckBtn').parentElement;
    
    const embedCheckboxDiv = document.createElement('div');
    embedCheckboxDiv.className = 'mb-3 form-check';
    embedCheckboxDiv.innerHTML = `
        <input type="checkbox" class="form-check-input" id="emphasizeEmbedding">
        <label class="form-check-label" for="emphasizeEmbedding">Emphasize semantic similarity (use embedding as primary score)</label>
    `;
    
    tokenBtnContainer.after(embedCheckboxDiv);
});

// Update the submitQA function to include the emphasize_embedding parameter
function submitQA() {
    // Get the document ID from the input field
    const documentId = document.getElementById('documentId').value;
    if (!documentId) {
        showError("Please enter a document ID");
        return;
    }

    // Get the course context
    const courseContext = document.getElementById('courseContext').value || '';

    // Get the emphasize_embedding option
    const emphasizeEmbedding = document.getElementById('emphasizeEmbedding').checked;
    
    // Collect all Q&A pairs
    const qaInputs = document.querySelectorAll('.qa-pair');
    const qaPairs = [];
    
    qaInputs.forEach(function(qaInput) {
        const question = qaInput.querySelector('.question-input').value;
        const answer = qaInput.querySelector('.answer-input').value;
        
        if (question && answer) {
            qaPairs.push({
                question: question,
                answer: answer
            });
        }
    });
    
    if (qaPairs.length === 0) {
        showError("Please add at least one Q&A pair");
        return;
    }
    
    // Show loading indicator
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    
    // Send data to backend
    fetch('/api/process-qa', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            document_id: documentId,
            qa_pairs: qaPairs,
            course_context: courseContext,
            emphasize_embedding: emphasizeEmbedding
        }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        // Hide loading indicator
        document.getElementById('loadingSpinner').style.display = 'none';
        
        // Display results
        displayResults(data);
    })
    .catch(error => {
        // Hide loading indicator
        document.getElementById('loadingSpinner').style.display = 'none';
        
        // Show error message
        showError(error.message);
    });
} 