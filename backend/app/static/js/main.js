// Function to toggle showing/hiding the structure analysis details
function toggleDetails(button) {
    const detailsDiv = button.nextElementSibling;
    if (detailsDiv.style.display === "none") {
        detailsDiv.style.display = "block";
        button.textContent = "Hide Full Structure Analysis";
    } else {
        detailsDiv.style.display = "none";
        button.textContent = "Show Full Structure Analysis";
    }
}

// Global error handler for catching null reference errors
window.addEventListener('error', function(event) {
    // Check if the error is a null reference error
    if (event.message && event.message.includes("Cannot read properties of null")) {
        console.error(`Null reference error caught: ${event.message}`);
        console.error(`At: ${event.filename}:${event.lineno}:${event.colno}`);
        console.error('Stack:', event.error.stack);
        
        // Prevent the default browser error handling
        event.preventDefault();
        
        // If needed, show a user-friendly error message
        // alert("An error occurred. Please try again or contact support if the issue persists.");
    }
});

// Progress stepper management
const STEPS = ['embedding', 'mapping', 'scoring', 'feedback'];

function updateStepperProgress(currentStep, error = null) {
    const steps = document.querySelectorAll('.stepper-step');
    
    steps.forEach((step, index) => {
        const stepName = step.dataset.step;
        const circle = step.querySelector('.step-circle');
        const currentStepIndex = STEPS.indexOf(currentStep);
        
        // Remove any existing content
        circle.innerHTML = '';
        step.classList.remove('completed', 'active', 'error', 'blocked');
        
        // Remove any existing error message
        const existingError = step.querySelector('.step-error');
        if (existingError) {
            existingError.remove();
        }
        
        if (error && stepName === currentStep) {
            // Error state
            step.classList.add('error');
            circle.innerHTML = '<span>✕</span>';
            
            // Add error message
            const errorMsg = document.createElement('div');
            errorMsg.className = 'step-error';
            errorMsg.textContent = error;
            step.appendChild(errorMsg);
            
            // Mark subsequent steps as blocked
            const subsequentSteps = Array.from(steps).slice(index + 1);
            subsequentSteps.forEach(subsequentStep => {
                subsequentStep.classList.add('blocked');
                const subsequentCircle = subsequentStep.querySelector('.step-circle');
                subsequentCircle.innerHTML = '<span>✕</span>';
            });
        } else if (STEPS.indexOf(stepName) < currentStepIndex) {
            // Completed state
            step.classList.add('completed');
            circle.innerHTML = '<span>✓</span>';
        } else if (stepName === currentStep) {
            // Active state
            step.classList.add('active');
            circle.innerHTML = '<div class="step-spinner"></div>';
        } else if (!error) {
            // Upcoming state
            circle.innerHTML = `<span class="step-number">${index + 1}</span>`;
        }
    });
}

function showSection(sectionId, content = null) {
    const section = document.getElementById(sectionId);
    const contentDiv = section.querySelector('.qa-content, .summary-content') || section;
    const skeleton = section.querySelector('.qa-skeleton, .summary-skeleton, .score-skeleton');
    
    // Show the section first
    section.style.display = 'block';
    
    if (content) {
        // If we have content, show it with animation
        if (contentDiv) {
            contentDiv.innerHTML = content;
            contentDiv.classList.add('slide-down');
        }
        
        // Hide the skeleton loader
        if (skeleton) {
            skeleton.style.display = 'none';
        }
    } else {
        // If no content yet, show the skeleton loader
        if (contentDiv) {
            contentDiv.innerHTML = '';
        }
        if (skeleton) {
            skeleton.style.display = 'block';
        }
    }
}

// Initialize the form when the document is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Safety function to safely get elements
    function safeGetElement(id) {
        const element = document.getElementById(id);
        if (!element) {
            console.warn(`Element with ID '${id}' not found in DOM. Creating a dummy element.`);
            const dummyElement = document.createElement('div');
            dummyElement.id = id;
            dummyElement.style.display = 'none';
            dummyElement.dataset.dummy = true;
            document.body.appendChild(dummyElement);
            return dummyElement;
        }
        return element;
    }

    // Check for common form elements and create if missing
    const elementIds = [
        'uploadForm', 'textForm', 'submission', 'ideal', 'model',
         'textSubmission', 'textIdeal',
        'textModel', 'evaluateBtn', 'evaluateTextBtn', 'loading'
    ];
    
    // Create any missing elements to prevent null reference errors
    elementIds.forEach(safeGetElement);
    
    const uploadForm = document.getElementById('uploadForm');
    const textForm = document.getElementById('textForm');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const evaluateBtn = document.getElementById('evaluateBtn');
    const modelSelect = document.getElementById('model');
    
    // Debug logs to verify elements are found
    console.log("Upload form found:", !!uploadForm);
    console.log("Evaluate button found:", !!evaluateBtn);
    
    // Add a global form submission listener to catch all form submissions
    document.addEventListener('submit', function(e) {
        console.log("Form submission detected:", e.target.id);
    });
    
    // Function to check if files are selected
    function checkFilesSelected() {
        const submission = document.getElementById('submission').files.length > 0;
        const ideal = document.getElementById('ideal').files.length > 0;
        return submission && ideal;
    }
    
    // Update button states based on file selection
    function updateButtonStates() {
        evaluateBtn.disabled = !checkFilesSelected();
    }
    
    // Event listeners for file input changes
    document.getElementById('submission').addEventListener('change', updateButtonStates);
    document.getElementById('ideal').addEventListener('change', updateButtonStates);
    
    // Initial button state
    updateButtonStates();
    
    // Add event listener for evaluate button - simplify the approach
    evaluateBtn.addEventListener('click', function(e) {
        e.preventDefault(); // Prevent default button behavior
        console.log("Evaluate button clicked");
        
        // Check if files are selected
        const submission = document.getElementById('submission').files[0];
        const ideal = document.getElementById('ideal').files[0];
        
        if (!submission || !ideal) {
            alert('Please select both student submission and ideal solution files.');
            return;
        }
        
        const model = document.getElementById('model').value;
        
        // No longer using the use_openai checkbox - model dropdown is the source of truth
        
        // Show loading indicator and prepare result area
        loading.style.display = 'block';
        result.style.display = 'block';
        updateStepperProgress('embedding');
            
        // Show skeleton loaders for all sections
        showSection('score-section');
        showSection('qa-evaluations-section');
        showSection('summary-section');
        
        // Create form data and submit manually
        const formData = new FormData();
        formData.append('submission', submission);
        formData.append('ideal', ideal);
        formData.append('model', model);
        formData.append('use_openai_feedback', useOpenAIFeedback);
        
        // Submit directly with fetch
        fetch('/evaluate', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log("API Response:", data);
            
            if (handleMongoDBErrors(data)) {
                updateStepperProgress('embedding', 'Database connection error');
                loading.style.display = 'none';
                return;
            }
            
            if (data.status === 'error' || data.error) {
                const errorMessage = data.message || data.error;
                updateStepperProgress('embedding', errorMessage);
                result.innerHTML = `<div class="error-message"><h3>Error</h3><p>${errorMessage}</p></div>`;
                loading.style.display = 'none';
                return;
            }
            
            // Process each stage
            try {
                // Embedding stage
                updateStepperProgress('mapping');
                
                // Mapping stage
                if (data.qa_pairs) {
                    updateStepperProgress('scoring');
                
                    // Scoring stage - Show the score first
                    if (data.overall_score) {
                        const overallScore = parseFloat(data.overall_score.replace('%', '')) / 100;
                        const scoreClass = getScoreClass(overallScore);
                        const scoreHtml = `
                        <div class="score-container">
                            <div class="score-circle ${scoreClass}">
                                <span>${Math.round(overallScore * 100)}%</span>
                            </div>
                            <p>Overall Score</p>
                        </div>
                        
                        <div class="stats-container">
                            <div class="stat-item">
                                <div class="stat-value high-score">${data.evaluations ? data.evaluations.filter(e => e.quality === 'high').length : 0}</div>
                                <p>High</p>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value medium-score">${data.evaluations ? data.evaluations.filter(e => e.quality === 'medium').length : 0}</div>
                                <p>Medium</p>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value low-score">${data.evaluations ? data.evaluations.filter(e => e.quality === 'low').length : 0}</div>
                                <p>Low</p>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value poor-score">${data.evaluations ? data.evaluations.filter(e => e.quality === 'poor').length : 0}</div>
                                <p>Poor</p>
                            </div>
                        </div>
                        `;
                        showSection('score-section', scoreHtml);
                    }
                    
                    // Show individual evaluations
                    if (data.qa_evaluations) {
                        updateStepperProgress('feedback');
                        const qaHtml = generateQAEvaluationsHtml(data.qa_evaluations);
                        showSection('qa-evaluations-section', qaHtml);
                    }
                    
                    // Finally, show the summary feedback
                    if (data.summary_feedback) {
                        const summaryHtml = `<div class="summary-content">${data.summary_feedback}</div>`;
                        showSection('summary-section', summaryHtml);
                        updateStepperProgress('feedback');
                        
                        // Mark process as complete
                        setTimeout(() => {
                            const feedbackStep = document.querySelector('.stepper-step[data-step="feedback"]');
                            feedbackStep.classList.remove('active');
                            feedbackStep.classList.add('completed');
                            feedbackStep.querySelector('.step-circle').innerHTML = '<span>✓</span>';
                        }, 500);
                    }
                }
                
                // Hide loading indicator
                loading.style.display = 'none';
            } catch (error) {
                console.error('Error processing evaluation stages:', error);
                updateStepperProgress(currentStep, 'Error processing results');
                loading.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error submitting evaluation:', error);
            updateStepperProgress('embedding', 'Network error occurred');
            loading.style.display = 'none';
        });
    });
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            console.log("Form submission handler called");
            
            const submission = document.getElementById('submission').files[0];
            const ideal = document.getElementById('ideal').files[0];
            const model = document.getElementById('model').value;
            
            if (!submission || !ideal) {
                alert('Please select both student submission and ideal solution files.');
                return;
            }
            
            // Show loading indicator
            loading.style.display = 'block';
            
            const formData = new FormData();
            formData.append('submission', submission);
            formData.append('ideal', ideal);
            formData.append('model', model);
            formData.append('use_openai_feedback', useOpenAIFeedback);
            
            // Show the result container and initialize progress
            result.style.display = 'block';
            updateStepperProgress('embedding');
            
            // Show skeleton loaders for all sections
            showSection('score-section');
            showSection('qa-evaluations-section');
            showSection('summary-section');
            
            try {
                const response = await fetch('/evaluate', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                console.log("API Response:", data);
                
                if (handleMongoDBErrors(data)) {
                    updateStepperProgress('embedding', 'Database connection error');
                    return;
                }
                
                if (data.status === 'error' || data.error) {
                    const errorMessage = data.message || data.error;
                    updateStepperProgress('embedding', errorMessage);
                    result.innerHTML = `<div class="error-message"><h3>Error</h3><p>${errorMessage}</p></div>`;
                    return;
                }
                
                // Process each stage
                try {
                    // Embedding stage
                    updateStepperProgress('mapping');
                    
                    // Mapping stage
                    if (data.qa_pairs) {
                        updateStepperProgress('scoring');
                    
                        // Scoring stage - Show the score first
                        if (data.overall_score) {
                            const overallScore = parseFloat(data.overall_score.replace('%', '')) / 100;
                            const scoreClass = getScoreClass(overallScore);
                            const scoreHtml = `
                            <div class="score-container">
                                <div class="score-circle ${scoreClass}">
                                    <span>${Math.round(overallScore * 100)}%</span>
                                </div>
                                <p>Overall Score</p>
                            </div>
                            
                            <div class="stats-container">
                                <div class="stat-item">
                                    <div class="stat-value high-score">${data.evaluations ? data.evaluations.filter(e => e.quality === 'high').length : 0}</div>
                                    <p>High</p>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value medium-score">${data.evaluations ? data.evaluations.filter(e => e.quality === 'medium').length : 0}</div>
                                    <p>Medium</p>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value low-score">${data.evaluations ? data.evaluations.filter(e => e.quality === 'low').length : 0}</div>
                                    <p>Low</p>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value poor-score">${data.evaluations ? data.evaluations.filter(e => e.quality === 'poor').length : 0}</div>
                                    <p>Poor</p>
                                </div>
                            </div>
                            `;
                            showSection('score-section', scoreHtml);
                        }
                        
                        // Show individual evaluations
                        if (data.qa_evaluations) {
                            updateStepperProgress('feedback');
                            const qaHtml = generateQAEvaluationsHtml(data.qa_evaluations);
                            showSection('qa-evaluations-section', qaHtml);
                        }
                        
                        // Finally, show the summary feedback
                        if (data.summary_feedback) {
                            const summaryHtml = `<div class="summary-content">${data.summary_feedback}</div>`;
                            showSection('summary-section', summaryHtml);
                            updateStepperProgress('feedback');
                            
                            // Mark process as complete
                            setTimeout(() => {
                                const feedbackStep = document.querySelector('.stepper-step[data-step="feedback"]');
                                feedbackStep.classList.remove('active');
                                feedbackStep.classList.add('completed');
                                feedbackStep.querySelector('.step-circle').innerHTML = '<span>✓</span>';
                            }, 500);
                        }
                    }
                } catch (error) {
                    console.error('Error processing evaluation stages:', error);
                    updateStepperProgress(currentStep, 'Error processing results');
                }
            } catch (error) {
                console.error('Error submitting evaluation:', error);
                updateStepperProgress('embedding', 'Network error occurred');
            }
        });
    }
    
    if (textForm) {
        textForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submission = document.getElementById('textSubmission').files[0];
            const ideal = document.getElementById('textIdeal').files[0];
            const model = document.getElementById('textModel').value;
            
            if (!submission || !ideal) {
                alert('Please select both student submission and ideal answer files.');
                return;
            }
            
            // Prepare form data
            const formData = new FormData();
            formData.append('submission', submission);
            formData.append('ideal', ideal);
            formData.append('model', model);
            
            // Show textResult container and reset stepper
            const textResult = document.getElementById('textResult');
            textResult.style.display = 'block';
            updateStepperProgress('embedding');
            
            // Show skeletons
            document.querySelector('.text-score-skeleton').style.display = 'block';
            document.querySelector('.text-questions-skeleton').style.display = 'block';
            document.querySelector('.text-summary-skeleton').style.display = 'block';
            
            // Hide real sections initially
            document.getElementById('textScoreSection').style.display = 'none';
            document.getElementById('textQuestionsSection').style.display = 'none';
            document.getElementById('textSummarySection').style.display = 'none';
            
            try {
                // Add a timeout controller
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
                
                // Show a loading indicator
                const textLoading = document.getElementById('textLoading');
                if (textLoading) textLoading.style.display = 'block';
                
                const response = await fetch('/evaluate-text', {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal,
                    // Ensure we get the right response type and don't cache
                    headers: {
                        'Accept': 'application/json, text/event-stream',
                        'Cache-Control': 'no-cache'
                    }
                });
                
                // Clear the timeout as we got a response
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Network response error (${response.status}): ${errorText || response.statusText}`);
                }
                
                console.log('Response received, beginning to read stream');
                
                // Check if the response body is available
                if (!response.body) {
                    throw new Error('Response body stream is not available');
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                try {
                    let streamClosed = false;
                    
                    while (!streamClosed) {
                        const { value, done } = await reader.read();
                        if (done) {
                            console.log('Stream complete, processing any remaining data');
                            streamClosed = true;
                            // Process any remaining data in buffer
                            if (buffer.trim()) {
                                try {
                                    const obj = JSON.parse(buffer.trim());
                                    handleTextStageUpdate(obj);
                                } catch (jsonErr) {
                                    console.error('Error parsing final buffer JSON:', jsonErr);
                                }
                            }
                            break;
                        }
                        
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        // Keep the last potentially incomplete line in the buffer
                        buffer = lines.pop() || '';
                        
                        for (const line of lines) {
                            if (!line.trim()) continue;
                            try {
                                const cleaned = line.trim();
                                const obj = JSON.parse(cleaned);
                                console.log('Processing JSON object:', obj);
                                handleTextStageUpdate(obj);
                            } catch (err) {
                                console.error('JSON parse error', err, line);
                                continue;
                            }
                        }
                    }
                    
                    // Hide loading indicator when stream is complete
                    const textLoading = document.getElementById('textLoading');
                    if (textLoading) textLoading.style.display = 'none';
                    
                } catch (streamErr) {
                    console.error('Stream reading error:', streamErr);
                    // Handle stream being closed abruptly
                    if (streamErr.name === 'AbortError' || 
                        (streamErr.message && streamErr.message.includes('aborted'))) {
                        console.warn('Stream was aborted, attempting to finish processing');
                        updateStepperProgress('feedback');
                    } else {
                        updateStepperProgress('embedding', 'Error reading response stream: ' + streamErr.message);
                    }
                    
                    // Hide loading indicator on error
                    const textLoading = document.getElementById('textLoading');
                    if (textLoading) textLoading.style.display = 'none';
                }
            } catch (err) {
                console.error('Streaming error', err);
                updateStepperProgress('embedding', err.message || 'Stream error');
                
                // Hide loading indicator on error
                const textLoading = document.getElementById('textLoading');
                if (textLoading) textLoading.style.display = 'none';
            }
        });
    }
    
    function handleTextStageUpdate(update) {
        const stage = update.stage;
        const status = update.status;
        if (status === 'working') {
            updateStepperProgress(stage);
            return;
        }
        if (status === 'error') {
            updateStepperProgress(stage, update.message || 'Error');
            return;
        }
        if (status === 'success') {
            // Move to next stage unless this is feedback (final)
            if (stage === 'embedding') updateStepperProgress('mapping');
            else if (stage === 'mapping') updateStepperProgress('scoring');
            else if (stage === 'scoring') updateStepperProgress('feedback');
            else if (stage === 'feedback') {
                // Received final data
                const data = update.data || {};
                
                if (data.overall_score !== undefined) {
                    const score = parseInt(data.overall_score);
                    const scoreClass = score >= 80 ? 'high-score' : score >= 60 ? 'medium-score' : 'low-score';
                    
                    // Count different quality levels
                    const highCount = data.evaluations ? data.evaluations.filter(e => e.quality === 'high').length : 0;
                    const mediumCount = data.evaluations ? data.evaluations.filter(e => e.quality === 'medium').length : 0;
                    const lowCount = data.evaluations ? data.evaluations.filter(e => e.quality === 'low').length : 0;
                    const poorCount = data.evaluations ? data.evaluations.filter(e => e.quality === 'poor').length : 0;
                    
                    // Create enhanced score display with metrics
                    const scoreHtml = `
                        <div class="score-container">
                            <div class="score-circle ${scoreClass}">
                                <span>${score}%</span>
                            </div>
                            <p>Overall Score</p>
                        </div>
                        
                        <div class="stats-container">
                            <div class="stat-item">
                                <div class="stat-value high-score">${highCount}</div>
                                <p>High</p>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value medium-score">${mediumCount}</div>
                                <p>Medium</p>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value low-score">${lowCount}</div>
                                <p>Low</p>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value poor-score">${poorCount}</div>
                                <p>Poor</p>
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('textScoreSection').innerHTML = '<h3>Overall Score</h3>' + scoreHtml;
                    document.querySelector('.text-score-skeleton').style.display = 'none';
                    document.getElementById('textScoreSection').style.display = 'block';
                }
                
                if (data.evaluations && Array.isArray(data.evaluations)) {
                    // Add debugging to see the structure of evaluations
                    console.log("Text evaluations data received:", data.evaluations);
                    
                    const listDiv = document.getElementById('textQuestionsList');
                    listDiv.innerHTML = generateQAEvaluationsHtml(data.evaluations);
                    document.querySelector('.text-questions-skeleton').style.display = 'none';
                    document.getElementById('textQuestionsSection').style.display = 'block';
                    initializeAccordion();
                }
                
                if (data.summary) {
                    document.getElementById('textSummaryContent').innerHTML = formatFeedback(data.summary);
                    document.querySelector('.text-summary-skeleton').style.display = 'none';
                    document.getElementById('textSummarySection').style.display = 'block';
                }
                
                // Mark feedback step complete
                const feedbackStep = document.querySelector('.stepper-step[data-step="feedback"]');
                feedbackStep.classList.remove('active');
                feedbackStep.classList.add('completed');
                feedbackStep.querySelector('.step-circle').innerHTML = '<span>✓</span>';
            }
        }
    }
    
    // Helper functions
    function getScoreClass(score) {
        if (score >= 0.8) return 'high-score';
        if (score >= 0.6) return 'medium-score';
        return 'low-score';
    }
    
    function formatFeatureName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }
    
    function formatFeedback(feedback) {
        if (typeof feedback === 'string') {
            return feedback.replace(/\n/g, '<br>');
        }
        
        if (Array.isArray(feedback)) {
            return feedback.map(item => `<p>${item}</p>`).join('');
        }
        
        return JSON.stringify(feedback);
    }
    
    function initializeAccordion() {
        const accordionButtons = document.querySelectorAll('.accordion-button');
        accordionButtons.forEach(button => {
            button.addEventListener('click', function() {
                this.classList.toggle('active');
                const content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            });
        });
    }

    // Function to handle MongoDB connection errors in API responses
    function handleMongoDBErrors(response) {
        if (response.status === 'error' && response.message && 
            (response.message.includes('MongoDB') || response.message.includes('mongo'))) {
            document.getElementById('mongodb-error').style.display = 'block';
            document.getElementById('mongodb-error-message').textContent = response.message;
            return true;
        }
        return false;
    }

    // Look for a function or section that handles text evaluation results
    // Add a function to safely get the overall score from various response formats
    function getOverallScore(data) {
        // Try to get score from top level first
        if (data.overall_score !== undefined) {
            return data.overall_score;
        }
        
        // Try to get score from result object
        if (data.result && data.result.overall_score !== undefined) {
            return data.result.overall_score;
        }
        
        // If neither exists, return 0
        return 0;
    }

    // Helper function to generate QA evaluations HTML
    function generateQAEvaluationsHtml(evaluations) {
        let html = '<div class="qa-content">';
        
        // Add debugging to console
        console.log("Evaluations data:", evaluations);
        
        evaluations.forEach((evaluation, index) => {
            const qualityClass = evaluation.quality === 'high' ? 'high-score' : 
                               evaluation.quality === 'medium' ? 'medium-score' : 
                               evaluation.quality === 'low' ? 'low-score' : 'poor-score';
            
            // Always use index+1 for sequential numbering
            const questionNumber = index + 1;
            const similarityPercentage = Math.round(evaluation.similarity * 100);
            
            console.log(`Question ${index}: display number=${questionNumber}, quality=${evaluation.quality}`);
            
            html += `
                <div class="qa-item">
                    <button class="accordion-button ${qualityClass}">
                        <span class="question-number">Question ${questionNumber}</span>
                        <div class="button-metrics">
                            <span class="quality-indicator">${evaluation.quality}</span>
                            <span class="similarity-score">${similarityPercentage}%</span>
                        </div>
                    </button>
                    <div class="accordion-content">
                        <div class="similarity-metrics">
                            <p><strong>Similarity Score:</strong> ${similarityPercentage}%</p>
                            <p><strong>Quality:</strong> ${evaluation.quality}</p>
                        </div>
                        <div class="qa-details">
                            ${evaluation.student_answer ? `<p><strong>Student Answer:</strong> ${evaluation.student_answer}</p>` : ''}
                            ${evaluation.ideal_answer ? `<p><strong>Ideal Answer:</strong> ${evaluation.ideal_answer}</p>` : ''}
                        </div>
                        <div class="feedback-content">
                            <h5>Feedback:</h5>
                            ${evaluation.feedback}
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }
}); 

// Removed incorrect event listener for evaluationForm which doesn't exist in the HTML 