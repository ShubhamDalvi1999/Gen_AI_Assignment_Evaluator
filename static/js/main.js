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
    const uploadForm = document.getElementById('uploadForm');
    const textForm = document.getElementById('textForm');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const evaluateBtn = document.getElementById('evaluateBtn');
    const modelSelect = document.getElementById('model');
    
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
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submission = document.getElementById('submission').files[0];
            const ideal = document.getElementById('ideal').files[0];
            const model = document.getElementById('model').value;
            const useOpenAIFeedback = document.getElementById('use_openai_feedback').checked;
            
            if (!submission || !ideal) {
                alert('Please select both student submission and ideal solution files.');
                return;
            }
            
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
                            const scoreHtml = `
                            <div class="score-circle ${getScoreClass(overallScore)}">
                                ${Math.round(overallScore * 100)}%
                            </div>
                            <p>Similarity Score</p>
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
                const response = await fetch('/evaluate-text', {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) throw new Error('Network response was not ok');
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    let lines = buffer.split('\n');
                    buffer = lines.pop(); // keep incomplete line
                    for (const line of lines) {
                        if (!line.trim()) continue;
                        let obj;
                        try {
                            const cleaned = line.trim();
                            if (!cleaned) continue;
                            obj = JSON.parse(cleaned);
                        } catch (err) {
                            console.error('JSON parse error', err, line);
                            continue;
                        }
                        handleTextStageUpdate(obj);
                    }
                }
            } catch (err) {
                console.error('Streaming error', err);
                updateStepperProgress('embedding', err.message || 'Stream error');
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
                    document.getElementById('textOverallScore').textContent = score;
                    document.querySelector('.text-score-skeleton').style.display = 'none';
                    document.getElementById('textScoreSection').style.display = 'block';
                }
                if (data.evaluations && Array.isArray(data.evaluations)) {
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
        for (const evaluation of evaluations) {
            html += `
                <div class="qa-item">
                    <h4>Question ${evaluation.question_number}</h4>
                    <div class="similarity-metrics">
                        <p><strong>Similarity Score:</strong> ${Math.round(evaluation.similarity * 100)}%</p>
                        <p><strong>Quality:</strong> ${evaluation.quality}</p>
                    </div>
                    <div class="feedback-content">
                        ${evaluation.feedback}
                    </div>
                </div>
            `;
        }
        html += '</div>';
        return html;
    }
}); 

document.getElementById('evaluationForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submissionFile = document.getElementById('submission').files[0];
    const idealFile = document.getElementById('ideal').files[0];
    
    if (!submissionFile || !idealFile) {
        alert('Please select both submission and ideal solution files.');
        return;
    }
    
    // Show result section and reset progress
    document.getElementById('result').style.display = 'block';
    updateStepperProgress('embedding');
    
    // Show skeleton loaders
    document.querySelectorAll('.skeleton-loader').forEach(loader => {
        loader.style.display = 'block';
    });
    
    // Hide previous results
    document.getElementById('scoreSection').style.display = 'none';
    document.getElementById('questionsSection').style.display = 'none';
    document.getElementById('summarySection').style.display = 'none';
    
    const formData = new FormData();
    formData.append('submission', submissionFile);
    formData.append('ideal', idealFile);
    
    try {
        const response = await fetch('/evaluate', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'error') {
            // Show error in the stepper
            updateStepperProgress(result.stage, result.message);
            return;
        }
        
        // Update progress through stages
        updateStepperProgress('mapping');
        await new Promise(resolve => setTimeout(resolve, 500));
        updateStepperProgress('scoring');
        await new Promise(resolve => setTimeout(resolve, 500));
        updateStepperProgress('feedback');
        
        // Fade out skeleton loaders and show content sections with animation
        const fadeOutDuration = 300;
        const slideInDuration = 500;
        
        // Show score section
        document.querySelectorAll('.skeleton-loader').forEach(loader => {
            loader.style.opacity = 0;
            setTimeout(() => loader.style.display = 'none', fadeOutDuration);
        });
        
        // Animate in the score section
        const scoreSection = document.getElementById('scoreSection');
        scoreSection.style.display = 'block';
        scoreSection.style.opacity = 0;
        scoreSection.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            scoreSection.style.transition = `all ${slideInDuration}ms ease-out`;
            scoreSection.style.opacity = 1;
            scoreSection.style.transform = 'translateY(0)';
            
            // Update score content
            document.getElementById('overallScore').textContent = result.data.overall_score.toFixed(2);
            
            // Show questions section after score animation
            setTimeout(() => {
                const questionsSection = document.getElementById('questionsSection');
                questionsSection.style.display = 'block';
                questionsSection.style.opacity = 0;
                questionsSection.style.transform = 'translateY(20px)';
                
                // Populate questions
                const questionsContainer = document.getElementById('questionsList');
                questionsContainer.innerHTML = '';
                
                result.data.question_scores.forEach((score, index) => {
                    const questionDiv = document.createElement('div');
                    questionDiv.className = 'question-item';
                    questionDiv.innerHTML = `
                        <h4>Question ${index + 1}</h4>
                        <p>Score: ${score.score.toFixed(2)}</p>
                        <p>Feedback: ${score.feedback}</p>
                    `;
                    questionsContainer.appendChild(questionDiv);
                });
                
                setTimeout(() => {
                    questionsSection.style.transition = `all ${slideInDuration}ms ease-out`;
                    questionsSection.style.opacity = 1;
                    questionsSection.style.transform = 'translateY(0)';
                    
                    // Show summary section after questions animation
                    setTimeout(() => {
                        const summarySection = document.getElementById('summarySection');
                        summarySection.style.display = 'block';
                        summarySection.style.opacity = 0;
                        summarySection.style.transform = 'translateY(20px)';
                        
                        // Update summary content
                        document.getElementById('summaryFeedback').textContent = result.data.summary_feedback;
                        
                        setTimeout(() => {
                            summarySection.style.transition = `all ${slideInDuration}ms ease-out`;
                            summarySection.style.opacity = 1;
                            summarySection.style.transform = 'translateY(0)';
                        }, 50);
                    }, slideInDuration);
                }, 50);
            }, slideInDuration);
        }, fadeOutDuration);
        
    } catch (error) {
        console.error('Error:', error);
        updateStepperProgress('scoring', 'An unexpected error occurred. Please try again.');
    }
}); 