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
            
            loading.style.display = 'flex';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/evaluate', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                console.log("API Response:", data); // Log the full response
                
                if (handleMongoDBErrors(data)) {
                    return;
                }
                
                if (data.status === 'error') {
                    result.innerHTML = `<div class="error-message"><h3>Error</h3><p>${data.message}</p></div>`;
                } else if (data.error) {
                    result.innerHTML = `<div class="error-message"><h3>Error</h3><p>${data.error}</p></div>`;
                } else {
                    // Log the data for debugging
                    console.log("Data received from server:", data);
                    
                    // Calculate overall score first as it's used in multiple places
                    const overallScore = data.overall_score ? parseFloat(data.overall_score.replace('%', '')) / 100 : 0;
                    
                    // Check if we have valid response data
                    console.log("Processing response data structure:", {
                        hasReport: !!data.report,
                        reportType: data.report ? typeof data.report : "N/A",
                        hasOverallScore: !!data.overall_score,
                        hasSimilarityScore: !!data.similarity_score
                    });
                    
                    console.log("Calculated overallScore:", overallScore);
                    
                    // Create the results HTML
                    let resultHtml = `
                        <h2>Evaluation Results</h2>
                        <div class="score-container">
                            <div class="score-circle ${getScoreClass(overallScore)}">
                                ${Math.round(overallScore * 100)}%
                            </div>
                            <p>Similarity Score</p>
                        </div>
                    `;
                    
                    // Add function reports section if available
                    if (data.report && typeof data.report === 'object') {
                        console.log("Processing report object with keys:", Object.keys(data.report));
                        
                        resultHtml += `
                            <div class="function-reports">
                                <h3>Function Analysis</h3>
                                <div class="accordion">
                        `;
                        
                        // For each function in the report
                        try {
                            for (const funcName in data.report) {
                                if (!data.report.hasOwnProperty(funcName)) continue;
                                
                                const funcData = data.report[funcName];
                                console.log(`Processing function: ${funcName}`, funcData);
                                
                                if (!funcData) {
                                    console.log(`Skipping ${funcName}: funcData is null or undefined`);
                                    continue;
                                }
                                
                                const status = funcData.status || 'Unknown';
                                const similarity = funcData.similarity || 0;
                                const statusClass = status === 'Correct' ? 'status-match' : 
                                                  (status === 'Missing' ? 'status-missing' : 'status-mismatch');
                                
                                resultHtml += `
                                    <div class="accordion-item">
                                        <button class="accordion-button">
                                            <span class="function-status ${statusClass}"></span>
                                            ${funcName}: ${status} (${Math.round(similarity * 100)}% similar)
                                        </button>
                                        <div class="accordion-content">
                                            <div class="function-details">
                                `;
                                
                                // Add structure analysis if available
                                if (funcData.structure_analysis && typeof funcData.structure_analysis === 'object') {
                                    const structAnalysis = funcData.structure_analysis;
                                    console.log(`Processing structure analysis for ${funcName}:`, structAnalysis);
                                    
                                    // Extra safety check for variables, control_flow, and function_calls sections
                                    const hasVariables = structAnalysis.variables && typeof structAnalysis.variables === 'object';
                                    const hasControlFlow = structAnalysis.control_flow && typeof structAnalysis.control_flow === 'object';
                                    const hasFunctionCalls = structAnalysis.function_calls && typeof structAnalysis.function_calls === 'object';
                                    
                                    if (hasVariables || hasControlFlow || hasFunctionCalls) {
                                        resultHtml += `
                                            <div class="structure-analysis">
                                                <h4>Structure Analysis</h4>
                                                <table>
                                                    <thead>
                                                        <tr>
                                                            <th>Category</th>
                                                            <th>Missing</th>
                                                            <th>Extra</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                        `;
                                        
                                        // Variables
                                        if (hasVariables) {
                                            const missing = Array.isArray(structAnalysis.variables.missing_variables) ? 
                                                          structAnalysis.variables.missing_variables : [];
                                            const extra = Array.isArray(structAnalysis.variables.extra_variables) ? 
                                                        structAnalysis.variables.extra_variables : [];
                                            
                                            resultHtml += `
                                                <tr>
                                                    <td>Variables</td>
                                                    <td>${missing.length > 0 ? missing.join(', ') : 'None'}</td>
                                                    <td>${extra.length > 0 ? extra.join(', ') : 'None'}</td>
                                                </tr>
                                            `;
                                        }
                                        
                                        // Control flow
                                        if (hasControlFlow) {
                                            const missing = Array.isArray(structAnalysis.control_flow.missing_control_structures) ? 
                                                          structAnalysis.control_flow.missing_control_structures : [];
                                            const extra = Array.isArray(structAnalysis.control_flow.extra_control_structures) ? 
                                                        structAnalysis.control_flow.extra_control_structures : [];
                                            
                                            resultHtml += `
                                                <tr>
                                                    <td>Control Flow</td>
                                                    <td>${missing.length > 0 ? missing.join(', ') : 'None'}</td>
                                                    <td>${extra.length > 0 ? extra.join(', ') : 'None'}</td>
                                                </tr>
                                            `;
                                        }
                                        
                                        // Function calls
                                        if (hasFunctionCalls) {
                                            const missing = Array.isArray(structAnalysis.function_calls.missing_calls) ? 
                                                          structAnalysis.function_calls.missing_calls : [];
                                            const extra = Array.isArray(structAnalysis.function_calls.extra_calls) ? 
                                                        structAnalysis.function_calls.extra_calls : [];
                                            
                                            resultHtml += `
                                                <tr>
                                                    <td>Function Calls</td>
                                                    <td>${missing.length > 0 ? missing.join(', ') : 'None'}</td>
                                                    <td>${extra.length > 0 ? extra.join(', ') : 'None'}</td>
                                                </tr>
                                            `;
                                        }
                                        
                                        resultHtml += `
                                                    </tbody>
                                                </table>
                                            </div>
                                        `;
                                    } else {
                                        console.log(`No valid structure analysis sections found for ${funcName}`);
                                    }
                                }
                                
                                // Add recommendations if available
                                if (funcData.recommendations && Array.isArray(funcData.recommendations)) {
                                    resultHtml += `
                                        <div class="recommendations">
                                            <h4>Recommendations</h4>
                                            <ul>
                                                ${funcData.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                                            </ul>
                                        </div>
                                    `;
                                }
                                
                                // Add feedback if available
                                if (funcData.feedback) {
                                    resultHtml += `
                                        <div class="feedback">
                                            <h4>Feedback</h4>
                                            <div class="feedback-content">${formatFeedback(funcData.feedback)}</div>
                                        </div>
                                    `;
                                }
                                
                                resultHtml += `
                                            </div>
                                        </div>
                                    </div>
                                `;
                            }
                            
                            resultHtml += `
                                    </div>
                                </div>
                            `;
                        } catch (error) {
                            console.error("Error processing function reports:", error);
                            resultHtml += `
                                <div class="error-message"><h3>Error</h3><p>${error.message}</p></div>
                            `;
                        }
                    } else {
                        // Legacy structure analysis display (for backward compatibility)
                        if (data.structure_analysis && typeof data.structure_analysis === 'object') {
                            console.log("Processing legacy structure analysis");
                            resultHtml += `
                        <div class="structure-analysis">
                            <h3>Structure Analysis</h3>
                            <table>
                                <thead>
                                    <tr>
                                                <th>Category</th>
                                                <th>Expected</th>
                                                <th>Found</th>
                                                <th>Missing</th>
                                                <th>Extra</th>
                                    </tr>
                                </thead>
                                <tbody>
                    `;
                    
                            // Safely get structure analysis data
                            try {
                                const structureCategories = data.structure_analysis ? Object.keys(data.structure_analysis) : [];
                                console.log("Structure categories:", structureCategories);

                                if (structureCategories.length > 0) {
                                    for (const category of structureCategories) {
                                        if (!data.structure_analysis[category] || typeof data.structure_analysis[category] !== 'object') {
                                            console.log(`Skipping invalid category: ${category}`);
                                            continue;
                                        }

                                        const categoryData = data.structure_analysis[category];
                                        const expected = Array.isArray(categoryData.expected) ? categoryData.expected.length : 0;
                                        const found = Array.isArray(categoryData.found) ? categoryData.found.length : 0;
                                        const missing = Array.isArray(categoryData.missing) ? categoryData.missing : [];
                                        const extra = Array.isArray(categoryData.extra) ? categoryData.extra : [];

                                        resultHtml += `
                                            <tr>
                                                <td>${category}</td>
                                                <td>${expected}</td>
                                                <td>${found}</td>
                                                <td>${missing.length > 0 ? missing.join(', ') : 'None'}</td>
                                                <td>${extra.length > 0 ? extra.join(', ') : 'None'}</td>
                                            </tr>
                                        `;
                                    }
                                } else {
                                    resultHtml += `
                                        <tr>
                                            <td colspan="5">No structure analysis data available</td>
                                        </tr>
                                    `;
                                }
                            } catch (error) {
                                console.error("Error processing structure analysis:", error);
                        resultHtml += `
                            <tr>
                                        <td colspan="5">Error processing structure analysis: ${error.message}</td>
                            </tr>
                        `;
                    }
                    
                    resultHtml += `
                                </tbody>
                            </table>
                        </div>
                            `;
                        }
                        
                        // Legacy recommendations display (for backward compatibility)
                        if (data.recommendations) {
                            resultHtml += `
                        <div class="feedback-section">
                            <h3>Feedback</h3>
                            <div class="feedback-content">${formatFeedback(data.recommendations)}</div>
                        </div>
                    `;
                        }
                    }
                    
                    // Add similar functions section if present
                    if (data.similar_contexts && Array.isArray(data.similar_contexts) && data.similar_contexts.length > 0) {
                        resultHtml += `
                            <div class="similar-functions">
                                <h3>Similar Functions</h3>
                                <div class="accordion">
                        `;
                        
                        data.similar_contexts.forEach((context, index) => {
                            if (!context) return; // Skip if context is null or undefined
                            const functionName = context.function_name || 'Unknown';
                            const similarity = context.similarity || 0;
                            const code = context.code || 'No code available';
                            
                            resultHtml += `
                                <div class="accordion-item">
                                    <button class="accordion-button">${functionName} (${Math.round(similarity * 100)}% similar)</button>
                                    <div class="accordion-content">
                                        <pre><code>${code}</code></pre>
                                    </div>
                                </div>
                            `;
                        });
                        
                        resultHtml += `
                                </div>
                            </div>
                        `;
                    }
                    
                    // Add key concepts section if available
                    if (data.key_concepts && typeof data.key_concepts === 'object') {
                        console.log("Processing key concepts");
                        try {
                            resultHtml += `
                                <div class="key-concepts">
                                    <h3>Key Concepts</h3>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Concept</th>
                                                <th>Present</th>
                                                <th>Context</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                            `;

                            // Safely iterate through key concepts
                            for (const concept in data.key_concepts) {
                                if (!data.key_concepts.hasOwnProperty(concept)) continue;
                                
                                const conceptData = data.key_concepts[concept];
                                if (!conceptData || typeof conceptData !== 'object') {
                                    console.log(`Skipping invalid concept: ${concept}`);
                                    continue;
                                }
                                
                                const present = conceptData.present === true;
                                const context = conceptData.context || 'N/A';
                                
                                resultHtml += `
                                    <tr>
                                        <td>${concept}</td>
                                        <td>${present ? '✓' : '✗'}</td>
                                        <td>${context}</td>
                                    </tr>
                                `;
                            }

                            resultHtml += `
                                        </tbody>
                                    </table>
                                </div>
                            `;
                        } catch (error) {
                            console.error("Error processing key concepts:", error);
                            resultHtml += `
                                <div class="error-message">
                                    <h3>Error Processing Key Concepts</h3>
                                    <p>${error.message}</p>
                                </div>
                            `;
                        }
                    }
                    
                    // Add the detailed feedback
                    if (data.feedback && typeof data.feedback === 'string') {
                        resultHtml += `
                            <div class="detailed-feedback">
                                <h3>Detailed Feedback</h3>
                                <div class="feedback-content">${data.feedback}</div>
                            </div>
                        `;
                    }
                    
                    // Add the summary at the end
                    try {
                        // Log complete data for debugging summary issues
                        console.log("Creating summary with data:", {
                            total_functions: data.total_functions,
                            matched_functions: data.matched_functions,
                            function_stats: data.function_stats,
                            report_length: data.report ? Object.keys(data.report).length : 0,
                            model_info: data.model_used || data.model || 'Unknown model'
                        });
                        
                        // Extract function stats properly
                        let totalFunctions = 0;
                        let matchedFunctions = 0;
                        
                        // Try to get values from different possible locations in the response
                        if (data.total_functions !== undefined) {
                            totalFunctions = data.total_functions;
                        } else if (data.function_stats && data.function_stats.total !== undefined) {
                            totalFunctions = data.function_stats.total;
                        } else if (data.report) {
                            totalFunctions = Object.keys(data.report).length;
                        }
                        
                        if (data.matched_functions !== undefined) {
                            matchedFunctions = data.matched_functions;
                        } else if (data.function_stats && data.function_stats.matched !== undefined) {
                            matchedFunctions = data.function_stats.matched;
                        }
                        
                        const modelInfo = data.model_used || data.model || 'Unknown model';
                        
                        resultHtml += `
                            <div class="summary-section">
                                <h3>Summary</h3>
                                <p>Functions: ${matchedFunctions}/${totalFunctions} matched</p>
                                <p>Overall Score: ${Math.round(overallScore * 100)}%</p>
                                <p>Evaluated using: ${modelInfo}</p>
                            </div>
                        `;
                    } catch (error) {
                        console.error("Error creating summary section:", error);
                    }
                    
                    result.innerHTML = resultHtml;
                    
                    // Initialize accordion functionality
                    initializeAccordion();
                }
            } catch (error) {
                result.innerHTML = `<div class="error-message"><h3>Error</h3><p>${error.message}</p></div>`;
            } finally {
                loading.style.display = 'none';
                result.style.display = 'block';
            }
        });
    }
    
    if (textForm) {
        textForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submission = document.getElementById('textSubmission').files[0];
            const ideal = document.getElementById('textIdeal').files[0];
            const useOpenAI = document.getElementById('use_openai').checked;
            
            if (!submission || !ideal) {
                alert('Please select both student submission and ideal answer files.');
                return;
            }
            
            const formData = new FormData();
            formData.append('submission', submission);
            formData.append('ideal', ideal);
            formData.append('use_openai', useOpenAI);
            
            loading.style.display = 'flex';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/evaluate-text', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (handleMongoDBErrors(data)) {
                    return;
                }
                
                if (data.status === 'error') {
                    result.innerHTML = `<div class="error-message"><h3>Error</h3><p>${data.message}</p></div>`;
                } else {
                    let resultHtml = `
                        <h2>Text Q&A Evaluation Results</h2>
                        <div class="score-container">
                            <div class="score-circle ${getScoreClass(getOverallScore(data) / 100)}">
                                ${getOverallScore(data)}%
                            </div>
                            <p>Overall Score</p>
                        </div>
                        
                        <div class="stats-container">
                            <div class="stat-item">
                                <div class="stat-value high-score">${data.stats.high_count}</div>
                                <p>High Quality</p>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value medium-score">${data.stats.medium_count}</div>
                                <p>Medium Quality</p>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value low-score">${data.stats.low_count}</div>
                                <p>Low Quality</p>
                            </div>
                        </div>
                        
                        <div class="summary-section">
                            <h3>Summary Feedback</h3>
                            <div class="summary-content">
                                ${data.summary ? formatFeedback(data.summary) : 'No summary available.'}
                            </div>
                        </div>
                    `;
                    
                    // Add individual Q&A evaluations if available
                    if (data.evaluations && data.evaluations.length > 0) {
                        resultHtml += `
                            <div class="qa-evaluations">
                                <h3>Individual Question Evaluations</h3>
                                <div class="accordion">
                        `;
                        
                        data.evaluations.forEach((evaluation, index) => {
                            let qualityClass = 'low-score';
                            if (evaluation.quality === 'high') qualityClass = 'high-score';
                            else if (evaluation.quality === 'medium') qualityClass = 'medium-score';
                            
                            resultHtml += `
                                <div class="accordion-item">
                                    <button class="accordion-button">
                                        <span class="quality-indicator ${qualityClass}"></span>
                                        Q${index + 1}: ${evaluation.question.substring(0, 80)}${evaluation.question.length > 80 ? '...' : ''}
                                    </button>
                                    <div class="accordion-content">
                                        <div class="qa-comparison">
                                            <div class="qa-item">
                                                <h4>Question:</h4>
                                                <p>${evaluation.question}</p>
                                            </div>
                                            <div class="qa-item">
                                                <h4>Student's Answer:</h4>
                                                <p>${evaluation.student_answer}</p>
                                            </div>
                                            <div class="qa-item">
                                                <h4>Reference Answer:</h4>
                                                <p>${evaluation.reference_answer}</p>
                                            </div>
                                            <div class="qa-item">
                                                <h4>Quality: <span class="${qualityClass}">${evaluation.quality.toUpperCase()}</span></h4>
                                                <p>Similarity: ${Math.round(evaluation.similarity * 100)}%</p>
                                                ${evaluation.numerical_score ? `<p class="numerical-score">Numerical Score: ${evaluation.numerical_score}/100</p>` : ''}
                                            </div>
                                            <div class="qa-item">
                                                <h4>Feedback:</h4>
                                                <p>${formatFeedback(evaluation.feedback)}</p>
                                            </div>
                                            ${evaluation.key_concepts_present && evaluation.key_concepts_present.length > 0 ? `
                                            <div class="qa-item">
                                                <h4>Key Concepts Present:</h4>
                                                <ul class="key-concepts-present">
                                                    ${evaluation.key_concepts_present.map(concept => `<li>${concept}</li>`).join('')}
                                                </ul>
                                            </div>
                                            ` : ''}
                                            ${evaluation.key_concepts_missing && evaluation.key_concepts_missing.length > 0 ? `
                                            <div class="qa-item">
                                                <h4>Key Concepts Missing:</h4>
                                                <ul class="key-concepts-missing">
                                                    ${evaluation.key_concepts_missing.map(concept => `<li>${concept}</li>`).join('')}
                                                </ul>
                                            </div>
                                            ` : ''}
                                        </div>
                                    </div>
                                </div>
                            `;
                        });
                        
                        resultHtml += `
                                </div>
                            </div>
                        `;
                    }
                    
                    result.innerHTML = resultHtml;
                    
                    // Initialize accordion functionality
                    initializeAccordion();
                }
            } catch (error) {
                result.innerHTML = `<div class="error-message"><h3>Error</h3><p>${error.message}</p></div>`;
            } finally {
                loading.style.display = 'none';
                result.style.display = 'block';
            }
        });
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
}); 