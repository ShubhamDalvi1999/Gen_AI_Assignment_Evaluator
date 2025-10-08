import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import ModelSelector from '../components/ModelSelector';
import { useFileUpload } from '../hooks/useFileUpload';
import { evaluationApi, CodeEvaluationResult } from '../api/evaluationApi';
import { formatSimilarity, getQualityColor, formatQuality } from '../utils/formatters';
import { FormattedFeedback, formatSimpleFeedback } from '../utils/feedbackFormatter';
import ExportDropdown from '../components/ExportDropdown';
import { exportFeedbackToTXT } from '../utils/exportUtils';

const CodeEvaluation: React.FC = () => {
  const {
    submissionFile,
    idealFile,
    error,
    setSubmissionFile,
    setIdealFile,
    validateFiles,
  } = useFileUpload();

  const [selectedModel, setSelectedModel] = useState<'ollama' | 'openai'>('ollama');
  const [useOpenAIFeedback, setUseOpenAIFeedback] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [result, setResult] = useState<CodeEvaluationResult | null>(null);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [feedbackStates, setFeedbackStates] = useState<{ [key: string]: { loading: boolean; feedback: string | null; error: string | null } }>({});
  const [summaryFeedbackState, setSummaryFeedbackState] = useState<{ loading: boolean; feedback: string | null; error: string | null }>({ loading: false, feedback: null, error: null });
  const [isGeneratingFeedback] = useState(false);

  const handleEvaluate = async () => {
    if (!validateFiles()) {
      return;
    }

    if (!submissionFile || !idealFile) {
      return;
    }

    setIsEvaluating(true);
    setEvaluationError(null);
    setResult(null);

    try {
      const evaluationResult = await evaluationApi.evaluateCode({
        submission: submissionFile,
        ideal: idealFile,
        model: selectedModel,
        use_openai_feedback: useOpenAIFeedback,
      });

      setResult(evaluationResult);
    } catch (err: any) {
      setEvaluationError(err.response?.data?.detail || err.message || 'Evaluation failed');
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleGenerateFeedback = async (functionName: string, functionData: any) => {
    if (!result) return;

    setFeedbackStates(prev => ({
      ...prev,
      [functionName]: { loading: true, feedback: null, error: null }
    }));

    try {
      // For now, we'll use a simplified approach since we don't have the full structure analysis
      // In a real implementation, you'd need to pass the actual structure analysis from the evaluation
      const feedbackResult = await evaluationApi.generateCodeFeedback({
        student_code: functionData.student_code || 'Code not available',
        ideal_code: functionData.ideal_code || 'Code not available',
        similarity: functionData.similarity,
        structure_analysis: JSON.stringify({
          missing_variables: [],
          extra_variables: [],
          missing_control_structures: [],
          extra_control_structures: [],
          missing_function_calls: [],
          extra_function_calls: []
        }),
        similar_contexts: '[]',
        use_openai: useOpenAIFeedback
      });

      setFeedbackStates(prev => ({
        ...prev,
        [functionName]: { loading: false, feedback: feedbackResult.feedback, error: null }
      }));
    } catch (err: any) {
      setFeedbackStates(prev => ({
        ...prev,
        [functionName]: { 
          loading: false, 
          feedback: null, 
          error: err.response?.data?.detail || err.message || 'Failed to generate feedback' 
        }
      }));
    }
  };

  const handleGenerateSummaryFeedback = async () => {
    if (!result || !result.function_results) return;

    setSummaryFeedbackState({ loading: true, feedback: null, error: null });

    try {
      // Calculate quality counts
      const highCount = result.function_results.filter(f => f.quality === 'high').length;
      const mediumCount = result.function_results.filter(f => f.quality === 'medium').length;
      const lowCount = result.function_results.filter(f => f.quality === 'low').length;
      const poorCount = result.function_results.filter(f => f.quality === 'poor').length;
      const missingCount = result.function_results.filter(f => f.quality === 'missing').length;

      const feedbackResult = await evaluationApi.generateCodeSummaryFeedback({
        function_evaluations: JSON.stringify(result.function_results),
        total_functions: result.functions_evaluated,
        high_count: highCount,
        medium_count: mediumCount,
        low_count: lowCount,
        poor_count: poorCount,
        missing_count: missingCount,
        average_similarity: result.average_similarity,
        extra_functions: JSON.stringify(result.extra_functions),
        missing_functions: JSON.stringify(result.missing_functions),
        use_openai: useOpenAIFeedback
      });

      setSummaryFeedbackState({ loading: false, feedback: feedbackResult.feedback, error: null });
    } catch (err: any) {
      setSummaryFeedbackState({ 
        loading: false, 
        feedback: null, 
        error: err.response?.data?.detail || err.message || 'Failed to generate summary feedback' 
      });
    }
  };

  return (
    <div className="code-evaluation">
      <h1>Code Evaluation</h1>
      <p className="description">
        Upload ZIP files containing Python code to evaluate student submissions against ideal solutions.
      </p>

      <div className="evaluation-form">
        <div className="form-section">
          <h2>Upload Files</h2>
          
          <FileUpload
            label="Student Submission"
            file={submissionFile}
            onFileSelect={setSubmissionFile}
            accept=".zip"
            required
          />
          
          <FileUpload
            label="Ideal Solution"
            file={idealFile}
            onFileSelect={setIdealFile}
            accept=".zip"
            required
          />
        </div>

        <div className="form-section">
          <h2>Configuration</h2>
          
          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            disabled={isEvaluating}
          />
          
          {selectedModel === 'openai' && (
            <div className="checkbox-group">
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={useOpenAIFeedback}
                  onChange={(e) => setUseOpenAIFeedback(e.target.checked)}
                  disabled={isEvaluating}
                />
                <span>Use OpenAI for enhanced feedback</span>
              </label>
            </div>
          )}
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <div className="form-actions">
          <button
            type="button"
            onClick={handleEvaluate}
            disabled={isEvaluating || !submissionFile || !idealFile}
            className="btn btn-primary"
          >
            {isEvaluating ? 'Evaluating...' : 'Evaluate Code'}
          </button>
        </div>
      </div>

      {evaluationError && (
        <div className="error-message">
          <h3>Evaluation Error</h3>
          <p>{evaluationError}</p>
        </div>
      )}

      {result && (
        <div className="evaluation-results">
          <h2>Evaluation Results</h2>
          
          <div className="results-summary">
            <div className="summary-item">
              <span className="summary-label">Status:</span>
              <span className={`summary-value status-${result.status}`}>
                {result.status}
              </span>
            </div>
            
            {result.status === 'success' && (
              <>
                <div className="summary-item">
                  <span className="summary-label">Functions Evaluated:</span>
                  <span className="summary-value">{result.functions_evaluated}</span>
                </div>
                
                <div className="summary-item">
                  <span className="summary-label">Average Similarity:</span>
                  <span className="summary-value">{formatSimilarity(result.average_similarity)}</span>
                </div>
              </>
            )}
                     </div>

           {result.status === 'success' && (
             <div className="export-section">
               <h4>📄 Export Options</h4>
               <div className="export-actions">
                 <ExportDropdown
                   data={{
                     evaluationType: 'code',
                     timestamp: new Date().toLocaleString(),
                     modelUsed: selectedModel,
                     results: result,
                     feedback: summaryFeedbackState.feedback || undefined
                   }}
                 />
               </div>
             </div>
           )}

           {result.status === 'success' && result.function_results.length > 0 && (
            <div className="function-results">
              <h3>Function Results</h3>
              <div className="results-table">
                <table>
                  <thead>
                    <tr>
                      <th>Function</th>
                      <th>Similarity</th>
                      <th>Quality</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.function_results.map((func, index) => {
                      const feedbackState = feedbackStates[func.function_name] || { loading: false, feedback: null, error: null };
                      
                      return (
                        <tr key={index}>
                          <td>{func.function_name}</td>
                          <td>{formatSimilarity(func.similarity)}</td>
                          <td>
                            <span
                              className="quality-badge"
                              style={{ backgroundColor: getQualityColor(func.quality) }}
                            >
                              {formatQuality(func.quality)}
                            </span>
                          </td>
                          <td>
                            <div className="function-actions">
                              {!feedbackState.feedback && !feedbackState.loading && (
                                <button
                                  type="button"
                                  onClick={() => handleGenerateFeedback(func.function_name, func)}
                                  className="btn btn-secondary btn-sm"
                                  disabled={isGeneratingFeedback}
                                >
                                  Generate Feedback
                                </button>
                              )}
                              
                              {feedbackState.loading && (
                                <span className="loading-text">Generating feedback...</span>
                              )}
                              
                              {feedbackState.error && (
                                <div className="error-text">{feedbackState.error}</div>
                              )}
                              
                              {feedbackState.feedback && (
                                <div className="feedback-content">
                                  <h4>Detailed Feedback:</h4>
                                  <div className="feedback-text">
                                    {formatSimpleFeedback(feedbackState.feedback)}
                                  </div>
                                  <div className="feedback-actions">
                                    <button
                                      type="button"
                                      onClick={() => setFeedbackStates(prev => ({
                                        ...prev,
                                        [func.function_name]: { loading: false, feedback: null, error: null }
                                      }))}
                                      className="btn btn-secondary btn-sm"
                                    >
                                      Regenerate
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => exportFeedbackToTXT(
                                        feedbackState.feedback!,
                                        'code',
                                        func.function_name
                                      )}
                                      className="btn btn-primary btn-sm"
                                    >
                                      📄 Export Feedback
                                    </button>
                                  </div>
                                </div>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {result.extra_functions.length > 0 && (
            <div className="extra-functions">
              <h3>Extra Functions in Submission</h3>
              <ul>
                {result.extra_functions.map((func, index) => (
                  <li key={index}>{func}</li>
                ))}
              </ul>
            </div>
          )}

          {result.missing_functions.length > 0 && (
            <div className="missing-functions">
              <h3>Missing Functions</h3>
              <ul>
                {result.missing_functions.map((func, index) => (
                  <li key={index}>{func}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Summary Feedback Section */}
          <div className="summary-feedback-section">
            <h4>Comprehensive Feedback</h4>
            <div className="summary-feedback-actions">
              {!summaryFeedbackState.feedback && !summaryFeedbackState.loading && (
                <button
                  type="button"
                  onClick={handleGenerateSummaryFeedback}
                  className="btn btn-secondary"
                  disabled={isGeneratingFeedback}
                >
                  Generate Comprehensive Feedback
                </button>
              )}
              
              {summaryFeedbackState.loading && (
                <span className="loading-text">Generating comprehensive feedback...</span>
              )}
              
              {summaryFeedbackState.error && (
                <div className="error-text">{summaryFeedbackState.error}</div>
              )}
              
              {summaryFeedbackState.feedback && (
                <div className="feedback-content">
                  <h4>Comprehensive Analysis:</h4>
                  <div className="feedback-text">
                    <FormattedFeedback feedback={summaryFeedbackState.feedback} />
                  </div>
                  <div className="feedback-actions">
                    <button
                      type="button"
                      onClick={() => setSummaryFeedbackState({ loading: false, feedback: null, error: null })}
                      className="btn btn-secondary btn-sm"
                    >
                      Regenerate
                    </button>
                    <button
                      type="button"
                      onClick={() => exportFeedbackToTXT(
                        summaryFeedbackState.feedback!,
                        'code'
                      )}
                      className="btn btn-primary btn-sm"
                    >
                      📄 Export Comprehensive Feedback
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {result.message && (
            <div className="result-message">
              <p>{result.message}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CodeEvaluation;
