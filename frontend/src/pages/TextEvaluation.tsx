import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import ModelSelector from '../components/ModelSelector';
import { useFileUpload } from '../hooks/useFileUpload';
import { evaluationApi, TextEvaluationResult } from '../api/evaluationApi';
import { formatSimilarity, getQualityColor, formatQuality } from '../utils/formatters';
import { FormattedFeedback, formatSimpleFeedback } from '../utils/feedbackFormatter';
import ExportDropdown from '../components/ExportDropdown';
import { exportFeedbackToTXT } from '../utils/exportUtils';

const TextEvaluation: React.FC = () => {
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
  const [result, setResult] = useState<TextEvaluationResult | null>(null);
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
      const evaluationResult = await evaluationApi.evaluateText({
        submission: submissionFile,
        ideal: idealFile,
        model: selectedModel,
      });

      setResult(evaluationResult);
    } catch (err: any) {
      setEvaluationError(err.response?.data?.detail || err.message || 'Evaluation failed');
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleGenerateQuestionFeedback = async (questionIndex: number, evaluation: any) => {
    if (!result) return;

    const questionKey = `question_${questionIndex}`;
    setFeedbackStates(prev => ({
      ...prev,
      [questionKey]: { loading: true, feedback: null, error: null }
    }));

    try {
      const feedbackResult = await evaluationApi.generateTextFeedback({
        student_answer: evaluation.student_answer,
        reference_answer: evaluation.ideal_answer || 'Reference answer not available',
        similarity: evaluation.similarity,
        use_openai: useOpenAIFeedback
      });

      setFeedbackStates(prev => ({
        ...prev,
        [questionKey]: { loading: false, feedback: feedbackResult.feedback, error: null }
      }));
    } catch (err: any) {
      setFeedbackStates(prev => ({
        ...prev,
        [questionKey]: { 
          loading: false, 
          feedback: null, 
          error: err.response?.data?.detail || err.message || 'Failed to generate feedback' 
        }
      }));
    }
  };

  const handleGenerateSummaryFeedback = async () => {
    if (!result || !result.evaluations) return;

    setSummaryFeedbackState({ loading: true, feedback: null, error: null });

    try {
      const feedbackResult = await evaluationApi.generateSummaryFeedback({
        question_evaluations: JSON.stringify(result.evaluations),
        total_questions: result.stats.total_questions,
        high_count: result.stats.high_count,
        medium_count: result.stats.medium_count,
        low_count: result.stats.low_count,
        overall_score: result.overall_score,
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
    <div className="text-evaluation">
      <h1>Text Q&A Evaluation</h1>
      <p className="description">
        Upload DOCX files containing Q&A pairs to evaluate student submissions against ideal solutions.
      </p>

      <div className="evaluation-form">
        <div className="form-section">
          <h2>Upload Files</h2>
          
          <FileUpload
            label="Student Submission"
            file={submissionFile}
            onFileSelect={setSubmissionFile}
            accept=".docx,.doc,.txt"
            required
          />
          
          <FileUpload
            label="Ideal Solution"
            file={idealFile}
            onFileSelect={setIdealFile}
            accept=".docx,.doc,.txt"
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
            {isEvaluating ? 'Evaluating...' : 'Evaluate Text'}
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
                  <span className="summary-label">Overall Score:</span>
                  <span className="summary-value">{result.overall_score}%</span>
                </div>
                
                <div className="summary-item">
                  <span className="summary-label">Matched Questions:</span>
                  <span className="summary-value">{result.matched_questions}</span>
                </div>
                
                <div className="summary-item">
                  <span className="summary-label">Average Similarity:</span>
                  <span className="summary-value">{formatSimilarity(result.average_similarity)}</span>
                </div>
                
                <div className="summary-item">
                  <span className="summary-label">Model Used:</span>
                  <span className="summary-value">{result.model_used}</span>
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
                     evaluationType: 'text',
                     timestamp: new Date().toLocaleString(),
                     modelUsed: selectedModel,
                     overallScore: result.overall_score,
                     results: result,
                     feedback: summaryFeedbackState.feedback || undefined
                   }}
                 />
               </div>
             </div>
           )}

           {result.status === 'success' && result.evaluations.length > 0 && (
            <div className="question-results">
              <h3>Question Evaluations</h3>
              <div className="results-table">
                <table>
                  <thead>
                    <tr>
                      <th>Question</th>
                      <th>Student Answer</th>
                      <th>Similarity</th>
                      <th>Quality</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.evaluations.map((evaluation, index) => {
                      const questionKey = `question_${index}`;
                      const feedbackState = feedbackStates[questionKey] || { loading: false, feedback: null, error: null };
                      
                      return (
                        <tr key={index}>
                          <td>{evaluation.student_question}</td>
                          <td>{evaluation.student_answer}</td>
                          <td>{formatSimilarity(evaluation.similarity)}</td>
                          <td>
                            <span
                              className="quality-badge"
                              style={{ backgroundColor: getQualityColor(evaluation.quality) }}
                            >
                              {formatQuality(evaluation.quality)}
                            </span>
                          </td>
                          <td>
                            <div className="function-actions">
                              {!feedbackState.feedback && !feedbackState.loading && (
                                <button
                                  type="button"
                                  onClick={() => handleGenerateQuestionFeedback(index, evaluation)}
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
                                        [questionKey]: { loading: false, feedback: null, error: null }
                                      }))}
                                      className="btn btn-secondary btn-sm"
                                    >
                                      Regenerate
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => exportFeedbackToTXT(
                                        feedbackState.feedback!,
                                        'text',
                                        `Question_${index + 1}`
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

          {result.status === 'success' && result.stats && (
            <div className="evaluation-stats">
              <h3>Statistics</h3>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-label">Total Questions:</span>
                  <span className="stat-value">{result.stats.total_questions}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">High Quality:</span>
                  <span className="stat-value">{result.stats.high_count}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Medium Quality:</span>
                  <span className="stat-value">{result.stats.medium_count}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Low Quality:</span>
                  <span className="stat-value">{result.stats.low_count}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Poor Quality:</span>
                  <span className="stat-value">{result.stats.poor_count}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Missing:</span>
                  <span className="stat-value">{result.stats.missing_count}</span>
                </div>
              </div>
              
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
                            'text'
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
            </div>
          )}

          {result.summary && (
            <div className="evaluation-summary">
              <h3>Summary</h3>
              <div className="summary-text">
                {result.summary.split('\n').map((line, index) => (
                  <p key={index}>{line}</p>
                ))}
              </div>
            </div>
          )}

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

export default TextEvaluation;
