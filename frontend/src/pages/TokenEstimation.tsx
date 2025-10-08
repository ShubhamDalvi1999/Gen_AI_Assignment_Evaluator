import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import ModelSelector from '../components/ModelSelector';
import { useFileUpload } from '../hooks/useFileUpload';
import { evaluationApi, TokenEstimateResult } from '../api/evaluationApi';
import { formatCost } from '../utils/formatters';

const TokenEstimation: React.FC = () => {
  const {
    submissionFile,
    idealFile,
    error,
    setSubmissionFile,
    setIdealFile,
    validateFiles,
  } = useFileUpload();

  const [selectedModel, setSelectedModel] = useState<'ollama' | 'openai'>('ollama');
  const [isEstimating, setIsEstimating] = useState(false);
  const [result, setResult] = useState<TokenEstimateResult | null>(null);
  const [estimationError, setEstimationError] = useState<string | null>(null);

  const handleEstimate = async () => {
    if (!validateFiles()) {
      return;
    }

    if (!submissionFile || !idealFile) {
      return;
    }

    setIsEstimating(true);
    setEstimationError(null);
    setResult(null);

    try {
      const estimationResult = await evaluationApi.estimateTokens({
        submission: submissionFile,
        ideal: idealFile,
        model: selectedModel,
      });

      setResult(estimationResult);
    } catch (err: any) {
      setEstimationError(err.response?.data?.detail || err.message || 'Estimation failed');
    } finally {
      setIsEstimating(false);
    }
  };

  return (
    <div className="token-estimation">
      <h1>Token Estimation</h1>
      <p className="description">
        Estimate token usage and potential costs for evaluating your files before processing.
      </p>

      <div className="estimation-form">
        <div className="form-section">
          <h2>Upload Files</h2>
          
          <FileUpload
            label="Student Submission"
            file={submissionFile}
            onFileSelect={setSubmissionFile}
            required
          />
          
          <FileUpload
            label="Ideal Solution"
            file={idealFile}
            onFileSelect={setIdealFile}
            required
          />
        </div>

        <div className="form-section">
          <h2>Configuration</h2>
          
          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            disabled={isEstimating}
          />
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <div className="form-actions">
          <button
            type="button"
            onClick={handleEstimate}
            disabled={isEstimating || !submissionFile || !idealFile}
            className="btn btn-primary"
          >
            {isEstimating ? 'Estimating...' : 'Estimate Tokens'}
          </button>
        </div>
      </div>

      {estimationError && (
        <div className="error-message">
          <h3>Estimation Error</h3>
          <p>{estimationError}</p>
        </div>
      )}

      {result && (
        <div className="estimation-results">
          <h2>Estimation Results</h2>
          
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
                  <span className="summary-label">Estimated Tokens:</span>
                  <span className="summary-value">{result.estimated_tokens.toLocaleString()}</span>
                </div>
                
                {result.cost_estimate && (
                  <div className="summary-item">
                    <span className="summary-label">Estimated Cost:</span>
                    <span className="summary-value">{formatCost(result.cost_estimate)}</span>
                  </div>
                )}
              </>
            )}
          </div>

          {result.warnings && result.warnings.length > 0 && (
            <div className="estimation-warnings">
              <h3>Warnings</h3>
              <ul className="warnings-list">
                {result.warnings.map((warning, index) => (
                  <li key={index} className="warning-item">
                    ⚠️ {warning}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.message && (
            <div className="result-message">
              <p>{result.message}</p>
            </div>
          )}

          <div className="estimation-info">
            <h3>About Token Estimation</h3>
            <div className="info-content">
              <p>
                <strong>What are tokens?</strong> Tokens are the basic units of text that AI models process. 
                They can be words, parts of words, or punctuation marks.
              </p>
              <p>
                <strong>Cost implications:</strong> Token usage directly affects the cost of AI model operations. 
                The estimation helps you understand potential expenses before processing.
              </p>
              <p>
                <strong>Safe limits:</strong> The system has built-in safety limits to prevent excessive token usage. 
                Warnings will appear if estimates exceed recommended thresholds.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TokenEstimation;
