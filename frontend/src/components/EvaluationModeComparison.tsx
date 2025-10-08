import React from 'react';

interface EvaluationModeComparisonProps {
  currentMode: 'synchronous' | 'celery';
  onModeChange: (mode: 'synchronous' | 'celery') => void;
}

const EvaluationModeComparison: React.FC<EvaluationModeComparisonProps> = ({
  currentMode,
  onModeChange
}) => {
  return (
    <div className="evaluation-mode-comparison">
      <div className="comparison-header">
        <h3>Choose Evaluation Mode</h3>
        <p>Select between synchronous and parallel processing approaches</p>
      </div>

      <div className="mode-options">
        {/* Synchronous Mode */}
        <div 
          className={`mode-option ${currentMode === 'synchronous' ? 'selected' : ''}`}
          onClick={() => onModeChange('synchronous')}
        >
          <div className="mode-header">
            <div className="mode-icon">📝</div>
            <h4>Synchronous Evaluation</h4>
            <div className={`mode-badge ${currentMode === 'synchronous' ? 'active' : ''}`}>
              {currentMode === 'synchronous' ? 'Selected' : 'Available'}
            </div>
          </div>
          
          <div className="mode-description">
            <p>Traditional sequential processing approach</p>
          </div>

          <div className="mode-features">
            <div className="feature-list">
              <div className="feature-item">
                <span className="feature-icon">⏱️</span>
                <span>Processing Time: 25-50 seconds</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">🔄</span>
                <span>Sequential processing</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">👤</span>
                <span>Single user at a time</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">📊</span>
                <span>No progress tracking</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">🛑</span>
                <span>Cannot cancel</span>
              </div>
            </div>
          </div>

          <div className="mode-pros-cons">
            <div className="pros">
              <h5>✅ Pros</h5>
              <ul>
                <li>Simple and reliable</li>
                <li>No additional setup required</li>
                <li>Immediate results</li>
                <li>Lower resource usage</li>
              </ul>
            </div>
            <div className="cons">
              <h5>❌ Cons</h5>
              <ul>
                <li>Slower processing</li>
                <li>Blocks user interface</li>
                <li>No progress feedback</li>
                <li>Limited scalability</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Celery Mode */}
        <div 
          className={`mode-option celery-mode ${currentMode === 'celery' ? 'selected' : ''}`}
          onClick={() => onModeChange('celery')}
        >
          <div className="mode-header">
            <div className="mode-icon">🚀</div>
            <h4>Parallel Evaluation (Celery)</h4>
            <div className={`mode-badge celery-badge ${currentMode === 'celery' ? 'active' : ''}`}>
              {currentMode === 'celery' ? 'Selected' : 'Available'}
            </div>
          </div>
          
          <div className="mode-description">
            <p>Advanced parallel processing with real-time progress</p>
          </div>

          <div className="mode-features">
            <div className="feature-list">
              <div className="feature-item">
                <span className="feature-icon">⚡</span>
                <span>Processing Time: 8-15 seconds</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">🔄</span>
                <span>Parallel processing</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">👥</span>
                <span>Multiple users simultaneously</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">📊</span>
                <span>Real-time progress tracking</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">🛑</span>
                <span>Can cancel anytime</span>
              </div>
            </div>
          </div>

          <div className="mode-pros-cons">
            <div className="pros">
              <h5>✅ Pros</h5>
              <ul>
                <li>3-4x faster processing</li>
                <li>Real-time progress updates</li>
                <li>Non-blocking interface</li>
                <li>Highly scalable</li>
                <li>Task cancellation</li>
                <li>Concurrent evaluations</li>
              </ul>
            </div>
            <div className="cons">
              <h5>❌ Cons</h5>
              <ul>
                <li>Requires Redis setup</li>
                <li>More complex architecture</li>
                <li>Higher resource usage</li>
                <li>Additional monitoring needed</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="comparison-summary">
        <div className="summary-card">
          <h4>Performance Comparison</h4>
          <div className="performance-metrics">
            <div className="metric">
              <span className="metric-label">Speed</span>
              <div className="metric-bar">
                <div className="metric-fill synchronous" style={{width: '30%'}}>
                  <span>Synchronous: 30%</span>
                </div>
                <div className="metric-fill celery" style={{width: '100%'}}>
                  <span>Celery: 100%</span>
                </div>
              </div>
            </div>
            <div className="metric">
              <span className="metric-label">User Experience</span>
              <div className="metric-bar">
                <div className="metric-fill synchronous" style={{width: '40%'}}>
                  <span>Synchronous: 40%</span>
                </div>
                <div className="metric-fill celery" style={{width: '100%'}}>
                  <span>Celery: 100%</span>
                </div>
              </div>
            </div>
            <div className="metric">
              <span className="metric-label">Scalability</span>
              <div className="metric-bar">
                <div className="metric-fill synchronous" style={{width: '25%'}}>
                  <span>Synchronous: 25%</span>
                </div>
                <div className="metric-fill celery" style={{width: '100%'}}>
                  <span>Celery: 100%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EvaluationModeComparison;
