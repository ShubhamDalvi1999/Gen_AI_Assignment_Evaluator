import React, { useState } from 'react';
import './App.css';
import CodeEvaluation from './pages/CodeEvaluation';
import TextEvaluation from './pages/TextEvaluation';
import TokenEstimation from './pages/TokenEstimation';
import CeleryCodeEvaluation from './pages/CeleryCodeEvaluation';
import CeleryTextEvaluation from './pages/CeleryTextEvaluation';

type Page = 'code' | 'text' | 'tokens' | 'celery-code' | 'celery-text';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<Page>('code');

  const renderPage = () => {
    switch (currentPage) {
      case 'code':
        return <CodeEvaluation />;
      case 'text':
        return <TextEvaluation />;
      case 'tokens':
        return <TokenEstimation />;
      case 'celery-code':
        return <CeleryCodeEvaluation />;
      case 'celery-text':
        return <CeleryTextEvaluation />;
      default:
        return <CodeEvaluation />;
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1 className="app-title">AI Assignment Checker</h1>
          <p className="app-subtitle">Intelligent evaluation of student submissions</p>
        </div>
      </header>

      <nav className="App-nav">
        <div className="nav-container">
          {/* Synchronous Evaluation Section */}
          <div className="nav-section">
            <h3 className="nav-section-title">📝 Synchronous Evaluation</h3>
            <div className="nav-buttons-group">
              <button
                className={`nav-button ${currentPage === 'code' ? 'active' : ''}`}
                onClick={() => setCurrentPage('code')}
              >
                Code Evaluation
              </button>
              <button
                className={`nav-button ${currentPage === 'text' ? 'active' : ''}`}
                onClick={() => setCurrentPage('text')}
              >
                Text Q&A Evaluation
              </button>
            </div>
          </div>

          {/* Celery Evaluation Section */}
          <div className="nav-section">
            <h3 className="nav-section-title">🚀 Parallel Evaluation (Celery)</h3>
            <div className="nav-buttons-group">
              <button
                className={`nav-button celery-button ${currentPage === 'celery-code' ? 'active' : ''}`}
                onClick={() => setCurrentPage('celery-code')}
              >
                Code Evaluation
              </button>
              <button
                className={`nav-button celery-button ${currentPage === 'celery-text' ? 'active' : ''}`}
                onClick={() => setCurrentPage('celery-text')}
              >
                Text Q&A Evaluation
              </button>
            </div>
          </div>

          {/* Utility Section */}
          <div className="nav-section">
            <h3 className="nav-section-title">🔧 Utilities</h3>
            <div className="nav-buttons-group">
              <button
                className={`nav-button ${currentPage === 'tokens' ? 'active' : ''}`}
                onClick={() => setCurrentPage('tokens')}
              >
                Token Estimation
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="App-main">
        <div className="main-container">
          {renderPage()}
        </div>
      </main>

      <footer className="App-footer">
        <div className="footer-content">
          <p>&copy; 2024 AI Assignment Checker. Built with React and FastAPI.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
