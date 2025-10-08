import React, { useState, useRef, useEffect } from 'react';
import { ExportData, exportToJSON, exportToCSV, exportToTXT, exportComprehensiveReport } from '../utils/exportUtils';

interface ExportDropdownProps {
  data: ExportData;
  className?: string;
}

const ExportDropdown: React.FC<ExportDropdownProps> = ({ data, className = '' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleExport = (exportFunction: () => void) => {
    exportFunction();
    setIsOpen(false);
  };

  return (
    <div className={`export-dropdown ${className}`} ref={dropdownRef}>
      <button
        className="btn btn-secondary"
        onClick={() => setIsOpen(!isOpen)}
        type="button"
      >
        📄 Export Results
        <span className={`dropdown-arrow ${isOpen ? 'open' : ''}`}>▼</span>
      </button>
      
      {isOpen && (
        <div className="export-options">
          <button
            className="export-option"
            onClick={() => handleExport(() => exportToJSON(data))}
            type="button"
          >
            📊 JSON Format
            <span className="export-description">Structured data for analysis</span>
          </button>
          
          <button
            className="export-option"
            onClick={() => handleExport(() => exportToCSV(data))}
            type="button"
          >
            📈 CSV Format
            <span className="export-description">Spreadsheet compatible</span>
          </button>
          
          <button
            className="export-option"
            onClick={() => handleExport(() => exportToTXT(data))}
            type="button"
          >
            📝 Text Report
            <span className="export-description">Human-readable format</span>
          </button>
          
          {data.feedback && (
            <button
              className="export-option"
              onClick={() => handleExport(() => exportComprehensiveReport(data))}
              type="button"
            >
              📋 Comprehensive Report
              <span className="export-description">Results + Feedback</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default ExportDropdown;
