import React from 'react';
import { formatFileSize } from '../utils/formatters';

interface FileUploadProps {
  label: string;
  file: File | null;
  onFileSelect: (file: File | null) => void;
  accept?: string;
  required?: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({
  label,
  file,
  onFileSelect,
  accept = '.zip,.docx,.doc,.txt',
  required = false,
}) => {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0] || null;
    onFileSelect(selectedFile);
  };

  const handleRemoveFile = () => {
    onFileSelect(null);
  };

  return (
    <div className="file-upload">
      <label className="file-upload__label">
        {label}
        {required && <span className="required">*</span>}
      </label>
      
      <div className="file-upload__container">
        <input
          type="file"
          accept={accept}
          onChange={handleFileChange}
          className="file-upload__input"
          id={`file-${label.toLowerCase().replace(/\s+/g, '-')}`}
        />
        
        <label
          htmlFor={`file-${label.toLowerCase().replace(/\s+/g, '-')}`}
          className="file-upload__button"
        >
          {file ? 'Change File' : 'Choose File'}
        </label>
        
        {file && (
          <div className="file-upload__info">
            <span className="file-upload__name">{file.name}</span>
            <span className="file-upload__size">({formatFileSize(file.size)})</span>
            <button
              type="button"
              onClick={handleRemoveFile}
              className="file-upload__remove"
            >
              ×
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;
