import { useState, useCallback } from 'react';

interface FileUploadState {
  submissionFile: File | null;
  idealFile: File | null;
  isUploading: boolean;
  error: string | null;
}

interface FileUploadActions {
  setSubmissionFile: (file: File | null) => void;
  setIdealFile: (file: File | null) => void;
  clearFiles: () => void;
  validateFiles: () => boolean;
}

export const useFileUpload = (): FileUploadState & FileUploadActions => {
  const [submissionFile, setSubmissionFile] = useState<File | null>(null);
  const [idealFile, setIdealFile] = useState<File | null>(null);
  const [isUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const clearFiles = useCallback(() => {
    setSubmissionFile(null);
    setIdealFile(null);
    setError(null);
  }, []);

  const validateFiles = useCallback((): boolean => {
    setError(null);

    if (!submissionFile) {
      setError('Please select a submission file');
      return false;
    }

    if (!idealFile) {
      setError('Please select an ideal solution file');
      return false;
    }

    // Check file types
    const submissionExt = submissionFile.name.split('.').pop()?.toLowerCase();
    const idealExt = idealFile.name.split('.').pop()?.toLowerCase();

    const validExtensions = ['zip', 'docx', 'doc', 'txt'];

    if (!submissionExt || !validExtensions.includes(submissionExt)) {
      setError('Submission file must be a ZIP, DOCX, DOC, or TXT file');
      return false;
    }

    if (!idealExt || !validExtensions.includes(idealExt)) {
      setError('Ideal solution file must be a ZIP, DOCX, DOC, or TXT file');
      return false;
    }

    // Check file sizes (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (submissionFile.size > maxSize) {
      setError('Submission file size must be less than 50MB');
      return false;
    }

    if (idealFile.size > maxSize) {
      setError('Ideal solution file size must be less than 50MB');
      return false;
    }

    return true;
  }, [submissionFile, idealFile]);

  return {
    submissionFile,
    idealFile,
    isUploading,
    error,
    setSubmissionFile,
    setIdealFile,
    clearFiles,
    validateFiles,
  };
};
