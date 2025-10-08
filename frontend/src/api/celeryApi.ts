import React from 'react';
import apiClient from './apiClient';

// Types for Celery API requests and responses
export interface CeleryCodeEvaluateRequest {
  submission: File;
  ideal: File;
  model: 'ollama' | 'openai';
  use_openai_feedback?: boolean;
}

export interface CeleryTextEvaluateRequest {
  submission: File;
  ideal: File;
  model: 'ollama' | 'openai';
}

export interface CeleryTaskResponse {
  evaluation_id: string;
  task_id: string;
  status: 'PENDING' | 'PROGRESS' | 'SUCCESS' | 'FAILURE';
  message: string;
  check_status_url: string;
  progress?: number;
  result?: any;
  error?: string;
}

export interface CeleryStatusResponse {
  evaluation_id: string;
  task_id: string;
  status: 'PENDING' | 'PROGRESS' | 'SUCCESS' | 'FAILURE';
  message: string;
  progress?: number;
  result?: any;
  error?: string;
}

export interface ActiveTask {
  evaluation_id: string;
  task_id: string;
  started_at: string;
  model: string;
}

export interface ActiveTasksResponse {
  total_tasks: number;
  active_tasks: ActiveTask[];
}

// Celery API service functions
export const celeryApi = {
  // Health check endpoint
  async healthCheck(): Promise<{ status: string; message: string }> {
    const response = await apiClient.get('/api/v1/health');
    return response.data as { status: string; message: string };
  },

  // Start Celery code evaluation
  async startCodeEvaluation(request: CeleryCodeEvaluateRequest): Promise<CeleryTaskResponse> {
    const formData = new FormData();
    formData.append('submission', request.submission);
    formData.append('ideal', request.ideal);
    formData.append('model', request.model);
    if (request.use_openai_feedback !== undefined) {
      formData.append('use_openai_feedback', request.use_openai_feedback.toString());
    }

    const response = await apiClient.post('/api/v1/celery/evaluate/code', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data as CeleryTaskResponse;
  },

  // Start Celery text evaluation
  async startTextEvaluation(request: CeleryTextEvaluateRequest): Promise<CeleryTaskResponse> {
    const formData = new FormData();
    formData.append('submission', request.submission);
    formData.append('ideal', request.ideal);
    formData.append('model', request.model);

    const response = await apiClient.post('/api/v1/celery/evaluate/text', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data as CeleryTaskResponse;
  },

  // Check evaluation status
  async checkStatus(evaluationId: string): Promise<CeleryStatusResponse> {
    const response = await apiClient.get(`/api/v1/celery/status/${evaluationId}`);
    return response.data as CeleryStatusResponse;
  },

  // List active tasks
  async listActiveTasks(): Promise<ActiveTasksResponse> {
    const response = await apiClient.get('/api/v1/celery/tasks');
    return response.data as ActiveTasksResponse;
  },

  // Cancel evaluation
  async cancelEvaluation(evaluationId: string): Promise<{ status: string; message: string }> {
    const response = await apiClient.delete(`/api/v1/celery/tasks/${evaluationId}`);
    return response.data as { status: string; message: string };
  },

  // Poll for status updates (utility function)
  async pollStatus(
    evaluationId: string, 
    onProgress?: (status: CeleryStatusResponse) => void,
    onComplete?: (result: any) => void,
    onError?: (error: string) => void,
    interval: number = 2000,
    maxAttempts: number = 300 // 10 minutes max
  ): Promise<CeleryStatusResponse> {
    let attempts = 0;
    
    const poll = async (): Promise<CeleryStatusResponse> => {
      if (attempts >= maxAttempts) {
        throw new Error('Polling timeout: Maximum attempts reached');
      }
      
      attempts++;
      const status = await this.checkStatus(evaluationId);
      
      // Call progress callback
      if (onProgress && status.status === 'PROGRESS') {
        onProgress(status);
      }
      
      // Check if completed
      if (status.status === 'SUCCESS') {
        if (onComplete) {
          onComplete(status.result);
        }
        return status;
      }
      
      // Check if failed
      if (status.status === 'FAILURE') {
        if (onError) {
          onError(status.error || 'Task failed');
        }
        throw new Error(status.error || 'Task failed');
      }
      
      // Still pending or in progress, continue polling
      await new Promise(resolve => setTimeout(resolve, interval));
      return poll();
    };
    
    return poll();
  }
};

// Hook for using Celery evaluations in React components
export const useCeleryEvaluation = () => {
  const [isEvaluating, setIsEvaluating] = React.useState(false);
  const [evaluationId, setEvaluationId] = React.useState<string | null>(null);
  const [progress, setProgress] = React.useState(0);
  const [status, setStatus] = React.useState<string>('');
  const [result, setResult] = React.useState<any>(null);
  const [error, setError] = React.useState<string | null>(null);

  const startCodeEvaluation = async (request: CeleryCodeEvaluateRequest) => {
    try {
      setIsEvaluating(true);
      setError(null);
      setResult(null);
      setProgress(0);
      
      const taskResponse = await celeryApi.startCodeEvaluation(request);
      setEvaluationId(taskResponse.evaluation_id);
      
      // Start polling for status
      await celeryApi.pollStatus(
        taskResponse.evaluation_id,
        (status) => {
          setProgress(status.progress || 0);
          setStatus(status.message);
        },
        (result) => {
          setResult(result);
          setIsEvaluating(false);
        },
        (error) => {
          setError(error);
          setIsEvaluating(false);
        }
      );
    } catch (err: any) {
      setError(err.message || 'Evaluation failed');
      setIsEvaluating(false);
    }
  };

  const startTextEvaluation = async (request: CeleryTextEvaluateRequest) => {
    try {
      setIsEvaluating(true);
      setError(null);
      setResult(null);
      setProgress(0);
      
      const taskResponse = await celeryApi.startTextEvaluation(request);
      setEvaluationId(taskResponse.evaluation_id);
      
      // Start polling for status
      await celeryApi.pollStatus(
        taskResponse.evaluation_id,
        (status) => {
          setProgress(status.progress || 0);
          setStatus(status.message);
        },
        (result) => {
          setResult(result);
          setIsEvaluating(false);
        },
        (error) => {
          setError(error);
          setIsEvaluating(false);
        }
      );
    } catch (err: any) {
      setError(err.message || 'Evaluation failed');
      setIsEvaluating(false);
    }
  };

  const cancelEvaluation = async () => {
    if (evaluationId) {
      try {
        await celeryApi.cancelEvaluation(evaluationId);
        setIsEvaluating(false);
        setEvaluationId(null);
        setProgress(0);
        setStatus('');
      } catch (err: any) {
        setError(err.message || 'Failed to cancel evaluation');
      }
    }
  };

  return {
    isEvaluating,
    evaluationId,
    progress,
    status,
    result,
    error,
    startCodeEvaluation,
    startTextEvaluation,
    cancelEvaluation,
  };
};
