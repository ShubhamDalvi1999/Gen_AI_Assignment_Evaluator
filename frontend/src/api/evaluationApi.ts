import apiClient from './apiClient';

// Types for API requests and responses
export interface CodeEvaluateRequest {
  submission: File;
  ideal: File;
  model: 'ollama' | 'openai';
  use_openai_feedback?: boolean;
}

export interface TextEvaluateRequest {
  submission: File;
  ideal: File;
  model: 'ollama' | 'openai';
}

export interface TokenEstimateRequest {
  submission: File;
  ideal: File;
  model: 'ollama' | 'openai';
}

export interface EvaluationResult {
  status: string;
  message?: string;
}

export interface CodeEvaluationResult extends EvaluationResult {
  functions_evaluated: number;
  average_similarity: number;
  function_results: Array<{
    function_name: string;
    similarity: number;
    quality: string;
    feedback: string;
  }>;
  extra_functions: string[];
  missing_functions: string[];
}

export interface TextEvaluationResult extends EvaluationResult {
  session_id?: string;
  matched_questions: number;
  average_similarity: number;
  processed_questions: Array<{
    student_qa_id: string;
    ideal_qa_id?: string;
    question_similarity: number;
    answer_similarity: number;
    similarity: number;
    quality: string;
    student_question: string;
    student_answer: string;
  }>;
  model_used: string;
  overall_score: number;
  evaluations: Array<{
    student_qa_id: string;
    student_question: string;
    student_answer: string;
    quality: string;
    similarity: number;
    feedback: string;
    ideal_question?: string;
    ideal_answer?: string;
  }>;
  summary: string;
  stats: {
    total_questions: number;
    high_count: number;
    medium_count: number;
    low_count: number;
    poor_count: number;
    missing_count: number;
  };
}

export interface TokenEstimateResult extends EvaluationResult {
  estimated_tokens: number;
  cost_estimate?: number;
  warnings: string[];
}

// API service functions
export const evaluationApi = {
  // Health check endpoint
  async healthCheck(): Promise<{ status: string; message: string }> {
    const response = await apiClient.get('/api/v1/health');
    return response.data as { status: string; message: string };
  },

  // Code evaluation
  async evaluateCode(request: CodeEvaluateRequest): Promise<CodeEvaluationResult> {
    const formData = new FormData();
    formData.append('submission', request.submission);
    formData.append('ideal', request.ideal);
    formData.append('model', request.model);
    if (request.use_openai_feedback !== undefined) {
      formData.append('use_openai_feedback', request.use_openai_feedback.toString());
    }

    const response = await apiClient.post('/api/v1/evaluate/code', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data as CodeEvaluationResult;
  },

  // Text evaluation
  async evaluateText(request: TextEvaluateRequest): Promise<TextEvaluationResult> {
    const formData = new FormData();
    formData.append('submission', request.submission);
    formData.append('ideal', request.ideal);
    formData.append('model', request.model);

    const response = await apiClient.post('/api/v1/evaluate/text', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data as TextEvaluationResult;
  },

  // Token estimation
  async estimateTokens(request: TokenEstimateRequest): Promise<TokenEstimateResult> {
    const formData = new FormData();
    formData.append('submission', request.submission);
    formData.append('ideal', request.ideal);
    formData.append('model', request.model);

    const response = await apiClient.post('/api/v1/estimate/tokens', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data as TokenEstimateResult;
  },

  // Feedback API endpoints
  async generateCodeFeedback(request: {
    student_code: string;
    ideal_code: string;
    similarity: number;
    structure_analysis: string;
    similar_contexts?: string;
    use_openai?: boolean;
  }): Promise<{ status: string; feedback: string; generated_at: string }> {
    const formData = new FormData();
    formData.append('student_code', request.student_code);
    formData.append('ideal_code', request.ideal_code);
    formData.append('similarity', request.similarity.toString());
    formData.append('structure_analysis', request.structure_analysis);
    formData.append('similar_contexts', request.similar_contexts || '[]');
    formData.append('use_openai', (request.use_openai || false).toString());
    
    const response = await apiClient.post('/api/v1/feedback/code', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data as { status: string; feedback: string; generated_at: string };
  },

  async generateTextFeedback(request: {
    student_answer: string;
    reference_answer: string;
    similarity: number;
    use_openai?: boolean;
  }): Promise<{ status: string; feedback: string; generated_at: string }> {
    const formData = new FormData();
    formData.append('student_answer', request.student_answer);
    formData.append('reference_answer', request.reference_answer);
    formData.append('similarity', request.similarity.toString());
    formData.append('use_openai', (request.use_openai || false).toString());
    
    const response = await apiClient.post('/api/v1/feedback/text', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data as { status: string; feedback: string; generated_at: string };
  },

  async generateSummaryFeedback(request: {
    question_evaluations: string;
    total_questions: number;
    high_count: number;
    medium_count: number;
    low_count: number;
    overall_score: number;
    use_openai?: boolean;
  }): Promise<{ status: string; feedback: string; generated_at: string }> {
    const formData = new FormData();
    formData.append('question_evaluations', request.question_evaluations);
    formData.append('total_questions', request.total_questions.toString());
    formData.append('high_count', request.high_count.toString());
    formData.append('medium_count', request.medium_count.toString());
    formData.append('low_count', request.low_count.toString());
    formData.append('overall_score', request.overall_score.toString());
    formData.append('use_openai', (request.use_openai || false).toString());
    
    const response = await apiClient.post('/api/v1/feedback/summary', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data as { status: string; feedback: string; generated_at: string };
  },

  async generateCodeSummaryFeedback(request: {
    function_evaluations: string;
    total_functions: number;
    high_count: number;
    medium_count: number;
    low_count: number;
    poor_count: number;
    missing_count: number;
    average_similarity: number;
    extra_functions: string;
    missing_functions: string;
    use_openai?: boolean;
  }): Promise<{ status: string; feedback: string; generated_at: string }> {
    const formData = new FormData();
    formData.append('function_evaluations', request.function_evaluations);
    formData.append('total_functions', request.total_functions.toString());
    formData.append('high_count', request.high_count.toString());
    formData.append('medium_count', request.medium_count.toString());
    formData.append('low_count', request.low_count.toString());
    formData.append('poor_count', request.poor_count.toString());
    formData.append('missing_count', request.missing_count.toString());
    formData.append('average_similarity', request.average_similarity.toString());
    formData.append('extra_functions', request.extra_functions);
    formData.append('missing_functions', request.missing_functions);
    formData.append('use_openai', (request.use_openai || false).toString());
    
    const response = await apiClient.post('/api/v1/feedback/code-summary', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data as { status: string; feedback: string; generated_at: string };
  }
};
