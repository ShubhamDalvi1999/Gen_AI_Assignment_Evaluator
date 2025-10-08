// Export utilities for evaluation results and feedback

export interface ExportData {
  evaluationType: 'code' | 'text';
  timestamp: string;
  modelUsed: string;
  overallScore?: number;
  results: any;
  feedback?: any;
}

// Export to JSON format
export const exportToJSON = (data: ExportData): void => {
  const jsonString = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `evaluation_results_${data.evaluationType}_${new Date().toISOString().split('T')[0]}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

// Export to CSV format
export const exportToCSV = (data: ExportData): void => {
  let csvContent = '';
  
  // Header
  csvContent += `Evaluation Type,${data.evaluationType}\n`;
  csvContent += `Timestamp,${data.timestamp}\n`;
  csvContent += `Model Used,${data.modelUsed}\n`;
  if (data.overallScore !== undefined) {
    csvContent += `Overall Score,${data.overallScore}%\n`;
  }
  csvContent += '\n';
  
  if (data.evaluationType === 'code') {
    // Code evaluation results
    csvContent += 'Function Name,Similarity,Quality\n';
    data.results.function_results?.forEach((func: any) => {
      csvContent += `"${func.function_name}",${func.similarity},${func.quality}\n`;
    });
    
    if (data.results.extra_functions?.length > 0) {
      csvContent += '\nExtra Functions\n';
      data.results.extra_functions.forEach((func: string) => {
        csvContent += `"${func}"\n`;
      });
    }
    
    if (data.results.missing_functions?.length > 0) {
      csvContent += '\nMissing Functions\n';
      data.results.missing_functions.forEach((func: string) => {
        csvContent += `"${func}"\n`;
      });
    }
  } else {
    // Text evaluation results
    csvContent += 'Question,Student Answer,Similarity,Quality\n';
    data.results.evaluations?.forEach((evaluation: any) => {
      csvContent += `"${evaluation.student_question}","${evaluation.student_answer}",${evaluation.similarity},${evaluation.quality}\n`;
    });
  }
  
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `evaluation_results_${data.evaluationType}_${new Date().toISOString().split('T')[0]}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

// Export to TXT format (human-readable)
export const exportToTXT = (data: ExportData): void => {
  let txtContent = '';
  
  // Header
  txtContent += `AI Assignment Checker - Evaluation Report\n`;
  txtContent += `==========================================\n\n`;
  txtContent += `Evaluation Type: ${data.evaluationType.toUpperCase()}\n`;
  txtContent += `Timestamp: ${data.timestamp}\n`;
  txtContent += `Model Used: ${data.modelUsed}\n`;
  if (data.overallScore !== undefined) {
    txtContent += `Overall Score: ${data.overallScore}%\n`;
  }
  txtContent += '\n';
  
  if (data.evaluationType === 'code') {
    // Code evaluation results
    txtContent += `FUNCTIONS EVALUATED: ${data.results.functions_evaluated}\n`;
    txtContent += `AVERAGE SIMILARITY: ${(data.results.average_similarity * 100).toFixed(2)}%\n\n`;
    
    txtContent += 'DETAILED RESULTS:\n';
    txtContent += '=================\n';
    data.results.function_results?.forEach((func: any, index: number) => {
      txtContent += `${index + 1}. ${func.function_name}\n`;
      txtContent += `   Similarity: ${(func.similarity * 100).toFixed(2)}%\n`;
      txtContent += `   Quality: ${func.quality}\n`;
      txtContent += '\n';
    });
    
    if (data.results.extra_functions?.length > 0) {
      txtContent += 'EXTRA FUNCTIONS IN SUBMISSION:\n';
      txtContent += '==============================\n';
      data.results.extra_functions.forEach((func: string) => {
        txtContent += `- ${func}\n`;
      });
      txtContent += '\n';
    }
    
    if (data.results.missing_functions?.length > 0) {
      txtContent += 'MISSING FUNCTIONS:\n';
      txtContent += '==================\n';
      data.results.missing_functions.forEach((func: string) => {
        txtContent += `- ${func}\n`;
      });
      txtContent += '\n';
    }
  } else {
    // Text evaluation results
    txtContent += `MATCHED QUESTIONS: ${data.results.matched_questions}\n`;
    txtContent += `AVERAGE SIMILARITY: ${(data.results.average_similarity * 100).toFixed(2)}%\n\n`;
    
    txtContent += 'QUESTION EVALUATIONS:\n';
    txtContent += '=====================\n';
    data.results.evaluations?.forEach((evaluation: any, index: number) => {
      txtContent += `${index + 1}. Question: ${evaluation.student_question}\n`;
      txtContent += `   Student Answer: ${evaluation.student_answer}\n`;
      txtContent += `   Similarity: ${(evaluation.similarity * 100).toFixed(2)}%\n`;
      txtContent += `   Quality: ${evaluation.quality}\n`;
      txtContent += '\n';
    });
    
    if (data.results.stats) {
      txtContent += 'STATISTICS:\n';
      txtContent += '===========\n';
      txtContent += `Total Questions: ${data.results.stats.total_questions}\n`;
      txtContent += `High Quality: ${data.results.stats.high_count}\n`;
      txtContent += `Medium Quality: ${data.results.stats.medium_count}\n`;
      txtContent += `Low Quality: ${data.results.stats.low_count}\n`;
      txtContent += `Poor Quality: ${data.results.stats.poor_count}\n`;
      txtContent += `Missing: ${data.results.stats.missing_count}\n\n`;
    }
  }
  
  // Add feedback if available
  if (data.feedback) {
    txtContent += 'FEEDBACK:\n';
    txtContent += '=========\n';
    txtContent += data.feedback;
    txtContent += '\n';
  }
  
  const blob = new Blob([txtContent], { type: 'text/plain;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `evaluation_report_${data.evaluationType}_${new Date().toISOString().split('T')[0]}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

// Export feedback only to TXT
export const exportFeedbackToTXT = (feedback: string, evaluationType: 'code' | 'text', functionName?: string): void => {
  let txtContent = '';
  
  txtContent += `AI Assignment Checker - Feedback Report\n`;
  txtContent += `======================================\n\n`;
  txtContent += `Evaluation Type: ${evaluationType.toUpperCase()}\n`;
  if (functionName) {
    txtContent += `Function: ${functionName}\n`;
  }
  txtContent += `Generated: ${new Date().toLocaleString()}\n\n`;
  txtContent += 'FEEDBACK:\n';
  txtContent += '=========\n';
  txtContent += feedback;
  
  const blob = new Blob([txtContent], { type: 'text/plain;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  const fileName = functionName 
    ? `feedback_${evaluationType}_${functionName}_${new Date().toISOString().split('T')[0]}.txt`
    : `feedback_${evaluationType}_${new Date().toISOString().split('T')[0]}.txt`;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

// Export comprehensive report (results + feedback)
export const exportComprehensiveReport = (data: ExportData): void => {
  let reportContent = '';
  
  // Header
  reportContent += `AI Assignment Checker - Comprehensive Report\n`;
  reportContent += `============================================\n\n`;
  reportContent += `Evaluation Type: ${data.evaluationType.toUpperCase()}\n`;
  reportContent += `Timestamp: ${data.timestamp}\n`;
  reportContent += `Model Used: ${data.modelUsed}\n`;
  if (data.overallScore !== undefined) {
    reportContent += `Overall Score: ${data.overallScore}%\n`;
  }
  reportContent += '\n';
  
  // Results section
  reportContent += 'EVALUATION RESULTS:\n';
  reportContent += '===================\n';
  
  if (data.evaluationType === 'code') {
    reportContent += `Functions Evaluated: ${data.results.functions_evaluated}\n`;
    reportContent += `Average Similarity: ${(data.results.average_similarity * 100).toFixed(2)}%\n\n`;
    
    data.results.function_results?.forEach((func: any, index: number) => {
      reportContent += `${index + 1}. ${func.function_name}\n`;
      reportContent += `   Similarity: ${(func.similarity * 100).toFixed(2)}%\n`;
      reportContent += `   Quality: ${func.quality}\n`;
      reportContent += '\n';
    });
  } else {
    reportContent += `Matched Questions: ${data.results.matched_questions}\n`;
    reportContent += `Average Similarity: ${(data.results.average_similarity * 100).toFixed(2)}%\n\n`;
    
    data.results.evaluations?.forEach((evaluation: any, index: number) => {
      reportContent += `${index + 1}. Question: ${evaluation.student_question}\n`;
      reportContent += `   Student Answer: ${evaluation.student_answer}\n`;
      reportContent += `   Similarity: ${(evaluation.similarity * 100).toFixed(2)}%\n`;
      reportContent += `   Quality: ${evaluation.quality}\n`;
      reportContent += '\n';
    });
  }
  
  // Feedback section
  if (data.feedback) {
    reportContent += 'COMPREHENSIVE FEEDBACK:\n';
    reportContent += '=======================\n';
    reportContent += data.feedback;
    reportContent += '\n';
  }
  
  const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `comprehensive_report_${data.evaluationType}_${new Date().toISOString().split('T')[0]}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
