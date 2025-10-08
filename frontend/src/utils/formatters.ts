// Format file size in bytes to human readable format
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Format similarity score as percentage
export const formatSimilarity = (similarity: number): string => {
  return `${(similarity * 100).toFixed(1)}%`;
};

// Format quality level with color
export const getQualityColor = (quality: string | undefined | null): string => {
  if (!quality) {
    return '#6c757d'; // Gray for undefined/null quality
  }
  
  switch (quality.toLowerCase()) {
    case 'high':
      return '#28a745'; // Green
    case 'medium':
      return '#ffc107'; // Yellow
    case 'low':
      return '#fd7e14'; // Orange
    case 'poor':
      return '#dc3545'; // Red
    case 'missing':
      return '#6c757d'; // Gray
    default:
      return '#6c757d';
  }
};

// Format quality level for display
export const formatQuality = (quality: string | undefined | null): string => {
  if (!quality) {
    return 'Unknown';
  }
  
  return quality.charAt(0).toUpperCase() + quality.slice(1).toLowerCase();
};

// Format cost estimate
export const formatCost = (cost: number): string => {
  return `$${cost.toFixed(4)}`;
};
