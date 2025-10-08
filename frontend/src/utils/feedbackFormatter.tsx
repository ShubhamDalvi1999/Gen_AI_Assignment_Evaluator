import React from 'react';

interface FeedbackSection {
  title: string;
  content: string[];
  type: 'strengths' | 'improvements' | 'misconceptions' | 'study-plan' | 'resources' | 'assessment';
}

interface FormattedFeedbackProps {
  feedback: string;
}

const parseFeedback = (feedback: string): FeedbackSection[] => {
  const sections: FeedbackSection[] = [];
  const lines = feedback.split('\n').filter(line => line.trim());
  
  let currentSection: FeedbackSection | null = null;
  
  for (const line of lines) {
    const trimmedLine = line.trim();
    
    // Check for section headers
    if (trimmedLine.startsWith('### ')) {
      if (currentSection) {
        sections.push(currentSection);
      }
      
      const title = trimmedLine.replace('### ', '');
      let type: FeedbackSection['type'] = 'assessment';
      
      if (title.toLowerCase().includes('strength')) {
        type = 'strengths';
      } else if (title.toLowerCase().includes('improvement') || title.toLowerCase().includes('priority')) {
        type = 'improvements';
      } else if (title.toLowerCase().includes('misconception')) {
        type = 'misconceptions';
      } else if (title.toLowerCase().includes('study plan') || title.toLowerCase().includes('actionable')) {
        type = 'study-plan';
      } else if (title.toLowerCase().includes('resource') || title.toLowerCase().includes('reading')) {
        type = 'resources';
      }
      
      currentSection = {
        title,
        content: [],
        type
      };
    } else if (currentSection && trimmedLine) {
      currentSection.content.push(trimmedLine);
    }
  }
  
  if (currentSection) {
    sections.push(currentSection);
  }
  
  return sections;
};

const renderContent = (content: string[]): React.ReactNode => {
  return content.map((line, index) => {
    const trimmedLine = line.trim();
    
    // Handle numbered lists
    if (/^\d+\./.test(trimmedLine)) {
      return (
        <li key={index} className="action-item">
          {trimmedLine.replace(/^\d+\.\s*/, '')}
        </li>
      );
    }
    
    // Handle bullet points
    if (trimmedLine.startsWith('- ')) {
      return (
        <li key={index} className="action-item">
          {trimmedLine.replace('- ', '')}
        </li>
      );
    }
    
    // Handle bold text
    if (trimmedLine.includes('**')) {
      const parts = trimmedLine.split('**');
      return (
        <p key={index}>
          {parts.map((part, partIndex) => 
            partIndex % 2 === 1 ? <strong key={partIndex}>{part}</strong> : part
          )}
        </p>
      );
    }
    
    // Handle short-term and medium-term sections
    if (trimmedLine.includes('Short-term') || trimmedLine.includes('Medium-term')) {
      const isShortTerm = trimmedLine.includes('Short-term');
      const className = isShortTerm ? 'action-item short-term' : 'action-item medium-term';
      return (
        <div key={index} className={className}>
          <strong>{trimmedLine}</strong>
        </div>
      );
    }
    
    // Regular paragraph
    if (trimmedLine) {
      return <p key={index}>{trimmedLine}</p>;
    }
    
    return null;
  });
};

const getSectionIcon = (type: FeedbackSection['type']): string => {
  switch (type) {
    case 'strengths':
      return '✅';
    case 'improvements':
      return '🔧';
    case 'misconceptions':
      return '⚠️';
    case 'study-plan':
      return '📚';
    case 'resources':
      return '📖';
    default:
      return '📋';
  }
};

const getSectionClass = (type: FeedbackSection['type']): string => {
  return `feedback-section ${type}`;
};

export const FormattedFeedback: React.FC<FormattedFeedbackProps> = ({ feedback }) => {
  const sections = parseFeedback(feedback);
  
  return (
    <div className="formatted-feedback">
      {sections.map((section, index) => (
        <div key={index} className={getSectionClass(section.type)}>
          <div className="feedback-section-title">
            {getSectionIcon(section.type)} {section.title}
          </div>
          <div className="feedback-content">
            {renderContent(section.content)}
          </div>
        </div>
      ))}
    </div>
  );
};

// Simple text formatter for basic feedback (like code feedback)
export const formatSimpleFeedback = (feedback: string): React.ReactNode => {
  const lines = feedback.split('\n').filter(line => line.trim());
  
  return (
    <div className="simple-feedback">
      {lines.map((line, index) => {
        const trimmedLine = line.trim();
        
        if (trimmedLine.startsWith('**') && trimmedLine.endsWith('**')) {
          return (
            <h4 key={index}>
              {trimmedLine.replace(/\*\*/g, '')}
            </h4>
          );
        }
        
        if (trimmedLine.startsWith('- ')) {
          return (
            <li key={index} className="action-item">
              {trimmedLine.replace('- ', '')}
            </li>
          );
        }
        
        if (trimmedLine) {
          return <p key={index}>{trimmedLine}</p>;
        }
        
        return null;
      })}
    </div>
  );
};
