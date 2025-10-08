import React from 'react';

interface ModelSelectorProps {
  selectedModel: 'ollama' | 'openai';
  onModelChange: (model: 'ollama' | 'openai') => void;
  disabled?: boolean;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onModelChange,
  disabled = false,
}) => {
  return (
    <div className="model-selector">
      <label className="model-selector__label">
        Embedding Model
        <span className="required">*</span>
      </label>
      
      <div className="model-selector__options">
        <label className="model-selector__option">
          <input
            type="radio"
            name="model"
            value="ollama"
            checked={selectedModel === 'ollama'}
            onChange={(e) => onModelChange(e.target.value as 'ollama' | 'openai')}
            disabled={disabled}
          />
          <span className="model-selector__text">
            <strong>Ollama</strong> (Local - Free)
          </span>
        </label>
        
        <label className="model-selector__option">
          <input
            type="radio"
            name="model"
            value="openai"
            checked={selectedModel === 'openai'}
            onChange={(e) => onModelChange(e.target.value as 'ollama' | 'openai')}
            disabled={disabled}
          />
          <span className="model-selector__text">
            <strong>OpenAI</strong> (Cloud - Paid)
          </span>
        </label>
      </div>
      
      <div className="model-selector__info">
        {selectedModel === 'ollama' && (
          <p className="model-selector__description">
            Uses local Ollama server. Make sure Ollama is running on your machine.
          </p>
        )}
        {selectedModel === 'openai' && (
          <p className="model-selector__description">
            Uses OpenAI API. Requires valid API key in backend configuration.
          </p>
        )}
      </div>
    </div>
  );
};

export default ModelSelector;
