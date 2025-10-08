# AI Assignment Checker - Frontend

A modern React TypeScript frontend for the AI Assignment Checker application. This frontend provides an intuitive user interface for evaluating student submissions against ideal solutions using AI-powered similarity analysis.

## Features

- **Code Evaluation**: Upload ZIP files containing Python code for function-level evaluation
- **Text Q&A Evaluation**: Upload DOCX files for question-answer pair evaluation
- **Token Estimation**: Estimate token usage and costs before processing
- **Model Selection**: Choose between Ollama (local) and OpenAI (cloud) models
- **Real-time Feedback**: Get detailed evaluation results with similarity scores and quality assessments
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## Technology Stack

- **React 18** with TypeScript
- **Axios** for API communication
- **CSS3** with modern design patterns
- **Create React App** for build tooling

## Project Structure

```
frontend/
├── src/
│   ├── api/                 # API client and services
│   │   ├── apiClient.ts     # Axios configuration
│   │   └── evaluationApi.ts # Evaluation API endpoints
│   ├── components/          # Reusable UI components
│   │   ├── FileUpload.tsx   # File upload component
│   │   └── ModelSelector.tsx # Model selection component
│   ├── hooks/               # Custom React hooks
│   │   └── useFileUpload.ts # File upload logic
│   ├── pages/               # Page components
│   │   ├── CodeEvaluation.tsx
│   │   ├── TextEvaluation.tsx
│   │   └── TokenEstimation.tsx
│   ├── utils/               # Utility functions
│   │   └── formatters.ts    # Data formatting utilities
│   ├── App.tsx              # Main application component
│   ├── App.css              # Application styles
│   └── index.tsx            # Application entry point
├── public/                  # Static assets
└── package.json             # Dependencies and scripts
```

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Backend server running (see backend README)

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Environment Configuration

Create a `.env` file in the frontend directory to configure the API endpoint:

```env
REACT_APP_API_URL=http://localhost:8000
```

## Usage

### Code Evaluation

1. Navigate to the "Code Evaluation" tab
2. Upload a ZIP file containing student Python code
3. Upload a ZIP file containing the ideal solution
4. Select your preferred embedding model (Ollama or OpenAI)
5. Click "Evaluate Code" to start the evaluation
6. Review the results showing function similarity scores and feedback

### Text Q&A Evaluation

1. Navigate to the "Text Q&A Evaluation" tab
2. Upload a DOCX file containing student Q&A pairs
3. Upload a DOCX file containing ideal Q&A pairs
4. Select your preferred embedding model
5. Click "Evaluate Text" to start the evaluation
6. Review the results showing question-answer similarity and overall scores

### Token Estimation

1. Navigate to the "Token Estimation" tab
2. Upload your submission and ideal solution files
3. Select your preferred embedding model
4. Click "Estimate Tokens" to get usage estimates
5. Review the estimated token count and potential costs

## API Integration

The frontend communicates with the FastAPI backend through the following endpoints:

- `GET /api/v1/health` - Health check
- `POST /api/v1/evaluate/code` - Code evaluation
- `POST /api/v1/evaluate/text` - Text evaluation
- `POST /api/v1/estimate/tokens` - Token estimation

## File Requirements

### Code Evaluation
- **Submission**: ZIP file containing Python code files
- **Ideal Solution**: ZIP file containing Python code files
- **Size Limit**: 50MB per file

### Text Evaluation
- **Submission**: DOCX, DOC, or TXT file containing Q&A pairs
- **Ideal Solution**: DOCX, DOC, or TXT file containing Q&A pairs
- **Size Limit**: 50MB per file

## Model Options

### Ollama (Local)
- **Pros**: Free, runs locally, no API keys required
- **Cons**: Requires Ollama server running locally
- **Setup**: Install and run Ollama server on your machine

### OpenAI (Cloud)
- **Pros**: High-quality embeddings, no local setup required
- **Cons**: Requires API key, incurs costs
- **Setup**: Configure OpenAI API key in backend settings

## Development

### Available Scripts

- `npm start` - Start development server
- `npm test` - Run test suite
- `npm run build` - Build for production
- `npm run eject` - Eject from Create React App (irreversible)

### Code Style

- Use TypeScript for type safety
- Follow React functional component patterns
- Use CSS classes for styling
- Implement responsive design principles

### Testing

Run the test suite:
```bash
npm test
```

## Deployment

### Production Build

1. Create a production build:
   ```bash
   npm run build
   ```

2. The build artifacts will be stored in the `build/` directory

3. Deploy the contents of the `build/` directory to your web server

### Environment Variables

For production deployment, set the following environment variables:

```env
REACT_APP_API_URL=https://your-backend-domain.com
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Ensure the backend server is running
   - Check the `REACT_APP_API_URL` environment variable
   - Verify CORS settings in the backend

2. **File Upload Issues**
   - Check file size limits (50MB)
   - Verify file format requirements
   - Ensure files are not corrupted

3. **Model Selection Issues**
   - For Ollama: Ensure Ollama server is running locally
   - For OpenAI: Verify API key configuration in backend

### Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the backend README for API documentation
- Review the browser console for error messages
- Ensure all prerequisites are properly installed
