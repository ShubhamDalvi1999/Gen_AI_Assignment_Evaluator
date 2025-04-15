# AI Assignment Checker

An advanced tool for evaluating student assignments using AI techniques. This application leverages Retrieval Augmented Generation (RAG) to provide high-quality feedback on both code submissions and text-based Q&A responses.

## Features

- **Code Evaluation**: Compare student code against ideal solutions with structure analysis and similarity scoring
- **Text Q&A Evaluation**: Assess student answers against reference answers with quality ratings
- **RAG Pipeline**: Uses vector embeddings to enhance feedback quality through context retrieval
- **Comprehensive Logging**: Detailed logging of all stages in the RAG process
- **Token Estimation**: Calculate token usage before processing submissions
- **Multiple AI Models**: Support for both local (Ollama) and cloud (OpenAI) models
- **Detailed Feedback**: Generate comprehensive feedback for students with specific improvement suggestions
- **Enhanced UI**: Improved visualization of evaluation results and summaries

## Architecture

The application is built using a modern architecture:

- **Backend**: FastAPI for high-performance API endpoints
- **Frontend**: HTML/JavaScript for a responsive interface
- **Embedding Models**: Support for both local (Ollama) and cloud (OpenAI) embedding generation
- **Database**: MongoDB for storing embeddings and evaluation results
- **RAG Pipeline**: Custom implementation with retrieval, augmentation and generation stages

## RAG Process Stages

The application implements a comprehensive RAG pipeline with detailed logging:

1. **Document Processing Stage**: Extracts Q&A pairs from student and reference documents
2. **Embedding Generation Stage**: Converts text to vector embeddings
3. **Embedding Storage Stage**: Stores embeddings in MongoDB for retrieval
4. **Retrieval Stage**: Finds most relevant reference content for each student submission
5. **Question Mapping Stage**: Maps student questions to reference questions
6. **Augmentation Stage**: Enriches context with similarity metrics and analysis
7. **Evaluation Stage**: Evaluates the quality of student answers
8. **Scoring Stage**: Calculates overall scores and metrics
9. **Generation Stage**: Produces detailed feedback for students
10. **Summary Generation Stage**: Creates a comprehensive evaluation summary

## Installation

### Prerequisites

- Python 3.8 or higher
- MongoDB (optional, will use in-memory storage if not available)
- Ollama (optional, for local model support)

### Windows Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/ai-assignment-checker.git
   cd ai-assignment-checker
   ```

2. **Run the startup script**:
   ```
   start_app.bat
   ```
   
   This will:
   - Create a virtual environment if it doesn't exist
   - Install all dependencies
   - Start the application

3. **Access the application**:
   Open your browser and navigate to: http://localhost:8001

### Manual Installation (All Platforms)

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/ai-assignment-checker.git
   cd ai-assignment-checker
   ```

2. **Create a virtual environment**:
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

5. **Set up environment variables** (Optional):
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   OLLAMA_API_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2:3b
   MONGODB_URI=mongodb://localhost:27017/
   MONGODB_DB_NAME=assignment_checker
   MONGODB_COLLECTION_NAME=embeddings
   MONGODB_QA_COLLECTION_NAME=qa_embeddings
   LOG_LEVEL=INFO
   SIMILARITY_THRESHOLD=0.8
   ```

6. **Run the application**:
   ```
   python app.py
   ```

7. **Access the application**:
   Open your browser and navigate to: http://localhost:8001

## Usage

### Code Evaluation

1. Navigate to the "Code Evaluation" tab
2. Upload a student submission ZIP file containing Python code
3. Upload an ideal solution ZIP file with reference implementations
4. Select the embedding model (Ollama or OpenAI)
5. Check "Use OpenAI for Better Feedback Generation" for enhanced feedback (requires an API key)
6. Click "Check Tokens" to estimate token usage (optional)
7. Click "Evaluate Code" to process the submission
8. Review the detailed results including:
   - Function-by-function similarity scores
   - Code structure analysis
   - Automated recommendations
   - AI-generated feedback

### Text Q&A Evaluation

1. Navigate to the "Text Q&A Evaluation" tab
2. Upload a student submission DOCX file containing questions and answers
3. Upload an ideal answer DOCX file with reference answers
4. Check "Use OpenAI for Better Results" for enhanced evaluation (requires an API key)
5. Click "Evaluate Text" to process the submission
6. Review the comprehensive evaluation including:
   - Question-by-question quality assessment
   - Similarity scores for each answer
   - AI-generated feedback for improvement
   - Overall performance summary with recommendations

## Logging

The application includes comprehensive logging of all stages in the RAG process. Logs are stored in the `logs` directory and include:

- **app.log**: General application logs
- **mongodb.log**: Database operation logs
- **code_analyzer.log**: Code analysis logs
- **embedding.log**: Embedding generation logs
- **feedback.log**: Feedback generation logs

To change the log level, set the `LOG_LEVEL` environment variable to one of: DEBUG, INFO, WARNING, ERROR.

## Troubleshooting

### Common Issues

- **"This site can't be reached"** - Make sure you're accessing http://localhost:8001 in your browser, not http://0.0.0.0:8001
- **MongoDB connection errors** - Check that MongoDB is running or remove the MONGODB_URI from .env to use in-memory storage
- **Ollama errors** - Ensure Ollama is running locally or set up the correct API URL in .env
- **OpenAI API errors** - Verify your API key is correct and has sufficient credits
- **Missing logs** - Ensure the `logs` directory exists and is writable

### For Windows Users

If you encounter issues with the batch file:
1. Open a command prompt
2. Navigate to the project directory
3. Run these commands manually:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   python app.py
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
