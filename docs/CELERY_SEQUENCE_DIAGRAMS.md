# Celery Integration Sequence Diagrams

This document contains detailed sequence diagrams showing the complete flow of Celery integration in the AI Assignment Evaluator project.

## 📋 **Table of Contents**

1. [Complete Code Evaluation Flow](#complete-code-evaluation-flow)
2. [Text Q&A Evaluation Flow](#text-qa-evaluation-flow)
3. [Progress Tracking Flow](#progress-tracking-flow)
4. [Error Handling & Recovery Flow](#error-handling--recovery-flow)
5. [Task Orchestration Flow](#task-orchestration-flow)
6. [Worker Communication Flow](#worker-communication-flow)

---

## 🚀 **Complete Code Evaluation Flow**

This diagram shows the complete flow of a code evaluation using Celery, from initial request to final results.

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Redis
    participant CeleryBroker
    participant EmbeddingWorker
    participant LLMWorker
    participant ProcessingWorker
    participant EvaluationWorker
    participant MongoDB
    participant OpenAIAPI
    participant OllamaAPI

    Note over Client, OllamaAPI: 🚀 Code Evaluation with Celery Integration

    %% Initial Request
    Client->>FastAPI: POST /celery/evaluate/code<br/>(ZIP files + config)
    FastAPI->>FastAPI: Save temp files<br/>Generate evaluation_id
    FastAPI->>CeleryBroker: evaluate_code_parallel_task.delay()
    CeleryBroker->>Redis: Store task in queue
    FastAPI->>Client: Return task_id + evaluation_id<br/>Status: PENDING

    %% Task Processing Begins
    CeleryBroker->>EvaluationWorker: Pick up evaluation task
    EvaluationWorker->>EvaluationWorker: Start fresh RAG session<br/>Clear previous logs

    %% Step 1: File Processing (Parallel)
    par File Processing
        EvaluationWorker->>ProcessingWorker: extract_functions_from_zip_task<br/>(student ZIP)
        EvaluationWorker->>ProcessingWorker: extract_functions_from_zip_task<br/>(ideal ZIP)
    end
    ProcessingWorker->>ProcessingWorker: Extract Python functions<br/>Parse code structure
    ProcessingWorker->>Redis: Return extracted functions
    EvaluationWorker->>Redis: Collect file processing results

    %% Step 2: Ideal Embeddings (Parallel)
    EvaluationWorker->>EmbeddingWorker: generate_ideal_embeddings_task<br/>(all ideal functions)
    loop For each ideal function
        par Embedding Generation
            EmbeddingWorker->>OllamaAPI: Generate embedding<br/>(if model=ollama)
            EmbeddingWorker->>OpenAIAPI: Generate embedding<br/>(if model=openai)
        end
        EmbeddingWorker->>MongoDB: Store ideal embedding
        EmbeddingWorker->>Redis: Update progress
    end
    EmbeddingWorker->>Redis: Return all ideal embeddings

    %% Step 3: Student Embeddings & Comparison (Parallel)
    loop For each student function
        par Student Processing
            EvaluationWorker->>EmbeddingWorker: generate_embedding_task<br/>(student function)
            EmbeddingWorker->>OllamaAPI: Generate student embedding
            EmbeddingWorker->>EvaluationWorker: Return student embedding
            EvaluationWorker->>EvaluationWorker: Calculate similarity<br/>with ideal embedding
        end
    end

    %% Step 4: Code Analysis (Parallel)
    loop For each matched function
        EvaluationWorker->>ProcessingWorker: analyze_code_structure_task<br/>(student vs ideal)
        ProcessingWorker->>ProcessingWorker: Analyze syntax, structure<br/>Generate recommendations
        ProcessingWorker->>Redis: Return structure analysis
    end

    %% Step 5: Feedback Generation (Parallel)
    loop For each function
        par Feedback Generation
            EvaluationWorker->>LLMWorker: generate_code_feedback_task<br/>(similarity + analysis)
            LLMWorker->>OpenAIAPI: Generate feedback<br/>(if use_openai_feedback=true)
            LLMWorker->>OllamaAPI: Generate feedback<br/>(if use_openai_feedback=false)
            LLMWorker->>Redis: Return feedback text
        end
    end

    %% Step 6: Compile Results
    EvaluationWorker->>EvaluationWorker: Calculate overall score<br/>Compile function results<br/>Generate summary
    EvaluationWorker->>Redis: Store final results
    EvaluationWorker->>FastAPI: Task completed (SUCCESS)

    %% Client Status Check
    Client->>FastAPI: GET /celery/status/{evaluation_id}
    FastAPI->>Redis: Check task status
    Redis->>FastAPI: Return task result
    FastAPI->>Client: Return complete evaluation results

    %% Cleanup
    FastAPI->>FastAPI: Delete temporary files<br/>Clean up resources
```

---

## 📄 **Text Q&A Evaluation Flow**

This diagram shows the complete flow of a text Q&A evaluation using Celery.

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Redis
    participant CeleryBroker
    participant EmbeddingWorker
    participant LLMWorker
    participant ProcessingWorker
    participant EvaluationWorker
    participant MongoDB
    participant OpenAIAPI
    participant OllamaAPI

    Note over Client, OllamaAPI: 📄 Text Q&A Evaluation with Celery Integration

    %% Initial Request
    Client->>FastAPI: POST /celery/evaluate/text<br/>(DOCX files + config)
    FastAPI->>FastAPI: Save temp files<br/>Generate evaluation_id
    FastAPI->>CeleryBroker: evaluate_text_parallel_task.delay()
    CeleryBroker->>Redis: Store task in queue
    FastAPI->>Client: Return task_id + evaluation_id<br/>Status: PENDING

    %% Task Processing Begins
    CeleryBroker->>EvaluationWorker: Pick up evaluation task
    EvaluationWorker->>EvaluationWorker: Start fresh RAG session<br/>Clear previous logs

    %% Step 1: Document Processing (Parallel)
    par Document Processing
        EvaluationWorker->>ProcessingWorker: process_docx_document_task<br/>(student DOCX)
        EvaluationWorker->>ProcessingWorker: process_docx_document_task<br/>(ideal DOCX)
    end
    ProcessingWorker->>ProcessingWorker: Extract Q&A pairs<br/>Parse document structure
    ProcessingWorker->>Redis: Return Q&A pairs
    EvaluationWorker->>Redis: Collect document processing results

    %% Step 2: Question Embeddings (Parallel)
    loop For each ideal question
        par Question Embedding
            EvaluationWorker->>EmbeddingWorker: generate_embedding_task<br/>(ideal question)
            EmbeddingWorker->>OllamaAPI: Generate question embedding
            EmbeddingWorker->>MongoDB: Store ideal question embedding
        end
    end

    loop For each student question
        par Student Question Embedding
            EvaluationWorker->>EmbeddingWorker: generate_embedding_task<br/>(student question)
            EmbeddingWorker->>OllamaAPI: Generate student question embedding
        end
    end

    %% Step 3: Question Matching (Parallel)
    loop For each student question
        EvaluationWorker->>EvaluationWorker: Find best matching ideal question<br/>using cosine similarity
        alt Question similarity > threshold
            EvaluationWorker->>EvaluationWorker: Mark as matched
        else Question similarity < threshold
            EvaluationWorker->>EvaluationWorker: Mark as unmatched
        end
    end

    %% Step 4: Answer Embeddings & Comparison (Parallel)
    loop For each matched Q&A pair
        par Answer Processing
            EvaluationWorker->>EmbeddingWorker: generate_embedding_task<br/>(student answer)
            EvaluationWorker->>EmbeddingWorker: generate_embedding_task<br/>(ideal answer)
            EmbeddingWorker->>OllamaAPI: Generate answer embeddings
            EmbeddingWorker->>EvaluationWorker: Return answer embeddings
            EvaluationWorker->>EvaluationWorker: Calculate answer similarity
        end
    end

    %% Step 5: Feedback Generation (Parallel)
    loop For each matched answer
        par Answer Feedback
            EvaluationWorker->>LLMWorker: generate_text_feedback_task<br/>(student vs ideal answer)
            LLMWorker->>OpenAIAPI: Generate feedback<br/>(if use_openai=true)
            LLMWorker->>OllamaAPI: Generate feedback<br/>(if use_openai=false)
            LLMWorker->>Redis: Return feedback text
        end
    end

    %% Step 6: Compile Results
    EvaluationWorker->>EvaluationWorker: Calculate overall score<br/>Compile question results<br/>Generate summary
    EvaluationWorker->>Redis: Store final results
    EvaluationWorker->>FastAPI: Task completed (SUCCESS)

    %% Client Status Check
    Client->>FastAPI: GET /celery/status/{evaluation_id}
    FastAPI->>Redis: Check task status
    Redis->>FastAPI: Return task result
    FastAPI->>Client: Return complete evaluation results

    %% Cleanup
    FastAPI->>FastAPI: Delete temporary files<br/>Clean up resources
```

---

## 📊 **Progress Tracking Flow**

This diagram shows how real-time progress tracking works throughout the evaluation process.

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Redis
    participant CeleryBroker
    participant Worker

    Note over Client, Worker: 📊 Real-time Progress Tracking

    Client->>FastAPI: POST /celery/evaluate/code
    FastAPI->>CeleryBroker: Start evaluation task
    FastAPI->>Client: Return task_id (PENDING)

    %% Progress Updates
    loop During Processing
        Worker->>Redis: Update task state<br/>PROGRESS: "Processing files..." (10%)
        Client->>FastAPI: GET /celery/status/{evaluation_id}
        FastAPI->>Redis: Get task status
        Redis->>FastAPI: Return progress: 10%
        FastAPI->>Client: Status: PROGRESS, Progress: 10%

        Worker->>Redis: Update task state<br/>PROGRESS: "Generating embeddings..." (30%)
        Client->>FastAPI: GET /celery/status/{evaluation_id}
        FastAPI->>Redis: Get task status
        Redis->>FastAPI: Return progress: 30%
        FastAPI->>Client: Status: PROGRESS, Progress: 30%

        Worker->>Redis: Update task state<br/>PROGRESS: "Comparing functions..." (60%)
        Client->>FastAPI: GET /celery/status/{evaluation_id}
        FastAPI->>Redis: Get task status
        Redis->>FastAPI: Return progress: 60%
        FastAPI->>Client: Status: PROGRESS, Progress: 60%

        Worker->>Redis: Update task state<br/>PROGRESS: "Generating feedback..." (80%)
        Client->>FastAPI: GET /celery/status/{evaluation_id}
        FastAPI->>Redis: Get task status
        Redis->>FastAPI: Return progress: 80%
        FastAPI->>Client: Status: PROGRESS, Progress: 80%
    end

    %% Completion
    Worker->>Redis: Update task state<br/>SUCCESS: "Evaluation completed" (100%)
    Client->>FastAPI: GET /celery/status/{evaluation_id}
    FastAPI->>Redis: Get task status
    Redis->>FastAPI: Return final results
    FastAPI->>Client: Status: SUCCESS, Results: {...}
```

---

## 🛡️ **Error Handling & Recovery Flow**

This diagram shows how the system handles errors and implements recovery mechanisms.

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Redis
    participant CeleryBroker
    participant Worker
    participant FallbackWorker

    Note over Client, FallbackWorker: 🛡️ Error Handling & Recovery

    Client->>FastAPI: POST /celery/evaluate/code
    FastAPI->>CeleryBroker: Start evaluation task
    FastAPI->>Client: Return task_id (PENDING)

    CeleryBroker->>Worker: Assign evaluation task
    Worker->>Worker: Start processing

    %% Error Occurs
    Worker->>Worker: Embedding generation fails<br/>(API timeout/error)
    Worker->>Redis: Update task state<br/>FAILURE: "Embedding failed"
    
    %% Client checks status
    Client->>FastAPI: GET /celery/status/{evaluation_id}
    FastAPI->>Redis: Get task status
    Redis->>FastAPI: Return error status
    FastAPI->>Client: Status: FAILURE, Error: "Embedding failed"

    %% Retry with fallback
    alt Automatic Retry
        CeleryBroker->>FallbackWorker: Retry with different model<br/>(Ollama → OpenAI)
        FallbackWorker->>FallbackWorker: Process with fallback
        FallbackWorker->>Redis: Update task state<br/>SUCCESS: "Completed with fallback"
    else Manual Retry
        Client->>FastAPI: POST /celery/evaluate/code<br/>(with different config)
        FastAPI->>CeleryBroker: Start new evaluation task
        FastAPI->>Client: Return new task_id
    end

    %% Final Success
    Client->>FastAPI: GET /celery/status/{evaluation_id}
    FastAPI->>Redis: Get task status
    Redis->>FastAPI: Return success results
    FastAPI->>Client: Status: SUCCESS, Results: {...}
```

---

## 🎭 **Task Orchestration Flow**

This diagram shows how different types of workers coordinate to complete complex evaluations.

```mermaid
sequenceDiagram
    participant EvaluationWorker
    participant EmbeddingWorker
    participant LLMWorker
    participant ProcessingWorker
    participant Redis
    participant MongoDB

    Note over EvaluationWorker, MongoDB: 🎭 Task Orchestration & Coordination

    %% Initial Orchestration
    EvaluationWorker->>EvaluationWorker: Start evaluation orchestration
    EvaluationWorker->>Redis: Create task coordination plan

    %% Phase 1: File Processing
    par File Processing Phase
        EvaluationWorker->>ProcessingWorker: extract_functions_from_zip_task
        EvaluationWorker->>ProcessingWorker: process_docx_document_task
    end
    ProcessingWorker->>Redis: Return processing results
    EvaluationWorker->>Redis: Collect all processing results

    %% Phase 2: Embedding Generation
    par Embedding Phase
        EvaluationWorker->>EmbeddingWorker: generate_ideal_embeddings_task
        EvaluationWorker->>EmbeddingWorker: generate_embeddings_batch_task
    end
    EmbeddingWorker->>MongoDB: Store embeddings
    EmbeddingWorker->>Redis: Return embedding results
    EvaluationWorker->>Redis: Collect all embedding results

    %% Phase 3: Analysis & Comparison
    EvaluationWorker->>EvaluationWorker: Calculate similarities<br/>Perform analysis
    EvaluationWorker->>Redis: Store analysis results

    %% Phase 4: Feedback Generation
    par Feedback Phase
        EvaluationWorker->>LLMWorker: generate_feedback_batch_task
        EvaluationWorker->>LLMWorker: generate_summary_feedback_task
    end
    LLMWorker->>Redis: Return feedback results
    EvaluationWorker->>Redis: Collect all feedback results

    %% Final Compilation
    EvaluationWorker->>EvaluationWorker: Compile final results<br/>Calculate scores
    EvaluationWorker->>Redis: Store final evaluation results
    EvaluationWorker->>Redis: Mark task as completed
```

---

## 🔄 **Worker Communication Flow**

This diagram shows how different worker types communicate and share data through Redis.

```mermaid
sequenceDiagram
    participant EmbeddingWorker1
    participant EmbeddingWorker2
    participant LLMWorker1
    participant LLMWorker2
    participant ProcessingWorker1
    participant Redis
    participant MongoDB

    Note over EmbeddingWorker1, MongoDB: 🔄 Worker Communication & Data Sharing

    %% Shared Data Storage
    ProcessingWorker1->>Redis: Store extracted functions<br/>Key: "functions:{task_id}"
    ProcessingWorker1->>Redis: Store Q&A pairs<br/>Key: "qa_pairs:{task_id}"

    %% Embedding Workers Coordination
    par Parallel Embedding Generation
        EmbeddingWorker1->>Redis: Get functions from "functions:{task_id}"
        EmbeddingWorker2->>Redis: Get functions from "functions:{task_id}"
    end

    EmbeddingWorker1->>MongoDB: Store ideal embeddings
    EmbeddingWorker2->>MongoDB: Store student embeddings

    EmbeddingWorker1->>Redis: Store embedding results<br/>Key: "embeddings:{task_id}:ideal"
    EmbeddingWorker2->>Redis: Store embedding results<br/>Key: "embeddings:{task_id}:student"

    %% LLM Workers Coordination
    par Parallel Feedback Generation
        LLMWorker1->>Redis: Get embedding results from "embeddings:{task_id}:ideal"
        LLMWorker2->>Redis: Get embedding results from "embeddings:{task_id}:student"
    end

    LLMWorker1->>LLMWorker1: Generate code feedback
    LLMWorker2->>LLMWorker2: Generate text feedback

    LLMWorker1->>Redis: Store feedback results<br/>Key: "feedback:{task_id}:code"
    LLMWorker2->>Redis: Store feedback results<br/>Key: "feedback:{task_id}:text"

    %% Result Aggregation
    Redis->>Redis: Aggregate all results<br/>Key: "final_results:{task_id}"
```

---

## 📈 **Performance Monitoring Flow**

This diagram shows how the system monitors performance and resource usage.

```mermaid
sequenceDiagram
    participant Flower
    participant Redis
    participant Worker1
    participant Worker2
    participant Worker3
    participant FastAPI

    Note over Flower, FastAPI: 📈 Performance Monitoring & Metrics

    %% Worker Registration
    Worker1->>Redis: Register worker status<br/>Queue: embeddings, Status: active
    Worker2->>Redis: Register worker status<br/>Queue: llm, Status: active
    Worker3->>Redis: Register worker status<br/>Queue: processing, Status: active

    %% Task Execution Monitoring
    Worker1->>Redis: Update task progress<br/>Task: embedding_001, Progress: 50%
    Worker2->>Redis: Update task progress<br/>Task: feedback_001, Progress: 75%
    Worker3->>Redis: Update task progress<br/>Task: processing_001, Progress: 25%

    %% Flower Monitoring
    Flower->>Redis: Query worker statuses
    Redis->>Flower: Return worker metrics
    Flower->>Flower: Calculate performance stats<br/>Tasks/min, Queue lengths, Worker load

    %% API Monitoring
    FastAPI->>Redis: Query task statistics
    Redis->>FastAPI: Return task metrics
    FastAPI->>FastAPI: Calculate API performance<br/>Response times, Success rates

    %% Performance Alerts
    alt High Load Detected
        Flower->>Flower: Alert: High queue length<br/>Recommend: Scale workers
    else Worker Failure Detected
        Flower->>Flower: Alert: Worker offline<br/>Recommend: Restart worker
    end
```

---

## 🔧 **Configuration & Scaling Flow**

This diagram shows how the system can be configured and scaled dynamically.

```mermaid
sequenceDiagram
    participant Admin
    participant FastAPI
    participant CeleryBroker
    participant Redis
    participant NewWorker
    participant ExistingWorker

    Note over Admin, ExistingWorker: 🔧 Dynamic Configuration & Scaling

    %% Configuration Update
    Admin->>FastAPI: Update worker configuration<br/>Increase embedding workers to 8
    FastAPI->>Redis: Store new configuration<br/>Key: "config:workers"
    FastAPI->>CeleryBroker: Apply configuration changes

    %% Worker Scaling
    Admin->>NewWorker: Start new embedding worker<br/>Concurrency: 4
    NewWorker->>Redis: Register new worker<br/>Queue: embeddings, Concurrency: 4
    NewWorker->>CeleryBroker: Join worker pool

    %% Load Balancing
    CeleryBroker->>Redis: Query worker capacities
    Redis->>CeleryBroker: Return worker statuses
    CeleryBroker->>CeleryBroker: Redistribute tasks<br/>Balance load across workers

    %% Performance Optimization
    ExistingWorker->>Redis: Report performance metrics
    Redis->>FastAPI: Aggregate performance data
    FastAPI->>FastAPI: Analyze performance trends<br/>Identify bottlenecks

    %% Auto-scaling Decision
    alt High Load Detected
        FastAPI->>NewWorker: Auto-scale: Start additional worker
        NewWorker->>Redis: Register additional worker
    else Low Load Detected
        FastAPI->>ExistingWorker: Auto-scale: Reduce worker concurrency
        ExistingWorker->>Redis: Update worker concurrency
    end
```

---

## 📝 **Notes**

### **Key Benefits Shown in These Diagrams:**

1. **Parallel Processing**: Multiple workers can process different parts simultaneously
2. **Fault Tolerance**: Error handling and recovery mechanisms are built-in
3. **Progress Tracking**: Real-time status updates throughout the process
4. **Resource Management**: Workers can be scaled dynamically based on load
5. **Data Sharing**: Redis acts as a central coordination point for all workers
6. **Monitoring**: Comprehensive monitoring and alerting capabilities

### **Performance Improvements:**

- **Sequential Processing**: 1 task at a time
- **Parallel Processing**: 10+ tasks simultaneously
- **Speed Improvement**: 3-4x faster evaluations
- **Scalability**: Horizontal scaling across multiple machines
- **Reliability**: Automatic retry and fallback mechanisms

### **Use Cases:**

- **Single Evaluation**: Use synchronous endpoints for simple cases
- **Batch Evaluations**: Use Celery for multiple concurrent evaluations
- **Production Deployments**: Use Celery for high-availability systems
- **Development/Testing**: Use synchronous endpoints for quick testing
