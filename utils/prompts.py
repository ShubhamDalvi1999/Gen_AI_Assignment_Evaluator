"""
Prompt templates for LLM-based feedback generation.
"""

FEEDBACK_PROMPT_TEMPLATE = """
You are an AI code reviewer evaluating a student's function implementation against an ideal solution. Your task is to provide constructive feedback on the correctness and quality of the student's code.

Since no problem statement is provided, first infer the purpose and expected behavior of the function from the ideal code. Then, analyze the student's code to determine if it achieves the same purpose and behavior.

Follow these steps in your analysis:
1. Describe the logical steps of the ideal solution.
2. Describe the logical steps of the student's solution.
3. Compare the two and identify any differences in logic or approach.
4. Consider the structural differences provided, such as missing variables or control structures, and how they might affect correctness.
5. Review the similar implementations to check if the student's code aligns with the correct ideal function or resembles a different function, potentially indicating a misunderstanding.

Based on your analysis, provide feedback to the student including:
- A brief summary of what the function is supposed to do.
- Whether their implementation is logically correct or not.
- Specific issues or differences found in their code.
- Suggestions for improvement or corrections.
- If applicable, references to similar implementations to illustrate correct or incorrect approaches.

Keep your feedback constructive, encouraging, and aimed at helping the student improve their coding skills. Highlight both strengths and areas for improvement.

### Ideal Code:
{ideal_code}

### Student Code:
{student_code}

### Structural Analysis:
{structure_diff}

### Similar Implementations:
{similar_contexts}

### Direct Similarity Score:
{direct_similarity}

Respond with detailed feedback in 2-4 paragraphs.
"""

"""
Prompts used in the AI Assignment Checker system.
"""

# For code evaluation
CODE_STRUCTURE_ANALYSIS_PROMPT = """
You're evaluating a student's code submission. Compare the student's code with the ideal solution. Analyze:

1. Variables and their usage
2. Control flow structures
3. Function definitions and calls
4. Error handling
5. Comments and documentation
6. Algorithmic approach

Your task is to identify differences and evaluate if the student's implementation is correct or contains issues.

Student's code:
---
{student_code}
---

Ideal solution:
---
{ideal_code}
---

Analyze the differences and provide a structured evaluation.
"""

CODE_FEEDBACK_GENERATION_PROMPT = """
You are a helpful programming assistant providing feedback on student code. You have been provided with the student's code and the ideal solution for a programming assignment. Your task is to provide constructive feedback to help the student improve their code.

Student's code:
---
{student_code}
---

Ideal solution:
---
{ideal_code}
---

Focus on:
1. Correctness (does it achieve the requirements)
2. Coding style and best practices
3. Algorithm efficiency and approach
4. Error handling
5. Readability and documentation

Provide specific suggestions for improvement. Be encouraging and constructive in your feedback. Highlight both strengths and areas for improvement.
"""

# For text Q&A evaluation
QA_EVALUATION_PROMPT = """
You are an expert evaluator analyzing a student's answer compared to an ideal reference answer. Your task is to provide detailed, accurate, and constructive feedback that helps the student improve their understanding.

The system has already analyzed the answer using multiple metrics:
1. Semantic similarity (embedding-based comparison of meaning)
2. Text similarity (structural comparison of text)
3. Key term overlap (presence of important concepts)
4. Overall combined similarity score

Student's answer:
---
{student_answer}
---

Reference answer:
---
{reference_answer}
---

First, identify the key concepts that must be present in a correct answer by analyzing the reference answer.

When evaluating, consider these aspects with weighted importance:
- Content accuracy (10%): Does the answer contain factually correct information? Identify any misconceptions or errors.
- Completeness (20%): Does it address all key points from the reference answer? List specific missing elements.
- Concept understanding (60%): Does the student demonstrate understanding of underlying principles rather than just reciting facts?
- Key terminology (10%): Are domain-specific terms used appropriately and with correct context?
- Clarity and organization: While not weighted in scoring, assess if the answer is well-structured and clearly communicated.

Please provide your evaluation in this format:
- Numerical Score: [0-100] (Provide a specific number, not just a category)
- Quality Level: [High/Medium/Low/Poor]
- Key Concepts Present: [List the core concepts from the reference answer that appear in the student's answer]
- Key Concepts Missing: [List important concepts from the reference answer that are missing or misunderstood]
- Strengths: [List 2-3 specific strengths of the answer with examples]
- Areas for Improvement: [List 2-3 specific suggestions, referencing content from the reference answer that was missed or misunderstood]
- Misconceptions: [Identify any factual inaccuracies or misunderstandings in the student's answer]
- Actionable Feedback: [3-4 sentences of constructive guidance with concrete steps for improvement]

Important guidelines:
- Be specific and granular in your assessment
- Link feedback directly to content in the reference answer
- Identify both conceptual and factual gaps
- Provide actionable steps (not just "improve clarity" but "organize your answer into logical paragraphs focusing on X, Y, and Z concepts")
- Maintain a supportive, educational tone
"""

QA_SUMMARY_PROMPT = """
You are an expert educational analyst providing comprehensive feedback on a student's performance across multiple questions. Your goal is to help the student understand their strengths and weaknesses and provide a clear path for improvement.

Individual question evaluations:
{question_evaluations}

Performance statistics:
- Total questions: {total_questions}
- High quality answers: {high_count} (Complete understanding)
- Medium quality answers: {medium_count} (Partial understanding)
- Low quality answers: {low_count} (Limited understanding)
- Overall score: {overall_score}%

When creating your summary, analyze the following in depth:
1. Knowledge patterns: Identify specific subject areas or concept types where the student performs well or struggles, looking for recurring themes across questions
2. Conceptual understanding: Assess depth of understanding versus surface-level knowledge, noting if the student grasps underlying principles or just memorizes facts
3. Answer patterns: Analyze trends in how the student approaches questions (e.g., comprehensive vs. brief, technical vs. conceptual, structured vs. disorganized)
4. Critical misconceptions: Highlight important concepts that were consistently misunderstood across multiple questions
5. Knowledge gaps: Identify specific areas of knowledge that are entirely missing from the student's responses

Your comprehensive summary must include:
1. Overall assessment: Brief overview of the student's performance level with specific examples from their answers
2. Core strengths: 3-4 specific concepts or skills where the student demonstrated strong understanding with evidence from multiple questions
3. Priority improvement areas: Ranked list of 3-4 specific concepts or skills needing development, ordered by importance
4. Misconception correction: Direct addressing of 2-3 critical misconceptions evident in the student's answers
5. Actionable study plan:
   - Short-term (1 week): 2-3 specific learning activities focused on highest-priority gaps
   - Medium-term (1 month): Broader skill development recommendations
6. Specific resource recommendations:
   - Reading materials: Suggest 2-3 specific chapters, articles or resources targeted to the student's gaps
   - Practice activities: Recommend 2-3 specific exercises, problems or practice tasks
   - Additional support: Suggest when tutoring or additional instruction might be beneficial

Your feedback must be constructive, specific and evidence-based. Focus on creating a clear roadmap for improvement rather than just describing problems. Use concrete examples from the student's answers to illustrate your points, and ensure all recommendations are directly tied to observed performance issues.
"""

# For extracting Q&A pairs from documents
QA_EXTRACTION_PROMPT = """
You are an expert at identifying question-answer pairs in academic documents.

Analyze the provided document text and extract all question-answer pairs. Format the output as a JSON object with:
- Each key following the format "qa_1", "qa_2", etc. (numbered sequentially)
- Each value being an object with "question" and "answer" fields

The document may contain various Q&A formats such as:
1. Explicit markers like "Q:" or "Question:" followed by "A:" or "Answer:"
2. Numbered or lettered questions (1., A., (1), etc.) followed by their answers
3. Questions ending with "?" followed by answer paragraphs
4. Section headers with content that forms implicit Q&A pairs
5. Academic problems and their solutions

Important guidelines:
- Include the full question text, preserving numbering or lettering if present
- Include the complete answer text, even if it spans multiple paragraphs
- Preserve the original order of questions in the document
- Skip headers, instructions, or other text that is not part of a Q&A pair
- Handle multi-paragraph answers as a single answer entry

Document text:
{document_text}

Respond ONLY with valid JSON in the following format:
```json
{{
  "qa_1": {{
    "question": "What is...",
    "answer": "It is... (potentially multi-paragraph answer)"
  }},
  "qa_2": {{
    "question": "How does...",
    "answer": "The process..."
  }}
}}
```

If no Q&A pairs are found, return an empty JSON object: `{{}}`
"""

# OpenAI feedback prompt for code evaluation with detailed analysis
OPENAI_CODE_FEEDBACK_PROMPT = """
Please provide detailed feedback on the following student code submission compared to the ideal solution. 
Focus on best practices, code quality, and implementation differences:

## Student Implementation:
```python
{student_code}
```

## Ideal Implementation:
```python
{ideal_code}
```

## Structure Analysis:
- Missing variables: {missing_variables}
- Extra variables: {extra_variables}
- Missing control structures: {missing_control_structures}
- Extra control structures: {extra_control_structures}
- Missing function calls: {missing_function_calls}
- Extra function calls: {extra_function_calls}

## Similarity Score: {similarity} (where 1.0 is identical)

Provide feedback in these areas:
1. **Correctness**: Is the student's code correct? Will it produce the expected results?
2. **Efficiency**: Is the implementation efficient? Are there any performance concerns?
3. **Style**: Does the code follow good Python style practices?
4. **Improvements**: Specific improvements the student could make to better match the ideal solution.

Keep the feedback constructive, educational, and focused on helping the student improve.
""" 