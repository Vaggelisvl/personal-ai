# Personal AI - Evangelos Vrailas CV-Based AI Model

An AI model trained on Evangelos Vrailas's CV data that can answer questions about his background, experience, skills, and general knowledge questions.

## Features

✨ **CV-Specific Question Answering**: Ask anything about Evangelos's:
- Professional experience
- Technical skills and expertise
- Education background
- Contact information
- Projects and responsibilities

🧠 **General Knowledge**: The AI can also answer basic questions about:
- Technology definitions (Java, Kubernetes, Docker, etc.)
- Simple mathematics
- World capitals
- General greetings and small talk

## Quick Start

**Run the demo** to see all capabilities:
```bash
python demo.py
```

**Run the test suite** to verify everything works:
```bash
python test_ai_model.py
```

**Interactive mode** for asking your own questions:
```bash
python ai_model.py
```

## What the Model Covers

The AI model is trained on comprehensive CV data including:

### Personal Information
- Full name, title, and location
- Contact information (email, LinkedIn, portfolio)

### Professional Experience
1. **Full Stack Developer** at Netcompany-Intrasoft (Oct 2024 - Present)
   - Java 17, Spring Boot, React, Angular
   - Kubernetes, RabbitMQ, microservices
   - Cross-team collaboration

2. **Junior Software Developer** at Public Group (2022 - 2024)
   - Enterprise applications with Java and Spring Boot
   - ActiveMQ and RabbitMQ messaging
   - MongoDB, Oracle, SQL Server databases

3. **Computer Science Professor Assistant** at NKUA (2021 - 2022)
   - Data structures and programming techniques
   - C programming and algorithm design

### Technical Skills
- **Languages**: Java (11, 17, 21), C++, SQL, Python
- **Frameworks**: Spring Boot, ReactJS, Angular
- **Messaging**: RabbitMQ, ActiveMQ
- **Databases**: Oracle, MongoDB, Microsoft SQL Server, NoSQL
- **DevOps**: Docker, Kubernetes, Git, CI/CD (Bitbucket, Azure DevOps), Jira
- **Practices**: Clean code, TDD, Agile (Scrum), CI/CD

### Education
- Bachelor's in Informatics and Telecommunications
- National and Kapodistrian University of Athens (2019-2024)

### Certifications
- Public Next Graduate Program (500-hour comprehensive training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Vaggelisvl/personal-ai.git
cd personal-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode

Run the AI model in interactive mode:

```bash
python ai_model.py
```

This will start an interactive session where you can ask questions.

### Example Questions

**About Evangelos:**
- "Who are you?"
- "What is your current job?"
- "What are your skills?"
- "Where did you study?"
- "What experience do you have with Kubernetes?"
- "Tell me about your experience"
- "What is your email?"

**General Questions:**
- "What is Java?"
- "What is Kubernetes?"
- "What is the capital of Greece?"
- "What is 5 + 3?"
- "Hello!"

### Programmatic Usage

You can also use the AI model in your own Python code:

```python
from ai_model import PersonalAIModel

# Initialize the model
model = PersonalAIModel()

# Ask a question
answer = model.answer("What are your programming skills?")
print(answer)

# Get CV summary
summary = model.get_cv_summary()
print(summary)
```

## Architecture

The AI model uses a hybrid approach:

1. **Knowledge Base**: Structured CV data stored in `cv_knowledge_base.py`
2. **Retrieval System**: Pattern matching and similarity scoring for CV-specific questions
3. **General QA**: Rule-based system for handling general knowledge questions
4. **Fallback**: Graceful handling of unknown questions with helpful suggestions

## Files

- `ai_model.py`: Main AI model implementation with question-answering logic
- `cv_knowledge_base.py`: Structured CV data and pre-defined Q&A pairs
- `requirements.txt`: Python dependencies
- `Evangelos_Vrailas_Resume.pdf`: Original CV document

## How It Works

1. **Question Processing**: The model normalizes and analyzes the input question
2. **Q&A Matching**: Checks pre-defined Q&A pairs for direct matches
3. **CV Search**: Searches through structured CV data for relevant information
4. **General Knowledge**: Applies rule-based logic for general questions
5. **Response Generation**: Returns the most relevant answer or a helpful fallback

## Technical Details

- **Language**: Python 3.12+
- **Dependencies**: PyPDF2 for PDF parsing
- **Approach**: Retrieval-based QA with pattern matching
- **No external APIs**: Fully local, no internet required for inference

## Example Session

```
================================================================================
Personal AI Model - Trained on Evangelos Vrailas's CV
================================================================================

This AI can answer questions about:
- Evangelos's background, experience, and skills
- General knowledge questions (basic math, capitals, tech definitions, etc.)

Type 'quit' or 'exit' to end the conversation.
Type 'summary' to see a CV summary.
================================================================================

🤔 Your question: Who are you?

🤖 Answer: I am Evangelos Vrailas, a Software Developer from Athens, Greece, 
specializing in robust, production-ready systems built with Java and Spring Boot.

🤔 Your question: What are your skills?

🤖 Answer: I have expertise in Java (11, 17, 21), Spring Boot, C++, SQL, Python, 
ReactJS, Angular, RabbitMQ, ActiveMQ, Oracle, MongoDB, Docker, Kubernetes, and more. 
I follow clean code practices, TDD, and Agile methodologies.

🤔 Your question: What is Kubernetes?

🤖 Answer: Kubernetes (K8s) is an open-source container orchestration platform for 
automating deployment, scaling, and management of containerized applications. 
Evangelos has experience deploying and monitoring microservices in Kubernetes clusters.
```

## About Evangelos Vrailas

**Evangelos Vrailas** is a Software Developer based in Athens, Greece, specializing in:
- Java and Spring Boot development
- Full-stack development (React, Angular)
- Microservices architecture
- DevOps (Kubernetes, Docker)
- Message brokers (RabbitMQ, ActiveMQ)

**Contact:**
- Email: e.vrailas.dev@gmail.com
- LinkedIn: [evangelos-vrailas](https://www.linkedin.com/in/evangelos-vrailas-66b31122a/)
- Portfolio: [vaggelisvl.github.io/portfolio](https://vaggelisvl.github.io/portfolio/)

## License

This project is for demonstration purposes showcasing an AI model trained on CV data.

## Contributing

This is a personal portfolio project. For questions or feedback, please contact Evangelos Vrailas.
