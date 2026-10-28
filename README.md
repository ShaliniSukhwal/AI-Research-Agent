# AI Research Agent

An AI-powered research assistant built with Python, FastAPI, LangGraph, and Google Gemini. It answers research questions by combining AI reasoning with web search and calculator tools.

## Features

- AI-powered research using Google Gemini
- Web search for gathering relevant information
- Calculator tool for mathematical queries
- LangGraph-based agent workflow
- FastAPI backend
- Web-based chat interface
- Chat history saved in the browser
- Research categories for different topics

## Tech Stack

- **Language:** Python
- **LLM:** Google Gemini
- **Agent Framework:** LangGraph
- **Backend:** FastAPI
- **Tools:** Web Search, Calculator
- **Frontend:** HTML, CSS, JavaScript

## Project Structure

```text
AI-Research-Agent/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── agent.py
├── main.py
├── search_tool.py
├── tools.py
├── llm_test.py
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ShaliniSukhwal/AI-Research-Agent.git
cd AI-Research-Agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages required by the project:

```bash
pip install fastapi uvicorn langgraph langchain-google-genai python-dotenv
```

Also install any additional packages required by your search tool and other project files.

### 4. Configure environment variables

Create a `.env` file in the project root and add the API keys required by your configuration.

Never upload API keys or the `.env` file to GitHub.

### 5. Run the application

```bash
python -m uvicorn main:app --reload
```

Open your browser at:

http://127.0.0.1:8000

## What I Learned

- Building an AI agent workflow with LangGraph
- Integrating Google Gemini with external tools
- Connecting web search and calculator functionality
- Creating a FastAPI application with a web interface
- Managing chat history and asynchronous requests

## Author

**Shalini Sukhwal**

[GitHub Profile](https://github.com/ShaliniSukhwal)
