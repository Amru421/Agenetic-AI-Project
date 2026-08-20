# Agenetic-AI-Project

AI Resume Analyzer 📄🤖
An enterprise-grade Agentic AI Resume Analysis & Job Matching Platform powered by LangGraph, OpenAI LLM (gpt-4o-mini), and an interactive Streamlit dashboard.

🌟 1. Project Overview
The AI Resume Analyzer evaluates resumes using a collaborative multi-agent architecture. It goes beyond simple keyword matching by breaking down the evaluation process into discrete specialized agents (parsing, ATS scoring, skill matching, strengths/weaknesses, improvements, critique, and final judicial verdict) wrapped in strict input/output guardrails and rate limiters.

⚡ 2. Key Features
Multi-Format Ingestion: Upload resumes in PDF, DOCX, or TXT format, or directly paste text.
Job Description Alignment: Compare candidate qualifications against a target job description to compute detailed matching scores and skill gap analysis.
12-Stage Agentic Pipeline:
Input Guardrail (checks prompt safety, injection attacks, length caps)
Rate Limiter (sliding window traffic protection)
Orchestrator (state management & routing)
Resume Parser Agent (JSON entity extraction)
ATS Analyzer Agent (structure & keyword density scoring)
Skill Matching Agent (qualification alignment)
Strengths & Weaknesses Agent (recruiter impact analysis)
Improvement Agent (actionable resume enhancement advice)
Critic Agent (factual accuracy and sanity auditing)
Judge Agent (final synthesis & scoring)
Self-Correction Loop (re-evaluates weak analyses)
Output Guardrail (API key redaction & length limits)
Professional Dashboard:
Live progress feedback during agent execution
Real-time score metrics: ATS Score, Skill Match Score, Overall Score
Candidate profile breakdown (name, skills tags, experience, education, projects)
9 interactive deep-dive expanders
Export full analysis as JSON or formatted Text (.txt) reports
🏗️ 3. Agent Architecture (LangGraph DAG)

       [USER RESUME + JOB DESCRIPTION]
                      │
                      ▼
             [INPUT GUARDRAIL]
                      │
                      ▼
               [RATE LIMITER]
                      │
                      ▼
               [ORCHESTRATOR]
                      │
                      ▼
              [RESUME PARSER]
                      │
                      ▼
               [ATS ANALYZER]
                      │
                      ▼
              [SKILL MATCHING]
                      │
                      ▼
          [STRENGTHS & WEAKNESSES]
                      │
                      ▼
            [IMPROVEMENT AGENT]
                      │
                      ▼
               [CRITIC AGENT]
                      │
                      ▼
                [JUDGE AGENT]
                      │
          ┌───────────┴───────────┐
     Score < 50              Score >= 50
     Iteration < 2           or Iteration >= 2
          │                       │
          ▼ (Retry Loop)          ▼
   [ORCHESTRATOR]              [MEMORY]
                                  │
                                  ▼
                          [OUTPUT GUARDRAIL]
                                  │
                                  ▼
                            [FINAL REPORT]
📁 4. Project Structure

resume_analysis_agent/
│
├── resume_agent.py      # Core Agentic AI backend (LangGraph, agents, guardrails)
├── app.py               # Modern Streamlit UI dashboard
├── requirements.txt     # Python project dependencies
├── .env                 # API key configuration
└── README.md            # Project documentation
🚀 5. Installation & Setup
Prerequisites
Python 3.10+ installed
OpenAI API Key
Step 1: Clone or Navigate to Project Directory
bash

cd resume_analysis_agent
Step 2: Create a Virtual Environment
bash

# Windows
python -m venv venv
venv\Scripts\activate
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bash

pip install -r requirements.txt
🔑 6. Environment Variables
Create or edit the .env file in the project root:

env

OPENAI_API_KEY=sk-proj-your_actual_openai_api_key_here
(Alternatively, you can provide the API key directly via the Streamlit sidebar at runtime).

🖥️ 7. How to Run
Launch the Streamlit web application:

bash

streamlit run app.py
The application will open automatically in your default browser at http://localhost:8501.

🔌 8. How the Streamlit UI Connects to the Backend
The UI and backend are cleanly decoupled:

resume_agent.py contains all agent prompts, the LangGraph StateGraph, guardrail functions, and the run_resume_analysis() entrypoint.
app.py imports the backend functions:
python

from resume_agent import (
    input_guardrail,
    rate_limiter,
    output_guardrail,
    run_resume_analysis,
    RESUME_MEMORY
)
When the user clicks "Analyze Resume":
app.py extracts text from uploaded PDF/DOCX/TXT files or the text area.
Input is passed to input_guardrail(resume_text).
Rate limiting is verified via rate_limiter().
The multi-agent workflow is executed via run_resume_analysis().
The output is sanitized via output_guardrail().
The resulting state is displayed across interactive metrics, charts, and expanders.
