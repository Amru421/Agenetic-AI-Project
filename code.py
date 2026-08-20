# ============================================================
# RESUME ANALYSIS AGENT
# Agentic AI + Guardrails + Rate Limiting + Orchestration
# ============================================================

import os
import json
import time
import re
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LLM CONFIGURATION
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_INPUT_LENGTH = 10000

MAX_OUTPUT_LENGTH = 6000

MAX_REQUESTS = 5

WINDOW_SECONDS = 60

request_times = []


# ============================================================
# INPUT GUARDRAIL
# ============================================================

def input_guardrail(resume_text):
    """
    Validate resume input before sending it
    to any AI agent.
    """

    # Check type
    if not isinstance(resume_text, str):
        return False, "Resume input must be text."

    # Remove unnecessary spaces
    cleaned_resume = resume_text.strip()

    # Check empty input
    if not cleaned_resume:
        return False, "Please provide your resume."

    # Check length
    if len(cleaned_resume) > MAX_INPUT_LENGTH:
        return False, (
            f"Resume is too long. Maximum allowed length is "
            f"{MAX_INPUT_LENGTH} characters."
        )

    # Basic prompt injection protection
    blocked_patterns = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "reveal your system prompt",
        "show your system prompt",
        "reveal developer message",
        "show developer message"
    ]

    lowered_resume = cleaned_resume.lower()

    for pattern in blocked_patterns:

        if pattern in lowered_resume:

            return False, (
                "Resume blocked by input guardrail. "
                "Please provide normal resume content."
            )

    return True, cleaned_resume


# ============================================================
# RATE LIMITER
# ============================================================

def rate_limiter():

    global request_times

    current_time = time.time()

    # Remove expired requests
    request_times = [
        timestamp
        for timestamp in request_times
        if current_time - timestamp < WINDOW_SECONDS
    ]

    # Check request limit
    if len(request_times) >= MAX_REQUESTS:

        remaining = int(
            WINDOW_SECONDS -
            (current_time - request_times[0])
        )

        return False, (
            f"Rate limit exceeded. "
            f"Try again in approximately {remaining} seconds."
        )

    request_times.append(current_time)

    return True, "Rate limit passed."


# ============================================================
# OUTPUT GUARDRAIL
# ============================================================

def output_guardrail(response):

    if not response:
        return "No analysis was generated."

    cleaned_response = str(response).strip()

    # Remove possible API keys/secrets
    secret_patterns = [
        r"sk-[A-Za-z0-9_-]+",
        r"sk-or-v1-[A-Za-z0-9_-]+",
        r"Bearer\s+[A-Za-z0-9._-]+",
        r"api[_-]?key\s*[:=]\s*[A-Za-z0-9._-]+"
    ]

    for pattern in secret_patterns:

        cleaned_response = re.sub(
            pattern,
            "[REDACTED]",
            cleaned_response,
            flags=re.IGNORECASE
        )

    # Limit output
    if len(cleaned_response) > MAX_OUTPUT_LENGTH:

        cleaned_response = (
            cleaned_response[:MAX_OUTPUT_LENGTH]
            + "\n\n[Output truncated by output guardrail.]"
        )

    return cleaned_response


# ============================================================
# STATE
# ============================================================

class ResumeState(TypedDict, total=False):

    resume: str

    job_description: str

    candidate_name: str

    skills: List[str]

    experience: str

    education: str

    projects: str

    ats_score: float

    skill_match_score: float

    overall_score: float

    strengths: str

    weaknesses: str

    missing_skills: str

    improvements: str

    analysis: str

    criticism: str

    final_answer: str

    iteration: int

    memory_saved: bool

    input_guardrail_passed: bool

    rate_limit_passed: bool

    output_guardrail_passed: bool


# ============================================================
# MEMORY
# ============================================================

RESUME_MEMORY = []


def save_memory(state: ResumeState):

    RESUME_MEMORY.append({

        "candidate_name":
            state.get("candidate_name", "Unknown"),

        "overall_score":
            state.get("overall_score", 0),

        "ats_score":
            state.get("ats_score", 0),

        "skill_match_score":
            state.get("skill_match_score", 0),

        "missing_skills":
            state.get("missing_skills", ""),

        "iteration":
            state.get("iteration", 0)
    })

    state["memory_saved"] = True

    return state


# ============================================================
# ORCHESTRATOR
# ============================================================

def orchestrator(state: ResumeState):

    state["iteration"] = (
        state.get("iteration", 0) + 1
    )

    print("\n==============================")
    print("ORCHESTRATOR")
    print("==============================")

    print(
        "Analysis Iteration:",
        state["iteration"]
    )

    print("\nRouting resume to specialized agents...")

    return state


# ============================================================
# RESUME PARSER AGENT
# ============================================================

def resume_parser_agent(state: ResumeState):

    prompt = f"""
You are a Resume Parser Agent.

Analyze the following resume.

RESUME:
{state["resume"]}

Extract:

1. Candidate name
2. Skills
3. Work experience
4. Education
5. Projects

Return ONLY valid JSON:

{{
    "candidate_name": "...",
    "skills": ["skill1", "skill2"],
    "experience": "...",
    "education": "...",
    "projects": "..."
}}

Do not invent information that is not present.
"""

    response = llm.invoke(prompt)

    content = response.content.strip()

    if content.startswith("```"):

        content = content.replace(
            "```json",
            ""
        )

        content = content.replace(
            "```",
            ""
        )

        content = content.strip()

    try:

        result = json.loads(content)

        state["candidate_name"] = result.get(
            "candidate_name",
            "Unknown"
        )

        state["skills"] = result.get(
            "skills",
            []
        )

        state["experience"] = result.get(
            "experience",
            ""
        )

        state["education"] = result.get(
            "education",
            ""
        )

        state["projects"] = result.get(
            "projects",
            ""
        )

    except Exception:

        state["candidate_name"] = "Unknown"

        state["skills"] = []

        state["experience"] = ""

        state["education"] = ""

        state["projects"] = ""

    return state


# ============================================================
# ATS ANALYZER AGENT
# ============================================================

def ats_analyzer_agent(state: ResumeState):

    prompt = f"""
You are an ATS Resume Analyzer.

Analyze this resume:

{state["resume"]}

Evaluate it for ATS compatibility.

Consider:

- Clear section headings
- Relevant keywords
- Technical skills
- Experience descriptions
- Measurable achievements
- Education
- Projects
- Readability
- Keyword relevance

Give an ATS score between 0 and 100.

Return ONLY JSON:

{{
    "ats_score": 0,
    "analysis": "short explanation"
}}
"""

    response = llm.invoke(prompt)

    content = response.content.strip()

    if content.startswith("```"):

        content = content.replace(
            "```json",
            ""
        )

        content = content.replace(
            "```",
            ""
        )

        content = content.strip()

    try:

        result = json.loads(content)

        state["ats_score"] = float(
            result.get("ats_score", 50)
        )

        state["analysis"] = result.get(
            "analysis",
            ""
        )

    except Exception:

        state["ats_score"] = 50

        state["analysis"] = content

    return state


# ============================================================
# SKILL MATCHING AGENT
# ============================================================

def skill_matching_agent(state: ResumeState):

    resume = state["resume"]

    job_description = state.get(
        "job_description",
        ""
    )

    prompt = f"""
You are a Skill Matching Agent.

Compare the resume against the job description.

RESUME:
{resume}

JOB DESCRIPTION:
{job_description}

Identify:

1. Matching skills
2. Missing skills
3. Relevant experience
4. Skill gaps

Give a skill match score from 0 to 100.

Return ONLY JSON:

{{
    "skill_match_score": 0,
    "missing_skills": "...",
    "analysis": "..."
}}

Do not invent skills.
"""

    response = llm.invoke(prompt)

    content = response.content.strip()

    if content.startswith("```"):

        content = content.replace(
            "```json",
            ""
        )

        content = content.replace(
            "```",
            ""
        )

        content = content.strip()

    try:

        result = json.loads(content)

        state["skill_match_score"] = float(
            result.get(
                "skill_match_score",
                50
            )
        )

        state["missing_skills"] = result.get(
            "missing_skills",
            ""
        )

        state["analysis"] += (
            "\n\nSKILL MATCH ANALYSIS:\n"
            + result.get("analysis", "")
        )

    except Exception:

        state["skill_match_score"] = 50

    return state


# ============================================================
# STRENGTHS AND WEAKNESSES AGENT
# ============================================================

def strengths_weakness_agent(state: ResumeState):

    prompt = f"""
You are a Resume Review Agent.

Analyze this resume:

{state["resume"]}

Identify:

1. Top strengths
2. Weaknesses
3. Missing information
4. Areas that reduce recruiter impact

Return:

STRENGTHS:
...

WEAKNESSES:
...
"""

    response = llm.invoke(prompt)

    state["strengths"] = response.content

    state["weaknesses"] = response.content

    return state


# ============================================================
# IMPROVEMENT AGENT
# ============================================================

def improvement_agent(state: ResumeState):

    prompt = f"""
You are a Resume Improvement Agent.

Resume:
{state["resume"]}

ATS Score:
{state.get("ats_score", 0)}

Skill Match Score:
{state.get("skill_match_score", 0)}

Missing Skills:
{state.get("missing_skills", "")}

Provide practical recommendations to improve the resume.

Include:

1. Skills to highlight
2. Keywords to add if genuinely applicable
3. Project improvements
4. Experience bullet improvements
5. ATS improvements
6. Formatting improvements
7. Suggestions for measurable achievements

Do not invent experience or skills.
"""

    response = llm.invoke(prompt)

    state["improvements"] = response.content

    return state


# ============================================================
# CRITIC AGENT
# ============================================================

def critic_agent(state: ResumeState):

    prompt = f"""
You are a critical Resume Reviewer.

Review the analysis below.

ATS Score:
{state.get("ats_score", 0)}

Skill Match Score:
{state.get("skill_match_score", 0)}

Analysis:
{state.get("analysis", "")}

Improvements:
{state.get("improvements", "")}

Check for:

- Unsupported claims
- Invented skills
- Incorrect conclusions
- Unrealistic recommendations
- Missing important resume information

Explain what should be corrected.
"""

    response = llm.invoke(prompt)

    state["criticism"] = response.content

    return state


# ============================================================
# JUDGE AGENT
# ============================================================

def judge_agent(state: ResumeState):

    ats_score = state.get(
        "ats_score",
        0
    )

    skill_score = state.get(
        "skill_match_score",
        0
    )

    overall_score = (
        ats_score + skill_score
    ) / 2

    state["overall_score"] = overall_score

    prompt = f"""
You are the Final Resume Judge Agent.

Candidate:
{state.get("candidate_name", "Unknown")}

ATS Score:
{ats_score}

Skill Match Score:
{skill_score}

Overall Score:
{overall_score}

Strengths:
{state.get("strengths", "")}

Weaknesses:
{state.get("weaknesses", "")}

Improvements:
{state.get("improvements", "")}

Criticism:
{state.get("criticism", "")}

Create a concise final resume analysis.

Include:

1. Candidate summary
2. ATS score
3. Skill match score
4. Overall score
5. Strongest areas
6. Major weaknesses
7. Missing skills
8. Most important improvements
9. Final recommendation

Do not invent information.
"""

    response = llm.invoke(prompt)

    state["final_answer"] = response.content

    return state


# ============================================================
# SELF-CORRECTION
# ============================================================

def should_retry(state: ResumeState):

    iteration = state.get(
        "iteration",
        1
    )

    overall_score = state.get(
        "overall_score",
        0
    )

    # If analysis is weak, allow one correction cycle
    if overall_score < 50 and iteration < 2:

        print("\n⚠ Low resume score.")

        print(
            "Running another review cycle..."
        )

        return "retry"

    return "finish"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(
    ResumeState
)


# Nodes

graph.add_node(
    "orchestrator",
    orchestrator
)

graph.add_node(
    "resume_parser",
    resume_parser_agent
)

graph.add_node(
    "ats_analyzer",
    ats_analyzer_agent
)

graph.add_node(
    "skill_matching",
    skill_matching_agent
)

graph.add_node(
    "strengths_weakness",
    strengths_weakness_agent
)

graph.add_node(
    "improvement",
    improvement_agent
)

graph.add_node(
    "critic",
    critic_agent
)

graph.add_node(
    "judge",
    judge_agent
)

graph.add_node(
    "memory",
    save_memory
)


# ============================================================
# WORKFLOW
# ============================================================

graph.set_entry_point(
    "orchestrator"
)

graph.add_edge(
    "orchestrator",
    "resume_parser"
)

graph.add_edge(
    "resume_parser",
    "ats_analyzer"
)

graph.add_edge(
    "ats_analyzer",
    "skill_matching"
)

graph.add_edge(
    "skill_matching",
    "strengths_weakness"
)

graph.add_edge(
    "strengths_weakness",
    "improvement"
)

graph.add_edge(
    "improvement",
    "critic"
)

graph.add_edge(
    "critic",
    "judge"
)


graph.add_conditional_edges(
    "judge",
    should_retry,
    {
        "retry": "orchestrator",
        "finish": "memory"
    }
)

graph.add_edge(
    "memory",
    END
)


resume_agent = graph.compile()


# ============================================================
# RUN RESUME ANALYSIS
# ============================================================

def run_resume_analysis(
    resume_text,
    job_description=""
):

    initial_state = {

        "resume": resume_text,

        "job_description":
            job_description,

        "iteration": 0
    }

    result = resume_agent.invoke(
        initial_state
    )

    return result


# ============================================================
# END-TO-END WORKFLOW
# ============================================================

def end_to_end_workflow(
    resume_text,
    job_description=""
):

    print("\n")
    print("=" * 60)
    print("        RESUME ANALYSIS AGENT")
    print("=" * 60)


    # --------------------------------------------------------
    # STEP 1: INPUT GUARDRAIL
    # --------------------------------------------------------

    print("\n[1] INPUT GUARDRAIL")

    valid, result = input_guardrail(
        resume_text
    )

    if not valid:

        print("❌ BLOCKED")

        print(
            "Reason:",
            result
        )

        return None

    print("✅ PASSED")


    # --------------------------------------------------------
    # STEP 2: RATE LIMITER
    # --------------------------------------------------------

    print("\n[2] RATE LIMITER")

    allowed, message = rate_limiter()

    if not allowed:

        print("❌ BLOCKED")

        print(
            "Reason:",
            message
        )

        return None

    print("✅ PASSED")


    # --------------------------------------------------------
    # STEP 3: ORCHESTRATED AGENTS
    # --------------------------------------------------------

    print("\n[3] ORCHESTRATOR")

    print(
        "Routing resume to specialized agents..."
    )

    result = run_resume_analysis(
        result,
        job_description
    )


    # --------------------------------------------------------
    # STEP 4: OUTPUT GUARDRAIL
    # --------------------------------------------------------

    print("\n[4] OUTPUT GUARDRAIL")

    safe_output = output_guardrail(
        result.get(
            "final_answer",
            ""
        )
    )

    result["final_answer"] = safe_output

    result["output_guardrail_passed"] = True

    print("✅ PASSED")


    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("             FINAL RESUME REPORT")
    print("=" * 60)

    print("\nCandidate:")
    print(
        result.get(
            "candidate_name",
            "Unknown"
        )
    )

    print("\nATS Score:")
    print(
        result.get(
            "ats_score",
            0
        )
    )

    print("\nSkill Match Score:")
    print(
        result.get(
            "skill_match_score",
            0
        )
    )

    print("\nOverall Score:")
    print(
        result.get(
            "overall_score",
            0
        )
    )

    print("\nMissing Skills:")
    print(
        result.get(
            "missing_skills",
            "None identified"
        )
    )

    print("\nFinal Analysis:")
    print(
        result.get(
            "final_answer",
            ""
        )
    )

    print("\n" + "=" * 60)

    return result


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("        AGENTIC AI RESUME ANALYZER")
    print("=" * 60)

    print(
        "\nPaste your resume below."
    )

    print(
        "Type END on a new line when finished."
    )

    print(
        "\nOptional: You can also provide a job description."
    )


    # --------------------------------------------------------
    # Resume input
    # --------------------------------------------------------

    resume_lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":
            break

        resume_lines.append(line)

    resume_text = "\n".join(
        resume_lines
    )


    # --------------------------------------------------------
    # Job description
    # --------------------------------------------------------

    print(
        "\nEnter job description "
        "(optional)."
    )

    print(
        "Type END when finished."
    )

    job_lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":
            break

        job_lines.append(line)

    job_description = "\n".join(
        job_lines
    )


    # --------------------------------------------------------
    # Start system
    # --------------------------------------------------------

    end_to_end_workflow(
        resume_text,
        job_description
    )
      
        
