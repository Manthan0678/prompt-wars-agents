# 🤖 Multi-Agent Hiring Committee
*Prompt Wars Hackathon Submission*

## 🎯 Overview
This project is an autonomous, multi-agent AI system designed to evaluate candidate resumes and interview transcripts. Rather than relying on a single AI prompt or basic mathematical averaging, this system deploys four distinct AI personas that independently evaluate the candidate, debate each other using factual evidence, and reach a final synthesized verdict.

## ✨ Key Features
* **Candidate Fact Extraction:** Parses PDF resumes and transcripts into a structured, objective JSON fact-base.
* **Isolated Persona Evaluations:** Four specialized agents (Technical, HR/Culture, Hiring Manager, and Skeptic) assess the candidate entirely independently.
* **Evidence-Based Debate Engine:** Agents challenge and corroborate each other's initial stances using exact verbatim quotes from the source documents.
* **Arbiter Synthesis:** A final reasoning step that weighs debate shifts, red flags, and technical proofs to deliver a deterministic verdict.
* **Dynamic Audio Synthesis:** Utilizes Edge-TTS to map unique voices to each persona, generating a seamless master audio track of the live debate.
* **Secure UI:** Built with Streamlit, featuring dynamic, in-memory API key configuration (no hardcoded secrets).

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit
* **AI/LLM:** Groq API (OpenAI Python SDK), Pydantic (Strict JSON formatting)
* **Document Processing:** PyPDF2
* **Audio Processing:** edge-tts, asyncio

## 🚀 How to Run Locally
1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
