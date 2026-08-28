import os
import json
import re
import PyPDF2
import asyncio
import tempfile
import edge_tts
import streamlit as st
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from openai import OpenAI

# ==========================================
# 0. PDF & TEXT/AUDIO UTILITIES
# ==========================================
def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def parse_llm_json(content: str) -> dict:
    if not content or not content.strip():
        return {}
    content = content.strip()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group(0)
    try:
        return json.loads(content)
    except Exception:
        return {}

def generate_audio_sync(text: str, voice: str, filepath: str):
    communicate = edge_tts.Communicate(text, voice)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(communicate.save(filepath))
    loop.close()

# ==========================================
# 1. DATA SCHEMAS
# ==========================================
class CandidateProfile(BaseModel):
    name: str = "Unknown"
    target_role: str = "Unknown"
    verified_skills: List[str] = []
    timeline_and_experience: List[Dict[str, str]] = []
    verbatim_claims: List[str] = []

class AgentAssessment(BaseModel):
    persona_name: str = "Agent"
    stance: str = "Neutral"
    confidence_score: float = 0.5
    key_arguments: List[str] = []

class DebateTurn(BaseModel):
    speaker: str = "Agent"
    target_agent: str = "Agent"
    argument: str = "No argument provided."
    cited_quote: str = ""

class FinalDecisionReport(BaseModel):
    candidate_name: str = "Unknown"
    final_recommendation: str = "Inconclusive"
    system_confidence: float = 0.0
    key_strengths: List[str] = []
    critical_concerns: List[str] = []
    unresolved_disagreements: List[str] = []
    decision_rationalization: str = "Arbitration completed."

# ==========================================
# 2. CORE AGENT LOGIC
# ==========================================
def build_candidate_profile(client: OpenAI, model: str, resume_text: str, transcript_text: str) -> CandidateProfile:
    prompt = f"Extract fact base from resume and transcript. Return raw JSON matching schema: {json.dumps(CandidateProfile.model_json_schema())}. Use single quotes internally. RESUME: {resume_text} TRANSCRIPT: {transcript_text}"
    res = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1)
    return CandidateProfile.model_validate(parse_llm_json(res.choices[0].message.content))

PERSONA_PROMPTS = {
    "Technical Agent": "Evaluate technical depth and logic.",
    "HR / Culture Agent": "Evaluate communication and honesty.",
    "Hiring Manager": "Evaluate business impact and execution.",
    "Skeptic Agent": "Look aggressively for inflated claims and contradictions."
}

def run_isolated_agent(client: OpenAI, model: str, persona: str, prompt: str, profile: CandidateProfile) -> AgentAssessment:
    user_prompt = f"Evaluate candidate based on facts: {profile.model_dump_json()}. Return raw JSON matching schema: {json.dumps(AgentAssessment.model_json_schema())}. Use single quotes."
    res = client.chat.completions.create(model=model, messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_prompt}], temperature=0.2)
    parsed = parse_llm_json(res.choices[0].message.content)
    parsed["persona_name"] = persona
    return AgentAssessment.model_validate(parsed)

def run_debate_round(client: OpenAI, model: str, profile: CandidateProfile, assessments: Dict[str, AgentAssessment]) -> List[DebateTurn]:
    transcript = []
    for speaker, prompt in PERSONA_PROMPTS.items():
        debate_prompt = f"Debate context: {profile.model_dump_json()}. Initial: {json.dumps({k: v.model_dump() for k, v in assessments.items()})}. Challenge another agent. Return raw JSON matching schema: {json.dumps(DebateTurn.model_json_schema())}."
        res = client.chat.completions.create(model=model, messages=[{"role": "system", "content": prompt}, {"role": "user", "content": debate_prompt}], temperature=0.3)
        parsed = parse_llm_json(res.choices[0].message.content)
        parsed["speaker"] = speaker
        transcript.append(DebateTurn.model_validate(parsed))
    return transcript

def arbitrate_decision(client: OpenAI, model: str, profile: CandidateProfile, assessments: Dict[str, AgentAssessment], debate: List[DebateTurn]) -> FinalDecisionReport:
    arbiter_prompt = f"Synthesize arguments and make a final hiring decision. Return raw JSON matching schema: {json.dumps(FinalDecisionReport.model_json_schema())}. DATA: {profile.model_dump_json()} ASSESSMENTS: {json.dumps({k:v.model_dump() for k,v in assessments.items()})} DEBATE: {json.dumps([t.model_dump() for t in debate])}"
    res = client.chat.completions.create(model=model, messages=[{"role": "user", "content": arbiter_prompt}], temperature=0.1)
    return FinalDecisionReport.model_validate(parse_llm_json(res.choices[0].message.content))

# ==========================================
# 3. STREAMLIT WEB INTERFACE
# ==========================================
st.set_page_config(page_title="Multi-Agent Hiring AI", layout="wide")

with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key here.")
    user_model = st.selectbox(
        "Select AI Model",
        ["openai/gpt-oss-20b", "qwen/qwen3.6-27b", "llama3-70b-8192", "gemma2-9b-it"],
        index=0
    )
    st.markdown("---")
    st.caption("API keys are processed entirely in memory and are never saved to disk.")

st.title("🤖 Multi-Agent Hiring Committee")
st.markdown("Upload a candidate's resume and interview transcript to trigger an autonomous debate between four AI personas.")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (.pdf)", type="pdf")
with col2:
    transcript_file = st.file_uploader("Upload Transcript (.pdf)", type="pdf")

if st.button("Start Committee Evaluation", type="primary"):
    if not user_api_key:
        st.error("Please enter a valid API Key in the left sidebar to begin.")
    elif not resume_file or not transcript_file:
        st.warning("Please upload both the Resume and Transcript PDFs to begin.")
    else:
        dynamic_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=user_api_key
        )
        
        with st.status(f"Running AI Hiring Pipeline via {user_model}...", expanded=True) as status:
            st.write("📄 Extracting text from PDFs...")
            resume_text = extract_text_from_pdf(resume_file)
            transcript_text = extract_text_from_pdf(transcript_file)

            st.write("🧠 Building candidate fact profile...")
            profile = build_candidate_profile(dynamic_client, user_model, resume_text, transcript_text)

            st.write("🕵️‍♂️ Running independent persona evaluations...")
            assessments = {}
            for name, prompt in PERSONA_PROMPTS.items():
                assessments[name] = run_isolated_agent(dynamic_client, user_model, name, prompt, profile)

            st.write("💬 Initiating inter-agent debate...")
            debate_log = run_debate_round(dynamic_client, user_model, profile, assessments)

            st.write("⚖️ Arbiter synthesizing final verdict...")
            final_report = arbitrate_decision(dynamic_client, user_model, profile, assessments, debate_log)
            status.update(label="Evaluation Complete!", state="complete", expanded=False)

        st.header(f"Final Verdict: {final_report.final_recommendation}")
        st.progress(final_report.system_confidence)
        st.caption(f"Confidence Level: {final_report.system_confidence * 100:.0f}%")

        st.subheader("Arbiter Rationale")
        st.info(final_report.decision_rationalization)

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Key Strengths")
            for s in final_report.key_strengths: st.success(f"+ {s}")
        with colB:
            st.subheader("Critical Concerns")
            for c in final_report.critical_concerns: st.error(f"- {c}")

        st.divider()
        st.subheader("Behind the Scenes: Committee Debate Log & Audio")
        
        voice_map = {
            "Technical Agent": "en-US-GuyNeural",
            "HR / Culture Agent": "en-US-JennyNeural",
            "Hiring Manager": "en-US-AriaNeural",
            "Skeptic Agent": "en-GB-RyanNeural"
        }

        # Master byte string to hold the concatenated audio
        combined_audio_bytes = b""

        for turn in debate_log:
            with st.chat_message("user"):
                st.markdown(f"**{turn.speaker}** addressing **{turn.target_agent}**:")
                st.write(turn.argument)
                if turn.cited_quote:
                    st.caption(f"Evidence: *\"{turn.cited_quote}\"*")
                
                voice = voice_map.get(turn.speaker, "en-US-ChristopherNeural")
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                generate_audio_sync(turn.argument, voice, temp_audio.name)
                
                # Append individual MP3 bytes to the master track
                with open(temp_audio.name, "rb") as f:
                    combined_audio_bytes += f.read()
        
        st.markdown("---")
        st.subheader("🎧 Play Full Debate Sequence")
        # Render a single audio player that plays all turns back-to-back
        st.audio(combined_audio_bytes, format="audio/mp3")