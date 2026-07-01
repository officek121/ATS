import os
import asyncio
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load local .env if it exists
load_dotenv()

from src.parser import ResumeParser
from src.retriever import HybridRetriever
from src.evaluator import LLMEvaluator

# Page configuration
st.set_page_config(page_title="Zero-Cost ATS", page_icon="📝", layout="wide")

# Initialize session state for singletons
if 'retriever' not in st.session_state:
    st.session_state.retriever = HybridRetriever()
if 'parser' not in st.session_state:
    st.session_state.parser = ResumeParser()
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = LLMEvaluator()
if 'candidates' not in st.session_state:
    st.session_state.candidates = []

st.title("📝 Zero-Cost Applicant Tracking System")
st.markdown("""
This ATS utilizes **Hybrid RAG** (FAISS + BM25) for semantic & lexical matching, 
and **Groq's Llama 3.3** for lightning-fast candidate evaluation.
""")

# Sidebar for controls
with st.sidebar:
    st.header("1. Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF/DOCX Resumes", accept_multiple_files=True, type=['pdf', 'docx'])
    
    if st.button("Process Resumes"):
        if uploaded_files:
            with st.spinner("Parsing and embedding resumes..."):
                new_docs = []
                for file in uploaded_files:
                    # Save temporarily to parse
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Parse
                    text = st.session_state.parser.parse_file(temp_path)
                    os.remove(temp_path)
                    
                    if text:
                        new_docs.append({
                            'id': file.name,
                            'text': text,
                            'metadata': {'filename': file.name}
                        })
                
                if new_docs:
                    st.session_state.retriever.add_documents(new_docs)
                    st.session_state.candidates.extend(new_docs)
                    st.success(f"Processed {len(new_docs)} resumes!")
        else:
            st.warning("Please upload files first.")

# Main dashboard
col1, col2 = st.columns([1, 1])

with col1:
    st.header("2. Job Description")
    job_desc = st.text_area("Paste the job description here:", height=200)
    
    if st.button("Search Candidates"):
        if not job_desc:
            st.warning("Please enter a job description.")
        elif len(st.session_state.candidates) == 0:
            st.warning("Please upload and process resumes first.")
        else:
            st.session_state.search_results = st.session_state.retriever.retrieve(job_desc, top_k=10)

with col2:
    st.header("3. Top Matches")
    if 'search_results' in st.session_state and st.session_state.search_results:
        results = st.session_state.search_results
        
        # Display as a dataframe
        df_data = []
        for res in results:
            df_data.append({
                "Candidate": res['metadata']['filename'],
                "Hybrid Score": round(res['hybrid_score'], 4)
            })
        st.dataframe(pd.DataFrame(df_data), use_container_width=True)
        
        # Select candidate to evaluate
        selected_cand_name = st.selectbox("Select Candidate to Evaluate", [r['metadata']['filename'] for r in results])
        
        # Find selected text
        selected_text = next(r['text'] for r in results if r['metadata']['filename'] == selected_cand_name)
        
        if st.button("Evaluate Candidate (AI)"):
            st.session_state.eval_trigger = True
            st.session_state.eval_cand = selected_text
    else:
        st.info("Run a search to see candidates.")

# Streamlit Fragment for AI Evaluation (runs in isolation)
@st.fragment
def evaluation_panel():
    if getattr(st.session_state, 'eval_trigger', False):
        st.header("🤖 AI Evaluation")
        
        # We use asyncio to run the async evaluation
        async def run_eval():
            with st.spinner("Generating structured evaluation via Groq..."):
                try:
                    result = await st.session_state.evaluator.evaluate_candidate_json(
                        job_description=job_desc,
                        resume_text=st.session_state.eval_cand
                    )
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.metric("Match Percentage", f"{result.get('match_percentage', 0)}%")
                        
                        st.subheader("Key Strengths")
                        for s in result.get('key_strengths', []):
                            st.write(f"- {s}")
                            
                        st.subheader("Missing Skills")
                        for m in result.get('missing_skills', []):
                            st.write(f"- {m}")
                            
                        st.subheader("Summary")
                        st.write(result.get('summary', ''))
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    
        asyncio.run(run_eval())
        
        # Reset trigger so it doesn't run on unrelated reruns
        st.session_state.eval_trigger = False

evaluation_panel()
