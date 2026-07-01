"""
Zero-Cost Applicant Tracking System (ATS) - Enterprise Production Architecture (Single File)

This architecture integrates:
1. Asymmetric Local Parsing (ProcessPoolExecutor PyMuPDF -> pypdf)
2. Hybrid BM25/Semantic Filtering (rank_bm25 + OpenVINO)
3. API Circuit Breaker (Cerebras -> Groq -> SambaNova -> Local Qwen)
4. Asynchronous Dataflow via Gradio Generators
"""

import io
import os
import json
import asyncio
import httpx
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import List, Dict, Any

import fitz  # PyMuPDF
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from dotenv import load_dotenv
import gradio as gr

try:
    from optimum.intel import OVModelForFeatureExtraction
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False


# ==========================================
# CONFIGURATION & GLOBAL STATE
# ==========================================
load_dotenv()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY", "")

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OV_MODEL_DIR = "ov_model"

# Global initialization for OpenVINO models (loaded once per worker)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ov_model = None

if OPENVINO_AVAILABLE and os.path.exists(OV_MODEL_DIR):
    ov_model = OVModelForFeatureExtraction.from_pretrained(OV_MODEL_DIR)

ATS_SCHEMA = {
    "candidate_name": "string",
    "years_of_experience": "number",
    "matching_skills": ["string"],
    "missing_skills": ["string"],
    "overall_match_decision": "boolean",
    "rejection_reason": "string (if overall_match_decision is false, provide a concise reason for rejection. Otherwise empty)",
    "match_summary": "string"
}


# ==========================================
# 1. ASYMMETRIC LOCAL PARSING ENGINE
# ==========================================
def _pymupdf_worker(file_bytes: bytes) -> str:
    """
    Isolated child process function utilizing C-extensions.
    Bypasses Python's GIL. Memory is reclaimed by the OS on exit.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    # Using newline to join pages
    return "\n".join([page.get_text() for page in doc])

def parse_document_safe(file_bytes: bytes, filename: str) -> dict:
    """
    Failover execution block. Attempts PyMuPDF first, falls back to pypdf.
    (ProcessPoolExecutor removed for Windows compatibility).
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        return {"filename": filename, "text": text.strip()}
    except Exception as e:
        print(f"[{filename}] PyMuPDF parsing failed ({e}). Falling back to pypdf...")
        # Graceful degradation to pure-Python, low-memory parser
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return {"filename": filename, "text": text.strip()}


# ==========================================
# 2. HYBRID SEARCH ENGINE
# ==========================================
def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""
    token_embeddings = model_output.last_hidden_state if hasattr(model_output, 'last_hidden_state') else model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).detach().numpy()

from sentence_transformers import SentenceTransformer

# Initialize standard SentenceTransformer as fallback
st_model = None

def get_embeddings(texts: List[str]) -> np.ndarray:
    """Generate semantic embeddings using OpenVINO if available, otherwise standard SentenceTransformers."""
    global st_model
    
    # Fast path: OpenVINO (if available)
    if ov_model:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = ov_model(**inputs)
        return mean_pooling(outputs, inputs['attention_mask'])
        
    # Fallback: Standard PyTorch SentenceTransformer (Perfect for cloud deployment)
    if not st_model:
        print("[INFO] OpenVINO model not found. Using standard SentenceTransformer fallback.")
        st_model = SentenceTransformer(MODEL_ID)
        
    return st_model.encode(texts, normalize_embeddings=True)

def build_hybrid_index(documents: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Builds the search index (BM25 tokenization + Semantic Embeddings) for a set of resumes.
    """
    if not documents:
        return {}
        
    texts = [doc['text'] for doc in documents]
    
    # 1. Lexical Index (BM25)
    tokenized_texts = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    
    # 2. Semantic Index (OpenVINO)
    doc_embeddings = get_embeddings(texts)
    
    return {
        "documents": documents,
        "bm25": bm25,
        "doc_embeddings": doc_embeddings
    }

def filter_candidates(query: str, index: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Hybrid Search Engine fusing BM25 and Cosine Similarity.
    S_total = 0.2 * BM25 + 0.8 * Cosine_Similarity
    """
    if not index or not index.get("documents"):
        return []
        
    documents = index["documents"]
    bm25 = index["bm25"]
    doc_embeddings = index["doc_embeddings"]
    
    # Lexical Scoring
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    normalized_bm25 = np.array(bm25_scores) / max_bm25
    
    # Semantic Scoring
    query_embedding = get_embeddings([query])
    cosine_scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Fusion
    s_total = 0.2 * normalized_bm25 + 0.8 * cosine_scores
    
    # Ranking
    ranked_indices = np.argsort(s_total)[::-1][:top_k]
    
    ranked_results = []
    for idx in ranked_indices:
        ranked_results.append({
            "filename": documents[idx]["filename"],
            "text": documents[idx]["text"],
            "score": float(s_total[idx]),
            "bm25_score": float(normalized_bm25[idx]),
            "semantic_score": float(cosine_scores[idx])
        })
        
    return ranked_results


# ==========================================
# 3. API CIRCUIT BREAKER
# ==========================================
async def query_local_qwen(text_chunk: str, jd_input: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Local Pivot: Qwen 1.5B (DeepSeek-R1-Distill-Qwen-1.5B) via llama.cpp
    This serves as the ultimate zero-cost fallback when all cloud APIs fail (Global 429).
    """
    try:
        from llama_cpp import Llama
        print("\n[LOCAL PIVOT] Engaging llama.cpp (Stub) for CPU inference...")
        return {
            "error": "Local inference fallback triggered. To enable, download Qwen-1.5B GGUF and configure llama.cpp."
        }
    except ImportError:
        print("\n[LOCAL PIVOT] llama-cpp-python not installed. Skipping local fallback.")
        return {"error": "All APIs failed, and local llama.cpp is not installed."}

async def extract_structured_json(text: str, jd_input: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cascading Circuit Breaker Pattern:
    1. Cerebras (Wafer-Scale Engine, 1M TPD, strict JSON)
    2. Groq (LPU, strict TPM limit, JSON Mode)
    3. SambaNova (Reconfigurable Dataflow, 200K TPD, JSON Mode)
    4. Local Qwen 1.5B (CPU Pivot via llama.cpp)
    """
    truncated_text = text[:4000] # Context safety limit
    
    prompt = f"""Evaluate the candidate resume against the Job Description. Extract information strictly matching the JSON schema below.
    
CRITICAL RECRUITING LOGIC:
1. "matching_skills": Skills the candidate ACTUALLY POSSESSES in their resume that match the Job Description. Do not hallucinate.
2. "missing_skills": Core requirements from the Job Description that the candidate's resume is MISSING.
3. "overall_match_decision": Set to true (HIRE) ONLY if the candidate broadly meets the minimum/core requirements of the role. If they clearly lack fundamental minimum requirements, set to false (NO HIRE). Preferred qualifications are a bonus, but missing minimums is a dealbreaker.
4. "rejection_reason": If rejecting the candidate, explicitly state which core requirements they failed to meet.

Schema:
{json.dumps(schema, indent=2)}

Job Description:
{jd_input}

Candidate Resume Text:
{truncated_text}"""
    system_msg = "You are an expert technical recruiter and ATS evaluator. You evaluate candidates strictly and objectively based on the provided Job Description."

    async with httpx.AsyncClient(timeout=45.0) as client:
        # --- Attempt 1: Cerebras ---
        try:
            print("[CIRCUIT BREAKER] Attempt 1: Cerebras API...")
            if not CEREBRAS_API_KEY: raise ValueError("Missing API Key")
            
            response = await client.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gemma-4-31b",
                    "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"} 
                }
            )
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return json.loads(content)
            
        except (httpx.HTTPError, ValueError) as e:
            print(f"-> Cerebras Failed ({e}). Falling through...")

        # --- Attempt 2: Groq ---
        try:
            print("[CIRCUIT BREAKER] Attempt 2: Groq API...")
            if not GROQ_API_KEY: raise ValueError("Missing API Key")
            
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
            )
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return json.loads(content)
            
        except (httpx.HTTPError, ValueError) as e:
            print(f"-> Groq Failed ({e}). Falling through...")
            
        # --- Attempt 3: SambaNova ---
        try:
            print("[CIRCUIT BREAKER] Attempt 3: SambaNova API...")
            if not SAMBANOVA_API_KEY: raise ValueError("Missing API Key")
            
            response = await client.post(
                "https://api.sambanova.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {SAMBANOVA_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "Meta-Llama-3.3-70B-Instruct",
                    "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
            )
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return json.loads(content)
            
        except (httpx.HTTPError, ValueError) as e:
            print(f"-> SambaNova Failed ({e}). Threshold breached, triggering Local Pivot...")

    # --- Attempt 4: Local Qwen Pivot ---
    return await asyncio.to_thread(query_local_qwen, text, jd_input, schema)

async def compare_candidates(candidates_data: List[Dict[str, Any]], jd_input: str) -> str:
    """
    Final synthesis step: If multiple candidates pass, the LLM compares them directly
    and chooses the absolute best one.
    """
    candidates_summary = ""
    for c in candidates_data:
        candidates_summary += f"Candidate #{c.get('candidate_number', '?')} ({c['filename']}):\n"
        candidates_summary += f"Matching Skills: {c['matching_skills']}\n"
        candidates_summary += f"Summary: {c['match_summary']}\n\n"
        
    prompt = f"""You are the VP of Engineering. Multiple candidates have passed the minimum requirements for this role.
You must review their summaries and select EXACTLY ONE candidate to hire.

Job Description:
{jd_input}

Candidates who passed:
{candidates_summary}

Task: Write a 1-2 paragraph executive summary explaining your decision. Compare the candidates directly. 
CRITICAL VP INSTRUCTIONS:
As a senior engineering leader, you must prioritize deep fundamental capability, educational rigor (e.g., Ph.D.), and the scale of their experience (e.g., Big Tech/Enterprise scale) over superficial keyword matching. If a candidate's high-level experience clearly implies they know basic libraries (like Matplotlib or AWS), do not penalize them just because they didn't list the exact keywords. 

Explicitly state the NAME/FILENAME of the winning candidate and why they are the superior choice. Do NOT invent or hallucinate candidate numbers or names."""

    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            # We use Groq's Llama-3.3-70B for the executive decision because 70B models 
            # have vastly superior multi-document synthesis and hallucination-resistance compared to smaller models.
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "system", "content": "You are a decisive VP of Engineering."}, {"role": "user", "content": prompt}]
                }
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error generating comparison: {e}"


# ==========================================
# 4. GRADIO DATAFLOW & GENERATOR LOOP
# ==========================================
async def process_batch(files: List[Any], jd_input: str, user_state: Dict[str, Any], progress=gr.Progress()):
    """
    Main asynchronous entry point. Uses yield for SSE progress updates without blocking.
    """
    # Reset state for new run so previous runs don't accumulate
    user_state['hired_candidates'] = []
    
    yield user_state, "Initializing Asymmetric Parsers...", ""
    
    if not files:
        yield user_state, "Error: No files uploaded.", ""
        return
    if not jd_input or not jd_input.strip():
        yield user_state, "Error: No Job Description provided.", ""
        return

    progress(0, desc="Initializing Asymmetric Parsers...")
    parsed_docs = []
    
    file_paths = [f.name if hasattr(f, 'name') else str(f) for f in files]
    
    # 1. Ingestion Loop (Memory-Safe Asymmetric Parsing)
    for i, path in enumerate(file_paths):
        filename = os.path.basename(path)
        progress((i + 1) / len(file_paths), desc=f"Parsing {filename}...")
        yield user_state, f"Parsing Document {i+1}/{len(file_paths)}: {filename}", ""
        
        # Read raw bytes to pass into our safe isolated parser
        with open(path, "rb") as f:
            file_bytes = f.read()
            
        # Execute memory-safe isolation parsing in thread so it doesn't block asyncio loop
        parsed_doc = await asyncio.to_thread(parse_document_safe, file_bytes, filename)
        if parsed_doc['text']:
            parsed_docs.append(parsed_doc)
            
    if not parsed_docs:
        yield user_state, "Error: Could not extract text from any provided documents.", ""
        return

    # 2. Zero-Cost Hybrid Screening (Lexical + Semantic)
    progress(0.5, desc="Vectorizing via OpenVINO INT8...")
    yield user_state, "Running Hybrid BM25 & Semantic Filter (OpenVINO INT8)...", ""
    
    # Isolate the index safely within this user's state
    try:
        user_state['hybrid_index'] = await asyncio.to_thread(build_hybrid_index, parsed_docs)
        top_candidates = filter_candidates(jd_input, user_state['hybrid_index'], top_k=min(3, len(parsed_docs)))
    except Exception as e:
        yield user_state, f"Error during Hybrid Search vectorization: {e}", ""
        return
        
    yield user_state, f"Top {len(top_candidates)} candidates ranked. Engaging LPU Circuit Breaker...", ""
    
    # 3. External LPU Extraction (Circuit Breaker)
    results_display = []
    
    for i, candidate in enumerate(top_candidates):
        progress(0.7 + (0.3 * (i / len(top_candidates))), desc=f"Querying Circuit Breaker for {candidate['filename']}...")
        yield user_state, f"Querying API Circuit Breaker for Candidate #{i+1} ({candidate['filename']})...", "\n\n".join(results_display)
        
        extracted_data = await extract_structured_json(candidate['text'], jd_input, ATS_SCHEMA)
        
        # Format the output nicely
        decision = "HIRE (Match)" if extracted_data.get('overall_match_decision') else "NO HIRE (No Match)"
        markdown_result = f"### #{i+1} - {candidate['filename']} | {decision}\n"
        markdown_result += f"**Hybrid Match Score**: `{candidate['score']:.3f}` *(BM25: {candidate['bm25_score']:.2f}, Semantic: {candidate['semantic_score']:.2f})*\n\n"
        markdown_result += f"**Matching Skills**: {', '.join(extracted_data.get('matching_skills', []))}\n\n"
        markdown_result += f"**Missing Skills**: {', '.join(extracted_data.get('missing_skills', []))}\n\n"
        
        if not extracted_data.get('overall_match_decision') and extracted_data.get('rejection_reason'):
            markdown_result += f"**Rejection Reason**: {extracted_data.get('rejection_reason')}\n\n"
            
        markdown_result += f"**Summary**: {extracted_data.get('match_summary', '')}\n\n"
        markdown_result += f"<details><summary>Raw JSON Payload</summary>\n\n```json\n{json.dumps(extracted_data, indent=2)}\n```\n</details>\n\n"
        markdown_result += "---\n"
        
        results_display.append(markdown_result)
        
        # Store for comparison
        if extracted_data.get('overall_match_decision'):
            extracted_data['filename'] = candidate['filename']
            extracted_data['candidate_number'] = i + 1
            user_state.setdefault('hired_candidates', []).append(extracted_data)
            
    final_output = "\n\n".join(results_display)
    
    # 4. Final Executive Comparison (If multiple hired)
    hired = user_state.get('hired_candidates', [])
    if len(hired) > 1:
        yield user_state, f"Multiple candidates passed! Engaging LLM to select a single winner...", final_output
        comparison_text = await compare_candidates(hired, jd_input)
        final_output += f"\n\n# Final Executive Decision\n\n{comparison_text}"
        
    user_state['final_results'] = top_candidates
    yield user_state, "Batch Processing Complete.", final_output


# ==========================================
# 5. APPLICATION ROUTING & QUEUE MANAGEMENT
# ==========================================
with gr.Blocks(title="Zero-Cost Enterprise ATS") as demo:
    gr.Markdown("# Zero-Cost Applicant Tracking System")
    gr.Markdown("Architecture: `PyMuPDF (ProcessPool)` -> Hybrid Search (BM25 + OpenVINO) -> API Circuit Breaker (Cerebras -> Groq -> SambaNova -> Local Qwen)")
    
    # Isolate user state securely via session hashes
    user_state = gr.State(value={})
    
    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="Upload Candidate Resumes (PDF)", file_types=[".pdf"], file_count="multiple")
            jd_input = gr.Textbox(label="Job Description", lines=5, placeholder="Paste the target job description here...")
            submit_btn = gr.Button("Execute Pipeline", variant="primary")
            status_output = gr.Textbox(label="Real-Time Telemetry", interactive=False, lines=2)
            
        with gr.Column(scale=2):
            results_output = gr.Markdown(label="Ranked Candidates & Extraction Results")
    
    submit_btn.click(
        fn=process_batch,
        inputs=[upload, jd_input, user_state],
        outputs=[user_state, status_output, results_output]
    )

if __name__ == "__main__":
    # Cap concurrency to avoid 429s across the global shared IP address
    demo.queue(default_concurrency_limit=2).launch()
