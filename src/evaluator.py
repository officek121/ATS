import os
import json
import asyncio
from typing import Dict, Any
from groq import AsyncGroq
from pydantic import BaseModel, Field

# Define expected output structure
class CandidateEvaluation(BaseModel):
    match_percentage: int = Field(description="Calculated match percentage between 0 and 100")
    key_strengths: list[str] = Field(description="List of top 3 skills matching the job description")
    missing_skills: list[str] = Field(description="List of required skills the candidate is missing")
    summary: str = Field(description="A brief 2-sentence summary of the candidate's fit")

class LLMEvaluator:
    def __init__(self):
        """
        Initializes the Groq client. Requires GROQ_API_KEY environment variable.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY not found in environment variables.")
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile" # Highly capable model on Groq free tier

    async def evaluate_candidate_stream(self, job_description: str, resume_text: str):
        """
        Streams the evaluation of a candidate.
        Yields text chunks. The final chunk will be the structured JSON.
        We instruct the LLM to provide a thought process, then a JSON block.
        """
        prompt = f"""
You are an expert technical recruiter. Evaluate the candidate's resume against the job description.
First, provide a brief thought process of your evaluation.
Then, output a strictly valid JSON object conforming to this schema:
{CandidateEvaluation.schema_json()}

Job Description:
{job_description}

Candidate Resume:
{resume_text}
"""
        
        try:
            stream = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert ATS AI. Always include the final JSON block."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.2,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error during evaluation: {str(e)}"

    async def evaluate_candidate_json(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        """
        Non-streaming evaluation returning pure JSON using Groq's JSON mode.
        """
        prompt = f"""
You are an expert technical recruiter. Evaluate the candidate's resume against the job description.
You MUST respond with a strictly valid JSON object conforming to this schema:
{CandidateEvaluation.schema_json()}

Job Description:
{job_description}

Candidate Resume:
{resume_text}
"""
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a JSON-only API. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error during JSON evaluation: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    evaluator = LLMEvaluator()
    print("Evaluator initialized.")
