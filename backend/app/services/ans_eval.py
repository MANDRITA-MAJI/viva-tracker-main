from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class AnswerEvaluation(BaseModel):
    score: float = Field(description="Score from 0.0 to 10.0 based on factual correctness and completeness")
    is_correct: bool = Field(description="True if score >= 6.0, False otherwise")
    feedback: str = Field(description="Constructive feedback explaining what was correct and what was missing")
    matched_key_points: List[str] = Field(description="Key concepts correctly covered by the student")
    missing_key_points: List[str] = Field(description="Important concepts missed or explained incorrectly")

def evaluate_student_answer(
    question: str,
    ideal_answer: str,
    key_points: List[str],
    student_answer: str,
    source_text: str
) -> AnswerEvaluation:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0) 
    structured_evaluator = llm.with_structured_output(AnswerEvaluation)
    system_prompt = (
        "You are a fair and intelligent academic evaluator. Your job is to grade the student's answer "
        "using the source text as the ground truth. The ideal answer and key points are a helpful guide, "
        "but if they ever conflict with the source text, the source text wins.\n\n"
        "CRITICAL INSTRUCTION: Grade based on CONCEPTUAL UNDERSTANDING, not exact wording. "
        "Students may use different vocabulary, phrasing, or articulation. If the underlying logic, "
        "meaning, and facts are correct, give them full credit for those points. Do not penalize them "
        "for missing specific keywords if they successfully explained the concept.\n\n"
        "Rubric:\n"
        "- 9.0 - 10.0: Fully correct and covers all key points.\n"
        "- 8.0 - 9.0: Mostly correct, covers main ideas but misses secondary details.\n"
        "- 6.0 - 8.0 : Partially correct with notable omissions or minor misconceptions.\n"
        "- 4.0 - 5.0 : Incorrect, completely off-topic, or containing severe factual errors."
    )
    user_prompt = (
        f"Source Text:\n{source_text}\n\n"
        f"Question: {question}\n"
        f"Ideal Answer: {ideal_answer}\n"
        f"Required Key Points: {key_points}\n"
        f"Student's Answer: {student_answer}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    chain = prompt | structured_evaluator
    return chain.invoke({})