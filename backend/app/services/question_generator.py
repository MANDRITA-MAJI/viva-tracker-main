from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class VivaQuestion(BaseModel):
    difficulty: Literal["Easy", "Medium", "Hard"]
    question: str = Field(description="The viva question asked to the student")
    ideal_answer: str = Field(description="A concise ideal answer based strictly on the text")
    key_points: List[str] = Field(description="Key concepts or facts required for full credit")

class VivaQuestionSet(BaseModel):
    questions: List[VivaQuestion] = Field(
        description="Exactly 5 questions: 2 Easy, 2 Medium, and 1 Hard"
    )

def generate_viva_questions(context_text: str, previously_asked: List[str] = None) -> VivaQuestionSet:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
    structured_llm = llm.with_structured_output(VivaQuestionSet)

    system_prompt = (
        "You are an academic examiner conducting a spoken viva exam. Using ONLY the provided context "
        "text as your source of facts, generate 5 viva questions with this exact distribution:\n"
        "- 2 Easy questions (basic definitions, direct recall)\n"
        "- 2 Medium questions (explanations, relationships between concepts)\n"
        "- 1 Hard question (analytical comparison, consequence, application)\n\n"
        "CRITICAL RULE: The student answering these questions will NEVER see the context text. They "
        "are recalling this from what they studied, exactly like a real oral exam. Therefore every "
        "question must be fully self-contained and must NEVER refer to 'the passage', 'the text', "
        "'the document', 'the given content', 'mentioned above', or any similar phrase. Ask about the "
        "actual subject matter directly, naming the specific topic, term, or example instead of "
        "pointing at the source.\n\n"
        "Bad: \"What are the two main components of soil mentioned in the passage?\"\n"
        "Good: \"What are the two main components of soil?\"\n"
        "Bad: \"List the factors used to classify soils of India according to the text.\"\n"
        "Good: \"What factors are used to classify the soils of India?\"\n\n"
        "Avoid any questions that overlap with previously asked questions."
    )
    user_prompt = f"Context:\n{context_text}\n\nPreviously Asked Questions:\n{previously_asked or []}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    chain = prompt | structured_llm
    return chain.invoke({})