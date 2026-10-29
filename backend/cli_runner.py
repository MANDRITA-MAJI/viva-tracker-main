import os
import json
from dotenv import load_dotenv
from app.services.chunk_engine import get_random_chunk
from app.services.question_generator import generate_viva_questions
from app.services.ans_eval import evaluate_student_answer

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_FILE = os.path.join(DATA_DIR, "viva_results.json")

def save_result_to_json(student_record: dict):
    """Appends the student's complete viva record into viva_results.json"""
    os.makedirs(DATA_DIR, exist_ok=True)
    records = []

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []

    records.append(student_record)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

def run_cli_viva():
    print("=" * 55)
    print("            VIVA TRACKER - TERMINAL TEST           ")
    print("=" * 55)

    # 1. Collect Student Info
    roll_no = input("Enter Student Roll No: ").strip()
    name = input("Enter Student Name: ").strip()

    if not roll_no or not name:
        print("Error: Roll number and name cannot be empty.")
        return

    # 2. Retrieve a Chunk from ChromaDB
    print("\n[1/4] Fetching a relevant chunk from ChromaDB...")
    try:
        chunk_id, chunk = get_random_chunk()
        print(f"✓ Retrieved chunk snippet: \"{chunk[:80]}...\"")
    except Exception as e:
        print(f"Failed to retrieve chunk: {e}")
        print("Did you run 'python -m scripts.ingest' first?")
        return

    # 3. Generate Questions
    print("\n[2/4] Generating viva questions with LLM...")
    try:
        questions = generate_viva_questions(context_text=chunk).questions
        print(f"✓ Generated {len(questions)} questions successfully!\n")
    except Exception as e:
        print(f"Failed to generate questions: {e}")
        return

    # 4. Interactive Q&A & Evaluation Loop
    qa_records = []
    total_score = 0.0

    for idx, q in enumerate(questions, start=1):
        print("-" * 55)
        print(f"Question {idx}/5 [{q.difficulty.upper()}]:")
        print(f"{q.question}\n")
        print(f"-> Ideal Answer: {q.ideal_answer}\n")

        # Get student's answer via terminal
        student_ans = input("Your Answer: ").strip()

        print("\nEvaluating answer...")
        try:
            eval_result = evaluate_student_answer(
                question=q.question,
                ideal_answer=q.ideal_answer,
                key_points=q.key_points,
                student_answer=student_ans,
                source_text=chunk
            )

            score = eval_result.score
            feedback = eval_result.feedback

            total_score += score
            print(f"-> Score: {score}/10")
            print(f"-> Feedback: {feedback}\n")
            print(f"-> Ideal Answer: {q.ideal_answer}\n")

            qa_records.append({
                "question": q.question,
                "difficulty": q.difficulty,
                "ideal_answer": q.ideal_answer,
                "student_answer": student_ans,
                "score": score,
                "feedback": feedback
            })
        except Exception as e:
            print(f"Evaluation error: {e}")

    # 5. Final Summary & Save to File
    final_score = round(total_score / len(questions), 2) if questions else 0.0

    student_data = {
        "roll_no": roll_no,
        "name": name,
        "final_average_score": final_score,
        "questions_data": qa_records
    }

    save_result_to_json(student_data)

    print("=" * 55)
    print(f"VIVA COMPLETED FOR {name.upper()} (Roll No: {roll_no})")
    print(f"Overall Average Score: {final_score}/10")
    print(f"Saved results to: data/viva_results.json")
    print("=" * 55)

if __name__ == "__main__":
    run_cli_viva()