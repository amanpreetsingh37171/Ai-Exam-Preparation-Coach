import streamlit as st
import os, json, re, random
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
from fpdf import FPDF

# ================= CONFIG =================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY missing in .env")
    st.stop()

client = Groq(api_key=api_key)
st.set_page_config(page_title="AI Exam Coach", layout="wide")

# ================= SESSION =================
defaults = {
    "chat_history": [],
    "vector_db": None,
    "quiz_data": None,
    "quiz_answers": {},
    "quiz_submitted": False,
    "score_recorded": False,
    "retake_count": 0,
    "quiz_history": [],
    "generated_notes": ""
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= PDF =================
def get_pdf_text(files):
    text = ""
    for file in files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)

def export_as_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Clean text for latin-1 compatibility (FPDF default)
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean_text)
    output = pdf.output(dest='S')
    return output.encode('latin-1') if isinstance(output, str) else output

@st.cache_resource
def embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector(chunks):
    return FAISS.from_texts(chunks, embedding=embeddings())

def load_vector():
    return st.session_state.get("vector_db")

# ================= AI =================
def generate_ai(messages):
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1200,
            temperature=0.7
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ AI Error: {e}"

# ================= CHAT =================
def chat_with_memory(context, question):
    messages = []

    for role, msg in st.session_state.chat_history[-6:]:
        messages.append({"role": role, "content": msg})

    messages.append({
        "role": "system",
        "content": f"""
        You are an AI Exam Coach.

        Give detailed answers:
        - 300–500+ words
        - Examples
        - Clear explanation

        Context:
        {context}
        """
    })

    messages.append({"role": "user", "content": question})
    return generate_ai(messages)

def download_chat():
    text = ""
    for role, msg in st.session_state.chat_history:
        text += f"{role.upper()}: {msg}\n\n"
    return text

# ================= QUIZ =================
def extract_json(text):
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        return None
    return None

def generate_quiz(context):
    prompt = f"""
    Generate 5 multiple-choice questions (MCQs) based on the provided context.
    
    Return ONLY a valid JSON list of objects with this structure:
    [
      {{
        "question": "The actual question text",
        "options": ["Option 1 content", "Option 2 content", "Option 3 content", "Option 4 content"],
        "answer": "The exact content of the correct option",
        "explanation": "A short explanation of why this answer is correct",
        "topic": "The specific subject/topic"
      }}
    ]

    IMPORTANT: 
    1. The 'options' list MUST contain actual text content derived from the context, NOT just letters like 'A', 'B', 'C', 'D'.
    2. The 'answer' field MUST match one of the strings in the 'options' list exactly.

    Context:
    {context}
    """
    raw = generate_ai([{"role": "user", "content": prompt}])
    quiz = extract_json(raw)
    if quiz:
        for q in quiz:
            random.shuffle(q["options"])
    return quiz

# ================= NOTES =================
def generate_notes(context):
    prompt = f"""
    Create detailed exam notes:
    - Headings
    - Key points
    - Examples
    - Easy revision

    Content:
    {context}
    """
    return generate_ai([{"role": "user", "content": prompt}])

# ================= MAIN =================
st.title("🎓 AI Exam Preparation Coach")

menu = st.sidebar.selectbox("Menu", [
    "Chat", "Quiz", "Flashcards", "Notes", "Planner", "Analytics"
])

# ---------- SIDEBAR ----------
with st.sidebar:
    files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if files:
            text = get_pdf_text(files)
            chunks = get_chunks(text)
            st.session_state.vector_db = create_vector(chunks)
            st.success("✅ PDFs processed!")
        else:
            st.warning("Upload PDFs")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []

    if st.session_state.chat_history:
        st.download_button("📥 Download Chat", download_chat(), "chat.txt")

# ---------- CHAT ----------
if menu == "Chat":
    st.title("💬 Smart Chat")

    db = load_vector()
    if not db:
        st.warning("Upload PDFs first")
    else:
        q = st.chat_input("Ask something...")

        if q:
            docs = db.similarity_search(q, k=5)
            context = "\n".join([d.page_content for d in docs])

            ans = chat_with_memory(context, q)

            st.session_state.chat_history.append(("user", q))
            st.session_state.chat_history.append(("assistant", ans))

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(msg)

# ---------- QUIZ ----------
elif menu == "Quiz":
    st.title("🧪 Interactive Quiz")

    db = load_vector()
    if not db:
        st.warning("⚠️ Please upload and process PDFs in the sidebar first!")
    elif st.button("Generate Quiz"):
        docs = db.similarity_search("important topics", k=5)
        context = "\n".join([d.page_content for d in docs])

        quiz = generate_quiz(context)
        if quiz:
            st.session_state.quiz_data = quiz
            st.session_state.quiz_answers = {i: None for i in range(len(quiz))}
            st.session_state.quiz_submitted = False
            st.session_state.score_recorded = False

    if st.session_state.quiz_data:
        for i, q in enumerate(st.session_state.quiz_data):
            st.write(f"### Q{i+1}: {q['question']}")
            ans = st.radio("Choose:", q["options"], key=f"q_{i}_{st.session_state.retake_count}", index=None)
            st.session_state.quiz_answers[i] = ans

        if not st.session_state.quiz_submitted:
            if st.button("Submit Quiz"):
                if any(v is None for v in st.session_state.quiz_answers.values()):
                    st.error("Please answer all questions before submitting!")
                else:
                    st.session_state.quiz_submitted = True
                    st.rerun()

        if st.session_state.quiz_submitted:
            score = 0
            topic_perf = {}

            for i, q in enumerate(st.session_state.quiz_data):
                sel = st.session_state.quiz_answers.get(i)
                correct = q.get("answer", "")
                topic = q.get("topic", "General")

                topic_perf.setdefault(topic, {"correct": 0, "total": 0})
                topic_perf[topic]["total"] += 1

                # Standardize comparison to fix logic errors
                if sel and str(sel).strip().lower() == str(correct).strip().lower():
                    score += 1
                    topic_perf[topic]["correct"] += 1
                    st.write(f"✅ Q{i+1}: Correct!")
                else:
                    st.write(f"❌ Q{i+1}: Wrong. Correct answer: {correct}")

            if not st.session_state.score_recorded:
                st.session_state.quiz_history.append({
                    "score": score,
                    "topics": topic_perf
                })
                st.session_state.score_recorded = True

            st.success(f"🎯 Final Score: {score}/{len(st.session_state.quiz_data)}")

            if st.button("🔄 Retake Quiz"):
                st.session_state.quiz_data = None
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.session_state.score_recorded = False
                st.session_state.retake_count += 1
                st.rerun()

# ---------- FLASHCARDS ----------
elif menu == "Flashcards":
    st.title("🧠 Flashcards")

    db = load_vector()
    if db and st.button("Generate"):
        docs = db.similarity_search("important", k=5)
        context = "\n".join([d.page_content for d in docs])
        res = generate_ai([{"role": "user", "content": f"Generate flashcards:\n{context}"}])
        st.write(res)

# ---------- NOTES ----------
elif menu == "Notes":
    st.title("📝 Generate Notes")

    db = load_vector()
    if db and st.button("Generate Notes"):
        docs = db.similarity_search("important concepts", k=8)
        context = "\n".join([d.page_content for d in docs])
        st.session_state.generated_notes = generate_notes(context)

    if st.session_state.generated_notes:
        st.write(st.session_state.generated_notes)

        pdf_bytes = export_as_pdf(st.session_state.generated_notes)
        st.download_button(
            "📥 Download Notes (PDF)",
            pdf_bytes,
            "notes.pdf",
            "application/pdf"
        )

# ---------- PLANNER ----------
elif menu == "Planner":
    topic = st.text_input("Enter topic")

    if st.button("Generate Plan"):
        plan = generate_ai([{"role": "user", "content": f"Create study plan for {topic}"}])
        st.write(plan)

# ---------- ANALYTICS ----------
elif menu == "Analytics":
    st.title("📊 Analytics")

    if not st.session_state.quiz_history:
        st.info("No quiz attempts yet")
    else:
        scores = [q["score"] for q in st.session_state.quiz_history]
        st.line_chart(scores)

        topic_stats = {}
        for q in st.session_state.quiz_history:
            for t, d in q["topics"].items():
                topic_stats.setdefault(t, {"correct": 0, "total": 0})
                topic_stats[t]["correct"] += d["correct"]
                topic_stats[t]["total"] += d["total"]

        weak = []
        for t, d in topic_stats.items():
            acc = d["correct"] / d["total"]
            st.write(f"{t}: {acc:.2f}")
            if acc < 0.5:
                weak.append(t)

        if weak:
            st.error("Weak Topics:")
            for w in weak:
                st.write(f"🔴 {w}")
        else:
            st.success("Strong in all topics!")