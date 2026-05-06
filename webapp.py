import streamlit as st
import os, json, re
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv

# ================= CONFIG =================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY missing in .env")
    st.stop()

genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-2.5-flash")

st.set_page_config(page_title="AI Exam Coach", layout="wide")

# ================= SESSION =================
defaults = {
    "chat_history": [],
    "quiz_data": None,
    "quiz_answers": {},
    "quiz_submitted": False,
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_text(text)

@st.cache_resource
def embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector(chunks):
    db = FAISS.from_texts(chunks, embedding=embeddings())
    db.save_local("faiss_index")

def load_vector():
    if not os.path.exists("faiss_index"):
        return None
    return FAISS.load_local("faiss_index", embeddings(), allow_dangerous_deserialization=True)

# ================= GEMINI AI =================
def generate_ai(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Error: {e}"

# ================= CHAT =================
def chat_with_memory(context, question):
    history_text = ""
    for role, msg in st.session_state.chat_history[-6:]:
        history_text += f"{role}: {msg}\n"

    prompt = f"""
    You are an AI Exam Coach.

    Give LONG detailed answers:
    - Minimum 500 words
    - Use examples
    - Use headings

    Chat History:
    {history_text}

    Context:
    {context}

    Question:
    {question}
    """

    return generate_ai(prompt)

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
    Generate 5 MCQs.

    ONLY JSON FORMAT:

    [
      {{
        "question": "text",
        "options": ["A","B","C","D"],
        "answer": "correct option",
        "topic": "topic"
      }}
    ]

    Context:
    {context}
    """

    raw = generate_ai(prompt)
    return extract_json(raw)

# ================= NOTES =================
def generate_notes(context):
    prompt = f"""
    Create detailed exam notes:
    - Headings
    - Key points
    - Examples
    - 800+ words

    Content:
    {context}
    """
    return generate_ai(prompt)

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
            create_vector(chunks)
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
    if db and st.button("Generate Quiz"):
        docs = db.similarity_search("important topics", k=5)
        context = "\n".join([d.page_content for d in docs])

        quiz = generate_quiz(context)
        if quiz:
            st.session_state.quiz_data = quiz
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False

    if st.session_state.quiz_data:
        for i, q in enumerate(st.session_state.quiz_data):
            st.write(f"### Q{i+1}: {q['question']}")
            ans = st.radio("Choose:", q["options"], key=f"q_{i}")
            st.session_state.quiz_answers[i] = ans

        if not st.session_state.quiz_submitted:
            if st.button("Submit Quiz"):
                st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:
            score = 0

            for i, q in enumerate(st.session_state.quiz_data):
                if st.session_state.quiz_answers.get(i) == q["answer"]:
                    score += 1

            st.success(f"🎯 Score: {score}/{len(st.session_state.quiz_data)}")

# ---------- FLASHCARDS ----------
elif menu == "Flashcards":
    st.title("🧠 Flashcards")

    db = load_vector()
    if db and st.button("Generate"):
        docs = db.similarity_search("important", k=5)
        context = "\n".join([d.page_content for d in docs])
        res = generate_ai(f"Generate flashcards:\n{context}")
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

        st.download_button(
            "📥 Download Notes",
            st.session_state.generated_notes,
            "notes.txt"
        )

# ---------- PLANNER ----------
elif menu == "Planner":
    topic = st.text_input("Enter topic")

    if st.button("Generate Plan"):
        plan = generate_ai(f"Create a detailed study plan for {topic}")
        st.write(plan)

# ---------- ANALYTICS ----------
elif menu == "Analytics":
    st.title("📊 Analytics")

    if not st.session_state.quiz_history:
        st.info("No quiz attempts yet")
    else:
        scores = [q["score"] for q in st.session_state.quiz_history]
        st.line_chart(scores)