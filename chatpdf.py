import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# 🔗 Modern QA Chain (NO load_qa_chain ❌)
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say:
    "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # model = ChatGoogleGenerativeAI(
    #     model="gemini-1.5-flash",
    #     temperature=0.3
    # )
    
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # ✅ Custom chain (modern approach)
    def chain(docs, question):
        context = "\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.format(
            context=context,
            question=question
        )

        # response = model.invoke(final_prompt)

        # return response.content
        
        response = model.generate_content(final_prompt)
        return response.text

    return chain



# 💬 Handle user query
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ✅ Load FAISS safely
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    answer = chain(docs, user_question)

    st.write("Reply:", answer)




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()