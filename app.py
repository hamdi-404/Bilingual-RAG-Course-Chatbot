import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import tempfile
from langdetect import detect


#  Page Config

st.set_page_config(page_title="AI Bilingual Tutor 💬", layout="wide")
st.title("📘 AI Course Chatbot (Arabic + English)")

# Load Lightweight Model (fast)

@st.cache_resource(show_spinner=True)
def load_model():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config
    )
    return tokenizer, model

tokenizer, model = load_model()

# Session State

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF

if st.session_state.vectorstore is None:
    uploaded_file = st.file_uploader("📂 Upload your course/book (PDF) to begin chatting", type=["pdf"])
    if uploaded_file:
        with st.spinner("📚 Processing your file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = splitter.split_documents(pages)

            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embedder)
            st.session_state.vectorstore = vectorstore

        st.success("✅ Textbook processed! Start chatting below.")
        st.rerun()

else:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    st.markdown("### 💬 Chat with your course")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything (English or Arabic)...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Detect language
                try:
                    lang = detect(user_input)
                except:
                    lang = "en"

                relevant_docs = retriever.invoke(user_input)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                if lang == "ar":
                    prompt = f"""
أنت مساعد ذكي متخصص في شرح المناهج الدراسية.
استخدم النص التالي من الكتاب لتقديم إجابة مختصرة وواضحة فقط.
لا تكرر السؤال أو النص، فقط أعد الجواب النهائي.

النص:
{context}

السؤال:
{user_input}

الجواب:
"""
                else:
                    prompt = f"""
You are a helpful AI tutor.
Use the textbook context below to give a clear, short, direct answer.
Do NOT repeat the question or context. Output only the final answer.

Context:
{context}

Question:
{user_input}

Answer:
"""

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        temperature=0.3,
                        do_sample=False
                    )
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                answer = (
                    output_text.replace(prompt, "")
                    .split("Answer:")[-1]
                    .split("الجواب:")[-1]
                    .strip()
                )

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
