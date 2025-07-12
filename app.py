import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import tempfile
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
from io import StringIO, BytesIO

# Streamlit Configuration
st.set_page_config(page_title="📘 QueryGenius", layout="wide")
st.markdown("""
<style>
    .main { 
        background-color: #1a1a1a; 
        color: #ffffff; 
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content { 
        background-color: #2d2d2d; 
        border-radius: 10px;
        padding: 20px;
    }
    .stTextInput textarea, .stChatInput input { 
        color: #ffffff !important; 
        background-color: #3d3d3d !important;
        border-radius: 8px;
        padding: 10px;
    }
    .stSelectbox div[data-baseweb="select"], div[role="listbox"] div {
        color: white !important; 
        background-color: #3d3d3d !important;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #2d3748;
        margin-left: auto;
    }
    .ai-message {
        background-color: #4a5568;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

st.title("📘 QueryGenius")
st.caption("🚀 Your Intelligent Document & Web Assistant")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    data_source = st.radio("Select Data Source", ["None", "Upload PDF", "Upload CSV/JSON", "Scrape Website"])
    file = None
    website_url = ""
    if data_source == "Upload PDF":
        file = st.file_uploader("Upload a PDF", type=["pdf"])
    elif data_source == "Upload CSV/JSON":
        file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    elif data_source == "Scrape Website":
        website_url = st.text_input("Enter website URL")

    st.divider()
    st.markdown("### 🤖 Select OpenRouter Model")
    selected_model = st.selectbox(
        "Choose a model:",
        [
            "mistralai/mistral-nemo:free",
            "qwen/qwen3-235b-a22b:free",
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-4-maverick:free"
        ],
        index=0
    )
    st.markdown("Built with [LangChain](https://docs.langchain.com) + [OpenRouter](https://openrouter.ai)")

# Load and Chunk Document
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return "\n".join([doc.page_content for doc in loader.load()])

def load_tabular(file, file_type):
    if file_type == "csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_json(file)
    return df.to_string(index=False)

def scrape_website(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return "\n".join([p.get_text() for p in soup.find_all("p")])
    except:
        return "Error fetching website."

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.create_documents([text])

# OpenRouter-Compatible Chat Model
class OpenRouterLLM(ChatOpenAI):
    def __init__(self, model_name):
        super().__init__(
            openai_api_key=st.secrets.get("api", {}).get("openrouter_api_key", "your-api-key"),
            base_url="https://openrouter.ai/api/v1",
            model_name=model_name,
        )

# Build RAG Pipeline
def build_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

def build_rag_chain(vector_store, model_name):
    retriever = vector_store.as_retriever()
    prompt = PromptTemplate.from_template("""
    You are an intelligent assistant. Use the context to answer the user's query.
    If no context is provided or the answer is unknown, provide a general response based on your knowledge.

    Question: {question}
    Context: {context}
    Answer:
    """)
    return RetrievalQA.from_chain_type(
        llm=OpenRouterLLM(model_name=model_name),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# Load & Process Documents
knowledge_base = None
if data_source != "None" and (file or website_url):
    try:
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            if file.name.endswith(".pdf"):
                text_data = load_pdf(tmp_path)
            else:
                text_data = load_tabular(tmp_path, file.name.split('.')[-1])
        elif website_url:
            text_data = scrape_website(website_url)
        else:
            text_data = None

        if text_data:
            docs = chunk_text(text_data)
            knowledge_base = build_vector_store(docs)
            rag_chain = build_rag_chain(knowledge_base, selected_model)
            st.success("✅ Knowledge base loaded and indexed successfully.")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")

# Initialize default RAG chain for no-document case
if not knowledge_base:
    # Create a dummy vector store with a generic document
    dummy_text = "This is a general knowledge assistant. Ask me anything!"
    docs = chunk_text(dummy_text)
    knowledge_base = build_vector_store(docs)
    rag_chain = build_rag_chain(knowledge_base, selected_model)

# Chat Interface
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! How can I assist you today?"}]
    st.session_state.responses = []

chat_box = st.container()
with chat_box:
    for message in st.session_state.message_log:
        role_class = "user-message" if message["role"] == "user" else "ai-message"
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "ai" else "👤"):
            st.markdown(f'<div class="chat-message {role_class}">{message["content"]}</div>', unsafe_allow_html=True)

user_input = st.chat_input("Ask me anything...")

def respond_to_query(query):
    try:
        response = rag_chain.run(query)
        st.session_state.responses.append({"question": query, "answer": response})
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"

if user_input:
    st.session_state.message_log.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        response = respond_to_query(user_input)
    st.session_state.message_log.append({"role": "ai", "content": response})
    with st.chat_message("ai", avatar="🤖"):
        st.markdown(f'<div class="chat-message ai-message">{response}</div>', unsafe_allow_html=True)
    st.rerun()

# Download Q&A Session as JSON
if st.sidebar.button("📥 Download Q&A Session"):
    if st.session_state.responses:
        json_str = json.dumps(st.session_state.responses, indent=2)
        json_bytes = json_str.encode('utf-8')
        st.sidebar.download_button(
            label="Download JSON",
            data=json_bytes,
            file_name="qa_responses.json",
            mime="application/json"
        )
        st.sidebar.success("Ready to download Q&A session!")
    else:
        st.sidebar.warning("No responses to download yet.")
