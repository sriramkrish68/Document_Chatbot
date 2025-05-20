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

 
# Streamlit Configuration
 

st.set_page_config(page_title="📘 QueryGenius", layout="wide")
st.markdown("""
<style>
    .main { background-color: #1a1a1a; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #2d2d2d; }
    .stTextInput textarea, .stChatInput input { color: #ffffff !important; }
    .stSelectbox div[data-baseweb="select"], div[role="listbox"] div {
        color: white !important; background-color: #3d3d3d !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📘 QueryGenius")
st.caption("🚀 Your Intelligent Document & Web Assistant")

 
# Sidebar Configuration
 

# Sidebar Configuration (model selection updated)
with st.sidebar:
    st.header("⚙️ Configuration")
    data_source = st.radio("Select Data Source", ["Upload PDF", "Upload CSV/JSON", "Scrape Website"])
    file = None
    website_url = ""
    if data_source == "Upload PDF":
        file = st.file_uploader("Upload a PDF", type=["pdf"])
    elif data_source == "Upload CSV/JSON":
        file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    else:
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
    response = requests.get(url)
    if response.ok:
        soup = BeautifulSoup(response.text, "html.parser")
        return "\n".join([p.get_text() for p in soup.find_all("p")])
    return "Error fetching website."

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.create_documents([text])

 
# OpenRouter-Compatible Chat Model

class OpenRouterLLM(ChatOpenAI):
    def __init__(self, model_name):
        super().__init__(
            openai_api_key=st.secrets["api"]["openrouter_api_key"],
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
    If the answer is unknown, say "I don't know."

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

 
# Chat Interface
 

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! How can I assist you today?"}]
    st.session_state.responses = []

chat_box = st.container()

with chat_box:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_input = st.chat_input("Ask me anything...")

def respond_to_query(query):
    response = rag_chain.run(query)
    st.session_state.responses.append({"question": query, "answer": response})
    return response

if user_input and knowledge_base:
    st.session_state.message_log.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        response = respond_to_query(user_input)
    st.session_state.message_log.append({"role": "ai", "content": response})
    with st.chat_message("ai"):
        st.markdown(response)

 
# Save Q&A Session to JSON
 

if st.sidebar.button("📥 Save Q&A Session"):
    output_file = "qa_responses.json"
    with open(output_file, "w") as f:
        json.dump(st.session_state.responses, f, indent=2)
    st.sidebar.success(f"Responses saved to `{output_file}`")
