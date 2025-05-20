# 📘 QueryGenius: AI-Powered Document & Web Assistant

🚀 **QueryGenius** is a powerful, user-friendly RAG (Retrieval-Augmented Generation) system that enables users to upload documents or enter URLs and interact with them via natural language queries.

It uses **LangChain**, **FAISS**, and **OpenRouter LLMs** with model-switching capability to dynamically integrate state-of-the-art open-source models like **DeepSeek**, **LLaMA**, **Qwen**, **Gemini**, and **Mistral**.

> 🌐 **Live Demo**: [Click to Try the App](https://documentchatbot-dmlqpeszsfpttc9gw5gqng.streamlit.app/)

---

## ✨ Features

* 📄 Upload and query **PDF**, **CSV**, or **JSON** files
* 🌍 Scrape and query **website content**
* 🧠 RAG pipeline using **LangChain + FAISS** for contextual answers
* 🔄 **Model-switching dropdown** to choose from multiple OpenRouter-hosted LLMs
* 💬 Conversational chat interface powered by **Streamlit**
* 💾 Save full Q\&A sessions to JSON
* 🔒 API Key integration using Streamlit secrets

---

## ⚙️ Available LLMs (via OpenRouter)

Select your preferred model from the sidebar:

| Model Name                         | Description                                  |
| ---------------------------------- | -------------------------------------------- |
| `mistralai/mistral-nemo:free`      | Lightweight, fast model by Mistral           |
| `qwen/qwen3-235b-a22b:free`        | High-performance Chinese-English model       |
| `google/gemini-2.0-flash-exp:free` | Google's fast inference Gemini 2.0           |
| `meta-llama/llama-4-maverick:free` | Cutting-edge open-source LLaMA 4-based model |

---

## 🧱 Tech Stack

| Component      | Technology                                         |
| -------------- | -------------------------------------------------- |
| Backend        | Python                                             |
| Framework      | [Streamlit](https://streamlit.io)                  |
| Vector Store   | [FAISS](https://github.com/facebookresearch/faiss) |
| Language Model | OpenRouter API (various LLMs)                      |
| Text Chunking  | `RecursiveCharacterTextSplitter` from LangChain    |
| RAG Pipeline   | `RetrievalQA` from LangChain                       |
| Parsing        | `PDFPlumber`, `pandas`, `BeautifulSoup`            |
| Deployment     | Streamlit Cloud                                    |

---

## 🧠 System Architecture

```plaintext
             ┌────────────────────┐
             │  User Input (UI)   │
             └─────────┬──────────┘
                       │
            ┌──────────▼───────────┐
            │ Upload / Scrape Data │
            └──────────┬───────────┘
                       │
             ┌─────────▼─────────┐
             │ Chunk & Vectorize │ ← FAISS + LangChain
             └─────────┬─────────┘
                       │
              ┌────────▼────────┐
              │   Model Switch  │ ← via Streamlit selectbox
              └────────┬────────┘
                       │
         ┌─────────────▼────────────────┐
         │ Query RAG Chain (Retriever + │
         │    Selected LLM via OpenRouter) │
         └─────────────┬────────────────┘
                       │
             ┌─────────▼────────┐
             │   Display Answer │
             └──────────────────┘
```

---

## 🔄 Application Flow

1. **User selects** a data source:

   * Upload a PDF
   * Upload a CSV or JSON
   * Input a website URL

2. The content is:

   * Loaded via appropriate loaders
   * Chunked into semantic blocks using `RecursiveCharacterTextSplitter`
   * Converted into vector embeddings via `OpenAIEmbeddings` (used with OpenRouter API)

3. The app builds a FAISS vector store and configures a **retriever**

4. The user **selects one of the supported models** from a dropdown

5. On query, LangChain’s `RetrievalQA` chain is used with:

   * Selected model from OpenRouter
   * Prompt template to ensure structured and factual answers

6. Chat is streamed via Streamlit’s interactive `st.chat_message` UI

7. Q\&A logs can be saved locally in `qa_responses.json`

---
### Development
## 📦 File Structure

```bash
📦 Document_RAG/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Required packages
├── qa_responses.json       # Saved Q&A (optional, auto-generated)
└── .streamlit/
    └── secrets.toml        # API key for OpenRouter (not version controlled)
```

---

## 🔑 Setup & Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/QueryGenius-RAG.git
cd QueryGenius-RAG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add OpenRouter API Key

Create `.streamlit/secrets.toml`:

```toml
[api]
openrouter_api_key = "sk-or-your-openrouter-key"
```

### 4. Run Locally

```bash
streamlit run app.py
```


---

## 🔐 Deployment on Streamlit Cloud

1. Push your project to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and deploy your app.
3. Add `secrets.toml` API key via the "Secrets" section in Streamlit Cloud Settings.

---

## 💡 Additional Features

* ✅ **Model Switching**: Easily switch between high-performance OpenRouter-hosted models without changing code
* 🌍 **Web Scraping**: Enter a URL and instantly query its content
* 📁 **Tabular Data Handling**: Intelligent parsing of both JSON and CSV data formats
* 💬 **Chat Log**: View full chat history with option to export
* 🧠 **Context-aware Answers**: Using FAISS + LangChain RAG
* ☁️ **Deploy on Streamlit Cloud** with one click

---

## 📚 Citations & Credits

| Task                                | Source                                         |
| ----------------------------------- | ---------------------------------------------- |
| API Errors & Fixes                  | [Stack Overflow](https://stackoverflow.com)    |
| Code Structuring, Model Integration | ChatGPT (OpenAI)                               |
| Prompt Design, Grammar Fixes        | ChatGPT (Grammer)                              |
| UI & Component Styling              | [Streamlit AI Assistant](https://streamlit.io) |
| Readme Styling                      | Claude Sonnet 3.5                              |
---

## 🙌 Acknowledgements
Thanks to:
* **OpenRouter** for hosting excellent open LLMs
* **LangChain** for robust RAG support
* **Streamlit** for an incredibly smooth developer UX

---
