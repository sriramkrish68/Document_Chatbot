# 📘 QueryGenius: Intelligent Document & Web Assistant

QueryGenius is a **Retrieval-Augmented Generation (RAG)** powered assistant that can understand your **PDFs, CSVs, JSONs**, and even **scrape and query websites**. It integrates **DeepSeek LLM (via OpenRouter API)** for fluent natural language responses and uses **HuggingFace embeddings** for document understanding and search.

Built with **LangChain**, **Streamlit**, and **FAISS**, this tool offers an end-to-end smart assistant for document analysis, web scraping, and conversational Q\&A.

---

## 🚀 Features

* ✅ Upload and query **PDF** documents
* ✅ Load and analyze **CSV / JSON** files
* ✅ **Scrape website content** and query it
* ✅ **Conversational Q\&A** using `deepseek-chat-v3-0324:free` via OpenRouter
* ✅ Uses **HuggingFace's `all-MiniLM-L6-v2`** for fast and free local embedding
* ✅ Saves Q\&A sessions in `qa_responses.json`
* ✅ Clean UI with **Streamlit Dark Theme**
* ✅ Modular RAG pipeline using LangChain's components

---

## 📂 Project Structure

```
Document_RAG/
│
├── app.py                     # Main Streamlit app
├── qa_responses.json          # Saved Q&A session (generated at runtime)
├── .streamlit/
│   └── secrets.toml           # API keys for OpenRouter
└── requirements.txt           # All required dependencies
```

---

## 🛠️ Tech Stack Used

| Purpose                   | Library / Tool                   |
| ------------------------- | -------------------------------- |
| Web UI                    | Streamlit                        |
| PDF/CSV/JSON Reading      | pandas, PDFPlumber               |
| Web Scraping              | BeautifulSoup, requests          |
| Document Chunking         | LangChain Text Splitters         |
| Embeddings (local)        | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector DB                 | FAISS                            |
| Natural Language Response | DeepSeek v3 via OpenRouter       |
| RAG Framework             | LangChain                        |

---

## 🧠 Flow of the Application

1. **User Uploads or Enters Data Source**

   * Options: PDF, CSV/JSON, or Website URL.

2. **Document Loader**

   * PDF: Parsed using `PDFPlumberLoader`
   * CSV/JSON: Read via pandas
   * Website: Scraped using `requests` + `BeautifulSoup`

3. **Text Chunking**

   * Long text is broken into smaller chunks using `RecursiveCharacterTextSplitter`.

4. **Vector Store Creation**

   * Chunks are embedded using `HuggingFaceEmbeddings`.
   * Stored in a FAISS vector store for similarity search.

5. **RAG Retrieval Chain**

   * When the user asks a question:

     * Relevant chunks are retrieved from FAISS.
     * Prompt is formatted using `PromptTemplate`.
     * Answer generated using `deepseek-chat-v3-0324:free` LLM via OpenRouter API.

6. **Chat Interface**

   * Previous messages are saved in session state.
   * Users can ask follow-ups conversationally.

7. **Export**

   * Users can download the full Q\&A session as `qa_responses.json`.

---

## 🔐 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/QueryGenius.git
cd QueryGenius
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API Key

Create a file at `.streamlit/secrets.toml`:

```toml
[api]
openrouter_api_key = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Get your API key from [https://openrouter.ai](https://openrouter.ai)

---

## 📦 Requirements

### `requirements.txt`

```txt
streamlit
langchain
langchain-community
langchain-core
faiss-cpu
huggingface_hub
pdfplumber
beautifulsoup4
pandas
openai
```

---

## 🧪 Example Use Cases

* 🔍 Upload a company’s financial report PDF → Ask: “What is the net profit for Q4?”
* 📈 Load a JSON export of analytics → Ask: “Which day had the highest traffic?”
* 🌐 Scrape a website blog → Ask: “What are the key ideas in the latest post?”

---

## ✨ Highlights

| Feature             | Description                                                                   |
| ------------------- | ----------------------------------------------------------------------------- |
| **RAG-based**       | Uses Retrieval-Augmented Generation to ground LLM answers in document content |
| **DeepSeek API**    | High-performance free model integrated via OpenRouter                         |
| **Free Embeddings** | No OpenAI billing; uses HuggingFace locally                                   |
| **Web Scraper**     | Supports live web page ingestion                                              |
| **Chat History**    | Remembers past conversations per session                                      |
| **Exportable**      | Easily save your full session as JSON                                         |

---

## 🧾 Credits & Citations

* 🧠 **LLM Backend & Embedding Help**:

  * [LangChain Documentation](https://docs.langchain.com/)
  * [HuggingFace Sentence Transformers](https://www.sbert.net/)
  * [OpenRouter](https://openrouter.ai/)

* 🐞 **Error Fixing & Debugging References**:

  * [Streamlit Secrets KeyError - Stack Overflow](https://stackoverflow.com/questions/75599488/keyerror-st-secrets-has-no-key)

* 🛠 **Code Structuring, Prompt Design, Rephrasing**: ChatGPT (OpenAI)

* 🖼 **UI Design Suggestions**: Streamlit AI Assistant

---
