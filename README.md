# 📘 Investment RAG System

A powerful Retrieval-Augmented Generation (RAG) application designed to analyze investment textbooks and documents. This system allows users to upload PDF files and ask complex questions, receiving answers strictly based on the provided content.

Live Link - https://8s4qbncbbyqwgrjfh5ddyf.streamlit.app/

## 🚀 Features

- **PDF Analysis**: Upload any investment-related PDF for instant indexing.
- **Local Embeddings**: Uses `sentence-transformers` for efficient, local vector generation.
- **Fast Search**: Powered by `FAISS` for high-performance similarity search.
- **Multi-LLM Support**: 
  - **Groq** (Llama 3.1) - Optimized for speed.
  - **OpenAI** (GPT-3.5 Turbo) - Industry standard.
  - **Google Gemini** (Gemini 1.5 Pro) - Advanced reasoning.
- **Interactive UI**: Built with Streamlit for a smooth, responsive user experience.
- **Quick Questions**: Pre-configured test queries for rapid evaluation.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Framework**: [LangChain](https://www.langchain.com/)
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLMs**: Groq, OpenAI, Google Gemini

## 📋 Prerequisites

- Python 3.10+
- At least one API key from:
  - [Groq Cloud](https://console.groq.com/)
  - [OpenAI](https://platform.openai.com/)
  - [Google AI Studio](https://aistudio.google.com/)

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "Rag document"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_key_here
   OPENAI_API_KEY=your_openai_key_here
   GOOGLE_API_KEY=your_google_key_here
   ```

## 🎮 Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

1. Upload a PDF document via the sidebar/main area.
2. Wait for the "Document Indexing Complete" notification.
3. Click on a "Quick Question" or type your own query in the text input.
4. The system will provide an answer derived **only** from the uploaded document.

## 📁 Project Structure

```text
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (API keys)
├── .gitignore          # Files to ignore in Git
└── README.md           # Project documentation
```

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Built with ❤️ for Investment Analysis*
