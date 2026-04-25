import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# 1. SETUP & CONFIG
load_dotenv()
st.set_page_config(page_title="Investment RAG System", page_icon="📘")
st.title("📘 Investment RAG System")

# 2. INITIALIZE LLM & EMBEDDINGS
# Custom Robust Embedding Wrapper (Fixes all dependency conflicts)
class SimpleEmbeddings(Embeddings):
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()
    def __call__(self, text):
        return self.embed_query(text)

@st.cache_resource
def load_embeddings():
    # Using the custom wrapper to avoid LangChain version conflicts
    return SimpleEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()
st.sidebar.success("Embeddings: Local (Sentence-Transformers)")

# LLM: Prioritizing Groq for speed
if os.getenv("GROQ_API_KEY"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
    st.sidebar.info("LLM: Groq (Llama 3)")
elif os.getenv("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    st.sidebar.info("LLM: OpenAI")
elif os.getenv("GOOGLE_API_KEY"):
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    st.sidebar.info("LLM: Gemini")
else:
    st.error("❌ No LLM API Key found.")
    st.stop()

# RESET BUTTON (Critical for picking up code changes)
if st.sidebar.button("Reset System"):
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()

# 3. PDF PROCESSING FUNCTION
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    # DEBUG: Terminal output requirements
    print("\n--- DEBUG: FIRST 2 CHUNKS ---")
    for i, chunk in enumerate(chunks[:2]):
        print(f"Chunk {i+1}: {chunk.page_content[:200]}...")
    
    print("\n--- DEBUG: FIRST EMBEDDING VECTOR (First 10 values) ---")
    vector = embeddings.embed_query(chunks[0].page_content)
    print(vector[:10])
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.remove(tmp_path)
    return vectorstore

# 4. UI COMPONENTS
uploaded_file = st.file_uploader("Upload Investment Textbook (PDF)", type="pdf")

if uploaded_file:
    if "vectorstore" not in st.session_state:
        with st.spinner("Analyzing document..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
            st.success("✅ Document Indexing Complete!")

    # Question Buttons
    st.markdown("### 🧪 Quick Questions")
    questions = [
        "how to deal with brokerage houses?",
        "what is theory of diversification?",
        "how to become intelligent investor?",
        "how to do business valuation?",
        "what is putting all eggs in one basket analogy?"
    ]
    
    selected_q = None
    for q in questions:
        if st.button(q):
            selected_q = q

    query = st.text_input("Or enter your own question:", value=selected_q if selected_q else "")

    if st.button("Ask") or selected_q:
        if not query:
            st.warning("Please enter or select a question.")
        else:
            with st.spinner("Searching for answers..."):
                docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(query, k=3)
                context = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])
                
                # Answer generation with strict prompt
                prompt = f"""Answer ONLY from the provided context. If answer is not found, say 'Not found in document'.
                
                Context:
                {context}
                
                Question: {query}
                
                Answer:"""
                
                response = llm.invoke(prompt).content
                
                st.markdown("---")
                st.markdown(f"### 🤖 Answer:\n{response}")
else:
    st.info("Please upload a PDF file to start.")
