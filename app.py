# ---------------- IMPORTS ----------------
import os
import json
from pathlib import Path
import faiss
import streamlit as st
import google.generativeai as genai

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Optional

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


# ---------------- CONFIG ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path("storage")
INDEX_DIR = BASE_DIR / "index"
FAISS_DIR = BASE_DIR / "faiss"
PDF_DIR = Path("data/pdfs")

INDEX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- EMBEDDING SETUP ----------------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbedding(model_name=EMBED_MODEL)


# ---------------- BUILD / LOAD INDEX ----------------
def build_index_from_pdfs(pdf_paths):
    st.info("📄 Building FAISS index from uploaded PDFs...")
    documents = []
    for pdf_path in pdf_paths:
        docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
        documents.extend(docs)

    embed_model = get_embedding_model()
    sample_emb = embed_model.get_text_embedding("hello")
    embedding_dim = len(sample_emb)

    faiss_index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    Settings.embed_model = embed_model
    Settings.chunk_size = 3000
    Settings.chunk_overlap = 1000

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    faiss.write_index(faiss_index, str(FAISS_DIR / "faiss.index"))

    st.success("✅ Index built successfully!")
    return index


def load_index():
    faiss_index_path = FAISS_DIR / "faiss.index"
    if not faiss_index_path.exists():
        raise FileNotFoundError("FAISS index not found. Please upload PDFs first.")
    faiss_index = faiss.read_index(str(faiss_index_path))
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=str(INDEX_DIR)
    )
    return load_index_from_storage(storage_context)


# ---------------- LANGGRAPH WORKFLOW ----------------
class SelfRAGState(TypedDict, total=False):
    query: str
    retriever: Optional[str]
    answer: Optional[str]
    critic_result: Optional[dict]


def create_selfrag_graph(index):
    """Creates Self-RAG workflow using latest LangGraph API (>=0.2)"""

    # ---- Tool: Retriever ----
    def retriever_node(state: SelfRAGState):
        query = state["query"]
        retriever = index.as_retriever(similarity_top_k=8)
        results = retriever.retrieve(query)
        context = "\n\n".join([r.text for r in results])
        return {"retriever": context}

    # ---- Node: Generator ----
    def generator_node(state: SelfRAGState):
        context = state.get("retriever", "")
        query = state["query"]

        # 🧠 Add conversation memory context
        memory_context = st.session_state.memory.get_context()

        prompt = f"""
            You are a helpful assistant that answers based on PDFs and past chat.
            Use only the CONTEXT and CHAT HISTORY below to answer.
            If you don't find the answer, say "I don't know based on the provided documents."

            CHAT HISTORY:
            {memory_context}

            CONTEXT FROM DOCUMENTS:
            {context}

            USER QUESTION:
            {query}
        """

        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        return {"answer": response.text}


    # ---- Node: Critic ----
    def critic_node(state: SelfRAGState):
        context = state.get("retriever", "")
        answer = state.get("answer", "")
        query = state["query"]

        prompt = f"""
        You are a critic evaluating an AI's answer.
        Return ONLY JSON:
        {{
          "decision": "ACCEPT" or "RETRIEVE_MORE",
          "reason": "short explanation",
          "refine_query": "improved query if more retrieval needed"
        }}

        Original Question: {query}
        Answer: {answer}
        Context: {context}
        """

        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        try:
            result = json.loads(response.text)
        except Exception:
            result = {"decision": "ACCEPT", "reason": "Parsing error"}
        return {"critic_result": result}

    # ---- Graph Build ----
    graph = StateGraph(SelfRAGState)  # ✅ Required schema
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "critic")

    def feedback_condition(state: SelfRAGState):
        result = state.get("critic_result", {})
        return "retriever" if result.get("decision") == "RETRIEVE_MORE" else END

    graph.add_conditional_edges("critic", feedback_condition)
    return graph.compile()

# ---------------- MEMORY ----------------
class ConversationMemory:
    """Simple in-memory chat history for context-aware RAG."""
    def __init__(self):
        self.history = []

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > 10:  # limit memory size
            self.history.pop(0)

    def get_context(self):
        """Returns a combined text of past messages"""
        return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.history])

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Self-RAG PDF Chatbot (LangGraph)", page_icon="🤖", layout="wide")
st.title("🤖 Self-RAG PDF Chatbot (LangGraph + LlamaIndex + Gemini)")

if "index" not in st.session_state:
    st.session_state.index = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

uploaded_files = st.file_uploader("📁 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    saved_paths = []
    for f in uploaded_files:
        path = PDF_DIR / f.name
        with open(path, "wb") as fh:
            fh.write(f.getbuffer())
        saved_paths.append(path)

    st.session_state.index = build_index_from_pdfs(saved_paths)
    st.session_state.graph = create_selfrag_graph(st.session_state.index)

# Chat UI
st.subheader("💬 Ask questions about your PDFs")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.memory.add_message("user", user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.graph is None:
        with st.chat_message("assistant"):
            st.markdown("⚠️ Please upload and index PDFs first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking with Self-RAG (LangGraph)..."):
                result = st.session_state.graph.invoke({"query": user_input})
                answer = result.get("answer", "⚠️ No answer generated.")
                st.markdown(answer)
                st.session_state.memory.add_message("assistant", answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.caption("🔁 Powered by LangGraph | Self-RAG (Retriever → Generator → Critic loop)")
