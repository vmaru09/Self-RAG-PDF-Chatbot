# 🤖 Self-RAG PDF Chatbot  
A Streamlit app that uses LangGraph, LlamaIndex, Gemini, and FAISS to answer questions based on your uploaded PDFs.

---

## 🧠 About the Project  
This project is an AI-powered PDF question-answering chatbot built using:

- **Self-RAG pipeline** (Retriever → Generator → Critic) using LangGraph  
- **LlamaIndex** for PDF loading and document chunking  
- **FAISS** for vector search  
- **Gemini 2.5 Flash** as the LLM  
- **Streamlit** for the interactive chat UI  

Upload PDFs → FAISS builds an index → Ask questions → Get accurate answers using retrieval + reasoning.

---

## 🚀 Features  

- Upload multiple PDFs  
- Builds FAISS vector index  
- Self-RAG loop improves answer quality  
- Memory of last 10 chat messages  
- Streamlit chat interface  
- Uses sentence-transformers for embeddings  
- Stores index locally in `storage/`

---

## 📂 Project Structure  

your_project/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│── storage/ # auto-created
│── data/
│ └── pdfs/ # uploaded PDFs saved here


---

## 🔐 Environment Variables  

Create a `.env` file:

GOOGLE_API_KEY=your_gemini_api_key_here

Make sure `.env` is added to `.gitignore`.

---

## 📦 Installation  

### 1. Clone the Repository  

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
---

### 2. Create Virtual Environment
```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
---

### 3. Install Requirements
```
pip install -r requirements.txt
```
---
## ▶️ Run the App
```
streamlit run app.py
```
Then open the URL shown in terminal (usually http://localhost:8501).

## 📝 How It Works

1. PDF Upload
- Reads PDFs using LlamaIndex SimpleDirectoryReader.

2. Embedding + Indexing
- Embeddings via HuggingFace MiniLM
- Stored in FAISS index
- Persistent storage in storage/

3. Self-RAG Workflow
- Retriever fetches relevant chunks
- Generator answers using PDF context + chat memory
- Critic evaluates the answer and may request more retrieval

4. Conversation Memory
- Keeps last 10 messages for context-aware responses.

---
## 📘 Tech Stack

- Python
- Streamlit
- LlamaIndex
- FAISS
- LangGraph
- Google Gemini API
- Sentence Transformers

---
## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs.

---
## 🛡 License
This project is licensed under the MIT License.

---
## ⭐ Support
If this project helped you, give it a star ⭐ on GitHub!
