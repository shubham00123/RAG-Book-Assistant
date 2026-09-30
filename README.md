# 📚 RAG Book Assistant

A practical **Retrieval-Augmented Generation (RAG)** application that allows users to **upload a PDF book, create a vector database from its content, and ask questions based on the document**.

The application uses **Hugging Face embeddings** to convert text into vectors, **ChromaDB** to store and retrieve relevant information, and **Groq LLM** to generate answers using the retrieved context.

---

## 🚀 What I Built

I built a **RAG Book Assistant** where a user can:

1. Upload a PDF book through a Streamlit interface.
2. Extract text from the PDF.
3. Split the document into smaller chunks.
4. Convert the chunks into vector embeddings.
5. Store the embeddings in ChromaDB.
6. Retrieve relevant chunks when a question is asked.
7. Send the retrieved context to an LLM.
8. Generate an answer based only on the document.

The goal was to understand how a basic **RAG pipeline works from document upload to final AI answer**.

---

## 🧠 What I Learned

While building this project, I learned:

* How **RAG (Retrieval-Augmented Generation)** works
* How to load PDF documents using `PyPDFLoader`
* Why documents are divided into smaller **chunks**
* How **text embeddings** convert text into vectors
* How Hugging Face embedding models can be used with RAG
* How **ChromaDB** works as a vector database
* How a **retriever** finds relevant information
* How **MMR (Maximal Marginal Relevance)** retrieval works
* How an LLM uses retrieved context to generate an answer
* How to create RAG prompts using LangChain
* How to build a simple AI interface using **Streamlit**
* How to use `.env` for API keys
* How to handle uploaded files using temporary files

---

## 🧩 Problems I Faced & How I Solved Them

### 1. Large PDF Processing

Processing a complete PDF directly can make retrieval inefficient.

**Solution:**
I used `RecursiveCharacterTextSplitter` to divide the document into smaller chunks with overlap.

```text
Chunk Size: 1000
Chunk Overlap: 200
```

This helped create smaller and more useful pieces of information for retrieval.

### 2. Finding Relevant Information

A simple similarity search can sometimes return similar information repeatedly.

**Solution:**
I used **MMR retrieval** with:

```text
k = 4
fetch_k = 10
lambda_mult = 0.5
```

This helps retrieve relevant information while maintaining some diversity between the retrieved chunks.

### 3. AI Hallucination

The LLM could potentially answer using information that was not present in the uploaded document.

**Solution:**
I designed the prompt to instruct the model to use **only the retrieved context** and return:

> "I could not find the answer in the document."

when the required information is not available.

### 4. API Key Security

The application requires an API key, but storing it directly inside the code is unsafe.

**Solution:**
I stored the API key in `.env` and added `.env` to `.gitignore` so it is not uploaded to GitHub.

---

## 📁 Project Structure

```text
RAG/
│
├── document_loaders/
│   ├── deep learning.pdf
│   ├── notes.txt
│   ├── page.py
│   ├── pdf.py
│   ├── report.pdf
│   └── test.py
│
├── vector store/
│   └── db.py
│
├── app.py
├── create_databae.py
├── main.py
├── requirement.txt
├── .gitignore
└── README.md
```

> `chroma_db/`, `.env`, and `.venv/` are excluded from GitHub using `.gitignore`.

---

## 🛠️ Tech Stack

**Python · LangChain · Hugging Face · ChromaDB · Groq · Streamlit · PyPDF · RAG**

---

## 🔄 RAG Pipeline

```text
PDF Upload
    ↓
PDF Loader
    ↓
Text Splitting
    ↓
Hugging Face Embeddings
    ↓
ChromaDB
    ↓
User Question
    ↓
Retriever + MMR
    ↓
Relevant Context
    ↓
Groq LLM
    ↓
AI Answer
```

---

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirement.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🎯 Current Focus

This project helped me build a practical understanding of **RAG, embeddings, vector databases, information retrieval, LLM integration, and AI application development**.

I am currently continuing to build deeper skills in **Generative AI, RAG, AI Agents, AI Engineering, and Full-Stack Development**.
