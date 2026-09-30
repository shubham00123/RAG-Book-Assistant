
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# Chroma vector database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)


print("RAG system created")
print("Press 0 to exit")


while True:

    query = input("You : ")

    if query == "0":
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # Create context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create final prompt
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    # Generate answer
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")

