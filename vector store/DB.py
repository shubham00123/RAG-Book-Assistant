
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"}
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"}
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"}
    ),
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vectorstore.similarity_search(
    "what is used for data analysis?",
    k=2
)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)
