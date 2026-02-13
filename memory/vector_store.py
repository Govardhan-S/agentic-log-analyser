from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def build_vectorstore():
    embeddings = OllamaEmbeddings(
        model="all-minilm:33m",
        base_url="http://localhost:11434"
    )

    vectorstore = FAISS.from_texts(
        ["Hello world"],  # replace with your docs
        embedding=embeddings
    )

    return vectorstore
