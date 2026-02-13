from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

def build_rag_chain(retriever):
    llm = Ollama(model="deepseek-r1:1.5b")

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
