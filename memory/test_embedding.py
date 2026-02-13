# from langchain_community.embeddings import OllamaEmbeddings

# embeddings = OllamaEmbeddings(
#     model="all-minilm:33m",
#     base_url="http://localhost:11434"
# )

# vector = embeddings.embed_query(
#     "Agentic AI system using LangChain and Ollama"
# )

# print("Vector length:", len(vector))
# print("First 5 values:", vector[:5])

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

texts = [
    "Karpenter automatically provisions nodes in EKS",
    "LangChain helps build agentic AI systems",
    "Redis is an in-memory data store",
    "Qwen-VL supports vision-language tasks"
]

embeddings = OllamaEmbeddings(
    model="all-minilm:33m",
    base_url="http://localhost:11434"
)

db = FAISS.from_texts(texts, embeddings)

query = "How do I build AI agents?"
results = db.similarity_search(query, k=2)

for r in results:
    print(r.page_content)
