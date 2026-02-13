from chains.perception_chain import perception_chain
from chains.rag_chain import build_rag_chain
from memory.vector_store import build_vectorstore
from agent.agent import build_agent

def main():
    print("🔹 Initializing vector store...")
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever()

    rag_chain = build_rag_chain(retriever)
    agent = build_agent()

    while True:
        user_input = input("\nAsk something (or type 'exit'): ")
        if user_input.lower() == "exit":
            break

        if user_input.startswith("image:"):
            perception = perception_chain.run(user_input)
            response = agent.run(perception)
        else:
            context = rag_chain.run(user_input)
            response = agent.run(context)

        print("\n🤖 Response:\n", response)

if __name__ == "__main__":
    main()
