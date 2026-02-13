import streamlit as st
from agent.agent import Agent
from memory.vector_store import build_vectorstore

st.set_page_config(page_title="Agentic AI Chat", page_icon="🤖")

@st.cache_resource
def load_agent():
    return Agent()

@st.cache_resource
def load_vectorstore():
    return build_vectorstore()

def main():
    st.title("🤖 Agentic AI Assistant")
    
    # Initialize
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        with st.spinner("Loading AI agent..."):
            st.session_state.agent = load_agent()
            st.session_state.vectorstore = load_vectorstore()
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.run(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()