import streamlit as st
from agent.agent import Agent
from chains.multimodal_rag import MultimodalRAG
import tempfile
import os

st.set_page_config(page_title="Multimodal RAG", page_icon="🖼️")

@st.cache_resource
def load_agent():
    return Agent()

@st.cache_resource
def load_multimodal_rag():
    return MultimodalRAG()

def main():
    st.title("🖼️ Multimodal RAG Assistant")
    st.caption("Upload screenshots and ask questions - Qwen3 for vision, DeepSeek for reasoning")
    
    # Initialize
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.agent = load_agent()
            st.session_state.multimodal_rag = load_multimodal_rag()
    
    # Sidebar for image upload
    with st.sidebar:
        st.header("Upload Screenshot")
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Screenshot", use_column_width=True)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.temp_image_path = tmp_file.name
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask about the screenshot or anything else..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # Check if there's an uploaded image and the query seems image-related
                if hasattr(st.session_state, 'temp_image_path') and any(word in prompt.lower() for word in ['screenshot', 'image', 'see', 'show', 'what', 'describe']):
                    result = st.session_state.multimodal_rag.process_query(st.session_state.temp_image_path, prompt)
                    response = f"**Vision Analysis (Qwen):**\\n{result['screenshot_analysis']}\\n\\n**Reasoning (DeepSeek):**\\n{result['final_answer']}"
                else:
                    response = st.session_state.agent.run(prompt)
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()