import streamlit as st
from chains.multimodal_rag import MultimodalRAG
import tempfile
import os

st.set_page_config(page_title="Kubernetes Error Analyzer", page_icon="🔍")

@st.cache_resource
def load_multimodal_rag():
    return MultimodalRAG()

def main():
    st.title("🔍 Kubernetes Error Analyzer")
    st.caption("Upload error screenshots - Qwen for vision, DeepSeek for troubleshooting")
    
    # Initialize
    if "multimodal_rag" not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.multimodal_rag = load_multimodal_rag()
    
    # Image upload
    uploaded_file = st.file_uploader("Upload Kubernetes Error Screenshot", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        # Display uploaded image
        st.image(uploaded_file, caption="Error Screenshot", use_column_width=True)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_image_path = tmp_file.name
        
        # Query input
        query = st.text_input("Ask about the error:", placeholder="What went wrong in this Kubernetes error?")
        
        if st.button("Analyze Error") and query:
            with st.spinner("Analyzing screenshot..."):
                result = st.session_state.multimodal_rag.process_query(temp_image_path, query)
                
                st.subheader("🔍 Vision Analysis (Qwen)")
                st.write(result['screenshot_analysis'])
                
                st.subheader("🧠 Troubleshooting (DeepSeek)")
                st.write(result['final_answer'])
            
            # Cleanup temp file
            os.unlink(temp_image_path)

if __name__ == "__main__":
    main()