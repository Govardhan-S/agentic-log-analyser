from langchain_community.llms import Ollama
import base64
import os

class MultimodalRAG:
    def __init__(self):
        # Try different Qwen vision models in order of preference
        models_to_try = ["qwen3-vl:4b"]
        
        self.vision_model = None
        for model in models_to_try:
            try:
                self.vision_model = Ollama(model=model)
                # Test the model with a simple call
                self.vision_model.invoke("test")
                print(f"Using vision model: {model}")
                break
            except Exception as e:
                print(f"Model {model} not available: {e}")
                continue
        
        if not self.vision_model:
            raise Exception("No vision model available. Please pull a vision model with: ollama pull qwen2-vl:4b")
        
        self.reasoning_model = Ollama(model="deepseek-r1:1.5b")
    
    def analyze_screenshot(self, image_path, query_type="general"):
        """Analyze screenshot using Qwen vision model"""
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
        
        if query_type == "kubernetes_error":
            prompt = """Analyze this Kubernetes error screenshot carefully. Extract and describe:
1. The exact error message(s) displayed
2. The Kubernetes resource type (pod, deployment, service, etc.)
3. The resource name and namespace if visible
4. Any status codes or error types
5. Timestamps if present
6. Any stack traces or detailed error information
7. Command that was executed (if visible)

Provide a clear, structured description of what error occurred."""
        else:
            prompt = "Describe what you see in this screenshot in detail. Focus on text, UI elements, and any important information."
        
        response = self.vision_model.invoke(
            prompt,
            images=[image_data]
        )
        return response
    
    def reason_with_context(self, screenshot_analysis, user_query):
        """Use DeepSeek for reasoning based on screenshot analysis"""
        context_prompt = f"""Kubernetes Error Analysis: {screenshot_analysis}

User Question: {user_query}

Based on the Kubernetes error analysis above:
1. Explain what went wrong
2. Identify the root cause
3. Provide specific troubleshooting steps
4. Suggest kubectl commands to investigate further
5. Recommend potential fixes

Be specific and actionable in your response."""
        
        return self.reasoning_model.invoke(context_prompt)
    
    def process_query(self, image_path, user_query):
        """Complete RAG pipeline: analyze image then reason"""
        # Detect if this is a Kubernetes error query
        query_type = "kubernetes_error" if any(word in user_query.lower() for word in ['kubernetes', 'k8s', 'error', 'kubectl', 'pod', 'deployment']) else "general"
        
        # Step 1: Analyze screenshot with Qwen
        screenshot_analysis = self.analyze_screenshot(image_path, query_type)
        
        # Step 2: Reason with DeepSeek
        final_answer = self.reason_with_context(screenshot_analysis, user_query)
        
        return {
            "screenshot_analysis": screenshot_analysis,
            "final_answer": final_answer
        }