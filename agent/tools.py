import subprocess
from langchain.tools import Tool
from chains.multimodal_rag import MultimodalRAG

def run_shell(command: str):
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout or result.stderr

def analyze_screenshot(query: str):
    """Analyze screenshot and answer questions about it"""
    # Assumes screenshot is saved as 'screenshot.png'
    image_path = "screenshot.png"
    rag = MultimodalRAG()
    result = rag.process_query(image_path, query)
    return f"Analysis: {result['screenshot_analysis']}\n\nAnswer: {result['final_answer']}"

shell_tool = Tool(
    name="ShellCommand",
    func=run_shell,
    description="Run shell commands like kubectl, ls, cat, etc."
)

screenshot_tool = Tool(
    name="ScreenshotAnalyzer",
    func=analyze_screenshot,
    description="Analyze screenshots and answer questions about what's visible in the image"
)
