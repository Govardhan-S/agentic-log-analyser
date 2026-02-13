from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from agent.tools import shell_tool, screenshot_tool
from chains.multimodal_rag import MultimodalRAG

class Agent:
    def __init__(self):
        self.llm = Ollama(model="deepseek-r1:1.5b")
        self.multimodal_rag = MultimodalRAG()
        self.agent = initialize_agent(
            tools=[shell_tool, screenshot_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def run(self, query):
        return self.agent.run(query)
    
    def analyze_screenshot(self, image_path, query):
        return self.multimodal_rag.process_query(image_path, query)

def build_agent():
    llm = Ollama(model="deepseek-r1:1.5b")
    return initialize_agent(
        tools=[shell_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
