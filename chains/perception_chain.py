from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

vision_llm = Ollama(model="qwen3-vl:4b")

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    You are a vision-capable assistant.
    Analyze and explain the following image description or screenshot text:

    {input}
    """
)

perception_chain = LLMChain(
    llm=vision_llm,
    prompt=prompt
)
