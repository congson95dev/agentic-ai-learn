from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
from typing import TypedDict
import time
from rich import print

_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define the state
class PythonTipState(TypedDict):
    input: str
    response: str

# Define steps
def generate_python_tips(state: PythonTipState):
    prompt = [
        HumanMessage(content=f"""Give 5 tips to learn Python programming effectively.""")
    ]
    response = llm.invoke(prompt).content
    state['response'] = response

    return state

# Build the graph
builder = StateGraph(PythonTipState)
builder.add_node("generate_python_tips", generate_python_tips)

builder.add_edge(START, "generate_python_tips")
builder.add_edge("generate_python_tips", END)

workflow = builder.compile()

# Enable tracing in Langsmith
# Sau khi đã bật tracing, mỗi khi invoke, sẽ lưu trực tiếp logs lên https://smith.langchain.com/, có thể vào trang đó để monitoring
CONFIG = {
    "run_name": "chat_trace"
}

# Execute the graph
result = workflow.invoke({"input": ""}, config=CONFIG) # Sau khi invoke, logs đã đc lưu trực tiếp logs lên https://smith.langchain.com/, có thể vào trang đó để monitoring
print(result)