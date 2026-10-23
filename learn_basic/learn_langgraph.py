# ================================================================================================
# Tạo React Agent có tool search web và tool search dữ liệu thời tiết
# Ngoài ra, sử dụng thêm Built-in Middleware (PIIMiddleware, HumanInTheLoopMiddleware) của Langchain và Custom Middleware để validate input
# ================================================================================================

import os
from dotenv import load_dotenv, find_dotenv
import requests
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware, AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from rich import print

_ = load_dotenv(find_dotenv())
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Dùng Tavily để tạo search tool bằng langchain
# Tavily API Key Free lấy ở trang https://app.tavily.com/playground
search_tool = TavilySearch(max_results=3) # Search và trả về tối đa 3 kết quả
# response = search_tool.invoke("Who are the top stars of the 2024 Eurocup?")  # trong TH hỏi những câu hỏi vượt quá dữ liệu lưu trữ của OpenAI model, thì nó mới gọi tool search, còn nếu câu hỏi nằm trong knowledge base thì nó không search mà trả luôn kq.
# print(response)

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    print("Trigger send_email function")
    return f"Email sent to {to} with subject: {subject}"

# Import tools to tool list
tools = [search_tool, get_weather_data, send_email]

# Generate memory
memory = MemorySaver()

# Custom Middleware
class ContentFilterMiddleware(AgentMiddleware):
    """
    Deterministic guardrail: Block requests containing banned keywords.
    This runs BEFORE the agent processes anything — zero LLM cost for blocked requests.
    """

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    # Use before_agent() to validate requests before any LLM processing begins.
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                print(f"🚫 Blocked — keyword detected: '{keyword}'")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I cannot process requests containing inappropriate content. "
                            "Please rephrase your request."
                        )
                    }],
                    "jump_to": "end"
                }
        return None

# Tạo Agent có chứa tools và memory
agent_executor = create_agent(
    llm, tools, checkpointer=memory,
    middleware=[
        PIIMiddleware(
            "email",
            strategy="redact", # Redact emails in user input before sending to model, it will display the email as "[REDACTED_EMAIL]"
            apply_to_input=True,
        ),
        PIIMiddleware(
            "credit_card",
            strategy="mask", # Mask credit cards in user input, it will display the credit card as "****-****-****-5100"
            apply_to_input=True,
        ),
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block", # Block API keys - raise error if detected
            apply_to_input=True,
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True, # Require approval if user want to send email
            }
        ),
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "malware", "jailbreak", "bypass"] # Custom Middleware, block if found banned keywords
        ),
    ],
)

config = {"configurable": {"thread_id": "001"}} # thread_id để nhận biết đc cuộc trò truyện này thuộc thread nào

# # Normal case, output the result from LLM
# response = agent_executor.invoke({"messages": [HumanMessage(content="Find the capital of Vietnam and then find its current weather.")]}, config)
# print(response["messages"][-1].content)  # Output: {'output': 'The capital of Vietnam is Hanoi. The current weather in Hanoi is as follows:\n- Temperature: 38°C\n- Weather: Dust storm\n- Humidity: 20%'}

# # Demo PIIMiddleware, the input contain email and credit card information, when the LLM receive the input, it automatically mask the PII data.
# response = agent_executor.invoke({"messages": [HumanMessage(content="My email is john.doe@example.com and my card is 5105-1051-0510-5100. Can you help me?")]}, config)
# print(response["messages"][0].content) # Output: My email is [REDACTED_EMAIL] and my card is ****-****-****-5100. Can you help me?

# # Demo PIIMiddleware, the input contain api key information, when the LLM receive the input, it return error immediately.
# try:
#     response = agent_executor.invoke({"messages": [HumanMessage(content="My key is sk-abcdefghijklmnopqrstuvwxyz123456. Can you help me?")]}, config)
#     print(response["messages"][0].content)
# except Exception as e:
#     print(f"🚫 Blocked as expected: {e}") # Output: 🚫 Blocked as expected: Detected 1 instance(s) of api_key in text content

# # Demo HumanInTheLoopMiddleware, the input contain request to send an email, when the LLM receive the input, it stop and ask for approval.
# response = agent_executor.invoke({"messages": [HumanMessage(content="Send an email to team@company.com about the Q4 results")]}, config)
# # print(response["messages"][0].content) # Output: Send an email to [REDACTED_EMAIL] about the Q4 results
# print(response) # Check the output, you will see it waiting for approval

# # Approved
# response = agent_executor.invoke(
#     Command(resume={"decisions": [{"type": "approve"}]}),
#     config=config
# )
# print(response["messages"][-1].content) # Output: I have sent an email to the specified recipient regarding the Q4 results.

# # Demo Custom Middleware, the input contain banned keywords, when the LLM receive the input, it return error immediately.
response = agent_executor.invoke({"messages": [{"role": "user", "content": "How do I hack into a server?"}]}, config)
print("🚫 Unsafe request response:")
print(response["messages"][-1].content) # Output: I cannot process requests containing inappropriate content. Please rephrase your request.




# ================================================================================================
# Langgraph workflow example về State, Node, Edge, Parallel Execution, Conditional Branching, Iterative (Retry)
# Flow detail check ở temperature_workflow.png
# ================================================================================================


import operator
from rich import print

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define structured output schema
class PostAnalysisSchema(BaseModel):
    content_type: Literal[
        "educational",
        "promotional",
        "showcase",
        "testimonial",
        "announcement",
        "engagement"
    ] = Field(description="The primary type of content in the post.")
    tone: Literal[
        "professional",
        "friendly",
        "casual",
        "inspiring",
        "persuasive",
        "informative"
    ] = Field(description="The overall tone of the post.")
    quality: Literal[
        "poor",
        "acceptable",
        "good",
        "excellent"
    ] = Field(description="Overall quality of the generated post.")
    engagement_potential: Literal[
        "low",
        "medium",
        "high"
    ] = Field(description="Estimated potential for audience engagement.")

# Define state
class TemperatureState(TypedDict):
    temp_celsius: float
    temp_fahrenheit: float
    weather_status: str
    weather_status_vn: str
    weather_status_fr: str
    post_content: str
    diagnosis: dict
    review_result: str
    feedback: str
    iteration: int
    max_iteration: int
    post_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

# Generate a structured output model
post_analysis_structured_model = llm.with_structured_output(PostAnalysisSchema)

def convert_temp(state: TemperatureState):
    celsius = state['temp_celsius']
    fahrenheit = (celsius * 9/5) + 32 # Convert Temperature Celsius to Fahrenheit
    state['temp_fahrenheit'] = round(fahrenheit, 2)

    return state

def label_weather(state: TemperatureState):
    fahrenheit = state['temp_fahrenheit']

    # Mark the weather status based on the Fahrenheit temperature
    if fahrenheit < 50:
        state["weather_status"] = "Cold"
    elif 50 <= fahrenheit < 77:
        state["weather_status"] = "Mild"
    elif 77 <= fahrenheit < 95:
        state["weather_status"] = "Hot"
    else:
        state["weather_status"] = "Extreme Heat"
        
    return state

def translate_to_vn(state: TemperatureState):
    weather_status = state['weather_status']

    # translate the weather status to Vietnamese
    prompt = f'Translate to Vietnamese the following weather status - {weather_status}'
    weather_status_vn = llm.invoke(prompt).content

    return {
        'weather_status_vn': weather_status_vn
    }

def translate_to_fr(state: TemperatureState):
    weather_status = state['weather_status']

    # translate the weather status to French
    prompt = f'Translate to French the following weather status - {weather_status}'
    weather_status_fr = llm.invoke(prompt).content

    return {
        'weather_status_fr': weather_status_fr
    }

def aggregate_translated_text(state: TemperatureState) -> TemperatureState:
    return state 

def create_post(state: TemperatureState) -> TemperatureState:
    prompt = [
        SystemMessage(content="You are a funny and clever Facebook influencer."),
        HumanMessage(content=f"""Create a post content around 100 words that describe about weather \n {state["weather_status"]}""")
    ]
    post_content = llm.invoke(prompt).content

    return {'post_content': post_content, 'post_history': [post_content]}

def analyze_content(state: TemperatureState):
    prompt = f"""Diagnose this post content :\n\n{state['post_content']}\n"""
    response = post_analysis_structured_model.invoke(prompt)

    return {
        'diagnosis': response.model_dump() # Return the structured output as a dictionary
    }

def check_condition(state: TemperatureState):
    if state['diagnosis']['quality'] != 'poor' or state['iteration'] >= state['max_iteration']:
        return "approve_post"
    else:
        return "reject_post"

def approve_post(state: TemperatureState):
    result = 'Post approved.'
    return {
        'review_result': result
    }

def reject_post(state: TemperatureState):
    result = 'Post rejected.'
    return {
        'review_result': result
    }

def create_feedback(state: TemperatureState):
    prompt = [
        SystemMessage(content="You are a ruthless, no-laugh-given Facebook critic. You punch up Facebook posts for virality and humor."),
        HumanMessage(
            content=f"""
                Create feedback to improve the following Facebook post:

                Post: "{state['post_content']}"

                ### Respond ONLY in structured format:
                - feedback: One paragraph explaining the strengths and weaknesses
            """
        )
    ]
    feedback = llm.invoke(prompt).feedback

    return {'feedback': feedback, 'feedback_history': [feedback]}

def optimize_post(state: TemperatureState):
    prompt = [
        SystemMessage(content="You punch up Facebook posts for virality and humor based on given feedback."),
        HumanMessage(content=f"""
            Improve the Facebook post based on this feedback:
            "{state['feedback']}"

            Weather: "{state['weather_status']}"
            Original Post:
            {state['post_content']}

            Re-write it as a short, viral-worthy Facebook post. Avoid Q&A style and stay under 500 characters.
            """)
        ]

    response = llm.invoke(prompt).content

    iteration = state['iteration'] + 1

    return {'post_content': response, 'iteration': iteration, 'post_history': [response]}

# Define the graph
graph = StateGraph(TemperatureState)

# Add nodes to the graph
graph.add_node('convert_temp',convert_temp)
graph.add_node('label_weather', label_weather)
graph.add_node('translate_to_vn', translate_to_vn)
graph.add_node('translate_to_fr', translate_to_fr)
graph.add_node('aggregate_translated_text', aggregate_translated_text)
graph.add_node('create_post', create_post)
graph.add_node('analyze_content', analyze_content)
graph.add_node('approve_post', approve_post)
graph.add_node('reject_post', reject_post)
graph.add_node('create_feedback', create_feedback)
graph.add_node('optimize_post', optimize_post)


# Workflow:

# Add edges to the graph
graph.add_edge(START, 'convert_temp')
graph.add_edge('convert_temp', 'label_weather')

# Chạy song song dịch tiếng Việt và tiếng Đức
graph.add_edge('label_weather', 'translate_to_vn')
graph.add_edge('label_weather', 'translate_to_fr')
graph.add_edge('translate_to_vn', 'aggregate_translated_text')
graph.add_edge('translate_to_fr', 'aggregate_translated_text')

# Create post and analyze post content
graph.add_edge('aggregate_translated_text', 'create_post')
graph.add_edge('create_post', 'analyze_content')

# Conditional branching
graph.add_conditional_edges('analyze_content', check_condition, {
    'approve_post': 'approve_post',
    'reject_post': 'reject_post',
})

# If approved, then end the workflow
graph.add_edge('approve_post', END)

# If rejected, create feedback and optimize post, then analyze content again, keep doing this until the post is approved or retried 3 times
graph.add_edge('reject_post', 'create_feedback')
graph.add_edge('create_feedback', 'optimize_post')
graph.add_edge('optimize_post', 'analyze_content')

# Compile the graph
workflow = graph.compile()

# Init state
initial_state = {'temp_celsius': 28.5, "iteration": 1, "max_iteration": 3}

# Execute the graph
# final_state = workflow.invoke(initial_state)
# print(final_state)

# Visualize the graph
# png_data = workflow.get_graph().draw_mermaid_png()
# with open("learn_basic/temperature_workflow.png", "wb") as f:
#     f.write(png_data)





# ================================================================================================
# Demo use-case của checkpointer trong langgraph về fault-tolerance (retry khi có lỗi xảy ra)
# Ở step 2, ấn Ctrl C để giả lập bị lỗi, lúc này nó tự chạy, nhưng thay vì chạy lại từ đầu, nó chạy từ step 2.
# ================================================================================================

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
import time
from rich import print


# Define the state
class CrashState(TypedDict):
    input: str
    step1: str
    step2: str
    step3: str

# Define steps
def step_1(state: CrashState) -> CrashState:
    print("✅ Step 1 executed")
    return {"step1": "done", "input": state["input"]}

def step_2(state: CrashState) -> CrashState:
    print("⏳ Step 2 hanging... now manually interrupt from the notebook toolbar (STOP button)")
    time.sleep(30)  # Simulate long-running hang
    return {"step2": "done"}

def step_3(state: CrashState) -> CrashState:
    print("✅ Step 3 executed")
    return {"step3": "done"}

# Build the graph
builder = StateGraph(CrashState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.set_entry_point("step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# # Execute the graph
# try:
#     print("▶️ Running graph: Please manually interrupt during Step 2...") # simulate the crash in step 2
#     graph.invoke({"input": "start"}, config={"configurable": {"thread_id": 'thread-1'}})
# except KeyboardInterrupt:
#     print("❌ Kernel manually interrupted (crash simulated).")

# # Re-run the graph
# print("\n🔁 Re-running the graph to demonstrate fault tolerance...")
# final_state = graph.invoke(None, config={"configurable": {"thread_id": 'thread-1'}})
# print("\n✅ Final State:", final_state)

# print(list(graph.get_state_history({"configurable": {"thread_id": 'thread-1'}}))) # At this point, you should see it running from step 2 instead of start over from step 1




# ================================================================================================
# Demo subgraph trong langgraph
# Có 2 cách gọi subgraph trong langgraph
# 1. Khai báo và gọi trực tiếp bên trong function bằng invoke()
# VD: Graph translate to vietnamese dùng cách này.
# 2. Khái báo graph thành 1 node, sau đó gọi vào node đó
# VD: Graph translate to france dùng cách này.
# ================================================================================================

from rich import print

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
subgraph_llm = ChatOpenAI(model="gpt-3.5-turbo")
parent_llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define Sub Graph 1 (translate text to Vietnamese)
class SubState1(TypedDict):
    question: str
    translated_text_vn: str

def translate_text_to_vn(state: SubState1):
    prompt = f"""
        Translate the following text to Vietnamese.
        Keep it natural and clear. Do not add extra content.

        Text:
        {state["question"]}
        """.strip()
    
    translated_text_vn = subgraph_llm.invoke(prompt).content

    return {'translated_text_vn': translated_text_vn}

subgraph1_builder = StateGraph(SubState1)

subgraph1_builder.add_node('translate_text_to_vn', translate_text_to_vn)

subgraph1_builder.add_edge(START, 'translate_text_to_vn')
subgraph1_builder.add_edge('translate_text_to_vn', END)

subgraph1 = subgraph1_builder.compile()

# Define Sub Graph 2 (translate text to France)
class SubState2(TypedDict):
    answer_eng: str
    answer_fr: str

def translate_text_to_fr(state: SubState2):
    prompt = f"""
        Translate the following text to France.
        Keep it natural and clear. Do not add extra content.

        Text:
        {state["answer_eng"]}
        """.strip()
    
    answer_fr = subgraph_llm.invoke(prompt).content

    return {'answer_fr': answer_fr}

subgraph2_builder = StateGraph(SubState2)

subgraph2_builder.add_node('translate_text_to_fr', translate_text_to_fr)

subgraph2_builder.add_edge(START, 'translate_text_to_fr')
subgraph2_builder.add_edge('translate_text_to_fr', END)

subgraph2 = subgraph2_builder.compile()

# Define Parent Graph (This graph will call to 2 Sub Graph above by 2 different way)
class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_vn: str
    answer_fr: str

def generate_answer(state: ParentState):
    answer = parent_llm.invoke(f"You are a helpful assistant. Answer clearly.\n\nQuestion: {state['question']}").content

    return {'answer_eng': answer}

def translate_answer_to_vn(state: ParentState):
    result = subgraph1.invoke({'question': state['answer_eng']}) # Call the Sub Graph 1 directly in function

    return {'answer_vn': result['translated_text_vn']}

parent_builder = StateGraph(ParentState)

parent_builder.add_node("answer", generate_answer)
parent_builder.add_node("translate_answer_to_vn", translate_answer_to_vn)
parent_builder.add_node("translate_answer_to_fr", subgraph2) # Define the Sub Graph 2 as a node

parent_builder.add_edge(START, 'answer')
parent_builder.add_edge('answer', 'translate_answer_to_vn')
parent_builder.add_edge('translate_answer_to_vn', 'translate_answer_to_fr') # Call the Sub Graph 2 as a node
parent_builder.add_edge('translate_answer_to_fr', END)

# graph = parent_builder.compile()

# response = graph.invoke({'question': 'What is yugioh'})
# print(response)