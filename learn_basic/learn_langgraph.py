# # ================================================================================================
# # Tạo React Agent có tool search web và tool search dữ liệu thời tiết
# # ================================================================================================

# import os
# from dotenv import load_dotenv, find_dotenv
# import requests
# from langchain_openai import ChatOpenAI
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages import HumanMessage
# from langgraph.checkpoint.memory import MemorySaver
# from langchain.tools import tool
# from rich import print

# _ = load_dotenv(find_dotenv())
# WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
# llm = ChatOpenAI(model="gpt-3.5-turbo")

# # Dùng Tavily để tạo search tool bằng langchain
# # Tavily API Key Free lấy ở trang https://app.tavily.com/playground
# search_tool = TavilySearchResults(max_results=3) # Search và trả về tối đa 3 kết quả
# # response = search_tool.invoke("Who are the top stars of the 2024 Eurocup?")  # trong TH hỏi những câu hỏi vượt quá dữ liệu lưu trữ của OpenAI model, thì nó mới gọi tool search, còn nếu câu hỏi nằm trong knowledge base thì nó không search mà trả luôn kq.
# # print(response)

# @tool
# def get_weather_data(city: str) -> str:
#     """
#     Fetch current weather information for a city.
#     """

#     url = (
#         f"https://api.weatherstack.com/current?"
#         f"access_key={WEATHERSTACK_API_KEY}&query={city}"
#     )

#     response = requests.get(url)

#     data = response.json()

#     if "current" not in data:
#         return f"Could not fetch weather data for {city}"

#     return (
#         f"City: {city}\n"
#         f"Temperature: {data['current']['temperature']}°C\n"
#         f"Weather: {data['current']['weather_descriptions'][0]}\n"
#         f"Humidity: {data['current']['humidity']}%"
#     )

# # Import tools to tool list
# tools = [search_tool, get_weather_data]

# # Generate memory
# memory = MemorySaver()

# # Tạo Agent có chứa tools và memory
# agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# config = {"configurable": {"thread_id": "001"}} # thread_id để nhận biết đc cuộc trò truyện này thuộc thread nào

# response = agent_executor.invoke({"messages": [HumanMessage(content="Find the capital of Vietnam and then find its current weather.")]}, config)
# print(response)  # Output: {'output': 'The capital of Vietnam is Hanoi. The current weather in Hanoi is as follows:\n- Temperature: 38°C\n- Weather: Dust storm\n- Humidity: 20%'}


# ================================================================================================
# Langgraph workflow example về State, Node, Edge, Parallel Execution, Conditional Branching, Iterative (Retry)
# Flow detail check ở temperature_workflow.png
# ================================================================================================


import operator
from rich import print

from langchain.messages import HumanMessage, HumanMessage, SystemMessage
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
final_state = workflow.invoke(initial_state)
print(final_state)

# Visualize the graph
png_data = workflow.get_graph().draw_mermaid_png()
with open("learn_basic/temperature_workflow.png", "wb") as f:
    f.write(png_data)


