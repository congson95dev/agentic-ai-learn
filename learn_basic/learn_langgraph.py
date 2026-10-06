import os
from dotenv import load_dotenv, find_dotenv
import requests
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from rich import print

_ = load_dotenv(find_dotenv())
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo")


# ================================================================================================
# Tạo React Agent có tool search web và tool search dữ liệu thời tiết
# ================================================================================================


# Dùng Tavily để tạo search tool bằng langchain
# Tavily API Key Free lấy ở trang https://app.tavily.com/playground
search_tool = TavilySearchResults(max_results=3) # Search và trả về tối đa 3 kết quả
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

# Import tools to tool list
tools = [search_tool, get_weather_data]

# Generate memory
memory = MemorySaver()

# Tạo Agent có chứa tools và memory
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "001"}} # thread_id để nhận biết đc cuộc trò truyện này thuộc thread nào

# response = agent_executor.invoke({"messages": [HumanMessage(content="Find the capital of Vietnam and then find its current weather.")]}, config)
# print(response)  # Output: {'output': 'The capital of Vietnam is Hanoi. The current weather in Hanoi is as follows:\n- Temperature: 38°C\n- Weather: Dust storm\n- Humidity: 20%'}


# ================================================================================================
# Langgraph example đơn giản về State, Node, Edge
# ================================================================================================



from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
import operator

# Define state
class TemperatureState(TypedDict):
    temp_celsius: float
    temp_fahrenheit: float
    weather_status: str
    weather_status_translated: Annotated[list, operator.add]

def convert_temp(state: TemperatureState) -> TemperatureState:
    celsius = state['temp_celsius']
    fahrenheit = (celsius * 9/5) + 32 # Convert Temperature Celsius to Fahrenheit
    state['temp_fahrenheit'] = round(fahrenheit, 2)

    return state # Sau mỗi node đều phải trả nguyên state, trừ khi chạy parallel

def label_weather(state: TemperatureState) -> TemperatureState:
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
        
    return state # Sau mỗi node đều phải trả nguyên state, trừ khi chạy parallel

def translate_to_vn(state: TemperatureState) -> TemperatureState:
    weather_status = state['weather_status']

    # translate the weather status to Vietnamese
    prompt = f'Translate to Vietnamese the following weather status - {weather_status}'
    weather_status_translated = llm.invoke(prompt).content

    state['weather_status_translated'] = weather_status_translated

    return {
        'weather_status_translated': [weather_status_translated] # Vì đang chạy parallel, nên trả dict của value được update thôi
    }

def translate_to_fr(state: TemperatureState) -> TemperatureState:
    weather_status = state['weather_status']

    # translate the weather status to French
    prompt = f'Translate to French the following weather status - {weather_status}'
    weather_status_translated = llm.invoke(prompt).content

    state['weather_status_translated'] = weather_status_translated

    return {
        'weather_status_translated': [weather_status_translated] # Vì đang chạy parallel, nên trả dict của value được update thôi
    }

# Define the graph
graph = StateGraph(TemperatureState)

# Add nodes to the graph
graph.add_node('convert_temp',convert_temp)
graph.add_node('label_weather', label_weather)
graph.add_node('translate_to_vn', translate_to_vn)
graph.add_node('translate_to_fr', translate_to_fr)

# Add edges to the graph
graph.add_edge(START, 'convert_temp')
graph.add_edge('convert_temp', 'label_weather')
# Chạy song song dịch tiếng Việt và tiếng Đức
graph.add_edge('label_weather', 'translate_to_vn')
graph.add_edge('label_weather', 'translate_to_fr')
graph.add_edge('translate_to_vn', END)
graph.add_edge('translate_to_fr', END)

# Compile the graph
workflow = graph.compile()

# Execute the graph
initial_state = {'temp_celsius': 28.5}
final_state = workflow.invoke(initial_state)
print(final_state) # output: {'temp_celsius': 28.5, 'temp_fahrenheit': 83.3, 'weather_status': 'Hot', 'weather_status_translated': ['Nắng nóng', 'Chaud']}

# Visualize the graph
# png_data = workflow.get_graph().draw_mermaid_png()
# with open("learn_basic/temperature_workflow.png", "wb") as f:
#     f.write(png_data)


