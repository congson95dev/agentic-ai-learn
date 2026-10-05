import os
from dotenv import load_dotenv, find_dotenv
import requests
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool

_ = load_dotenv(find_dotenv())
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo")

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

# Import search_tool to tools list
tools = [search_tool, get_weather_data]

# Generate memory
memory = MemorySaver()

# Tạo Agent có chứa tools và memory
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "001"}} # thread_id để nhận biết đc cuộc trò truyện này thuộc thread nào

response = agent_executor.invoke({"messages": [HumanMessage(content="Find the capital of India and then find its current weather.")]}, config)
print(response)  # Output: {'output': 'The capital of India is New Delhi. The current weather in New Delhi is as follows:\n- Temperature: 38°C\n- Weather: Dust storm\n- Humidity: 20%'}