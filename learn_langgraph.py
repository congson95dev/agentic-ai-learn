import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Dùng Tavily để tạo search tool bằng langchain
# Tavily API Key Free lấy ở trang https://app.tavily.com/playground
search_tool = TavilySearchResults(max_results=3) # Search và trả về tối đa 3 kết quả
response = search_tool.invoke("Who are the top stars of the 2024 Eurocup?")  # trong TH hỏi những câu hỏi vượt quá dữ liệu lưu trữ của OpenAI model, thì nó mới gọi tool search, còn nếu câu hỏi nằm trong knowledge base thì nó không search mà trả luôn kq.
print(response)

# Import search_tool to tools list
tools = [search_tool]

# Generate memory
memory = MemorySaver()

# Tạo Agent có chứa tools và memory
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "001"}} # thread_id để nhận biết đc cuộc trò truyện này thuộc thread nào

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="Who won the 2024 soccer Eurocup?")]}, config
):
    print(chunk) # output: in ra 3 kq
    print("----")