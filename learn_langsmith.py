import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

load_dotenv()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Set LangChain API Key and enable tracing
# Sau khi đã set API Key và bật tracing, mỗi khi invoke, sẽ lưu trực tiếp logs lên https://smith.langchain.com/, có thể vào trang đó để monitoring
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Init Langchain
llmModel = ChatOpenAI(model="gpt-3.5-turbo-0125")

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)

json_parser = SimpleJsonOutputParser()

json_chain = json_prompt | llmModel | json_parser

res  = json_chain.invoke({"question": "What is the biggest country?"})
print(res) # Output: {'answer': 'Russia'}
# Sau khi invoke, logs đã đc lưu trực tiếp logs lên https://smith.langchain.com/, có thể vào trang đó để monitoring