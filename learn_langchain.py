import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

_ = load_dotenv(find_dotenv())


# Set OpenAI API Key
openai_api_key = os.environ["OPENAI_API_KEY"]


# Init Langchain
llmModel = OpenAI()
chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")


# Chat Prompt Template
## English-Spanish translator bot
examples = [
    {"input": "hi!", "output": "¡hola!"},
    {"input": "bye!", "output": "¡adiós!"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Spanish translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# Chain
## With chain, It's run by sequence:
##   1. final_prompt
##   2. chatModel

# chain = final_prompt | chatModel

# res = chain.invoke({"input": "How are you?"})
# print(res.content) # Output: ¿Cómo estás?


# ==============================================================


# Output Parser (Simple)
## Code sau convert output từ LLM sang dạng JSON
json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)

json_parser = SimpleJsonOutputParser()

# json_chain = json_prompt | llmModel | json_parser

# res  = json_chain.invoke({"question": "What is the biggest country?"})
# print(res) # Output: {'answer': 'Russia'}


# ==============================================================


# Output Parser with Pydantic
## Code sau ép output về dạng JSON mà được pre-defined trong Pydantic model
## VD với 2 field "setup" và "punchline" như example sau
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# chain = prompt | chatModel | parser

# res = chain.invoke({"query": "Tell me a joke."})
# print(res) # Output: {'setup': "Why couldn't the bicycle find its way home?", 'punchline': 'Because it lost its bearings!'}


# ==============================================================


# Loader
## langchain có thể load file như txt, pdf, csv, html, website url, wikipedia search thành context để AI có dữ liệu để trả lời
## tuy nhiên nếu dữ liệu quá lớn, sẽ rất tốn token khi làm theo cách thông thường đó
## => cách xử lý: dùng RAG

## Load a text file
# from langchain_community.document_loaders import TextLoader
# loader = TextLoader("sample-data/sample-text.txt")
# loaded_data = loader.load()

## Load a CSV file
# from langchain_community.document_loaders import CSVLoader
# loader = CSVLoader('./data/Street_Tree_List.csv')
# loaded_data = loader.load()

## Load a PDF file
# from langchain_community.document_loaders import PyPDFLoader
# loader = PyPDFLoader('./data/5pages.pdf')
# loaded_data = loader.load_and_split()

## Load a Wikipedia page
# from langchain_community.document_loaders import WikipediaLoader
# loader = WikipediaLoader(query="Tesla", load_max_docs=1)
# loaded_data = loader.load()

# ## Use the loaded data as context in a chat prompt
# chat_template = ChatPromptTemplate.from_messages(
#     [
#         ("human", "Answer this {question}, here is some extra {context}"),
#     ]
# )
# messages = chat_template.format_messages(
#     question="Tell me about Tesla",
#     context=loaded_data
# )


# ==============================================================


# Splitter
## Có 2 loại Splitter: CharacterTextSplitter và RecursiveCharacterTextSplitter
## CharacterTextSplitter chunk theo 1 ký tự cụ thể
## RecursiveCharacterTextSplitter chunk không theo 1 ký tự cụ thể, mà nó tự động tìm điểm chung để chunk, nó linh hoạt hơn CharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
loader = TextLoader("sample-data/sample-long-text.txt")
loaded_data = loader.load()

# Giải thích các params của hàm CharacterTextSplitter

# separator
## Ký tự ưu tiên dùng để chia text
## Lưu ý: không phải cứ gặp "\n\n" là cắt dù chưa đủ 1000 ký tự, mà nó sẽ cố gắng gom thêm trước khi cắt

# chunk_size
## Kích thước tối đa của mỗi chunk
## Lưu ý: nó tính cả các ký tự trống, hoặc xuống dòng

# chunk_overlap
## Số ký tự overlap giữa các chunk
# VD: chunk_size=1000, chunk_overlap=200:
# Chunk 1: [800 ký tự A][200 ký tự B]
# Chunk 2: [200 ký tự B][800 ký tự C]
# Kết quả: 200 ký tự cuối của Chunk 1 được lặp lại ở đầu Chunk 2.
# Mục đích: giúp giữ context và tránh mất thông tin tại điểm chia.

# length_function
## Hàm dùng để tính độ dài text
## Có thể dùng hàm custom

# is_separator_regex
## Xác định separator có phải regex hay không
## Nếu separator là regex thay vì "\n\n" thì set is_separator_regex=True

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([loaded_data[0].page_content])

# print(texts) # output: chunked text
# print(len(texts)) # output: number of chunks, e.g: 1, 2 ..

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = TextLoader("sample-data/sample-text.txt")
loaded_data = loader.load()

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=26,
    chunk_overlap=4
)

texts = recursive_splitter.split_text(loaded_data[0].page_content)

# print(texts) # output: chunked text
# print(len(texts)) # output: number of chunks, e.g: 1, 2 ..


# ==============================================================


# Embeddings and Vector DB (Vector Store)
## Convert text thành dạng embedding sau đó lưu vào Vector DB
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loaded_document = TextLoader('sample-data/sample-text.txt').load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=26,
    chunk_overlap=4
)

chunks_of_text = text_splitter.split_documents(loaded_document)

## Có 2 loại vector DB trong langchain
## Chroma và FAISS (facebook AI Similarity Search)
## Chroma và FAISS đều như nhau nhưng Chroma thường đc sử dụng cho case RAG thực tế, còn FAISS được sử dụng làm quick demo vì độ nhanh và tiện lợi của nó

## Chroma
# from langchain_chroma import Chroma

# vector_db = Chroma.from_documents(chunks_of_text, OpenAIEmbeddings()) # code này tốn token

# question = "What did the president say about the John Lewis Voting Rights Act?"

# response = vector_db.similarity_search(question) # code này tốn token

# print(response[0].page_content) # output: 4 câu trả lời gần nhất

## FAISS
from langchain_community.vectorstores import FAISS

vector_db = FAISS.from_documents(chunks_of_text, OpenAIEmbeddings()) # code này tốn token

retriever = vector_db.as_retriever(search_kwargs={"k": 4}) # k=4 nghĩa là lấy 4 câu trả lời gần nhất, có thể thay đổi k=1,2,3.. tùy nhu cầu

question = "What is dswithbappy teaching?"

response = retriever.invoke(question) # code này tốn token

print(response) # output: 4 câu trả lời gần nhất