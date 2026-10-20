# Chatbot Agentic phong cách ChatGPT

Project này là phiên bản giao diện web thay cho ứng dụng Streamlit trong `../langgraph-chatbot`. 
Những thay đổi khác:
chuyển từ single agent thành multi agent
sử dụng MCP

Techstack: **FastAPI**, **Jinja2**, **LangGraph**, **LangChain**, **MCP**

Chatbot có thể tự quyết định khi nào cần trả lời trực tiếp và khi nào cần gọi tool để:

- tìm kiếm web qua Tavily Remote MCP;
- tra cứu dữ liệu chuyến bay qua Aviationstack MCP;
- tính toán;
- xem giá cổ phiếu, thời tiết hiện tại và dự báo năm ngày;
- hỏi đáp trên các tệp người dùng tải lên (RAG);
- lưu và truy xuất bộ nhớ dài hạn;
- yêu cầu người dùng phê duyệt trước khi mô phỏng mua cổ phiếu.

Giao diện cũng hỗ trợ tạo và quản lý nhiều cuộc hội thoại, stream câu trả lời theo thời gian thực, tải tệp và nhập bằng giọng nói trên trình duyệt.

## Layers

| Layer | File / thư mục chính | Trách nhiệm |
| --- | --- | --- |
| Presentation | `main.py`, `templates/index.html`, `static/` | Cung cấp giao diện chat, API FastAPI và stream phản hồi đến trình duyệt qua Server-Sent Events. |
| Orchestration | `agent.py`, `mcp_client.py` | Định nghĩa đồ thị LangGraph đa agent, điều phối model, tool, kết nối MCP và checkpoint cho từng cuộc hội thoại. |
| Tools | `tools.py`, `custom_mcp.py` | `tools.py` chứa các tool cục bộ; `custom_mcp.py` là local MCP server cho dữ liệu thời tiết. |
| RAG | `rag.py`, `uploads/`, `vector_store/` | Đọc tệp tải lên, chia nhỏ nội dung, tạo embeddings và truy xuất bằng FAISS cục bộ. |
| Data & memory | `database.py`, `databases/` | Lưu cuộc hội thoại, tin nhắn và bộ nhớ dài hạn trong SQLite; lưu checkpoint LangGraph trong một SQLite riêng. |

## Available tools

| Tool | Khi nào sử dụng | Nguồn / hoạt động |
| --- | --- | --- |
| Tavily MCP tools | Thông tin hiện tại hoặc cần tìm trên web | Tool được tự khám phá từ Tavily Remote MCP (`tavily_search`, `tavily_extract`, …). |
| Aviationstack MCP tools | Chuyến bay, hãng bay, sân bay, tuyến bay hoặc loại máy bay | Tool được tự khám phá từ Aviationstack MCP, ví dụ `get_flight_status`, `flights_with_airline`, `list_airports` và `list_airlines`. |
| `calculator` | Biểu thức toán học | Tính biểu thức với các hàm được giới hạn như `math.sqrt`, `round`, `min`, `max` và `sum`. |
| `get_stock_price` | Giá cổ phiếu hiện tại, ví dụ `AAPL` hoặc `TSLA` | Gọi Alpha Vantage qua `ALPHAVANTAGE_API_KEY`. |
| Local Weather MCP tools | Thời tiết hiện tại hoặc dự báo cho một thành phố/địa điểm | `get_current_weather` và `get_weather_forecast` được tự khám phá từ `custom_mcp.py`, chạy local qua stdio và dùng `OPENWEATHER_API_KEY`. Forecast của OpenWeather gồm các mốc ba giờ trong năm ngày. |
| `rag_tool` | Câu hỏi về tệp đã tải lên | Truy xuất 4 đoạn nội dung tương đồng nhất từ FAISS index cục bộ. |
| `save_memory` | Người dùng yêu cầu chatbot ghi nhớ hoặc cần dùng thông tin đã nhớ | Lưu hoặc tìm tối đa 8 bộ nhớ phù hợp trong SQLite. |
| `purchase_stock` | Yêu cầu mua cổ phiếu | Mô phỏng lệnh mua và tạm dừng graph để chờ phê duyệt. Chỉ `yes` mới hoàn tất lệnh mô phỏng; các lựa chọn khác sẽ hủy lệnh. |

## Multi-agent

`agent.py` dùng mô hình **router + specialist agents (small agents) + final agent**. Router chọn một hoặc nhiều specialist agents phù hợp; các specialist agents được chọn chạy song song nếu chúng không phụ thuộc vào nhau. Sau đó `final_agent` tổng hợp kết quả thành một câu trả lời duy nhất. Chỉ gọi những specialist agents được router chọn.

Các specialist agents gồm web research, weather, aviation, data/calculator, RAG documents, memory, simulated stock transaction và general chat.

## Tệp được hỗ trợ cho RAG

Người dùng có thể tải lên các tệp sau để chatbot đọc và trả lời câu hỏi: `.pdf`, `.docx`, `.txt`, `.md`, `.py` và `.csv`.

Nội dung được tách thành các đoạn, tạo embeddings bằng OpenAI và lưu vào FAISS cục bộ. Các tệp đã tải lên dùng chung một vector store; vì vậy nội dung của những tệp trước đó vẫn có thể được truy xuất trong các cuộc hội thoại sau, cho đến khi bạn xóa dữ liệu cục bộ.

## Setup

Cài đặt dependencies từ thư mục gốc của repository:

```bash
pip install -r requirements.txt
```

Tạo tệp `.env` tại thư mục gốc của repository:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

`OPENAI_API_KEY` là bắt buộc để chạy chat model và tạo embeddings cho RAG. Những key còn lại chỉ cần thiết khi sử dụng tool tương ứng.

## MCP:

`mcp_client.py` là file MCP Client để kết nối tới các MCP Server.

- **Tavily** là Remote MCP dùng Streamable HTTP, dùng để internet search.
- **Aviationstack** là Local MCP được khởi chạy cục bộ bằng `uvx --python 3.13 aviationstack-mcp`, dùng để lấy thông tin chuyến bay qua API của Aviationstack.
- **Local Weather** là custom Local MCP Server trong `custom_mcp.py`. Gồm có 2 tool sau bên trong: `get_current_weather` và `get_weather_forecast`.


# Setup:

```bash
pip install -r requirements.txt
```

## How to run

Từ thư mục gốc của repository:

```bash
cd chatgpt-clone
uvicorn main:app --reload
```

Sau đó mở [http://127.0.0.1:8000](http://127.0.0.1:8000) trên trình duyệt.

## Lưu trữ dữ liệu

| Đường dẫn | Nội dung |
| --- | --- |
| `uploads/` | Bản gốc của các tệp người dùng tải lên. |
| `vector_store/` | FAISS index cục bộ được tạo từ nội dung tệp. |
| `databases/chatbot.sqlite3` | Dữ liệu SQLAlchemy: cuộc hội thoại, tin nhắn và bộ nhớ dài hạn. |
| `databases/langgraph_checkpoints.sqlite3` | Checkpoint LangGraph theo từng cuộc hội thoại; cần thiết để tiếp tục luồng agent và quy trình phê duyệt. |

Khi người dùng yêu cầu chatbot ghi nhớ một thông tin hoặc sở thích, agent sẽ gọi `save_memory(action="save")`. Khi một thông tin đã lưu có thể giúp trả lời câu hỏi, agent gọi `save_memory(action="retrieve")` để tìm lại thông tin phù hợp.

## Nhập bằng giọng nói

Tính năng nhập giọng nói dùng Web Speech API của trình duyệt (`SpeechRecognition` hoặc `webkitSpeechRecognition`). Khả năng sử dụng phụ thuộc vào trình duyệt và quyền microphone. Âm thanh không được tải lên ứng dụng này; trình duyệt chỉ chuyển lời nói thành văn bản trước khi gửi tin nhắn.
