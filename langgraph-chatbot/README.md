# Agentic Chatbot with LangGraph

Project này là một chatbot agentic sử dụng **LangGraph**, **LangChain** và **Streamlit**. Chatbot có thể tự quyết định khi nào cần trả lời trực tiếp và khi nào cần gọi tool để :
- web search
- math calculate
- PDF reader
- RAG
- HITL

## Layers

| Layer | File chính | Trách nhiệm |
| --- | --- | --- |
| Presentation | `app.py` | Cung cấp giao diện chat Streamlit, stream câu trả lời. |
| Orchestration | `backend.py` | Quản lý flow LangGraph. |
| Tools | `tools.py` | Định nghĩa các tools. |
| Data & memory | `helper.py` | Lưu checkpoint LangGraph trong SQLite, đồng thời nạp và truy xuất nội dung PDF từ FAISS vector store cục bộ. |

## Available tools

| Tool | Khi nào sử dụng | Nguồn / hoạt động |
| --- | --- | --- |
| `search_tool` | Sự kiện hiện tại hoặc thông tin cần tìm trên web | Tavily Search; trả về tối đa 5 kết quả tìm kiếm chuyên sâu. |
| `calculator` | Biểu thức toán học | Tính toán biểu thức bằng tập hàm được giới hạn như `math.sqrt`, `round`, `min` và `max`. |
| `get_stock_price` | Giá cổ phiếu hiện tại, ví dụ `AAPL` hoặc `TSLA` | Gọi Alpha Vantage qua `ALPHAVANTAGE_API_KEY`. |
| `get_current_weather` | Thời tiết hiện tại của thành phố hoặc địa điểm | Dùng OpenWeather để lấy tọa độ, sau đó gọi API thời tiết qua `OPENWEATHER_API_KEY`. |
| `rag_tool` | Câu hỏi về PDF đã tải lên | Lấy 4 đoạn nội dung tương đồng nhất từ FAISS index cục bộ. |
| `purchase_stock` | Yêu cầu mua cổ phiếu | Mô phỏng lệnh mua, sau đó tạm dừng graph để chờ người dùng phê duyệt. Chỉ `yes` mới hoàn tất lệnh mô phỏng; các lựa chọn khác sẽ hủy lệnh. |

## Setup

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

`OPENAI_API_KEY` is required for the chat model and PDF embeddings. The other keys are required only when using their corresponding tools.

## How to run

From the repository root:

```bash
streamlit run langgraph-chatbot/app.py
```

Then open the local URL printed by Streamlit.
