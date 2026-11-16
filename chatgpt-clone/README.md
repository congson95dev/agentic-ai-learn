# Chatbot Agentic phong cách ChatGPT

Project này là phiên bản giao diện web thay cho ứng dụng Streamlit trong `../langgraph-chatbot`. 

Những thay đổi khác:
- chuyển từ Single Agent thành Multi Agent
- sử dụng MCP
- sử dụng Custom Middleware / Guardrail để chặn các request có chứa banned keywords, và mask PII data.
- sử dụng RAG (tự phân tích và sử dụng 3 dạng search khác nhau sao cho phù hợp: vector search, keyword search, hybrid search)
- có evaluation (1 dạng unit test cho Agentic AI)

Techstack: **FastAPI**, **Jinja2**, **LangGraph**, **LangChain**, **MCP**

Chatbot có thể tự quyết định khi nào cần trả lời trực tiếp và khi nào cần gọi tool để:

- tìm kiếm web qua Tavily Remote MCP;
- tra cứu dữ liệu chuyến bay qua Aviationstack MCP;
- tính toán;
- xem giá cổ phiếu, thời tiết hiện tại và dự báo năm ngày;
- hỏi đáp trên các tệp người dùng tải lên (RAG);
- lưu và truy xuất bộ nhớ dài hạn;
- yêu cầu người dùng phê duyệt trước khi mô phỏng mua cổ phiếu.

Giao diện cũng hỗ trợ tạo và quản lý nhiều cuộc hội thoại, hiển thị tiến trình tool qua stream, tải tệp và nhập bằng giọng nói trên trình duyệt. Câu trả lời cuối được hiển thị sau khi model hoàn tất và ứng dụng xử lý nội dung.

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
| `rag_tool` | Câu hỏi về tệp đã tải lên | Tự chọn vector, keyword hoặc hybrid search và truy xuất tối đa 4 đoạn từ FAISS index cục bộ. |
| `save_memory` | Người dùng yêu cầu chatbot ghi nhớ hoặc cần dùng thông tin đã nhớ | Lưu hoặc tìm tối đa 8 bộ nhớ phù hợp trong SQLite. |
| `purchase_stock` | Yêu cầu mua cổ phiếu | Mô phỏng lệnh mua và tạm dừng graph để chờ phê duyệt. Chỉ `yes` mới hoàn tất lệnh mô phỏng; các lựa chọn khác sẽ hủy lệnh. |

## Cải thiện chất lượng retrieval

Truy xuất và dịch chạy cục bộ; agent dùng model đã cấu hình để nhận diện điều kiện metadata từ câu hỏi:

1. **Adaptive chunking strategy**: chọn cách chia theo loại tệp và cấu trúc nội dung. Năm chiến lược thường gặp và trạng thái trong ứng dụng:

   | Strategy | Cách chia và điều kiện | Trạng thái |
   | --- | --- | --- |
   | **Structure-based** | Chia tại heading của Markdown khi có ít nhất hai heading, hoặc tại định nghĩa `def`, `async def`, `class` cấp cao nhất của Python khi có ít nhất hai định nghĩa. Với CSV, mỗi dòng là một đơn vị ban đầu (`row`). | Đang áp dụng. |
   | **Fixed-size** | Chia theo số ký tự khi văn bản không có xuống dòng và không có ranh giới câu được nhận diện. | Đang áp dụng. |
   | **Recursive** | Lần lượt ưu tiên các ranh giới văn bản như đoạn, dòng và khoảng trắng; dùng khi không thỏa điều kiện của structure-based hoặc fixed-size. Cũng dùng để chia tiếp section hoặc dòng quá dài. | Đang áp dụng. |
   | **Semantic** | Chia theo thay đổi về chủ đề hoặc ý nghĩa giữa các đoạn. | Chưa triển khai. Lý do: cần thêm cơ chế đánh giá sự thay đổi ý nghĩa giữa các đoạn, thường dùng embeddings. |
   | **Agentic** | Dùng agent/LLM để xác định ranh giới nội dung dựa trên ngữ cảnh. | Chưa triển khai. Lý do: cần gọi LLM hoặc agent để phân tích nội dung, làm tăng thời gian xử lý và có thể phát sinh chi phí model. |

   Các đoạn dài được giới hạn tối đa 1.000 ký tự với overlap 150 ký tự. Log `strategy=row` thể hiện trường hợp CSV; `structure_based`, `fixed_size` và `recursive` là các giá trị còn lại.
2. **Adaptive search strategy**: tự chọn theo nội dung câu hỏi:

   - **Vector Search**: mặc định cho câu hỏi thông thường có thể diễn đạt khác nội dung tài liệu.
   - **Keyword Search**: dùng khi câu hỏi chứa tên biến/config hoặc chuỗi kỹ thuật cần khớp nguyên văn.
   - **Hybrid Search**: dùng cho truy vấn về tài liệu kỹ thuật, mã lỗi, thuật ngữ chuyên ngành, endpoint, schema hoặc stack trace; kết hợp thứ hạng vector và điểm lexical BM25-style.

   Backend ghi `RAG search mode=<vector|keyword|hybrid>` để kiểm tra chế độ đã chọn.
3. **Query rewriting**: `langdetect` nhận diện ngôn ngữ; với câu tiếng Việt, Argos Translate dịch toàn bộ câu sang tiếng Anh trên máy và nối bản dịch vào câu gốc trước khi truy xuất. Câu gốc được giữ lại để tìm trong tài liệu tiếng Việt và giữ các mã kỹ thuật. Bản dịch chỉ dùng để tìm kiếm; agent được hướng dẫn trả lời theo ngôn ngữ câu hỏi gốc, trừ khi người dùng yêu cầu ngôn ngữ khác. Không cần LLM. Nếu thiếu model hoặc dịch lỗi, hệ thống ghi cảnh báo và tìm bằng câu gốc. Nhận diện ngôn ngữ có thể sai với câu quá ngắn, không dấu hoặc trộn nhiều ngôn ngữ; bản dịch máy cũng có thể sai.
4. **Metadata filtering**: gắn thông tin như phòng ban và tác giả vào từng đoạn tài liệu, sau đó dùng thông tin này để giới hạn phạm vi tìm kiếm. Xem ví dụ ở phần dưới.

   **Metadata filtering hoạt động như thế nào?**

   Ghi thông tin ở đầu tệp văn bản hoặc Markdown, trước nội dung tài liệu:

   ```markdown
   ---
   category: policy
   date: 2026-01-15
   author: Nguyen An
   department: HR
   ---
   # Chính sách nghỉ phép
   Nhân viên có 12 ngày nghỉ phép năm.
   ```

   Hệ thống lưu các thông tin này cùng từng đoạn khi tải tệp lên. Sau đó, bạn có thể hỏi tự nhiên: “Theo tài liệu của phòng HR, nhân viên có bao nhiêu ngày nghỉ phép?” Agent dùng LLM và quy tắc cục bộ để nhận diện điều kiện phòng ban và truyền vào bộ lọc. Quy tắc cục bộ đối chiếu giá trị đã lưu trong index và vẫn hoạt động khi LLM gặp lỗi. Có thể yêu cầu nhiều điều kiện; giá trị đã khai báo phải khớp đầy đủ, không phân biệt hoa thường. Các đoạn thiếu trường được lọc vẫn được tìm, nhưng không thể xác nhận chúng thuộc phòng ban hay tác giả đã yêu cầu.

   Với CSV, dùng các cột cùng tên cho từng dòng. Với PDF/DOCX, khối metadata phải nằm ở đầu phần văn bản được trích xuất. Tài liệu không có metadata vẫn được tìm kể cả khi có bộ lọc. Nếu cần lọc chính xác các tài liệu cũ, hãy tải lại sau khi thêm metadata; upload lại sẽ thêm bản mới. Nhận diện cục bộ chỉ hỗ trợ một số cách diễn đạt rõ ràng; nếu cả model và quy tắc đều không nhận ra điều kiện, tìm kiếm có thể không được lọc. Kiểm tra `RAG metadata extraction` và `metadata_filter` trong log.

   Hệ thống cũng tự lưu tên tệp, chiến lược chia, số thứ tự đoạn và metadata do trình đọc tệp cung cấp.

5. **Re-ranking**: lấy tối đa 12 đoạn tài liệu tìm được, sau đó xếp hạng lại dựa trên tỷ lệ từ trong câu hỏi xuất hiện trong đoạn, các cụm từ đặt trong dấu ngoặc kép và một ưu tiên nhỏ cho thứ hạng ban đầu. Hệ thống trả về tối đa 4 đoạn. Đây là lexical reranker, không phải cross-encoder.

### Câu hỏi thử nghiệm

Tải [`hybrid-search-sample.md`](sample-data/hybrid-search-sample.md) để thử các chế độ search:

1. General semantic/vector search: `Vì sao hệ thống cần idempotency key khi thanh toán?`
2. Hybrid search for an error code: `ERR-PAY-409 nghĩa là gì?`
3. Hybrid search for an HTTP error: `HTTP 429 cần retry như thế nào?`
4. Hybrid search for technical documentation: `Tài liệu kỹ thuật nói gì về cách xử lý lỗi ConnectionPoolTimeout?`
5. Hybrid search for terminology: `Giải thích thuật ngữ exponential backoff trong API thanh toán.`
6. Keyword search for an exact config key: `Trường idempotency_key có tác dụng gì khi thanh toán?`


Tải cả hai tệp [`hr-leave-policy.md`](sample-data/hr-leave-policy.md) và [`engineering-leave-policy.md`](sample-data/engineering-leave-policy.md), sau đó thử:

| Câu hỏi | File mong đợi | Kết quả mong đợi |
| --- | --- | --- |
| `Theo tài liệu của phòng HR, nhân viên có bao nhiêu ngày nghỉ phép năm?` | `hr-leave-policy.md` | 12 ngày; lọc department và dịch truy vấn tiếng Việt. |
| `Theo tài liệu của phòng Engineering, khi nghỉ phép cần làm gì với lịch trực?` | `engineering-leave-policy.md` | Bàn giao lịch trực. |
| `Theo tài liệu của tác giả Nguyen An đề ngày 15/01/2026, ngày phép chuyển sang năm sau thế nào?` | `hr-leave-policy.md` | Tối đa 5 ngày, dùng trước 31/3; hai điều kiện AND. |
| `Theo tài liệu của phòng Finance, nhân viên được nghỉ phép thế nào?` | Không có nếu chỉ tải hai tệp trên | Hai tệp có phòng ban khác nên bị loại; tài liệu thiếu thông tin phòng ban vẫn có thể xuất hiện. |

Tải các tệp sau để thử metadata trong CSV, DOCX và PDF:

| Tệp | Câu hỏi thử nghiệm | Kết quả mong đợi |
| --- | --- | --- |
| [`department-benefits.csv`](sample-data/department-benefits.csv) | Theo tài liệu của phòng HR, nhân viên được hỗ trợ bao nhiêu tiền mỗi năm cho khóa học? | 2 triệu đồng mỗi năm; dòng Engineering có mức hỗ trợ khác. |
| [`department-benefits.csv`](sample-data/department-benefits.csv) | Theo tài liệu của phòng Engineering, nhân viên được hỗ trợ bao nhiêu tiền mỗi năm cho khóa học kỹ thuật? | 5 triệu đồng mỗi năm. |
| [`hr-training-policy.docx`](sample-data/hr-training-policy.docx) | Theo tài liệu của phòng HR, nhân viên cần đăng ký khóa đào tạo trước bao nhiêu ngày làm việc? | Ít nhất 7 ngày làm việc. |
| [`engineering-equipment-policy.pdf`](sample-data/engineering-equipment-policy.pdf) | Theo tài liệu của phòng Engineering, nhân viên nhận máy tính trong bao lâu kể từ ngày bắt đầu làm việc? | Trong vòng 3 ngày làm việc. |

Log backend minh họa (nội dung và số chunk tùy tài liệu):

```text
RAG chunking source=hr-leave-policy.md strategy=structure_based metadata={'category': 'policy', 'date': '2026-01-15', 'author': 'Nguyen An', 'department': 'HR'} chunks=3
RAG translation status=translated provider=argos from=vi to=en
RAG query rewriting original='Theo tài liệu của phòng HR, nhân viên có bao nhiêu ngày nghỉ phép năm?' rewritten='Theo tài liệu của phòng HR, nhân viên có bao nhiêu ngày nghỉ phép năm? According to the HR document, how many days of annual leave do employees have?'
RAG metadata extraction local_filters={'department': 'HR'} applied_filters={'department': 'HR'}
RAG search mode=vector metadata_filter={'department': 'HR'}
RAG reranking method=lexical_coverage candidates=3 returned=3
```

Log truy vấn chứa câu hỏi gốc và câu đã viết lại; cân nhắc cấu hình logging phù hợp nếu dữ liệu nhạy cảm.

## Multi-agent

`agent.py` dùng mô hình **router + specialist agents (small agents) + final agent**. Router chọn một hoặc nhiều specialist agents phù hợp; các specialist agents được chọn chạy song song nếu chúng không phụ thuộc vào nhau. Sau đó `final_agent` tổng hợp kết quả thành một câu trả lời duy nhất. Chỉ gọi những specialist agents được router chọn.

Các specialist agents gồm web research, weather, aviation, data/calculator, RAG documents, memory, simulated stock transaction và general chat.

## Evaluation

Là dạng unit test của Agentic AI

Nó cover 5 tiêu chí:
1. Answer Correctness => check xem trả lời đúng không
2. Tool Selection => check xem có gọi đúng tool không
3. Trajectory => check xem thứ tự gọi tool có đúng không
4. Latency => check thời gian xử lý
5. Safety / Reliability => check kết quả có loại bỏ secret information hoặc bị hallucinations hay không

How to run:

```bash
python3 evaluation.py
```

Lệnh trên in bảng PASS/FAIL theo năm nhóm trên. Từ đó, có thể dựa vào kết quả này để làm căn cứ dừng deployment tự động trong CICD, hoặc dùng để làm báo cáo.

## Guardrail và bảo vệ dữ liệu đầu vào

- Request chứa các keyword bị cấm (`hack`, `exploit`, `malware`, `jailbreak`, `bypass`) được dừng trước khi gọi LLM hoặc tool.
- Mask PII data: Email được thay bằng `[REDACTED_EMAIL]`; credit card được mask; API key dạng `sk-...` được thay bằng `[REDACTED_API_KEY]`.

## Tệp được hỗ trợ cho RAG

Người dùng có thể tải lên các tệp sau để chatbot đọc và trả lời câu hỏi: `.pdf`, `.docx`, `.txt`, `.md`, `.py` và `.csv`.

Nội dung được tách thành các đoạn, tạo embeddings bằng bộ băm từ và ký tự chạy cục bộ và lưu vào FAISS. Các tệp đã tải lên dùng chung một vector store; vì vậy nội dung của những tệp trước đó vẫn có thể được truy xuất trong các cuộc hội thoại sau, cho đến khi bạn xóa dữ liệu cục bộ.

## Setup

Cài đặt dependencies từ thư mục gốc của repository:

```bash
pip install -r requirements.txt
python chatgpt-clone/install_translation_model.py
```

**Lưu ý về model dịch:** Lệnh thứ hai tải và cài model Argos Translate Việt → Anh; cần Internet lúc cài, sau đó dịch chạy offline. Model được lưu trong thư mục dữ liệu của người dùng, không nằm trong repository. Trên Linux, đường dẫn mặc định là `~/.local/share/argos-translate/packages` (có thể khác nếu cấu hình `XDG_DATA_HOME`).

Hãy chạy lệnh cài bằng cùng môi trường Python dùng để chạy server. Ví dụ, nếu môi trường ảo nằm tại `venv/` ở thư mục gốc:

```bash
venv/bin/python -m pip install -r requirements.txt
venv/bin/python chatgpt-clone/install_translation_model.py
```

Nếu log báo `RAG translation status=unavailable reason=vi_en_model_missing`, hãy chạy lại lệnh cài model bằng môi trường và tài khoản đang chạy server; khởi động lại server nếu cần. Khi thiếu model, RAG vẫn tìm bằng truy vấn gốc.

Tạo tệp `.env` tại thư mục gốc của repository:

```env
GEMINI_API_KEY=your_google_ai_api_key
GEMINI_MODEL=gemini-2.5-flash
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=groq/openai/gpt-oss-20b
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

Gemini là provider chính. `GEMINI_API_KEY` dùng cho Gemini và `GEMINI_MODEL` mặc định là `gemini-2.5-flash`. Nếu Gemini không được cấu hình hoặc một lần gọi Gemini bị lỗi, ứng dụng tự động thử lại bằng Groq qua `GROQ_API_KEY`/`GROQ_MODEL`. Embeddings cho RAG được chạy cục bộ và không cần API key. Những key còn lại chỉ cần thiết khi sử dụng tool tương ứng.

Nếu `vector_store/` được tạo bởi phiên bản cũ dùng OpenAI embeddings, lần upload tiếp theo sẽ tự động tạo lại index bằng embedding local mới.

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

Tin nhắn mới được che email, credit card và API key trước khi ghi vào hai nơi lưu trữ trên.

## Nhập bằng giọng nói

Tính năng nhập giọng nói dùng Web Speech API của trình duyệt (`SpeechRecognition` hoặc `webkitSpeechRecognition`). Khả năng sử dụng phụ thuộc vào trình duyệt và quyền microphone. Âm thanh không được tải lên ứng dụng này; trình duyệt chỉ chuyển lời nói thành văn bản trước khi gửi tin nhắn.
