# Chatbot Agentic phong cách ChatGPT

Project này là phiên bản giao diện web thay cho ứng dụng Streamlit trong `../langgraph-chatbot`. 

Những thay đổi khác:
- chuyển từ Single Agent thành Multi Agent
- sử dụng MCP
- sử dụng Custom Middleware / Guardrail để chặn các request có chứa banned keywords, và mask PII data.
- sử dụng RAG
- áp dụng các cách sau để cải thiện retrieval quality trong RAG: adaptive chunking strategy, adaptive search strategy, re-rank, query rewriting trước khi thực hiện retrieval, meta-data filter
- áp dụng Agentic RAG để khi gặp câu hỏi phức tạp thì chia thành tối đa 4 câu query để truy vấn và liên kết document để đưa ra câu trả lời cuối cùng
- xử lý lỗi "lost in the middle" bằng 4 steps: Re-rank, Contextualization, Context Compression, Context Ordering
- có evaluation (1 dạng unit test cho Agentic AI) bằng cách viết tay (evaluation.py) và framework DeepEval (deepeval_rag.py)

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
| `rag_tool` | Câu hỏi về tệp đã tải lên | Tự áp dụng các strategy và truy xuất tối đa 4 đoạn mỗi câu hỏi từ FAISS index cục bộ. |
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

   Khi có bộ lọc metadata, truy vấn vốn dùng vector sẽ chuyển sang hybrid để tìm thêm theo từ khóa. Các đoạn khớp metadata rõ ràng được ưu tiên trong nhóm ứng viên; đoạn thiếu metadata vẫn đủ điều kiện.

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
RAG search mode=hybrid metadata_filter={'department': 'HR'}
RAG reranking method=lexical_coverage candidates=3 returned=3
```

Log truy vấn chứa câu hỏi gốc và câu đã viết lại; cân nhắc cấu hình logging phù hợp nếu dữ liệu nhạy cảm.

## Agentic RAG: phân rã câu hỏi và tổng hợp bằng chứng

Agent có thể phân rã câu hỏi thành tối đa 4 câu truy xuất, gộp kết quả và đánh giá bằng chứng. Nếu thông tin chưa đủ, agent có thể thực hiện tối đa 2 vòng tìm bổ sung. Khi có nhiều câu hỏi và nhiều nguồn, hệ thống chọn các đoạn liên quan, thêm metadata sẵn có, loại dòng trùng và sắp xếp lại context trước khi trả lời.


### Lost in the middle

**Lost in the middle** là hiện tượng model có thể sử dụng thông tin ở giữa context dài kém hiệu quả hơn thông tin ở đầu hoặc cuối; mức độ phụ thuộc model và tác vụ, không có nghĩa model luôn bỏ qua phần giữa. Hiện tượng này có thể xảy ra với Agentic RAG nhiều lượt retrieval: ví dụ, một câu hỏi tách thành 4 câu con, mỗi câu lấy tối đa 4 đoạn, tạo ra tối đa 16 đoạn trước khi gộp và loại trùng.

Khi câu hỏi ban đầu được phân rã và kết quả đến từ nhiều nguồn, dự án giảm nhiễu và hạn chế hiện tượng này theo thứ tự:

1. **Re-rank:** xếp hạng theo mức khớp từ khóa với câu hỏi, giữ tối đa 6 đoạn phù hợp. Xếp hạng theo từ khóa có thể bỏ sót đoạn liên quan dùng từ khác.
2. **Contextualization:** thêm nguyên văn metadata có sẵn vào đoạn. Ví dụ chung, nếu metadata xác định rõ loại chính sách, thông tin đó có thể làm rõ “This policy applies to premium customers.” đang nói về chính sách nào. Dự án chỉ thêm các trường metadata có sẵn, không tự viết lại câu hay suy đoán khi thiếu metadata để hạn chế hallucination.
3. **Context compression:** loại các dòng giống hệt nhau liền kề; không dùng model tóm tắt để tránh làm sai lệch nội dung.
4. **Context ordering:** đặt đoạn xếp hạng cao nhất ở đầu, đoạn cao thứ hai ở cuối, rồi xếp các đoạn còn lại từ hai phía vào giữa.

Bốn bước này chỉ hỗ trợ giảm nhiễu và rủi ro mất chú ý, không bảo đảm model sử dụng đúng mọi bằng chứng.

### Kiểm tra điều kiện

Với câu hỏi về điều kiện, hệ thống trích xuất các yêu cầu và bằng chứng, kiểm tra trích dẫn rồi xác định kết luận. Khi chưa xác minh được điều kiện bắt buộc, kết quả là **chưa đủ thông tin**. Việc truy xuất và đánh giá bằng model vẫn có thể bỏ sót hoặc hiểu sai thông tin; phép tính tỷ lệ khoản vay hiện chỉ hỗ trợ một mẫu tiếng Việt và dùng năm trước năm hiện tại như một giả định kỹ thuật.

### Câu hỏi thử nghiệm

Tải [`loan-company-a.md`](sample-data/loan-company-a.md) và [`loan-credit-policy.md`](sample-data/loan-credit-policy.md), sau đó hỏi:

`Dựa trên các tài liệu, Công ty A có đủ điều kiện vay 2 tỷ đồng không? Hãy phân tích doanh thu, chính sách tín dụng và rủi ro.`

Kết quả mong đợi: **chưa đủ thông tin** vì thiếu tiêu chí và kết quả đánh giá rủi ro. Nếu tải thêm [`loan-risk-policy.md`](sample-data/loan-risk-policy.md), dữ liệu mẫu cho thấy công ty đáp ứng các điều kiện được nêu. Vector store dùng chung giữa các cuộc hội thoại, nên tài liệu đã tải trước đó có thể ảnh hưởng kết quả.

Log `RAG plan`, `RAG retrieval`, `RAG assessment`, `RAG final context` và `RAG eligibility` giúp theo dõi các bước. Log context có thể chứa nội dung tài liệu.

## Multi-agent

`agent.py` dùng mô hình **router + specialist agents (small agents) + final agent**. Router chọn một hoặc nhiều specialist agents phù hợp; các specialist agents được chọn chạy song song nếu chúng không phụ thuộc vào nhau. Sau đó `final_agent` tổng hợp kết quả thành một câu trả lời duy nhất. Chỉ gọi những specialist agents được router chọn.

Các specialist agents gồm web research, weather, aviation, data/calculator, RAG documents, memory, simulated stock transaction và general chat.

## Evaluation

Chạy các lệnh sau từ thư mục gốc repository sau khi cài `requirements.txt`:

```bash
# Các kiểm tra viết tay và một luồng calculator thực tế
python evaluation.py

# Kiểm thử RAG, eligibility và context bằng model/retrieval giả lập
python -m unittest discover -s tests

# DeepEval: đánh giá câu trả lời thực với tài liệu mẫu cố định
python deepeval_rag.py
```

`evaluation.py` kiểm tra eligibility, bằng chứng, guardrail, redaction, routing, kết quả calculator, thứ tự node và latency. Router và calculator gọi model thật; các kiểm tra eligibility và evidence không gọi model. Báo cáo là bảng PASS/FAIL và mã thoát khác 0 nếu có case thất bại. Ngưỡng latency mặc định 30 giây, thay đổi qua `AGENT_EVALUATION_MAX_LATENCY_SECONDS`.

DeepEval chạy hai câu hỏi trên nội dung cố định từ `sample-data/loan-company-a.md`, lấy câu trả lời thật từ luồng document answer và dùng Faithfulness cùng Answer Relevancy để chấm. Đây là đánh giá **khâu trả lời với context cho sẵn**, không đo chất lượng retrieval hoặc toàn bộ graph. Ngưỡng mặc định là 0.7, có thể đổi bằng `DEEPEVAL_RAG_THRESHOLD` (từ 0 đến 1); kết quả lỗi hoặc dưới ngưỡng trả mã thoát khác 0.

Hai lệnh gọi model cần cấu hình ít nhất một API key OpenAI, Gemini hoặc Groq như ứng dụng. DeepEval dùng cùng thứ tự provider của ứng dụng cho model giám khảo và phát sinh thêm lượt gọi, chi phí và thời gian; điểm LLM-as-a-judge có thể thay đổi giữa các lần chạy. Bộ case hiện là baseline nhỏ, không chứng minh hệ thống không hallucinate. LangSmith tiếp tục hoạt động theo cấu hình hiện có; dữ liệu gửi đến model và trace cần phù hợp với môi trường sử dụng.

### Các tiêu chí cần phát triển tiếp

| Tiêu chí | Hiện tại | Cần bổ sung |
| --- | --- | --- |
| Tỷ lệ hoàn thành task | Chưa đo trực tiếp; báo cáo chỉ đếm số check pass/fail. | Định nghĩa tiêu chí thành công cho từng task và tính số task thành công trên tổng số task. |
| Độ chính xác khi gọi tool | Kiểm tra calculator được gọi trong một task và lựa chọn agent ở ba case routing. | Thêm nhiều task với tool kỳ vọng; đo lựa chọn tool, tham số và các lượt gọi thừa hoặc thiếu. |
| Số bước trung bình | Có thu thập danh sách node cho task calculator để kiểm tra thứ tự. | Định nghĩa một bước là gì, ghi số bước từng task và tính trung bình trên dataset. |
| Token được dùng | Báo cáo eval chưa thu thập hoặc tổng hợp token. | Ghi input/output token theo task và model; tính tổng, trung bình và chi phí khi có dữ liệu giá. LangSmith có thể cung cấp thông tin cho một số trace tùy provider. |
| Độ trễ mỗi task | Đo end-to-end cho một task calculator và so với ngưỡng. | Đo từng task trong dataset; tổng hợp trung bình và p95, có thể chia theo từng giai đoạn. |
| Độ đúng và chất lượng kết quả | Kiểm tra đáp án calculator, logic eligibility và hai câu trả lời với DeepEval Faithfulness/Answer Relevancy. | Thêm các task có đáp án hoặc điều kiện kỳ vọng, gồm đủ điều kiện, không đủ điều kiện và thiếu bằng chứng; đánh giá độ đúng và đầy đủ ngoài hai metric hiện có. |
| Tỷ lệ lỗi | Có số check thất bại và test xử lý lỗi giả lập, nhưng chưa tính tỷ lệ task lỗi. | Ghi lỗi model, tool, retrieval và timeout theo task; tính tỷ lệ task lỗi và phân loại nguyên nhân. |
| Safety | Có case guardrail và kiểm tra hàm che PII/API key với dữ liệu mẫu. | Kiểm tra rò rỉ dữ liệu trong câu trả lời và trace, các trường hợp vượt guardrail, và hành động cần phê duyệt khi chưa được phép. |
| Chất lượng retrieval của RAG | Test giả lập kiểm tra truy xuất bổ sung, giới hạn vòng và xử lý lỗi; DeepEval dùng context cố định. | Tạo dataset có tài liệu hoặc đoạn liên quan được đánh nhãn; chạy retriever thực tế và đo khả năng tìm đúng, tìm đủ bằng chứng. |

Một task có thể tạo nhiều check, nên tỷ lệ check pass trong báo cáo **không phải** tỷ lệ hoàn thành task hay tỷ lệ task không lỗi. Để tổng hợp các tiêu chí trên, cần lưu kết quả theo từng task cùng trace, tool call, context, lỗi và các chỉ số sử dụng tài nguyên; tránh ghi dữ liệu nhạy cảm không cần thiết.

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
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
GEMINI_API_KEY=your_google_ai_api_key
GEMINI_MODEL=gemini-2.5-flash
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=groq/openai/gpt-oss-20b
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

Ứng dụng thử các provider đã cấu hình theo thứ tự OpenAI → Gemini → Groq. Mỗi lần gọi thông thường có tối đa 5 giây; riêng bước phân tích và rà soát điều kiện có tối đa 30 giây cho mỗi provider. Nếu quá thời gian hoặc gặp lỗi, ứng dụng chuyển sang provider tiếp theo; khi nhiều provider cùng chậm, tổng thời gian chờ có thể kéo dài. Với Gemini, hai bước này dùng JSON schema. Log `RAG eligibility model` ghi thời gian và provider đã thành công; log lỗi ghi thời gian cùng giới hạn timeout. Cấu hình API key tương ứng và tùy chọn `OPENAI_MODEL` (mặc định `gpt-4o-mini`), `GEMINI_MODEL` hoặc `GROQ_MODEL`. Embeddings cho RAG chạy cục bộ và không cần API key. Những key còn lại chỉ cần thiết khi sử dụng tool tương ứng.

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
