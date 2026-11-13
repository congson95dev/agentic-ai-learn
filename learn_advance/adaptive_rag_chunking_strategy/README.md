# Adaptive RAG – Bộ lựa chọn Chunking Strategy

Đây là một prototype phục vụ mục đích học tập. Công cụ đọc document thật, phân tích các tín hiệu có thể quan sát được và đề xuất một Chunking Strategy phù hợp cho từng document.

Nó chỉ chọn strategy; không thực hiện tạo chunks, embeddings, vector store, truy xuất tài liệu hoặc gọi LLM.

## Đọc file thật

Backend hỗ trợ:

- File text/code UTF-8 như `.txt`, `.md`, `.html`, `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.csv` và một số extension code phổ biến.
- PDF thật thông qua `pypdf`.
- DOCX thật thông qua `docx2txt`.

PDF/DOCX được extract thành text trước, sau đó selector áp dụng các heuristic giống như với text truyền trực tiếp. Reader cũng tạo một số metadata bảo thủ từ nội dung, chẳng hạn nested markers và cross-references; metadata thủ công sẽ được merge/override nếu caller cung cấp. Không có bước nào gọi LLM.

Các dependency web và document parser đã có trong `requirements.txt`.

## Setup

Thực hiện từ thư mục gốc của repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu môi trường đã có sẵn Python và dependency, có thể bỏ qua bước tạo
virtual environment.

## Các strategies được hỗ trợ

- `structure_based`: Sử dụng khi phát hiện được cấu trúc đáng tin cậy như Markdown headings, HTML sections, các định nghĩa trong source code hoặc ranh giới sections do parser cung cấp.
- `agentic`: Đề xuất một cách thận trọng cho những nội dung có cấu trúc bất thường, lồng nhau và phụ thuộc lẫn nhau. Không thực sự thực thi Agent.
- `semantic`: Sử dụng khi phát hiện các sự thay đổi chủ đề rõ ràng nhưng tài liệu không có cấu trúc tự nhiên.
- `recursive`: Lựa chọn mặc định an toàn, phù hợp với văn bản thông thường hoặc khi không có đủ thông tin đầu vào.
- `fixed_size`: Chỉ sử dụng cho các luồng văn bản phẳng, không có ranh giới nội dung hữu ích.

Các quy tắc được thực thi theo thứ tự trên.

Giá trị `confidence` là mức độ tin cậy ước tính dựa trên các quy tắc xác định trước, không phải xác suất được học từ dữ liệu.

Chỉ dựa vào tên file PDF hoặc DOCX không đủ để xác định tài liệu có cấu trúc. Metadata do parser cung cấp phải chứa các headings hoặc ranh giới sections có thể nhận diện được.

## Năm file sample và logic phân định

Năm nút sample trong FastAPI dùng đúng năm file sau trong
`learn_advance/sample-data/`. Các file này được đọc qua cùng pipeline với
file người dùng upload; tên nút không ép kết quả.

### 1. `adaptive-structure.md` → `structure_based`

Ví dụ dữ liệu:

```markdown
# Product Overview

## Architecture

## Deployment
```

Reader đếm được ít nhất hai Markdown heading. Vì có ranh giới native rõ
ràng, selector ưu tiên `structure_based`.

### 2. `adaptive-recursive.txt` → `recursive`

Ví dụ dữ liệu:

```text
This ordinary report explains the project goals and implementation approach.

The second paragraph describes the rollout plan and maintenance work.
```

Có paragraph boundaries nhưng không có headings, code definitions, topic
cues hoặc layout bất thường. Selector dùng `recursive` làm lựa chọn an toàn.

### 3. `adaptive-semantic.txt` → `semantic`

Ví dụ dữ liệu:

```text
Topic: Volcanoes erupt when magma pressure fractures rock layers.

Topic: Jazz improvisation uses harmony, rhythm, and ensemble interaction.

Topic: Database indexing improves query planning and storage layouts.
```

Ba paragraph có topic cues rõ ràng và vocabulary gần như tách biệt. Không
có native structure nên selector chọn `semantic`; prototype không tạo
embedding và không tính semantic similarity thật.

### 4. `adaptive-fixed-size.txt` → `fixed_size`

Ví dụ dữ liệu:

```text
alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu
```

Đây là một text stream phẳng, không có newline nội dung, paragraph hoặc
ranh giới hữu ích. Selector chọn `fixed_size`.

### 5. `adaptive-agentic.txt` → `agentic`

Ví dụ dữ liệu có nested markers và cross-references:

```text
Fragment A
    - level one
        - level two
            - level three
See section 2.1 and refer to appendix A.
See section 3.4 and refer to appendix B.
```

Reader tự đếm nested markers và reference markers, sau đó tạo metadata
`irregular_content=true`. Selector nhận diện nội dung fragmented,
interdependent và đề xuất `agentic`. Đây chỉ là recommendation; không có
LLM/agent nào được thực thi.

### Thứ tự rule thực tế

1. Có cấu trúc đáng tin cậy (headings, HTML sections, code definitions,
   parser boundaries) → `structure_based`.
2. Có nested structure và cross-references bất thường → `agentic`.
3. Có nhiều topic khác nhau nhưng không có structure → `semantic`.
4. Text phẳng, không có boundary hữu ích → `fixed_size`.
5. Các trường hợp còn lại → `recursive`.

Với PDF/DOCX thật, backend dùng parser tương ứng để extract text. Với PDF
không có text layer (ví dụ scan image), selector sẽ không thể suy luận đầy
đủ cấu trúc nếu chưa có OCR/layout metadata.

## FastAPI upload UI

`app.py` cung cấp một giao diện upload đơn giản, các nút sample cho đủ 5 strategy và một JSON API. UI chỉ hiển thị strategy, lý do, confidence và các signals; nó không triển khai chunking.

## How to run

Khởi động FastAPI từ thư mục module:

```bash
cd learn_advance/adaptive_rag_chunking_strategy
uvicorn app:app --reload
```

Mở trình duyệt tại [http://127.0.0.1:8000/](http://127.0.0.1:8000/). Bạn có thể chọn file thật bằng nút **Choose File**, hoặc bấm một trong năm nút sample để kiểm tra từng strategy.

Khi chạy xong, dừng server bằng `Ctrl+C` trong terminal.

Khi FastAPI service đã được mở, truy cập `http://127.0.0.1:8000/` để upload
file hoặc bấm một trong năm nút sample. Kết quả hiển thị strategy, lý do,
confidence và các tín hiệu phân tích.

Các endpoint chính:

- `GET /`: HTML upload form.
- `POST /analyze`: upload từ form và nhận kết quả HTML.
- `POST /analyze-sample`: chạy một trong 5 file sample có sẵn từ UI.
- `POST /api/analyze`: upload multipart và nhận JSON.
- `GET /health`: health check.

Flow của backend là `UploadFile -> parser -> extracted text/metadata -> select_chunking_strategy -> strategy + reason`. Không có LLM trong flow này. Nếu sau này muốn dùng LLM để phân tích các tài liệu mơ hồ, có thể thêm một bước tạo metadata trước selector, nhưng đó không phải dependency của prototype hiện tại.

## Các test case hiện có

File `test_adaptive_rag_chunking_strategy.py` hiện bao phủ 11 trường hợp:

| Test | Input chính | Strategy kỳ vọng | Mục đích kiểm tra |
|---|---|---|---|
| `test_markdown_headings_choose_structure_based` | File Markdown có `#`, `##` và nhiều heading | `structure_based` | Xác nhận heading Markdown cung cấp ranh giới cấu trúc đáng tin cậy. |
| `test_source_definitions_choose_structure_based` | File Python có một `class` và một `def` | `structure_based` | Xác nhận nhiều định nghĩa trong source code được nhận diện như cấu trúc tự nhiên. |
| `test_parser_supplied_pdf_sections_choose_structure_based` | PDF có metadata `section_boundaries` gồm 2 section | `structure_based` | Xác nhận selector dùng section boundaries do parser cung cấp, không chỉ dựa vào đuôi `.pdf`. |
| `test_ordinary_prose_chooses_recursive` | Văn bản thường gồm 2 đoạn văn | `recursive` | Kiểm tra lựa chọn mặc định an toàn cho prose có paragraph nhưng không có cấu trúc đặc biệt. |
| `test_flat_text_chooses_fixed_size` | Một chuỗi text phẳng, không có newline/ranh giới hữu ích | `fixed_size` | Kiểm tra fallback fixed-size cho input rất đơn giản. |
| `test_topic_shifts_choose_semantic` | 3 paragraph bắt đầu bằng `Topic:` và nói về các chủ đề khác nhau | `semantic` | Kiểm tra phát hiện nhiều topic rõ ràng nhưng không có native structure. |
| `test_irregular_interdependent_content_recommends_agentic` | Metadata có `complex_layout=true`, `cross_references=true` | `agentic` | Kiểm tra recommendation agentic cho layout bất thường và nội dung phụ thuộc lẫn nhau; test không chạy LLM. |
| `test_insufficient_input_chooses_recursive` | Không truyền file, type, content hoặc metadata | `recursive` | Kiểm tra fallback khi không đủ thông tin để đánh giá cấu trúc hay topic. |
| `test_plain_text_file_reader_returns_text_and_metadata` | Bytes của Markdown có 2 heading | Reader trả text + metadata | Kiểm tra file reader text thật và metadata cấu trúc do parser tạo ra. |
| `test_binary_unknown_file_is_rejected` | Bytes không phải UTF-8 với extension không hỗ trợ | `DocumentReadError` | Kiểm tra backend từ chối file không thể đọc an toàn thay vì đoán nội dung. |
| `test_irregular_file_reader_infers_agentic_signals` | Text có 3 nested markers và 4 cross-references | `agentic` | Kiểm tra reader tự tạo tín hiệu irregular mà không cần metadata JSON riêng. |

Mỗi test dùng helper `assert_strategy`, nên ngoài strategy còn xác nhận rằng kết quả có:

- `reason` không rỗng;
- trường `signals`;
- `signals.file_type`.

Các test đều là unit test thuần Python: không cần API key, model, internet, embedding, vector database hoặc dịch vụ bên ngoài.
