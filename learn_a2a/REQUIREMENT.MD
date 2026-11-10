# REQUIREMENT: Multi-Agent A2A System

## 1. Mục tiêu

Xây dựng một repository dạng **Monorepo** chứa 5 AI Agent độc lập (Agent
A, B, C, D, E), giao tiếp với nhau thông qua **A2A (Agent2Agent)
protocol**.

Yêu cầu chính:

-   Mỗi Agent là một service độc lập, có thể chạy trong container riêng.
-   Các Agent giao tiếp qua A2A, không gọi trực tiếp function nội bộ của
    Agent khác.
-   Mỗi Agent cung cấp A2A Server và Agent Card để Agent khác khám phá,
    kết nối.
-   Agent nào cần gọi Agent khác thì sử dụng A2A Client.
-   Có một Agent Registry tập trung để đăng ký và tra cứu Agent.
-   Sử dụng Python và A2A Python SDK.
-   Có thể sử dụng LangGraph để triển khai logic/workflow bên trong từng
    Agent.
-   Có thể chạy toàn bộ hệ thống bằng Docker Compose.

## 2. Kiến trúc tổng thể

Repository gồm các thành phần:

1.  **Agent A -- Coordinator Agent**
    -   Là điểm tiếp nhận yêu cầu từ người dùng.
    -   Phân tích yêu cầu và quyết định Agent nào cần thực hiện công
        việc.
    -   Tra cứu Agent Registry để tìm Agent phù hợp.
    -   Gửi task đến Agent B/C/D/E thông qua A2A Client.
    -   Tổng hợp kết quả và trả về cho người dùng.
2.  **Agent B -- Data Analysis Agent**
    -   Chuyên phân tích dữ liệu.
    -   Expose các khả năng thông qua A2A Server.
3.  **Agent C -- Web Research Agent**
    -   Chuyên nghiên cứu/tìm kiếm thông tin.
    -   Expose các khả năng thông qua A2A Server.
4.  **Agent D -- Coding Agent**
    -   Chuyên xử lý các nhiệm vụ liên quan đến lập trình.
    -   Expose các khả năng thông qua A2A Server.
5.  **Agent E -- Report Agent**
    -   Chuyên tổng hợp thông tin và tạo báo cáo.
    -   Expose các khả năng thông qua A2A Server.
6.  **Agent Registry**
    -   Lưu thông tin Agent Card và endpoint của các Agent.
    -   Cung cấp API để Agent đăng ký, cập nhật, xóa và tra cứu Agent.
    -   Cho phép tìm Agent theo tên hoặc skill.
    -   Registry là service nội bộ của project, không phải thành phần
        bắt buộc của A2A protocol.
7.  **Shared**
    -   Chứa các thành phần kỹ thuật dùng chung như A2A client factory,
        card loader, task helpers, cấu hình và logging.
    -   Không chứa logic nghiệp vụ riêng của từng Agent.

## 3. Folder structure

Tạo cấu trúc repository như sau:

``` text
agentic-system/
│
├── agents/
│   ├── agent_a/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── server.py
│   │   ├── client.py
│   │   ├── agent_card.json
│   │   ├── graph.py
│   │   ├── nodes/
│   │   ├── skills/
│   │   │   └── coordinator.py
│   │   ├── config.py
│   │   └── Dockerfile
│   │
│   ├── agent_b/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── server.py
│   │   ├── agent_card.json
│   │   ├── graph.py
│   │   ├── nodes/
│   │   ├── skills/
│   │   │   └── data_analysis.py
│   │   ├── config.py
│   │   └── Dockerfile
│   │
│   ├── agent_c/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── server.py
│   │   ├── agent_card.json
│   │   ├── graph.py
│   │   ├── nodes/
│   │   ├── skills/
│   │   │   └── web_research.py
│   │   ├── config.py
│   │   └── Dockerfile
│   │
│   ├── agent_d/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── server.py
│   │   ├── agent_card.json
│   │   ├── graph.py
│   │   ├── nodes/
│   │   ├── skills/
│   │   │   └── coding.py
│   │   ├── config.py
│   │   └── Dockerfile
│   │
│   └── agent_e/
│       ├── __init__.py
│       ├── main.py
│       ├── server.py
│       ├── agent_card.json
│       ├── graph.py
│       ├── nodes/
│       ├── skills/
│       │   └── report.py
│       ├── config.py
│       └── Dockerfile
│
├── services/
│   └── registry/
│       ├── __init__.py
│       ├── main.py
│       ├── registry.py
│       ├── storage.py
│       ├── models.py
│       └── Dockerfile
│
├── shared/
│   ├── a2a/
│   │   ├── client_factory.py
│   │   ├── card_loader.py
│   │   └── task_helpers.py
│   ├── config/
│   │   └── settings.py
│   ├── logging/
│   │   └── logger.py
│   └── exceptions/
│       └── errors.py
│
├── tests/
│   ├── integration/
│   │   └── test_a2a_communication.py
│   └── e2e/
│       └── test_multi_agent_workflow.py
│
├── docker/
│   └── docker-compose.yml
│
├── pyproject.toml
├── .env.example
└── README.md
```

Lưu ý: - Không tạo thư mục `scripts/`. - Không tạo thư mục `deploy/`. -
Không tạo unit tests. - Dockerfile của từng service đặt cùng thư mục
service tương ứng. - Docker Compose đặt tại `docker/docker-compose.yml`.

## 4. Yêu cầu đối với từng Agent

### 4.1. Cấu trúc chung

Mỗi Agent cần có các thành phần phù hợp với vai trò:

-   `main.py`: entry point, khởi tạo và chạy service.
-   `server.py`: cấu hình A2A Server, expose endpoint để Agent khác gọi.
-   `client.py`: chỉ tạo nếu Agent cần gọi Agent khác. Agent A chắc chắn
    cần client; các Agent còn lại có thể bổ sung khi có nhu cầu.
-   `agent_card.json`: mô tả Agent theo schema của A2A SDK/specification
    đang sử dụng.
-   `graph.py`: định nghĩa LangGraph workflow nếu Agent sử dụng
    LangGraph.
-   `nodes/`: các node trong graph.
-   `skills/`: implementation các nghiệp vụ/skill mà Agent cung cấp.
-   `config.py`: cấu hình riêng của Agent.
-   `Dockerfile`: build image độc lập cho Agent.

Không bắt buộc mọi Agent phải có cả A2A Client và A2A Server. Trong phạm
vi project này, cả 5 Agent đều cần cung cấp A2A Server; chỉ Agent cần
điều phối hoặc gọi Agent khác mới cần A2A Client.

### 4.2. Agent Card

Mỗi Agent phải có Agent Card hợp lệ theo phiên bản A2A SDK/specification
được chọn.

Agent Card cần khai báo tối thiểu các thông tin phù hợp với schema chính
thức:

-   Tên Agent.
-   Mô tả Agent.
-   URL A2A endpoint.
-   Version.
-   Skills.
-   Capabilities.
-   Authentication requirements nếu có.

Không tự định nghĩa schema Agent Card riêng nếu A2A SDK đã cung cấp
schema tương ứng.

Agent Card phải được server cung cấp tại well-known URL theo
specification mà SDK/version đang sử dụng hỗ trợ. Không giả định
endpoint lấy Agent Card và endpoint gửi task là cùng một URL.

Các URL trong Agent Card phải được cấu hình theo môi trường. Khi chạy
Docker Compose, sử dụng hostname/service name nội bộ phù hợp; khi deploy
độc lập, có thể cấu hình public URL hoặc private URL có thể truy cập
được.

### 4.3. A2A Server

Mỗi Agent phải:

-   Khởi tạo A2A Server bằng API của A2A Python SDK.
-   Expose HTTP endpoint để nhận A2A request.
-   Cung cấp Agent Card.
-   Xử lý Message/Task theo protocol.
-   Trả trạng thái Task và kết quả theo đúng schema của SDK.
-   Hỗ trợ cơ chế streaming nếu được SDK hỗ trợ và được bật trong cấu
    hình.
-   Không expose logic nội bộ trực tiếp; chỉ expose các skill được khai
    báo.

### 4.4. A2A Client

A2A Client phải:

-   Lấy/đọc Agent Card của Agent đích.
-   Khởi tạo client dựa trên Agent Card.
-   Gửi message/task theo A2A protocol.
-   Nhận và xử lý trạng thái cũng như kết quả task.
-   Có timeout và xử lý lỗi kết nối.
-   Không hardcode endpoint rải rác trong source code.

Ưu tiên sử dụng API chính thức của `a2a-sdk`. Không tự xây dựng HTTP
payload nếu SDK đã cung cấp client abstraction tương ứng.

## 5. Agent Registry

Xây dựng Registry service đơn giản cho project.

### Chức năng

-   Đăng ký Agent với Agent Card hoặc thông tin Agent Card URL.
-   Cập nhật thông tin Agent.
-   Xóa Agent khỏi Registry.
-   Lấy danh sách Agent.
-   Tra cứu Agent theo ID/name.
-   Tìm Agent theo skill.
-   Trả về Agent Card hoặc Agent Card URL để client có thể kết nối.

### Lưu trữ

Có thể bắt đầu bằng in-memory storage hoặc file-based storage để đơn
giản hóa prototype. Thiết kế storage thông qua interface để có thể thay
thế bằng database sau này.

### Luồng đăng ký

1.  Mỗi Agent khởi động A2A Server.
2.  Agent tạo/cung cấp Agent Card với endpoint đúng theo môi trường.
3.  Agent đăng ký thông tin của mình vào Registry.
4.  Registry lưu thông tin và cho phép Agent khác tra cứu.

Registry không xử lý task thay cho Agent và không nằm trên đường truyền
dữ liệu task giữa hai Agent. Sau khi discovery, A2A Client kết nối trực
tiếp đến A2A Server của Agent đích.

## 6. Luồng hoạt động chính

Ví dụ user gửi yêu cầu cho Agent A:

"Research thị trường AI, phân tích dữ liệu và tạo báo cáo."

Luồng xử lý mong muốn:

1.  User gửi yêu cầu đến Agent A.
2.  Agent A phân tích yêu cầu và xác định các skill cần dùng.
3.  Agent A truy vấn Registry để tìm Agent có skill phù hợp.
4.  Registry trả về thông tin/Agent Card của Agent C (Web Research).
5.  Agent A sử dụng A2A Client gửi task đến Agent C.
6.  Agent C xử lý và trả kết quả cho Agent A.
7.  Agent A tiếp tục tìm Agent B (Data Analysis) và gửi task kèm dữ liệu
    cần phân tích.
8.  Agent B xử lý và trả kết quả.
9.  Agent A tìm Agent E (Report) và gửi dữ liệu research cùng kết quả
    phân tích.
10. Agent E tạo báo cáo và trả kết quả.
11. Agent A tổng hợp và trả kết quả cuối cùng cho user.

Agent A là coordinator của workflow. Không để các Agent gọi trực tiếp
Python function hoặc import logic nghiệp vụ của nhau.

## 7. Docker và chạy local

Sử dụng Docker Compose để chạy các service:

-   agent-a
-   agent-b
-   agent-c
-   agent-d
-   agent-e
-   registry

Yêu cầu:

-   Mỗi Agent chạy trong container riêng.
-   Registry chạy trong container riêng.
-   Các container có thể giao tiếp với nhau qua Docker Network.
-   Sử dụng Docker service name làm hostname nội bộ khi chạy cùng Docker
    Compose.
-   Không hardcode `localhost` để Agent này gọi Agent khác bên trong
    container.
-   Mỗi Agent có port nội bộ riêng hoặc dùng chung port trong các
    container độc lập.
-   Cấu hình endpoint trong Agent Card phải khớp với địa chỉ mà Agent
    Client có thể truy cập.
-   Có healthcheck hoặc cơ chế kiểm tra service readiness phù hợp.
-   Dùng environment variables cho cấu hình môi trường và secrets.
-   Không commit file `.env` chứa secret.
-   Có thể build và chạy toàn bộ hệ thống bằng một lệnh Docker Compose.

Docker Compose đặt tại `docker/docker-compose.yml`. Cần cấu hình build
context phù hợp với repository root để các Dockerfile có thể copy source
code từ đúng vị trí.

## 8. Cấu hình và dependency

-   Sử dụng Python.
-   Sử dụng `a2a-sdk` chính thức.
-   Có thể sử dụng LangGraph cho agent workflow.
-   Quản lý dependency tập trung bằng `pyproject.toml`, trừ khi có lý do
    kỹ thuật rõ ràng để tách dependency từng service.
-   Cấu hình endpoint, port, Registry URL, model credentials và các
    secrets bằng environment variables.
-   Cung cấp `.env.example` với các biến cần thiết và giá trị mẫu không
    chứa secret thật.
-   Không hardcode API key hoặc credentials trong source code.

## 9. Testing

Không tạo unit tests.

Chỉ tạo:

-   **Integration test:** kiểm tra A2A Client có thể discovery Agent
    Card, gửi task và nhận kết quả từ A2A Server.
-   **End-to-end test:** kiểm tra workflow hoàn chỉnh từ Agent A qua các
    Agent cần thiết và trả kết quả về.

Test phải có thể chạy sau khi các service đã được khởi động bằng Docker
Compose.

## 10. Yêu cầu về chất lượng implementation

-   Ưu tiên code rõ ràng, ít abstraction không cần thiết.
-   Không tạo framework nội bộ phức tạp nếu A2A SDK đã cung cấp
    abstraction.
-   Tách biệt logic nghiệp vụ của từng Agent.
-   Tái sử dụng code kỹ thuật chung trong `shared/`.
-   Không để Registry trở thành proxy cho toàn bộ A2A traffic.
-   Không giả định mọi task đều hoàn thành ngay lập tức; xử lý trạng
    thái task theo A2A protocol.
-   Có logging cho request, task ID, trạng thái và lỗi; không log
    secrets.
-   Xử lý timeout, lỗi kết nối, Agent không khả dụng và lỗi từ Agent
    đích.
-   Đảm bảo các Agent có thể được chạy/deploy độc lập trong tương lai.

## 11. Hướng dẫn triển khai cho Codex

Trước khi code:

1.  Kiểm tra phiên bản hiện tại của A2A Python SDK và tài liệu API tương
    ứng.
2.  Chọn một phiên bản SDK ổn định và pin dependency.
3.  Xác minh chính xác cách khởi tạo A2A Server, A2A Client, Agent Card
    và xử lý Task theo phiên bản SDK đã chọn.
4.  Không sử dụng API giả định hoặc ví dụ từ phiên bản SDK cũ.
5.  Nếu cấu trúc/API thực tế khác với mô tả trong requirement, ưu tiên
    specification và SDK chính thức, đồng thời cập nhật README để giải
    thích.

Triển khai theo từng bước:

1.  Tạo shared A2A infrastructure.
2.  Tạo Agent Registry.
3.  Tạo một Agent mẫu (Agent B) có A2A Server và Agent Card.
4.  Tạo Agent A có A2A Client, discovery Agent B và gửi task thành công.
5.  Hoàn thiện Agent C, D, E theo cùng convention.
6.  Tích hợp workflow điều phối tại Agent A.
7.  Hoàn thiện Docker Compose.
8.  Viết integration test và E2E test.
9.  Cập nhật README hướng dẫn cấu hình, chạy local, kiểm tra giao tiếp
    và mở rộng thêm Agent mới.

Không cần triển khai nghiệp vụ AI phức tạp trong prototype. Có thể dùng
logic mẫu đơn giản cho từng skill để tập trung xác minh A2A
communication trước.
