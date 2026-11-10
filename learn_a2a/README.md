# Multi-Agent A2A System

Monorepo học A2A gồm năm service agent độc lập và một Registry service.

```text
User -> Agent A (Coordinator)
          -> Registry: tìm skill
          -> Agent C (Web Research)   -- A2A
          -> Agent B (Data Analysis)  -- A2A
          -> Agent D (Coding)         -- A2A
          -> Agent E (Report)         -- A2A
```

Registry chỉ quản lý discovery. Sau khi nhận `base_url`, Agent A dùng A2A Python SDK để đọc Agent Card tại `/.well-known/agent-card.json` và gọi trực tiếp agent đích. Registry không proxy A2A task.

## Cấu trúc

- `agents/agent_a` đến `agents/agent_e`: mỗi agent là FastAPI/A2A service độc lập, có `main.py`, `server.py`, `config.py`, `graph.py`, `skills/`, `nodes/`, Agent Card template và Dockerfile riêng.
- `services/registry`: Registry HTTP nội bộ với in-memory storage.
- `shared/a2a`: A2A client, card loader, task executor và server factory dùng SDK chính thức.
- `docker/docker-compose.yml`: chạy sáu service bằng một lệnh.
- `tests/integration` và `tests/e2e`: chỉ integration/E2E test, không có unit test.

Các `agent_card.json` là template URL Docker. Khi service chạy, SDK tạo Agent Card thực tế theo `AGENT_*_BASE_URL`, nên URL luôn khớp môi trường hiện tại.

## Cài đặt local

```bash
cd learn_a2a
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Mở sáu terminal, sau khi activate venv trong mỗi terminal:

```bash
uvicorn services.registry.main:app --host 127.0.0.1 --port 9000
python -m agents.agent_b.main
python -m agents.agent_c.main
python -m agents.agent_d.main
python -m agents.agent_e.main
python -m agents.agent_a.main
```

Gọi Agent A bằng A2A client:

```bash
python -c "import asyncio; from shared.a2a.client_factory import send_task; print(asyncio.run(send_task('http://127.0.0.1:8001', 'Research thị trường AI và tạo báo cáo')))"
```

## Docker Compose

```bash
cd learn_a2a
docker compose -f docker/docker-compose.yml up --build
```

Compose dùng hostname nội bộ `agent-a`, `agent-b`, …, `registry`. Không agent nào gọi agent khác bằng `localhost` bên trong container.

Sau khi sáu service đã chạy, test từ host:

```bash
source venv/bin/activate
python -m tests.integration.test_a2a_communication
python -m tests.e2e.test_multi_agent_workflow
```

## Cấu hình

Copy `.env.example` để tham khảo các URL. Các biến quan trọng:

- `REGISTRY_URL`: URL Registry mà mỗi agent dùng để tự đăng ký và Agent A dùng để discovery.
- `AGENT_A_BASE_URL` … `AGENT_E_BASE_URL`: base URL xuất hiện trong Agent Card và Registry.
- `AGENT_A_PORT` … `AGENT_E_PORT`: port local khi không dùng Docker.

Không commit `.env` chứa secret. Demo không yêu cầu key và chỉ dùng logic dummy; thay các file trong `skills/` hoặc `graph.py` bằng logic/LangGraph thực tế khi cần.

## Endpoint chính

| Service | Agent Card | A2A JSON-RPC |
| --- | --- | --- |
| Agent A | `:8001/.well-known/agent-card.json` | `/a2a/coordinator` |
| Agent B | `:8002/.well-known/agent-card.json` | `/a2a/data-analysis` |
| Agent C | `:8003/.well-known/agent-card.json` | `/a2a/web-research` |
| Agent D | `:8004/.well-known/agent-card.json` | `/a2a/coding` |
| Agent E | `:8005/.well-known/agent-card.json` | `/a2a/report` |

Registry API: `PUT /agents/{id}`, `GET /agents`, `GET /agents/{id}`, `GET /agents/by-skill/{skill}`, `DELETE /agents/{id}`.
