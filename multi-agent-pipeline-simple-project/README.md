# Multi-Agent Pipeline Research System

Project này là một ví dụ đơn giản về **Multi-Agent Research System** sử dụng LangChain.

Mục tiêu là chia bài toán research thành nhiều bước, mỗi agent/chain đảm nhiệm một nhiệm vụ riêng.

## Architecture

```text
User
 │
 ▼
Search Agent
 │
 │ tìm kiếm thông tin / URLs
 ▼
Reader Agent
 │
 │ đọc và lấy nội dung từ URLs
 ▼
Writer
 │
 │ tổng hợp research thành report
 ▼
Critic
 │
 │ đánh giá chất lượng report
 ▼
Final Report
```

## How to Run

### Streamlit

```bash
streamlit run multi-agent-pipeline-simple-project/app.py
```

### Python

```bash
python3 multi-agent-pipeline-simple-project/main.py
```
