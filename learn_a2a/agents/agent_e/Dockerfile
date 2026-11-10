FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
CMD ["python", "-m", "agents.agent_e.main"]
