FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
CMD ["uvicorn", "services.registry.main:app", "--host", "0.0.0.0", "--port", "9000"]
