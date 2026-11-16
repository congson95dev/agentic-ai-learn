# Hybrid Search sample data

Tài liệu này dùng để kiểm thử ba chế độ truy xuất của chatgpt-clone. Hãy tải tệp lên giao diện trước khi đặt câu hỏi.

## Technical runbook

The payment API returns **ERR-PAY-409** when the same idempotency key is submitted more than once. Reuse of an idempotency key is safe only when the original request is still being retried; otherwise create a new key.

For **HTTP 429**, the client must use exponential backoff starting at two seconds and respect the `Retry-After` response header. A maximum of five retries is recommended.

The `POST /v1/payments` endpoint requires `amount`, `currency`, and `idempotency_key`. The `currency` value is an ISO 4217 code such as `USD` or `VND`.

When the worker logs `ConnectionPoolTimeout`, inspect the database pool size and the `DB_POOL_TIMEOUT` configuration before restarting the service. Restarting can hide the symptom but does not resolve pool exhaustion.

## Glossary

- **Idempotency key**: a client supplied identifier that prevents a retried request from creating a duplicate payment.
- **Exponential backoff**: a retry policy that increases the wait duration after each failure.
- **Connection pool exhaustion**: a condition where all database connections are busy and new work must wait or time out.
