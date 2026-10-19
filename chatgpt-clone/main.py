from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from sqlalchemy.orm import Session

from agent import MCP_TOOL_NAMES, create_chatbot, pending_interrupt
from database import ChatMessage, Conversation, get_db, init_database
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rag import SUPPORTED_EXTENSIONS, read_file_text

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
app = FastAPI(title="Agentic Chatbot")
logger = logging.getLogger(__name__)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.on_event("startup")
async def startup() -> None:
    init_database()
    UPLOADS_DIR.mkdir(exist_ok=True)
    checkpoint_context = AsyncSqliteSaver.from_conn_string(str(BASE_DIR / "databases" / "langgraph_checkpoints.sqlite3"))
    app.state.checkpoint_context = checkpoint_context
    app.state.chatbot = create_chatbot(await checkpoint_context.__aenter__())


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.checkpoint_context.__aexit__(None, None, None)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


def conversation_or_404(conversation_id: str, db: Session) -> Conversation:
    item = db.get(Conversation, conversation_id)
    if not item:
        raise HTTPException(404, "Conversation not found")
    return item


def serialize_conversation(item: Conversation) -> dict:
    return {"id": item.id, "title": item.title, "updated_at": item.updated_at.isoformat()}


@app.get("/api/conversations")
def list_conversations(db: Session = Depends(get_db)):
    return [serialize_conversation(item) for item in db.query(Conversation).order_by(Conversation.updated_at.desc()).all()]


@app.post("/api/conversations", status_code=201)
def create_conversation(db: Session = Depends(get_db)):
    item = Conversation()
    db.add(item)
    db.commit()
    db.refresh(item)
    return serialize_conversation(item)


@app.get("/api/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str, db: Session = Depends(get_db)):
    item = conversation_or_404(conversation_id, db)
    return [{"id": msg.id, "role": msg.role, "content": msg.content, "tool_calls": json.loads(msg.tool_calls or "[]")} for msg in item.messages]


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Delete a conversation and its SQLAlchemy-managed message history."""
    conversation = conversation_or_404(conversation_id, db)
    db.delete(conversation)
    db.commit()


async def run_graph(payload, thread_id: str) -> tuple[str, list[dict[str, str]], str | None]:
    chatbot = app.state.chatbot
    config = {"configurable": {"thread_id": thread_id}, "metadata": {"thread_id": thread_id}, "run_name": "fastapi_chat"}
    chunks: list[str] = []
    calls: list[dict[str, str]] = []
    async for message, _metadata in chatbot.astream(payload, config=config, stream_mode="messages"):
        log_mcp_activity(message)
        if isinstance(message, AIMessage):
            for name in tool_call_names(message):
                if name and not any(item["name"] == name for item in calls):
                    calls.append(tool_call_metadata(name))
            if message.content:
                chunks.append(message.content if isinstance(message.content, str) else str(message.content))
    unresolved = await pending_interrupt(chatbot, thread_id)
    return "".join(chunks), calls, str(unresolved.value) if unresolved else None


def sse(event: str, payload: dict) -> str:
    """Format one Server-Sent Event for the browser fetch stream."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def tool_call_metadata(name: str) -> dict[str, str]:
    """Format a tool call for the browser and persisted conversation history."""
    return {"name": name, "source": "MCP" if name in MCP_TOOL_NAMES else "Tool"}


def tool_call_names(message: AIMessage) -> list[str]:
    """Return tool names from both streamed chunks and completed AI messages."""
    names: list[str] = []
    for call in [*getattr(message, "tool_call_chunks", []), *getattr(message, "tool_calls", [])]:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name and name not in names:
            names.append(name)
    return names


def log_mcp_activity(message) -> None:
    """Write visible logs only for tools supplied by an MCP server."""
    if isinstance(message, AIMessage):
        for name in tool_call_names(message):
            if name in MCP_TOOL_NAMES:
                logger.info("MCP tool requested: %s", name)
    elif isinstance(message, ToolMessage) and message.name in MCP_TOOL_NAMES:
        logger.info("MCP tool completed: %s", message.name)


@app.post("/api/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    message: str = Form(""),
    files: list[UploadFile] | None = File(None),
    db: Session = Depends(get_db),
):
    conversation = conversation_or_404(conversation_id, db)
    chatbot = app.state.chatbot
    if await pending_interrupt(chatbot, conversation_id):
        raise HTTPException(409, "A stock purchase is awaiting approval.")
    uploaded: list[str] = []
    for file in files or []:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {suffix or 'unknown'}")
        stored_name = f"{uuid4().hex}{suffix}"
        destination = UPLOADS_DIR / stored_name
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
            read_file_text(destination)
            uploaded.append(file.filename or stored_name)
        except Exception as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(400, f"Could not read {file.filename}: {error}") from error
        finally:
            file.file.close()
    clean_message = message.strip()
    if not clean_message and not uploaded:
        raise HTTPException(400, "Enter a message or attach a file.")
    agent_message = clean_message or "Please confirm the uploaded file is ready for questions."
    if uploaded:
        agent_message = f"The user uploaded these files in this message: {', '.join(uploaded)}. Use rag_tool if they ask about them.\n\nUser message: {agent_message}"
    db.add(ChatMessage(conversation_id=conversation_id, role="user", content=clean_message or f"Uploaded: {', '.join(uploaded)}"))
    if conversation.title == "New chat" and clean_message:
        conversation.title = clean_message[:60]
    # Save the user's message before the response stays open for streaming.
    db.commit()

    async def generate_response():
        config = {"configurable": {"thread_id": conversation_id}, "metadata": {"thread_id": conversation_id}, "run_name": "fastapi_chat"}
        chunks: list[str] = []
        tool_calls: list[dict[str, str]] = []
        try:
            async for message_chunk, _metadata in chatbot.astream(
                {"messages": [HumanMessage(content=agent_message)]}, config=config, stream_mode="messages"
            ):
                log_mcp_activity(message_chunk)
                if not isinstance(message_chunk, AIMessage):
                    continue
                for tool_name in tool_call_names(message_chunk):
                    if tool_name and not any(item["name"] == tool_name for item in tool_calls):
                        tool = tool_call_metadata(tool_name)
                        tool_calls.append(tool)
                        yield sse("tool", tool)
                if message_chunk.content:
                    token = message_chunk.content if isinstance(message_chunk.content, str) else str(message_chunk.content)
                    chunks.append(token)
                    yield sse("token", {"text": token})
            answer = "".join(chunks)
            unresolved = await pending_interrupt(chatbot, conversation_id)
            interrupt = str(unresolved.value) if unresolved else None
            db.add(ChatMessage(conversation_id=conversation_id, role="assistant", content=answer, tool_calls=json.dumps(tool_calls)))
            db.commit()
            yield sse("done", {"tool_calls": tool_calls, "uploaded": uploaded, "interrupt": interrupt, "title": conversation.title})
        except Exception as error:
            db.rollback()
            logger.exception("Chat stream failed for conversation %s", conversation_id)
            yield sse("error", {"detail": str(error)})

    return StreamingResponse(
        generate_response(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/conversations/{conversation_id}/approval")
async def approve_purchase(conversation_id: str, approved: bool = Form(...), db: Session = Depends(get_db)):
    conversation_or_404(conversation_id, db)
    chatbot = app.state.chatbot
    if not await pending_interrupt(chatbot, conversation_id):
        raise HTTPException(409, "There is no action awaiting approval.")
    answer, tool_calls, interrupt = await run_graph(Command(resume="yes" if approved else "no"), conversation_id)
    db.add(ChatMessage(conversation_id=conversation_id, role="assistant", content=answer, tool_calls=json.dumps(tool_calls)))
    db.commit()
    return {"answer": answer, "tool_calls": tool_calls, "interrupt": interrupt}
