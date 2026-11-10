import uvicorn
from agents.agent_b.config import PORT
from agents.agent_b.server import app

uvicorn.run(app, host="0.0.0.0", port=PORT)
