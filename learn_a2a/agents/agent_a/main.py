import uvicorn
from agents.agent_a.config import PORT
from agents.agent_a.server import app

uvicorn.run(app, host="0.0.0.0", port=PORT)
