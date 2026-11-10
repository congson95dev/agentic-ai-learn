import uvicorn
from agents.agent_e.config import PORT
from agents.agent_e.server import app

uvicorn.run(app, host="0.0.0.0", port=PORT)
