import uvicorn
from agents.agent_d.config import PORT
from agents.agent_d.server import app

uvicorn.run(app, host="0.0.0.0", port=PORT)
