import uvicorn
from agents.agent_c.config import PORT
from agents.agent_c.server import app

uvicorn.run(app, host="0.0.0.0", port=PORT)
