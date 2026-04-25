"""
FastAPI application for the Multi-Agent Hyperlocal Catalog Ops environment.
"""

import os

from openenv.core.env_server.http_server import create_app # type: ignore

try:
    from ..models import MultiAgentAction, MultiAgentObservation
    from .environment import MultiAgentHyperlocalCatalogOpsEnvironment
except ImportError:
    from models import MultiAgentAction, MultiAgentObservation
    from server.environment import MultiAgentHyperlocalCatalogOpsEnvironment


app = create_app(
    MultiAgentHyperlocalCatalogOpsEnvironment,
    MultiAgentAction,
    MultiAgentObservation,
    env_name="multi_agent_hyperlocal_catalog_ops",
    max_concurrent_envs=4,
)
from fastapi.responses import JSONResponse # type: ignore

@app.get("/")
def root():
    return JSONResponse(
        {
            "message": "Multi-Agent Hyperlocal Catalog Ops API is running",
            "docs": "/docs",
            "health": "/health",
            "metadata": "/metadata",
        }
    )

def main() -> None:
    import uvicorn # type: ignore

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
