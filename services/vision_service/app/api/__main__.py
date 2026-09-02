"""
Convenience entrypoint:

    python -m services.vision_service.app.api

Equivalent to running uvicorn on
services.vision_service.app.api.main:app
"""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "services.vision_service.app.api.main:app",
        host=os.getenv("VISION_HOST", "0.0.0.0"),
        port=int(os.getenv("VISION_PORT", "8000")),
        reload=bool(os.getenv("VISION_RELOAD")),
    )
