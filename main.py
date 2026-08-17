import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from swarmagent.config.db import init_db  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # await MongoDB.connect()
    init_routes()
    # await test_runner()
    yield
    # await MongoDB.close()


app = FastAPI(lifespan=lifespan)


def init_routes():
    print("Initializing routes")
    # app.include_router(node_router, prefix="/node")
    # app.include_router(graph_router, prefix="/graph")
    # app.include_router(edge_router, prefix="/edge")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
