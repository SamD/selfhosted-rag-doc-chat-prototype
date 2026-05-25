#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.endpoints import router
from .config import HUB_PORT, STATIC_DIR
from .mqtt.client import AgentHubClient
from .mqtt.registry import AgentRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("hub_server")

registry = AgentRegistry()
mqtt_client = AgentHubClient(registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await mqtt_client.connect()
    app.state.mqtt_client = mqtt_client
    app.state.registry = registry
    log.info("MQTT Agent Hub started on port %s", HUB_PORT)
    yield
    await mqtt_client.disconnect()
    log.info("MQTT Agent Hub shut down")


app = FastAPI(title="MQTT Agent Hub", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

app.state.mqtt_client = None
app.state.registry = None

if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "mqtt_agent_hub.hub_server:app",
        host="0.0.0.0",
        port=HUB_PORT,
        reload=True,
    )
