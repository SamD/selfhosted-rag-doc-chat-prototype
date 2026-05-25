import os

from shared.defaults import (
    DEFAULT_HUB_PORT,
    DEFAULT_MQTT_BROKER_HOST,
    DEFAULT_MQTT_BROKER_PORT,
    DEFAULT_MQTT_HUB_TOKEN,
    DEFAULT_MQTT_WS_PORT,
)
from shared.env_names import (
    ENV_HUB_PORT,
    ENV_MQTT_BROKER_HOST,
    ENV_MQTT_BROKER_PORT,
    ENV_MQTT_HUB_TOKEN,
    ENV_MQTT_WS_PORT,
)
from shared.topics import (
    DISCOVERY_TOPIC,  # noqa: F401 — re-exported for backward compat
    GLOBAL_TOPIC,  # noqa: F401
    TASK_RESULT_TOPIC,  # noqa: F401
    TASK_TOPIC,  # noqa: F401
    TELEMETRY_TOPIC,  # noqa: F401
)

MQTT_BROKER_HOST = os.getenv(ENV_MQTT_BROKER_HOST, DEFAULT_MQTT_BROKER_HOST)
MQTT_BROKER_PORT = int(os.getenv(ENV_MQTT_BROKER_PORT, str(DEFAULT_MQTT_BROKER_PORT)))
MQTT_WS_PORT = int(os.getenv(ENV_MQTT_WS_PORT, str(DEFAULT_MQTT_WS_PORT)))
MQTT_HUB_TOKEN = os.getenv(ENV_MQTT_HUB_TOKEN, DEFAULT_MQTT_HUB_TOKEN)
HUB_PORT = int(os.getenv(ENV_HUB_PORT, str(DEFAULT_HUB_PORT)))

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astro-dashboard", "dist")
