#!/bin/sh
set -e

echo "========================================="
echo "  Pipeline Initialization (Clean Slate)"
echo "========================================="

echo ""
echo "--- Step 1: Flushing Redis Queues ---"
python3 /app/flush_redis.py

echo ""
echo "--- Step 2: Deploying NiFi Processors ---"
if [ -n "$NIFI_SSH_HOST" ]; then
    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    
    echo "   Copying processors to $NIFI_SSH_HOST..."
    ssh $SSH_OPTS "$NIFI_SSH_HOST" "mkdir -p ${NIFI_EXTENSIONS_DIR:-/opt/nifi/python_extensions}"
    scp $SSH_OPTS /app/processors/*.py "$NIFI_SSH_HOST:${NIFI_EXTENSIONS_DIR:-/opt/nifi/python_extensions}/"
    echo "   Restarting NiFi..."
    ssh $SSH_OPTS "$NIFI_SSH_HOST" "docker restart ${NIFI_CONTAINER_NAME:-nifi-2.0-ai}" || true
else
    echo "⏭️  NIFI_SSH_HOST not set, skipping processor deploy"
fi

echo ""
echo "--- Step 3: Deleting NiFi Process Groups ---"
python3 /app/nuke_nifi.py

echo ""
echo "========================================="
echo "  ✅ Clean Slate Complete"
echo "========================================="
