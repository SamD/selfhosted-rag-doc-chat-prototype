#!/bin/sh
set -e

NIFI_HOST="${NIFI_HOST}"
NIFI_SSH_USER="${NIFI_SSH_USER:-root}"
NIFI_SSH_PASS="${NIFI_SSH_PASS}"
NIFI_EXTENSIONS_DIR="${NIFI_EXTENSIONS_DIR:-/opt/nifi/python_extensions}"
NIFI_CONTAINER_NAME="${NIFI_CONTAINER_NAME:-nifi-2.0-ai}"

echo "📦 Deploying NiFi Python processors to ${NIFI_HOST}:${NIFI_EXTENSIONS_DIR}"

export SSHPASS="${NIFI_SSH_PASS}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "   Creating remote directory..."
sshpass -e ssh ${SSH_OPTS} "${NIFI_SSH_USER}@${NIFI_HOST}" \
    "mkdir -p ${NIFI_EXTENSIONS_DIR}"

echo "   Copying processor files..."
sshpass -e scp ${SSH_OPTS} /app/nifi/python/extensions/*.py \
    "${NIFI_SSH_USER}@${NIFI_HOST}:${NIFI_EXTENSIONS_DIR}/"

echo "   Restarting NiFi container (${NIFI_CONTAINER_NAME})..."
sshpass -e ssh ${SSH_OPTS} "${NIFI_SSH_USER}@${NIFI_HOST}" \
    "docker restart ${NIFI_CONTAINER_NAME}" || {
    echo "   ⚠️  Failed to restart NiFi container"
    echo "   Processors copied but may need manual restart"
}

echo "   Waiting for NiFi to restart..."
sleep 10

echo "   Verifying deployment..."
sshpass -e ssh ${SSH_OPTS} "${NIFI_SSH_USER}@${NIFI_HOST}" \
    "ls -la ${NIFI_EXTENSIONS_DIR}/"

echo "✅ NiFi processors deployed"
