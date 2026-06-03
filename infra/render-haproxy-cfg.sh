#!/usr/bin/env bash
# Render HAProxy configs from environment variables.
#
# SUPERVISOR_LLM_ENDPOINTS  — comma-separated list of supervisor LLM backends
# EMBEDDING_ENDPOINTS         — comma-separated list of embedding backends
#
# Each list is parsed into HAProxy server lines with health checks.
# If a list has 0 or 1 entries, the corresponding HAProxy container is
# unnecessary and the Python worker talks directly to the endpoint.
#
# Usage:
#   ./render-haproxy-cfg.sh [OUTPUT_DIR]
#
# Default OUTPUT_DIR is the directory containing this script (infra/).
# Writes:
#   haproxy-supervisor.cfg
#   haproxy-embedding.cfg
# Only writes a file if the corresponding env var has 2+ endpoints.

set -euo pipefail

OUTPUT_DIR="${1:-$(dirname "$0")}"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Helper: parse comma-separated URLs into HAProxy server lines
# ---------------------------------------------------------------------------
parse_server_lines() {
    local raw="$1"
    local lines=""
    local idx=0

    # Split on comma, strip whitespace
    IFS=',' read -ra parts <<< "$raw"
    for part in "${parts[@]}"; do
        part="$(echo "$part" | xargs)"  # trim
        [[ -z "$part" ]] && continue

        # Extract host:port from URL
        # Remove protocol
        local host_port="${part#http://}"
        host_port="${host_port#https://}"
        # Remove path
        host_port="${host_port%%/*}"
        # Remove trailing slash (shouldn't be any after above)
        host_port="${host_port%/}"

        lines="${lines}    server srv${idx} ${host_port} check inter 2s fall 3 rise 2
"
        ((idx++))
    done

    echo "$lines"
}

# ---------------------------------------------------------------------------
# Supervisor HAProxy config
# ---------------------------------------------------------------------------
SUPERVISOR_ENDPOINTS="${SUPERVISOR_LLM_ENDPOINTS:-}"
SUPERVISOR_COUNT=0
if [[ -n "$SUPERVISOR_ENDPOINTS" ]]; then
    # Count non-empty entries
    IFS=',' read -ra parts <<< "$SUPERVISOR_ENDPOINTS"
    for p in "${parts[@]}"; do
        p="$(echo "$p" | xargs)"
        [[ -n "$p" ]] && SUPERVISOR_COUNT=$((SUPERVISOR_COUNT + 1))
    done
fi

if [[ "$SUPERVISOR_COUNT" -gt 1 ]]; then
    SUPERVISOR_SERVERS="$(parse_server_lines "$SUPERVISOR_ENDPOINTS")"
    cat > "$OUTPUT_DIR/haproxy-supervisor.cfg" <<HAPROXY_EOF
global
    log stdout format raw daemon info
    maxconn 4096

defaults
    mode http
    log global
    option httplog
    option dontlognull
    option log-health-checks
    option redispatch
    timeout connect 5s
    timeout client  300s
    timeout server  300s
    retries 3
    retry-on all-retryable-errors

frontend fe_supervisor
    bind *:11437
    default_backend be_supervisor

backend be_supervisor
    balance leastconn
    option httpchk GET /models
    http-check expect status 200
    option http-keep-alive
    timeout http-keep-alive 1s
${SUPERVISOR_SERVERS}
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
HAPROXY_EOF
    echo "✅ Wrote $OUTPUT_DIR/haproxy-supervisor.cfg ($SUPERVISOR_COUNT backends)"
else
    rm -f "$OUTPUT_DIR/haproxy-supervisor.cfg"
    echo "ℹ️  Supervisor: 0-1 endpoints, HAProxy not needed"
fi

# ---------------------------------------------------------------------------
# Embedding HAProxy config
# ---------------------------------------------------------------------------
EMBEDDING_ENDPOINTS="${EMBEDDING_ENDPOINTS:-}"
EMBEDDING_COUNT=0
if [[ -n "$EMBEDDING_ENDPOINTS" ]]; then
    IFS=',' read -ra parts <<< "$EMBEDDING_ENDPOINTS"
    for p in "${parts[@]}"; do
        p="$(echo "$p" | xargs)"
        [[ -n "$p" ]] && EMBEDDING_COUNT=$((EMBEDDING_COUNT + 1))
    done
fi

if [[ "$EMBEDDING_COUNT" -gt 1 ]]; then
    EMBEDDING_SERVERS="$(parse_server_lines "$EMBEDDING_ENDPOINTS")"
    cat > "$OUTPUT_DIR/haproxy-embedding.cfg" <<HAPROXY_EOF
global
    log stdout format raw daemon info
    maxconn 4096

defaults
    mode http
    log global
    option httplog
    option dontlognull
    option log-health-checks
    option redispatch
    timeout connect 5s
    timeout client  60s
    timeout server  60s
    retries 3
    retry-on all-retryable-errors

frontend fe_embedding
    bind *:11438
    default_backend be_embedding

backend be_embedding
    balance leastconn
    option httpchk GET /models
    http-check expect status 200
    option http-keep-alive
    timeout http-keep-alive 1s
${EMBEDDING_SERVERS}
listen stats
    bind *:8405
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
HAPROXY_EOF
    echo "✅ Wrote $OUTPUT_DIR/haproxy-embedding.cfg ($EMBEDDING_COUNT backends)"
else
    rm -f "$OUTPUT_DIR/haproxy-embedding.cfg"
    echo "ℹ️  Embedding: 0-1 endpoints, HAProxy not needed"
fi