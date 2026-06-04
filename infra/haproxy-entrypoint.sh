#!/bin/sh
# HAProxy entrypoint: renders config from environment variables.
#
# Usage: haproxy-entrypoint.sh supervisor|embedding
#
# Reads SUPERVISOR_LLM_ENDPOINTS or EMBEDDING_ENDPOINTS and generates
# the appropriate HAProxy config.
#
#   0 endpoints  → 503 dummy (container starts but rejects all traffic)
#   1 endpoint   → transparent proxy to that single backend
#   2+ endpoints → leastconn load balancing with health checks
#
# This script is designed to run inside the haproxy:2.8-alpine container.
# The entrypoint generates the config, then exec's haproxy -f /tmp/haproxy.cfg.

set -e

SERVICE="${1:-supervisor}"
CFG="/tmp/haproxy.cfg"

# ---------------------------------------------------------------------------
# Helper: parse comma-separated URLs into HAProxy server lines
# ---------------------------------------------------------------------------
parse_server_lines() {
    raw="$1"
    idx=0
    OLD_IFS="$IFS"
    IFS=','
    for part in $raw; do
        part=$(echo "$part" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [ -z "$part" ] && continue

        # Extract host:port from URL
        host_port="${part#http://}"
        host_port="${host_port#https://}"
        host_port="${host_port%%/*}"

        echo "    server srv${idx} ${host_port} check inter 2s fall 3 rise 2"
        idx=$((idx + 1))
    done
    IFS="$OLD_IFS"
}

# ---------------------------------------------------------------------------
# Helper: extract host:port from a single URL
# ---------------------------------------------------------------------------
parse_single_host_port() {
    url="$1"
    url=$(echo "$url" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    host_port="${url#http://}"
    host_port="${host_port#https://}"
    host_port="${host_port%%/*}"
    echo "$host_port"
}

# ---------------------------------------------------------------------------
# Count non-empty entries in a comma-separated list
# ---------------------------------------------------------------------------
count_entries() {
    raw="$1"
    count=0
    OLD_IFS="$IFS"
    IFS=','
    for part in $raw; do
        part=$(echo "$part" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [ -z "$part" ] && continue
        count=$((count + 1))
    done
    IFS="$OLD_IFS"
    echo "$count"
}

# ---------------------------------------------------------------------------
# Supervisor config
# ---------------------------------------------------------------------------
if [ "$SERVICE" = "supervisor" ]; then
    ENDPOINTS="${SUPERVISOR_LLM_ENDPOINTS:-}"
    COUNT=$(count_entries "$ENDPOINTS")

    if [ "$COUNT" -eq 0 ]; then
        cat > "$CFG" <<'EOF'
global
    log stdout format raw daemon info
    maxconn 1

defaults
    mode http
    log global
    option dontlognull

frontend fe_dummy
    bind *:11437
    mode http
    http-request deny deny_status 503
EOF
        echo "ℹ️  Supervisor: 0 endpoints, HAProxy idle (503)"

    elif [ "$COUNT" -eq 1 ]; then
        HOST_PORT=$(parse_single_host_port "$ENDPOINTS")
        cat > "$CFG" <<HAPROXY_EOF
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
    option httpchk GET /models
    http-check expect status 200
    server srv0 ${HOST_PORT} check inter 5s fall 3 rise 2

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
HAPROXY_EOF
        echo "✅ Supervisor HAProxy config: 1 backend (${HOST_PORT})"

    else
        SERVERS=$(parse_server_lines "$ENDPOINTS")
        cat > "$CFG" <<HAPROXY_EOF
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
    balance roundrobin
    option httpchk GET /models
    http-check expect status 200
    option httpclose
${SERVERS}
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
HAPROXY_EOF
        echo "✅ Supervisor HAProxy config: $COUNT backends"
    fi

# ---------------------------------------------------------------------------
# Embedding config
# ---------------------------------------------------------------------------
elif [ "$SERVICE" = "embedding" ]; then
    ENDPOINTS="${EMBEDDING_ENDPOINTS:-}"
    COUNT=$(count_entries "$ENDPOINTS")

    if [ "$COUNT" -eq 0 ]; then
        cat > "$CFG" <<'EOF'
global
    log stdout format raw daemon info
    maxconn 1

defaults
    mode http
    log global
    option dontlognull

frontend fe_dummy
    bind *:11438
    mode http
    http-request deny deny_status 503
EOF
        echo "ℹ️  Embedding: 0 endpoints, HAProxy idle (503)"

    elif [ "$COUNT" -eq 1 ]; then
        HOST_PORT=$(parse_single_host_port "$ENDPOINTS")
        cat > "$CFG" <<HAPROXY_EOF
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
    option httpchk GET /models
    http-check expect status 200
    server srv0 ${HOST_PORT} check inter 5s fall 3 rise 2

listen stats
    bind *:8405
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
HAPROXY_EOF
        echo "✅ Embedding HAProxy config: 1 backend (${HOST_PORT})"

    else
        SERVERS=$(parse_server_lines "$ENDPOINTS")
        cat > "$CFG" <<HAPROXY_EOF
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
    balance roundrobin
    option httpchk GET /models
    http-check expect status 200
    option httpclose
${SERVERS}
listen stats
    bind *:8405
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
HAPROXY_EOF
        echo "✅ Embedding HAProxy config: $COUNT backends"
    fi

# ---------------------------------------------------------------------------
# Whisper config
# ---------------------------------------------------------------------------
elif [ "$SERVICE" = "whisper" ]; then
    ENDPOINTS="${WHISPER_MODEL_ENDPOINTS:-}"
    COUNT=$(count_entries "$ENDPOINTS")

    if [ "$COUNT" -eq 0 ]; then
        cat > "$CFG" <<'EOF'
global
    log stdout format raw daemon info
    maxconn 1

defaults
    mode http
    log global
    option dontlognull

frontend fe_dummy
    bind *:11439
    mode http
    http-request deny deny_status 503
EOF
        echo "ℹ️  Whisper: 0 endpoints, HAProxy idle (503)"

    elif [ "$COUNT" -eq 1 ]; then
        HOST_PORT=$(parse_single_host_port "$ENDPOINTS")
        cat > "$CFG" <<HAPROXY_EOF
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

frontend fe_whisper
    bind *:11439
    default_backend be_whisper

backend be_whisper
    option httpchk GET /health
    http-check expect status 200
    server srv0 ${HOST_PORT} check inter 5s fall 3 rise 2

listen stats
    bind *:8406
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
HAPROXY_EOF
        echo "✅ Whisper HAProxy config: 1 backend (${HOST_PORT})"

    else
        SERVERS=$(parse_server_lines "$ENDPOINTS")
        cat > "$CFG" <<HAPROXY_EOF
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

frontend fe_whisper
    bind *:11439
    default_backend be_whisper

backend be_whisper
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    option httpclose
${SERVERS}
listen stats
    bind *:8406
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
HAPROXY_EOF
        echo "✅ Whisper HAProxy config: $COUNT backends"
    fi

# ---------------------------------------------------------------------------
# OCR config
# ---------------------------------------------------------------------------
elif [ "$SERVICE" = "ocr" ]; then
    ENDPOINTS="${OCR_ENDPOINTS:-}"
    COUNT=$(count_entries "$ENDPOINTS")

    if [ "$COUNT" -eq 0 ]; then
        cat > "$CFG" <<'EOF'
global
    log stdout format raw daemon info
    maxconn 1

defaults
    mode http
    log global
    option dontlognull

frontend fe_dummy
    bind *:11440
    mode http
    http-request deny deny_status 503
EOF
        echo "ℹ️  OCR: 0 endpoints, HAProxy idle (503)"

    elif [ "$COUNT" -eq 1 ]; then
        HOST_PORT=$(parse_single_host_port "$ENDPOINTS")
        cat > "$CFG" <<HAPROXY_EOF
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

frontend fe_ocr
    bind *:11440
    default_backend be_ocr

backend be_ocr
    option httpchk GET /health
    http-check expect status 200
    server srv0 ${HOST_PORT} check inter 5s fall 3 rise 2

listen stats
    bind *:8407
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
HAPROXY_EOF
        echo "✅ OCR HAProxy config: 1 backend (${HOST_PORT})"

    else
        SERVERS=$(parse_server_lines "$ENDPOINTS")
        cat > "$CFG" <<HAPROXY_EOF
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

frontend fe_ocr
    bind *:11440
    default_backend be_ocr

backend be_ocr
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    option httpclose
${SERVERS}
listen stats
    bind *:8407
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
HAPROXY_EOF
        echo "✅ OCR HAProxy config: $COUNT backends"
    fi

else
    echo "❌ Unknown service: $SERVICE (must be 'supervisor', 'embedding', 'whisper', or 'ocr')"
    exit 1
fi

# ---------------------------------------------------------------------------
# Start HAProxy
# ---------------------------------------------------------------------------
exec haproxy -f "$CFG" -db