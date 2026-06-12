#!/usr/bin/env bash
set -euo pipefail

#######################################
# ✅ REQUIRED ENVIRONMENT VARIABLES  #
#######################################
# DEFAULT_DOC_INGEST_ROOT
#   - The parent directory for all lifecycle stages
#
# LLM_PATH
#   - Must be a valid .gguf file for local mode
#   - OR an http(s) URL for remote mode
#
# SUPERVISOR_LLM_ENDPOINTS
#   - Must be a valid .gguf file for local mode
#   - OR an http(s) URL for remote mode
#   - OR comma-separated URLs for HAProxy load balancing
#
# EMBEDDING_ENDPOINTS
#   - Must be a directory containing config.json or remote URL
#   - OR comma-separated URLs for HAProxy load balancing
#
#######################################

export DOCKER_HOST="${DOCKER_HOST:=unix://$XDG_RUNTIME_DIR/docker.sock}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 0. GLOBAL DEFINITIONS
REQUIRED_VARS=(
  "DEFAULT_DOC_INGEST_ROOT:dir"
  "LLM_PATH:gguf"
  "SUPERVISOR_LLM_ENDPOINTS:any"
  "EMBEDDING_ENDPOINTS:any"
  "WHISPER_MODEL_ENDPOINTS:any"
  "OCR_ENDPOINTS:any"
  "PDF_FORCE_OCR:any"
  "VECTOR_DB_TIMEOUT:any"
  "VECTOR_DB_BATCH_SIZE:any"
  "REDIS_HOST:any"
  "REDIS_PORT:any"
)

# 1. LOAD ENV FILE INTO MAP FIRST
ENV_FILE="ingest-svc.env"
declare -A env_map

if [[ -f "$ENV_FILE" ]]; then
  echo "🔄 Loading variables from $ENV_FILE"
  while IFS='=' read -r key val || [[ -n "$key" ]]; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    clean_key=$(echo "$key" | xargs)
    clean_val=$(echo "$val" | xargs | tr -d '\r')
    env_map["$clean_key"]="$clean_val"
  done < "$ENV_FILE"
fi

# 2. SET DEFAULTS AND EXPORTS
if [[ -z "${VECTOR_DB_PROFILE:-}" ]]; then
  VECTOR_DB_PROFILE="${env_map[VECTOR_DB_PROFILE]:-qdrant}"
fi

export VECTOR_DB_URL="${VECTOR_DB_URL:-${env_map[VECTOR_DB_URL]:-}}"
export OCR_ENDPOINTS="${OCR_ENDPOINTS:-${env_map[OCR_ENDPOINTS]:-LOCAL}}"
export PDF_FORCE_OCR="${PDF_FORCE_OCR:-${env_map[PDF_FORCE_OCR]:-false}}"
export VECTOR_DB_TIMEOUT="${VECTOR_DB_TIMEOUT:-${env_map[VECTOR_DB_TIMEOUT]:-60.0}}"
export VECTOR_DB_BATCH_SIZE="${VECTOR_DB_BATCH_SIZE:-${env_map[VECTOR_DB_BATCH_SIZE]:-20}}"
export VECTOR_DB_PROFILE

# 3. PROFILE LOGIC
COMPOSE_FILE_OPTS="--profile cuda-$VECTOR_DB_PROFILE --profile cuda --profile with-frontend"

case "$VECTOR_DB_URL" in
  http://*|https://*)
    echo "📡 Remote Vector DB detected: $VECTOR_DB_URL"
    echo "📡 Skipping local Qdrant instance."
    ;;
  *)
    echo "🏠 Using local Qdrant configuration."
    COMPOSE_FILE_OPTS="$COMPOSE_FILE_OPTS --profile local-qdrant"
    ;;
esac

# 4b. MULTI-ENDPOINT: export endpoints and auto-override to haproxy URL
# Save originals for HAProxy containers (they need the raw endpoint list)
SUPERVISOR_ENDPOINTS="${SUPERVISOR_LLM_ENDPOINTS:-${env_map[SUPERVISOR_LLM_ENDPOINTS]:-}}"
EMBEDDING_ENDPOINTS_VAL="${EMBEDDING_ENDPOINTS:-${env_map[EMBEDDING_ENDPOINTS]:-}}"
WHISPER_ENDPOINTS_VAL="${WHISPER_MODEL_ENDPOINTS:-${env_map[WHISPER_MODEL_ENDPOINTS]:-}}"
OCR_ENDPOINTS_VAL="${OCR_ENDPOINTS:-${env_map[OCR_ENDPOINTS]:-}}"

# Export originals for HAProxy containers
export SUPERVISOR_LLM_ENDPOINTS="$SUPERVISOR_ENDPOINTS"
export EMBEDDING_ENDPOINTS="$EMBEDDING_ENDPOINTS_VAL"
export WHISPER_MODEL_ENDPOINTS="$WHISPER_ENDPOINTS_VAL"
export OCR_ENDPOINTS="$OCR_ENDPOINTS_VAL"

# Also export as HAPROXY_* for compose to pass to haproxy containers
export HAPROXY_SUPERVISOR_ENDPOINTS="$SUPERVISOR_ENDPOINTS"
export HAPROXY_EMBEDDING_ENDPOINTS="$EMBEDDING_ENDPOINTS_VAL"
export HAPROXY_WHISPER_ENDPOINTS="$WHISPER_ENDPOINTS_VAL"
export HAPROXY_OCR_ENDPOINTS="$OCR_ENDPOINTS_VAL"

# Auto-override to haproxy URL when multi-endpoint detected
if [[ -n "$SUPERVISOR_ENDPOINTS" ]]; then
  count=$(echo "$SUPERVISOR_ENDPOINTS" | tr ',' '\n' | grep -c 'http' || true)
  if [[ "$count" -gt 0 ]]; then
    export SUPERVISOR_LLM_ENDPOINTS="http://haproxy_supervisor:11437/v1"
    echo "🔀 SUPERVISOR_LLM_ENDPOINTS detected ($count endpoints) → SUPERVISOR_LLM_ENDPOINTS=$SUPERVISOR_LLM_ENDPOINTS"
  fi
fi

if [[ -n "$EMBEDDING_ENDPOINTS_VAL" ]]; then
  count=$(echo "$EMBEDDING_ENDPOINTS_VAL" | tr ',' '\n' | grep -c 'http' || true)
  if [[ "$count" -gt 0 ]]; then
    export EMBEDDING_ENDPOINTS="http://haproxy_embd:11438/v1"
    echo "🔀 EMBEDDING_ENDPOINTS detected ($count endpoints) → EMBEDDING_ENDPOINTS=$EMBEDDING_ENDPOINTS"
  fi
fi

if [[ -n "$WHISPER_ENDPOINTS_VAL" ]]; then
  count=$(echo "$WHISPER_ENDPOINTS_VAL" | tr ',' '\n' | grep -c 'http' || true)
  if [[ "$count" -gt 0 ]]; then
    export WHISPER_MODEL_ENDPOINTS="http://haproxy_whisper:11439/inference"
    echo "🔀 WHISPER_MODEL_ENDPOINTS detected ($count endpoints) → WHISPER_MODEL_ENDPOINTS=$WHISPER_MODEL_ENDPOINTS"
  fi
fi

if [[ -n "$OCR_ENDPOINTS_VAL" ]]; then
  count=$(echo "$OCR_ENDPOINTS_VAL" | tr ',' '\n' | grep -c 'http' || true)
  if [[ "$count" -gt 0 ]]; then
    export OCR_ENDPOINTS="http://haproxy_ocr:11440/v1/convert/file"
    echo "🔀 OCR_ENDPOINTS detected ($count endpoints) → OCR_ENDPOINTS=$OCR_ENDPOINTS"
  fi
fi

######################################
# Function to validate and export   #
######################################
validate_var_path() {
  local var_name="$1"
  local required_type="$2"
  # Priority: Shell Environment > .env file
  local val="${!var_name:-${env_map[$var_name]:-}}"

  if [[ -z "$val" ]]; then
    if [[ "$required_type" == "any" ]]; then
        val="NOT_SET"
    else
        echo "❌ $var_name is not set in $ENV_FILE or environment"
        exit 1
    fi
  fi

  case "$required_type" in
    "gguf")
      if [[ ! "$val" =~ ^https?:// ]] && [[ ! -f "$val" ]]; then
          echo "❌ $var_name must be a valid file or URL"
          exit 1
      fi
      ;;
    "e5")
      if [[ ! "$val" =~ ^https?:// ]] && [[ ! -d "$val" ]]; then
        echo "❌ $var_name must be a directory or URL"
        exit 1
      fi
      ;;
    "dir")
      if [[ ! -d "$val" ]]; then
        echo "❌ $var_name must be a directory : $val"
        exit 1
      fi
      ;;
  esac
  
  local icon="✅"
  if [[ "$val" =~ ^https?:// ]]; then icon="📡";
  elif [[ "$val" == "LOCAL" || -f "$val" || -d "$val" ]]; then icon="🏠";
  elif [[ "$val" == "NOT_SET" ]]; then icon="⚠️"; fi
  
  echo "$icon $var_name is valid : $val"
  export "$var_name"="$val"
}

# Run validation
REMOTE_DUMMY_DIR="/tmp/rag_remote_mount"
mkdir -p "$REMOTE_DUMMY_DIR"

for entry in "${REQUIRED_VARS[@]}"; do
  var_name="${entry%%:*}"
  validation_type="${entry##*:}"
  validate_var_path "$var_name" "$validation_type"

  # Setup MOUNT variables for Compose
  val="${!var_name}"
  mount_var_name="${var_name}_MOUNT"
  if [[ "$val" =~ ^https?:// || "$val" == "NOT_SET" ]]; then
    export "$mount_var_name"="$REMOTE_DUMMY_DIR"
  elif [[ -f "$val" ]]; then
    export "$mount_var_name"="$(dirname "$val")"
  else
    export "$mount_var_name"="$val"
  fi
done

# WHISPER CONFIGURATION GUARD (FOR LOGS)
if [[ "$WHISPER_MODEL_ENDPOINTS" == "NOT_SET" ]]; then
  echo "⚠️  WARNING: WHISPER_MODEL_ENDPOINTS is not set. Media will fail."
fi

# 6. Lifecycle Path Exports (Anchored to Root)
export STAGING_DIR="${DEFAULT_DOC_INGEST_ROOT}/staging"
export PREPROCESSING_DIR="${DEFAULT_DOC_INGEST_ROOT}/preprocessing"
export INGESTION_DIR="${DEFAULT_DOC_INGEST_ROOT}/ingestion"
export CONSUMING_DIR="${DEFAULT_DOC_INGEST_ROOT}/consuming"
export SUCCESS_DIR="${DEFAULT_DOC_INGEST_ROOT}/success"
export FAILED_DIR="${DEFAULT_DOC_INGEST_ROOT}/failed"
export VECTOR_DB_DATA_DIR="${DEFAULT_DOC_INGEST_ROOT}/qdrant_data"

COMPOSE_FILE="ingest-dockercompose.yaml"

echo "🚀 Launching LifeCycle-Aware Stack"

if command -v docker-compose &> /dev/null; then
  docker-compose -f "$COMPOSE_FILE" $COMPOSE_FILE_OPTS up "$@"
else
  docker compose -f "$COMPOSE_FILE" $COMPOSE_FILE_OPTS up "$@"
fi
