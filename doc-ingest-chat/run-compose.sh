#!/usr/bin/env bash
set -euo pipefail

#######################################
# ✅ REQUIRED ENVIRONMENT VARIABLES  #
#######################################
# DEFAULT_DOC_INGEST_ROOT
#   - The parent directory for all lifecycle stages
#
# LLM_PATH
#   - Must be a .gguf file or remote URL
#
# EMBEDDING_MODEL_PATH
#   - Must be a directory containing config.json or tokenizer_config.json
#
#######################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

export COMPOSE_BAKE=true
export LLAMA_TEMPERATURE='0.1'

# redis host name since this docker-compose service
export REDIS_HOST=redis
export REDIS_PORT=6379

COMPOSE_FILE="ingest-dockercompose.yaml"

if [[ -z "${GPU_CPU_PROFILE:-}" ]]; then
  GPU_CPU_PROFILE="cuda"
fi

if [[ -z "${VECTOR_DB_PROFILE:-}" ]]; then
  VECTOR_DB_PROFILE="qdrant"
fi

export VECTOR_DB_PROFILE
COMBINED_PROFILE="${GPU_CPU_PROFILE}-${VECTOR_DB_PROFILE}"
COMPOSE_FILE_OPTS="--profile $COMBINED_PROFILE --profile $GPU_CPU_PROFILE --profile $VECTOR_DB_PROFILE --profile with-frontend"

REQUIRED_VARS=(
  "DEFAULT_DOC_INGEST_ROOT:dir"
  "LLM_PATH:gguf"
  "SUPERVISOR_LLM_PATH:gguf"
  "EMBEDDING_MODEL_PATH:e5"
)

###################################
# Setup script directory and env #
###################################
cd "$SCRIPT_DIR"
ENV_FILE="ingest-svc.env"
declare -A env_map

if [[ -f "$ENV_FILE" ]]; then
  echo "🔄 Loading variables from $ENV_FILE"
  while IFS='=' read -r key val; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    env_map["$key"]="$val"
  done < <(grep -Ev '^\s*#|^\s*$' "$ENV_FILE")
fi

######################################
# Function to validate and export   #
######################################
validate_var_path() {
  local var_name="$1"
  local required_type="$2"
  local val="${env_map[$var_name]:-${!var_name:-}}"

  if [[ -z "$val" ]]; then
    echo "❌ $var_name is not set in $ENV_FILE or environment"
    exit 1
  fi

  case "$required_type" in
    "gguf")
      if [[ ! "$val" =~ ^https?:// ]] && [[ ! -f "$val" ]]; then
          echo "❌ $var_name must be a valid file or URL"
          exit 1
      fi
      ;;
    "e5")
      if [[ ! -d "$val" ]]; then
        echo "❌ $var_name must be a directory"
        exit 1
      fi
      ;;
    "dir")
      if [[ ! -d "$val" ]]; then
        echo "❌ $var_name must be a directory"
        exit 1
      fi
      ;;
  esac
  export "$var_name"="$val"
}

#################################
# Validate and export all vars #
#################################
DUMMY_MOUNT="/tmp/llm_remote_mount.gguf"
touch "$DUMMY_MOUNT"

for entry in "${REQUIRED_VARS[@]}"; do
  var_name="${entry%%:*}"
  validation_type="${entry##*:}"
  validate_var_path "$var_name" "$validation_type"

  val="${!var_name}"
  mount_var_name="${var_name}_MOUNT"
  if [[ "$val" =~ ^https?:// ]]; then
    export "$mount_var_name"="$DUMMY_MOUNT"
  else
    export "$mount_var_name"="$val"
  fi
done

# Anchored paths for the Compose volumes
export STAGING_DIR="${DEFAULT_DOC_INGEST_ROOT}/staging"
export PREPROCESSING_DIR="${DEFAULT_DOC_INGEST_ROOT}/preprocessing"
export INGESTION_DIR="${DEFAULT_DOC_INGEST_ROOT}/ingestion"
export CONSUMING_DIR="${DEFAULT_DOC_INGEST_ROOT}/consuming"
export SUCCESS_DIR="${DEFAULT_DOC_INGEST_ROOT}/success"
export FAILED_DIR="${DEFAULT_DOC_INGEST_ROOT}/failed"
export VECTOR_DB_DATA_DIR="${DEFAULT_DOC_INGEST_ROOT}/qdrant_data"

#################################
# Launch Docker Compose         #
#################################
echo "🚀 Launching LifeCycle-Aware Stack"

if command -v docker-compose &> /dev/null; then
  docker-compose -f "$COMPOSE_FILE" $COMPOSE_FILE_OPTS up "$@"
else
  docker compose -f "$COMPOSE_FILE" $COMPOSE_FILE_OPTS up "$@"
fi
