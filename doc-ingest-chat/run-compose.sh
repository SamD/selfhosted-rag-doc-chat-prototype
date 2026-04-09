#!/usr/bin/env bash
set -euo pipefail

#######################################
# ✅ REQUIRED ENVIRONMENT VARIABLES  #
#######################################
# LLM_PATH
#   - Must be a .gguf file that exists
#
# EMBEDDING_MODEL_PATH
#   - Must be a directory containing config.json or tokenizer_config.json
#
# INGEST_FOLDER
#   - Must be a valid directory where docs/html are stored
#
# These will be validated from either a .env file or the shell environment.
# After validation, they are exported for use in the Compose file.

#######################################
# ✅ DOCKER COMPOSE CONFIGURATION    #
#######################################
# The Compose file used and the default profiles to enable
#
# Compose file must exist in the same directory as this script.
#######################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

export COMPOSE_BAKE=true

# Set deterministic temperature by default
export LLAMA_TEMPERATURE='0.1'

# redis host name since this docker-compose service
export REDIS_HOST=redis
export REDIS_PORT=6380

COMPOSE_FILE="ingest-dockercompose.yaml"

if [[ -z "${GPU_CPU_PROFILE:-}" ]]; then
  # default to gpu
  GPU_CPU_PROFILE="cuda"
fi

if [[ -z "${VECTOR_DB_PROFILE:-}" ]]; then
  # default to qdrant instead of chroma
  VECTOR_DB_PROFILE="qdrant"
fi

# Export VECTOR_DB_PROFILE so containers can see it
export VECTOR_DB_PROFILE

# Combine GPU_CPU_PROFILE and VECTOR_DB_PROFILE for services that need combined profile
# (e.g., consumer_worker needs "cuda-qdrant" not separate "cuda" and "qdrant")
COMBINED_PROFILE="${GPU_CPU_PROFILE}-${VECTOR_DB_PROFILE}"

# Profiles needed:
# - COMBINED_PROFILE: for consumer_worker, api services (e.g., cuda-qdrant)
# - GPU_CPU_PROFILE: for ocr_worker, producer_worker (e.g., cuda)
# - VECTOR_DB_PROFILE: for vector db services (e.g., qdrant)
# - with-frontend: for frontend service
COMPOSE_FILE_OPTS="--profile $COMBINED_PROFILE --profile $GPU_CPU_PROFILE --profile $VECTOR_DB_PROFILE --profile with-frontend"

REQUIRED_VARS=(
  "LLM_PATH:gguf"
  "SUPERVISOR_LLM_PATH:gguf"
  "EMBEDDING_MODEL_PATH:e5"
  "INGEST_FOLDER:dir"
  "STAGING_FOLDER:dir"
)

###################################
# Setup script directory and env #
###################################
cd "$SCRIPT_DIR"
echo "📁 Working directory: $SCRIPT_DIR"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "❌ ERROR: Compose file '$COMPOSE_FILE' not found"
  exit 1
else
  echo "✅ Found compose file: $COMPOSE_FILE"
fi

# ENV_FILE="${1:-.env}"
ENV_FILE="ingest-svc.env"
declare -A env_map

if [[ -f "$ENV_FILE" ]]; then
  echo "🔄 Loading variables from $ENV_FILE"
  while IFS='=' read -r key val; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    env_map["$key"]="$val"
  done < <(grep -Ev '^\s*#|^\s*$' "$ENV_FILE")
else
  echo "⚠️ No .env file found at $ENV_FILE"
fi

######################################
# Function to validate and export   #
######################################
validate_var_path() {
  local var_name="$1"
  local required_type="$2"
  local val="${env_map[$var_name]:-${!var_name:-}}"

  echo "🔍 Checking $var_name..."

  if [[ -z "$val" ]]; then
    echo "❌ $var_name is not set in .env or environment"
    exit 1
  fi

  echo "   → Value: $val"

  case "$required_type" in
    "gguf")
      if [[ ! -f "$val" ]]; then
        echo "❌ $var_name must be a valid file"
        exit 1
      elif [[ "${val##*.}" != "gguf" ]]; then
        echo "❌ $var_name must end in .gguf"
        exit 1
      fi
      ;;
    "e5")
      if [[ ! -d "$val" ]]; then
        echo "❌ $var_name must be a directory"
        exit 1
      elif [[ ! -f "$val/config.json" && ! -f "$val/tokenizer_config.json" ]]; then
        echo "❌ $var_name must contain config.json or tokenizer_config.json"
        exit 1
      fi
      ;;
    "dir")
      if [[ ! -d "$val" ]]; then
        echo "❌ $var_name must be a directory"
        exit 1
      fi
      ;;
    *)
      echo "❌ Unknown validation type: $required_type"
      exit 1
      ;;
  esac

  echo "✅ $var_name is valid"
  export "$var_name"="$val"
}

#################################
# Validate and export all vars #
#################################
for entry in "${REQUIRED_VARS[@]}"; do
  var_name="${entry%%:*}"
  validation_type="${entry##*:}"
  validate_var_path "$var_name" "$validation_type"
done

# NOW that INGEST_FOLDER is validated and exported, we can set data dirs
export CHROMA_DATA_DIR="${INGEST_FOLDER}/chroma_db"
export QDRANT_DATA_DIR="${INGEST_FOLDER}/qdrant_data"
export VECTOR_DB_DATA_DIR="${INGEST_FOLDER}/qdrant_data" # Ensure it matches docker-compose.yaml

#################################
# Launch Docker Compose         #
#################################
echo "🚀 Launching Docker Compose with profiles: $COMPOSE_FILE_OPTS"

if command -v docker-compose &> /dev/null; then
  echo "💡 Using docker-compose (v2.36.2)"
  docker-compose -f "$COMPOSE_FILE" $COMPOSE_FILE_OPTS up --build "$@"
elif docker compose version &> /dev/null; then
  echo "💡 Using docker compose"
  docker compose -f "$COMPOSE_FILE" $COMPOSE_FILE_OPTS up --build "$@"
else
  echo "❌ ERROR: Neither docker-compose nor docker compose found"
  exit 1
fi
