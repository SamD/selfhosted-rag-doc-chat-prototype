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
# INGEST_FOLDER: Directory inside the container where files are read for ingestion. Must match the right side of the data volume mount.
# INGEST_FOLDER=.
# export INGEST_FOLDER=/home/myname/Projects/selfhosted-rag-doc-chat-prototype/Docs

# Model Paths
# EMBEDDING_MODEL_PATH: Path inside the container to the E5 model directory. Must match the right side of the E5 model volume mount.
# Only tested with e5-large-v2
# https://huggingface.co/intfloat/e5-large-v2/blob/main/model.safetensors
# export EMBEDDING_MODEL_PATH=/home/myname/AI/models/e5-large-v2

# LLM_PATH: Path inside the container to the Llama model file. Must match the right side of the Llama model volume mount.
# Only tested with Meta-LLama
# https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf
# export LLM_PATH=/home/myname/AI/models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf

# This is the path chroma will use for its files
# CHROMA_DATA_DIR=/home/myname/Projects/selfhosted-rag-doc-chat-prototype/Docs


# curl -LsS https://archive.org/download/outlineofhistory01welluoft/outlineofhistory01welluoft.pdf -o outline_of_history_pt1.pdf
# curl -LsS https://archive.org/download/outlineofhistory02welluoft/outlineofhistory02welluoft.pdf -o outline_of_history_pt2.pdf


export INGEST_FOLDER=/home/samueldoyle/Projects/GitHub/SamD/selfhosted-rag-doc-chat-prototype/Docs
export EMBEDDING_MODEL_PATH=/home/samueldoyle/AI/e5-large-v2
export LLM_PATH=/home/samueldoyle/AI/models/Phi/Phi-3.5-mini-instruct-Q4_K_M.gguf

export CHROMA_DATA_DIR=${INGEST_FOLDER}/chroma_db
export QDRANT_DATA_DIR=${INGEST_FOLDER}/qdrant_data

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

#: "${INGEST_FOLDER:-}" || INGEST_FOLDER="${SCRIPT_DIR}/Docs"
#if [[ -z "${INGEST_FOLDER:-}" ]]; then
#  INGEST_FOLDER="${SCRIPT_DIR}/Docs"
#fi

echo "✅ INGEST_FOLDER: $INGEST_FOLDER"

REQUIRED_VARS=(
  "LLM_PATH:gguf"
  "EMBEDDING_MODEL_PATH:e5"
  "INGEST_FOLDER:dir"
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

#################################
# Launch Docker Compose         #
#################################
echo "🚀 Launching Docker Compose with profiles: $COMPOSE_FILE_OPTS"

if command -v docker-compose &> /dev/null; then
  echo "💡 Using docker-compose (v2.36.2)"
  exec docker-compose -f "$COMPOSE_FILE" $COMPOSE_FILE_OPTS up --build "$@"
elif docker compose version &> /dev/null; then
  echo "💡 Using docker compose"
  exec docker compose -f "$COMPOSE_FILE" $COMPOSE_FILE_OPTS up --build "$@"
else
  echo "❌ ERROR: Neither docker-compose nor docker compose found"
  exit 1
fi

