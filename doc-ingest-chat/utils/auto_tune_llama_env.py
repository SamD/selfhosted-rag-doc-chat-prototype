#!/usr/bin/env python3

"""
Auto-tune LLaMA config and write to .env file
"""

import argparse
import os
import sys

import psutil
from logging_config import setup_logging

log = setup_logging("auto_tune_llama.log")


def get_cpu_ram_gb():
    return psutil.virtual_memory().total / 1e9


def get_gpu_info():
    try:
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory
            return mem / 1e9  # in GB
    except ImportError:
        pass
    return None


def suggest_llama_config():
    ram_gb = get_cpu_ram_gb()
    gpu_gb = get_gpu_info()

    config = {
        "LLAMA_N_CTX": 2048,
        "LLAMA_N_BATCH": 256,
        "LLAMA_N_GPU_LAYERS": 0,
    }

    if gpu_gb:
        if gpu_gb >= 24:
            config["LLAMA_N_CTX"] = 8192
            config["LLAMA_N_BATCH"] = 512
            config["LLAMA_N_GPU_LAYERS"] = 35
        elif gpu_gb >= 16:
            config["LLAMA_N_CTX"] = 4096
            config["LLAMA_N_BATCH"] = 384
            config["LLAMA_N_GPU_LAYERS"] = 20
        elif gpu_gb >= 8:
            config["LLAMA_N_CTX"] = 2048
            config["LLAMA_N_BATCH"] = 256
            config["LLAMA_N_GPU_LAYERS"] = 10
        else:
            config["LLAMA_N_CTX"] = 2048
            config["LLAMA_N_BATCH"] = 128
            config["LLAMA_N_GPU_LAYERS"] = 0
    else:
        if ram_gb >= 64:
            config["LLAMA_N_CTX"] = 4096
            config["LLAMA_N_BATCH"] = 256
        elif ram_gb >= 32:
            config["LLAMA_N_CTX"] = 2048
            config["LLAMA_N_BATCH"] = 128
        else:
            config["LLAMA_N_CTX"] = 1024
            config["LLAMA_N_BATCH"] = 64

    config["LLAMA_N_THREADS"] = psutil.cpu_count(logical=False)
    return config


def validate_env_path(path):
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)

    if not os.path.exists(parent):
        log.error(f"❌ Error: Parent directory does not exist: {parent}")
        sys.exit(1)

    if os.path.isdir(abs_path):
        log.error(f"❌ Error: Provided path is a directory, not a file: {abs_path}")
        sys.exit(1)

    return abs_path


def write_to_env_file(config, env_path):
    lines = [f"{key}={value}" for key, value in config.items()]
    content = "\n".join(lines)

    if os.path.exists(env_path):
        backup_path = env_path + ".bak"
        os.rename(env_path, backup_path)
        log.info(f"🔁 Backed up existing env file to: {backup_path}")

    with open(env_path, "w") as f:
        f.write(content + "\n")

    log.info(f"✅ Wrote optimized LLaMA config to: {env_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-tune LLaMA config and write to .env file")
    parser.add_argument("env_file", help="Full path to .env file to write")
    args = parser.parse_args()

    validated_path = validate_env_path(args.env_file)
    config = suggest_llama_config()
    write_to_env_file(config, validated_path)
