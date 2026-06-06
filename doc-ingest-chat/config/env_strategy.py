import os
import sys

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.path.dirname(_app_dir)
for _candidate in (_app_dir, _repo_root):
    if os.path.isdir(os.path.join(_candidate, "shared")):
        sys.path.insert(0, _candidate)
        break


class GPUEnvConfig:
    def apply(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_env_strategy():
    return GPUEnvConfig()
