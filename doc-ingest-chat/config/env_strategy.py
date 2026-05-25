import os
import sys

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.path.dirname(_app_dir)
for _candidate in (_app_dir, _repo_root):
    if os.path.isdir(os.path.join(_candidate, "shared")):
        sys.path.insert(0, _candidate)
        break

from shared.defaults import DEFAULT_LLAMA_USE_GPU  # noqa: E402
from shared.env_names import ENV_LLAMA_USE_GPU  # noqa: E402


class BaseEnvConfig:
    def apply(self):
        raise NotImplementedError()


class CPUEnvConfig(BaseEnvConfig):
    def apply(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


class GPUEnvConfig(BaseEnvConfig):
    def apply(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_env_strategy():
    use_gpu = os.getenv(ENV_LLAMA_USE_GPU, DEFAULT_LLAMA_USE_GPU).lower() == "true"
    return GPUEnvConfig() if use_gpu else CPUEnvConfig()
