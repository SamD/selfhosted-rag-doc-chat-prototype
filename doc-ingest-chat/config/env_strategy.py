import os


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
    use_gpu = os.getenv("LLAMA_USE_GPU", "true").lower() == "true"
    return GPUEnvConfig() if use_gpu else CPUEnvConfig()
