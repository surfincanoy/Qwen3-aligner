import os
import gc

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "Qwen")


def _get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_asr_model(model_name: str = "Qwen/Qwen3-ASR-0.6B"):
    from qwen_asr import Qwen3ASRModel

    _clear_cache()
    device = _get_device()
    print(f"Using device: {device}")

    folder_name = model_name.split("/")[-1]
    local_model_dir = os.path.join(MODELS_DIR, folder_name)

    if os.path.isdir(local_model_dir):
        print(f"Loading local model: {local_model_dir}")
        try:
            model = Qwen3ASRModel.from_pretrained(
                local_model_dir, dtype=torch.bfloat16, device_map=device,
                max_inference_batch_size=32, max_new_tokens=1024,
            )
            print("Local model loaded")
            return model
        except Exception as e:
            print(f"Local load failed: {e}")

    try:
        print(f"Downloading from HuggingFace: {model_name}")
        model = Qwen3ASRModel.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=device,
            max_inference_batch_size=32, max_new_tokens=1024,
        )
        print("HuggingFace model loaded")
        try:
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(local_model_dir)
        except Exception as e:
            print(f"Save model failed: {e}")
        return model
    except Exception as e:
        print(f"HuggingFace failed: {e}")

    try:
        from modelscope import snapshot_download
        ms_id = model_name.replace("Qwen/", "qwen/")
        print(f"Downloading from ModelScope: {ms_id}")
        snapshot_download(ms_id, local_dir=local_model_dir)
        model = Qwen3ASRModel.from_pretrained(
            local_model_dir, dtype=torch.bfloat16, device_map=device,
            max_inference_batch_size=32, max_new_tokens=1024,
        )
        print("ModelScope model loaded")
        return model
    except ImportError:
        print("modelscope not installed")
    except Exception as e:
        print(f"ModelScope failed: {e}")

    raise RuntimeError("All model loading methods failed")


def load_aligner_model():
    from qwen_asr import Qwen3ForcedAligner

    _clear_cache()
    device = _get_device()
    print(f"Using device: {device}")

    local_model_dir = os.path.join(MODELS_DIR, "Qwen3-ForcedAligner-0.6B")

    if os.path.isdir(local_model_dir):
        print(f"Loading local aligner: {local_model_dir}")
        try:
            model = Qwen3ForcedAligner.from_pretrained(local_model_dir, dtype=torch.bfloat16, device_map=device)
            print("Local aligner loaded")
            return model
        except Exception as e:
            print(f"Local aligner load failed: {e}")

    hf_model_id = "Qwen/Qwen3-ForcedAligner-0.6B"
    try:
        print(f"Downloading aligner from HuggingFace: {hf_model_id}")
        model = Qwen3ForcedAligner.from_pretrained(hf_model_id, dtype=torch.bfloat16, device_map=device)
        print("HuggingFace aligner loaded")
        print(f"Saving aligner to local: {local_model_dir}")
        model.save_pretrained(local_model_dir)
        return model
    except Exception as e:
        print(f"HuggingFace aligner failed: {e}")

    try:
        from modelscope import snapshot_download
        print("Downloading aligner from ModelScope...")
        snapshot_download("qwen/Qwen3-ForcedAligner-0.6B", local_dir=local_model_dir)
        model = Qwen3ForcedAligner.from_pretrained(local_model_dir, dtype=torch.bfloat16, device_map=device)
        print("ModelScope aligner loaded")
        return model
    except ImportError:
        print("modelscope not installed")
    except Exception as e:
        print(f"ModelScope aligner failed: {e}")

    raise RuntimeError("All aligner model loading methods failed")
