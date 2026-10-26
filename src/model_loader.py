import argparse
import gc
import os
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
DOLOS_REPO = 'Poseidon-Research/Dolos-1.5B'
DOLOS_REVISION = '63e7031a0535a49f86e08af4e7bdc45cfa22fe6d'
QWEN_REPO = 'Qwen/Qwen2.5-1.5B-Instruct'
QWEN_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'

def load_model(model_name: str, device: torch.device, dtype: torch.dtype, weights: Path | None=None):
    """Load weights with the public Qwen architecture and restore tied embeddings."""
    repo, revision = (DOLOS_REPO, DOLOS_REVISION) if model_name == 'dolos' else (QWEN_REPO, QWEN_REVISION)
    if weights is None or model_name == 'qwen':
        weights = Path(hf_hub_download(repo, 'model.safetensors', revision=revision, token=False))
    weights = weights.expanduser().absolute()
    if not weights.is_file() or weights.name != 'model.safetensors':
        raise ValueError('--weights must point to a file named model.safetensors.')
    config = AutoConfig.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    model, info = AutoModelForCausalLM.from_pretrained(weights.parent, config=config, generation_config=GenerationConfig.from_model_config(config), dtype=dtype, attn_implementation='sdpa', use_safetensors=True, local_files_only=True, output_loading_info=True)
    problems = {key: info[key] for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs') if info.get(key)}
    if problems:
        raise RuntimeError(f'The weights did not load completely: {problems}')
    model.tie_weights()
    return model.to(device).eval()
