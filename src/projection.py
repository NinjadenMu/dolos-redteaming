"""Small reusable intervention for the pinned Dolos Qwen2.5 architecture."""
from contextlib import contextmanager
import torch

@contextmanager
def erase_readout(model, basis_path, rank=8):
    """Mean-ablate a fitted subspace after final attention's output projection.

    Apply during both prefill and decoding. Does not modify weights or the KV
    cache, and always removes the hook on exit. This suppresses a measured
    channel; it is not a confidentiality guarantee.
    """
    basis=torch.load(basis_path,map_location='cpu',weights_only=True)
    module=model.model.layers[27].self_attn.o_proj
    device=module.weight.device
    v=basis['directions'][:rank].to(device=device,dtype=torch.float32)
    mean=basis['mean'].to(device=device,dtype=torch.float32)
    assert v.shape==(rank,1536)
    def erase(_module,_inputs,output):
        x=output.float()
        return (x-((x-mean)@v.T)@v).to(output.dtype)
    handle=module.register_forward_hook(erase)
    try:yield
    finally:handle.remove()
