"""Sequential 16-token likelihood decoder; truth is not an input."""
import numpy as np
import torch
import common as c
from common import tok, models, prompt
alphabet=c.HEX
@torch.inference_mode()
def decode_blocks(task, ids, name='dolos', template=c.ORIGINAL_TEMPLATE):
    dummy = '0123456789abcdef0123456789abcdef'
    pred = ''
    blocks = []
    for j, start in enumerate(range(0, min(512, len(ids)), 16)):
        end = min(start + 16, len(ids))
        outids = ids[:end]
        kk = []
        for c in alphabet:
            k = pred + c + dummy[j + 1:]
            kk.append(tok.encode(prompt(task, k, template), add_special_tokens=False) + outids)
        scores = []
        for s in range(0, 16, 4):
            enc = tok.pad({'input_ids': kk[s:s + 4]}, padding=True, return_tensors='pt').to('cuda')
            lp = models[name](**enc, logits_to_keep=end - start + 1).logits[:, :-1].float().log_softmax(-1)
            target = torch.tensor(ids[start:end], device='cuda')[None, :, None].expand(min(4, len(kk) - s), -1, -1)
            scores.extend(lp.gather(-1, target).squeeze(-1).sum(-1).cpu().tolist())
        pred += alphabet[int(np.argmax(scores))]
        blocks.append(scores)
    return (pred, blocks)
