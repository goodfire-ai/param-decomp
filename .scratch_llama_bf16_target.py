"""Does casting the frozen target to bf16 matter? Compare clean-forward logits:
fp32-weights-under-bf16-autocast (current) vs bf16-weights (proposed), on real 8B."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama

dev = "cuda"
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
ids = tok(
    "The capital of France is Paris, and the largest planet in the solar system is",
    return_tensors="pt",
).input_ids.to(dev)

m = VendoredLlama.from_hf_pretrained("meta-llama/Llama-3.1-8B").to(dev).eval()

with torch.no_grad():
    fp32_pure = m(ids).float()  # fp32 weights, fp32 math
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fp32_autocast = m(ids).float()  # CURRENT: fp32 wts + bf16 autocast

m_bf16 = m.to(torch.bfloat16)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    bf16_wts = m_bf16(ids).float()  # PROPOSED: bf16 weights


def cmp(name, a, b):
    d = (a - b).abs()
    p = F.log_softmax(a, -1)
    q = F.log_softmax(b, -1)
    kl = (p.exp() * (p - q)).sum(-1).mean().item()  # KL(a||b) per token, nats
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"{name:34s} max|Δ|={d.max():.3e}  KL={kl:.3e} nats  argmax_agree={agree:.3f}")


print("reference = fp32 pure-math forward")
cmp("fp32+autocast (CURRENT) vs fp32", fp32_autocast, fp32_pure)
cmp("bf16 weights (PROPOSED) vs fp32", bf16_wts, fp32_pure)
cmp("bf16 weights vs fp32+autocast", bf16_wts, fp32_autocast)
print(
    "next-token:",
    "fp32",
    repr(tok.decode(fp32_pure[0, -1].argmax())),
    "| bf16",
    repr(tok.decode(bf16_wts[0, -1].argmax())),
)
