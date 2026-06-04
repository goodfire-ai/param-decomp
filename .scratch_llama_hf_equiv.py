"""Validate VendoredLlama.from_hf_pretrained == HF LlamaForCausalLM logits (fp32, GPU)."""

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from param_decomp.components import make_components
from param_decomp_lab.experiments.lm.vendored.llama_3_1.components import componentize_llama
from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama

NAME = "meta-llama/Llama-3.1-8B"
dev = "cuda"
tok = AutoTokenizer.from_pretrained(NAME)
text = "The capital of France is Paris, and the capital of Japan is"
ids = tok(text, return_tensors="pt").input_ids.to(dev)
print("input ids:", ids.shape)

vend = VendoredLlama.from_hf_pretrained(NAME).to(dev).eval()
with torch.no_grad():
    lv = vend(ids).float()

hf = LlamaForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32).to(dev).eval()
with torch.no_grad():
    lh = hf(ids).logits.float()

d = (lv - lh).abs()
print(
    f"max|Δ|={d.max().item():.3e}  mean|Δ|={d.mean().item():.3e}  "
    f"rel={d.max().item() / lh.abs().max().item():.3e}"
)
print("argmax agreement:", (lv.argmax(-1) == lh.argmax(-1)).float().mean().item())
print("vend next-token:", tok.decode(lv[0, -1].argmax()), "| hf:", tok.decode(lh[0, -1].argmax()))
del hf
torch.cuda.empty_cache()

# componentized clean forward (identity decomposition is NOT identity; just check the
# no-mask target path still matches HF after componentize)
targets = {f"layers.18.mlp.{p}": 4096 for p in ("gate_proj", "up_proj", "down_proj")}
comps = make_components(vend, targets)
cm = componentize_llama(vend, comps)
with torch.no_grad():
    lc = cm(ids).float()
print("componentized no-mask max|Δ| vs HF:", (lc - lh).abs().max().item())
