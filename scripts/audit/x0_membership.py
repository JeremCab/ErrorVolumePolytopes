"""x0 appartient-il a son propre polytope ? C'est l'invariant le plus basique :
P2 est par construction la region de linearite CONTENANT x0."""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from src.models.networks import FashionCNN_Small, FashionMLP_Large
from src.quantization.quantize import quantize_model
from src.optim.build_polytopes import build_all_polytopes_per_class
from src.optim.build_polytopes_cnn import build_cnn_all_polytopes_per_class

def diag(tag, A, b, xf):
    v = (A @ xf + b)
    viol = v[v > 0]
    print(f"  {tag:<22} {A.shape[0]:>6} contraintes | "
          f"violees : {len(viol):>5} | max = {v.max().item():+.3e} | "
          f"pire violation = {viol.max().item() if len(viol) else 0.0:+.3e}", flush=True)
    return v

print("=========== MLP ===========", flush=True)
m = FashionMLP_Large(); m.load_state_dict(torch.load("checkpoints/fashion_mlp_best.pth",
      map_location="cpu", weights_only=True)); m.eval()
ds = torch.load("data/fashionMNIST_correct_mlp.pt", weights_only=False)
for i in (0, 1, 2):
    x, c = ds[i]; c = int(c); xf = x.flatten()
    q = quantize_model(m, bits=16).eval()
    A0, b0, poly = build_all_polytopes_per_class(m, {16: q}, x.unsqueeze(0), c)
    A, b, per = poly[16]
    diag(f"sample {i} — P2", A, b, xf)
    diag(f"sample {i} — P3^c", *per[c], xf)

print("\n=========== CNN ===========", flush=True)
n = FashionCNN_Small(); n.load_state_dict(torch.load("checkpoints/fashion_cnn_best.pth",
      map_location="cpu", weights_only=True)); n.eval()
ds = torch.load("data/fashionMNIST_correct_cnn.pt", weights_only=False)
for i in (0, 1, 2):
    x, c = ds[i]; c = int(c); xf = x.flatten()
    q = quantize_model(n, bits=16).eval()
    A0, b0, poly = build_cnn_all_polytopes_per_class(n, {16: q}, x.unsqueeze(0), c)
    A, b, per = poly[16]
    diag(f"sample {i} — P2", A, b, xf)
    diag(f"sample {i} — P3^c", *per[c], xf)
