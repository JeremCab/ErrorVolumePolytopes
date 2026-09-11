"""INVARIANT : les poids raccourcis doivent redonner exactement les logits.
MLP et CNN ont DEUX implementations distinctes — on teste les deux."""
import sys; sys.path.insert(0,'.')
import torch
from src.models.networks import FashionCNN_Small, FashionMLP_Large
torch.manual_seed(0)

def report(tag, pred, true):
    e = (pred - true).abs().max().item()
    rel = e / true.abs().max().item()
    flag = "  <<< INCOHERENT" if rel > 1e-4 else ""
    print(f"  {tag:<30} err_max={e:.3e}  rel={rel:.2e}{flag}", flush=True)
    return rel

# ---------------- MLP ----------------
print("=========== MLP (temoin) ===========", flush=True)
from src.shortcuts.shortcut_weights import compute_shortcut_weights
m = FashionMLP_Large(); m.load_state_dict(torch.load("checkpoints/fashion_mlp_best.pth",
      map_location="cpu", weights_only=True)); m.eval()
ds = torch.load("data/fashionMNIST_correct_mlp.pt", weights_only=False)
x, c = ds[0]; xf = x.flatten()
W_l, B_l, _ = compute_shortcut_weights(m, x)   # forme d'origine (le MLP a un Flatten)
W, B = W_l[-1], B_l[-1]
with torch.no_grad(): true = m(x.unsqueeze(0)).squeeze(0)
report("x0", W @ xf + B, true)
from src.optim.build_polytopes import build_all_polytopes_per_class
from src.quantization.quantize import quantize_model
q = quantize_model(m, bits=16).eval()
_, _, poly = build_all_polytopes_per_class(m, {16: q}, x.unsqueeze(0), int(c))
A, b, _ = poly[16]
sc = 1e-2; found = False
for _ in range(80):
    d = torch.randn_like(xf); d /= d.norm(); y = (xf + sc*d).clamp(-1, 1)
    if (A @ y + b).max() <= 0: found = True; break
    sc *= 0.7
if found:
    with torch.no_grad(): t2 = m(y.reshape(x.shape).unsqueeze(0)).squeeze(0)
    report(f"point perturbe (|d|={sc:.1e})", W @ y + B, t2)
else: print("  (aucun point perturbe trouve)", flush=True)

# ---------------- CNN ----------------
print("\n=========== CNN (suspect) ===========", flush=True)
from src.optim.build_polytopes_cnn import (_collect, _Constraints, _Message,
                                           model_to_sequential,
                                           build_cnn_all_polytopes_per_class)
n = FashionCNN_Small(); n.load_state_dict(torch.load("checkpoints/fashion_cnn_best.pth",
      map_location="cpu", weights_only=True)); n.eval()
ds = torch.load("data/fashionMNIST_correct_cnn.pt", weights_only=False)
x, c = ds[0]; c = int(c)
seq = model_to_sequential(n)
cst = _Constraints(); msg = _Message(x)
with torch.no_grad(): msg = _collect(seq, msg, cst)
D = x.numel()
Wc = msg.s_weight.reshape(D, -1)        # (784, 10)
Bc = msg.s_bias.flatten()
xf = x.flatten()
with torch.no_grad(): true = n(x.unsqueeze(0)).squeeze(0)
print(f"  s_weight {tuple(msg.s_weight.shape)} -> ({D}, {Wc.shape[1]})   s_bias {tuple(msg.s_bias.shape)}", flush=True)
report("x0", xf @ Wc + Bc, true)
q = quantize_model(n, bits=16).eval()
_, _, poly = build_cnn_all_polytopes_per_class(n, {16: q}, x.unsqueeze(0), c)
A, b, _ = poly[16]
sc = 1e-2; found = False
for _ in range(80):
    d = torch.randn_like(xf); d /= d.norm(); y = (xf + sc*d).clamp(-1, 1)
    if (A @ y + b).max() <= 0: found = True; break
    sc *= 0.7
if found:
    with torch.no_grad(): t2 = n(y.reshape(x.shape).unsqueeze(0)).squeeze(0)
    report(f"point perturbe (|d|={sc:.1e})", y @ Wc + Bc, t2)
else: print("  (aucun point perturbe trouve)", flush=True)
