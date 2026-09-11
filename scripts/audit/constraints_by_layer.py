"""D'ou viennent les contraintes saturees du CNN ? On separe par couche :
ReLU (actifs / satures) et max-pool."""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from src.models.networks import FashionCNN_Small
from src.optim.build_polytopes_cnn import (_collect, _Constraints, _Message,
                                           model_to_sequential)
n = FashionCNN_Small(); n.load_state_dict(torch.load("checkpoints/fashion_cnn_best.pth",
      map_location="cpu", weights_only=True)); n.eval()
ds = torch.load("data/fashionMNIST_correct_cnn.pt", weights_only=False)
x, c = ds[0]
seq = model_to_sequential(n)
print("couches de la version sequentielle :", flush=True)
for i, mod in enumerate(seq): print(f"   [{i}] {type(mod).__name__}", flush=True)

cst = _Constraints(); msg = _Message(x)
with torch.no_grad(): msg = _collect(seq, msg, cst)
xf = x.flatten()

def stats(tag, W, B):
    if W.numel() == 0: print(f"  {tag:<26} (vide)", flush=True); return
    W = W.reshape(W.shape[0], -1) if W.dim() > 2 else W
    nrm = W.norm(dim=1)
    zero = (nrm <= 1e-10).sum().item()
    # convention du code : U -> Wx+b >= 0 ; S -> Wx+b <= 0
    val = W @ xf + B
    print(f"  {tag:<26} {W.shape[0]:>6} lignes | norme nulle : {zero:>6} "
          f"| |Wx+b| median = {val.abs().median().item():.2e} "
          f"| serrees(<1e-6) : {(val.abs()<1e-6).sum().item():>6}", flush=True)

print("\nContraintes NON SATUREES (ReLU actifs, Wx+b >= 0) :", flush=True)
for i, (W, B) in enumerate(zip(cst.U_weight, cst.U_bias)):
    stats(f"U[{i}]", W, B)
print("\nContraintes SATUREES (ReLU inactifs + max-pool, Wx+b <= 0) :", flush=True)
for i, (W, B) in enumerate(zip(cst.S_weight, cst.S_bias)):
    stats(f"S[{i}]", W, B)
