"""Is the 98.35%-dense readout actually LOW-RANK?

If W = w_dend_out (8192 x 11520) has a small effective rank, then:
  - storing/factoring it is the real win (not CSR), and
  - it reveals what the dendritic readout actually learned.
Cheap test: SVD on a random column subset (columns = tokens), plus
rank-reconstruction error at several ranks.
"""
import torch
p = r"D:\Ziwago18-storage\Produce\checkpoints\ziwago_v19_bankfixed_20261109.pt"
obj = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
crs = obj["core_runtime_state"]["core"]["dendritic_plateau_readout"]["attributes"]
W = crs["w_dend_out"]

torch.manual_seed(11)
cols = torch.randperm(W.shape[1])[:1024]
sub = W[:, cols].float()          # (8192, 1024)
print("sample:", tuple(sub.shape))

U, S, Vh = torch.linalg.svd(sub, full_matrices=False)
energy = (S ** 2)
tot = float(energy.sum())
cum = torch.cumsum(energy, 0) / tot
print("energy captured by top-k singular directions:")
for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    print("   k=%4d -> %.4f%%" % (k, 100 * float(cum[k-1])))

print()
print("rank-k reconstruction error (Frobenius, relative):")
for k in (8, 16, 32, 64, 128, 256):
    kk = min(k, S.numel())
    Ur = U[:, :kk]
    approx = Ur @ (Ur.T @ sub)
    rel = float((approx - sub).norm() / sub.norm())
    print("   rank %4d -> rel err %.4f" % (kk, rel))

print()
print("=> dense does NOT imply high-rank; see energy curve above.")
print("   If a small k captures ~99% energy, a factored readout is the real win")
print("   (and it is a lossy-but-testable proposal, unlike CSR which is 1.97x worse).")
