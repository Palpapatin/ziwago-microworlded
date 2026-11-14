import sys, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch.nn.functional as F
from train_ziwago_v14_pure_synaptic_dialogue import DendriticPlateauMatrixReadout

ro = DendriticPlateauMatrixReadout(d_vec=64, d_dend=8192, max_pos=64, seed=1405)

pole = F.normalize(torch.randn(64), dim=0)
ro.w_ca3_attractors = pole.unsqueeze(0).clone()
sdr = torch.zeros(2048); sdr[[10, 20, 30, 40]] = 1.0
ro.w_ca3_sdr = sdr.unsqueeze(0).clone()
ro.w_ca3_salience = torch.tensor([1.5])

idx, res = ro.resolve_episode_pole(pole)
print("resolve:", idx, round(res, 6))
assert idx == 0 and res > 0.99

other = F.normalize(torch.randn(64), dim=0)
idx2, res2 = ro.resolve_episode_pole(other)
print("other:", idx2, round(res2, 4))
assert idx2 is None

# Траектория эпизода
ro.upsert_episode_trace(0, [101, 102, 103])
print("trace:", ro.episode_trace(0))
assert ro.episode_trace(0) == [101, 102, 103]

# Код — функция ЭЛЕМЕНТА, не позиции: одинаковый элемент -> одинаковый код
cue = F.normalize(torch.randn(64), dim=0)
c1 = ro.apical_episode_code(cue)
c2 = ro.apical_episode_code(cue)
print("code nnz", int((c1 != 0).sum()), "norm", float(c1.norm()),
      "deterministic", bool(torch.equal(c1, c2)))
assert torch.equal(c1, c2)

# Компоненты без апикального — набор прежний
d = torch.zeros(8192); d[[1, 5, 9]] = 0.5
ro.w_dend_out = torch.zeros(8192, 4); ro.w_dend_out[1] = 1.0
ro.col_to_tok = [10, 11, 12, 13]; ro.tok_to_col = {10: 0, 11: 1, 12: 2, 13: 3}
ro.w_core_dend_out = torch.zeros(8192, 4)
ro.w_swr_dend_out = torch.zeros(8192, 4)
comp = ro._readout_components(d, pole, 4)
print("components:", sorted(comp.keys()))
assert set(comp) == {"dendritic", "core", "swr", "bank"}

al = ro.apical_logits(c1, 1.0)
print("apical logits:", tuple(al.shape), float(al.abs().sum()))

st = ro.imprint_apical_replay([cue, cue, cue], [10, 11, 12], f_mod=1.5)
print("replay stats", st)
print("swr absmax", float(ro.w_swr_dend_out.abs().max()))
assert st["apical_replay_updates"] == 3

# Пустая траектория -> канал молчит
ro2 = DendriticPlateauMatrixReadout(d_vec=64, d_dend=8192, max_pos=64, seed=1405)
ro2.w_ca3_attractors = pole.unsqueeze(0).clone()
print("empty trace:", ro2.episode_trace(0))
assert ro2.episode_trace(0) == []
print("OK")
