"""Scratch: check the corrected toroidal kernel (correct axes) against v21 acceptance."""
import torch
import torch.nn.functional as F

from ziwago.memory.tcc import ToroidalRelationAttractor, toroidal_distance


def corrected_drive(self):
    a = self.activity
    m_c1 = (a * self.c1[:, None]).mean(dim=0)   # [N] over dim0 (mode-1 amplitude per col)
    m_s1 = (a * self.s1[:, None]).mean(dim=0)
    m_c2 = (a * self.c2[None, :]).mean(dim=1)   # [N] over dim1 (mode-2 amplitude per row)
    m_s2 = (a * self.s2[None, :]).mean(dim=1)
    return (-self.w_inh * a.mean()
            + self.w_exc * (self.c1[:, None] * m_c1[None, :] + self.s1[:, None] * m_s1[None, :]
                            + m_c2[:, None] * self.c2[None, :] + m_s2[:, None] * self.s2[None, :]))


def vec(seed, d=32):
    return F.normalize(torch.randn(d, generator=torch.Generator().manual_seed(seed)), dim=0)


def stored(n=6, d=32, **kw):
    params = dict(grid_size=24, d_vec=32, seed=0, enabled=True)
    params.update(kw)
    t = ToroidalRelationAttractor(**params)
    for i in range(n):
        t.store_relation("rel_%d" % i, vec(i, d))
    return t


def report(label):
    t = stored()
    stab = t.bump_stability("rel_0", steps=60, anchor_gain=0.0)
    print(label, "stab:", {k: round(v, 4) for k, v in stab.items()})
    t2 = stored(w_exc=0.0)
    s2 = t2.bump_stability("rel_0", steps=60, anchor_gain=0.0)
    print(label, "stab w_exc=0:", {k: round(v, 4) for k, v in s2.items()})
    t3 = stored(n=8)
    hits = sum(t3.recall("rel_%d" % i, settle_steps=12)["decoded"] == "rel_%d" % i for i in range(8))
    print(label, "decode hits:", hits, "/8")
    t4 = stored(n=8)
    labels = list(t4.anchors)
    pairs = [(a, b, toroidal_distance(t4.anchors[a], t4.anchors[b]))
             for i, a in enumerate(labels) for b in labels[i + 1:]]
    a, b, sep = max(pairs, key=lambda x: x[2])
    res = t4.interference(a, b, steps=30)
    print(label, "interference sep=%.3f" % sep, {k: round(v, 4) for k, v in res.items()})
    t5 = stored(n=8)
    stabp = t5.bump_stability("rel_1", steps=40, anchor_gain=0.5)
    print(label, "pinned:", {k: round(v, 4) for k, v in stabp.items()})
    t6 = stored(n=6)
    t6.place_bump(*t6.anchors["rel_3"])
    t6.step(steps=12)
    print(label, "decode reads activity:", t6.decode()["decoded"])


report("ORIGINAL")
ToroidalRelationAttractor.recurrent_drive = corrected_drive
report("CORRECTED")
