"""B.1: верность загрузчика runtime-состояния (staged allocation + READONLY mmap).

Лёгкий регрессионный тест на tiny_core: export -> load roundtrip побайтно
идентичен (state_fingerprint совпадает). Проверяет механику загрузчика
(load_runtime_state / load_runtime_tree: поатрибутное восстановление, shared
tensor identity, миграция тегов) без тяжёлого чекпоинта.

Полный чекпоинт 4.5 ГБ (ziwago_v19_apical2_20261109.pt) проверяется тяжёлым
прибором Produce/b1_production_probe.py через tools/limited_run.py: там
NORM_MATCH=True (42/42 ключа ядра побайтно, fp acd64957856e8dda) и пик USS
5.94 ГБ. В обычный pytest полный чекпоинт не входит — OOM (§5.3/§5.4).
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from test_v19_spiking_language import tiny_core  # noqa: E402


def _hash_tensor(digest, value):
    digest.update(str((value.dtype, tuple(value.shape))).encode())
    tensor = value.detach().cpu()
    if tensor.numel() == 0:
        return
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))


def state_fingerprint(state):
    """sha256 по дереву состояния: структура + dtype/shape/байты тензоров."""
    digest = hashlib.sha256()

    def visit(value):
        digest.update(type(value).__name__.encode())
        if isinstance(value, torch.Tensor):
            _hash_tensor(digest, value)
        elif isinstance(value, dict):
            digest.update(str(len(value)).encode())
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(str(len(value)).encode())
            for item in value:
                visit(item)
        else:
            digest.update(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=repr).encode())

    visit(state)
    return digest.hexdigest()


class LoadFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_runtime_state_roundtrip_byte_identical(self):
        core = tiny_core()
        # Возмущаем НЕПУСТЫЕ данные, чтобы восстановление было нетривиальным.
        # w_dend_out в tiny_core пуст (readout.py:85 torch.empty(d_dend, 0)),
        # поэтому берём _vec64_cache[0] (в "core") и w_ff.w_slow (в "moe_cortex").
        with torch.no_grad():
            core._vec64_cache[0].add_(0.25)
            core.moe_cortex.columns[0].w_ff.w_slow.add_(0.1)
        source = core.export_runtime_state()
        fp_source_core = state_fingerprint(source["core"])
        fp_source_moe = state_fingerprint(source["moe_cortex"])

        fresh = tiny_core()
        fp_before_core = state_fingerprint(fresh.export_runtime_state()["core"])
        self.assertNotEqual(fp_before_core, fp_source_core,
                            "fresh tiny_core должен отличаться от возмущённого source")
        fresh.load_runtime_state(source)
        restored = fresh.export_runtime_state()
        self.assertEqual(state_fingerprint(restored["core"]), fp_source_core,
                         "load_runtime_state должен восстановить ядро побайтно")
        self.assertEqual(state_fingerprint(restored["moe_cortex"]), fp_source_moe,
                         "load_runtime_state должен восстановить moe_cortex побайтно")

    def test_shared_tensor_identity_preserved(self):
        from ziwago.cortex.moe_cortex import load_runtime_tree
        shared = torch.randn(8, 8)
        tree = {"__object__": "types:SimpleNamespace",
                "attributes": {"a": shared, "b": shared}}
        # export_runtime_tree мемоизирует shared -> один клон; load тоже.
        from ziwago.cortex.moe_cortex import export_runtime_tree
        exported = export_runtime_tree(tree)
        restored = load_runtime_tree(None, exported)
        self.assertIs(restored.a, restored.b,
                      "общий тензор должен остаться одним объектом после restore")
        self.assertTrue(torch.equal(restored.a, shared))


if __name__ == "__main__":
    unittest.main()