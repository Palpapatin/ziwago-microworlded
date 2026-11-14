"""Selective snapshot equivalence and lookup semantics; no models or checkpoints."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ziwago.config import USNeuronCfg
from ziwago.cortex.moe_cortex import (
    _get_runtime_path,
    export_runtime_paths,
    export_runtime_tree,
    load_runtime_paths,
)


def _reference_export(root, paths):
    """The uncached implementation, including its single shared tensor memo."""
    paths = [tuple(path) for path in paths]
    values = []
    for path in paths:
        try:
            values.append(_get_runtime_path(root, path))
        except (AttributeError, KeyError, IndexError, TypeError):
            values.append({"__runtime_path_missing__": True})
    return {"runtime_version": 1, "paths": paths,
            "values": export_runtime_tree(values)}


def _assert_tree_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert actual.requires_grad == expected.requires_grad
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert list(actual) == list(expected)
        for key in actual:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize(("wrap", "prefix"), [
    pytest.param(lambda value: {"root": value}, ("root",), id="dict-root"),
    pytest.param(lambda value: [value], (0,), id="list-root"),
    pytest.param(lambda value: (value,), (0,), id="tuple-root"),
    pytest.param(lambda value: SimpleNamespace(root=value), ("root",), id="object-root"),
])
def test_mixed_containers_preserve_wire_format_and_path_order(wrap, prefix):
    root = wrap({
        "nodes": [SimpleNamespace(pair=({"value": 11}, {"value": 22}), flag=True)],
        "label": "saved",
    })
    paths = [
        prefix + ("nodes", 0, "pair", 1, "value"),
        (),
        prefix + ("nodes",),
        prefix + ("label",),
        prefix + ("nodes", 0, "pair", 0),
        prefix,
        prefix + ("nodes", 0),
        prefix + ("nodes", 0, "pair"),
        prefix + ("nodes", 0, "flag"),
        prefix + ("nodes", 0, "pair", -1, "value"),
        prefix + ("label",),
        (),
    ]
    actual = export_runtime_paths(root, (list(path) for path in paths))
    _assert_tree_equal(actual, _reference_export(root, paths))
    assert list(actual) == ["runtime_version", "paths", "values"]
    assert actual["runtime_version"] == 1
    assert actual["paths"] == paths


def test_empty_path_collection():
    assert export_runtime_paths(SimpleNamespace(value=1), iter(())) == {
        "runtime_version": 1, "paths": [], "values": [],
    }


def test_missing_nested_parents_and_present_sentinel_data():
    root = SimpleNamespace(
        mapping={"none": None, "marker": {"__runtime_path_missing__": True}},
        seq=[SimpleNamespace(value=7)], pair=(8,),
    )
    missing = [
        ("absent", "a"), ("absent", "b"),
        ("mapping", "absent", "a"), ("mapping", "absent", "b"),
        ("mapping", "none", "value"),
        ("seq", 3, "a"), ("seq", 3, "b"), ("pair", -2),
        ("seq", 0, "absent"), ("seq", "bad-index"), ("seq", 0, 2),
        ("mapping", []), ("seq", {}),
    ]
    paths = missing + [("mapping", "none"), ("seq", 0, "value"), ("mapping", "marker")]
    actual = export_runtime_paths(root, paths)
    _assert_tree_equal(actual, _reference_export(root, paths))
    assert actual["values"][:len(missing)] == [
        {"__runtime_path_missing__": True} for _ in missing
    ]
    assert actual["values"][len(missing):] == [None, 7, {"__runtime_path_missing__": True}]


def test_unusual_keys_keep_uncached_indexing_semantics():
    class UnhashableIndex:
        __hash__ = None

        def __index__(self):
            return 1

    root = {
        "seq": [10, 20, 30], "pair": (40, 50),
        "mapping": {(1, 2): "tuple", 1: "integer", "1": "string", None: "none", b"k": "bytes"},
    }
    paths = [
        ("seq", 0), ("seq", 0.0), ("seq", True), ("seq", 1), ("seq", False),
        ("seq", slice(None, None, -1)), ("seq", slice(0, 2), 1),
        ("seq", UnhashableIndex()),
        ("pair", 1), ("pair", 1.0), ("pair", slice(None, None, -1), 0),
        ("mapping", (1, 2)), ("mapping", 1), ("mapping", True),
        ("mapping", 1.0), ("mapping", "1"), ("mapping", None), ("mapping", b"k"),
    ]
    actual = export_runtime_paths(root, paths)
    _assert_tree_equal(actual, _reference_export(root, paths))
    assert actual["values"][:5] == [10, {"__runtime_path_missing__": True}, 20, 20, 10]
    assert actual["values"][7] == 20
    assert actual["values"][9] == {"__runtime_path_missing__": True}
    assert actual["values"][11:] == ["tuple", "integer", "integer", "integer", "string", "none", "bytes"]


def test_tensor_aliases_span_repeated_paths_and_whole_subtrees():
    shared = torch.tensor([1.0, 2.0], requires_grad=True)
    root = SimpleNamespace(branch={"tensor": shared, "seq": [shared, (shared,)]}, alias=shared)
    paths = [("branch", "tensor"), ("alias",), ("branch",), ("branch", "tensor"), ()]
    actual = export_runtime_paths(root, paths)
    _assert_tree_equal(actual, _reference_export(root, paths))
    values = actual["values"]
    copies = [
        values[0], values[1], values[2]["tensor"], values[2]["seq"][0],
        values[2]["seq"][1]["items"][0], values[3],
        values[4]["attributes"]["alias"], values[4]["attributes"]["branch"]["tensor"],
    ]
    assert all(value is copies[0] for value in copies)
    assert copies[0] is not shared
    assert copies[0].data_ptr() != shared.data_ptr()
    assert not copies[0].requires_grad
    with torch.no_grad():
        shared.fill_(9.0)
    assert torch.equal(copies[0], torch.tensor([1.0, 2.0]))
    copies[0].zero_()
    assert torch.equal(shared, torch.tensor([9.0, 9.0]))


@pytest.mark.parametrize("replacement_mode", ["assignment", "load-runtime-paths"])
def test_replacing_subobjects_between_exports_does_not_reuse_cached_nodes(replacement_mode):
    old, new = torch.tensor([1.0]), torch.tensor([2.0])
    root = SimpleNamespace(branch={"items": [SimpleNamespace(weight=old), (old,)], "removed": 1})
    replacement = SimpleNamespace(branch={"items": [SimpleNamespace(weight=new), (new,)], "added": 2})
    paths = [
        ("branch", "items", 0, "weight"), ("branch", "items", 1, 0),
        ("branch",), ("branch", "added"), ("branch", "removed"),
    ]
    before = export_runtime_paths(root, paths)
    previous_branch = root.branch
    if replacement_mode == "assignment":
        root.branch = replacement.branch
    else:
        load_runtime_paths(root, export_runtime_paths(replacement, [("branch",)]))
    assert root.branch is not previous_branch
    after = export_runtime_paths(root, paths)
    _assert_tree_equal(after, _reference_export(root, paths))
    assert torch.equal(before["values"][0], old)
    assert torch.equal(after["values"][0], new)
    assert after["values"][0] is after["values"][1]
    assert after["values"][0] is after["values"][2]["items"][0]["attributes"]["weight"]
    assert after["values"][0] is not before["values"][0]
    assert before["values"][3:] == [{"__runtime_path_missing__": True}, 1]
    assert after["values"][3:] == [2, {"__runtime_path_missing__": True}]


@pytest.mark.parametrize("error_type", [AttributeError, KeyError, IndexError, TypeError, ValueError, RuntimeError])
def test_lookup_exception_types_are_preserved(monkeypatch, error_type):
    def fail(self):
        raise error_type("path read failed")

    monkeypatch.setattr(USNeuronCfg, "broken", property(fail), raising=False)
    root, paths = USNeuronCfg(), [("c_m",), ("broken", "child")]
    if error_type in (AttributeError, KeyError, IndexError, TypeError):
        assert export_runtime_paths(root, paths) == _reference_export(root, paths)
    else:
        with pytest.raises(error_type) as reference_error:
            _reference_export(root, paths)
        with pytest.raises(error_type) as actual_error:
            export_runtime_paths(root, paths)
        assert type(actual_error.value) is type(reference_error.value)
        assert str(actual_error.value) == str(reference_error.value)


@pytest.mark.parametrize("value", [pytest.param(object(), id="unsupported"), pytest.param(lambda: None, id="callable")])
def test_serialization_errors_are_not_converted_to_missing(value):
    root, paths = {"item": value}, [("item",), ("item",)]
    with pytest.raises(TypeError) as reference_error:
        _reference_export(root, paths)
    with pytest.raises(TypeError) as actual_error:
        export_runtime_paths(root, paths)
    assert type(actual_error.value) is type(reference_error.value)
    assert str(actual_error.value) == str(reference_error.value)


@pytest.mark.parametrize("paths", [None, [None], [5]])
def test_path_normalization_errors_are_preserved(paths):
    with pytest.raises(TypeError) as reference_error:
        _reference_export({}, paths)
    with pytest.raises(TypeError) as actual_error:
        export_runtime_paths({}, paths)
    assert type(actual_error.value) is type(reference_error.value)
    assert str(actual_error.value) == str(reference_error.value)


def test_stateful_properties_keep_sequential_lookup_semantics(monkeypatch):
    reads = 0

    def sequence(self):
        nonlocal reads
        reads += 1
        return {"value": reads}

    monkeypatch.setattr(USNeuronCfg, "sequence", property(sequence), raising=False)
    root = USNeuronCfg()
    paths = [("sequence", "value")] * 3
    actual = export_runtime_paths(root, paths)
    assert actual["values"] == [1, 2, 3]
    reads = 0
    _assert_tree_equal(actual, _reference_export(root, paths))
