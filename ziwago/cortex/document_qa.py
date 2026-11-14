from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA = "neuroweave.document_evidence.v1"
HASH_SCHEME = "blake2b-word-1:3-char-3:5-v1"
_WORDS = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


def word_tokens(text: str) -> list[str]:
    return _WORDS.findall(text.casefold())


@dataclass(frozen=True)
class RankerConfig:
    buckets: int = 8192
    width: int = 64
    max_question_tokens: int = 64
    max_sentence_tokens: int = 160
    max_token_chars: int = 96
    seed: int = 42
    hash_scheme: str = HASH_SCHEME

    def __post_init__(self):
        integers = (self.buckets, self.width, self.max_question_tokens,
                    self.max_sentence_tokens, self.max_token_chars, self.seed)
        if any(type(value) is not int for value in integers):
            raise ValueError("Ranker configuration requires integer sizes and seed")
        if self.buckets != 8192 or self.width != 64 or self.hash_scheme != HASH_SCHEME:
            raise ValueError("Unsupported ranker architecture or hashing scheme")
        if not (1 <= self.max_question_tokens <= 256
                and 1 <= self.max_sentence_tokens <= 512
                and 1 <= self.max_token_chars <= 256
                and 0 <= self.seed < 2 ** 63):
            raise ValueError("Invalid ranker token limits or seed")


@dataclass(frozen=True)
class EvidenceGroup:
    question: str
    sentences: tuple[str, ...]
    positive_indices: tuple[int, ...] = ()
    family_id: str = ""


def _checked_tokens(text: str, limit: int, config: RankerConfig, kind: str) -> list[str]:
    if type(text) is not str:
        raise ValueError(f"{kind} must be a string")
    tokens = word_tokens(text)
    if not tokens:
        raise ValueError(f"{kind} must contain at least one word token")
    if len(tokens) > limit or any(len(token) > config.max_token_chars for token in tokens):
        raise ValueError(f"{kind} exceeds token limits; truncation is not supported")
    return tokens


def _bucket(feature: str) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8,
                             person=b"nw-doc-qa-v1").digest()
    return 1 + int.from_bytes(digest, "little") % 8191


def _features(tokens: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    rows = []
    for index, token in enumerate(tokens):
        features = [_bucket("w:" + token)]
        for size in (2, 3):
            if index + 1 >= size:
                features.append(_bucket("w:" + "\0".join(tokens[index + 1 - size:index + 1])))
        bounded = "^" + token + "$"
        for size in (3, 4, 5):
            features.extend(_bucket("c:" + bounded[pos:pos + size])
                            for pos in range(len(bounded) - size + 1))
        rows.append(tuple(features))
    return tuple(rows)


def _sequence_batch(rows, device):
    length = max(len(row) for row in rows)
    ids, offsets, masks = [], [0], []
    for row in rows:
        masks.append([True] * len(row) + [False] * (length - len(row)))
        for position in range(length):
            if position < len(row):
                ids.extend(row[position])
            offsets.append(len(ids))
    return (torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(offsets, dtype=torch.long, device=device),
            torch.tensor(masks, dtype=torch.bool, device=device))


def _encode_pairs(pairs, config: RankerConfig, device, cache=None):
    def encode(text, limit, kind):
        key = (text, limit)
        if cache is not None and key in cache:
            return cache[key]
        tokens = _checked_tokens(text, limit, config, kind)
        result = tokens, _features(tokens)
        if cache is not None:
            cache[key] = result
        return result

    questions, sentences = [], []
    for question, sentence in pairs:
        questions.append(encode(question, config.max_question_tokens, "Question"))
        sentences.append(encode(sentence, config.max_sentence_tokens, "Sentence"))
    q_ids, q_offsets, q_mask = _sequence_batch([item[1] for item in questions], device)
    s_ids, s_offsets, s_mask = _sequence_batch([item[1] for item in sentences], device)
    exact = torch.zeros((len(pairs), q_mask.shape[1], s_mask.shape[1]), dtype=torch.bool)
    for index, ((q_tokens, _), (s_tokens, _)) in enumerate(zip(questions, sentences)):
        exact[index, :len(q_tokens), :len(s_tokens)] = torch.tensor(
            [[q == s for s in s_tokens] for q in q_tokens], dtype=torch.bool)
    return {"q_ids": q_ids, "q_offsets": q_offsets, "q_mask": q_mask,
            "s_ids": s_ids, "s_offsets": s_offsets, "s_mask": s_mask,
            "exact": exact.to(device)}


class OrderedEvidenceModel(nn.Module):
    def __init__(self, config: RankerConfig):
        super().__init__()
        width = config.width
        self.embedding = nn.EmbeddingBag(config.buckets, width, mode="mean",
                                         include_last_offset=True, padding_idx=0)
        self.context = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1) for _ in range(2)])
        self.context_norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.match_bias = nn.Parameter(torch.zeros(()))
        self.interaction = nn.Linear(4 * width + 1, width)
        self.ordered = nn.Conv1d(width, width, 3, padding=1)
        self.ordered_norm = nn.LayerNorm(width)
        self.hidden = nn.Linear(4 * width, width)
        self.head = nn.Linear(width, 2)

    def _contextualize(self, ids, offsets, mask):
        value = self.embedding(ids, offsets).reshape(*mask.shape, -1)
        for conv, norm in zip(self.context, self.context_norm):
            value = norm(value + F.gelu(conv(value.transpose(1, 2)).transpose(1, 2)))
            value = value * mask.unsqueeze(-1)
        return value

    @staticmethod
    def _mean(value, mask):
        return (value * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

    def forward(self, batch):
        q_mask, s_mask = batch["q_mask"], batch["s_mask"]
        question = self._contextualize(batch["q_ids"], batch["q_offsets"], q_mask)
        sentence = self._contextualize(batch["s_ids"], batch["s_offsets"], s_mask)
        attention = question @ sentence.transpose(1, 2) / math.sqrt(question.shape[-1])
        attention = attention + self.match_bias * batch["exact"]
        attention = attention.masked_fill(~s_mask.unsqueeze(1), -torch.inf).softmax(-1)
        attended = attention @ sentence
        exact = batch["exact"].any(-1).to(question.dtype).unsqueeze(-1)
        interaction = F.gelu(self.interaction(torch.cat(
            (question, attended, question * attended, question - attended, exact), dim=-1)))
        interaction = interaction * q_mask.unsqueeze(-1)
        ordered = self.ordered_norm(
            interaction + F.gelu(self.ordered(interaction.transpose(1, 2)).transpose(1, 2)))
        ordered = ordered * q_mask.unsqueeze(-1)
        pooled = torch.cat((self._mean(ordered, q_mask),
                            ordered.masked_fill(~q_mask.unsqueeze(-1), -torch.inf).amax(1),
                            self._mean(question, q_mask), self._mean(sentence, s_mask)), dim=-1)
        output = self.head(F.gelu(self.hidden(pooled)))
        return output[:, 0], output[:, 1]


def _validate_groups(groups, config):
    result = list(groups)
    if not result:
        raise ValueError("At least one evidence group is required")
    for group in result:
        if not isinstance(group, EvidenceGroup) or not group.sentences:
            raise ValueError("Training requires nonempty EvidenceGroup instances")
        if type(group.family_id) is not str:
            raise ValueError("family_id must be a string")
        _checked_tokens(group.question, config.max_question_tokens, config, "Question")
        for sentence in group.sentences:
            _checked_tokens(sentence, config.max_sentence_tokens, config, "Sentence")
        positives = group.positive_indices
        if (any(type(index) is not int or not 0 <= index < len(group.sentences) for index in positives)
                or len(set(positives)) != len(positives)):
            raise ValueError("Positive sentence indices must be unique and in range")
    return result


def _group_losses(rank, support, groups, binary_weight):
    losses, offset = [], 0
    for group in groups:
        length = len(group.sentences)
        scores, logits = rank[offset:offset + length], support[offset:offset + length]
        labels = torch.zeros_like(logits)
        normalizer = torch.cat((scores.new_zeros(1), scores)).logsumexp(0)
        if group.positive_indices:
            labels[list(group.positive_indices)] = 1
            listwise = normalizer - scores[list(group.positive_indices)].logsumexp(0)
        else:
            listwise = normalizer
        binary = F.binary_cross_entropy_with_logits(logits, labels)
        losses.append(listwise + binary_weight * binary)
        offset += length
    return torch.stack(losses).mean()


_CALIBRATION_KEYS = {"method", "scope", "threshold", "groups", "positive_groups", "negative_groups",
                     "balanced_accuracy", "positive_accept_rate", "negative_accept_rate"}


def _validate_calibration(value):
    if value is None:
        return
    if type(value) is not dict or set(value) != _CALIBRATION_KEYS:
        raise ValueError("Malformed calibration state")
    if (type(value["method"]) is not str or type(value["scope"]) is not str
            or value["method"] != "ranked_sentence_support_balanced_accuracy"
            or value["scope"] != "official-train/internal-calibration/annotation-support"):
        raise ValueError("Unsupported calibration provenance")
    for key in ("groups", "positive_groups", "negative_groups"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError("Calibration counts must be positive integers")
    if value["groups"] != value["positive_groups"] + value["negative_groups"]:
        raise ValueError("Inconsistent calibration counts")
    for key in ("threshold", "balanced_accuracy", "positive_accept_rate", "negative_accept_rate"):
        if type(value[key]) is not float or not math.isfinite(value[key]) or not 0 <= value[key] <= 1:
            raise ValueError("Calibration metrics must be finite probabilities")
    expected = (value["positive_accept_rate"] + 1 - value["negative_accept_rate"]) / 2
    if not math.isclose(value["balanced_accuracy"], expected, abs_tol=1e-12):
        raise ValueError("Inconsistent calibration metrics")


def validate_training_metadata(metadata: dict) -> None:
    if type(metadata) is not dict or metadata.get("schema") != "ziwago.document_ranker.training.v1":
        raise ValueError("Missing document ranker training provenance")
    source = metadata.get("source")
    if (type(source) is not dict or source.get("repository") != "rajpurkar/squad"
            or source.get("split") != "train" or not isinstance(source.get("revision"), str)
            or re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is None
            or not isinstance(source.get("source_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", source["source_sha256"]) is None):
        raise ValueError("Invalid official-train source provenance")
    hash_keys = ("training_prompt_hashes", "training_document_hashes", "training_normalized_document_hashes",
                 "exposed_prompt_hashes", "exposed_document_hashes")
    for key in hash_keys:
        values = metadata.get(key)
        if (type(values) is not list or not values or any(type(value) is not str
                or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in values)
                or values != sorted(set(values))):
            raise ValueError(f"Invalid training identity hashes: {key}")
    for key in ("fit_family_ids", "internal_dev_family_ids", "calibration_family_ids"):
        values = metadata.get(key)
        if (type(values) is not list or any(type(value) is not str or not value.strip() for value in values)
                or values != sorted(set(values))):
            raise ValueError(f"Invalid ranker article identities: {key}")
    families = [set(metadata[key]) for key in ("fit_family_ids", "internal_dev_family_ids", "calibration_family_ids")]
    if not all(families) or any(families[i] & families[j] for i in range(3) for j in range(i)):
        raise ValueError("Missing or overlapping ranker article partitions")
    records = metadata.get("training_group_provenance")
    if type(records) is not list or not records:
        raise ValueError("Missing group identity provenance")
    fields = {"normalized_prompt_sha256", "raw_document_sha256", "normalized_document_sha256", "family_id", "partition"}
    for record in records:
        if (type(record) is not dict or set(record) != fields or record["partition"] not in ("fit", "internal_dev_pool")
                or type(record["family_id"]) is not str or not record["family_id"].strip()
                or any(type(record[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", record[key]) is None
                       for key in ("normalized_prompt_sha256", "raw_document_sha256", "normalized_document_sha256"))):
            raise ValueError("Malformed ranker group identity")
    fitted = [record for record in records if record["partition"] == "fit"]
    pooled_families = {record["family_id"] for record in records if record["partition"] == "internal_dev_pool"}
    if pooled_families != families[1] | families[2] or pooled_families & families[0]:
        raise ValueError("Ranker holdout article ledger disagrees with its group provenance")
    expected = {
        "training_prompt_hashes": {r["normalized_prompt_sha256"] for r in fitted},
        "training_document_hashes": {r["raw_document_sha256"] for r in fitted},
        "training_normalized_document_hashes": {r["normalized_document_sha256"] for r in fitted},
        "exposed_prompt_hashes": {r["normalized_prompt_sha256"] for r in records},
        "exposed_document_hashes": {r[key] for r in records for key in ("raw_document_sha256", "normalized_document_sha256")},
        "fit_family_ids": {r["family_id"] for r in fitted},
    }
    if any(metadata[key] != sorted(values) for key, values in expected.items()):
        raise ValueError("Ranker identity ledger disagrees with its group provenance")
    if any(metadata.get(key) is not False for key in ("official_validation_read", "acceptance_documents_read", "raw_answers_saved")):
        raise ValueError("Ranker training provenance includes prohibited evaluation data")
    digest = metadata.get("ranker_checkpoint_sha256")
    if digest is not None and (type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise ValueError("Malformed ranker checkpoint identity")


class DocumentEvidenceRanker:
    """Ordered sentence ranking; calibrated support reflects annotations, not semantic truth."""

    def __init__(self, seed: int = 42, *, max_question_tokens: int = 64,
                 max_sentence_tokens: int = 160, max_token_chars: int = 96):
        self.config = RankerConfig(seed=seed, max_question_tokens=max_question_tokens,
                                   max_sentence_tokens=max_sentence_tokens,
                                   max_token_chars=max_token_chars)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.model = OrderedEvidenceModel(self.config)
        self.model.eval()
        self.trained = False
        self.updates = 0
        self._calibration = None

    @property
    def support_threshold(self) -> float | None:
        return None if self._calibration is None else self._calibration["threshold"]

    @property
    def calibration(self) -> dict | None:
        return None if self._calibration is None else dict(self._calibration)

    def _outputs(self, question: str, sentences: Sequence[str]):
        _checked_tokens(question, self.config.max_question_tokens, self.config, "Question")
        if not isinstance(sentences, (list, tuple)):
            raise ValueError("Sentences must be a list or tuple of complete source sentences")
        for sentence in sentences:
            _checked_tokens(sentence, self.config.max_sentence_tokens, self.config, "Sentence")
        ranks, supports = [], []
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            for start in range(0, len(sentences), 32):
                pairs = [(question, sentence) for sentence in sentences[start:start + 32]]
                rank, support = self.model(_encode_pairs(pairs, self.config, device))
                if not (torch.isfinite(rank).all() and torch.isfinite(support).all()):
                    raise ValueError("Ranker produced nonfinite scores")
                ranks.extend(rank.cpu().tolist())
                supports.extend(support.sigmoid().cpu().tolist())
        return ranks, supports

    def score_sentences(self, question: str, sentences: list[str]) -> list[float]:
        return self._outputs(question, sentences)[0]

    def assess(self, question: str, sentences: list[str]) -> dict:
        scores, supports = self._outputs(question, sentences)
        best = max(range(len(scores)), key=scores.__getitem__) if scores else None
        probability = supports[best] if best is not None else 0.0
        threshold = self.support_threshold
        return {"scores": scores, "sentence_support": supports, "support_probability": probability,
                "support_threshold": threshold, "supported": bool(
                    self.trained and threshold is not None and supports and probability > threshold),
                "support_kind": "learned_annotation_support_not_entailment"}

    def _batch(self, groups, cache=None):
        pairs = [(group.question, sentence) for group in groups for sentence in group.sentences]
        return _encode_pairs(pairs, self.config, next(self.model.parameters()).device, cache)

    def evaluate_groups(self, groups, *, binary_weight: float = 1.0, batch_groups: int = 8) -> dict:
        groups = _validate_groups(groups, self.config)
        self._validate_training_options(1, 0.001, batch_groups, binary_weight)
        total_loss, top1, top2, positive_count = 0.0, 0, 0, 0
        accepted, accepted_annotated, no_positive_accepted = 0, 0, 0
        threshold = self.support_threshold if self.trained else None
        with torch.inference_mode():
            for start in range(0, len(groups), batch_groups):
                batch_groups_list = groups[start:start + batch_groups]
                rank, support = self.model(self._batch(batch_groups_list))
                loss = _group_losses(rank, support, batch_groups_list, binary_weight)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite evaluation loss")
                total_loss += float(loss) * len(batch_groups_list)
                offset = 0
                for group in batch_groups_list:
                    scores = rank[offset:offset + len(group.sentences)]
                    indices = scores.argsort(descending=True, stable=True).tolist()
                    positives = set(group.positive_indices)
                    if threshold is not None:
                        assessment = self.assess(group.question, list(group.sentences))
                        selected = max(range(len(group.sentences)), key=assessment["scores"].__getitem__)
                        if assessment["supported"]:
                            accepted += 1
                            accepted_annotated += int(selected in positives)
                            no_positive_accepted += int(not positives)
                    if group.positive_indices:
                        top1 += int(indices[0] in positives)
                        top2 += int(bool(positives.intersection(indices[:2])))
                        positive_count += 1
                    offset += len(group.sentences)
        return {"groups": len(groups), "positive_groups": positive_count,
                "no_positive_groups": len(groups) - positive_count, "loss": total_loss / len(groups),
                "recall_at_1": top1 / positive_count if positive_count else None,
                "recall_at_2": top2 / positive_count if positive_count else None,
                "selective_metrics_available": threshold is not None,
                "accepted_groups": accepted if threshold is not None else None,
                "accepted_annotated_sentences": accepted_annotated if threshold is not None else None,
                "accepted_unannotated_sentences": accepted - accepted_annotated if threshold is not None else None,
                "annotation_precision_among_accepted": accepted_annotated / accepted if accepted else None,
                "coverage": accepted / len(groups) if threshold is not None else None,
                "no_positive_accepted_groups": no_positive_accepted if threshold is not None else None,
                "no_positive_group_acceptance_rate": no_positive_accepted / (len(groups) - positive_count)
                if threshold is not None and len(groups) > positive_count else None}

    @staticmethod
    def _validate_training_options(epochs, lr, batch_groups, binary_weight):
        if type(epochs) is not int or epochs <= 0 or type(batch_groups) is not int or batch_groups <= 0:
            raise ValueError("epochs and batch_groups must be positive integers")
        for value in (lr, binary_weight):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Learning rate and binary loss weight must be finite and positive")

    def fit_groups(self, groups, epochs: int = 2, *, lr: float = 0.001,
                   batch_groups: int = 8, binary_weight: float = 1.0, seed: int | None = None) -> dict:
        self._validate_training_options(epochs, lr, batch_groups, binary_weight)
        groups = _validate_groups(groups, self.config)
        seed = self.config.seed if seed is None else seed
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError("Training seed must be a nonnegative integer below 2**63")
        if not any(group.positive_indices for group in groups) or not any(
                not group.positive_indices for group in groups):
            raise ValueError("Fit requires annotated-positive and no-positive groups")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        rng, cache = random.Random(seed), {}
        history = [self.evaluate_groups(groups, binary_weight=binary_weight, batch_groups=batch_groups)]
        self._calibration = None
        prior_mode = self.model.training
        try:
            self.model.train()
            for _ in range(epochs):
                order = list(range(len(groups)))
                rng.shuffle(order)
                for start in range(0, len(order), batch_groups):
                    batch = [groups[index] for index in order[start:start + batch_groups]]
                    optimizer.zero_grad(set_to_none=True)
                    rank, support = self.model(self._batch(batch, cache))
                    loss = _group_losses(rank, support, batch, binary_weight)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    self.updates += 1
                    self.trained = True
                history.append(self.evaluate_groups(groups, binary_weight=binary_weight,
                                                    batch_groups=batch_groups))
        finally:
            self.model.train(prior_mode)
            optimizer.zero_grad(set_to_none=True)
        return {"epochs": epochs, "updates": self.updates, "binary_weight": float(binary_weight),
                "history": history}

    def calibrate(self, internal_calibration_groups) -> dict:
        if not self.trained:
            raise ValueError("Fit the model before calibrating support")
        groups = _validate_groups(internal_calibration_groups, self.config)
        observations = []
        for group in groups:
            scores, support = self._outputs(group.question, group.sentences)
            best = max(range(len(scores)), key=scores.__getitem__)
            observations.append((support[best], best in group.positive_indices))
        positive = sum(label for _, label in observations)
        negative = len(observations) - positive
        if not positive or not negative:
            raise ValueError("Calibration requires positive and negative selected-sentence annotations")
        observations.sort()
        tp, fp = positive, negative
        cursor, best, best_key = 0, None, None
        for threshold in sorted({0.0, *(probability for probability, _ in observations)}):
            while cursor < len(observations) and observations[cursor][0] <= threshold:
                if observations[cursor][1]:
                    tp -= 1
                else:
                    fp -= 1
                cursor += 1
            tpr, fpr = tp / positive, fp / negative
            balanced = (tpr + 1 - fpr) / 2
            key = balanced, -fpr, threshold
            if best_key is None or key > best_key:
                best_key = key
                best = {"method": "ranked_sentence_support_balanced_accuracy",
                        "scope": "official-train/internal-calibration/annotation-support",
                        "threshold": float(threshold), "groups": len(groups),
                        "positive_groups": positive, "negative_groups": negative,
                        "balanced_accuracy": float(balanced), "positive_accept_rate": float(tpr),
                        "negative_accept_rate": float(fpr)}
        _validate_calibration(best)
        self._calibration = best
        return dict(best)

    def export_state_dict(self) -> dict:
        return {"schema": SCHEMA, "config": asdict(self.config), "trained": self.trained,
                "updates": self.updates, "calibration": self.calibration,
                "weights": {name: tensor.detach().cpu().clone()
                            for name, tensor in self.model.state_dict().items()}}

    def load_state_dict(self, state: dict) -> None:
        keys = {"schema", "config", "trained", "updates", "calibration", "weights"}
        if (type(state) is not dict or set(state) != keys or type(state["schema"]) is not str
                or state["schema"] != SCHEMA):
            raise ValueError("Malformed or unsupported document ranker state")
        config = state["config"]
        if type(config) is not dict or set(config) != set(asdict(RankerConfig())):
            raise ValueError("Malformed ranker configuration")
        if type(config["hash_scheme"]) is not str:
            raise ValueError("Malformed hashing scheme")
        try:
            config = RankerConfig(**config)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid ranker configuration") from error
        if (type(state["trained"]) is not bool or type(state["updates"]) is not int
                or state["updates"] < 0 or state["trained"] != (state["updates"] > 0)):
            raise ValueError("Inconsistent trained state or update count")
        _validate_calibration(state["calibration"])
        if not state["trained"] and state["calibration"] is not None:
            raise ValueError("An untrained ranker cannot have support calibration")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            candidate = OrderedEvidenceModel(config)
        weights, expected = state["weights"], candidate.state_dict()
        if type(weights) is not dict or set(weights) != set(expected):
            raise ValueError("Missing or unexpected ranker tensors")
        for name, tensor in weights.items():
            if (type(tensor) is not torch.Tensor or tensor.device.type != "cpu"
                    or tensor.layout != torch.strided or tensor.dtype != expected[name].dtype
                    or tensor.shape != expected[name].shape or tensor.requires_grad):
                raise ValueError(f"Invalid ranker tensor: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite ranker tensor: {name}")
        candidate.load_state_dict(weights, strict=True)
        candidate.train(self.model.training)
        self.config, self.model = config, candidate
        self.trained, self.updates = state["trained"], state["updates"]
        self._calibration = None if state["calibration"] is None else dict(state["calibration"])
