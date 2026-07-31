from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


@dataclass
class ModelPass:
    boxes_cxcywh: np.ndarray
    boxes_xyxy: np.ndarray
    category_scores: np.ndarray
    scores: np.ndarray
    category_indices: np.ndarray
    query_indices: np.ndarray
    hidden_states: np.ndarray
    reference_points: np.ndarray
    candidates_before_cap: int
    candidates_retained: int
    protected_candidates_retained: int


def select_candidate_indices(
    scores: np.ndarray,
    candidate_threshold: float,
    max_detections: int,
    required_query_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """Select top candidates while preserving eligible reference query IDs."""

    scores = np.asarray(scores, dtype=np.float64)
    eligible_all = np.flatnonzero(scores >= float(candidate_threshold))
    ordered_all = eligible_all[
        np.argsort(-scores[eligible_all], kind="stable")
    ]
    eligible = ordered_all[: int(max_detections)]
    protected_retained = 0
    if required_query_indices is not None and len(eligible_all):
        required = np.asarray(required_query_indices, dtype=np.int64)
        protected = eligible_all[np.isin(eligible_all, required)]
        protected_retained = int(len(protected))
        if len(protected):
            eligible = np.unique(np.concatenate([eligible, protected]))
            eligible = eligible[np.argsort(-scores[eligible], kind="stable")]
    return eligible.astype(np.int64), int(len(eligible_all)), protected_retained


def resolve_package_resource(value: str) -> Path:
    prefix = "package://"
    if not value.startswith(prefix):
        return Path(value).resolve()
    relative = value[len(prefix) :]
    package_name, _, resource = relative.partition("/")
    if not resource:
        raise ValueError(f"Package resource has no path: {value}")
    module = __import__(package_name)
    package_root = Path(module.__file__).resolve().parent
    path = package_root / resource
    if not path.exists():
        raise FileNotFoundError(f"Package resource not found: {path}")
    return path


def cxcywh_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    result = np.empty_like(boxes)
    result[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * width
    result[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * height
    result[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * width
    result[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * height
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0.0, float(width))
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0.0, float(height))
    return result


class PromptMapper:
    def __init__(self, tokenizer, classes: Sequence[str]):
        self.classes = tuple(classes)
        self.caption = " . ".join(self.classes) + " ."
        full_ids = tokenizer(
            self.caption, add_special_tokens=True, return_tensors=None
        )["input_ids"]
        self.token_indices: list[np.ndarray] = []
        cursor = 0
        for class_name in self.classes:
            class_ids = tokenizer(
                class_name, add_special_tokens=False, return_tensors=None
            )["input_ids"]
            match = self._find_subsequence(full_ids, class_ids, cursor)
            if match is None:
                raise ValueError(
                    f"Could not map class {class_name!r} into prompt tokens"
                )
            start, end = match
            self.token_indices.append(np.arange(start, end, dtype=np.int64))
            cursor = end

    @staticmethod
    def _find_subsequence(
        sequence: Sequence[int], target: Sequence[int], start: int
    ) -> tuple[int, int] | None:
        for index in range(start, len(sequence) - len(target) + 1):
            if list(sequence[index : index + len(target)]) == list(target):
                return index, index + len(target)
        return None

    def category_scores(self, token_probabilities: np.ndarray) -> np.ndarray:
        token_probabilities = np.asarray(token_probabilities)
        scores = [
            token_probabilities[:, token_ids].max(axis=1)
            for token_ids in self.token_indices
        ]
        return np.stack(scores, axis=1)


class _TransformerCapture:
    def __init__(self, transformer):
        self._latest = None
        self._handle = transformer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        hidden_states, references = output[0], output[1]
        import torch

        if isinstance(hidden_states, (list, tuple)):
            hidden_states = torch.stack(list(hidden_states))
        if isinstance(references, (list, tuple)):
            references = torch.stack(list(references))
        self._latest = (
            hidden_states.detach().float().cpu(),
            references.detach().float().cpu(),
        )

    def pop(self):
        if self._latest is None:
            raise RuntimeError("GroundingDINO transformer hook captured no output")
        latest = self._latest
        self._latest = None
        return latest

    def close(self) -> None:
        self._handle.remove()


class GroundingDinoAdapter:
    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        text_encoder_path: str | Path,
        classes: Sequence[str],
        stochastic_module_types: Sequence[str],
        device: str = "cuda",
        amp: bool = False,
    ):
        import torch
        from groundingdino.models import build_model
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        encoder_path = Path(text_encoder_path).resolve()
        if not encoder_path.is_dir():
            raise FileNotFoundError(
                f"Pinned text encoder not found: {encoder_path}. Run "
                "experiments2.0/scripts/prepare_model.py first."
            )
        arguments = SLConfig.fromfile(str(config_path))
        arguments.device = str(self.device)
        arguments.text_encoder_type = str(encoder_path)
        self.model = build_model(arguments)
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=True
        )
        incompatibility = self.model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        allowed_unexpected = {
            "label_enc.weight",
            "bert.embeddings.position_ids",
        }
        missing = sorted(incompatibility.missing_keys)
        unexpected = sorted(incompatibility.unexpected_keys)
        unsupported_unexpected = sorted(set(unexpected) - allowed_unexpected)
        if missing or unsupported_unexpected:
            raise RuntimeError(
                "Pinned GroundingDINO checkpoint is incompatible: "
                f"missing={missing}, unexpected={unexpected}, "
                f"allowed_unexpected={sorted(allowed_unexpected)}"
            )
        self.checkpoint_load_audit = {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "allowed_unexpected_keys": sorted(allowed_unexpected),
            "status": "validated",
        }
        self.model = self.model.to(self.device)
        self.model.eval()
        self.prompt = PromptMapper(self.model.tokenizer, classes)
        self.classes = tuple(classes)
        self.stochastic_module_types = set(stochastic_module_types)
        supported_types = {"Dropout", "DropPath", "BiMultiHeadAttention"}
        unknown_types = self.stochastic_module_types - supported_types
        if unknown_types:
            raise ValueError(
                f"Unsupported stochastic module types: {sorted(unknown_types)}"
            )
        self.amp = bool(amp)
        self.capture = _TransformerCapture(self.model.transformer)
        self.enabled_stochastic_modules: list[dict[str, str | float]] = []

    def close(self) -> None:
        self.capture.close()

    @contextmanager
    def stochastic_mode(
        self,
    ) -> Iterator[list[dict[str, str | float]]]:
        torch = self.torch
        original = {
            name: module.training for name, module in self.model.named_modules()
        }
        enabled: list[dict[str, str | float]] = []
        self.model.eval()
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__
            probability: float | None = None
            if (
                isinstance(module, torch.nn.Dropout)
                and "Dropout" in self.stochastic_module_types
            ):
                probability = float(module.p)
            elif (
                class_name == "DropPath"
                and "DropPath" in self.stochastic_module_types
            ):
                probability = float(module.drop_prob)
            elif (
                class_name == "BiMultiHeadAttention"
                and "BiMultiHeadAttention" in self.stochastic_module_types
            ):
                probability = float(module.dropout)
            if probability is not None and probability > 0.0:
                module.train()
                enabled.append(
                    {
                        "name": name,
                        "type": class_name,
                        "probability": probability,
                    }
                )
        self.enabled_stochastic_modules = enabled
        if not enabled:
            raise RuntimeError(
                "No configured stochastic module has probability > 0. "
                "The requested MC experiment would be deterministic."
            )
        try:
            yield enabled
        finally:
            for name, module in self.model.named_modules():
                module.train(original[name])

    def preprocess(self, image_path: str | Path):
        from groundingdino.util.inference import load_image

        source, tensor = load_image(str(image_path))
        return source, tensor

    def run(
        self,
        image_tensor,
        image_width: int,
        image_height: int,
        candidate_threshold: float,
        max_detections: int,
        required_query_indices: np.ndarray | None = None,
    ) -> ModelPass:
        torch = self.torch
        tensor = image_tensor.to(self.device)
        amp_enabled = self.amp and self.device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, enabled=amp_enabled
        ):
            outputs = self.model(tensor[None], captions=[self.prompt.caption])
        hidden_states, reference_points = self.capture.pop()

        token_probabilities = outputs["pred_logits"][0].sigmoid().float().cpu().numpy()
        boxes = outputs["pred_boxes"][0].float().cpu().numpy()
        category_scores = self.prompt.category_scores(token_probabilities)
        categories = category_scores.argmax(axis=1)
        scores = category_scores.max(axis=1)
        eligible, candidates_before_cap, protected_retained = (
            select_candidate_indices(
                scores,
                candidate_threshold,
                max_detections,
                required_query_indices,
            )
        )

        selected_boxes = boxes[eligible]
        selected_hidden = hidden_states[:, 0, eligible, :].numpy()
        selected_references = reference_points[:, 0, eligible, :].numpy()
        return ModelPass(
            boxes_cxcywh=selected_boxes,
            boxes_xyxy=cxcywh_to_xyxy(
                selected_boxes, width=image_width, height=image_height
            ),
            category_scores=category_scores[eligible],
            scores=scores[eligible],
            category_indices=categories[eligible],
            query_indices=eligible.astype(np.int64),
            hidden_states=selected_hidden,
            reference_points=selected_references,
            candidates_before_cap=candidates_before_cap,
            candidates_retained=int(len(eligible)),
            protected_candidates_retained=protected_retained,
        )
