"""fastai integration for the Human-Centered-xAI dropout AAE."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn


DEFAULT_HUMAN_CENTERED_XAI_MODEL = (
    Path(__file__).resolve().parents[2] / "Human-Centered-xAI" / "modelAAE_DROPOUT.py"
)


@dataclass
class HumanCenteredAAEConfig:
    """Constructor arguments used by `modelAAE_DROPOUT.AAE`."""

    input_size: int = 256
    input_channels: int = 3
    encoding_dims: int = 128
    classes: int = 2
    gen_train: bool = True
    skip_dropout: float = 1.0

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FastaiLatentResult:
    """Latents and predictions extracted from a fastai `Learner`."""

    latents: Tensor
    preds: Tensor | None = None
    targets: Tensor | None = None
    vocab: Any | None = None


@dataclass
class HumanCenteredFeatures:
    """Feature tensors exposed by the Human-Centered AAE."""

    logits: Tensor
    latents: Tensor
    encoder_features: Tensor
    reconstruction: Tensor | None = None


def resolve_human_centered_aae_path(model_path: str | Path | None = None) -> Path:
    """Resolve the absolute path to `modelAAE_DROPOUT.py`."""

    if model_path is None:
        model_path = os.getenv("HUMAN_CENTERED_XAI_MODEL_PATH")
    path = Path(model_path) if model_path is not None else DEFAULT_HUMAN_CENTERED_XAI_MODEL
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            "Cannot find modelAAE_DROPOUT.py. Pass `model_path` or set "
            "`HUMAN_CENTERED_XAI_MODEL_PATH`."
        )
    return path


def load_human_centered_aae_module(model_path: str | Path | None = None) -> ModuleType:
    """Import `Human-Centered-xAI/modelAAE_DROPOUT.py` as a Python module."""

    path = resolve_human_centered_aae_path(model_path)
    module_name = f"tell_me_why_human_centered_aae_{abs(hash(path))}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        missing = error.name or str(error)
        raise ModuleNotFoundError(
            f"`{path}` requires `{missing}`. Install the fastai environment "
            "declared by this package before loading the AAE."
        ) from error
    return module


def build_human_centered_aae(
    config: HumanCenteredAAEConfig | None = None,
    *,
    model_path: str | Path | None = None,
) -> nn.Module:
    """Instantiate the `AAE` class from `modelAAE_DROPOUT.py`."""

    module = load_human_centered_aae_module(model_path)
    if not hasattr(module, "AAE"):
        raise AttributeError(f"{resolve_human_centered_aae_path(model_path)} has no AAE class.")
    return module.AAE(**(config or HumanCenteredAAEConfig()).to_kwargs())


def _infer_out_features(module: nn.Module) -> int | None:
    if hasattr(module, "out_features"):
        return int(module.out_features)
    for child in reversed(list(module.modules())):
        if child is module:
            continue
        if hasattr(child, "out_features"):
            return int(child.out_features)
    return None


def graft_classifier_to_human_aae(
    aae: nn.Module,
    classifier: nn.Module,
    *,
    classifier_attr: str = "linear",
    freeze_aae_body: bool = True,
    update_classes: bool = True,
) -> nn.Module:
    """Replace the AAE classification head with a developer supplied classifier.

    `modelAAE_DROPOUT.AAE.forward` computes `self.zi` and then calls
    `self.linear(self.zi)`. Replacing `linear` preserves the fastai training
    loop, losses, metrics, reconstruction output, and adversarial head.
    """

    setattr(aae, classifier_attr, classifier)

    if update_classes:
        out_features = _infer_out_features(classifier)
        if out_features is not None:
            aae.classes = out_features

    if freeze_aae_body:
        prefix = f"{classifier_attr}."
        for name, parameter in aae.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))

    return aae


def make_grafted_learner(
    dls: Any,
    classifier: nn.Module,
    *,
    aae: nn.Module | None = None,
    config: HumanCenteredAAEConfig | None = None,
    model_path: str | Path | None = None,
    pretrained_weights: str | None = None,
    loss_func: Any | None = None,
    metrics: list[Any] | None = None,
    freeze_aae_body: bool = True,
    load_strict: bool = False,
    **learner_kwargs: Any,
) -> Any:
    """Create a fastai `Learner` using the Human-Centered AAE plus a new head."""

    from fastai.vision.all import CrossEntropyLossFlat, Learner, accuracy

    model = aae if aae is not None else build_human_centered_aae(config, model_path=model_path)
    resolved_loss = loss_func or CrossEntropyLossFlat()
    resolved_metrics = metrics if metrics is not None else [accuracy]

    if pretrained_weights is not None:
        preload = Learner(dls, model, loss_func=resolved_loss, metrics=resolved_metrics)
        preload.load(pretrained_weights, strict=load_strict)

    model = graft_classifier_to_human_aae(
        model,
        classifier,
        freeze_aae_body=freeze_aae_body,
    )
    return Learner(
        dls,
        model,
        loss_func=resolved_loss,
        metrics=resolved_metrics,
        **learner_kwargs,
    )


class CollectLatentSpaceCallback:
    """fastai callback collecting `learn.model.zi` during validation/test."""

    order = 60

    def before_validate(self) -> None:
        self.learn.zi_valid = []

    def before_batch(self) -> None:
        if not self.training and not hasattr(self.learn, "zi_valid"):
            self.learn.zi_valid = []

    def after_batch(self) -> None:
        if self.training:
            return
        zi = getattr(self.learn.model, "zi", None)
        if zi is not None:
            self.learn.zi_valid.append(zi.detach().cpu())

    def after_validate(self) -> None:
        zi_valid = getattr(self.learn, "zi_valid", [])
        if isinstance(zi_valid, list):
            self.learn.zi_valid = torch.cat(zi_valid) if zi_valid else torch.empty(0)


def _as_fastai_callback(callback: CollectLatentSpaceCallback) -> Any:
    from fastai.callback.core import Callback

    if isinstance(callback, Callback):
        return callback

    class _FastaiCollectLatentSpaceCallback(Callback):
        order = callback.order

        def before_validate(self) -> None:
            callback.learn = self.learn
            callback.before_validate()

        def before_batch(self) -> None:
            callback.learn = self.learn
            callback.training = self.training
            callback.before_batch()

        def after_batch(self) -> None:
            callback.learn = self.learn
            callback.training = self.training
            callback.after_batch()

        def after_validate(self) -> None:
            callback.learn = self.learn
            callback.after_validate()

    return _FastaiCollectLatentSpaceCallback()


def extract_latents_from_learner(
    learn: Any,
    *,
    dl: Any | None = None,
    ds_idx: int = 1,
    with_decoded: bool = False,
) -> FastaiLatentResult:
    """Run `learn.get_preds` and return the latent `zi` collected from the AAE."""

    callback = _as_fastai_callback(CollectLatentSpaceCallback())
    outputs = learn.get_preds(dl=dl, ds_idx=ds_idx, with_decoded=with_decoded, cbs=[callback])
    preds = outputs[0] if len(outputs) > 0 else None
    targets = outputs[1] if len(outputs) > 1 else None
    latents = getattr(learn, "zi_valid", torch.empty(0))
    vocab = getattr(getattr(learn, "dls", None), "vocab", None)
    return FastaiLatentResult(latents=latents, preds=preds, targets=targets, vocab=vocab)


@torch.no_grad()
def extract_human_aae_features(
    aae: nn.Module,
    inputs: Tensor,
    *,
    device: torch.device | str | None = None,
) -> HumanCenteredFeatures:
    """Return latent vectors, encoder feature maps, logits, and reconstruction."""

    was_training = aae.training
    aae.eval()
    if device is not None:
        aae.to(device)
        inputs = inputs.to(device)

    encoder_features = aae.unet.layers[0](inputs)
    logits = aae(inputs)
    latents = aae.zi
    reconstruction = getattr(aae, "decoder_output", None)
    aae.train(was_training)

    return HumanCenteredFeatures(
        logits=logits.detach().cpu(),
        latents=latents.detach().cpu(),
        encoder_features=encoder_features.detach().cpu(),
        reconstruction=None if reconstruction is None else reconstruction.detach().cpu(),
    )
