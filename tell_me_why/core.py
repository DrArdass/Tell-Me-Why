"""Core adapters for combining an AAE encoder with a user classifier."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

BatchGetter = Callable[[Any], Any]


@dataclass
class LatentBatch:
    """Latent vectors collected from a dataloader."""

    latents: Tensor
    labels: Tensor | None = None

    def numpy(self) -> tuple[Any, Any | None]:
        """Return `(latents, labels)` as NumPy arrays."""

        labels = None if self.labels is None else self.labels.numpy()
        return self.latents.numpy(), labels


def _select_tensor(value: Any, index: int = 0) -> Tensor:
    """Extract the latent tensor from common encoder return shapes."""

    if isinstance(value, Tensor):
        return value
    if isinstance(value, dict):
        for key in ("z", "latent", "latents", "embedding", "features"):
            if key in value:
                return _select_tensor(value[key], index=index)
        raise KeyError(
            "Encoder returned a dict without one of: z, latent, latents, embedding, features."
        )
    if isinstance(value, (tuple, list)):
        return _select_tensor(value[index], index=index)
    raise TypeError(f"Cannot extract a latent Tensor from {type(value)!r}.")


def _move_to_device(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _detach_cpu(value: Any) -> Tensor:
    tensor = _select_tensor(value) if not isinstance(value, Tensor) else value
    return tensor.detach().cpu()


def freeze_module(module: nn.Module, trainable: bool = False) -> nn.Module:
    """Toggle gradient updates for every parameter of a module."""

    for parameter in module.parameters():
        parameter.requires_grad = trainable
    return module


class AAEAdapter(nn.Module):
    """Expose a stable `encode` method for an arbitrary AAE implementation.

    The adapter supports the most common conventions:
    - an AAE with an `encode(x)` method;
    - an AAE with an `encoder` submodule;
    - an explicit encoder attribute such as `encoder_q`.
    """

    def __init__(
        self,
        aae: nn.Module,
        *,
        encode_method: str = "encode",
        encoder_attr: str | None = None,
        latent_index: int = 0,
    ) -> None:
        super().__init__()
        self.aae = aae
        self.encode_method = encode_method
        self.encoder_attr = encoder_attr
        self.latent_index = latent_index

    def encode(self, inputs: Any, *args: Any, **kwargs: Any) -> Tensor:
        """Return the latent representation produced by the wrapped AAE."""

        if self.encode_method and hasattr(self.aae, self.encode_method):
            encode = getattr(self.aae, self.encode_method)
            if callable(encode):
                return _select_tensor(encode(inputs, *args, **kwargs), self.latent_index)

        encoder_attr = self.encoder_attr or "encoder"
        if hasattr(self.aae, encoder_attr):
            encoder = getattr(self.aae, encoder_attr)
            return _select_tensor(encoder(inputs, *args, **kwargs), self.latent_index)

        raise AttributeError(
            "Could not find an encoder. Pass `encode_method` or `encoder_attr` "
            "matching your AAE implementation."
        )

    def forward(self, inputs: Any, *args: Any, **kwargs: Any) -> Tensor:
        return self.encode(inputs, *args, **kwargs)


class GraftedAAEClassifier(nn.Module):
    """Combine an AAE encoder with a classifier supplied by the developer."""

    def __init__(
        self,
        aae: nn.Module | AAEAdapter,
        classifier: nn.Module,
        *,
        freeze_aae: bool = True,
        adapter_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.aae_adapter = (
            aae
            if isinstance(aae, AAEAdapter)
            else AAEAdapter(aae, **(adapter_kwargs or {}))
        )
        self.classifier = classifier

        if freeze_aae:
            freeze_module(self.aae_adapter, trainable=False)

    def encode(self, inputs: Any, *args: Any, **kwargs: Any) -> Tensor:
        return self.aae_adapter.encode(inputs, *args, **kwargs)

    def forward(
        self,
        inputs: Any,
        *,
        return_latent: bool = False,
        **encode_kwargs: Any,
    ) -> Tensor | tuple[Tensor, Tensor]:
        latents = self.encode(inputs, **encode_kwargs)
        logits = self.classifier(latents)
        if return_latent:
            return logits, latents
        return logits

    @torch.no_grad()
    def predict(self, inputs: Any, **encode_kwargs: Any) -> Tensor:
        """Return class ids from classifier logits."""

        logits = self.forward(inputs, **encode_kwargs)
        if logits.ndim == 1 or logits.shape[-1] == 1:
            return (logits.reshape(-1) > 0).long()
        return logits.argmax(dim=-1)


def default_input_getter(batch: Any) -> Any:
    """Extract model inputs from common dataloader batch formats."""

    if isinstance(batch, dict):
        for key in ("x", "input", "inputs", "image", "images", "features"):
            if key in batch:
                return batch[key]
        raise KeyError(
            "Could not infer inputs from batch dict. Pass a custom `input_getter`."
        )
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def default_target_getter(batch: Any) -> Any | None:
    """Extract labels from common dataloader batch formats when available."""

    if isinstance(batch, dict):
        for key in ("y", "label", "labels", "target", "targets"):
            if key in batch:
                return batch[key]
        return None
    if isinstance(batch, (tuple, list)) and len(batch) > 1:
        return batch[1]
    return None


def collect_latents(
    model: nn.Module,
    dataloader: Iterable[Any],
    *,
    device: torch.device | str | None = None,
    input_getter: BatchGetter = default_input_getter,
    target_getter: BatchGetter | None = default_target_getter,
    max_batches: int | None = None,
) -> LatentBatch:
    """Encode batches and concatenate their latent vectors for visualization."""

    was_training = model.training
    model.eval()
    if device is not None:
        model.to(device)

    latents: list[Tensor] = []
    labels: list[Tensor] = []

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                inputs = input_getter(batch)
                if device is not None:
                    inputs = _move_to_device(inputs, device)

                if hasattr(model, "encode"):
                    encoded = model.encode(inputs)
                else:
                    output = model(inputs)
                    encoded = getattr(model, "zi", output)
                latents.append(_detach_cpu(encoded))

                if target_getter is not None:
                    target = target_getter(batch)
                    if target is not None:
                        labels.append(_detach_cpu(target))
    finally:
        model.train(was_training)

    if not latents:
        raise ValueError("No latent vectors were collected from the dataloader.")

    labels_tensor = torch.cat(labels) if labels else None
    return LatentBatch(latents=torch.cat(latents), labels=labels_tensor)
