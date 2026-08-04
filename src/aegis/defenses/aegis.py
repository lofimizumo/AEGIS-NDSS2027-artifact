"""Architecture-aware AEGIS gradient flooding."""

from __future__ import annotations

import torch
import torch.nn as nn

from aegis.models import (
    detect_architecture,
    get_block_attention_module,
    get_final_layer_norm,
    get_input_embedding,
    get_transformer_blocks,
)

from .base import Defense, GradientDefenseHook


class AegisDefense(Defense):
    """Freeze attention and flood embedding, attention, MLP, and norm gradients."""

    name = "aegis"

    def __init__(
        self,
        model: nn.Module | None = None,
        flood_scale: float = 1.0,
        real_token_retain: float = 0.3,
        mlp_flood_scale: float = 8.0,
        mlp_retain: float = 0.3,
        ln_flood_scale: float = 4.0,
    ) -> None:
        self.model = model
        self.flood_scale = flood_scale
        self.real_token_retain = real_token_retain
        self.mlp_flood_scale = mlp_flood_scale
        self.mlp_retain = mlp_retain
        self.ln_flood_scale = ln_flood_scale
        self.architecture = detect_architecture(model) if model is not None else None

    def _require_model(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("AegisDefense is not bound to a model; call pre_train(model) first")
        return self.model

    def pre_train(self, model: nn.Module) -> None:
        self.model = model
        self.architecture = detect_architecture(model)
        self.apply_defense_pre_training_to(model)

    def apply_defense_pre_training(self) -> None:
        """Apply one-time model changes to the bound model."""

        self.apply_defense_pre_training_to(self._require_model())

    def apply_defense_pre_training_to(
        self,
        target: nn.Module,
        note: str = "",
    ) -> None:
        """Freeze every parameter in each transformer block's attention module."""

        count = 0
        for block in get_transformer_blocks(target):
            attention = get_block_attention_module(block)
            if attention is None:
                continue
            for parameter in attention.parameters():
                parameter.requires_grad = False
                count += 1

        trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
        total = sum(p.numel() for p in target.parameters())
        percentage = 100 * trainable / total if total else 0.0
        extra = f" {note}" if note else ""
        print(
            f"  [AEGIS] frozen {count} attn tensors{extra}  "
            f"trainable {trainable:,}/{total:,} ({percentage:.1f}%)"
        )

    def after_backward(self, model: nn.Module) -> None:
        self.flood_gradients(model)

    def attack_hook(self, model: nn.Module) -> GradientDefenseHook:
        if self.model is None:
            self.model = model
            self.architecture = detect_architecture(model)
        return self

    def flood_gradients(self, target: nn.Module | None = None) -> None:
        """Flood gradients on the supplied or bound model."""

        model = target if target is not None else self._require_model()
        self._flood_embedding(model)
        self._flood_attention_projection_grads(model)
        self._flood_all_mlp(model)
        self._flood_layernorm(model)
        self._flood_lm_head(model)

    def _flood_embedding(self, model: nn.Module) -> None:
        embedding = get_input_embedding(model)
        if embedding is None or embedding.weight.grad is None:
            return

        gradient = embedding.weight.grad.float()
        row_norms = torch.norm(gradient, dim=1)
        max_norm = row_norms.max().item()
        if max_norm <= 0:
            return
        active_mask = row_norms > max_norm * 0.01
        if not active_mask.any():
            return

        mean_norm = row_norms[active_mask].mean().item()
        clip_threshold = 1.5 * mean_norm
        scales = torch.clamp(
            clip_threshold / (row_norms[active_mask] + 1e-12),
            max=1.0,
        )
        gradient[active_mask] *= scales.unsqueeze(1)
        gradient[active_mask] *= self.real_token_retain
        gradient += torch.randn_like(gradient) * mean_norm * self.flood_scale
        embedding.weight.grad.copy_(gradient.to(embedding.weight.grad.dtype))

    def _flood_attention_projection_grads(self, model: nn.Module) -> None:
        for block in get_transformer_blocks(model):
            attention = get_block_attention_module(block)
            if attention is None:
                continue
            for module in attention.modules():
                if isinstance(module, nn.LayerNorm):
                    continue
                weight = getattr(module, "weight", None)
                if weight is None or weight.grad is None or weight.dim() < 2:
                    continue
                gradient = weight.grad.float()
                if gradient.norm().item() < 1e-10:
                    continue
                gradient_std = gradient.std().item()
                gradient *= self.mlp_retain
                gradient += torch.randn_like(gradient) * gradient_std * self.mlp_flood_scale
                weight.grad.copy_(gradient.to(weight.grad.dtype))

    def _flood_all_mlp(self, model: nn.Module) -> None:
        for block in get_transformer_blocks(model):
            candidates: list[nn.Module] = []
            if hasattr(block, "mlp"):
                candidates = list(block.mlp.modules())
            elif hasattr(block, "fc1"):
                candidates = [block]
            elif hasattr(block, "intermediate"):
                candidates = list(block.intermediate.modules())
                output = getattr(block, "output", None)
                if output is not None:
                    candidates += list(output.modules())

            for module in candidates:
                weight = getattr(module, "weight", None)
                if weight is None or weight.grad is None or weight.dim() != 2:
                    continue
                gradient = weight.grad.float()
                if gradient.norm().item() < 1e-10:
                    continue
                gradient_std = gradient.std().item()
                gradient = torch.clamp(
                    gradient,
                    min=-3.0 * gradient_std,
                    max=3.0 * gradient_std,
                )
                gradient *= self.mlp_retain
                gradient += torch.randn_like(gradient) * gradient_std * self.mlp_flood_scale
                weight.grad.copy_(gradient.to(weight.grad.dtype))

    def _flood_layernorm(self, model: nn.Module) -> None:
        targets: list[nn.Module] = list(get_transformer_blocks(model))
        final_norm = get_final_layer_norm(model)
        if final_norm is not None:
            targets.append(final_norm)

        visited: set[int] = set()
        for target in targets:
            for module in target.modules():
                if id(module) in visited or not isinstance(module, nn.LayerNorm):
                    continue
                visited.add(id(module))
                if module.weight is None or module.weight.grad is None:
                    continue
                gradient = module.weight.grad.float()
                gradient_std = max(gradient.std().item(), 1e-8)
                gradient *= 0.1
                gradient += torch.randn_like(gradient) * gradient_std * self.ln_flood_scale
                module.weight.grad.copy_(gradient.to(module.weight.grad.dtype))

    def _flood_lm_head(self, model: nn.Module) -> None:
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        output = (
            get_output_embeddings()
            if callable(get_output_embeddings)
            else getattr(model, "lm_head", None)
        )
        if (
            output is None
            or not hasattr(output, "weight")
            or output.weight.grad is None
            or output.weight.grad.dim() != 2
        ):
            return

        try:
            embedding = get_input_embedding(model)
            if embedding is not None and output.weight.data_ptr() == embedding.weight.data_ptr():
                return
        except (AttributeError, RuntimeError):
            pass

        gradient = output.weight.grad.float()
        row_norms = torch.norm(gradient, dim=1)
        max_norm = row_norms.max().item()
        if max_norm <= 0:
            return
        active_mask = row_norms > max_norm * 0.01
        if not active_mask.any():
            return

        mean_norm = row_norms[active_mask].mean().item()
        clip_threshold = 1.5 * mean_norm
        scales = torch.clamp(
            clip_threshold / (row_norms[active_mask] + 1e-12),
            max=1.0,
        )
        gradient[active_mask] *= scales.unsqueeze(1)
        gradient[active_mask] *= self.real_token_retain
        gradient += torch.randn_like(gradient) * mean_norm * self.flood_scale
        output.weight.grad.copy_(gradient.to(output.weight.grad.dtype))


__all__ = ["AegisDefense"]
