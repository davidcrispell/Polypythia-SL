from __future__ import annotations

from typing import Any

import torch


def zeropower_via_newton_schulz(
    matrix: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    """Approximate the polar factor used by Muon with a quintic iteration."""
    if matrix.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(matrix.shape)}")
    if steps < 1:
        raise ValueError("Newton-Schulz steps must be positive")

    a, b, c = 3.4445, -4.7750, 2.0315
    original_dtype = matrix.dtype
    work = matrix.float()
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.mT
    work = work / (work.norm() + 1e-7)
    for _ in range(steps):
        gram = work @ work.mT
        polynomial = b * gram + c * (gram @ gram)
        work = a * work + polynomial @ work
    if transposed:
        work = work.mT
    return work.to(original_dtype)


def _is_muon_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    lower_name = name.lower()
    return (
        parameter.ndim == 2
        and "embed" not in lower_name
        and "lm_head" not in lower_name
    )


class HybridMuon(torch.optim.Optimizer):
    """Single-device Muon for hidden matrices plus AdamW for auxiliary tensors."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        muon_lr: float,
        aux_lr: float,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        aux_betas: tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8,
    ) -> None:
        muon_parameters = []
        auxiliary_parameters = []
        muon_names = []
        auxiliary_names = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if _is_muon_parameter(name, parameter):
                muon_parameters.append(parameter)
                muon_names.append(name)
            else:
                auxiliary_parameters.append(parameter)
                auxiliary_names.append(name)
        if not muon_parameters or not auxiliary_parameters:
            raise ValueError("Hybrid Muon requires both matrix and auxiliary parameters")

        parameter_groups = [
            {
                "params": muon_parameters,
                "algorithm": "muon",
                "lr": muon_lr,
                "momentum": momentum,
                "ns_steps": ns_steps,
                "nesterov": nesterov,
                "weight_decay": weight_decay,
            },
            {
                "params": auxiliary_parameters,
                "algorithm": "adamw",
                "lr": aux_lr,
                "betas": aux_betas,
                "eps": aux_eps,
                "weight_decay": weight_decay,
            },
        ]
        super().__init__(parameter_groups, defaults={})
        self.group_metadata = {
            "muon_parameter_names": muon_names,
            "auxiliary_parameter_names": auxiliary_names,
            "muon_parameter_count": sum(parameter.numel() for parameter in muon_parameters),
            "auxiliary_parameter_count": sum(
                parameter.numel() for parameter in auxiliary_parameters
            ),
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        beta = float(group["momentum"])
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            state = self.state[parameter]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
            momentum = state["momentum_buffer"]
            momentum.lerp_(gradient, 1.0 - beta)
            update = (
                torch.lerp(gradient, momentum, beta)
                if group["nesterov"]
                else momentum
            )
            update = zeropower_via_newton_schulz(update, int(group["ns_steps"]))
            update = update * max(
                1.0, parameter.shape[-2] / parameter.shape[-1]
            ) ** 0.5
            learning_rate = float(group["lr"])
            parameter.mul_(1.0 - learning_rate * float(group["weight_decay"]))
            parameter.add_(update, alpha=-learning_rate)

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("HybridMuon does not support sparse gradients")
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                gradient, gradient, value=1.0 - beta2
            )

            step = state["step"]
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5)
            denominator.add_(float(group["eps"]))
            learning_rate = float(group["lr"])
            parameter.mul_(1.0 - learning_rate * float(group["weight_decay"]))
            parameter.addcdiv_(
                exp_avg,
                denominator,
                value=-learning_rate / bias_correction1,
            )


def build_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    if optimizer_name == "adamw":
        # Project rule (2026-07-11): match EleutherAI's Pythia pretraining
        # optimizer geometry (Adam betas [0.9, 0.95], eps 1e-8) so fine-tuning
        # preconditioning is consistent with the landscape the base model was
        # trained under. Weight decay stays config-driven; Pythia used 0.1.
        betas = tuple(config.get("betas", (0.9, 0.95)))
        eps = float(config.get("eps", 1e-8))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            betas=betas,
            eps=eps,
            weight_decay=float(config["weight_decay"]),
        )
        return optimizer, {
            "name": "adamw",
            "learning_rate": float(config["learning_rate"]),
            "betas": list(betas),
            "eps": eps,
        }
    if optimizer_name != "muon":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    if "muon_learning_rate" not in config:
        raise ValueError("Muon training requires an explicit muon_learning_rate")

    optimizer = HybridMuon(
        model,
        muon_lr=float(config["muon_learning_rate"]),
        aux_lr=float(config["learning_rate"]),
        momentum=float(config.get("momentum", 0.95)),
        ns_steps=int(config.get("newton_schulz_steps", 5)),
        nesterov=bool(config.get("nesterov", True)),
        weight_decay=float(config["weight_decay"]),
        aux_betas=tuple(config.get("aux_betas", (0.9, 0.999))),
        aux_eps=float(config.get("aux_eps", 1e-8)),
    )
    return optimizer, {
        "name": "hybrid_muon",
        "muon_learning_rate": float(config["muon_learning_rate"]),
        "aux_learning_rate": float(config["learning_rate"]),
        "momentum": float(config.get("momentum", 0.95)),
        "newton_schulz_steps": int(config.get("newton_schulz_steps", 5)),
        "nesterov": bool(config.get("nesterov", True)),
        **optimizer.group_metadata,
    }
