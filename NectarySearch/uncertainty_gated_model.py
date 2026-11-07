from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Independent, Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

try:
    from model import AllocentricNectarySearchModel
    from uncertainty_reduction import (
        compute_action_smoothness_penalty,
        estimate_expected_information_gain,
        posterior_entropy,
    )
except ImportError:
    from .model import AllocentricNectarySearchModel
    from .uncertainty_reduction import (
        compute_action_smoothness_penalty,
        estimate_expected_information_gain,
        posterior_entropy,
    )


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    output_activation: Optional[type[nn.Module]] = None,
) -> nn.Sequential:
    dims = [input_dim, *hidden_dims]
    layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.extend([nn.Linear(in_dim, out_dim), activation()])
    layers.append(nn.Linear(dims[-1], output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def _discounted_cumsum(x: torch.Tensor, discount: float) -> torch.Tensor:
    out = torch.zeros_like(x)
    running = torch.zeros_like(x[..., -1])
    for t in range(x.shape[-1] - 1, -1, -1):
        running = x[..., t] + discount * running
        out[..., t] = running
    return out


@dataclass
class ActionSelection:
    action: torch.Tensor
    belief_features: torch.Tensor
    exploration_action: torch.Tensor
    exploration_log_prob: Optional[torch.Tensor]
    exploration_value: Optional[torch.Tensor]
    exploitation_action: torch.Tensor
    selected_is_exploration: torch.Tensor
    uncertainty: torch.Tensor
    explore_probability: torch.Tensor
    threshold: torch.Tensor
    gate_log_prob: Optional[torch.Tensor]
    gate_value: Optional[torch.Tensor]
    gate_entropy: Optional[torch.Tensor]
    exploitation_log_prob: Optional[torch.Tensor]
    exploitation_value: torch.Tensor


def apply_stateful_fixed_threshold_gate(
    selection: ActionSelection,
    previous_is_exploration: Optional[torch.Tensor],
    previous_smoothed_uncertainty: Optional[torch.Tensor],
    uncertainty_ema_alpha: Optional[float] = None,
    hysteresis: float = 0.0,
) -> tuple[ActionSelection, torch.Tensor, torch.Tensor]:
    """Apply EMA and hysteresis to an already-computed fixed-threshold gate choice."""

    hysteresis = float(hysteresis)
    if hysteresis < 0.0:
        raise ValueError("hysteresis must be non-negative.")
    alpha = None if uncertainty_ema_alpha is None else float(uncertainty_ema_alpha)
    if alpha is not None and (alpha < 0.0 or alpha >= 1.0):
        raise ValueError("uncertainty_ema_alpha must be in [0, 1).")

    raw_uncertainty = selection.uncertainty.detach().reshape(-1)
    if previous_smoothed_uncertainty is None or alpha is None:
        smoothed_uncertainty = raw_uncertainty
    else:
        previous_smoothed_uncertainty = previous_smoothed_uncertainty.to(
            device=raw_uncertainty.device,
            dtype=raw_uncertainty.dtype,
        ).reshape(-1)
        smoothed_uncertainty = (
            alpha * previous_smoothed_uncertainty
            + (1.0 - alpha) * raw_uncertainty
        )

    threshold = selection.threshold.detach().to(
        device=raw_uncertainty.device,
        dtype=raw_uncertainty.dtype,
    )
    lower_threshold = threshold - hysteresis
    upper_threshold = threshold + hysteresis

    if previous_is_exploration is None:
        selected_is_exploration = smoothed_uncertainty > lower_threshold
    else:
        previous_is_exploration = previous_is_exploration.to(
            device=raw_uncertainty.device,
            dtype=torch.bool,
        ).reshape(-1)
        selected_is_exploration = torch.where(
            previous_is_exploration,
            smoothed_uncertainty > lower_threshold,
            smoothed_uncertainty > upper_threshold,
        )

    selected_is_exploration = selected_is_exploration.reshape(
        selection.selected_is_exploration.shape
    )
    selection.selected_is_exploration = selected_is_exploration
    selection.action = torch.where(
        selected_is_exploration.unsqueeze(-1),
        selection.exploration_action,
        selection.exploitation_action,
    )
    return (
        selection,
        selected_is_exploration.detach().reshape(-1),
        smoothed_uncertainty.detach().reshape(-1),
    )


@dataclass
class RolloutBatch:
    belief_features: torch.Tensor
    prev_actions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor
    executed_actions: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    old_log_probs: Optional[torch.Tensor] = None
    action_mask: Optional[torch.Tensor] = None
    exploration_values: Optional[torch.Tensor] = None
    exploitation_values: Optional[torch.Tensor] = None
    raw_uncertainties: Optional[torch.Tensor] = None
    smoothed_uncertainties: Optional[torch.Tensor] = None


@dataclass
class GateRolloutBatch:
    belief_features: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor
    advantages: Optional[torch.Tensor] = None
    old_log_probs: Optional[torch.Tensor] = None


class DeterministicActionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        action_delta_low: torch.Tensor,
        action_delta_high: torch.Tensor,
        hidden_dims: Sequence[int] = (128, 128),
    ):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dims, action_low.numel())
        self.register_buffer("action_low", action_low.clone().detach().float())
        self.register_buffer("action_high", action_high.clone().detach().float())
        self.register_buffer(
            "action_delta_low",
            action_delta_low.clone().detach().float(),
        )
        self.register_buffer(
            "action_delta_high",
            action_delta_high.clone().detach().float(),
        )

    def action_bounds(self, prev_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower = torch.maximum(self.action_low, prev_action + self.action_delta_low)
        upper = torch.minimum(self.action_high, prev_action + self.action_delta_high)
        return lower, upper

    def squash(self, raw_action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        action_low, action_high = self.action_bounds(prev_action)
        action_bias = 0.5 * (action_low + action_high)
        action_scale = 0.5 * (action_high - action_low)
        return torch.tanh(raw_action) * action_scale + action_bias

    def forward(self, features: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        raw_action = self.net(features)
        return self.squash(raw_action, prev_action)


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        action_delta_low: torch.Tensor,
        action_delta_high: torch.Tensor,
        hidden_dims: Sequence[int] = (128, 128),
        init_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.backbone = _build_mlp(input_dim, hidden_dims, hidden_dims[-1])
        self.mean_head = nn.Linear(hidden_dims[-1], action_low.numel())
        self.log_std = nn.Parameter(
            torch.full((action_low.numel(),), float(init_log_std), dtype=torch.float32)
        )
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.register_buffer("action_low", action_low.clone().detach().float())
        self.register_buffer("action_high", action_high.clone().detach().float())
        self.register_buffer(
            "action_delta_low",
            action_delta_low.clone().detach().float(),
        )
        self.register_buffer(
            "action_delta_high",
            action_delta_high.clone().detach().float(),
        )

    def action_bounds(self, prev_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower = torch.maximum(self.action_low, prev_action + self.action_delta_low)
        upper = torch.minimum(self.action_high, prev_action + self.action_delta_high)
        return lower, upper

    def squash(self, raw_action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        action_low, action_high = self.action_bounds(prev_action)
        action_bias = 0.5 * (action_low + action_high)
        action_scale = 0.5 * (action_high - action_low)
        return torch.tanh(raw_action) * action_scale + action_bias

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        raw_mean = self.mean_head(hidden)
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        log_std = log_std.expand_as(raw_mean)
        return raw_mean, log_std

    def distribution(
        self,
        features: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> TransformedDistribution:
        raw_mean, log_std = self(features)
        action_low, action_high = self.action_bounds(prev_action)
        action_bias = 0.5 * (action_low + action_high)
        action_scale = 0.5 * (action_high - action_low)
        base_dist = Independent(Normal(raw_mean, log_std.exp()), 1)
        return TransformedDistribution(
            base_dist,
            [
                TanhTransform(cache_size=1),
                AffineTransform(loc=action_bias, scale=action_scale),
            ],
        )

    @staticmethod
    def _squashed_log_prob(
        raw_action: torch.Tensor,
        squashed_action: torch.Tensor,
        raw_mean: torch.Tensor,
        log_std: torch.Tensor,
        action_scale: torch.Tensor,
    ) -> torch.Tensor:
        normal = Normal(raw_mean, log_std.exp())
        tanh_log_abs_det = TanhTransform(cache_size=0).log_abs_det_jacobian(
            raw_action,
            squashed_action,
        )
        scale_log_abs_det = torch.log(
            action_scale.clamp_min(torch.finfo(action_scale.dtype).eps)
        )
        return (
            normal.log_prob(raw_action)
            - tanh_log_abs_det
            - scale_log_abs_det
        ).sum(dim=-1)

    def approx_entropy(
        self,
        features: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, log_std = self(features)
        entropy = Independent(Normal(torch.zeros_like(log_std), log_std.exp()), 1).entropy()
        if prev_action is None:
            return entropy
        action_low, action_high = self.action_bounds(prev_action)
        action_scale = 0.5 * (action_high - action_low)
        return entropy + torch.log(action_scale.clamp_min(1e-6)).sum(dim=-1)

    def sample(
        self,
        features: torch.Tensor,
        prev_action: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        raw_mean, log_std = self(features)
        action_low, action_high = self.action_bounds(prev_action)
        action_bias = 0.5 * (action_low + action_high)
        action_scale = 0.5 * (action_high - action_low)
        if deterministic:
            return (
                torch.tanh(raw_mean) * action_scale + action_bias,
                None,
                self.approx_entropy(features, prev_action),
            )

        raw_action = Normal(raw_mean, log_std.exp()).rsample()
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * action_scale + action_bias
        log_prob = self._squashed_log_prob(
            raw_action=raw_action,
            squashed_action=squashed_action,
            raw_mean=raw_mean,
            log_std=log_std,
            action_scale=action_scale,
        )
        entropy = self.approx_entropy(features, prev_action)
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_mean, log_std = self(features)
        action_low, action_high = self.action_bounds(prev_action)
        action_bias = 0.5 * (action_low + action_high)
        action_scale = 0.5 * (action_high - action_low)
        safe_scale = action_scale.clamp_min(torch.finfo(actions.dtype).eps)
        support_eps = 10.0 * torch.finfo(actions.dtype).eps
        squashed_action = ((actions - action_bias) / safe_scale).clamp(
            min=-1.0 + support_eps,
            max=1.0 - support_eps,
        )
        raw_action = torch.atanh(squashed_action)
        log_prob = self._squashed_log_prob(
            raw_action=raw_action,
            squashed_action=squashed_action,
            raw_mean=raw_mean,
            log_std=log_std,
            action_scale=action_scale,
        )
        entropy = self.approx_entropy(features, prev_action)
        return log_prob, entropy


class ActorCriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        action_delta_low: torch.Tensor,
        action_delta_high: torch.Tensor,
        policy_hidden_dims: Sequence[int] = (128, 128),
        value_hidden_dims: Sequence[int] = (128, 128),
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.actor = SquashedGaussianActor(
            input_dim=input_dim,
            action_low=action_low,
            action_high=action_high,
            action_delta_low=action_delta_low,
            action_delta_high=action_delta_high,
            hidden_dims=policy_hidden_dims,
            init_log_std=init_log_std,
        )
        self.critic = _build_mlp(input_dim, value_hidden_dims, 1)

    def act(
        self,
        features: torch.Tensor,
        prev_action: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        action, log_prob, entropy = self.actor.sample(
            features,
            prev_action,
            deterministic=deterministic,
        )
        value = self.critic(features).squeeze(-1)
        return action, log_prob, value, entropy

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob, entropy = self.actor.evaluate_actions(features, actions, prev_action)
        value = self.critic(features).squeeze(-1)
        return log_prob, value, entropy


class BernoulliGateActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
    ):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dims, 1)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def distribution(self, features: torch.Tensor) -> Bernoulli:
        return Bernoulli(logits=self.logits(features))

    def sample(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        dist = self.distribution(features)
        probs = dist.probs
        entropy = dist.entropy()
        if deterministic:
            action = (probs >= 0.5).to(dtype=torch.float32)
            return action, None, entropy, probs
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, entropy, probs

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, dist.probs


class GateActorCriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        policy_hidden_dims: Sequence[int] = (128, 128),
        value_hidden_dims: Sequence[int] = (128, 128),
    ):
        super().__init__()
        self.actor = BernoulliGateActor(
            input_dim=input_dim,
            hidden_dims=policy_hidden_dims,
        )
        self.critic = _build_mlp(input_dim, value_hidden_dims, 1)

    def act(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, entropy, probs = self.actor.sample(
            features,
            deterministic=deterministic,
        )
        value = self.critic(features).squeeze(-1)
        return action, log_prob, value, entropy, probs

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob, entropy, probs = self.actor.evaluate_actions(features, actions)
        value = self.critic(features).squeeze(-1)
        return log_prob, value, entropy, probs


class UncertaintyGatedAllocentricNectarySearchModel(AllocentricNectarySearchModel):
    """
    Allocentric perception model augmented with:

    1. An exploration pathway that can either maximize differentiable expected
       information gain directly or maximize realized information gain with RL.
    2. A pluggable continuous-action RL exploitation policy/value module.
    3. A gate that can either be a fixed uncertainty threshold or a learned
       binary actor-critic head over belief features.

    The perception pathway is inherited from AllocentricNectarySearchModel, so
    existing reconstruction/KL training remains available through forward().
    """

    SUPPORTED_RL_ALGORITHMS = {"reinforce", "a2c", "ppo"}
    SUPPORTED_EXPLORATION_OBJECTIVES = {
        "expected_information_gain",
        "realized_information_gain",
    }
    SUPPORTED_GATE_MODES = {"fixed_threshold", "rl_head"}
    SUPPORTED_ACTION_SELECTION_STRATEGIES = {
        "uncertainty_gate",
        "epsilon_greedy",
        "fixed_switching",
        "random_switching",
        "non_lexicographic",
    }
    SUPPORTED_UNCERTAINTY_METRICS = {
        "aggregate",
        "average_unimodal",
        "weighted_unimodal",
        "mech",
        "vis",
    }

    @classmethod
    def _normalize_uncertainty_metric(cls, uncertainty_metric: str) -> str:
        normalized = str(uncertainty_metric or "aggregate").strip().lower().replace("-", "_")
        aliases = {
            "joint": "aggregate",
            "aggregate_posterior": "aggregate",
            "average": "average_unimodal",
            "mean_unimodal": "average_unimodal",
            "weighted": "weighted_unimodal",
            "moe_weighted": "weighted_unimodal",
            "mechanosensory": "mech",
            "visual": "vis",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.SUPPORTED_UNCERTAINTY_METRICS:
            raise ValueError(
                f"Unsupported uncertainty_metric '{uncertainty_metric}'. "
                f"Expected one of {sorted(cls.SUPPORTED_UNCERTAINTY_METRICS)}."
            )
        return normalized

    @classmethod
    def _normalize_action_selection_strategy(cls, action_selection_strategy: str) -> str:
        normalized = (
            str(action_selection_strategy or "uncertainty_gate")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        aliases = {
            "gate": "uncertainty_gate",
            "gated": "uncertainty_gate",
            "uncertainty_gated": "uncertainty_gate",
            "epsilon": "epsilon_greedy",
            "eps_greedy": "epsilon_greedy",
            "random_exploration": "epsilon_greedy",
            "fixed_switch": "fixed_switching",
            "fixed": "fixed_switching",
            "random_switch": "random_switching",
            "stochastic_switching": "random_switching",
            "nonlexicographic": "non_lexicographic",
            "nonlex": "non_lexicographic",
            "combined_reward": "non_lexicographic",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.SUPPORTED_ACTION_SELECTION_STRATEGIES:
            raise ValueError(
                f"Unsupported action_selection_strategy '{action_selection_strategy}'. "
                f"Expected one of {sorted(cls.SUPPORTED_ACTION_SELECTION_STRATEGIES)}."
            )
        return normalized

    def __init__(
        self,
        mech_obs_dim: int,
        vis_obs_shape: Sequence[int],
        action_dim: int,
        mech_embedding_dim: Optional[int] = None,
        vis_embedding_dim: int = 64,
        fusion_dim: Optional[int] = None,
        latent_dim: int = 1,
        hidden_dim: int = 64,
        beta: float = 0.1,
        visual_likelihood: str = "gaussian",
        learn_decoder_variance: bool = True,
        posterior_fusion: str = "early",
        moment_matching: bool = True,
        rnn_type: str = "rnn",
        modality_rnn_mode: str = "shared",
        action_low: Optional[Sequence[float]] = None,
        action_high: Optional[Sequence[float]] = None,
        action_delta_low: Optional[Sequence[float]] = None,
        action_delta_high: Optional[Sequence[float]] = None,
        exploration_hidden_dims: Sequence[int] = (128, 128),
        exploitation_policy_hidden_dims: Sequence[int] = (128, 128),
        exploitation_value_hidden_dims: Sequence[int] = (128, 128),
        gate_hidden_dims: Sequence[int] = (128, 128),
        rl_algorithm: str = "ppo",
        gate_mode: str = "fixed_threshold",
        fixed_gate_threshold: Optional[float] = None,
        gating_temperature: float = 0.25,
        action_selection_strategy: str = "uncertainty_gate",
        epsilon_greedy_probability: float = 0.1,
        fixed_switch_timestep: Optional[int] = None,
        random_switch_probability: float = 0.1,
        information_gain_reward_coef: float = 1.0,
        exploration_objective: str = "expected_information_gain",
        uncertainty_metric: str = "aggregate",
        use_posterior_mean_for_planning: bool = True,
        exploration_action_l2: float = 0.0,
        ppo_clip_epsilon: float = 0.2,
        rl_value_coef: float = 0.5,
        rl_entropy_coef: float = 1e-3,
        gate_value_coef: Optional[float] = None,
        gate_entropy_coef: Optional[float] = None,
        exploration_mc_samples: int = 8,
        exploration_action_delta_huber_beta: float = 0.05,
    ):
        super().__init__(
            mech_obs_dim=mech_obs_dim,
            vis_obs_shape=tuple(vis_obs_shape),
            action_dim=action_dim,
            mech_embedding_dim=mech_embedding_dim,
            vis_embedding_dim=vis_embedding_dim,
            fusion_dim=fusion_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            beta=beta,
            visual_likelihood=visual_likelihood,
            learn_decoder_variance=learn_decoder_variance,
            posterior_fusion=posterior_fusion,
            moment_matching=moment_matching,
            rnn_type=rnn_type,
            modality_rnn_mode=modality_rnn_mode,
        )

        if rl_algorithm not in self.SUPPORTED_RL_ALGORITHMS:
            raise ValueError(
                f"Unsupported rl_algorithm '{rl_algorithm}'. "
                f"Expected one of {sorted(self.SUPPORTED_RL_ALGORITHMS)}."
            )
        if exploration_objective not in self.SUPPORTED_EXPLORATION_OBJECTIVES:
            raise ValueError(
                f"Unsupported exploration_objective '{exploration_objective}'. "
                f"Expected one of {sorted(self.SUPPORTED_EXPLORATION_OBJECTIVES)}."
            )
        normalized_uncertainty_metric = self._normalize_uncertainty_metric(
            uncertainty_metric
        )
        normalized_action_selection_strategy = self._normalize_action_selection_strategy(
            action_selection_strategy
        )
        if normalized_uncertainty_metric != "aggregate" and self.posterior_fusion == "early":
            raise ValueError(
                "Unimodal uncertainty metrics require posterior_fusion='poe' or a MoE fusion."
            )
        if (
            normalized_uncertainty_metric == "weighted_unimodal"
            and self.posterior_fusion not in self.MOE_POSTERIOR_FUSIONS
        ):
            raise ValueError(
                "uncertainty_metric='weighted_unimodal' requires a MoE posterior fusion."
            )
        if self.use_separate_modality_rnns and exploration_objective == "expected_information_gain":
            if normalized_action_selection_strategy in {
                "uncertainty_gate",
                "fixed_switching",
                "random_switching",
            }:
                raise ValueError(
                    "exploration_objective='expected_information_gain' is not supported "
                    "with modality_rnn_mode='separate'. Use realized_information_gain."
                )
        normalized_gate_mode = str(gate_mode).strip().lower()
        if normalized_gate_mode not in self.SUPPORTED_GATE_MODES:
            raise ValueError(
                f"Unsupported gate_mode '{gate_mode}'. "
                f"Expected one of {sorted(self.SUPPORTED_GATE_MODES)}."
            )
        if gating_temperature <= 0:
            raise ValueError("gating_temperature must be strictly positive.")
        epsilon_greedy_probability = float(epsilon_greedy_probability)
        if epsilon_greedy_probability < 0.0 or epsilon_greedy_probability > 1.0:
            raise ValueError("epsilon_greedy_probability must be in [0, 1].")
        random_switch_probability = float(random_switch_probability)
        if not 0.0 <= random_switch_probability <= 1.0:
            raise ValueError("random_switch_probability must be in [0, 1].")
        if fixed_switch_timestep is not None:
            if isinstance(fixed_switch_timestep, bool):
                raise ValueError("fixed_switch_timestep must be a non-negative integer.")
            normalized_switch_timestep = int(fixed_switch_timestep)
            if normalized_switch_timestep != fixed_switch_timestep:
                raise ValueError("fixed_switch_timestep must be a non-negative integer.")
            if normalized_switch_timestep < 0:
                raise ValueError("fixed_switch_timestep must be a non-negative integer.")
            fixed_switch_timestep = normalized_switch_timestep
        if (
            normalized_action_selection_strategy == "fixed_switching"
            and fixed_switch_timestep is None
        ):
            raise ValueError(
                "fixed_switch_timestep is required when "
                "action_selection_strategy='fixed_switching'."
            )

        if action_low is None:
            action_low = [0.0] * action_dim
        if action_high is None:
            action_high = [1.0] * action_dim

        action_low_t = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_t = torch.as_tensor(action_high, dtype=torch.float32)
        if action_low_t.shape != (action_dim,) or action_high_t.shape != (action_dim,):
            raise ValueError("action_low and action_high must have shape (action_dim,).")
        if action_delta_low is None:
            action_delta_low = (action_low_t - action_high_t).tolist()
        if action_delta_high is None:
            action_delta_high = (action_high_t - action_low_t).tolist()
        action_delta_low_t = torch.as_tensor(action_delta_low, dtype=torch.float32)
        action_delta_high_t = torch.as_tensor(action_delta_high, dtype=torch.float32)
        if action_delta_low_t.shape != (action_dim,) or action_delta_high_t.shape != (action_dim,):
            raise ValueError("action_delta_low and action_delta_high must have shape (action_dim,).")
        if torch.any(action_delta_low_t > 0) or torch.any(action_delta_high_t < 0):
            raise ValueError("action_delta_low must be <= 0 and action_delta_high must be >= 0.")
        if torch.any(action_delta_high_t - action_delta_low_t <= 0):
            raise ValueError("Each action delta interval must have positive width.")

        self.rl_algorithm = rl_algorithm
        self.exploration_objective = exploration_objective
        self.uncertainty_metric = normalized_uncertainty_metric
        self.gate_mode = normalized_gate_mode
        self.gating_temperature = float(gating_temperature)
        self.action_selection_strategy = normalized_action_selection_strategy
        self.epsilon_greedy_probability = epsilon_greedy_probability
        self.fixed_switch_timestep = fixed_switch_timestep
        self.random_switch_probability = random_switch_probability
        self.information_gain_reward_coef = float(information_gain_reward_coef)
        self.use_posterior_mean_for_planning = bool(use_posterior_mean_for_planning)
        # Backwards-compatible coefficient name: this now weights action-delta smoothness.
        self.exploration_action_l2 = float(exploration_action_l2)
        self.ppo_clip_epsilon = float(ppo_clip_epsilon)
        self.rl_value_coef = float(rl_value_coef)
        self.rl_entropy_coef = float(rl_entropy_coef)
        self.gate_value_coef = (
            self.rl_value_coef if gate_value_coef is None else float(gate_value_coef)
        )
        self.gate_entropy_coef = (
            self.rl_entropy_coef if gate_entropy_coef is None else float(gate_entropy_coef)
        )
        self.exploration_mc_samples = int(exploration_mc_samples)
        self.exploration_action_delta_huber_beta = float(exploration_action_delta_huber_beta)
        self.policy_input_dim = (2 * self.latent_dim) + self.recurrent_feature_dim
        if self.exploration_mc_samples <= 0:
            raise ValueError("exploration_mc_samples must be a positive integer.")
        if self.exploration_action_delta_huber_beta <= 0:
            raise ValueError("exploration_action_delta_huber_beta must be strictly positive.")

        self.register_buffer(
            "_runtime_reference",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        default_threshold = 0.5 * self.latent_dim * (1.0 + math.log(2.0 * math.pi))
        threshold_value = (
            default_threshold
            if fixed_gate_threshold is None
            else float(fixed_gate_threshold)
        )
        if self.gate_mode != "fixed_threshold":
            threshold_value = float("nan")
        self.register_buffer(
            "uncertainty_threshold",
            torch.tensor(threshold_value, dtype=torch.float32),
        )

        if self.exploration_uses_rl:
            self.exploration_head = ActorCriticHead(
                input_dim=self.policy_input_dim,
                action_low=action_low_t,
                action_high=action_high_t,
                action_delta_low=action_delta_low_t,
                action_delta_high=action_delta_high_t,
                policy_hidden_dims=exploration_hidden_dims,
                value_hidden_dims=exploration_hidden_dims,
            )
        else:
            self.exploration_head = DeterministicActionHead(
                input_dim=self.policy_input_dim,
                action_low=action_low_t,
                action_high=action_high_t,
                action_delta_low=action_delta_low_t,
                action_delta_high=action_delta_high_t,
                hidden_dims=exploration_hidden_dims,
            )
        self.exploitation_head = ActorCriticHead(
            input_dim=self.policy_input_dim,
            action_low=action_low_t,
            action_high=action_high_t,
            action_delta_low=action_delta_low_t,
            action_delta_high=action_delta_high_t,
            policy_hidden_dims=exploitation_policy_hidden_dims,
            value_hidden_dims=exploitation_value_hidden_dims,
        )
        self.gate_head = None
        if self.gate_uses_rl:
            self.gate_head = GateActorCriticHead(
                input_dim=self.policy_input_dim,
                policy_hidden_dims=gate_hidden_dims,
                value_hidden_dims=gate_hidden_dims,
            )

    @property
    def action_low(self) -> torch.Tensor:
        if self.exploration_uses_rl:
            return self.exploration_head.actor.action_low
        return self.exploration_head.action_low

    @property
    def action_high(self) -> torch.Tensor:
        if self.exploration_uses_rl:
            return self.exploration_head.actor.action_high
        return self.exploration_head.action_high

    @property
    def exploration_uses_rl(self) -> bool:
        return self.exploration_objective == "realized_information_gain"

    @property
    def gate_uses_rl(self) -> bool:
        return self.gate_mode == "rl_head"

    @property
    def uses_uncertainty_gate(self) -> bool:
        return self.action_selection_strategy == "uncertainty_gate"

    @property
    def uses_exploration_policy(self) -> bool:
        return self.action_selection_strategy in {
            "uncertainty_gate",
            "fixed_switching",
            "random_switching",
        }

    def initial_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if device is None:
            device = self._runtime_reference.device
        if dtype is None:
            dtype = self._runtime_reference.dtype
        return super().initial_hidden(batch_size, device=device, dtype=dtype)

    def perception_parameters(self):
        modules = [
            self.mech_encoder,
            self.visual_encoder,
            self.fusion_layer,
            self.y2h,
            self.h2h,
            self.lstm_cell,
            self.mech_y2h,
            self.mech_h2h,
            self.vis_y2h,
            self.vis_h2h,
            self.encoder,
            self.enc_mu,
            self.enc_logvar,
            self.mech_posterior_encoder,
            self.mech_post_mu,
            self.mech_post_logvar,
            self.vis_posterior_encoder,
            self.vis_post_mu,
            self.vis_post_logvar,
            self.moe_weight_net,
            self.mech_decoder,
            self.vis_decoder,
        ]
        for module in modules:
            if module is not None:
                yield from module.parameters()

    def exploration_parameters(self):
        yield from self.exploration_head.parameters()

    def exploitation_parameters(self):
        yield from self.exploitation_head.parameters()

    def gating_parameters(self):
        if self.gate_head is not None:
            yield from self.gate_head.parameters()

    def belief_features(
        self,
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
        h_t: torch.Tensor,
        s_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_state = post_mu if s_t is None else s_t
        return torch.cat([latent_state, post_logvar, h_t], dim=-1)

    @staticmethod
    def _required_response(
        responses_t: Mapping[str, torch.Tensor],
        key: str,
        metric: str,
    ) -> torch.Tensor:
        if key not in responses_t:
            raise ValueError(
                f"uncertainty_metric='{metric}' requires '{key}' in the model responses."
            )
        return responses_t[key]

    def uncertainty_logvar_from_responses(
        self,
        responses_t: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        metric = self.uncertainty_metric
        if metric == "aggregate":
            return responses_t["post_logvar"]

        mech_logvar = self._required_response(
            responses_t,
            "mech_post_logvar",
            metric,
        )
        vis_logvar = self._required_response(
            responses_t,
            "vis_post_logvar",
            metric,
        )
        if metric == "average_unimodal":
            return 0.5 * (mech_logvar + vis_logvar)
        if metric == "weighted_unimodal":
            weights = self._required_response(responses_t, "moe_weights", metric)
            return (
                weights[..., 0:1] * mech_logvar
                + weights[..., 1:2] * vis_logvar
            )
        if metric == "mech":
            return mech_logvar
        if metric == "vis":
            return vis_logvar
        raise RuntimeError(f"Unsupported uncertainty_metric '{metric}'.")

    def uncertainty_from_responses(
        self,
        responses_t: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return posterior_entropy(self.uncertainty_logvar_from_responses(responses_t))

    def belief_features_from_responses(
        self,
        responses_t: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        post_mu = responses_t["post_mu"]
        feature_logvar = self.uncertainty_logvar_from_responses(responses_t)
        h_t = responses_t["h"]
        s_t = None if self.use_posterior_mean_for_planning else responses_t.get("s")
        planning_state = self._planning_state(
            post_mu,
            responses_t["post_logvar"],
            s_t=s_t,
        )
        return self.belief_features(post_mu, feature_logvar, h_t, planning_state)

    def _planning_state(
        self,
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
        s_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_posterior_mean_for_planning:
            return post_mu
        if s_t is not None:
            return s_t
        return self.reparameterize(post_mu, post_logvar)

    def compute_exploration_loss(
        self,
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
        h_t: torch.Tensor,
        c_t: Optional[torch.Tensor] = None,
        s_t: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        prev_action: Optional[torch.Tensor] = None,
    ) -> Mapping[str, torch.Tensor]:
        if self.exploration_uses_rl:
            raise RuntimeError(
                "compute_exploration_loss is only defined for "
                "exploration_objective='expected_information_gain'."
            )
        planning_state = self._planning_state(post_mu, post_logvar, s_t=s_t)
        features = self.belief_features(post_mu, post_logvar, h_t, planning_state)
        exploration_action = self.exploration_head(features) if action is None else action
        return estimate_expected_information_gain(
            model=self,
            post_logvar=post_logvar,
            h_t=h_t,
            planning_state=planning_state,
            action=exploration_action,
            num_samples=self.exploration_mc_samples,
            c_t=c_t,
            action_l2=self.exploration_action_l2,
            prev_action=prev_action,
            action_delta_huber_beta=self.exploration_action_delta_huber_beta,
        )

    def compute_exploration_loss_from_responses(
        self,
        responses_t: Mapping[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        prev_action: Optional[torch.Tensor] = None,
    ) -> Mapping[str, torch.Tensor]:
        if self.exploration_uses_rl:
            raise RuntimeError(
                "compute_exploration_loss_from_responses is only defined for "
                "exploration_objective='expected_information_gain'."
            )
        return self.compute_exploration_loss(
            post_mu=responses_t["post_mu"],
            post_logvar=responses_t["post_logvar"],
            h_t=responses_t["h"],
            c_t=responses_t.get("c"),
            s_t=None if self.use_posterior_mean_for_planning else responses_t.get("s"),
            action=action,
            prev_action=prev_action,
        )

    def compute_realized_information_gain_from_responses(
        self,
        responses_t: Mapping[str, torch.Tensor],
        next_responses_t: Mapping[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
        action_l2: float = 0.0,
        prev_action: Optional[torch.Tensor] = None,
        action_delta_huber_beta: float = 0.05,
    ) -> Mapping[str, torch.Tensor]:
        current_uncertainty = self.uncertainty_from_responses(responses_t)
        next_uncertainty = self.uncertainty_from_responses(next_responses_t)
        information_gain = current_uncertainty - next_uncertainty

        action_penalty = torch.zeros_like(information_gain)
        if action_l2 > 0.0:
            if action is None or prev_action is None:
                raise ValueError(
                    "action and prev_action are required when action_l2 > 0."
                )
            if action_low is None:
                action_low = self.action_low
            if action_high is None:
                action_high = self.action_high
            action_penalty = compute_action_smoothness_penalty(
                action=action,
                prev_action=prev_action,
                action_low=action_low,
                action_high=action_high,
                action_l2=action_l2,
                huber_beta=action_delta_huber_beta,
            )

        reward = information_gain - action_penalty
        return {
            "reward": reward,
            "information_gain": information_gain,
            "current_entropy": current_uncertainty,
            "next_entropy": next_uncertainty,
            "action_penalty": action_penalty,
        }

    def fixed_threshold_explore_probability(self, uncertainty: torch.Tensor) -> torch.Tensor:
        logits = (uncertainty - self.uncertainty_threshold) / self.gating_temperature
        return torch.sigmoid(logits)

    def sample_uniform_action(self, prev_action: torch.Tensor) -> torch.Tensor:
        action_low, action_high = self.exploitation_head.actor.action_bounds(prev_action)
        return action_low + torch.rand_like(action_low) * (action_high - action_low)

    def _select_action_from_features(
        self,
        features: torch.Tensor,
        uncertainty: torch.Tensor,
        prev_action: torch.Tensor,
        deterministic_exploitation: bool = False,
        deterministic_exploration: bool = False,
        stochastic_gate: bool = False,
        detach_gate_input: bool = False,
        force_mode: Optional[str] = None,
        episode_timestep: Optional[int] = None,
    ) -> ActionSelection:
        forced_mode = None if force_mode is None else str(force_mode).strip().lower()
        if forced_mode in {"", "auto", "none"}:
            forced_mode = None
        if forced_mode is not None and forced_mode not in {"explore", "exploit"}:
            raise ValueError("force_mode must be one of: auto, explore, exploit.")

        strategy = self.action_selection_strategy
        if strategy == "fixed_switching":
            if episode_timestep is None:
                raise ValueError(
                    "episode_timestep is required when "
                    "action_selection_strategy='fixed_switching'."
                )
            if isinstance(episode_timestep, bool):
                raise ValueError("episode_timestep must be a non-negative integer.")
            normalized_episode_timestep = int(episode_timestep)
            if (
                normalized_episode_timestep != episode_timestep
                or normalized_episode_timestep < 0
            ):
                raise ValueError("episode_timestep must be a non-negative integer.")
            episode_timestep = normalized_episode_timestep
        exploration_log_prob = None
        exploration_value = None
        if strategy in {
            "uncertainty_gate",
            "fixed_switching",
            "random_switching",
        }:
            if self.exploration_uses_rl:
                (
                    exploration_action,
                    exploration_log_prob,
                    exploration_value,
                    _,
                ) = self.exploration_head.act(
                    features,
                    prev_action,
                    deterministic=deterministic_exploration,
                )
            else:
                exploration_action = self.exploration_head(features, prev_action)
        else:
            exploration_action = None

        exploitation_action, exploitation_log_prob, exploitation_value, _ = self.exploitation_head.act(
            features,
            prev_action,
            deterministic=deterministic_exploitation,
        )
        if strategy not in {
            "uncertainty_gate",
            "fixed_switching",
            "random_switching",
        }:
            if strategy == "epsilon_greedy" or forced_mode == "explore":
                exploration_action = self.sample_uniform_action(prev_action)
            else:
                exploration_action = exploitation_action.detach()

        gate_uncertainty = uncertainty.detach() if detach_gate_input else uncertainty
        threshold = self.uncertainty_threshold.detach()
        gate_log_prob = None
        gate_value = None
        gate_entropy = None
        if strategy == "uncertainty_gate" and self.gate_uses_rl:
            if self.gate_head is None:
                raise RuntimeError("gate_head must be initialized when gate_mode='rl_head'.")
            (
                gate_action,
                gate_log_prob,
                gate_value,
                gate_entropy,
                explore_prob,
            ) = self.gate_head.act(
                features,
                deterministic=not stochastic_gate,
            )
        elif strategy == "uncertainty_gate":
            explore_prob = self.fixed_threshold_explore_probability(gate_uncertainty)
        elif strategy == "epsilon_greedy":
            explore_prob = torch.full_like(
                uncertainty,
                float(self.epsilon_greedy_probability),
            )
        elif strategy == "fixed_switching":
            explore_prob = torch.full_like(
                uncertainty,
                float(episode_timestep < self.fixed_switch_timestep),
            )
        elif strategy == "random_switching":
            explore_prob = torch.full_like(
                uncertainty,
                self.random_switch_probability,
            )
        else:
            explore_prob = torch.zeros_like(uncertainty)

        if forced_mode == "explore":
            selected_is_exploration = torch.ones_like(explore_prob, dtype=torch.bool)
        elif forced_mode == "exploit":
            selected_is_exploration = torch.zeros_like(explore_prob, dtype=torch.bool)
        elif strategy == "uncertainty_gate" and self.gate_uses_rl:
            selected_is_exploration = gate_action.to(dtype=torch.bool)
        elif strategy == "uncertainty_gate":
            selected_is_exploration = gate_uncertainty > threshold
        elif strategy == "epsilon_greedy":
            selected_is_exploration = torch.rand_like(explore_prob) < explore_prob
        elif strategy == "fixed_switching":
            selected_is_exploration = torch.full_like(
                explore_prob,
                episode_timestep < self.fixed_switch_timestep,
                dtype=torch.bool,
            )
        elif strategy == "random_switching":
            selected_is_exploration = torch.rand_like(explore_prob) < explore_prob
        else:
            selected_is_exploration = torch.zeros_like(explore_prob, dtype=torch.bool)

        action = torch.where(
            selected_is_exploration.unsqueeze(-1),
            exploration_action,
            exploitation_action,
        )

        return ActionSelection(
            action=action,
            belief_features=features,
            exploration_action=exploration_action,
            exploration_log_prob=exploration_log_prob,
            exploration_value=exploration_value,
            exploitation_action=exploitation_action,
            selected_is_exploration=selected_is_exploration,
            uncertainty=uncertainty,
            explore_probability=explore_prob,
            threshold=threshold,
            gate_log_prob=gate_log_prob,
            gate_value=gate_value,
            gate_entropy=gate_entropy,
            exploitation_log_prob=exploitation_log_prob,
            exploitation_value=exploitation_value,
        )

    def select_action_from_responses(
        self,
        responses_t: Mapping[str, torch.Tensor],
        prev_action: torch.Tensor,
        deterministic_exploitation: bool = False,
        deterministic_exploration: bool = False,
        stochastic_gate: bool = False,
        detach_gate_input: bool = False,
        force_mode: Optional[str] = None,
        episode_timestep: Optional[int] = None,
    ) -> ActionSelection:
        features = self.belief_features_from_responses(responses_t)
        uncertainty = self.uncertainty_from_responses(responses_t)
        return self._select_action_from_features(
            features=features,
            uncertainty=uncertainty,
            prev_action=prev_action,
            deterministic_exploitation=deterministic_exploitation,
            deterministic_exploration=deterministic_exploration,
            stochastic_gate=stochastic_gate,
            detach_gate_input=detach_gate_input,
            force_mode=force_mode,
            episode_timestep=episode_timestep,
        )

    def select_action_from_belief(
        self,
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
        h_t: torch.Tensor,
        prev_action: torch.Tensor,
        s_t: Optional[torch.Tensor] = None,
        deterministic_exploitation: bool = False,
        deterministic_exploration: bool = False,
        stochastic_gate: bool = False,
        detach_gate_input: bool = False,
        force_mode: Optional[str] = None,
        episode_timestep: Optional[int] = None,
    ) -> ActionSelection:
        if self.uncertainty_metric != "aggregate":
            raise ValueError(
                "select_action_from_belief only has aggregate posterior inputs. "
                "Use select_action_from_responses for unimodal uncertainty metrics."
            )
        planning_state = self._planning_state(post_mu, post_logvar, s_t=s_t)
        features = self.belief_features(post_mu, post_logvar, h_t, planning_state)
        uncertainty = posterior_entropy(post_logvar)
        return self._select_action_from_features(
            features=features,
            uncertainty=uncertainty,
            prev_action=prev_action,
            deterministic_exploitation=deterministic_exploitation,
            deterministic_exploration=deterministic_exploration,
            stochastic_gate=stochastic_gate,
            detach_gate_input=detach_gate_input,
            force_mode=force_mode,
            episode_timestep=episode_timestep,
        )

    def act(
        self,
        x_mech_t: torch.Tensor,
        x_vis_t: torch.Tensor,
        a_prev: torch.Tensor,
        h_prev: torch.Tensor,
        deterministic_exploitation: bool = False,
        deterministic_exploration: bool = False,
        stochastic_gate: bool = False,
        detach_gate_input: bool = False,
        force_mode: Optional[str] = None,
        episode_timestep: Optional[int] = None,
    ) -> Mapping[str, Any]:
        responses_t, losses_t = self.forward_t(x_mech_t, x_vis_t, a_prev, h_prev)
        action_selection = self.select_action_from_responses(
            responses_t=responses_t,
            prev_action=a_prev,
            deterministic_exploitation=deterministic_exploitation,
            deterministic_exploration=deterministic_exploration,
            stochastic_gate=stochastic_gate,
            detach_gate_input=detach_gate_input,
            force_mode=force_mode,
            episode_timestep=episode_timestep,
        )
        exploration = None
        if self.uses_exploration_policy and not self.exploration_uses_rl:
            exploration = self.compute_exploration_loss_from_responses(
                responses_t,
                action=action_selection.exploration_action,
                prev_action=a_prev,
            )
        return {
            "responses": responses_t,
            "perception_losses": losses_t,
            "exploration_losses": exploration,
            "action_selection": action_selection,
        }

    @staticmethod
    def compute_returns_and_advantages(
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = rewards.float()
        dones = dones.float()

        if values is None:
            returns = _discounted_cumsum(rewards, gamma)
            return returns, returns.clone()

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        next_return = torch.zeros_like(rewards[..., 0])
        next_value = torch.zeros_like(rewards[..., 0])
        next_advantage = torch.zeros_like(rewards[..., 0])

        for t in range(rewards.shape[-1] - 1, -1, -1):
            not_done = 1.0 - dones[..., t]
            returns[..., t] = rewards[..., t] + gamma * next_return * not_done
            delta = rewards[..., t] + gamma * next_value * not_done - values[..., t]
            advantages[..., t] = delta + gamma * gae_lambda * next_advantage * not_done
            next_return = returns[..., t]
            next_value = values[..., t]
            next_advantage = advantages[..., t]

        return returns, advantages

    def _tensor_from_rollout(
        self,
        rollout: Union[RolloutBatch, GateRolloutBatch, Mapping[str, torch.Tensor]],
        name: str,
        default: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if hasattr(rollout, name):
            value = getattr(rollout, name)
        else:
            value = rollout.get(name, default)
        return default if value is None else value

    def compute_exploitation_loss(
        self,
        rollout: Union[RolloutBatch, Mapping[str, torch.Tensor]],
        algorithm: Optional[str] = None,
    ) -> Mapping[str, torch.Tensor]:
        losses = self._compute_actor_critic_loss(
            head=self.exploitation_head,
            rollout=rollout,
            algorithm=algorithm,
            value_coef=self.rl_value_coef,
            entropy_coef=self.rl_entropy_coef,
            mask_value_loss=False,
        )

        losses["base_loss"] = losses["loss"]
        losses["gate_loss"] = losses["policy_loss"].new_zeros(())
        return losses

    def compute_exploration_rl_loss(
        self,
        rollout: Union[RolloutBatch, Mapping[str, torch.Tensor]],
        algorithm: Optional[str] = None,
    ) -> Mapping[str, torch.Tensor]:
        if not self.exploration_uses_rl:
            raise RuntimeError(
                "compute_exploration_rl_loss is only defined for "
                "exploration_objective='realized_information_gain'."
            )
        losses = self._compute_actor_critic_loss(
            head=self.exploration_head,
            rollout=rollout,
            algorithm=algorithm,
            value_coef=self.rl_value_coef,
            entropy_coef=self.rl_entropy_coef,
        )
        losses["gate_loss"] = losses["policy_loss"].new_zeros(())
        return losses

    def compute_gate_rl_loss(
        self,
        rollout: Union[GateRolloutBatch, Mapping[str, torch.Tensor]],
        algorithm: Optional[str] = None,
    ) -> Mapping[str, torch.Tensor]:
        if not self.gate_uses_rl:
            zero = torch.zeros((), device=self._runtime_reference.device, dtype=torch.float32)
            return {
                "loss": zero,
                "policy_loss": zero,
                "value_loss": zero,
                "entropy_bonus": zero,
                "log_prob": zero,
                "value": zero,
                "advantages": zero,
                "explore_probability": zero,
            }
        if self.gate_head is None:
            raise RuntimeError("gate_head must be initialized when gate_mode='rl_head'.")

        algorithm = self.rl_algorithm if algorithm is None else algorithm
        if algorithm not in self.SUPPORTED_RL_ALGORITHMS:
            raise ValueError(
                f"Unsupported RL algorithm '{algorithm}'. "
                f"Expected one of {sorted(self.SUPPORTED_RL_ALGORITHMS)}."
            )

        belief_features = self._tensor_from_rollout(rollout, "belief_features")
        actions = self._tensor_from_rollout(rollout, "actions")
        returns = self._tensor_from_rollout(rollout, "returns")
        advantages = self._tensor_from_rollout(rollout, "advantages")
        old_log_probs = self._tensor_from_rollout(rollout, "old_log_probs")

        if belief_features is None or actions is None or returns is None:
            raise ValueError("gate rollout must provide belief_features, actions, and returns.")

        belief_features = belief_features.reshape(-1, belief_features.shape[-1])
        actions = actions.reshape(-1).float()
        returns = returns.reshape(-1)
        log_prob, values, entropy, probs = self.gate_head.evaluate_actions(
            belief_features,
            actions,
        )

        if advantages is None:
            advantages = returns - values.detach()
        else:
            advantages = advantages.reshape(-1)

        if advantages.numel() > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False).clamp_min(1e-8)
            advantages = (advantages - adv_mean) / adv_std
        elif advantages.numel() == 1:
            advantages = advantages - advantages[0]

        if algorithm == "reinforce":
            policy_loss = -(log_prob * advantages.detach()).mean()
            value_loss = torch.zeros((), device=values.device, dtype=values.dtype)
        elif algorithm == "a2c":
            policy_loss = -(log_prob * advantages.detach()).mean()
            value_loss = 0.5 * (values - returns).pow(2).mean()
        else:
            if old_log_probs is None:
                raise ValueError("PPO requires gate rollout.old_log_probs.")
            old_log_probs = old_log_probs.reshape(-1)
            ratio = torch.exp(log_prob - old_log_probs)
            clipped_ratio = ratio.clamp(
                1.0 - self.ppo_clip_epsilon,
                1.0 + self.ppo_clip_epsilon,
            )
            surrogate_1 = ratio * advantages.detach()
            surrogate_2 = clipped_ratio * advantages.detach()
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            value_loss = 0.5 * (values - returns).pow(2).mean()

        entropy_bonus = entropy.mean()
        total_loss = (
            policy_loss
            + self.gate_value_coef * value_loss
            - self.gate_entropy_coef * entropy_bonus
        )

        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_bonus": entropy_bonus,
            "log_prob": log_prob.mean(),
            "value": values.mean(),
            "advantages": advantages,
            "explore_probability": probs.mean(),
        }

    def _compute_actor_critic_loss(
        self,
        head: ActorCriticHead,
        rollout: Union[RolloutBatch, Mapping[str, torch.Tensor]],
        algorithm: Optional[str] = None,
        value_coef: Optional[float] = None,
        entropy_coef: Optional[float] = None,
        mask_value_loss: bool = True,
    ) -> Mapping[str, torch.Tensor]:
        algorithm = self.rl_algorithm if algorithm is None else algorithm
        if algorithm not in self.SUPPORTED_RL_ALGORITHMS:
            raise ValueError(
                f"Unsupported RL algorithm '{algorithm}'. "
                f"Expected one of {sorted(self.SUPPORTED_RL_ALGORITHMS)}."
            )
        value_coef = self.rl_value_coef if value_coef is None else float(value_coef)
        entropy_coef = self.rl_entropy_coef if entropy_coef is None else float(entropy_coef)

        belief_features = self._tensor_from_rollout(rollout, "belief_features")
        prev_actions = self._tensor_from_rollout(rollout, "prev_actions")
        actions = self._tensor_from_rollout(rollout, "actions")
        returns = self._tensor_from_rollout(rollout, "returns")
        advantages = self._tensor_from_rollout(rollout, "advantages")
        old_log_probs = self._tensor_from_rollout(rollout, "old_log_probs")
        action_mask = self._tensor_from_rollout(rollout, "action_mask")

        if belief_features is None or prev_actions is None or actions is None or returns is None:
            raise ValueError("rollout must provide belief_features, prev_actions, actions, and returns.")

        belief_features = belief_features.reshape(-1, belief_features.shape[-1])
        prev_actions = prev_actions.reshape(-1, prev_actions.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        returns = returns.reshape(-1)

        log_prob, values, entropy = head.evaluate_actions(
            belief_features,
            actions,
            prev_actions,
        )

        if advantages is None:
            advantages = returns - values.detach()
        else:
            advantages = advantages.reshape(-1)

        if action_mask is not None:
            action_mask = action_mask.reshape(-1).float()
        else:
            action_mask = torch.ones_like(advantages)

        def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            denom = mask.sum().clamp_min(1.0)
            return (x * mask).sum() / denom

        active_advantages = advantages[action_mask > 0]
        if active_advantages.numel() > 1:
            adv_mean = active_advantages.mean()
            adv_std = active_advantages.std(unbiased=False).clamp_min(1e-8)
            advantages = (advantages - adv_mean) / adv_std
        elif active_advantages.numel() == 1:
            advantages = advantages - active_advantages[0]

        if algorithm == "reinforce":
            policy_loss = -_masked_mean(log_prob * advantages.detach(), action_mask)
            value_loss = torch.zeros((), device=values.device, dtype=values.dtype)
        elif algorithm == "a2c":
            policy_loss = -_masked_mean(log_prob * advantages.detach(), action_mask)
            value_error = (values - returns).pow(2)
            value_loss = 0.5 * (
                _masked_mean(value_error, action_mask) if mask_value_loss else value_error.mean()
            )
        else:
            if old_log_probs is None:
                raise ValueError("PPO requires rollout.old_log_probs.")
            old_log_probs = old_log_probs.reshape(-1)
            ratio = torch.exp(log_prob - old_log_probs)
            clipped_ratio = ratio.clamp(
                1.0 - self.ppo_clip_epsilon,
                1.0 + self.ppo_clip_epsilon,
            )
            surrogate_1 = ratio * advantages.detach()
            surrogate_2 = clipped_ratio * advantages.detach()
            policy_loss = -_masked_mean(torch.min(surrogate_1, surrogate_2), action_mask)
            value_error = (values - returns).pow(2)
            value_loss = 0.5 * (
                _masked_mean(value_error, action_mask) if mask_value_loss else value_error.mean()
            )

        entropy_bonus = _masked_mean(entropy, action_mask)
        total_loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy_bonus
        )

        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_bonus": entropy_bonus,
            "log_prob": _masked_mean(log_prob, action_mask),
            "value": _masked_mean(values, action_mask),
            "advantages": advantages,
        }
