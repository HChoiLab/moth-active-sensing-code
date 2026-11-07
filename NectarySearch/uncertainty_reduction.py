from __future__ import annotations

import math
from typing import Mapping, Optional, Protocol, Sequence

import torch
from torch.distributions import ContinuousBernoulli


class SupportsExpectedInformationGainModel(Protocol):
    vis_obs_shape: Sequence[int]
    mech_obs_dim: int
    latent_dim: int
    hidden_dim: int
    action_low: torch.Tensor
    action_high: torch.Tensor
    visual_likelihood: str

    def decode_t(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: ...

    def integrate_multisensory(
        self,
        mech: torch.Tensor,
        vis: torch.Tensor,
    ) -> torch.Tensor: ...

    def encode_t(
        self,
        x_t: torch.Tensor,
        a_m_1: torch.Tensor,
        h_m_1,
    ): ...

    def encode_observation_t(
        self,
        x_mech_t: torch.Tensor,
        x_vis_t: torch.Tensor,
        a_m_1: torch.Tensor,
        h_m_1,
    ): ...


def gaussian_entropy_from_logvar(logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + math.log(2.0 * math.pi) + logvar)


def posterior_entropy(
    logvar: torch.Tensor,
    reduce_latent: bool = True,
) -> torch.Tensor:
    entropy = gaussian_entropy_from_logvar(logvar)
    return entropy.sum(dim=-1) if reduce_latent else entropy


def compute_action_smoothness_penalty(
    action: torch.Tensor,
    prev_action: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    action_l2: float,
    huber_beta: float = 0.05,
) -> torch.Tensor:
    if action_l2 <= 0.0:
        return torch.zeros(action.shape[:-1], device=action.device, dtype=action.dtype)
    if huber_beta <= 0.0:
        raise ValueError("huber_beta must be strictly positive.")

    normalized_delta = (action - prev_action) / ((action_high - action_low).clamp_min(1e-6))
    abs_delta = normalized_delta.abs()
    huber = torch.where(
        abs_delta <= huber_beta,
        0.5 * abs_delta.pow(2) / huber_beta,
        abs_delta - (0.5 * huber_beta),
    )
    return float(action_l2) * huber.mean(dim=-1)


def decoder_predictions_to_expected_obs(
    x_mech_hat: torch.Tensor,
    x_vis_hat: torch.Tensor,
    vis_obs_shape: Sequence[int],
    visual_likelihood: str = "gaussian",
) -> tuple[torch.Tensor, torch.Tensor]:
    visual_likelihood = str(visual_likelihood).lower()
    contact_prob = torch.sigmoid(x_mech_hat[..., :1])
    expected_mech = torch.cat([contact_prob, x_mech_hat[..., 1:]], dim=-1)
    if visual_likelihood == "gaussian":
        expected_vis_flat = x_vis_hat
    elif visual_likelihood == "continuous_bernoulli":
        probs = x_vis_hat.clamp(min=1e-6, max=1.0 - 1e-6)
        expected_vis_flat = ContinuousBernoulli(probs=probs).mean
    else:
        raise ValueError(
            f"Unsupported visual_likelihood '{visual_likelihood}'."
        )
    expected_vis = expected_vis_flat.reshape(*x_vis_hat.shape[:-1], *vis_obs_shape)
    return expected_mech, expected_vis


def sample_decoder_observations(
    x_mech_hat: torch.Tensor,
    x_mech_logvar: torch.Tensor,
    x_vis_hat: torch.Tensor,
    x_vis_logvar: Optional[torch.Tensor],
    vis_obs_shape: Sequence[int],
    num_samples: int,
    visual_likelihood: str = "gaussian",
    min_logvar: float = -8.0,
    max_logvar: float = 8.0,
    straight_through_binary: bool = True,
    clamp_visual_range: Optional[tuple[float, float]] = (0.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    visual_likelihood = str(visual_likelihood).lower()
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    contact_probs = torch.sigmoid(x_mech_hat[..., :1])
    contact_probs = contact_probs.unsqueeze(0).expand(num_samples, *contact_probs.shape)
    bernoulli_draws = torch.bernoulli(contact_probs)
    if straight_through_binary:
        contact_samples = bernoulli_draws + contact_probs - contact_probs.detach()
    else:
        contact_samples = bernoulli_draws

    mech_mean = x_mech_hat[..., 1:]
    if mech_mean.shape[-1] > 0:
        mech_std = torch.exp(0.5 * x_mech_logvar.clamp(min=min_logvar, max=max_logvar))
        mech_noise = torch.randn(
            (num_samples, *mech_mean.shape),
            device=mech_mean.device,
            dtype=mech_mean.dtype,
        )
        mech_samples = mech_mean.unsqueeze(0) + mech_noise * mech_std.unsqueeze(0)
        mech_obs_samples = torch.cat([contact_samples, mech_samples], dim=-1)
    else:
        mech_obs_samples = contact_samples

    if visual_likelihood == "gaussian":
        if x_vis_logvar is None:
            raise ValueError("x_vis_logvar is required when visual_likelihood='gaussian'.")
        vis_std = torch.exp(0.5 * x_vis_logvar.clamp(min=min_logvar, max=max_logvar))
        vis_noise = torch.randn(
            (num_samples, *x_vis_hat.shape),
            device=x_vis_hat.device,
            dtype=x_vis_hat.dtype,
        )
        vis_samples = x_vis_hat.unsqueeze(0) + vis_noise * vis_std.unsqueeze(0)
        if clamp_visual_range is not None:
            vis_samples = vis_samples.clamp(*clamp_visual_range)
    elif visual_likelihood == "continuous_bernoulli":
        vis_probs = x_vis_hat.clamp(min=1e-6, max=1.0 - 1e-6)
        vis_dist = ContinuousBernoulli(probs=vis_probs)
        if getattr(vis_dist, "has_rsample", False):
            vis_samples = vis_dist.rsample((num_samples,))
        else:
            vis_samples = vis_dist.sample((num_samples,))
    else:
        raise ValueError(f"Unsupported visual_likelihood '{visual_likelihood}'.")
    vis_obs_samples = vis_samples.reshape(num_samples, *x_vis_hat.shape[:-1], *vis_obs_shape)
    return mech_obs_samples, vis_obs_samples


def infer_observation_posterior(
    model: SupportsExpectedInformationGainModel,
    mech: torch.Tensor,
    vis: torch.Tensor,
    action: torch.Tensor,
    h_m_1,
    modality='joint'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hasattr(model, "encode_observation_t"):
        post_mu, post_logvar, _, _, h_t, modality_posteriors = model.encode_observation_t(
            mech,
            vis,
            action,
            h_m_1,
        )
        if modality == 'joint':
            return post_mu, post_logvar, h_t
        elif modality == 'vis':
            return modality_posteriors['vis_post_mu'], modality_posteriors['vis_post_logvar'], h_t
        elif modality == 'mech':
            return modality_posteriors['mech_post_mu'], modality_posteriors['mech_post_logvar'], h_t
        else:
            raise ValueError(f"Modality has to be one of ['joint', 'vis', 'mech']")

    x_t = model.integrate_multisensory(mech, vis)
    return model.encode_t(x_t, action, h_m_1)


def estimate_expected_information_gain(
    *,
    model: SupportsExpectedInformationGainModel,
    post_logvar: torch.Tensor,
    h_t: torch.Tensor,
    planning_state: torch.Tensor,
    action: torch.Tensor,
    num_samples: int,
    c_t: Optional[torch.Tensor] = None,
    action_l2: float = 0.0,
    prev_action: Optional[torch.Tensor] = None,
    action_delta_huber_beta: float = 0.05,
    blank_mechanosensory: bool = False,
    blank_visual: bool = False,
) -> Mapping[str, torch.Tensor]:
    if getattr(model, "modality_rnn_mode", "shared") == "separate":
        raise NotImplementedError(
            "Expected information gain is not implemented for "
            "modality_rnn_mode='separate'. Use realized information gain or "
            "disable expected-IG diagnostics for this architecture."
        )

    x_mech_hat, x_mech_logvar, x_vis_hat, x_vis_logvar = model.decode_t(
        planning_state,
        action,
    )

    sampled_mech, sampled_vis = sample_decoder_observations(
        x_mech_hat=x_mech_hat,
        x_mech_logvar=x_mech_logvar,
        x_vis_hat=x_vis_hat,
        x_vis_logvar=x_vis_logvar,
        vis_obs_shape=model.vis_obs_shape,
        num_samples=num_samples,
        visual_likelihood=getattr(model, "visual_likelihood", "gaussian"),
    )
    if blank_mechanosensory:
        sampled_mech = torch.zeros_like(sampled_mech)
    if blank_visual:
        sampled_vis = torch.zeros_like(sampled_vis)

    flat_mech = sampled_mech.reshape(-1, model.mech_obs_dim)
    flat_vis = sampled_vis.reshape(-1, *model.vis_obs_shape)
    flat_actions = action.unsqueeze(0).expand(num_samples, *action.shape).reshape(-1, action.shape[-1])
    flat_h = h_t.unsqueeze(0).expand(num_samples, *h_t.shape).reshape(-1, model.hidden_dim)
    flat_recurrent_state = flat_h
    if c_t is not None:
        flat_c = c_t.unsqueeze(0).expand(num_samples, *c_t.shape).reshape(-1, model.hidden_dim)
        flat_recurrent_state = (flat_h, flat_c)

    next_mu_mc, next_logvar_mc, next_h_mc = infer_observation_posterior(
        model,
        flat_mech,
        flat_vis,
        flat_actions,
        flat_recurrent_state,
    )
    next_mu_mc = next_mu_mc.reshape(num_samples, -1, model.latent_dim)
    next_logvar_mc = next_logvar_mc.reshape(num_samples, -1, model.latent_dim)
    next_h_mc = next_h_mc.reshape(num_samples, -1, model.hidden_dim)

    current_entropy = posterior_entropy(post_logvar)
    sampled_next_entropy = posterior_entropy(next_logvar_mc)
    expected_next_entropy = sampled_next_entropy.mean(dim=0)
    information_gain = current_entropy - expected_next_entropy

    action_penalty = torch.zeros_like(information_gain)
    if action_l2 > 0.0:
        if prev_action is None:
            raise ValueError("prev_action is required when action_l2 > 0.")
        action_low = getattr(model, "action_low", None)
        action_high = getattr(model, "action_high", None)
        if action_low is None or action_high is None:
            raise ValueError("model.action_low and model.action_high are required when action_l2 > 0.")
        action_penalty = compute_action_smoothness_penalty(
            action=action,
            prev_action=prev_action,
            action_low=action_low,
            action_high=action_high,
            action_l2=action_l2,
            huber_beta=action_delta_huber_beta,
        )

    loss = -(information_gain - action_penalty)
    return {
        "loss": loss.mean(),
        "loss_per_sample": loss,
        "information_gain": information_gain.mean(),
        "information_gain_per_sample": information_gain,
        "current_entropy": current_entropy.mean(),
        "current_entropy_per_sample": current_entropy,
        "next_entropy": expected_next_entropy.mean(),
        "next_entropy_per_sample": expected_next_entropy,
        "sampled_next_entropy": sampled_next_entropy.mean(),
        "action_penalty": action_penalty.mean(),
        "action_penalty_per_sample": action_penalty,
        "sampled_mech": sampled_mech,
        "sampled_vis": sampled_vis,
        "next_mu_mc": next_mu_mc,
        "next_logvar_mc": next_logvar_mc,
        "next_h_mc": next_h_mc,
    }


def compute_realized_information_gain(
    *,
    post_logvar: torch.Tensor,
    next_post_logvar: torch.Tensor,
    action: Optional[torch.Tensor] = None,
    action_low: Optional[torch.Tensor] = None,
    action_high: Optional[torch.Tensor] = None,
    action_l2: float = 0.0,
    prev_action: Optional[torch.Tensor] = None,
    action_delta_huber_beta: float = 0.05,
) -> Mapping[str, torch.Tensor]:
    current_entropy = posterior_entropy(post_logvar)
    next_entropy = posterior_entropy(next_post_logvar)
    information_gain = current_entropy - next_entropy

    action_penalty = torch.zeros_like(information_gain)
    if action_l2 > 0.0:
        if action is None or action_low is None or action_high is None or prev_action is None:
            raise ValueError(
                "action, prev_action, action_low, and action_high are required when action_l2 > 0."
            )
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
        "current_entropy": current_entropy,
        "next_entropy": next_entropy,
        "action_penalty": action_penalty,
    }
