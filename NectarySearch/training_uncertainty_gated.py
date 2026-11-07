from __future__ import annotations

import copy
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

try:
    from nectary_allocentric_env import AllocentricNectarSearchEnv
    from uncertainty_gated_model import (
        GateRolloutBatch,
        RolloutBatch,
        UncertaintyGatedAllocentricNectarySearchModel,
        apply_stateful_fixed_threshold_gate,
    )
except ImportError:
    from .nectary_allocentric_env import AllocentricNectarSearchEnv
    from .uncertainty_gated_model import (
        GateRolloutBatch,
        RolloutBatch,
        UncertaintyGatedAllocentricNectarySearchModel,
        apply_stateful_fixed_threshold_gate,
    )


def _allocentric_vision_shape(env: AllocentricNectarSearchEnv) -> Tuple[int, ...]:
    vision_shape = tuple(env.observation_space["vision"].shape)
    if len(vision_shape) == 2:
        return (1, *vision_shape)
    return vision_shape


def _resolve_training_device(device=None, local_rank=None):
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        if local_rank is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{local_rank}")

    return torch.device("cpu")


def _distributed_ready():
    return dist.is_available() and dist.is_initialized()


def _reduce_scalar(value, device, world_size):
    if not _distributed_ready() or world_size <= 1:
        return float(value)

    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor.item()


def _unique_parameters(parameters: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    unique = []
    seen = set()
    for parameter in parameters:
        if parameter is None or (not parameter.requires_grad):
            continue
        pid = id(parameter)
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(parameter)
    return unique


def _assign_gradients(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[Optional[torch.Tensor]],
):
    for parameter, grad in zip(parameters, gradients):
        parameter.grad = None if grad is None else grad.detach().clone()


def _average_gradients(
    parameters: Sequence[torch.nn.Parameter],
    world_size: int,
):
    if not _distributed_ready() or world_size <= 1:
        return

    for parameter in parameters:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad /= world_size


def _clip_gradients(
    parameters: Sequence[torch.nn.Parameter],
    max_norm: Optional[float],
):
    if max_norm is None:
        return
    params_with_grad = [parameter for parameter in parameters if parameter.grad is not None]
    if params_with_grad:
        torch.nn.utils.clip_grad_norm_(params_with_grad, float(max_norm))


def _assert_finite_parameters(
    label: str,
    parameters: Sequence[torch.nn.Parameter],
    epoch: int,
):
    bad_count = 0
    total_count = 0
    for parameter in parameters:
        total_count += parameter.numel()
        bad_count += int((~torch.isfinite(parameter.detach())).sum().item())
    if bad_count:
        raise FloatingPointError(
            f"Non-finite {label} parameters before epoch {epoch + 1}: "
            f"{bad_count}/{total_count} values. Recreate the model from a clean "
            "checkpoint before continuing."
        )


def _assert_finite_loss(
    label: str,
    loss: Optional[torch.Tensor],
    epoch: int,
):
    if loss is None or not torch.is_tensor(loss):
        return
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise FloatingPointError(
            f"Non-finite {label} at epoch {epoch + 1}. The optimizer step was "
            "not taken, so model parameters were left untouched for this epoch."
        )


def _assert_finite_gradients(
    label: str,
    parameters: Sequence[torch.nn.Parameter],
    epoch: int,
):
    bad_count = 0
    total_count = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total_count += parameter.grad.numel()
        bad_count += int((~torch.isfinite(parameter.grad)).sum().item())
    if bad_count:
        raise FloatingPointError(
            f"Non-finite {label} gradients at epoch {epoch + 1}: "
            f"{bad_count}/{total_count} values. The optimizer step was not taken."
        )


def _mean_tensor_or_zero(
    tensors: Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if not tensors:
        return torch.zeros((), device=device, dtype=torch.float32)
    return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0).mean()


def _autograd_grad_or_none(
    loss: Optional[torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    retain_graph: bool,
):
    if loss is None or (not loss.requires_grad):
        return [None] * len(parameters)
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _obs_dict_to_tensors(obs, device):
    mech = torch.as_tensor(obs["mech"], dtype=torch.float32, device=device).reshape(1, -1)
    vis = torch.as_tensor(obs["vision"], dtype=torch.float32, device=device)
    if vis.ndim == 2:
        vis = vis.unsqueeze(0)
    vis = vis.unsqueeze(0)
    return mech, vis


def _obs_batch_to_tensors(obs_batch, device):
    mech = torch.as_tensor(
        np.stack([obs["mech"] for obs in obs_batch], axis=0),
        dtype=torch.float32,
        device=device,
    ).reshape(len(obs_batch), -1)
    vis = torch.as_tensor(
        np.stack([obs["vision"] for obs in obs_batch], axis=0),
        dtype=torch.float32,
        device=device,
    )
    if vis.ndim == 3:
        vis = vis.unsqueeze(1)
    return mech, vis


def _allocentric_action_from_pose(env: AllocentricNectarSearchEnv) -> np.ndarray:
    head_norm = env._normalize_canvas_xy(env.head_xy)
    tip_norm = env._normalize_canvas_xy(env.tip_xy)
    return np.array(
        [head_norm[0], head_norm[1], tip_norm[0], tip_norm[1], float(env.tip_depth)],
        dtype=np.float32,
    )


def _allocentric_actions_from_poses(envs: Sequence[AllocentricNectarSearchEnv]) -> np.ndarray:
    return np.stack([_allocentric_action_from_pose(env) for env in envs], axis=0)


def _clone_allocentric_env(env: AllocentricNectarSearchEnv) -> AllocentricNectarSearchEnv:
    return AllocentricNectarSearchEnv(
        flower=copy.deepcopy(env.flower),
        canvas=copy.deepcopy(env.canvas),
        dt=env.dt,
        max_steps=env.max_steps,
        max_press_depth=env.max_press_depth,
        min_depth=env.min_depth,
        max_depth=env.max_depth,
        proboscis_length=env.proboscis_length,
        visual_field_size=env.visual_field_size,
        visual_patch_shape=env.visual_patch_shape,
        non_stationary=copy.deepcopy(env.non_stationary),
    )


def _index_response_batch(responses, indices: torch.Tensor):
    return {
        key: value.index_select(0, indices) if torch.is_tensor(value) else value
        for key, value in responses.items()
    }


def _hidden_from_responses_for_next_step(responses, detach_hidden_state: bool):
    h = responses["h"].detach() if detach_hidden_state else responses["h"]
    if "c" not in responses:
        return h
    c = responses["c"].detach() if detach_hidden_state else responses["c"]
    return h, c


def _loss_vector(losses, name: str) -> torch.Tensor:
    per_sample_name = f"{name}_per_sample"
    if per_sample_name in losses:
        return losses[per_sample_name].reshape(-1)
    return losses[name].reshape(1)


def _index_loss_batch(losses, indices: torch.Tensor):
    indexed = {}
    for key, value in losses.items():
        if not torch.is_tensor(value):
            indexed[key] = value
            continue
        if key.endswith("_per_sample"):
            indexed[key] = value.index_select(0, indices)
            continue
        per_sample_name = f"{key}_per_sample"
        per_sample = losses.get(per_sample_name)
        if torch.is_tensor(per_sample):
            indexed[key] = per_sample.index_select(0, indices).mean()
        else:
            indexed[key] = value
    return indexed


def _allocentric_action_delta_bounds(
    env: AllocentricNectarSearchEnv,
) -> tuple[np.ndarray, np.ndarray]:
    xy_step_x = min(1.0, float(env.proboscis_length / (np.sqrt(2.0) * env.canvas.width)))
    xy_step_y = min(1.0, float(env.proboscis_length / (np.sqrt(2.0) * env.canvas.height)))
    depth_step = min(float(env.max_press_depth), float(env.max_depth - env.min_depth))
    delta_high = np.array(
        [xy_step_x, xy_step_y, xy_step_x, xy_step_y, depth_step],
        dtype=np.float32,
    )
    delta_low = -delta_high
    return delta_low, delta_high


def _build_default_model(
    env: AllocentricNectarSearchEnv,
    latent_dim: int = 5,
    hidden_dim: int = 64,
    beta: float = 0.1,
    rl_algorithm: str = "ppo",
    exploration_objective: str = "expected_information_gain",
    visual_likelihood: str = "gaussian",
    learn_decoder_variance: bool = True,
    posterior_fusion: str = "early",
    moment_matching: bool = True,
    uncertainty_metric: str = "aggregate",
    rnn_type: str = "rnn",
    modality_rnn_mode: str = "shared",
    min_depth: Optional[float] = None,
    action_selection_strategy: str = "uncertainty_gate",
    epsilon_greedy_probability: float = 0.1,
    fixed_switch_timestep: Optional[int] = None,
    random_switch_probability: float = 0.1,
    information_gain_reward_coef: float = 1.0,
):
    action_low = env.action_space.low.copy()
    if min_depth is not None:
        action_low[-1] = max(float(action_low[-1]), float(min_depth))
    action_delta_low, action_delta_high = _allocentric_action_delta_bounds(env)
    return UncertaintyGatedAllocentricNectarySearchModel(
        mech_obs_dim=env.observation_space["mech"].shape[0],
        vis_obs_shape=_allocentric_vision_shape(env),
        action_dim=env.action_space.shape[0],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        beta=beta,
        visual_likelihood=visual_likelihood,
        learn_decoder_variance=learn_decoder_variance,
        posterior_fusion=posterior_fusion,
        moment_matching=moment_matching,
        uncertainty_metric=uncertainty_metric,
        rnn_type=rnn_type,
        modality_rnn_mode=modality_rnn_mode,
        action_low=action_low,
        action_high=env.action_space.high,
        action_delta_low=action_delta_low,
        action_delta_high=action_delta_high,
        rl_algorithm=rl_algorithm,
        exploration_objective=exploration_objective,
        action_selection_strategy=action_selection_strategy,
        epsilon_greedy_probability=epsilon_greedy_probability,
        fixed_switch_timestep=fixed_switch_timestep,
        random_switch_probability=random_switch_probability,
        information_gain_reward_coef=information_gain_reward_coef,
    )


def _make_training_action(
    chosen_action: torch.Tensor,
    min_depth: Optional[float],
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    env_action = chosen_action.detach().clone()
    env_action = torch.max(env_action, action_low)
    env_action = torch.min(env_action, action_high)
    if min_depth is not None:
        env_action[..., -1] = env_action[..., -1].clamp(min=float(min_depth))
    return env_action


def _initialize_metrics_csv(
    metrics_csv_path: str,
    fieldnames: Sequence[str],
    append: bool = False,
):
    csv_path = Path(metrics_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = (not append) or (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    mode = "a" if append else "w"
    with csv_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()


def _append_metrics_csv_row(
    metrics_csv_path: str,
    fieldnames: Sequence[str],
    row: Dict[str, float],
):
    csv_path = Path(metrics_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _load_optimizer_state_if_compatible(
    optimizer: Optional[torch.optim.Optimizer],
    state_dict,
    device: torch.device,
    label: str,
    *,
    rank: int = 0,
    parallel: bool = False,
):
    if optimizer is None or state_dict is None:
        return False
    try:
        optimizer.load_state_dict(state_dict)
    except ValueError as exc:
        if not parallel or rank == 0:
            print(
                f"Warning: Could not restore {label} optimizer state from the checkpoint "
                f"({exc}). Continuing with a freshly initialized {label} optimizer."
            )
        return False
    _move_optimizer_state_to_device(optimizer, device)
    return True


def _head_lr_scale_for_epoch(
    epoch: int,
    total_epochs: int,
    schedule_start_epoch: int,
) -> float:
    if schedule_start_epoch < 0:
        raise ValueError("head_lr_scheduler_start_epoch must be non-negative.")
    if total_epochs <= 0 or schedule_start_epoch >= total_epochs or epoch < schedule_start_epoch:
        return 1.0
    anneal_epochs = max(1, total_epochs - schedule_start_epoch)
    remaining_epochs = max(0, total_epochs - epoch)
    return max(0.0, min(1.0, remaining_epochs / anneal_epochs))


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
):
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(learning_rate)


def _stateful_fixed_gate_enabled(
    model: UncertaintyGatedAllocentricNectarySearchModel,
    force_mode: Optional[str],
    gate_hysteresis: float,
    gate_uncertainty_ema_alpha: Optional[float],
) -> bool:
    normalized_force_mode = (
        None if force_mode is None else str(force_mode).strip().lower()
    )
    if normalized_force_mode in {"", "auto", "none"}:
        normalized_force_mode = None

    gate_hysteresis = float(gate_hysteresis)
    if gate_hysteresis < 0.0:
        raise ValueError("gate_hysteresis must be non-negative.")
    if gate_uncertainty_ema_alpha is not None:
        gate_uncertainty_ema_alpha = float(gate_uncertainty_ema_alpha)
        if gate_uncertainty_ema_alpha < 0.0 or gate_uncertainty_ema_alpha >= 1.0:
            raise ValueError("gate_uncertainty_ema_alpha must be in [0, 1).")

    requested = gate_hysteresis > 0.0 or gate_uncertainty_ema_alpha is not None
    if not requested or normalized_force_mode is not None:
        return False
    if not model.uses_uncertainty_gate:
        return False
    if model.gate_uses_rl:
        raise ValueError(
            "EMA/hysteresis gating is only supported with gate_mode='fixed_threshold'."
        )
    return True


def collect_sequential_uncertainty_gated_rollout_batch(
    env: AllocentricNectarSearchEnv,
    model: UncertaintyGatedAllocentricNectarySearchModel,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    min_depth: Optional[float] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    deterministic_exploitation: bool = False,
    deterministic_exploration: bool = False,
    detach_hidden_state: bool = False,
    force_mode: Optional[str] = None,
    gate_hysteresis: float = 0.0,
    gate_uncertainty_ema_alpha: Optional[float] = None,
):
    train_exploration_policy = model.uses_exploration_policy
    exploration_uses_rl = train_exploration_policy and model.exploration_uses_rl
    differentiable_exploration = train_exploration_policy and not model.exploration_uses_rl
    use_combined_exploitation_reward = (
        model.action_selection_strategy == "non_lexicographic"
    )
    track_executed_information_gain = model.action_selection_strategy in {
        "epsilon_greedy",
        "non_lexicographic",
    }
    use_stateful_fixed_gate = _stateful_fixed_gate_enabled(
        model,
        force_mode,
        gate_hysteresis,
        gate_uncertainty_ema_alpha,
    )

    perception_loss_terms = []
    recon_loss_terms = []
    kl_loss_terms = []
    exploration_loss_terms = []
    information_gain_terms = []
    uncertainty_terms = []

    flat_belief_features = []
    flat_prev_actions = []
    flat_exploitation_actions = []
    flat_executed_actions = []
    flat_rewards = []
    flat_dones = []
    flat_returns = []
    flat_advantages = []
    flat_old_log_probs = []
    flat_action_masks = []
    flat_rollout_exploration_values = []
    flat_rollout_exploitation_values = []
    flat_rollout_raw_uncertainties = []
    flat_rollout_smoothed_uncertainties = []
    flat_explore_probabilities = []

    flat_gate_belief_features = []
    flat_gate_actions = []
    flat_gate_rewards = []
    flat_gate_dones = []
    flat_gate_returns = []
    flat_gate_advantages = []
    flat_gate_old_log_probs = []

    flat_exploration_belief_features = []
    flat_exploration_prev_actions = []
    flat_exploration_actions = []
    flat_exploration_executed_actions = []
    flat_exploration_rewards = []
    flat_exploration_dones = []
    flat_exploration_returns = []
    flat_exploration_advantages = []
    flat_exploration_old_log_probs = []
    flat_exploration_values = []
    flat_exploration_raw_uncertainties = []
    flat_exploration_smoothed_uncertainties = []

    episode_returns = []
    episode_policy_returns = []
    episode_lengths = []
    explore_step_count = 0.0
    success_count = 0.0
    total_steps = 0

    action_low = model.action_low.to(device)
    action_high = model.action_high.to(device)

    for _ in range(batch_size):
        obs, _ = env.reset(seed=random.randint(0, 1_000_000))
        a_prev_np = _allocentric_action_from_pose(env)
        a_prev_t = torch.as_tensor(a_prev_np, dtype=torch.float32, device=device).reshape(1, -1)
        h_prev = model.initial_hidden(batch_size=1, device=device, dtype=torch.float32)
        x_mech_t, x_vis_t = _obs_dict_to_tensors(obs, device)
        responses_t, perception_losses_t = model.forward_t(x_mech_t, x_vis_t, a_prev_t, h_prev)
        gate_is_exploration = None
        gate_uncertainty_ema = None

        episode_reward_scalars = []
        episode_done_scalars = []
        episode_value_terms = []

        episode_belief_features = []
        episode_prev_actions = []
        episode_exploitation_actions = []
        episode_executed_actions = []
        episode_old_log_probs = []
        episode_action_masks = []
        episode_exploration_values = []
        episode_exploitation_values = []
        episode_raw_uncertainties = []
        episode_smoothed_uncertainties = []
        episode_gate_actions = []
        episode_gate_values = []
        episode_gate_old_log_probs = []

        reward_sum = 0.0
        policy_reward_sum = 0.0

        for _step in range(seq_len):
            selection = model.select_action_from_responses(
                responses_t=responses_t,
                prev_action=a_prev_t,
                deterministic_exploitation=deterministic_exploitation,
                deterministic_exploration=deterministic_exploration,
                stochastic_gate=model.gate_uses_rl,
                detach_gate_input=True,
                force_mode=force_mode,
                episode_timestep=_step,
            )
            if use_stateful_fixed_gate:
                (
                    selection,
                    gate_is_exploration,
                    gate_uncertainty_ema,
                ) = apply_stateful_fixed_threshold_gate(
                    selection,
                    previous_is_exploration=gate_is_exploration,
                    previous_smoothed_uncertainty=gate_uncertainty_ema,
                    uncertainty_ema_alpha=gate_uncertainty_ema_alpha,
                    hysteresis=gate_hysteresis,
                )
            exploration_losses_t = None
            if differentiable_exploration:
                exploration_losses_t = model.compute_exploration_loss_from_responses(
                    responses_t,
                    action=selection.exploration_action,
                    prev_action=a_prev_t,
                )

            raw_uncertainty = selection.uncertainty.detach().reshape(())
            smoothed_uncertainty = (
                raw_uncertainty
                if gate_uncertainty_ema is None
                else gate_uncertainty_ema.detach().reshape(())
            )
            selected_is_exploration = selection.selected_is_exploration
            exploitation_value = selection.exploitation_value.detach().reshape(())
            exploration_value = (
                torch.full((), float("nan"), dtype=torch.float32, device=device)
                if selection.exploration_value is None
                else selection.exploration_value.detach().reshape(())
            )
            gate_value = (
                None
                if selection.gate_value is None
                else selection.gate_value.detach().reshape(())
            )

            env_action_t = _make_training_action(
                selection.action,
                min_depth=min_depth,
                action_low=action_low,
                action_high=action_high,
            )
            env_action_np = env_action_t.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(env_action_np)
            executed_action_t = torch.as_tensor(
                _allocentric_action_from_pose(env),
                dtype=torch.float32,
                device=device,
            ).reshape(1, -1)
            done = bool(terminated or truncated)
            next_x_mech_t, next_x_vis_t = _obs_dict_to_tensors(next_obs, device)
            if "c" in responses_t:
                if detach_hidden_state:
                    next_h_prev = (responses_t["h"].detach(), responses_t["c"].detach())
                else:
                    next_h_prev = (responses_t["h"], responses_t["c"])
            else:
                next_h_prev = responses_t["h"].detach() if detach_hidden_state else responses_t["h"]
            next_responses_t, next_perception_losses_t = model.forward_t(
                next_x_mech_t,
                next_x_vis_t,
                executed_action_t,
                next_h_prev,
            )

            perception_loss_terms.append(perception_losses_t["total"])
            recon_loss_terms.append(perception_losses_t["recon"].detach())
            kl_loss_terms.append(perception_losses_t["kl"].detach())
            uncertainty_terms.append(raw_uncertainty)
            flat_explore_probabilities.append(
                selection.explore_probability.detach().reshape(())
            )
            executed_information_gain = None
            if track_executed_information_gain:
                executed_stats = model.compute_realized_information_gain_from_responses(
                    responses_t={
                        key: value.detach()
                        for key, value in responses_t.items()
                        if torch.is_tensor(value)
                    },
                    next_responses_t={
                        key: value.detach()
                        for key, value in next_responses_t.items()
                        if torch.is_tensor(value)
                    },
                    action=selection.action.detach(),
                    prev_action=a_prev_t.detach(),
                )
                executed_information_gain = executed_stats[
                    "information_gain"
                ].detach().reshape(())
                information_gain_terms.append(executed_information_gain)
            if exploration_uses_rl:
                if bool(selected_is_exploration.reshape(()).item()):
                    if (not deterministic_exploration) and (selection.exploration_log_prob is None or selection.exploration_value is None):
                        raise RuntimeError(
                            "Realized-information-gain exploration requires stochastic "
                            "exploration log-probabilities and values."
                        )
                    exploration_stats = model.compute_realized_information_gain_from_responses(
                        responses_t={
                            key: value.detach()
                            for key, value in responses_t.items()
                            if torch.is_tensor(value)
                        },
                        next_responses_t={
                            key: value.detach()
                            for key, value in next_responses_t.items()
                            if torch.is_tensor(value)
                        },
                        action=selection.exploration_action.detach(),
                        prev_action=a_prev_t.detach(),
                        action_low=action_low,
                        action_high=action_high,
                        action_l2=model.exploration_action_l2,
                        action_delta_huber_beta=model.exploration_action_delta_huber_beta,
                    )
                    realized_reward = exploration_stats["reward"].detach().reshape(())
                    realized_information_gain = exploration_stats["information_gain"].detach().reshape(())

                    flat_exploration_belief_features.append(
                        selection.belief_features.detach().squeeze(0)
                    )
                    flat_exploration_prev_actions.append(a_prev_t.detach().squeeze(0))
                    flat_exploration_actions.append(
                        selection.exploration_action.detach().squeeze(0)
                    )
                    flat_exploration_executed_actions.append(
                        executed_action_t.detach().squeeze(0)
                    )
                    flat_exploration_rewards.append(realized_reward)
                    flat_exploration_dones.append(
                        torch.ones((), dtype=torch.float32, device=device)
                    )
                    flat_exploration_returns.append(realized_reward)
                    flat_exploration_advantages.append(
                        (realized_reward - exploration_value).detach()
                    )
                    flat_exploration_old_log_probs.append(
                        torch.zeros((), dtype=torch.float32, device=device)
                        if selection.exploration_log_prob is None
                        else selection.exploration_log_prob.detach().reshape(())
                    )
                    flat_exploration_values.append(exploration_value)
                    flat_exploration_raw_uncertainties.append(raw_uncertainty)
                    flat_exploration_smoothed_uncertainties.append(
                        smoothed_uncertainty
                    )
                    information_gain_terms.append(realized_information_gain)
            elif differentiable_exploration:
                exploration_loss_terms.append(exploration_losses_t["loss"])
                information_gain_terms.append(
                    exploration_losses_t["information_gain"].detach().reshape(())
                )

            episode_belief_features.append(
                selection.belief_features.detach().squeeze(0)
            )
            episode_prev_actions.append(a_prev_t.detach().squeeze(0))
            episode_exploitation_actions.append(selection.exploitation_action.detach().squeeze(0))
            episode_executed_actions.append(executed_action_t.detach().squeeze(0))
            if selection.exploitation_log_prob is None:
                episode_old_log_probs.append(
                    torch.zeros((), dtype=torch.float32, device=device)
                )
            else:
                episode_old_log_probs.append(selection.exploitation_log_prob.detach().reshape(()))
            episode_action_masks.append((~selected_is_exploration).float().reshape(()))
            episode_exploration_values.append(exploration_value)
            episode_exploitation_values.append(exploitation_value)
            episode_raw_uncertainties.append(raw_uncertainty)
            episode_smoothed_uncertainties.append(smoothed_uncertainty)
            episode_value_terms.append(exploitation_value)
            policy_reward = float(reward)
            if use_combined_exploitation_reward and executed_information_gain is not None:
                policy_reward += (
                    model.information_gain_reward_coef
                    * float(executed_information_gain.item())
                )
            episode_reward_scalars.append(policy_reward)
            episode_done_scalars.append(float(done))
            if (
                model.uses_uncertainty_gate
                and model.gate_uses_rl
                and force_mode is None
            ):
                if selection.gate_log_prob is None or gate_value is None:
                    raise RuntimeError(
                        "RL gate mode requires sampled gate log-probabilities and values."
                    )
                episode_gate_actions.append(
                    selected_is_exploration.float().reshape(())
                )
                episode_gate_values.append(gate_value)
                episode_gate_old_log_probs.append(
                    selection.gate_log_prob.detach().reshape(())
                )

            reward_sum += float(reward)
            policy_reward_sum += float(policy_reward)
            total_steps += 1
            explore_step_count += float(selected_is_exploration.float().mean().item())
            success_count += float(info.get("success", False))

            obs = next_obs
            a_prev_t = executed_action_t
            responses_t = next_responses_t
            perception_losses_t = next_perception_losses_t

            if done:
                break

        rewards_t = torch.as_tensor(episode_reward_scalars, dtype=torch.float32, device=device).unsqueeze(0)
        dones_t = torch.as_tensor(episode_done_scalars, dtype=torch.float32, device=device).unsqueeze(0)
        values_t = torch.stack(episode_value_terms, dim=0).unsqueeze(0)
        returns_t, advantages_t = model.compute_returns_and_advantages(
            rewards_t,
            dones_t,
            values=values_t,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        flat_belief_features.extend(episode_belief_features)
        flat_prev_actions.extend(episode_prev_actions)
        flat_exploitation_actions.extend(episode_exploitation_actions)
        flat_executed_actions.extend(episode_executed_actions)
        flat_rewards.extend(rewards_t.squeeze(0).unbind(dim=0))
        flat_dones.extend(dones_t.squeeze(0).unbind(dim=0))
        flat_returns.extend(returns_t.squeeze(0).unbind(dim=0))
        flat_advantages.extend(advantages_t.squeeze(0).unbind(dim=0))
        flat_old_log_probs.extend(episode_old_log_probs)
        flat_action_masks.extend(episode_action_masks)
        flat_rollout_exploration_values.extend(episode_exploration_values)
        flat_rollout_exploitation_values.extend(episode_exploitation_values)
        flat_rollout_raw_uncertainties.extend(episode_raw_uncertainties)
        flat_rollout_smoothed_uncertainties.extend(episode_smoothed_uncertainties)
        if episode_gate_actions:
            gate_values_t = torch.stack(episode_gate_values, dim=0).unsqueeze(0)
            gate_returns_t, gate_advantages_t = model.compute_returns_and_advantages(
                rewards_t,
                dones_t,
                values=gate_values_t,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            flat_gate_belief_features.extend(episode_belief_features)
            flat_gate_actions.extend(episode_gate_actions)
            flat_gate_rewards.extend(rewards_t.squeeze(0).unbind(dim=0))
            flat_gate_dones.extend(dones_t.squeeze(0).unbind(dim=0))
            flat_gate_returns.extend(gate_returns_t.squeeze(0).unbind(dim=0))
            flat_gate_advantages.extend(gate_advantages_t.squeeze(0).unbind(dim=0))
            flat_gate_old_log_probs.extend(episode_gate_old_log_probs)

        episode_returns.append(reward_sum)
        episode_policy_returns.append(policy_reward_sum)
        episode_lengths.append(len(episode_reward_scalars))

    perception_loss = _mean_tensor_or_zero(perception_loss_terms, device)
    exploration_loss = _mean_tensor_or_zero(exploration_loss_terms, device)
    uncertainty = _mean_tensor_or_zero(uncertainty_terms, device)

    rollout = RolloutBatch(
        belief_features=torch.stack(flat_belief_features, dim=0),
        prev_actions=torch.stack(flat_prev_actions, dim=0),
        actions=torch.stack(flat_exploitation_actions, dim=0),
        rewards=torch.stack(flat_rewards, dim=0),
        dones=torch.stack(flat_dones, dim=0),
        returns=torch.stack(flat_returns, dim=0),
        executed_actions=torch.stack(flat_executed_actions, dim=0),
        advantages=torch.stack(flat_advantages, dim=0),
        old_log_probs=torch.stack(flat_old_log_probs, dim=0),
        action_mask=torch.stack(flat_action_masks, dim=0),
        exploration_values=torch.stack(flat_rollout_exploration_values, dim=0),
        exploitation_values=torch.stack(flat_rollout_exploitation_values, dim=0),
        raw_uncertainties=torch.stack(flat_rollout_raw_uncertainties, dim=0),
        smoothed_uncertainties=torch.stack(
            flat_rollout_smoothed_uncertainties,
            dim=0,
        ),
    )
    gate_rollout = None
    if flat_gate_belief_features:
        gate_rollout = GateRolloutBatch(
            belief_features=torch.stack(flat_gate_belief_features, dim=0),
            actions=torch.stack(flat_gate_actions, dim=0),
            rewards=torch.stack(flat_gate_rewards, dim=0),
            dones=torch.stack(flat_gate_dones, dim=0),
            returns=torch.stack(flat_gate_returns, dim=0),
            advantages=torch.stack(flat_gate_advantages, dim=0),
            old_log_probs=torch.stack(flat_gate_old_log_probs, dim=0),
        )
    exploration_rollout = None
    if flat_exploration_belief_features:
        exploration_rollout = RolloutBatch(
            belief_features=torch.stack(flat_exploration_belief_features, dim=0),
            prev_actions=torch.stack(flat_exploration_prev_actions, dim=0),
            actions=torch.stack(flat_exploration_actions, dim=0),
            rewards=torch.stack(flat_exploration_rewards, dim=0),
            dones=torch.stack(flat_exploration_dones, dim=0),
            returns=torch.stack(flat_exploration_returns, dim=0),
            executed_actions=torch.stack(flat_exploration_executed_actions, dim=0),
            advantages=torch.stack(flat_exploration_advantages, dim=0),
            old_log_probs=torch.stack(flat_exploration_old_log_probs, dim=0),
            exploration_values=torch.stack(flat_exploration_values, dim=0),
            raw_uncertainties=torch.stack(
                flat_exploration_raw_uncertainties,
                dim=0,
            ),
            smoothed_uncertainties=torch.stack(
                flat_exploration_smoothed_uncertainties,
                dim=0,
            ),
        )

    metrics = {
        "perception_total": float(perception_loss.detach().item()),
        "perception_recon": float(_mean_tensor_or_zero(recon_loss_terms, device).item()),
        "perception_kl": float(_mean_tensor_or_zero(kl_loss_terms, device).item()),
        "uncertainty": float(uncertainty.item()),
        "exploration_loss": float(exploration_loss.detach().item()),
        "exploration_information_gain": float(
            _mean_tensor_or_zero(information_gain_terms, device).item()
        ),
        "exploration_sample_count": float(len(flat_exploration_rewards)),
        "gate_sample_count": float(len(flat_gate_actions)),
        "mean_explore_probability": float(
            _mean_tensor_or_zero(flat_explore_probabilities, device).item()
        ),
        "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_policy_return": (
            float(np.mean(episode_policy_returns)) if episode_policy_returns else 0.0
        ),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "explore_fraction": float(explore_step_count / max(total_steps, 1)),
        "success_rate": float(success_count / max(batch_size, 1)),
        "total_steps": float(total_steps),
    }

    return {
        "rollout": rollout,
        "gate_rollout": gate_rollout,
        "exploration_rollout": exploration_rollout,
        "perception_loss": perception_loss,
        "exploration_loss": exploration_loss,
        "metrics": metrics,
    }


def collect_uncertainty_gated_rollout_batch(
    env: AllocentricNectarSearchEnv,
    model: UncertaintyGatedAllocentricNectarySearchModel,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    min_depth: Optional[float] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    deterministic_exploitation: bool = False,
    deterministic_exploration: bool = False,
    detach_hidden_state: bool = False,
    force_mode: Optional[str] = None,
    gate_hysteresis: float = 0.0,
    gate_uncertainty_ema_alpha: Optional[float] = None,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    train_exploration_policy = model.uses_exploration_policy
    exploration_uses_rl = train_exploration_policy and model.exploration_uses_rl
    differentiable_exploration = train_exploration_policy and not model.exploration_uses_rl
    use_combined_exploitation_reward = (
        model.action_selection_strategy == "non_lexicographic"
    )
    track_executed_information_gain = model.action_selection_strategy in {
        "epsilon_greedy",
        "non_lexicographic",
    }
    use_stateful_fixed_gate = _stateful_fixed_gate_enabled(
        model,
        force_mode,
        gate_hysteresis,
        gate_uncertainty_ema_alpha,
    )
    rollout_envs = [env, *[_clone_allocentric_env(env) for _ in range(batch_size - 1)]]

    perception_loss_terms = []
    recon_loss_terms = []
    kl_loss_terms = []
    exploration_loss_terms = []
    information_gain_terms = []
    uncertainty_terms = []
    flat_explore_probabilities = []

    episode_belief_features = [[] for _ in range(batch_size)]
    episode_prev_actions = [[] for _ in range(batch_size)]
    episode_exploitation_actions = [[] for _ in range(batch_size)]
    episode_executed_actions = [[] for _ in range(batch_size)]
    episode_old_log_probs = [[] for _ in range(batch_size)]
    episode_action_masks = [[] for _ in range(batch_size)]
    episode_exploration_values = [[] for _ in range(batch_size)]
    episode_exploitation_values = [[] for _ in range(batch_size)]
    episode_raw_uncertainties = [[] for _ in range(batch_size)]
    episode_smoothed_uncertainties = [[] for _ in range(batch_size)]
    episode_reward_scalars = [[] for _ in range(batch_size)]
    episode_done_scalars = [[] for _ in range(batch_size)]
    episode_value_terms = [[] for _ in range(batch_size)]
    episode_gate_actions = [[] for _ in range(batch_size)]
    episode_gate_values = [[] for _ in range(batch_size)]
    episode_gate_old_log_probs = [[] for _ in range(batch_size)]
    episode_returns = [0.0 for _ in range(batch_size)]
    episode_policy_returns = [0.0 for _ in range(batch_size)]

    flat_belief_features = []
    flat_prev_actions = []
    flat_exploitation_actions = []
    flat_executed_actions = []
    flat_rewards = []
    flat_dones = []
    flat_returns = []
    flat_advantages = []
    flat_old_log_probs = []
    flat_action_masks = []
    flat_rollout_exploration_values = []
    flat_rollout_exploitation_values = []
    flat_rollout_raw_uncertainties = []
    flat_rollout_smoothed_uncertainties = []

    flat_gate_belief_features = []
    flat_gate_actions = []
    flat_gate_rewards = []
    flat_gate_dones = []
    flat_gate_returns = []
    flat_gate_advantages = []
    flat_gate_old_log_probs = []

    flat_exploration_belief_features = []
    flat_exploration_prev_actions = []
    flat_exploration_actions = []
    flat_exploration_executed_actions = []
    flat_exploration_rewards = []
    flat_exploration_dones = []
    flat_exploration_returns = []
    flat_exploration_advantages = []
    flat_exploration_old_log_probs = []
    flat_exploration_values = []
    flat_exploration_raw_uncertainties = []
    flat_exploration_smoothed_uncertainties = []

    explore_step_count = 0.0
    success_count = 0.0
    total_steps = 0

    action_low = model.action_low.to(device)
    action_high = model.action_high.to(device)

    obs_batch = []
    for rollout_env in rollout_envs:
        obs, _ = rollout_env.reset(seed=random.randint(0, 1_000_000))
        obs_batch.append(obs)

    active_envs = list(rollout_envs)
    active_episode_ids = list(range(batch_size))
    a_prev_t = torch.as_tensor(
        _allocentric_actions_from_poses(active_envs),
        dtype=torch.float32,
        device=device,
    )
    h_prev = model.initial_hidden(
        batch_size=len(active_envs),
        device=device,
        dtype=torch.float32,
    )
    x_mech_t, x_vis_t = _obs_batch_to_tensors(obs_batch, device)
    responses_t, perception_losses_t = model.forward_t(
        x_mech_t,
        x_vis_t,
        a_prev_t,
        h_prev,
    )
    active_gate_is_exploration = None
    active_gate_uncertainty_ema = None

    for _step in range(seq_len):
        if not active_envs:
            break

        selection = model.select_action_from_responses(
            responses_t=responses_t,
            prev_action=a_prev_t,
            deterministic_exploitation=deterministic_exploitation,
            deterministic_exploration=deterministic_exploration,
            stochastic_gate=model.gate_uses_rl,
            detach_gate_input=True,
            force_mode=force_mode,
            episode_timestep=_step,
        )
        if use_stateful_fixed_gate:
            (
                selection,
                active_gate_is_exploration,
                active_gate_uncertainty_ema,
            ) = apply_stateful_fixed_threshold_gate(
                selection,
                previous_is_exploration=active_gate_is_exploration,
                previous_smoothed_uncertainty=active_gate_uncertainty_ema,
                uncertainty_ema_alpha=gate_uncertainty_ema_alpha,
                hysteresis=gate_hysteresis,
            )
        exploration_losses_t = None
        if differentiable_exploration:
            exploration_losses_t = model.compute_exploration_loss_from_responses(
                responses_t,
                action=selection.exploration_action,
                prev_action=a_prev_t,
            )

        raw_uncertainty_batch = selection.uncertainty.detach().reshape(-1)
        smoothed_uncertainty_batch = (
            raw_uncertainty_batch
            if active_gate_uncertainty_ema is None
            else active_gate_uncertainty_ema.detach().reshape(-1)
        )

        perception_loss_terms.append(_loss_vector(perception_losses_t, "total"))
        recon_loss_terms.append(_loss_vector(perception_losses_t, "recon").detach())
        kl_loss_terms.append(_loss_vector(perception_losses_t, "kl").detach())
        uncertainty_terms.append(raw_uncertainty_batch)
        flat_explore_probabilities.append(
            selection.explore_probability.detach().reshape(-1)
        )
        if differentiable_exploration:
            exploration_loss_terms.append(_loss_vector(exploration_losses_t, "loss"))
            information_gain_terms.append(
                _loss_vector(exploration_losses_t, "information_gain").detach()
            )

        env_action_t = _make_training_action(
            selection.action,
            min_depth=min_depth,
            action_low=action_low,
            action_high=action_high,
        )
        env_action_np = env_action_t.detach().cpu().numpy()

        next_obs_batch = []
        rewards = []
        dones = []
        infos = []
        for rollout_env, action_np in zip(active_envs, env_action_np):
            next_obs, reward, terminated, truncated, info = rollout_env.step(action_np)
            next_obs_batch.append(next_obs)
            rewards.append(float(reward))
            dones.append(bool(terminated or truncated))
            infos.append(info)

        executed_action_t = torch.as_tensor(
            _allocentric_actions_from_poses(active_envs),
            dtype=torch.float32,
            device=device,
        )
        next_x_mech_t, next_x_vis_t = _obs_batch_to_tensors(next_obs_batch, device)
        next_h_prev = _hidden_from_responses_for_next_step(
            responses_t,
            detach_hidden_state=detach_hidden_state,
        )
        next_responses_t, next_perception_losses_t = model.forward_t(
            next_x_mech_t,
            next_x_vis_t,
            executed_action_t,
            next_h_prev,
        )

        realized_reward_batch = None
        realized_information_gain_batch = None
        executed_information_gain_batch = None
        if track_executed_information_gain:
            executed_stats = model.compute_realized_information_gain_from_responses(
                responses_t={
                    key: value.detach()
                    for key, value in responses_t.items()
                    if torch.is_tensor(value)
                },
                next_responses_t={
                    key: value.detach()
                    for key, value in next_responses_t.items()
                    if torch.is_tensor(value)
                },
                action=selection.action.detach(),
                prev_action=a_prev_t.detach(),
            )
            executed_information_gain_batch = executed_stats[
                "information_gain"
            ].detach().reshape(-1)
            information_gain_terms.append(executed_information_gain_batch)
        if exploration_uses_rl:
            selected_any = bool(selection.selected_is_exploration.any().item())
            if selected_any and (
                (not deterministic_exploration)
                and (
                    selection.exploration_log_prob is None
                    or selection.exploration_value is None
                )
            ):
                raise RuntimeError(
                    "Realized-information-gain exploration requires stochastic "
                    "exploration log-probabilities and values."
                )
            exploration_stats = model.compute_realized_information_gain_from_responses(
                responses_t={
                    key: value.detach()
                    for key, value in responses_t.items()
                    if torch.is_tensor(value)
                },
                next_responses_t={
                    key: value.detach()
                    for key, value in next_responses_t.items()
                    if torch.is_tensor(value)
                },
                action=selection.exploration_action.detach(),
                prev_action=a_prev_t.detach(),
                action_low=action_low,
                action_high=action_high,
                action_l2=model.exploration_action_l2,
                action_delta_huber_beta=model.exploration_action_delta_huber_beta,
            )
            realized_reward_batch = exploration_stats["reward"].detach().reshape(-1)
            realized_information_gain_batch = exploration_stats[
                "information_gain"
            ].detach().reshape(-1)

        selected_is_exploration = selection.selected_is_exploration.reshape(-1)
        for slot, episode_id in enumerate(active_episode_ids):
            selected_bool = bool(selected_is_exploration[slot].detach().item())
            exploitation_value = selection.exploitation_value[slot].detach().reshape(())
            exploration_value = (
                torch.full((), float("nan"), dtype=torch.float32, device=device)
                if selection.exploration_value is None
                else selection.exploration_value[slot].detach().reshape(())
            )
            gate_value = (
                None
                if selection.gate_value is None
                else selection.gate_value[slot].detach().reshape(())
            )

            episode_belief_features[episode_id].append(
                selection.belief_features[slot].detach()
            )
            episode_prev_actions[episode_id].append(a_prev_t[slot].detach())
            episode_exploitation_actions[episode_id].append(
                selection.exploitation_action[slot].detach()
            )
            episode_executed_actions[episode_id].append(
                executed_action_t[slot].detach()
            )
            if selection.exploitation_log_prob is None:
                episode_old_log_probs[episode_id].append(
                    torch.zeros((), dtype=torch.float32, device=device)
                )
            else:
                episode_old_log_probs[episode_id].append(
                    selection.exploitation_log_prob[slot].detach().reshape(())
                )
            episode_action_masks[episode_id].append(
                (~selected_is_exploration[slot]).float().reshape(())
            )
            episode_exploration_values[episode_id].append(exploration_value)
            episode_exploitation_values[episode_id].append(exploitation_value)
            episode_raw_uncertainties[episode_id].append(
                raw_uncertainty_batch[slot]
            )
            episode_smoothed_uncertainties[episode_id].append(
                smoothed_uncertainty_batch[slot]
            )
            episode_value_terms[episode_id].append(exploitation_value)
            policy_reward = float(rewards[slot])
            if (
                use_combined_exploitation_reward
                and executed_information_gain_batch is not None
            ):
                policy_reward += (
                    model.information_gain_reward_coef
                    * float(executed_information_gain_batch[slot].item())
                )
            episode_reward_scalars[episode_id].append(policy_reward)
            episode_done_scalars[episode_id].append(float(dones[slot]))

            if (
                model.uses_uncertainty_gate
                and model.gate_uses_rl
                and force_mode is None
            ):
                if selection.gate_log_prob is None or gate_value is None:
                    raise RuntimeError(
                        "RL gate mode requires sampled gate log-probabilities and values."
                    )
                episode_gate_actions[episode_id].append(
                    selected_is_exploration[slot].float().reshape(())
                )
                episode_gate_values[episode_id].append(gate_value)
                episode_gate_old_log_probs[episode_id].append(
                    selection.gate_log_prob[slot].detach().reshape(())
                )

            if exploration_uses_rl and selected_bool:
                realized_reward = realized_reward_batch[slot].reshape(())
                realized_information_gain = realized_information_gain_batch[slot].reshape(())
                flat_exploration_belief_features.append(
                    selection.belief_features[slot].detach()
                )
                flat_exploration_prev_actions.append(a_prev_t[slot].detach())
                flat_exploration_actions.append(
                    selection.exploration_action[slot].detach()
                )
                flat_exploration_executed_actions.append(
                    executed_action_t[slot].detach()
                )
                flat_exploration_rewards.append(realized_reward)
                flat_exploration_dones.append(
                    torch.ones((), dtype=torch.float32, device=device)
                )
                flat_exploration_returns.append(realized_reward)
                flat_exploration_advantages.append(
                    (realized_reward - exploration_value).detach()
                )
                flat_exploration_old_log_probs.append(
                    torch.zeros((), dtype=torch.float32, device=device)
                    if selection.exploration_log_prob is None
                    else selection.exploration_log_prob[slot].detach().reshape(())
                )
                flat_exploration_values.append(exploration_value)
                flat_exploration_raw_uncertainties.append(
                    raw_uncertainty_batch[slot]
                )
                flat_exploration_smoothed_uncertainties.append(
                    smoothed_uncertainty_batch[slot]
                )
                information_gain_terms.append(realized_information_gain)

            episode_returns[episode_id] += rewards[slot]
            episode_policy_returns[episode_id] += policy_reward
            total_steps += 1
            explore_step_count += float(selected_bool)
            success_count += float(infos[slot].get("success", False))

        keep_slots = [idx for idx, done in enumerate(dones) if not done]
        if not keep_slots:
            active_envs = []
            break

        keep_idx = torch.as_tensor(keep_slots, dtype=torch.long, device=device)
        active_envs = [active_envs[idx] for idx in keep_slots]
        active_episode_ids = [active_episode_ids[idx] for idx in keep_slots]
        a_prev_t = executed_action_t.index_select(0, keep_idx)
        if active_gate_is_exploration is not None:
            active_gate_is_exploration = active_gate_is_exploration.index_select(
                0,
                keep_idx,
            )
        if active_gate_uncertainty_ema is not None:
            active_gate_uncertainty_ema = active_gate_uncertainty_ema.index_select(
                0,
                keep_idx,
            )
        responses_t = _index_response_batch(next_responses_t, keep_idx)
        perception_losses_t = _index_loss_batch(next_perception_losses_t, keep_idx)

    for episode_id in range(batch_size):
        if not episode_reward_scalars[episode_id]:
            continue
        rewards_t = torch.as_tensor(
            episode_reward_scalars[episode_id],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        dones_t = torch.as_tensor(
            episode_done_scalars[episode_id],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        values_t = torch.stack(episode_value_terms[episode_id], dim=0).unsqueeze(0)
        returns_t, advantages_t = model.compute_returns_and_advantages(
            rewards_t,
            dones_t,
            values=values_t,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        flat_belief_features.extend(episode_belief_features[episode_id])
        flat_prev_actions.extend(episode_prev_actions[episode_id])
        flat_exploitation_actions.extend(episode_exploitation_actions[episode_id])
        flat_executed_actions.extend(episode_executed_actions[episode_id])
        flat_rewards.extend(rewards_t.squeeze(0).unbind(dim=0))
        flat_dones.extend(dones_t.squeeze(0).unbind(dim=0))
        flat_returns.extend(returns_t.squeeze(0).unbind(dim=0))
        flat_advantages.extend(advantages_t.squeeze(0).unbind(dim=0))
        flat_old_log_probs.extend(episode_old_log_probs[episode_id])
        flat_action_masks.extend(episode_action_masks[episode_id])
        flat_rollout_exploration_values.extend(
            episode_exploration_values[episode_id]
        )
        flat_rollout_exploitation_values.extend(
            episode_exploitation_values[episode_id]
        )
        flat_rollout_raw_uncertainties.extend(
            episode_raw_uncertainties[episode_id]
        )
        flat_rollout_smoothed_uncertainties.extend(
            episode_smoothed_uncertainties[episode_id]
        )

        if episode_gate_actions[episode_id]:
            gate_values_t = torch.stack(
                episode_gate_values[episode_id],
                dim=0,
            ).unsqueeze(0)
            gate_returns_t, gate_advantages_t = model.compute_returns_and_advantages(
                rewards_t,
                dones_t,
                values=gate_values_t,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            flat_gate_belief_features.extend(episode_belief_features[episode_id])
            flat_gate_actions.extend(episode_gate_actions[episode_id])
            flat_gate_rewards.extend(rewards_t.squeeze(0).unbind(dim=0))
            flat_gate_dones.extend(dones_t.squeeze(0).unbind(dim=0))
            flat_gate_returns.extend(gate_returns_t.squeeze(0).unbind(dim=0))
            flat_gate_advantages.extend(gate_advantages_t.squeeze(0).unbind(dim=0))
            flat_gate_old_log_probs.extend(episode_gate_old_log_probs[episode_id])

    perception_loss = _mean_tensor_or_zero(perception_loss_terms, device)
    exploration_loss = _mean_tensor_or_zero(exploration_loss_terms, device)
    uncertainty = _mean_tensor_or_zero(uncertainty_terms, device)

    rollout = RolloutBatch(
        belief_features=torch.stack(flat_belief_features, dim=0),
        prev_actions=torch.stack(flat_prev_actions, dim=0),
        actions=torch.stack(flat_exploitation_actions, dim=0),
        rewards=torch.stack(flat_rewards, dim=0),
        dones=torch.stack(flat_dones, dim=0),
        returns=torch.stack(flat_returns, dim=0),
        executed_actions=torch.stack(flat_executed_actions, dim=0),
        advantages=torch.stack(flat_advantages, dim=0),
        old_log_probs=torch.stack(flat_old_log_probs, dim=0),
        action_mask=torch.stack(flat_action_masks, dim=0),
        exploration_values=torch.stack(flat_rollout_exploration_values, dim=0),
        exploitation_values=torch.stack(flat_rollout_exploitation_values, dim=0),
        raw_uncertainties=torch.stack(flat_rollout_raw_uncertainties, dim=0),
        smoothed_uncertainties=torch.stack(
            flat_rollout_smoothed_uncertainties,
            dim=0,
        ),
    )

    gate_rollout = None
    if flat_gate_belief_features:
        gate_rollout = GateRolloutBatch(
            belief_features=torch.stack(flat_gate_belief_features, dim=0),
            actions=torch.stack(flat_gate_actions, dim=0),
            rewards=torch.stack(flat_gate_rewards, dim=0),
            dones=torch.stack(flat_gate_dones, dim=0),
            returns=torch.stack(flat_gate_returns, dim=0),
            advantages=torch.stack(flat_gate_advantages, dim=0),
            old_log_probs=torch.stack(flat_gate_old_log_probs, dim=0),
        )

    exploration_rollout = None
    if flat_exploration_belief_features:
        exploration_rollout = RolloutBatch(
            belief_features=torch.stack(flat_exploration_belief_features, dim=0),
            prev_actions=torch.stack(flat_exploration_prev_actions, dim=0),
            actions=torch.stack(flat_exploration_actions, dim=0),
            rewards=torch.stack(flat_exploration_rewards, dim=0),
            dones=torch.stack(flat_exploration_dones, dim=0),
            returns=torch.stack(flat_exploration_returns, dim=0),
            executed_actions=torch.stack(flat_exploration_executed_actions, dim=0),
            advantages=torch.stack(flat_exploration_advantages, dim=0),
            old_log_probs=torch.stack(flat_exploration_old_log_probs, dim=0),
            exploration_values=torch.stack(flat_exploration_values, dim=0),
            raw_uncertainties=torch.stack(
                flat_exploration_raw_uncertainties,
                dim=0,
            ),
            smoothed_uncertainties=torch.stack(
                flat_exploration_smoothed_uncertainties,
                dim=0,
            ),
        )

    episode_lengths = [len(rewards) for rewards in episode_reward_scalars if rewards]
    metrics = {
        "perception_total": float(perception_loss.detach().item()),
        "perception_recon": float(_mean_tensor_or_zero(recon_loss_terms, device).item()),
        "perception_kl": float(_mean_tensor_or_zero(kl_loss_terms, device).item()),
        "uncertainty": float(uncertainty.item()),
        "exploration_loss": float(exploration_loss.detach().item()),
        "exploration_information_gain": float(
            _mean_tensor_or_zero(information_gain_terms, device).item()
        ),
        "exploration_sample_count": float(len(flat_exploration_rewards)),
        "gate_sample_count": float(len(flat_gate_actions)),
        "mean_explore_probability": float(
            _mean_tensor_or_zero(flat_explore_probabilities, device).item()
        ),
        "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_policy_return": (
            float(np.mean(episode_policy_returns)) if episode_policy_returns else 0.0
        ),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "explore_fraction": float(explore_step_count / max(total_steps, 1)),
        "success_rate": float(success_count / max(batch_size, 1)),
        "total_steps": float(total_steps),
    }

    return {
        "rollout": rollout,
        "gate_rollout": gate_rollout,
        "exploration_rollout": exploration_rollout,
        "perception_loss": perception_loss,
        "exploration_loss": exploration_loss,
        "metrics": metrics,
    }


def train_uncertainty_gated_allocentric_model(
    envs=None,
    model: Optional[UncertaintyGatedAllocentricNectarySearchModel] = None,
    num_epochs: int = 10,
    batch_size: int = 4,
    seq_len: Optional[int] = None,
    perception_learning_rate: float = 1e-3,
    exploration_learning_rate: float = 1e-3,
    exploitation_learning_rate: float = 3e-4,
    gate_learning_rate: Optional[float] = None,
    print_every: int = 10,
    min_depth: Optional[float] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    exploration_warmup_epochs: int = 0,
    deterministic_exploitation: bool = False,
    deterministic_exploration: bool = False,
    detach_hidden_state: bool = False,
    grad_clip_norm: Optional[float] = None,
    metrics_csv_path: Optional[str] = None,
    append_metrics_csv: bool = False,
    exploration_objective: str = "expected_information_gain",
    visual_likelihood: str = "gaussian",
    learn_decoder_variance: bool = True,
    posterior_fusion: str = "early",
    moment_matching: bool = True,
    uncertainty_metric: str = "aggregate",
    rnn_type: str = "rnn",
    modality_rnn_mode: str = "shared",
    action_selection_strategy: str = "uncertainty_gate",
    epsilon_greedy_probability: float = 0.1,
    fixed_switch_timestep: Optional[int] = None,
    random_switch_probability: float = 0.1,
    information_gain_reward_coef: float = 1.0,
    force_mode: Optional[str] = None,
    gate_hysteresis: float = 0.0,
    gate_uncertainty_ema_alpha: Optional[float] = None,
    parallel: bool = False,
    rank: int = 0,
    world_size: int = 1,
    local_rank=None,
    device=None,
    wrap_ddp: bool = False,
    resume_state: Optional[Dict[str, object]] = None,
    return_training_state: bool = False,
    head_lr_scheduler: bool = False,
    head_lr_scheduler_start_epoch: int = 0,
):
    del wrap_ddp  # This trainer uses manual gradient averaging instead of DDP wrapping.
    normalized_force_mode = None if force_mode is None else str(force_mode).strip().lower()
    if normalized_force_mode in {"", "auto", "none"}:
        normalized_force_mode = None
    if normalized_force_mode not in {None, "explore", "exploit"}:
        raise ValueError("force_mode must be one of: auto, explore, exploit.")
    gate_hysteresis = float(gate_hysteresis)
    if gate_hysteresis < 0.0:
        raise ValueError("gate_hysteresis must be non-negative.")
    if gate_uncertainty_ema_alpha is not None:
        gate_uncertainty_ema_alpha = float(gate_uncertainty_ema_alpha)
        if gate_uncertainty_ema_alpha < 0.0 or gate_uncertainty_ema_alpha >= 1.0:
            raise ValueError("gate_uncertainty_ema_alpha must be in [0, 1).")
    exploration_warmup_epochs = int(exploration_warmup_epochs)
    if exploration_warmup_epochs < 0:
        raise ValueError("exploration_warmup_epochs must be non-negative.")

    if envs is None:
        envs = [AllocentricNectarSearchEnv()]
    if not envs:
        raise ValueError("envs must contain at least one allocentric environment.")
    if parallel and world_size > 1 and not _distributed_ready():
        raise RuntimeError(
            "parallel=True with world_size > 1 requires torch.distributed to be initialized."
        )

    if model is None:
        model = _build_default_model(
            envs[0],
            min_depth=min_depth,
            exploration_objective=exploration_objective,
            visual_likelihood=visual_likelihood,
            learn_decoder_variance=learn_decoder_variance,
            posterior_fusion=posterior_fusion,
            moment_matching=moment_matching,
            uncertainty_metric=uncertainty_metric,
            rnn_type=rnn_type,
            modality_rnn_mode=modality_rnn_mode,
            action_selection_strategy=action_selection_strategy,
            epsilon_greedy_probability=epsilon_greedy_probability,
            fixed_switch_timestep=fixed_switch_timestep,
            random_switch_probability=random_switch_probability,
            information_gain_reward_coef=information_gain_reward_coef,
        )

    device = _resolve_training_device(device=device, local_rank=local_rank)
    if device.type == "cuda" and local_rank is not None:
        torch.cuda.set_device(local_rank)
    model.to(device)

    if seq_len is None:
        seq_len = envs[0].max_steps

    perception_params = _unique_parameters(model.perception_parameters())
    exploration_params = _unique_parameters(model.exploration_parameters())
    policy_params = _unique_parameters(model.exploitation_parameters())
    gate_params = _unique_parameters(model.gating_parameters())

    perception_optimizer = torch.optim.Adam(perception_params, lr=perception_learning_rate)
    exploration_optimizer = torch.optim.Adam(exploration_params, lr=exploration_learning_rate)
    policy_optimizer = torch.optim.Adam(policy_params, lr=exploitation_learning_rate)
    gate_learning_rate = (
        exploitation_learning_rate
        if gate_learning_rate is None
        else float(gate_learning_rate)
    )
    gate_optimizer = (
        None
        if not gate_params
        else torch.optim.Adam(gate_params, lr=gate_learning_rate)
    )
    resume_state = dict(resume_state or {})
    start_epoch = int(resume_state.get("completed_epochs", 0))
    if start_epoch < 0:
        raise ValueError("resume_state.completed_epochs must be non-negative.")
    if start_epoch > int(num_epochs):
        raise ValueError(
            f"Cannot resume from epoch {start_epoch} when num_epochs={num_epochs}. "
            "Increase num_epochs to continue training."
        )

    optimizer_state_dicts = resume_state.get("optimizer_state_dicts")
    if optimizer_state_dicts is not None:
        if not isinstance(optimizer_state_dicts, dict):
            raise ValueError("resume_state.optimizer_state_dicts must be a mapping.")
        perception_state = optimizer_state_dicts.get("perception")
        exploration_state = optimizer_state_dicts.get("exploration")
        policy_state = optimizer_state_dicts.get("policy")
        gate_state = optimizer_state_dicts.get("gate")
        _load_optimizer_state_if_compatible(
            perception_optimizer,
            perception_state,
            device,
            "perception",
            rank=rank,
            parallel=parallel,
        )
        _load_optimizer_state_if_compatible(
            exploration_optimizer,
            exploration_state,
            device,
            "exploration",
            rank=rank,
            parallel=parallel,
        )
        _load_optimizer_state_if_compatible(
            policy_optimizer,
            policy_state,
            device,
            "policy",
            rank=rank,
            parallel=parallel,
        )
        _load_optimizer_state_if_compatible(
            gate_optimizer,
            gate_state,
            device,
            "gate",
            rank=rank,
            parallel=parallel,
        )

    csv_fieldnames = [
        "epoch",
        "perception_total",
        "perception_recon",
        "perception_kl",
        "uncertainty",
        "exploration_loss",
        "exploration_information_gain",
        "exploration_sample_count",
        "gate_sample_count",
        "policy_loss",
        "value_loss",
        "gate_policy_loss",
        "gate_value_loss",
        "gate_loss",
        "entropy_bonus",
        "gate_entropy_bonus",
        "mean_explore_probability",
        "mean_episode_return",
        "mean_policy_return",
        "mean_episode_length",
        "explore_fraction",
        "success_rate",
        "threshold",
        "flower_c",
    ]
    should_write_metrics_csv = metrics_csv_path is not None and (not parallel or rank == 0)
    if should_write_metrics_csv:
        _initialize_metrics_csv(
            metrics_csv_path=metrics_csv_path,
            fieldnames=csv_fieldnames,
            append=append_metrics_csv,
        )

    for epoch in range(start_epoch, num_epochs):
        model.train()
        _assert_finite_parameters("perception", perception_params, epoch)
        _assert_finite_parameters("exploration", exploration_params, epoch)
        _assert_finite_parameters("policy", policy_params, epoch)
        _assert_finite_parameters("gate", gate_params, epoch)
        if head_lr_scheduler:
            head_lr_scale = _head_lr_scale_for_epoch(
                epoch=epoch,
                total_epochs=num_epochs,
                schedule_start_epoch=head_lr_scheduler_start_epoch,
            )
            _set_optimizer_learning_rate(
                exploration_optimizer,
                exploration_learning_rate * head_lr_scale,
            )
            _set_optimizer_learning_rate(
                policy_optimizer,
                exploitation_learning_rate * head_lr_scale,
            )
            if gate_optimizer is not None:
                _set_optimizer_learning_rate(
                    gate_optimizer,
                    gate_learning_rate * head_lr_scale,
                )
        active_env = envs[rank % len(envs)] if parallel else random.choice(envs)
        epoch_force_mode = normalized_force_mode
        if epoch_force_mode is None and epoch < exploration_warmup_epochs:
            epoch_force_mode = "explore"

        batch = collect_uncertainty_gated_rollout_batch(
            env=active_env,
            model=model,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            min_depth=min_depth,
            gamma=gamma,
            gae_lambda=gae_lambda,
            deterministic_exploitation=deterministic_exploitation,
            deterministic_exploration=deterministic_exploration,
            detach_hidden_state=detach_hidden_state,
            force_mode=epoch_force_mode,
            gate_hysteresis=gate_hysteresis,
            gate_uncertainty_ema_alpha=gate_uncertainty_ema_alpha,
        )
        rollout = batch["rollout"]
        gate_rollout = batch["gate_rollout"]
        exploration_rollout = batch["exploration_rollout"]
        perception_loss = batch["perception_loss"]
        exploration_losses = None
        if model.exploration_uses_rl:
            if exploration_rollout is not None:
                exploration_losses = model.compute_exploration_rl_loss(exploration_rollout)
                exploration_loss = exploration_losses["loss"]
            else:
                exploration_loss = torch.zeros((), device=device, dtype=torch.float32)
        else:
            exploration_loss = batch["exploration_loss"]
            if normalized_force_mode == "exploit":
                exploration_loss = torch.zeros((), device=device, dtype=torch.float32)
        policy_losses = model.compute_exploitation_loss(rollout)
        policy_loss = policy_losses["loss"]
        if normalized_force_mode == "explore":
            policy_loss = torch.zeros((), device=device, dtype=torch.float32)
        if gate_rollout is not None:
            gate_losses = model.compute_gate_rl_loss(gate_rollout)
            gate_loss = gate_losses["loss"]
        else:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            gate_losses = {
                "loss": zero,
                "policy_loss": zero,
                "value_loss": zero,
                "entropy_bonus": zero,
                "explore_probability": zero,
            }
            gate_loss = zero

        _assert_finite_loss("perception_loss", perception_loss, epoch)
        _assert_finite_loss("exploration_loss", exploration_loss, epoch)
        _assert_finite_loss("policy_loss", policy_loss, epoch)
        _assert_finite_loss("gate_loss", gate_loss, epoch)

        metrics = dict(batch["metrics"])
        metrics["exploration_loss"] = float(exploration_loss.detach().item())

        perception_optimizer.zero_grad(set_to_none=True)
        exploration_optimizer.zero_grad(set_to_none=True)
        policy_optimizer.zero_grad(set_to_none=True)
        if gate_optimizer is not None:
            gate_optimizer.zero_grad(set_to_none=True)

        perception_grads = _autograd_grad_or_none(
            perception_loss,
            perception_params,
            retain_graph=True,
        )
        exploration_grads = _autograd_grad_or_none(
            exploration_loss,
            exploration_params,
            retain_graph=False,
        )
        policy_grads = _autograd_grad_or_none(
            policy_loss,
            policy_params,
            retain_graph=False,
        )
        gate_grads = _autograd_grad_or_none(
            gate_loss,
            gate_params,
            retain_graph=False,
        )

        _assign_gradients(perception_params, perception_grads)
        _assign_gradients(exploration_params, exploration_grads)
        _assign_gradients(policy_params, policy_grads)
        _assign_gradients(gate_params, gate_grads)

        _average_gradients(perception_params, world_size)
        _average_gradients(exploration_params, world_size)
        _average_gradients(policy_params, world_size)
        _average_gradients(gate_params, world_size)

        _assert_finite_gradients("perception", perception_params, epoch)
        _assert_finite_gradients("exploration", exploration_params, epoch)
        _assert_finite_gradients("policy", policy_params, epoch)
        _assert_finite_gradients("gate", gate_params, epoch)

        _clip_gradients(perception_params, grad_clip_norm)
        _clip_gradients(exploration_params, grad_clip_norm)
        _clip_gradients(policy_params, grad_clip_norm)
        _clip_gradients(gate_params, grad_clip_norm)

        perception_optimizer.step()
        exploration_optimizer.step()
        policy_optimizer.step()
        if gate_optimizer is not None:
            gate_optimizer.step()

        reduced_metrics = {
            "perception_total": _reduce_scalar(metrics["perception_total"], device, world_size),
            "perception_recon": _reduce_scalar(metrics["perception_recon"], device, world_size),
            "perception_kl": _reduce_scalar(metrics["perception_kl"], device, world_size),
            "uncertainty": _reduce_scalar(metrics["uncertainty"], device, world_size),
            "exploration_loss": _reduce_scalar(metrics["exploration_loss"], device, world_size),
            "exploration_information_gain": _reduce_scalar(
                metrics["exploration_information_gain"], device, world_size
            ),
            "exploration_sample_count": _reduce_scalar(
                metrics["exploration_sample_count"], device, world_size
            ),
            "gate_sample_count": _reduce_scalar(
                metrics["gate_sample_count"], device, world_size
            ),
            "policy_loss": _reduce_scalar(policy_losses["policy_loss"].detach().item(), device, world_size),
            "value_loss": _reduce_scalar(policy_losses["value_loss"].detach().item(), device, world_size),
            "gate_policy_loss": _reduce_scalar(gate_losses["policy_loss"].detach().item(), device, world_size),
            "gate_value_loss": _reduce_scalar(gate_losses["value_loss"].detach().item(), device, world_size),
            "gate_loss": _reduce_scalar(gate_losses["loss"].detach().item(), device, world_size),
            "entropy_bonus": _reduce_scalar(policy_losses["entropy_bonus"].detach().item(), device, world_size),
            "gate_entropy_bonus": _reduce_scalar(gate_losses["entropy_bonus"].detach().item(), device, world_size),
            "mean_explore_probability": _reduce_scalar(
                metrics["mean_explore_probability"], device, world_size
            ),
            "mean_episode_return": _reduce_scalar(metrics["mean_episode_return"], device, world_size),
            "mean_policy_return": _reduce_scalar(metrics["mean_policy_return"], device, world_size),
            "mean_episode_length": _reduce_scalar(metrics["mean_episode_length"], device, world_size),
            "explore_fraction": _reduce_scalar(metrics["explore_fraction"], device, world_size),
            "success_rate": _reduce_scalar(metrics["success_rate"], device, world_size),
        }
        reduced_threshold = _reduce_scalar(
            model.uncertainty_threshold.detach().item(),
            device,
            world_size,
        )
        if should_write_metrics_csv:
            _append_metrics_csv_row(
                metrics_csv_path=metrics_csv_path,
                fieldnames=csv_fieldnames,
                row={
                    "epoch": float(epoch + 1),
                    **reduced_metrics,
                    "threshold": reduced_threshold,
                    "flower_c": float(active_env.flower.c),
                },
            )

        should_log = (epoch % print_every == 0) and (not parallel or rank == 0)
        if should_log:
            mode_label = "Global" if parallel else "Loss"
            print(
                f"Epoch {epoch + 1}: {mode_label} = {reduced_metrics['perception_total']:.4f}, "
                f"Recon = {reduced_metrics['perception_recon']:.4f}, "
                f"KL = {reduced_metrics['perception_kl']:.4f}, "
                f"ExploreLoss = {reduced_metrics['exploration_loss']:.4f}, "
                f"InfoGain = {reduced_metrics['exploration_information_gain']:.4f}, "
                f"ExploreSamples = {reduced_metrics['exploration_sample_count']:.1f}, "
                f"Policy = {reduced_metrics['policy_loss']:.4f}, "
                f"Value = {reduced_metrics['value_loss']:.4f}, "
                f"Gate = {reduced_metrics['gate_loss']:.4f}, "
                f"GatePolicy = {reduced_metrics['gate_policy_loss']:.4f}, "
                f"GateValue = {reduced_metrics['gate_value_loss']:.4f}, "
                f"Entropy = {reduced_metrics['entropy_bonus']:.4f}, "
                f"GateEntropy = {reduced_metrics['gate_entropy_bonus']:.4f}, "
                f"Return = {reduced_metrics['mean_episode_return']:.4f}, "
                f"PolicyReturn = {reduced_metrics['mean_policy_return']:.4f}, "
                f"ExploreFrac = {reduced_metrics['explore_fraction']:.3f}, "
                f"ExploreProb = {reduced_metrics['mean_explore_probability']:.3f}, "
                f"Success = {reduced_metrics['success_rate']:.3f}, "
                f"Uncertainty = {reduced_metrics['uncertainty']:.3f}, "
                f"Threshold = {reduced_threshold:.3f}, "
                f"c = {active_env.flower.c}"
            )

    training_state = {
        "completed_epochs": int(num_epochs),
        "optimizer_state_dicts": {
            "perception": perception_optimizer.state_dict(),
            "exploration": exploration_optimizer.state_dict(),
            "policy": policy_optimizer.state_dict(),
            "gate": None if gate_optimizer is None else gate_optimizer.state_dict(),
        },
    }
    if return_training_state:
        return model, training_state
    return model
