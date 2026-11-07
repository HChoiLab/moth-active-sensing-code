import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from experiment_registry import load_pretrained_perception_components
from nectary_allocentric_env import AllocentricNectarSearchEnv
from nectary_env import FlowerParams
from training_uncertainty_gated import (
    _allocentric_action_delta_bounds,
    _allocentric_vision_shape,
    train_uncertainty_gated_allocentric_model,
)
from uncertainty_gated_model import UncertaintyGatedAllocentricNectarySearchModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed training entrypoint for the uncertainty-gated allocentric nectary search model."
    )
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=None, help="Defaults to env.max_steps.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument(
        "--visual-likelihood",
        type=str,
        default="gaussian",
        choices=["gaussian", "continuous_bernoulli"],
        help="Likelihood model used for allocentric visual observations.",
    )
    parser.add_argument(
        "--fixed-decoder-variance",
        action="store_true",
        help="Use fixed unit variance for Gaussian decoder likelihoods instead of learning log variance.",
    )
    parser.add_argument(
        "--posterior-fusion",
        type=str,
        default="early",
        choices=[
            "early",
            "early_fusion",
            "poe",
            "product_of_experts",
            "moe",
            "mixture_of_experts",
            "uniform_moe",
            "weighted_moe",
        ],
        help="Posterior fusion for allocentric observations.",
    )
    parser.add_argument(
        "--no-moment-matching",
        dest="moment_matching",
        action="store_false",
        default=True,
        help="Use stratified MoE mixture samples instead of moment matching for MoE posterior training.",
    )
    parser.add_argument(
        "--uncertainty-metric",
        type=str,
        default="aggregate",
        choices=[
            "aggregate",
            "average_unimodal",
            "weighted_unimodal",
            "mech",
            "vis",
        ],
        help="Posterior uncertainty metric used by the gate and realized information gain.",
    )
    parser.add_argument(
        "--rnn-type",
        type=str,
        default="rnn",
        choices=["rnn", "lstm"],
        help="Recurrent cell used by the allocentric perception model.",
    )
    parser.add_argument(
        "--modality-rnn-mode",
        type=str,
        default="shared",
        choices=["shared", "separate"],
        help="Use shared fused history or separate modality histories for MoE/PoE posteriors.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--perception-lr", type=float, default=1e-3)
    parser.add_argument("--exploration-lr", type=float, default=1e-3)
    parser.add_argument("--exploitation-lr", type=float, default=3e-4)
    parser.add_argument("--gate-lr", type=float, default=None)
    parser.add_argument(
        "--head-lr-scheduler",
        action="store_true",
        help="Enable linear annealing for the exploration and exploitation head learning rates.",
    )
    parser.add_argument(
        "--head-lr-scheduler-start-epoch",
        type=int,
        default=0,
        help="Number of epochs to run before head learning-rate annealing begins.",
    )
    parser.add_argument("--rl-algorithm", type=str, default="ppo", choices=["reinforce", "a2c", "ppo"])
    parser.add_argument(
        "--exploration-objective",
        type=str,
        default="expected_information_gain",
        choices=["expected_information_gain", "realized_information_gain"],
    )
    parser.add_argument(
        "--gate-mode",
        type=str,
        default="fixed_threshold",
        choices=["fixed_threshold", "rl_head"],
    )
    parser.add_argument("--fixed-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--action-selection-strategy",
        type=str,
        default="uncertainty_gate",
        choices=[
            "uncertainty_gate",
            "epsilon_greedy",
            "fixed_switching",
            "random_switching",
            "non_lexicographic",
        ],
        help="Action selection strategy used during rollout collection.",
    )
    parser.add_argument(
        "--epsilon-greedy-probability",
        type=float,
        default=0.1,
        help="Probability of random exploration for epsilon-greedy selection.",
    )
    parser.add_argument(
        "--fixed-switch-timestep",
        type=int,
        default=None,
        help=(
            "For fixed switching, explore for timesteps [0, x) and exploit "
            "from timestep x onward."
        ),
    )
    parser.add_argument(
        "--random-switch-probability",
        type=float,
        default=0.1,
        help=(
            "Probability of selecting the learned exploration policy at each "
            "timestep for random switching."
        ),
    )
    parser.add_argument(
        "--information-gain-reward-coef",
        type=float,
        default=1.0,
        help="Scale on realized information gain in non-lexicographic rewards.",
    )
    parser.add_argument(
        "--gate-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden sizes for the learned gate actor-critic head.",
    )
    parser.add_argument("--gate-value-coef", type=float, default=None)
    parser.add_argument("--gate-entropy-coef", type=float, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-press-depth", type=float, default=4.0)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument(
        "--metrics-csv-path",
        type=str,
        default=None,
        help="Optional CSV path for per-epoch training metrics. Written by rank 0 only.",
    )
    parser.add_argument(
        "--append-metrics-csv",
        action="store_true",
        help="Append to the metrics CSV instead of overwriting it at the start of training.",
    )
    parser.add_argument("--detach-hidden-state", action="store_true")
    parser.add_argument("--deterministic-exploitation", action="store_true")
    parser.add_argument(
        "--exploration-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Force exploration for the first N epochs before handing control "
            "back to the learned gate."
        ),
    )
    parser.add_argument(
        "--gate-hysteresis",
        type=float,
        default=0.0,
        help="Half-width around the fixed gate threshold used to prevent mode chatter.",
    )
    parser.add_argument(
        "--gate-uncertainty-ema-alpha",
        type=float,
        default=None,
        help="Optional EMA alpha in [0, 1) for smoothing fixed-gate uncertainty.",
    )
    parser.add_argument(
        "--c-vals",
        type=float,
        nargs="+",
        default=[-1.0, -2.0, -100.0, -1000.0],
        help="Flower c values. In distributed mode, rank i uses c_vals[i].",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--pretrained-perception-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint from a standard allocentric nectary-search model. "
            "Matching perception-component weights are loaded into the uncertainty-gated model "
            "with strict=False semantics before training."
        ),
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where rank 0 writes the final checkpoint. Defaults under NectarySearch/checkpoints.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="torch.distributed backend. Use gloo if CUDA is unavailable.",
    )
    return parser.parse_args()


def setup_distributed(backend):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_distributed = world_size > 1
    if use_distributed:
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device, use_distributed


def cleanup_distributed(use_distributed):
    if use_distributed and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed, rank):
    worker_seed = seed + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


def build_envs(args):
    return [
        AllocentricNectarSearchEnv(
            flower=FlowerParams(c=c),
            dt=args.dt,
            max_steps=args.max_steps,
            max_press_depth=args.max_press_depth,
        )
        for c in args.c_vals
    ]


def build_model(args, env):
    action_low = env.action_space.low.copy()
    action_low[-1] = max(float(action_low[-1]), float(args.min_depth))
    action_delta_low, action_delta_high = _allocentric_action_delta_bounds(env)
    return UncertaintyGatedAllocentricNectarySearchModel(
        mech_obs_dim=env.observation_space["mech"].shape[0],
        vis_obs_shape=_allocentric_vision_shape(env),
        action_dim=env.action_space.shape[0],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        visual_likelihood=args.visual_likelihood,
        learn_decoder_variance=not args.fixed_decoder_variance,
        posterior_fusion=args.posterior_fusion,
        moment_matching=args.moment_matching,
        uncertainty_metric=args.uncertainty_metric,
        rnn_type=args.rnn_type,
        modality_rnn_mode=args.modality_rnn_mode,
        action_low=action_low,
        action_high=env.action_space.high,
        action_delta_low=action_delta_low,
        action_delta_high=action_delta_high,
        rl_algorithm=args.rl_algorithm,
        exploration_objective=args.exploration_objective,
        gate_mode=args.gate_mode,
        fixed_gate_threshold=args.fixed_gate_threshold,
        action_selection_strategy=args.action_selection_strategy,
        epsilon_greedy_probability=args.epsilon_greedy_probability,
        fixed_switch_timestep=args.fixed_switch_timestep,
        random_switch_probability=args.random_switch_probability,
        information_gain_reward_coef=args.information_gain_reward_coef,
        gate_hidden_dims=args.gate_hidden_dims,
        gate_value_coef=args.gate_value_coef,
        gate_entropy_coef=args.gate_entropy_coef,
    )


def default_save_path():
    return Path(__file__).resolve().parent / "checkpoints" / "nectary_uncertainty_gated_allocentric.pt"


def save_checkpoint(model, args, envs, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": type(model).__name__,
        "c_vals": [env.flower.c for env in envs],
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "perception_learning_rate": args.perception_lr,
        "exploration_learning_rate": args.exploration_lr,
        "exploitation_learning_rate": args.exploitation_lr,
        "gate_learning_rate": args.gate_lr,
        "head_lr_scheduler": args.head_lr_scheduler,
        "head_lr_scheduler_start_epoch": args.head_lr_scheduler_start_epoch,
        "min_depth": args.min_depth,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "beta": args.beta,
        "visual_likelihood": args.visual_likelihood,
        "learn_decoder_variance": not args.fixed_decoder_variance,
        "posterior_fusion": args.posterior_fusion,
        "moment_matching": args.moment_matching,
        "uncertainty_metric": args.uncertainty_metric,
        "rnn_type": args.rnn_type,
        "modality_rnn_mode": args.modality_rnn_mode,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "rl_algorithm": args.rl_algorithm,
        "exploration_objective": args.exploration_objective,
        "gate_mode": args.gate_mode,
        "fixed_gate_threshold": args.fixed_gate_threshold,
        "action_selection_strategy": args.action_selection_strategy,
        "epsilon_greedy_probability": args.epsilon_greedy_probability,
        "fixed_switch_timestep": args.fixed_switch_timestep,
        "random_switch_probability": args.random_switch_probability,
        "information_gain_reward_coef": args.information_gain_reward_coef,
        "gate_hidden_dims": list(args.gate_hidden_dims),
        "gate_value_coef": args.gate_value_coef,
        "gate_entropy_coef": args.gate_entropy_coef,
        "dt": args.dt,
        "max_steps": args.max_steps,
        "max_press_depth": args.max_press_depth,
        "metrics_csv_path": args.metrics_csv_path,
        "append_metrics_csv": args.append_metrics_csv,
        "detach_hidden_state": args.detach_hidden_state,
        "deterministic_exploitation": args.deterministic_exploitation,
        "exploration_warmup_epochs": args.exploration_warmup_epochs,
        "gate_hysteresis": args.gate_hysteresis,
        "gate_uncertainty_ema_alpha": args.gate_uncertainty_ema_alpha,
        "seed": args.seed,
        "pretrained_perception_checkpoint": args.pretrained_perception_checkpoint,
    }
    torch.save(checkpoint, save_path)


def main():
    args = parse_args()

    if not torch.cuda.is_available() and args.backend == "nccl":
        args.backend = "gloo"

    rank, world_size, local_rank, device, use_distributed = setup_distributed(args.backend)
    set_seed(args.seed, rank)

    envs = build_envs(args)
    if use_distributed and world_size > len(envs):
        raise ValueError(
            f"WORLD_SIZE={world_size} exceeds the number of environments ({len(envs)})."
        )

    if use_distributed and rank == 0 and world_size != len(envs):
        print(
            f"Warning: WORLD_SIZE={world_size} but {len(envs)} environments were configured. "
            "Only the first WORLD_SIZE environments will be used."
        )

    local_env = envs[rank % len(envs)]
    seq_len = args.seq_len if args.seq_len is not None else local_env.max_steps
    model = build_model(args, local_env)
    pretrained_init_info = None
    if args.pretrained_perception_checkpoint is not None:
        pretrained_init_info = load_pretrained_perception_components(
            model,
            args.pretrained_perception_checkpoint,
            map_location=device,
        )

    if rank == 0:
        assigned = [envs[i].flower.c for i in range(min(world_size, len(envs)))]
        print(
            f"Launching uncertainty-gated training on {world_size} process(es) in allocentric mode."
        )
        print(f"Active environment c values by rank: {assigned}")
        if pretrained_init_info is not None:
            print(
                "Loaded pretrained perception weights from "
                f"{pretrained_init_info['checkpoint_path']} "
                f"({pretrained_init_info['num_loaded_keys']} tensors)."
            )

    trained_model = train_uncertainty_gated_allocentric_model(
        envs=envs,
        model=model,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seq_len=seq_len,
        perception_learning_rate=args.perception_lr,
        exploration_learning_rate=args.exploration_lr,
        exploitation_learning_rate=args.exploitation_lr,
        gate_learning_rate=args.gate_lr,
        head_lr_scheduler=args.head_lr_scheduler,
        head_lr_scheduler_start_epoch=args.head_lr_scheduler_start_epoch,
        print_every=args.print_every,
        min_depth=args.min_depth,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        exploration_warmup_epochs=args.exploration_warmup_epochs,
        deterministic_exploitation=args.deterministic_exploitation,
        detach_hidden_state=args.detach_hidden_state,
        grad_clip_norm=args.grad_clip_norm,
        metrics_csv_path=args.metrics_csv_path,
        append_metrics_csv=args.append_metrics_csv,
        exploration_objective=args.exploration_objective,
        visual_likelihood=args.visual_likelihood,
        learn_decoder_variance=not args.fixed_decoder_variance,
        posterior_fusion=args.posterior_fusion,
        moment_matching=args.moment_matching,
        uncertainty_metric=args.uncertainty_metric,
        rnn_type=args.rnn_type,
        modality_rnn_mode=args.modality_rnn_mode,
        action_selection_strategy=args.action_selection_strategy,
        epsilon_greedy_probability=args.epsilon_greedy_probability,
        fixed_switch_timestep=args.fixed_switch_timestep,
        random_switch_probability=args.random_switch_probability,
        information_gain_reward_coef=args.information_gain_reward_coef,
        parallel=use_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        gate_hysteresis=args.gate_hysteresis,
        gate_uncertainty_ema_alpha=args.gate_uncertainty_ema_alpha,
    )

    if use_distributed:
        dist.barrier()

    if rank == 0:
        save_path = args.save_path or default_save_path()
        save_checkpoint(trained_model, args, envs, save_path)
        print(f"Saved checkpoint to {save_path}")

    cleanup_distributed(use_distributed)


if __name__ == "__main__":
    main()
