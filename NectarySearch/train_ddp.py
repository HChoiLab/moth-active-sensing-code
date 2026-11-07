import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from model import AllocentricNectarySearchModel, NectarySearchModel
from nectary_allocentric_env import AllocentricNectarSearchEnv, CanvasParams
from nectary_env import FlowerParams, NectarSearchEnv
from training import _allocentric_vision_shape, train_nectary_search_model

DEFAULT_FLOWER = FlowerParams(
    r0= 2.0,      # hole radius
    R= 40.0,      # corolla radius
    L= 25.0,      # flower length scale
    c= -1.0,      # curvature: -1 funnel, -2 near-funnel, -3 near-flat, np.inf flat
)

DEFAULT_CANVAS = CanvasParams(
    width= 200.0,
    height= 200.0,
    background_value= 0.0,
    flower_value= 1.0,
    hole_value= 0.0,
    pattern_type= "none",  # one of: none, cross, stripes, circular
    pattern_value= 0.5,
    cross_width= 5.0,
    stripe_width= 5.0,
    stripe_offset=10.0,
    stripe_orientation= "vertical",  # one of: vertical, horizontal
    circular_radius= 5.0
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed training entrypoint for the nectary search task."
    )
    parser.add_argument("--allocentric", default=True, action="store_true", help="Train the allocentric model.")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=None, help="Defaults to env.max_steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--mech-embedding-dim", type=int, default=16)
    parser.add_argument("--vis-embedding-dim", type=int, default=16)
    parser.add_argument("--fusion-dim", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.003)
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
        "--rnn-type",
        type=str,
        default="rnn",
        choices=["rnn", "lstm"],
        help="Recurrent cell used by the allocentric model.",
    )
    parser.add_argument(
        "--modality-rnn-mode",
        type=str,
        default="shared",
        choices=["shared", "separate"],
        help="Use shared fused history or separate modality histories for MoE/PoE posteriors.",
    )
    parser.add_argument("--anneal-beta", action="store_true")
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--annealing-epochs", type=int, default=None) 
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-press-depth", type=float, default=5.0)
    parser.add_argument("--visual-field-size", type=int, default=15)
    parser.add_argument("--flower-pattern", type=str, default="none")
    parser.add_argument(
        "--randomize-visual-pattern-each-epoch",
        action="store_true",
        help="Randomly choose the allocentric flower pattern at the start of each training epoch.",
    )
    parser.add_argument(
        "--randomize-visual-colors-each-epoch",
        action="store_true",
        help=(
            "Randomly switch between bright flower and dark flower visual values "
            "at the start of each training epoch."
        ),
    )
    parser.add_argument(
        "--visual-pattern-choices",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of allocentric pattern types to sample from each epoch.",
    )
    parser.add_argument(
        "--c-vals",
        type=float,
        nargs="+",
        default=[-1.0, -2.0, -100.0, -1000.0],
        help="Flower c values. In distributed mode, rank i uses c_vals[i].",
    )
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where rank 0 writes the final checkpoint. Defaults under NectarySearch/checkpoints.",
    )
    parser.add_argument(
        "--save-suffix",
        type=str,
        default="",
        help="Suffix to add to the model name when saving."
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
    envs = []
    for c in args.c_vals:
        flower = FlowerParams(
            r0=DEFAULT_FLOWER.r0,
            R=DEFAULT_FLOWER.R,
            L=DEFAULT_FLOWER.L,
            c=c,
        )
        canvas = CanvasParams(
            width=DEFAULT_CANVAS.width,
            height=DEFAULT_CANVAS.height,
            background_value=DEFAULT_CANVAS.background_value,
            flower_value=DEFAULT_CANVAS.flower_value,
            hole_value=DEFAULT_CANVAS.hole_value,
            pattern_type=args.flower_pattern,
            pattern_value=DEFAULT_CANVAS.pattern_value,
            cross_width=DEFAULT_CANVAS.cross_width,
            stripe_width=DEFAULT_CANVAS.stripe_width,
            stripe_offset=DEFAULT_CANVAS.stripe_offset,
            stripe_orientation=DEFAULT_CANVAS.stripe_orientation,
            circular_radius=DEFAULT_CANVAS.circular_radius,
        )
        if not args.allocentric:
            envs.append(
                NectarSearchEnv(
                    flower=flower,
                    dt=args.dt,
                    max_steps=args.max_steps,
                    max_press_depth=args.max_press_depth
                )
            )
        else:
            envs.append(
                AllocentricNectarSearchEnv(
                    flower=flower,
                    canvas=canvas,
                    dt=args.dt,
                    max_steps=args.max_steps,
                    max_press_depth=args.max_press_depth,
                    visual_field_size=args.visual_field_size
                )
            )
    return envs


def build_model(args, env):
    if args.allocentric:
        return AllocentricNectarySearchModel(
            mech_obs_dim=env.observation_space["mech"].shape[0],
            vis_obs_shape=_allocentric_vision_shape(env),
            action_dim=env.action_space.shape[0],
            mech_embedding_dim=args.mech_embedding_dim,
            vis_embedding_dim=args.vis_embedding_dim,
            fusion_dim=args.fusion_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            beta=args.beta,
            visual_likelihood=args.visual_likelihood,
            learn_decoder_variance=not args.fixed_decoder_variance,
            posterior_fusion=args.posterior_fusion,
            moment_matching=args.moment_matching,
            rnn_type=args.rnn_type,
            modality_rnn_mode=args.modality_rnn_mode,
        )
    return NectarySearchModel(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
    )


def default_save_path(args):
    mode = "allocentric" if args.allocentric else "egocentric"
    return Path(__file__).resolve().parent / "checkpoints" / f"nectary_{mode}_{args.seed}_{args.save_suffix}.pt"


def save_checkpoint(model, args, envs, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "allocentric": args.allocentric,
        "c_vals": [env.flower.c for env in envs],
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate,
        "min_depth": args.min_depth,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "mech_embedding_dim": args.mech_embedding_dim,
        "vis_embedding_dim": args.vis_embedding_dim,
        "fusion_dim": args.fusion_dim,
        "beta": args.beta,
        "visual_likelihood": args.visual_likelihood,
        "learn_decoder_variance": not args.fixed_decoder_variance,
        "posterior_fusion": args.posterior_fusion,
        "moment_matching": args.moment_matching,
        "rnn_type": args.rnn_type,
        "modality_rnn_mode": args.modality_rnn_mode,
        "dt": args.dt,
        "max_steps": args.max_steps,
        "max_press_depth": args.max_press_depth,
        "visual_field_size": args.visual_field_size,
        "flower_pattern": args.flower_pattern,
        "randomize_visual_pattern_each_epoch": args.randomize_visual_pattern_each_epoch,
        "randomize_visual_colors_each_epoch": args.randomize_visual_colors_each_epoch,
        "visual_pattern_choices": args.visual_pattern_choices,
        "seed": args.seed,
    }
    torch.save(checkpoint, save_path)


def main():
    args = parse_args()

    if not torch.cuda.is_available() and args.backend == "nccl":
        args.backend = "gloo"

    rank, world_size, local_rank, device, use_distributed = setup_distributed(args.backend)
    set_seed(args.seed, rank)

    envs = build_envs(args)
    if use_distributed and world_size < len(envs):
        raise ValueError(
            f"The number of environments ({len(envs)}) exceeds WORLD_SIZE={world_size}."
        )

    local_env = envs[rank % len(envs)]
    seq_len = args.seq_len if args.seq_len is not None else local_env.max_steps
    model = build_model(args, local_env)

    if rank == 0:
        assigned = [envs[i].flower.c for i in range(min(world_size, len(envs)))]
        print(
            f"Launching training on {world_size} process(es) with "
            f"{'allocentric' if args.allocentric else 'egocentric'} mode."
        )
        print(f"Active environment c values by rank: {assigned}")

    trained_model = train_nectary_search_model(
        envs=envs,
        model=model,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seq_len=seq_len,
        learning_rate=args.learning_rate,
        print_every=args.print_every,
        allocentric=args.allocentric,
        min_depth=args.min_depth,
        parallel=use_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        wrap_ddp=use_distributed,
        allocentric_trajectory_mode="smooth_modes",
        beta_anneal=args.anneal_beta,
        beta_start=args.beta_start,
        beta_end=args.beta,
        beta_anneal_epochs=args.annealing_epochs,
        randomize_visual_pattern_each_epoch=args.randomize_visual_pattern_each_epoch,
        randomize_visual_colors_each_epoch=args.randomize_visual_colors_each_epoch,
        visual_pattern_choices=args.visual_pattern_choices,
        visual_likelihood=args.visual_likelihood,
        learn_decoder_variance=not args.fixed_decoder_variance,
        posterior_fusion=args.posterior_fusion,
        moment_matching=args.moment_matching,
        rnn_type=args.rnn_type,
        modality_rnn_mode=args.modality_rnn_mode,
    )

    if use_distributed:
        dist.barrier()

    if rank == 0:
        save_path = args.save_path or default_save_path(args)
        save_checkpoint(trained_model, args, envs, save_path)
        print(f"Saved checkpoint to {save_path}")

    cleanup_distributed(use_distributed)


if __name__ == "__main__":
    main()
