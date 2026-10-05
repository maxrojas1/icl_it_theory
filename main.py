#!/usr/bin/env python3
import os

venv_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "bin"))
if os.path.exists(venv_bin) and venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

if "FLASHINFER_CUDA_ARCH_LIST" not in os.environ:
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "8.9 9.0"

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torch_inductor")
os.environ.setdefault("VLLM_CACHE_ROOT", "/tmp/vllm")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import Any
import argparse

from src.utils.config import get_experiment_config, get_plot_config
from src.evaluation.runner import run_icl_training
from src.analysis.plotting import get_plots
from src.analysis.comparative import (
    plot_and_save_icl_entropy_and_model_scaling,
    plot_and_save_icl_power_law_bound,
)


def parse_args() -> argparse.Namespace:
    """
    Parses CLI flags for evaluation, run plotting, and customization.
    """
    parser = argparse.ArgumentParser(description="In-Context Learning evaluation & analysis")
    parser.add_argument("--run", type=str, nargs="+", default=None, help="Run timestamp(s) to plot")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Models to evaluate (e.g. 0.5B 7B)")
    parser.add_argument("--max_train", type=int, default=100, help="Max in-context training examples")
    parser.add_argument("--num_train", type=int, default=16, help="Number of training sizes to evaluate")
    parser.add_argument("--num_val", type=int, default=64, help="Validation queries per context prompt")
    parser.add_argument("--num_contexts", "--num_context", type=int, default=256, dest="num_contexts", help="Context prompts per task and size")
    parser.add_argument("--num_entropy_levels", type=int, default=1, help="Number of entropy levels to sample")
    parser.add_argument("--entropies", type=float, nargs="+", default=None, help="Explicit list of target entropy levels (e.g. --entropies 0.69)")
    parser.add_argument("--dim", type=int, default=1, help="Input feature dimension")
    parser.add_argument("--seed_train", type=int, default=43, help="Random seed for training points")
    parser.add_argument("--seed_val", type=int, default=44, help="Random seed for validation points")
    parser.add_argument("--seed_entropy", type=int, default=45, help="Random seed for entropy task calibration")
    parser.add_argument("--seed_task", type=int, default=42, help="Random seed for tasks")
    parser.add_argument("--entropy_num_tasks", type=int, default=100_000, help="Number of simulated tasks for entropy calibration")
    parser.add_argument("--gpu_ptge", type=float, default=0.9, help="GPU memory fraction allocated to vLLM")
    parser.add_argument("--old_geom", action="store_true", default=False, help="Use integer geometric spacing for training sizes")
    parser.add_argument("--only_params", action=argparse.BooleanOptionalAction, default=True, help="Display continuous parameter colorbar instead of legend")
    parser.add_argument("--target_entropy", type=float, default=None, help="Specific target conditional entropy H(Y|X) to highlight")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, help="Set x-axis limits (e.g. --xlim 1 100)")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, help="Set y-axis limits (e.g. --ylim 0.001 1.0)")
    parser.add_argument("--dgp", action="store_true", help="Generate the 1D DGP sigmoid family theoretical figure")
    parser.add_argument("--plot", type=str, default="both", choices=["asymptotic", "scaling", "both"],
                        help="Which plots to generate: 'asymptotic', 'scaling', or 'both' (default)")

    args, unknown = parser.parse_known_args()

    # Support legacy "entropies=[...]" if passed in command line
    for arg in unknown:
        if arg.startswith("entropies="):
            val = arg.split("=", 1)[1].strip("[]")
            args.entropies = [float(x.strip()) for x in val.split(",") if x.strip()]

    return args


def apply_plot_args(cfg: Any,
                    args: argparse.Namespace
) -> Any:
    """
    Transfers command-line plotting flags onto the active configuration.

    Args:
        cfg: Target OmegaConf configuration.
        args: Parsed command-line namespace.
    """
    cfg.only_params = args.only_params
    cfg.entropy_num_tasks = args.entropy_num_tasks
    if args.xlim is not None:
        cfg.xlim = list(args.xlim)
    if args.ylim is not None:
        cfg.ylim = list(args.ylim)
    return cfg


def main() -> None:
    """
    Main CLI entrypoint to execute experiments, plot single runs, or generate comparative figures.
    """
    args = parse_args()

    if args.dgp:
        import numpy as np
        from pathlib import Path
        from src.analysis.theoretical import plot_DGP_1D_family
        save_dir = Path("results/figures")
        save_dir.mkdir(parents=True, exist_ok=True)
        x = np.linspace(-5, 5, 500)
        weights_small = np.linspace(-5, 5, 40)
        biases_small = np.linspace(-5, 5, 40)
        plot_DGP_1D_family(x, weights_small, biases_small, fixed_weight=1.5, fixed_bias=0.0, save_dir=save_dir)
        print(f"Saved dgp_1d_family.pdf in {save_dir}")
        return

    if args.run is not None:
        if len(args.run) == 1 and " " in args.run[0]:
            args.run = args.run[0].split()
        args.run = [r[4:] if r.startswith("run_") else r for r in args.run]

    if args.run is None:
        cfg_exp = get_experiment_config(args)
        apply_plot_args(cfg_exp, args)

        run_icl_training(cfg_exp)
        get_plots(cfg_exp, plot_mode=args.plot)

    elif len(args.run) == 1:
        cfg_plot = apply_plot_args(get_plot_config(args.run[0]), args)
        get_plots(cfg_plot, plot_mode=args.plot)

    elif len(args.run) == 2:
        cfg_plot_1 = apply_plot_args(get_plot_config(args.run[0]), args)
        cfg_plot_2 = apply_plot_args(get_plot_config(args.run[1]), args)

        if args.plot in ("asymptotic", "both"):
            plot_and_save_icl_entropy_and_model_scaling(cfg_plot_1, cfg_plot_2, target_entropy=args.target_entropy, xlim=args.xlim, ylim=args.ylim)
        if args.plot in ("scaling", "both"):
            plot_and_save_icl_power_law_bound(cfg_plot_1, cfg_plot_2, target_entropy=args.target_entropy, xlim=args.xlim, ylim=args.ylim)

    else:
        print("More than 2 runs not supported for combined plots.")


if __name__ == "__main__":
    main()