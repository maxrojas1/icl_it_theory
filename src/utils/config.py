import argparse
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf


def get_experiment_config(args: argparse.Namespace
) -> Any:
    """
    Constructs the experiment configuration by combining CLI arguments
    with model specifications from configs/models.yaml.

    Args:
        args: Parsed command-line arguments.
    """
    models_cfg = OmegaConf.load("configs/models.yaml")

    if args.models is not None:
        target_models = {x.lower().rstrip("b") for x in args.models}
        models_filtered = {
            k: m for k, m in models_cfg.models.items()
            if k in args.models or str(m.get("params", "")).lower().rstrip("b") in target_models
        }
        if models_filtered:
            models_cfg.models = models_filtered

    args_dict = vars(args).copy()
    for k in ["run", "dgp", "xlim", "ylim", "models"]:
        args_dict.pop(k, None)

    cfg = OmegaConf.create(args_dict)
    cfg = OmegaConf.merge(cfg, models_cfg)
    return cfg


def get_plot_config(*args
) -> Any:
    """
    Loads the saved config.yaml associated with an existing experiment run.

    Args:
        *args: Run identifier string (e.g., '2026_09_23_20_10_18') or plot type and run ID.
    """
    if len(args) == 1:
        run = args[0]
    elif len(args) >= 2:
        run = args[1]
    else:
        raise ValueError("Run timestamp must be provided to get_plot_config")

    clean_run = run[4:] if run.startswith("run_") else run
    run_path = Path("results/runs") / f"run_{clean_run}"
    if not run_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml for run_{clean_run} in results/runs/")

    cfg = OmegaConf.load(run_path / "config.yaml")
    return cfg


def save_config(cfg,
                run_path: Path
) -> None:
    """
    Strips runtime transient flags and serializes resolved config to run directory.

    Args:
        cfg: Resolved OmegaConf configuration.
        run_path: Directory where config.yaml is written.
    """
    cfg_to_save = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for k in ["--exp", "icl_training", "experiment", "only_params", "orientation", "format", "target_entropy"]:
        if k in cfg_to_save:
            del cfg_to_save[k]
    OmegaConf.save(cfg_to_save, run_path / "config.yaml")
