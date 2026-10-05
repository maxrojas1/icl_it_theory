from pathlib import Path

from .styles import save_fig
from .power_law import (
    preprocessing_icl_training,
    preprocessing_entropy_icl_training,
    preprocessing_power_law_scaling,
    preprocessing_entropy_power_law_scaling,
)
from .single_run import (
    plot_icl_training,
    plot_entropy_icl_training,
    plot_power_law,
    plot_entropy_power_law,
)

__all__ = ["get_plots"]


def get_plots(cfg,
              plot_mode: str = "both"
) -> None:
    """
    Generates and saves all requested plot families for a single experiment run.

    Args:
        cfg: Experiment configuration object containing run path and plot flags.
        plot_mode: Which plots to generate — 'asymptotic', 'scaling', or 'both'.
    """
    clean_run = str(cfg.run)[4:] if str(cfg.run).startswith("run_") else str(cfg.run)
    print(f"Starting plots for run: {clean_run}")

    FIG_PATH = Path("results") / "figures" / f"run_{clean_run}"
    FIG_PATH.mkdir(parents=True, exist_ok=True)
    target_paths = [FIG_PATH]

    def save_by_entropy(figures_dict, base_paths, prefix):
        """
        Saves generated figures partitioned by conditional entropy level.

        Args:
            figures_dict: Mapping of entropy levels to Matplotlib figures.
            base_paths: Destination directory paths.
            prefix: Filename prefix describing the plot type.
        """
        for entropy_level, fig in figures_dict.items():
            val = round(float(entropy_level), 2)
            filename_base = f"run_{clean_run}_{prefix}_H_Y_X_{val}"
            for p in base_paths:
                subfolder = p / f"H_Y_X_{val}"
                subfolder.mkdir(parents=True, exist_ok=True)
                save_fig(fig, subfolder, filename_base)

    def save_by_model(fig, base_paths, config, suffix):
        """
        Saves generated figures partitioned by model architecture and scale.

        Args:
            fig: Matplotlib figure to save.
            base_paths: Destination directory paths.
            config: Model configuration dictionary.
            suffix: Filename suffix describing the plot type.
        """
        folder_name = f"{config['name']} ({config['params']})"
        filename_base = f"run_{clean_run}_{suffix}"
        for p in base_paths:
            subfolder = p / folder_name
            subfolder.mkdir(parents=True, exist_ok=True)
            save_fig(fig, subfolder, filename_base)

    only_params = getattr(cfg, 'only_params', False)
    xlim = getattr(cfg, 'xlim', None)
    ylim = getattr(cfg, 'ylim', None)

    if plot_mode in ("asymptotic", "both"):
        _, train_sizes, entropies, entropy_config_and_results = preprocessing_icl_training(cfg)
        _, train_sizes_ent, entropies_ent, model_config_and_results = preprocessing_entropy_icl_training(cfg)

        for metric in ["risk", "loss"]:
            figures = plot_icl_training(
                train_sizes,
                entropy_config_and_results,
                metric=metric,
                only_params=only_params,
                xlim=xlim,
                ylim=ylim,
            )
            save_by_entropy(figures, target_paths, f"icl_asymptotic_{metric}")

            for model_id, config_and_results in model_config_and_results.items():
                single_model_dict = {model_id: config_and_results}
                fig = plot_entropy_icl_training(
                    train_sizes_ent,
                    entropies_ent,
                    single_model_dict,
                    metric=metric,
                    xlim=xlim,
                    ylim=ylim,
                )
                save_by_model(fig, target_paths, config_and_results, f"entropy_icl_asymptotic_{metric}")

        print("   Plot: icl_asymptotic saved")

    if plot_mode in ("scaling", "both"):
        _, train_list, train_continuous, entropy_config_and_results = preprocessing_power_law_scaling(cfg)
        figures = plot_power_law(
            train_list,
            train_continuous,
            entropy_config_and_results,
            only_params=only_params,
            xlim=xlim,
            ylim=ylim,
        )
        save_by_entropy(figures, target_paths, "icl_scaling")
        print("   Plot: icl_scaling saved")

        _, train_list, train_continuous, entropies, model_config_and_results = preprocessing_entropy_power_law_scaling(cfg)
        for model_id, config_and_results in model_config_and_results.items():
            fig = plot_entropy_power_law(
                train_list,
                train_continuous,
                entropies,
                config_and_results,
                xlim=xlim,
                ylim=ylim,
            )
            save_by_model(fig, target_paths, config_and_results, "entropy_icl_scaling")
        print("   Plot: entropy_icl_scaling saved")

    print(f"\nAll plots saved in: {FIG_PATH}")

