from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from .styles import (
    parse_params,
    format_param_tick,
    save_fig,
)
from .power_law import (
    preprocessing_icl_training,
    preprocessing_entropy_icl_training,
    preprocessing_entropy_power_law_scaling,
    preprocessing_power_law_scaling,
)
from src.utils.config import get_plot_config



def plot_entropy_and_icl_training(cfg_entropy,
                                  cfg_size,
                                  metric: str = "risk",
                                  target_entropy: float | None = None,
                                  xlim: tuple[float, float] | list[float] | None = None,
                                  ylim: tuple[float, float] | list[float] | None = None
) -> plt.Figure:
    """
    Plots two-panel figure contrasting task entropy modulation against model size scaling.

    Args:
        cfg_entropy: Configuration or run ID for the entropy modulation experiment.
        cfg_size: Configuration or run ID for the model size scaling experiment.
        metric: 'risk' or 'loss' to display.
        target_entropy: Specific entropy H(Y|X) level to isolate in the right panel.
        xlim: Optional (min, max) limits for x-axis.
        ylim: Optional (min, max) limits for y-axis.
    """
    if isinstance(cfg_entropy, str):
        r_id = cfg_entropy[4:] if cfg_entropy.startswith("run_") else cfg_entropy
        cfg_entropy = get_plot_config("icl_training", r_id)
    if isinstance(cfg_size, str):
        r_id = cfg_size[4:] if cfg_size.startswith("run_") else cfg_size
        cfg_size = get_plot_config("icl_training", r_id)

    models_1 = getattr(cfg_entropy, 'models', {})
    models_2 = getattr(cfg_size, 'models', {})
    if len(models_1) > 1 and len(models_2) == 1:
        cfg_entropy, cfg_size = cfg_size, cfg_entropy

    _, train_sizes_ent, entropies_ent, model_cfg_res_ent = preprocessing_entropy_icl_training(cfg_entropy)
    _, train_sizes_size, _, entropy_cfg_res_size = preprocessing_icl_training(cfg_size)

    line_alpha = 0.9
    grid_ls = '--'
    grid_lw = 0.5
    grid_alpha = 0.45
    ylabel = "Regret" if metric == "risk" else "Loss (nats)"

    cmap_ent = plt.get_cmap('cividis')
    float_entropies = [float(e) for e in entropies_ent]
    min_ent = min(float_entropies)
    max_ent = max(float_entropies)
    norm_ent = mcolors.Normalize(vmin=min_ent, vmax=max_ent)

    avail_keys = list(entropy_cfg_res_size.keys())
    if target_entropy is not None:
        size_ent_key = min(avail_keys, key=lambda k: abs(float(k) - float(target_entropy)))
    else:
        size_ent_key = avail_keys[0]
    config_and_results_size = entropy_cfg_res_size[size_ent_key]

    models_list = [v for k, v in config_and_results_size.items() if isinstance(v, dict)]
    models_sorted = sorted(models_list, key=lambda m: parse_params(m.get('params', '0')))
    colors_size = [m['color'] for m in models_sorted]
    labels_size = [format_param_tick(m.get('params', '')) for m in models_sorted]

    figsize = (6.75, 2.15)
    label_fs = 7.5
    tick_fs = 6.5
    cbar_tick_fs = 5.5
    line_w = 1.15
    letter_fs = 11.0
    wspace = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for model_id, config_and_results in model_cfg_res_ent.items():
        if metric == "risk":
            criterion = config_and_results["risks"].to_numpy()
        else:
            criterion = config_and_results["losses"].to_numpy()
        num_entropies = criterion.shape[0]
        for i in range(num_entropies):
            ent_val = float(entropies_ent[i])
            ent_color = cmap_ent(norm_ent(max(ent_val, min_ent))) if num_entropies > 1 else cmap_ent(0.5)
            if metric == "loss":
                ax1.axhline(y=ent_val, color=ent_color, linestyle="dotted", alpha=line_alpha, linewidth=1.0, zorder=2)
            ax1.plot(train_sizes_ent, criterion[i, :], color=ent_color, alpha=line_alpha, linewidth=line_w, zorder=3)

    ax1.set_ylabel(ylabel, fontsize=label_fs)
    ax1.set_xlabel(r"In-Context Examples ($n$)", fontsize=label_fs)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which='major', linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='both', which='both', labelsize=tick_fs)

    if metric == "loss":
        ax2.axhline(y=float(size_ent_key), color="black", linestyle="dotted", alpha=line_alpha, linewidth=1.0, zorder=2)

    for model_data in models_sorted:
        color = model_data['color']
        criterion = model_data['risks'] if metric == "risk" else model_data['losses']
        ax2.plot(train_sizes_size, criterion, color=color, alpha=line_alpha, linewidth=line_w, zorder=3)

    ax2.set_xlabel(r"In-Context Examples ($n$)", fontsize=label_fs)
    ax2.set_ylabel(ylabel, fontsize=label_fs)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, which='major', linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis='both', which='both', labelsize=tick_fs)

    all_train_sizes = np.concatenate([train_sizes_ent, train_sizes_size])
    for ax in [ax1, ax2]:
        if xlim is not None:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        else:
            ax.set_xlim(min(all_train_sizes) * 0.85, max(all_train_sizes) * 1.15)

    if ylim is not None:
        ax1.set_ylim(float(ylim[0]), float(ylim[1]))
        ax2.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)

    fig.subplots_adjust(left=0.15, right=0.88, top=0.86, bottom=0.22, wspace=wspace)
    fig.canvas.draw()

    p_left = ax1.get_position()
    cbar_w = 0.015
    cax_ent = fig.add_axes([p_left.x0 - 0.08, p_left.y0, cbar_w, p_left.height])
    sm_ent = cm.ScalarMappable(cmap=cmap_ent, norm=norm_ent)
    sm_ent.set_array([])
    cbar_ent = fig.colorbar(sm_ent, cax=cax_ent)
    cbar_ent.ax.yaxis.set_ticks_position('left')
    cbar_ent.ax.yaxis.set_label_position('left')
    cbar_ent.set_ticks([min_ent, max_ent])
    cbar_ent.set_ticklabels([f"{min_ent:.2f}", f"{max_ent:.2f}"])
    cbar_ent.set_label(r"$H(Y \mid X)$ (nats)", fontsize=label_fs)
    cbar_ent.ax.tick_params(labelsize=cbar_tick_fs)

    p_right = ax2.get_position()
    cax_size = fig.add_axes([p_right.x1 + 0.02, p_right.y0, cbar_w, p_right.height])
    cmap_size = mcolors.ListedColormap(colors_size)
    norm_size = mcolors.BoundaryNorm(range(len(colors_size) + 1), cmap_size.N)
    sm_size = cm.ScalarMappable(cmap=cmap_size, norm=norm_size)
    sm_size.set_array([])
    cbar_size = fig.colorbar(sm_size, cax=cax_size)
    cbar_size.set_ticks([i + 0.5 for i in range(len(colors_size))])
    cbar_size.set_ticklabels(labels_size)
    cbar_size.set_label('Parameters (B)', fontsize=label_fs)
    cbar_size.ax.tick_params(labelsize=cbar_tick_fs)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    min_tick_x0 = min(t.get_window_extent(r).x0 for t in cbar_ent.ax.get_yticklabels())
    inv_ent = cbar_ent.ax.transAxes.inverted()
    min_tick_x0_axes = inv_ent.transform((min_tick_x0, 0))[0]
    cbar_ent.ax.yaxis.set_label_coords(min_tick_x0_axes - 0.28, 0.5)

    max_tick_x1 = max(t.get_window_extent(r).x1 for t in cbar_size.ax.get_yticklabels())
    inv_size = cbar_size.ax.transAxes.inverted()
    max_tick_x1_axes = inv_size.transform((max_tick_x1, 0))[0]
    cbar_size.ax.yaxis.set_label_coords(max_tick_x1_axes + 0.28, 0.5)

    lbl_x0 = cbar_ent.ax.yaxis.label.get_window_extent(r).x0
    inv_ax1 = ax1.transAxes.inverted()
    a_x = inv_ax1.transform((lbl_x0, 0))[0]

    lbl_b_x0 = ax2.yaxis.label.get_window_extent(r).x0
    inv_ax2 = ax2.transAxes.inverted()
    b_x = inv_ax2.transform((lbl_b_x0, 0))[0]

    letter_y = 1.05
    ax1.text(a_x, letter_y, r"\textbf{(a)}", transform=ax1.transAxes, fontsize=letter_fs, fontweight='bold', va='bottom', ha='left', zorder=10)
    ax2.text(b_x, letter_y, r"\textbf{(b)}", transform=ax2.transAxes, fontsize=letter_fs, fontweight='bold', va='bottom', ha='left', zorder=10)

    return fig


def plot_and_save_entropy_and_icl_training(cfg_entropy,
                                          cfg_size,
                                          metric: str = "risk",
                                          formats: list[str] | None = None,
                                          target_entropy: float | None = None,
                                          xlim: tuple[float, float] | list[float] | None = None,
                                          ylim: tuple[float, float] | list[float] | None = None
) -> plt.Figure | None:
    """
    Generates and saves the two-panel entropy vs. model size figures to PDF.

    Args:
        cfg_entropy: Configuration or run ID for the entropy experiment.
        cfg_size: Configuration or run ID for the model size experiment.
        metric: 'risk' or 'loss'.
        formats: Kept for backwards compatibility (exports PDF).
        target_entropy: Specific entropy level to match (or None for all).
        xlim: Optional (min, max) limits for x-axis.
        ylim: Optional (min, max) limits for y-axis.
    """
    if isinstance(cfg_entropy, str):
        r_id = cfg_entropy[4:] if cfg_entropy.startswith("run_") else cfg_entropy
        cfg_entropy = get_plot_config("icl_training", r_id)
    if isinstance(cfg_size, str):
        r_id = cfg_size[4:] if cfg_size.startswith("run_") else cfg_size
        cfg_size = get_plot_config("icl_training", r_id)

    xlim = xlim or getattr(cfg_entropy, 'xlim', None) or getattr(cfg_size, 'xlim', None)
    ylim = ylim or getattr(cfg_entropy, 'ylim', None) or getattr(cfg_size, 'ylim', None)

    models_1 = getattr(cfg_entropy, 'models', {})
    models_2 = getattr(cfg_size, 'models', {})
    if len(models_1) > 1 and len(models_2) == 1:
        cfg_entropy, cfg_size = cfg_size, cfg_entropy

    FIG_PATH = Path("results") / "figures"
    FIG_PATH.mkdir(parents=True, exist_ok=True)

    _, _, _, entropy_cfg_res_size = preprocessing_icl_training(cfg_size)
    available_entropies = list(entropy_cfg_res_size.keys())

    if target_entropy is not None:
        selected_entropies = [min(available_entropies, key=lambda k: abs(float(k) - float(target_entropy)))]
    else:
        selected_entropies = available_entropies

    last_fig = None
    for ent_val in selected_entropies:
        fig = plot_entropy_and_icl_training(
            cfg_entropy,
            cfg_size,
            metric=metric,
            target_entropy=ent_val,
            xlim=xlim,
            ylim=ylim
        )
        last_fig = fig

        metric_suffix = f"_{metric}" if metric != "risk" else ""
        filename = f"run_{cfg_entropy.run}__{cfg_size.run}_H_Y_X_{ent_val}_icl_asymptotic{metric_suffix}"
        save_fig(fig, FIG_PATH, filename)
        print(f"Saved icl_asymptotic (H={ent_val}):")
        print(f"  {FIG_PATH / filename}")

    return last_fig


def plot_icl_power_law_bound(cfg_entropy,
                              cfg_size,
                              target_entropy: float | None = None,
                              xlim: tuple[float, float] | list[float] | None = None,
                              ylim: tuple[float, float] | list[float] | None = None
) -> plt.Figure:
    """
    Plots a 4-panel figure contrasting power-law curves and instantaneous scaling rates.

    Args:
        cfg_entropy: Configuration or run ID for the multi-entropy experiment.
        cfg_size: Configuration or run ID for the multi-model experiment.
        target_entropy: Target entropy level to match for the model scaling panel.
        xlim: Optional (min, max) limits for x-axis.
        ylim: Optional (min, max) limits for y-axis.
    """
    models_1 = getattr(cfg_entropy, 'models', {})
    models_2 = getattr(cfg_size, 'models', {})
    if len(models_1) > 1 and len(models_2) == 1:
        cfg_entropy, cfg_size = cfg_size, cfg_entropy

    _, train_list_ent, train_cont_ent, entropies_ent, model_cfg_res_ent = preprocessing_entropy_power_law_scaling(cfg_entropy)
    _, train_list_size, train_cont_size, entropy_cfg_res_size = preprocessing_power_law_scaling(cfg_size)

    figsize = (6.75, 3.85)
    label_fs = 7.5
    tick_fs = 6.5
    cbar_tick_fs = 5.5
    line_w = 1.25
    scatter_s = 2.0
    letter_fs = 11.0
    hspace = 0.10
    wspace = 0.32

    scatter_alpha = 0.08
    fill_alpha = 0.15
    grid_lw = 0.5
    grid_ls = '--'
    grid_alpha = 0.45

    if xlim is not None:
        min_examples = float(xlim[0])
        max_examples = float(xlim[1])
    else:
        min_examples = 1.0
        max_examples = float(max(max(train_list_ent), max(train_list_size)))

    mask_list_ent = (np.asarray(train_list_ent) >= min_examples) & (np.asarray(train_list_ent) <= max_examples)
    mask_cont_ent = (np.asarray(train_cont_ent) >= min_examples) & (np.asarray(train_cont_ent) <= max_examples)
    train_list_ent_plot = np.asarray(train_list_ent)[mask_list_ent]
    train_cont_ent_plot = np.asarray(train_cont_ent)[mask_cont_ent]

    mask_list_size = (np.asarray(train_list_size) >= min_examples) & (np.asarray(train_list_size) <= max_examples)
    mask_cont_size = (np.asarray(train_cont_size) >= min_examples) & (np.asarray(train_cont_size) <= max_examples)
    train_list_size_plot = np.asarray(train_list_size)[mask_list_size]
    train_cont_size_plot = np.asarray(train_cont_size)[mask_cont_size]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="col", figsize=figsize)

    model_id_ent = list(model_cfg_res_ent.keys())[0]
    ent_results = model_cfg_res_ent[model_id_ent]["entropies_results"]

    cmap_ent = plt.get_cmap("cividis")
    float_entropies = [float(e) for e in entropies_ent]
    min_ent = min(float_entropies)
    max_ent = max(float_entropies)
    norm_ent = mcolors.Normalize(vmin=min_ent, vmax=max_ent)

    for w in entropies_ent:
        f_ent = float(w)
        entropy_config = ent_results[round(f_ent, 2)]
        color = cmap_ent(norm_ent(max(f_ent, min_ent)))

        risks = np.asarray(entropy_config["risks"])[mask_list_ent]
        y_real = np.asarray(entropy_config["y_real"])[mask_cont_ent]
        y_real_der = np.asarray(entropy_config["y_real_der"])[mask_cont_ent]

        alpha = entropy_config["alpha"]
        model_label = fr"$\alpha = {alpha:.2f}$"

        ax1.scatter(train_list_ent_plot, risks, color=color, alpha=scatter_alpha, s=scatter_s, zorder=3, edgecolors='none')
        ax1.plot(train_cont_ent_plot, y_real, color=color, linewidth=line_w, label=model_label, zorder=5)

        ax3.plot(train_cont_ent_plot, y_real_der, color=color, linewidth=line_w, zorder=5)

    avail_keys = list(entropy_cfg_res_size.keys())
    if target_entropy is not None:
        size_ent_key = min(avail_keys, key=lambda k: abs(float(k) - float(target_entropy)))
    else:
        size_ent_key = avail_keys[0]
    config_and_results_size = entropy_cfg_res_size[size_ent_key]

    models_list = [v for k, v in config_and_results_size.items() if isinstance(v, dict)]
    models_sorted = sorted(models_list, key=lambda m: parse_params(m.get("params", "0")))
    colors_size = [m["color"] for m in models_sorted]
    labels_size = [format_param_tick(m.get("params", "")) for m in models_sorted]

    for model_data in models_sorted:
        color = model_data["color"]
        alpha = model_data["alpha"]
        model_label = fr"$\alpha = {alpha:.2f}$"

        risks = np.asarray(model_data["risks"])[mask_list_size]
        y_real = np.asarray(model_data["y_real"])[mask_cont_size]
        y_real_der = np.asarray(model_data["y_real_der"])[mask_cont_size]

        ax2.scatter(train_list_size_plot, risks, color=color, alpha=scatter_alpha, s=scatter_s, zorder=3, edgecolors='none')
        ax2.plot(train_cont_size_plot, y_real, color=color, linewidth=line_w, label=model_label, zorder=5)

        ax4.plot(train_cont_size_plot, y_real_der, color=color, linewidth=line_w, zorder=5)

    for ax in [ax1, ax2]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Regret", fontsize=label_fs)
        ax.set_xlim(min_examples, max_examples)
        ax.grid(True, which="major", linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", labelsize=tick_fs)

    legend_fs = 4.0
    ax1.legend(fontsize=legend_fs, loc="lower left", ncol=2, frameon=True, handlelength=0.8, handletextpad=0.2, labelspacing=0.1, borderpad=0.2, columnspacing=0.6)
    ax2.legend(fontsize=legend_fs, loc="lower left", ncol=1, frameon=True, handlelength=0.8, handletextpad=0.2, labelspacing=0.1, borderpad=0.2)

    if ylim is not None:
        ax1.set_ylim(float(ylim[0]), float(ylim[1]))
        ax2.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        y_min_top = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        y_max_top = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(y_min_top, y_max_top)
        ax2.set_ylim(y_min_top, y_max_top)

    minimax_rate = 1.0
    ax3.relim()
    ax3.autoscale_view()
    ax4.relim()
    ax4.autoscale_view()
    y_min_bot = min(ax3.get_ylim()[0], ax4.get_ylim()[0])
    y_max_bot = max(ax3.get_ylim()[1], ax4.get_ylim()[1])
    y_min_bot = max(0.0, np.floor(y_min_bot * 10) / 10)
    y_max_bot = max(1.15, np.ceil(y_max_bot * 10) / 10 + 0.05)

    x_fill = np.logspace(np.log10(min_examples), np.log10(max_examples), 500)

    for ax in [ax3, ax4]:
        ax.set_xscale("log")
        ax.set_ylabel("Scaling Exponent", fontsize=label_fs)
        ax.set_xlabel(r"In-Context Examples ($n$)", fontsize=label_fs)
        ax.set_xlim(min_examples, max_examples)
        ax.axhline(y=minimax_rate, color="black", linewidth=0.8, linestyle="dotted", zorder=2)
        ax.grid(True, which="major", linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", labelsize=tick_fs)
        ax.set_ylim(y_min_bot, y_max_bot)
        ax.fill_between(x_fill, y1=minimax_rate, y2=y_max_bot, facecolor="g", alpha=fill_alpha, zorder=1)
        ax.fill_between(x_fill, y1=y_min_bot, y2=minimax_rate, facecolor="r", alpha=fill_alpha, zorder=1)

    fig.subplots_adjust(left=0.15, right=0.88, top=0.92, bottom=0.12, wspace=wspace, hspace=hspace)
    fig.canvas.draw()

    p_top_left = ax1.get_position()
    p_bot_left = ax3.get_position()
    cbar_w = 0.015
    cax_ent = fig.add_axes([p_bot_left.x0 - 0.08, p_bot_left.y0, cbar_w, p_top_left.y1 - p_bot_left.y0])
    sm_ent = cm.ScalarMappable(cmap=cmap_ent, norm=norm_ent)
    sm_ent.set_array([])
    cbar_ent = fig.colorbar(sm_ent, cax=cax_ent)
    cbar_ent.ax.yaxis.set_ticks_position("left")
    cbar_ent.ax.yaxis.set_label_position("left")
    cbar_ent.set_ticks([min_ent, max_ent])
    cbar_ent.set_ticklabels([f"{min_ent:.2f}", f"{max_ent:.2f}"])
    cbar_ent.set_label(r"$H(Y \mid X)$ (nats)", fontsize=label_fs)
    cbar_ent.ax.tick_params(labelsize=cbar_tick_fs)

    p_top_right = ax2.get_position()
    p_bot_right = ax4.get_position()
    cax_size = fig.add_axes([p_top_right.x1 + 0.02, p_bot_right.y0, cbar_w, p_top_right.y1 - p_bot_right.y0])
    cmap_size = mcolors.ListedColormap(colors_size)
    norm_size = mcolors.BoundaryNorm(range(len(colors_size) + 1), cmap_size.N)
    sm_size = cm.ScalarMappable(cmap=cmap_size, norm=norm_size)
    sm_size.set_array([])
    cbar_size = fig.colorbar(sm_size, cax=cax_size)
    cbar_size.set_ticks([i + 0.5 for i in range(len(colors_size))])
    cbar_size.set_ticklabels(labels_size)
    cbar_size.set_label("Parameters (B)", fontsize=label_fs)
    cbar_size.ax.tick_params(labelsize=cbar_tick_fs)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    min_tick_x0 = min(t.get_window_extent(r).x0 for t in cbar_ent.ax.get_yticklabels())
    inv_ent = cbar_ent.ax.transAxes.inverted()
    min_tick_x0_axes = inv_ent.transform((min_tick_x0, 0))[0]
    cbar_ent.ax.yaxis.set_label_coords(min_tick_x0_axes - 0.28, 0.5)

    max_tick_x1 = max(t.get_window_extent(r).x1 for t in cbar_size.ax.get_yticklabels())
    inv_size = cbar_size.ax.transAxes.inverted()
    max_tick_x1_axes = inv_size.transform((max_tick_x1, 0))[0]
    cbar_size.ax.yaxis.set_label_coords(max_tick_x1_axes + 0.28, 0.5)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    lbl_x0 = cbar_ent.ax.yaxis.label.get_window_extent(r).x0
    inv_ax1 = ax1.transAxes.inverted()
    a_x = inv_ax1.transform((lbl_x0, 0))[0]

    lbl_b_x0 = ax2.yaxis.label.get_window_extent(r).x0
    inv_ax2 = ax2.transAxes.inverted()
    b_x = inv_ax2.transform((lbl_b_x0, 0))[0]

    letter_y = 1.05
    ax1.text(a_x, letter_y, r"\textbf{(a)}", transform=ax1.transAxes, fontsize=letter_fs, fontweight="bold", va="bottom", ha="left", zorder=10)
    ax2.text(b_x, letter_y, r"\textbf{(b)}", transform=ax2.transAxes, fontsize=letter_fs, fontweight="bold", va="bottom", ha="left", zorder=10)

    return fig


def plot_and_save_icl_power_law_bound(cfg_entropy,
                                      cfg_size,
                                      target_entropy: float | None = None,
                                      xlim: tuple[float, float] | list[float] | None = None,
                                      ylim: tuple[float, float] | list[float] | None = None
) -> plt.Figure | None:
    """
    Generates and saves the 4-panel power-law scaling figures to PDF.

    Args:
        cfg_entropy: Configuration or run ID for the entropy experiment.
        cfg_size: Configuration or run ID for the model scaling experiment.
        target_entropy: Target entropy level to isolate.
        xlim: Optional (min, max) limits for x-axis.
        ylim: Optional (min, max) limits for y-axis.
    """
    if isinstance(cfg_entropy, str):
        r_id = cfg_entropy[4:] if cfg_entropy.startswith("run_") else cfg_entropy
        cfg_entropy = get_plot_config("icl_training", r_id)
    if isinstance(cfg_size, str):
        r_id = cfg_size[4:] if cfg_size.startswith("run_") else cfg_size
        cfg_size = get_plot_config("icl_training", r_id)

    xlim = xlim or getattr(cfg_entropy, 'xlim', None) or getattr(cfg_size, 'xlim', None)
    ylim = ylim or getattr(cfg_entropy, 'ylim', None) or getattr(cfg_size, 'ylim', None)

    models_1 = getattr(cfg_entropy, 'models', {})
    models_2 = getattr(cfg_size, 'models', {})
    if len(models_1) > 1 and len(models_2) == 1:
        cfg_entropy, cfg_size = cfg_size, cfg_entropy

    FIG_PATH = Path("results") / "figures"
    FIG_PATH.mkdir(parents=True, exist_ok=True)

    _, _, _, entropy_cfg_res_size = preprocessing_power_law_scaling(cfg_size)
    available_entropies = list(entropy_cfg_res_size.keys())

    if target_entropy is not None:
        selected_entropies = [min(available_entropies, key=lambda k: abs(float(k) - float(target_entropy)))]
    else:
        selected_entropies = available_entropies

    last_fig = None
    for ent_val in selected_entropies:
        fig = plot_icl_power_law_bound(cfg_entropy, cfg_size, target_entropy=ent_val, xlim=xlim, ylim=ylim)
        last_fig = fig

        filename = f"run_{cfg_entropy.run}__{cfg_size.run}_H_Y_X_{ent_val}_icl_scaling"
        save_fig(fig, FIG_PATH, filename)
        print(f"Saved icl_scaling (H={ent_val}):")
        print(f"  {FIG_PATH / filename}")

    return last_fig


# Clean aliases
plot_and_save_icl_entropy_and_model_scaling = plot_and_save_entropy_and_icl_training
plot_icl_entropy_and_model_scaling = plot_entropy_and_icl_training
