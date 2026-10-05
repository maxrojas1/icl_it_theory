import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .styles import add_params_colorbar, get_mapped_axes


def plot_icl_training(train_sizes: list[int],
                      entropy_config_and_results: dict,
                      metric: str,
                      subplots_to_plot: list[int] | None = None,
                      only_params: bool = False,
                      **kwargs
) -> dict[str, plt.Figure]:
    """
    Plots performance metrics (loss or excess risk) against context size for each entropy level.

    Args:
        train_sizes: Sequence of in-context training sample sizes (n).
        entropy_config_and_results: Preprocessed results grouped by entropy level.
        metric: Target evaluation metric ('risk' or 'loss').
        subplots_to_plot: Optional list of subplot indices to render.
        only_params: If True, displays a continuous model parameter colorbar instead of legend.
    """
    figsize = (3.5, 2.3)
    title_fs = 7.5
    label_fs = 7.0
    tick_fs = 6.0
    legend_fs = 4.5
    line_w = 1.1

    horiz_margin_left = 0.18
    horiz_margin_right = 0.85
    horiz_margin_top = 0.90
    horiz_margin_bottom = 0.18
    horiz_cax_rect = [0.87, 0.18, 0.035, 0.72]

    legend_loc = 'upper right'
    line_alpha = 0.9
    grid_ls = '--'
    grid_lw = 0.5
    grid_alpha = 0.7

    figures = {}

    for entropy_level, config_and_results in entropy_config_and_results.items():
        fig, mapped_axs = get_mapped_axes(figsize, 1, subplots_to_plot, sharex=True)
        ax1 = mapped_axs[0]

        if metric == "risk":
            title = config_and_results["risk_title"]
        else:
            title = config_and_results["loss_title"]

        if metric != "risk" and ax1 is not None:
            ax1.axhline(y=float(entropy_level), color="black", linestyle="dotted", alpha=line_alpha, linewidth=1.0, zorder=2)

        for _, model_data in config_and_results.items():
            if not isinstance(model_data, dict):
                continue

            name = model_data["name"]
            color = model_data["color"]
            params = model_data["params"]
            model_label = f"{name} ({params})"

            if metric == "risk":
                criterion = model_data["risks"]
            else:
                criterion = model_data["losses"]

            if ax1 is not None:
                ax1.plot(train_sizes, criterion, color=color, alpha=line_alpha, linewidth=line_w, label=model_label, zorder=3)

        if ax1 is not None:
            ylabel = "Regret" if metric == "risk" else "Loss (nats)"
            ax1.set_ylabel(ylabel, fontsize=label_fs)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            if 'xlim' in kwargs and kwargs['xlim'] is not None:
                ax1.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
            else:
                ax1.set_xlim(train_sizes[0] * 0.85, train_sizes[-1] * 1.15)
            if 'ylim' in kwargs and kwargs['ylim'] is not None:
                ax1.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
            ax1.grid(True, which='both', linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
            ax1.set_axisbelow(True)
            ax1.tick_params(axis='both', labelsize=tick_fs)
            ax1.set_xlabel("In-Context Examples (n)", fontsize=label_fs)

        if only_params:
            fig.suptitle(title, fontsize=title_fs, y=0.98)
            fig.subplots_adjust(left=horiz_margin_left, right=horiz_margin_right, top=horiz_margin_top, bottom=horiz_margin_bottom)
            add_params_colorbar(fig, config_and_results, "vertical", horiz_cax_rect, label_fs, tick_fs)
        else:
            if ax1 is not None:
                ax1.legend(fontsize=legend_fs, loc=legend_loc, frameon=True, handlelength=0.8, handletextpad=0.2, labelspacing=0.1, borderpad=0.2)
            fig.suptitle(title, fontsize=title_fs, y=0.98)
            fig.subplots_adjust(left=horiz_margin_left, right=horiz_margin_right, top=horiz_margin_top, bottom=horiz_margin_bottom)

        figures[entropy_level] = fig

    return figures


def plot_entropy_icl_training(train_sizes: list[int],
                              entropies: list[float],
                              model_config_and_results: dict,
                              metric: str,
                              subplots_to_plot: list[int] | None = None,
                              **kwargs
) -> plt.Figure:
    """
    Plots in-context performance curves across varying task entropies for a single model.

    Args:
        train_sizes: Evaluated context training sizes.
        entropies: List of conditional entropy levels H(Y|X).
        model_config_and_results: Dictionary containing evaluated losses and model metadata.
        metric: 'risk' or 'loss' metric to plot.
        subplots_to_plot: Optional subplot index filter.
    """
    figsize = (3.5, 2.3)
    title_fs = 7.5
    label_fs = 7.0
    tick_fs = 6.0
    line_w = 1.1

    horiz_suptitle_y = 0.96
    horiz_margin_left = 0.14
    horiz_margin_right = 0.83
    horiz_margin_top = 0.86
    horiz_margin_bottom = 0.16
    horiz_cax_rect = [0.85, 0.16, 0.025, 0.68]

    line_alpha = 0.9
    grid_alpha = 0.5
    grid_lw = 0.5
    grid_ls = '--'

    fig, mapped_axs = get_mapped_axes(figsize, 1, subplots_to_plot, sharex=True)
    ax1 = mapped_axs[0]

    cmap = plt.get_cmap('cividis')
    float_entropies = [float(e) for e in entropies]
    min_ent = min(float_entropies)
    max_ent = max(float_entropies)
    norm = mcolors.Normalize(vmin=min_ent, vmax=max_ent)

    for _, config_and_results in model_config_and_results.items():
        if metric == "risk":
            criterion = config_and_results["risks"].to_numpy()
            title = config_and_results["risk_title"]
        else: 
            criterion = config_and_results["losses"].to_numpy()
            title = config_and_results["loss_title"]

        num_entropies = criterion.shape[0]
        
        for i in range(num_entropies):
            entropy_level = float(entropies[i])
            entropy_color = cmap(norm(entropy_level)) if num_entropies > 1 else cmap(0.5)

            if metric != "risk" and ax1 is not None:
                ax1.axhline(y=entropy_level, color=entropy_color, linestyle="dotted", alpha=line_alpha, linewidth=1.0, zorder=2)

            if ax1 is not None:
                ax1.plot(train_sizes, criterion[i, :], color=entropy_color, alpha=line_alpha, linewidth=line_w, zorder=3)

    if ax1 is not None:
        ylabel = "Regret" if metric == "risk" else "Loss (nats)"
        ax1.set_ylabel(ylabel, fontsize=label_fs)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        if 'xlim' in kwargs and kwargs['xlim'] is not None:
            ax1.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        else:
            ax1.set_xlim(train_sizes[0] * 0.85, train_sizes[-1] * 1.15)
        if 'ylim' in kwargs and kwargs['ylim'] is not None:
            ax1.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
        ax1.grid(True, which='both', linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
        ax1.set_axisbelow(True)
        ax1.tick_params(axis='both', labelsize=tick_fs)
        ax1.set_xlabel("In-Context Examples (n)", fontsize=label_fs)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    fig.suptitle(title, fontsize=title_fs, y=horiz_suptitle_y) 
    fig.subplots_adjust(left=horiz_margin_left, right=horiz_margin_right, top=horiz_margin_top, bottom=horiz_margin_bottom)
    cax = fig.add_axes(horiz_cax_rect)
        
    cbar = fig.colorbar(sm, cax=cax, format='%.2f')
    cbar.set_label(r"$H(Y \mid X)$", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    current_ticks = cbar.get_ticks()
    valid_ticks = [t for t in current_ticks if min_ent < t < max_ent]
    cbar.set_ticks([min_ent] + valid_ticks + [max_ent])

    return fig


def plot_power_law(train_list: list[int],
                   train_continuous: np.ndarray,
                   entropy_config_and_results: dict,
                   subplots_to_plot: list[int] | None = None,
                   only_params: bool = False,
                   **kwargs
) -> dict[str, plt.Figure]:
    """
    Plots empirical excess risk alongside fitted power-law curves and instantaneous scaling rates.

    Args:
        train_list: Discrete context sizes evaluated.
        train_continuous: Dense evaluation points for fitted power law curves.
        entropy_config_and_results: Dictionary of model fit parameters and risks per entropy level.
        subplots_to_plot: Optional subplot indices to render (1: fit, 2: derivative).
        only_params: If True, shows colorbar rather than model legend.
    """
    figsize = (7.0, 2.8)
    suptitle_fs = 8.5
    title_fs = 7.5
    label_fs = 7.0   
    tick_fs = 6.0    
    legend_fs = 4.5  
    line_w = 1.1   
    scatter_s = 6
    pad_title = 4
    
    horiz_suptitle_y = 0.98
    horiz_margin_left = 0.07
    horiz_margin_right = 0.98
    horiz_margin_top = 0.82
    horiz_margin_bottom = 0.16
    horiz_margin_wspace = 0.15
    scatter_alpha = 0.15
    fill_alpha = 0.20
    grid_alpha = 0.1
    grid_lw = 0.5
    grid_ls = '--'
    xlim = kwargs.get('xlim')
    ylim = kwargs.get('ylim')
    min_examples = float(xlim[0]) if xlim is not None else float(train_list[0])
    max_examples = float(xlim[1]) if xlim is not None else float(train_list[-1])

    mask_list = (np.asarray(train_list) <= max_examples)
    mask_cont = (np.asarray(train_continuous) <= max_examples)
    train_list_plot = np.asarray(train_list)[mask_list]
    train_continuous_plot = np.asarray(train_continuous)[mask_cont]

    figures = {}

    for entropy_level, config_and_results in entropy_config_and_results.items():
        fig, mapped_axs = get_mapped_axes(figsize, 2, subplots_to_plot, sharex=True)
        ax1, ax2 = mapped_axs
            
        title = config_and_results.get("title", fr"ICL Scaling Law for $H(Y \mid X)$ = {entropy_level}")
        y_minimax_der_shared = None

        for key, model_data in config_and_results.items():
            if not isinstance(model_data, dict):
                continue

            risks = np.asarray(model_data["risks"])[mask_list]
            name = model_data["name"]
            color = model_data["color"]
            params = model_data["params"]
            y_real = np.asarray(model_data["y_real"])[mask_cont]
            y_real_der = np.asarray(model_data["y_real_der"])[mask_cont]
            y_minimax_der_shared = np.asarray(model_data["y_minimax_der"])[mask_cont]
            
            alpha = model_data["alpha"]
            model_label = fr"{name} ({params}) $\rightarrow \alpha = {alpha:.2f}$"
            
            if ax1 is not None:
                ax1.scatter(train_list_plot, risks, color=color, alpha=scatter_alpha, s=scatter_s, zorder=3)
                ax1.plot(train_list_plot, risks, color=color, alpha=scatter_alpha)
                ax1.plot(train_continuous_plot, y_real, color=color, linewidth=line_w, label=model_label, zorder=5)
            
            if ax2 is not None:
                ax2.plot(train_continuous_plot, y_real_der, color=color, linewidth=line_w, zorder=5)
            
        if y_minimax_der_shared is not None and ax2 is not None:
            ax2.relim()
            ax2.autoscale_view()
            y_min, y_max = ax2.get_ylim()
            if y_max > 1.0:
                ax2.fill_between(train_continuous_plot, y1=np.maximum(y_minimax_der_shared, y_min), y2=y_max, facecolor='g', alpha=fill_alpha)
            if y_min < 1.0:
                ax2.fill_between(train_continuous_plot, y1=y_min, y2=np.minimum(y_minimax_der_shared, y_max), facecolor='r', alpha=fill_alpha)
            ax2.set_ylim(y_min, y_max)

        if ax1 is not None:
            ax1.set_xlabel(r"In-Context Examples ($n$)", fontsize=label_fs)
            ax1.set_ylabel("Regret", fontsize=label_fs)
            ax1.set_xscale('log')
            ax1.set_yscale('log') 
            ax1.set_xlim(min_examples, max_examples)
            if ylim is not None:
                ax1.set_ylim(ylim[0], ylim[1])
            ax1.grid(True, which='both', linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)  
            ax1.set_title("Power Law Fitting", fontsize=title_fs, fontweight='normal', pad=pad_title)
            ax1.tick_params(axis='both', labelsize=tick_fs)

        if ax2 is not None:
            ax2.set_xlabel(r"In-Context Examples ($n$)", fontsize=label_fs)
            ax2.set_ylabel(r"Scaling Exponent", fontsize=label_fs)
            ax2.set_xscale('log') 
            ax2.set_xlim(min_examples, max_examples)
            ax2.grid(True, which='both', linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)  
            ax2.set_title(r"Instantaneous Scaling Rate", fontsize=title_fs, fontweight='normal', pad=pad_title)
            ax2.tick_params(axis='both', labelsize=tick_fs)

        if only_params:
            fig.suptitle(title, fontsize=suptitle_fs, fontweight='normal', y=horiz_suptitle_y)
            fig.subplots_adjust(left=horiz_margin_left, right=0.87, bottom=horiz_margin_bottom, top=horiz_margin_top, wspace=horiz_margin_wspace)
            add_params_colorbar(fig, config_and_results, "vertical", [0.89, 0.16, 0.015, 0.66], label_fs, tick_fs)
        else:
            if ax1 is not None:
                ax1.legend(fontsize=legend_fs, loc='lower left', frameon=True, handlelength=0.8, handletextpad=0.2, labelspacing=0.1, borderpad=0.2)
            fig.suptitle(title, fontsize=suptitle_fs, fontweight='normal', y=horiz_suptitle_y)
            fig.subplots_adjust(left=horiz_margin_left, right=horiz_margin_right, bottom=horiz_margin_bottom, top=horiz_margin_top, wspace=horiz_margin_wspace)
        
        figures[entropy_level] = fig
        
    return figures


def plot_entropy_power_law(train_list: list[int],
                           train_continuous: np.ndarray,
                           entropies: list[float],
                           model_config_and_results: dict,
                           subplots_to_plot: list[int] | None = None,
                           **kwargs
) -> plt.Figure:
    """
    Plots power-law fits and instantaneous scaling rates across all task entropies for a model.

    Args:
        train_list: Discrete training sizes.
        train_continuous: Dense evaluation points for curves.
        entropies: Evaluated task entropy levels.
        model_config_and_results: Power-law fit results and metadata for the target model.
        subplots_to_plot: Optional subplot indices to render.
    """
    figsize = (7.0, 2.8)
    suptitle_fs = 8.5
    title_fs = 7.5
    label_fs = 7.0   
    tick_fs = 6.0    
    line_w = 1.1   
    scatter_s = 6
    pad_title = 4
    
    horiz_suptitle_y = 0.98
    horiz_margin_left = 0.08
    horiz_margin_right = 0.87
    horiz_margin_top = 0.82
    horiz_margin_bottom = 0.16
    horiz_margin_wspace = 0.25
    horiz_cax_rect = [0.89, 0.16, 0.015, 0.66]
    scatter_alpha = 0.15
    fill_alpha = 0.20
    grid_alpha = 0.1
    grid_lw = 0.5
    grid_ls = '--'
    xlim = kwargs.get('xlim')
    ylim = kwargs.get('ylim')
    min_examples = float(xlim[0]) if xlim is not None else float(train_list[0])
    max_examples = float(xlim[1]) if xlim is not None else float(train_list[-1])

    mask_list = (np.asarray(train_list) <= max_examples)
    mask_cont = (np.asarray(train_continuous) <= max_examples)
    train_list_plot = np.asarray(train_list)[mask_list]
    train_continuous_plot = np.asarray(train_continuous)[mask_cont]

    fig, mapped_axs = get_mapped_axes(figsize, 2, subplots_to_plot, sharex=True)
    ax1, ax2 = mapped_axs

    cmap = plt.get_cmap('cividis')
    float_entropies = [float(e) for e in entropies]
    min_ent = min(float_entropies)
    max_ent = max(float_entropies)
    norm = mcolors.Normalize(vmin=min_ent, vmax=max_ent)

    entropies_results = model_config_and_results["entropies_results"]
    title = model_config_and_results["title"]
    y_minimax_der_shared = None

    for w_val, results in entropies_results.items():
        color = cmap(norm(float(w_val)))
        risks = np.asarray(results["risks"])[mask_list]
        y_real = np.asarray(results["y_real"])[mask_cont]
        y_real_der = np.asarray(results["y_real_der"])[mask_cont]
        y_minimax_der_shared = np.asarray(results["y_minimax_der"])[mask_cont]

        if ax1 is not None:
            ax1.scatter(train_list_plot, risks, color=color, alpha=scatter_alpha, s=scatter_s, zorder=3)
            ax1.plot(train_list_plot, risks, color=color, alpha=scatter_alpha)
            ax1.plot(train_continuous_plot, y_real, color=color, linewidth=line_w, zorder=5)

        if ax2 is not None:
            ax2.plot(train_continuous_plot, y_real_der, color=color, linewidth=line_w, zorder=5)

    if y_minimax_der_shared is not None and ax2 is not None:
        ax2.relim()
        ax2.autoscale_view()
        y_min, y_max = ax2.get_ylim()
        if y_max > 1.0:
            ax2.fill_between(train_continuous_plot, y1=np.maximum(y_minimax_der_shared, y_min), y2=y_max, facecolor='g', alpha=fill_alpha)
        if y_min < 1.0:
            ax2.fill_between(train_continuous_plot, y1=y_min, y2=np.minimum(y_minimax_der_shared, y_max), facecolor='r', alpha=fill_alpha)
        ax2.set_ylim(y_min, y_max)

    if ax1 is not None:
        ax1.set_xlabel(r"In-Context Examples ($n$)", fontsize=label_fs)
        ax1.set_ylabel("Regret", fontsize=label_fs)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(min_examples, max_examples)
        if ylim is not None:
            ax1.set_ylim(ylim[0], ylim[1])
        ax1.grid(True, which='both', linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
        ax1.set_title("Power Law Fitting", fontsize=title_fs, fontweight='normal', pad=pad_title)
        ax1.tick_params(axis='both', labelsize=tick_fs)

    if ax2 is not None:
        ax2.set_xlabel(r"In-Context Examples ($n$)", fontsize=label_fs)
        ax2.set_ylabel(r"Scaling Exponent", fontsize=label_fs)
        ax2.set_xscale('log')
        ax2.set_xlim(min_examples, max_examples)
        ax2.grid(True, which='both', linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
        ax2.set_title(r"Instantaneous Scaling Rate", fontsize=title_fs, fontweight='normal', pad=pad_title)
        ax2.tick_params(axis='both', labelsize=tick_fs)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig.suptitle(title, fontsize=suptitle_fs, fontweight='normal', y=horiz_suptitle_y)
    fig.subplots_adjust(left=horiz_margin_left, right=horiz_margin_right, bottom=horiz_margin_bottom, top=horiz_margin_top, wspace=horiz_margin_wspace)
    cax = fig.add_axes(horiz_cax_rect)

    cbar = fig.colorbar(sm, cax=cax, format='%.2f')
    cbar.set_label(r"$H(Y \mid X)$", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    current_ticks = cbar.get_ticks()
    valid_ticks = [t for t in current_ticks if min_ent < t < max_ent]
    cbar.set_ticks([min_ent] + valid_ticks + [max_ent])

    return fig


def plot_conditional_entropy(con_entropies: list[float] | np.ndarray,
                             selected_entropies: list[float]
) -> plt.Figure:
    """
    Plots the sorted conditional entropy spectrum across simulated tasks with selected thresholds.

    Args:
        con_entropies: Conditional entropy values across all simulated tasks.
        selected_entropies: Highlighted entropy levels selected for experiment evaluation.
    """
    figsize = (3.5, 2.3)
    line_w = 1.1
    title_fs = 8.0
    label_fs = 7.0
    tick_fs = 6.0

    line_color = "teal"
    line_alpha = 0.9
    hline_color = "black"
    hline_ls = "--"
    y_lim = (0, 0.75)
    grid_ls = '--'
    grid_lw = 0.5
    grid_alpha = 0.7

    fig, ax1 = plt.subplots(figsize=figsize)
    
    ax1.plot(range(1, len(con_entropies) + 1), con_entropies, linewidth=line_w, color=line_color, alpha=line_alpha)

    for entropy in selected_entropies:
        ax1.axhline(y=entropy, color=hline_color, linestyle=hline_ls)

    ax1.set_title("Conditional Entropy for each Task", fontsize=title_fs)
    ax1.set_xlabel("Task ID", fontsize=label_fs)
    ax1.set_ylabel("Conditional Entropy (Nats)", fontsize=label_fs)
    ax1.set_ylim(y_lim)
    ax1.grid(True, which='both', linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
    ax1.tick_params(axis='both', labelsize=tick_fs)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.90, bottom=0.18)
    
    return fig
