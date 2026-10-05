from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science'])


def parse_params(param_str: str | float | int
) -> float:
    """
    Parses a model parameter size string (e.g. '0.5B', '7B', '350M') into billions of parameters.

    Args:
        param_str: Parameter size descriptor.
    """
    s = str(param_str).strip().upper()
    if s.endswith('B'):
        return float(s[:-1])
    elif s.endswith('M'):
        return float(s[:-1]) / 1000.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def format_param_tick(p: str | float
) -> str:
    """
    Strips trailing 'B' unit suffix for clean numerical colorbar tick labels.

    Args:
        p: Raw parameter label.
    """
    s = str(p).strip()
    if s.upper().endswith('B'):
        return s[:-1]
    return s


def get_mapped_axes(figsize: tuple[float, float],
                    num_subplots: int,
                    subplots_to_plot: list[int] | None = None,
                    sharex: bool = True,
                    is_combined: bool = False
) -> tuple:
    """
    Creates subplots and maps them to original indices for selective rendering.

    Args:
        figsize: Figure dimensions in inches (width, height).
        num_subplots: Total number of canonical subplots in the figure family.
        subplots_to_plot: 1-indexed list of subplots to actually render.
        sharex: Whether created axes share the horizontal axis.
        is_combined: True when creating side-by-side comparative subplots.
    """
    if subplots_to_plot is None:
        subplots_to_plot = list(range(1, num_subplots + 1))
    actual_num = len(subplots_to_plot)
    
    if is_combined:
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=sharex, sharey="row", figsize=figsize)
        return fig, axs

    fig, axs = plt.subplots(nrows=1, ncols=actual_num, sharex=sharex, figsize=figsize)
        
    if actual_num == 1:
        axs = [axs]
    else:
        axs = list(axs.flatten())
        
    mapped_axs = []
    curr = 0
    for j in range(1, num_subplots + 1):
        if j in subplots_to_plot:
            mapped_axs.append(axs[curr])
            curr += 1
        else:
            mapped_axs.append(None)
    return fig, mapped_axs


def add_params_colorbar(fig: plt.Figure,
                       models_dict: dict,
                       orientation: str,
                       cax_rect: list[float],
                       label_fs: int,
                       tick_fs: int
) -> cm.ScalarMappable | None:
    """
    Adds a custom model-size parameter colorbar mapped across evaluated checkpoints.

    Args:
        fig: Target Matplotlib figure.
        models_dict: Dictionary containing evaluated models and their color/param configs.
        orientation: Colorbar direction ('vertical' or 'horizontal').
        cax_rect: Bounding box [left, bottom, width, height] for the colorbar axes.
        label_fs: Font size for colorbar title label.
        tick_fs: Font size for tick annotations.
    """
    models_list = [v for _, v in models_dict.items() if isinstance(v, dict)]
    models_sorted = sorted(models_list, key=lambda m: parse_params(m.get('params', '0')))
    colors = [m['color'] for m in models_sorted]
    labels = [str(m['params']) for m in models_sorted]

    cax = fig.add_axes(cax_rect)
    cbar_orientation = "vertical" if orientation == "vertical" else "horizontal"

    if not colors:
        return None
    elif len(colors) == 1:
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation=cbar_orientation)
        cbar.set_ticks([0.5])
        cbar.set_ticklabels(labels)
    else:
        cmap = mcolors.LinearSegmentedColormap.from_list("model_param_cmap", colors)
        norm = mcolors.Normalize(vmin=0, vmax=len(colors) - 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation=cbar_orientation)
        cbar.set_ticks(range(len(colors)))
        cbar.set_ticklabels(labels)

    cbar.set_label("Parameters", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)
    return cbar


def save_fig(fig: plt.Figure,
             path: Path,
             filename_base: str,
             formats: list[str] | None = None
) -> None:
    """
    Saves a Matplotlib figure as publication-ready vector PDF and closes it.

    Args:
        fig: Figure object to serialize.
        path: Target directory path.
        filename_base: File name without extension.
        formats: Kept for backwards compatibility (exports PDF).
    """
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{filename_base}.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
