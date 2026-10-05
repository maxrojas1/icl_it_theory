from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

from src.theory.dgp import (
    DGP_1D,
    calculate_conditional_entropy,
)
from .styles import save_fig


def plot_DGP_1D_family(x: np.ndarray,
                       weights: np.ndarray,
                       biases: np.ndarray,
                       fixed_weight: float = 1.0,
                       fixed_bias: float = 0.0,
                       base_filename: str = 'dgp_1d_family',
                       save_dir: str = 'results'
) -> plt.Figure:
    """
    Plots the 1D data-generating process sigmoid family across weight and bias configurations.

    Args:
        x: Input feature grid.
        weights: Array of candidate slope weights.
        biases: Array of candidate bias shifts.
        fixed_weight: Reference weight for bias sweep.
        fixed_bias: Reference bias for weight sweep.
        base_filename: Output PDF figure filename base.
        save_dir: Directory where the output PDF is saved.
    """
    title_fs = 9
    label_fs = 8
    tick_fs = 7
    legend_fs = 5
    line_w = 1.2
    
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))
    
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.90, wspace=0.12, hspace=0.35)
    
    cmap = plt.get_cmap("coolwarm")
    
    min_val = min(min(weights), min(biases))
    max_val = max(max(weights), max(biases))
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    
    for weight in weights:
        p = DGP_1D(x, weight, fixed_bias)
        axes[0, 0].plot(x, p, color=cmap(norm(weight)), lw=0.6, alpha=0.8)

    axes[0, 0].set_title('Sigmoid Family varying Weight\n($w_0 = 0$)', fontsize=title_fs, pad=4)
    axes[0, 0].set_xlabel('X', fontsize=label_fs)
    axes[0, 0].set_ylabel(r'$\mathbb{P}(Y=1 \mid X)$', fontsize=label_fs)
    axes[0, 0].set_xlim(-3, 3)
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].grid(alpha=0.1, linestyle='--', linewidth=0.5)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=tick_fs)
    
    for bias in biases:
        p = DGP_1D(x, fixed_weight, bias)
        axes[0, 1].plot(x, p, color=cmap(norm(bias)), lw=0.6, alpha=0.8)

    axes[0, 1].set_title('Sigmoid Family varying Bias\n($w_1 = 1$)', fontsize=title_fs, pad=4)
    axes[0, 1].set_xlabel('X', fontsize=label_fs)
    axes[0, 1].set_xlim(-3, 3)
    axes[0, 1].grid(alpha=0.1, linestyle='--', linewidth=0.5)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=tick_fs)
    axes[0, 1].sharey(axes[0, 0])
    plt.setp(axes[0, 1].get_yticklabels(), visible=False)

    np.random.seed(42)
    p0 = DGP_1D(x, 1, 0)
    axes[0, 2].plot(x, p0, color='black', lw=line_w, alpha=0.3)
    
    x_samples = np.random.uniform(-3, 3, size=75)
    p_samples = DGP_1D(x_samples, 1, 0)
    y_samples = np.random.binomial(1, p_samples)
    
    mask_1 = y_samples == 1
    mask_0 = y_samples == 0
    
    axes[0, 2].scatter(x_samples[mask_1], y_samples[mask_1], 
                       color='green', edgecolor='black', alpha=0.6, s=6, lw=0.3, label='Y = 1')
    axes[0, 2].scatter(x_samples[mask_0], y_samples[mask_0], 
                       color='purple', edgecolor='black', alpha=0.6, s=6, lw=0.3, label='Y = 0')
    
    axes[0, 2].set_title('Sigmoid \\& Samples\n($w_1=1, w_0=0$)', fontsize=title_fs, pad=4)
    axes[0, 2].set_xlabel('X', fontsize=label_fs)
    axes[0, 2].set_xlim(-3, 3)
    axes[0, 2].grid(alpha=0.1, linestyle='--', linewidth=0.5)
    axes[0, 2].tick_params(axis='both', which='major', labelsize=tick_fs)
    axes[0, 2].sharey(axes[0, 0])
    plt.setp(axes[0, 2].get_yticklabels(), visible=False)
    
    axes[0, 2].legend(fontsize=legend_fs, loc='center right', frameon=True, handletextpad=0.1, borderpad=0.2)
    
    ax2_twin = axes[0, 2].twinx()
    ax2_twin.set_ylabel('Label', fontsize=label_fs)
    ax2_twin.set_ylim(axes[0, 0].get_ylim())
    ax2_twin.set_yticks([0, 1])
    ax2_twin.tick_params(axis='y', which='major', labelsize=tick_fs)

    mc_x = np.random.normal(0, 1, size=10000)
    ent_w, ent_b, _, _, H_grid = calculate_conditional_entropy(weights, biases, mc_x)

    axes[1, 0].plot(weights, ent_w, color='black', lw=line_w)
    axes[1, 0].set_xlabel('$w_1$', fontsize=label_fs)
    axes[1, 0].set_ylabel(r'$H(Y \mid X)$ (nats)', fontsize=label_fs)
    axes[1, 0].grid(alpha=0.1, linestyle='--', linewidth=0.5)
    axes[1, 0].set_ylim(0.0, 0.75)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=tick_fs)

    axes[1, 1].plot(biases, ent_b, color='black', lw=line_w)
    axes[1, 1].set_xlabel('$w_0$', fontsize=label_fs)
    axes[1, 1].grid(alpha=0.1, linestyle='--', linewidth=0.5)
    axes[1, 1].set_ylim(0.0, 0.75)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=tick_fs)
    axes[1, 1].sharey(axes[1, 0])
    plt.setp(axes[1, 1].get_yticklabels(), visible=False)

    axes[1, 2].imshow(H_grid, extent=[weights.min(), weights.max(), biases.min(), biases.max()],
                      origin='lower', aspect='auto', cmap='cividis')
    axes[1, 2].set_xlabel('$w_1$', fontsize=label_fs)
    axes[1, 2].grid(alpha=0.1, linestyle='--', linewidth=0.5)
    axes[1, 2].tick_params(axis='both', which='major', labelsize=tick_fs)
    
    axes[1, 2].set_ylim(biases.min(), biases.max())
    axes[1, 2].yaxis.tick_right()
    axes[1, 2].yaxis.set_label_position("right")
    axes[1, 2].set_ylabel('$w_0$', fontsize=label_fs)

    sm1 = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar1_ax = fig.add_axes([0.10, 0.05, 0.35, 0.02])
    cbar1 = fig.colorbar(sm1, cax=cbar1_ax, orientation='horizontal')
    cbar1.set_label('Parameter (Weight or Bias)', fontsize=label_fs)
    cbar1.ax.tick_params(labelsize=tick_fs)

    sm2 = cm.ScalarMappable(cmap='cividis', norm=mcolors.Normalize(vmin=H_grid.min(), vmax=H_grid.max()))
    cbar2_ax = fig.add_axes([0.55, 0.05, 0.35, 0.02])
    cbar2 = fig.colorbar(sm2, cax=cbar2_ax, orientation='horizontal')
    cbar2.set_label(r'$H(Y \mid X)$ (nats)', fontsize=label_fs)
    cbar2.ax.tick_params(labelsize=tick_fs)

    save_path = Path(save_dir) if save_dir else Path("results")
    out_file = Path(base_filename)
    if out_file.parent != Path("."):
        target_dir = out_file.parent
        target_name = out_file.name
    else:
        target_dir = save_path
        target_name = out_file.name

    save_fig(fig, target_dir, target_name)
    return fig

