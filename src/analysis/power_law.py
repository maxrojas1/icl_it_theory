import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.theory.entropy import get_all_simulated_tasks_and_ces, get_tasks_by_entropy

def load_fitted_weights(run_path: Path | str
) -> dict:
    """
    Loads cached power-law fit parameters from fitted_weights.json.

    Args:
        run_path: Directory path for the experiment run.
    """
    weights_path = Path(run_path) / "fitted_weights.json"
    if weights_path.exists():
        try:
            with open(weights_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_fitted_weights(run_path: Path | str,
                        weights: dict
) -> None:
    """
    Persists power-law curve fit parameters to JSON cache.

    Args:
        run_path: Directory path for the experiment run.
        weights: Dictionary mapping fit keys to parameter values (A, alpha, c).
    """
    weights_path = Path(run_path) / "fitted_weights.json"
    try:
        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=4)
    except Exception:
        pass


def power_law_fit_func(n,
                       A: float, 
                       alpha: float, 
                       c: float
) -> np.ndarray:
    """
    Evaluates y = A * n^(-alpha) + c in log10 space for scipy curve fitting.

    Args:
        n: Training context sample sizes.
        A: Scaling amplitude prefactor.
        alpha: Power-law decay exponent.
        c: Asymptotic risk plateau / offset.
    """
    y_real = A * (n ** (-alpha)) + c
    return np.log10(np.clip(y_real, 1e-10, None))


def get_power_law_fits(n: np.ndarray,
                       A: float,
                       alpha: float,
                       c: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates power-law fit trajectories and log-log derivatives across context sizes.

    Args:
        n: Continuous sequence of training sizes.
        A: Scaling amplitude.
        alpha: Power-law decay exponent.
        c: Plateau constant.
    """
    y_real = A * (1.0 / (n**alpha)) + c
    y_ideal = A * (1.0 / (n**alpha))

    dy_real_dn = -A * alpha / (n ** (alpha + 1.0))
    y_real_der = -1.0 * (n / np.maximum(y_real, 1e-10)) * dy_real_dn
    y_ideal_der = np.full_like(n, alpha, dtype=float)
    y_minimax_der = np.ones_like(n, dtype=float)

    return y_real, y_ideal, y_real_der, y_ideal_der, y_minimax_der


def get_base_preprocessing(cfg,
                           include_continuous: bool = False
) -> tuple:
    """
    Resolves filesystem paths, training sizes, and task entropies common to all plotting routines.

    Args:
        cfg: Experiment configuration object.
        include_continuous: Whether to generate dense evaluation points for fitted curves.
    """
    if hasattr(cfg, "run_dir") and cfg.run_dir and Path(cfg.run_dir).exists():
        RUN_PATH = Path(cfg.run_dir)
    else:
        clean_run = str(cfg.run)[4:] if str(cfg.run).startswith("run_") else str(cfg.run)
        RUN_PATH = Path("results") / "runs" / f"run_{clean_run}"

    if getattr(cfg, "old_geom", False):
        train_list = np.unique(np.geomspace(1, cfg.max_train, num=cfg.num_train, dtype=int))
    else:
        train_list = np.unique(np.round(np.geomspace(1, cfg.max_train, num=cfg.num_train)).astype(int))

    if hasattr(cfg, "entropies") and cfg.entropies is not None:
        entropies = [float(e) for e in cfg.entropies]
        tasks_list = None
    else:
        tasks_list, entropies = get_tasks_by_entropy(
            dim=cfg.dim,
            seed_entropy=cfg.seed_entropy,
            entropies=None,
            num_entropy_levels=getattr(cfg, 'num_entropy_levels', 10),
            num_task=getattr(cfg, 'entropy_num_tasks', 100_000)
        )

    if include_continuous:
        train_continuous = np.geomspace(1, cfg.max_train, 1_000)
        return RUN_PATH, train_list, train_continuous, tasks_list, entropies

    return RUN_PATH, train_list, tasks_list, entropies


def preprocessing_icl_training(cfg
) -> tuple:
    """
    Loads raw losses and organizes excess risks grouped by target entropy level.

    Args:
        cfg: Experiment configuration object.
    """
    RUN_PATH, train_list, _, entropies = get_base_preprocessing(cfg)
    if hasattr(cfg, "entropies") and cfg.entropies is not None:
        min_entropy = min(entropies)
        max_entropy = max(entropies)
    else:
        _, ces = get_all_simulated_tasks_and_ces(cfg.dim, cfg.seed_entropy, getattr(cfg, 'entropy_num_tasks', 100_000))
        min_entropy = min(ces)
        max_entropy = max(ces)

    entropies_arr = np.array(entropies)
    entropies_2d = entropies_arr.reshape(-1, 1)

    model_data = {}
    for _, model_cfg in cfg.models.items():
        MODEL_PATH = RUN_PATH / f"{model_cfg.name} ({model_cfg.params})"
        losses_file = MODEL_PATH / "losses.parquet"
        if not losses_file.exists():
            continue
        losses = pd.read_parquet(losses_file)
        risks = losses.values - entropies_2d

        model_data[model_cfg.id] = {
            "name": model_cfg.name,
            "params": model_cfg.params,
            "color": model_cfg.color,
            "losses": losses,
            "risks": risks
        }

    entropy_config_and_results = {}
    for w_i, w in enumerate(entropies):
        rounded_w = round(float(w), 2)

        risk_title_str = fr"In-Context Learning Regret for $H(Y \mid X) = {float(w):.2f}$"
        loss_title_str = fr"In-Context Learning Loss for $H(Y \mid X) = {float(w):.2f}$"

        model_config_and_results = {
            "risk_title": risk_title_str,
            "loss_title": loss_title_str,
        }

        for _, model_cfg in cfg.models.items():
            if model_cfg.id not in model_data:
                continue
            losses = model_data[model_cfg.id]["losses"].iloc[w_i].to_numpy()
            risks = model_data[model_cfg.id]["risks"][w_i]

            model_config_and_results[model_cfg.id] = {
                "name": model_data[model_cfg.id]["name"],
                "params": model_data[model_cfg.id]["params"],
                "color": model_data[model_cfg.id]["color"],
                "losses": losses,
                "risks": risks,
            }

        entropy_config_and_results[rounded_w] = model_config_and_results

    return RUN_PATH, train_list, model_data, entropy_config_and_results


def preprocessing_entropy_icl_training(cfg
) -> tuple:
    """
    Preprocesses losses and excess risks structured by model checkpoint across all entropies.

    Args:
        cfg: Experiment configuration object.
    """
    RUN_PATH, train_list, _, entropies = get_base_preprocessing(cfg)
    entropies_arr = np.array(entropies)
    entropies_2d = entropies_arr.reshape(-1, 1)

    model_config_and_results = {}
    for _, model_cfg in cfg.models.items():
        MODEL_PATH = RUN_PATH / f"{model_cfg.name} ({model_cfg.params})"
        losses_file = MODEL_PATH / "losses.parquet"
        if not losses_file.exists():
            continue

        losses = pd.read_parquet(losses_file)
        risks = pd.DataFrame(losses.values - entropies_2d)

        risk_title = fr"In-Context Learning Regret for {model_cfg.name} ({model_cfg.params})"
        loss_title = fr"In-Context Learning Loss for {model_cfg.name} ({model_cfg.params})"

        model_config_and_results[model_cfg.id] = {
            "losses": losses,
            "risks": risks,
            "name": model_cfg.name,
            "params": model_cfg.params,
            "risk_title": risk_title,
            "loss_title": loss_title,
        }

    return RUN_PATH, train_list, entropies, model_config_and_results


def preprocessing_entropy_power_law_scaling(cfg
) -> tuple:
    """
    Fits power-law scaling curves and computes scaling derivatives across entropies per model.

    Args:
        cfg: Experiment configuration object.
    """
    RUN_PATH, train_list, train_continuous, _, entropies = get_base_preprocessing(cfg, include_continuous=True)
    entropies_2d = np.array(entropies).reshape(-1, 1)

    cached_weights = load_fitted_weights(RUN_PATH)
    updated_weights = False

    model_config_and_results = {}
    for _, model_cfg in cfg.models.items():
        MODEL_PATH = RUN_PATH / f"{model_cfg.name} ({model_cfg.params})"
        losses_file = MODEL_PATH / "losses.parquet"
        if not losses_file.exists():
            continue

        model_losses = pd.read_parquet(losses_file)
        model_risks = model_losses.values - entropies_2d
        title = fr"ICL Scaling Law of {model_cfg.name} ({model_cfg.params})"

        entropy_config_and_results = {}
        for w_i, w in enumerate(entropies):
            rounded_w = round(float(w), 2)
            risks = model_risks[w_i]

            cache_key = f"power_law_{model_cfg.id}_{rounded_w}"
            if cache_key in cached_weights:
                wf = cached_weights[cache_key]
                A, alpha, c = wf["A"], wf["alpha"], wf["c"]
            else:
                (A, alpha, c), _ = curve_fit(
                    power_law_fit_func, train_list, np.log10(risks),    
                    p0=[1.0, 0.5, 0.01], bounds=([1e-7, 0.01, 1e-7], [10.0, 5.0, 1.0]) 
                )
                cached_weights[cache_key] = {"A": float(A), "alpha": float(alpha), "c": float(c)}
                updated_weights = True

            y_real, y_ideal, y_real_der, y_ideal_der, y_minimax_der = get_power_law_fits(
                train_continuous, A, alpha, c
            )

            entropy_config_and_results[rounded_w] = {
                "risks": risks,
                "y_real": y_real, "y_real_der": y_real_der,
                "y_ideal": y_ideal, "y_ideal_der": y_ideal_der,
                "y_minimax_der": y_minimax_der,
                "A": A, "alpha": alpha, "c": c
            }

        model_config_and_results[model_cfg.id] = {
            "entropies_results": entropy_config_and_results,
            "name": model_cfg.name,
            "params": model_cfg.params,
            "title": title
        }

    if updated_weights:
        save_fitted_weights(RUN_PATH, cached_weights)

    return RUN_PATH, train_list, train_continuous, entropies, model_config_and_results


def preprocessing_power_law_scaling(cfg
) -> tuple:
    """
    Fits power-law scaling curves grouped by entropy level across all evaluated models.

    Args:
        cfg: Experiment configuration object.
    """
    RUN_PATH, train_list, train_continuous, _, entropies = get_base_preprocessing(cfg, include_continuous=True)
    if hasattr(cfg, "entropies") and cfg.entropies is not None:
        min_entropy = min(entropies)
        max_entropy = max(entropies)
    else:
        _, ces = get_all_simulated_tasks_and_ces(cfg.dim, cfg.seed_entropy, getattr(cfg, 'entropy_num_tasks', 100_000))
        min_entropy = min(ces)
        max_entropy = max(ces)

    entropies_arr = np.array(entropies)
    entropies_2d = entropies_arr.reshape(-1, 1)

    cached_weights = load_fitted_weights(RUN_PATH)
    updated_weights = False

    model_data = {}
    for _, model_cfg in cfg.models.items():
        MODEL_PATH = RUN_PATH / f"{model_cfg.name} ({model_cfg.params})"
        losses_file = MODEL_PATH / "losses.parquet"
        if not losses_file.exists():
            continue
        losses = pd.read_parquet(losses_file)
        model_risks = losses.values - entropies_2d

        model_data[model_cfg.id] = {
            "name": model_cfg.name,
            "params": model_cfg.params,
            "color": model_cfg.color,
            "risks": model_risks
        }

    entropy_config_and_results = {}
    for w_i, w in enumerate(entropies):
        rounded_w = round(float(w), 2)

        title_str = fr"ICL Scaling Law for $H(Y \mid X) = {float(w):.2f}$"

        model_config_and_results = {
            "title": title_str
        }

        for _, model_cfg in cfg.models.items():
            if model_cfg.id not in model_data:
                continue
            risks = model_data[model_cfg.id]["risks"][w_i]

            cache_key = f"power_law_{model_cfg.id}_{rounded_w}"
            if cache_key in cached_weights:
                wf = cached_weights[cache_key]
                A, alpha, c = wf["A"], wf["alpha"], wf["c"]
            else:
                (A, alpha, c), _ = curve_fit(
                    power_law_fit_func, train_list, np.log10(risks),    
                    p0=[1.0, 0.5, 0.01], bounds=([1e-7, 0.01, 1e-7], [10.0, 5.0, 1.0]) 
                )
                cached_weights[cache_key] = {"A": float(A), "alpha": float(alpha), "c": float(c)}
                updated_weights = True

            y_real, y_ideal, y_real_der, y_ideal_der, y_minimax_der = get_power_law_fits(
                train_continuous, A, alpha, c
            )

            model_config_and_results[model_cfg.id] = {
                "name": model_data[model_cfg.id]["name"],
                "params": model_data[model_cfg.id]["params"],
                "color": model_data[model_cfg.id]["color"],
                "risks": risks,
                "y_real": y_real, "y_real_der": y_real_der,
                "y_ideal": y_ideal, "y_ideal_der": y_ideal_der,
                "y_minimax_der": y_minimax_der,
                "A": A, "alpha": alpha, "c": c
            }

        entropy_config_and_results[rounded_w] = model_config_and_results 

    if updated_weights:
        save_fitted_weights(RUN_PATH, cached_weights)

    return RUN_PATH, train_list, train_continuous, entropy_config_and_results
