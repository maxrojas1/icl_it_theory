import functools
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]




@functools.lru_cache(maxsize=4)
def get_all_simulated_tasks_and_ces(dim: int,
                                    seed_entropy: int,
                                    num_task: int
) -> tuple[list[Float[torch.Tensor, "D 1"]], list[float]]:
    """
    Simulates a pool of random tasks and computes their conditional entropies with caching.

    Args:
        dim: Task feature dimension.
        seed_entropy: Random seed for sampling task weight vectors.
        num_task: Total number of random tasks to generate.
    """
    assert num_task > 0, "num_task must be positive"
    cache_dir = ROOT_DIR / "results" / ".cache"
    cache_file = cache_dir / f"simulated_tasks_d{dim}_s{seed_entropy}_n{num_task}.pt"

    if cache_file.exists():
        try:
            data = torch.load(cache_file, weights_only=False)
            tasks, ces = data["tasks"], data["ces"]
            assert len(tasks) == num_task and len(ces) == num_task
            return tasks, ces
        except Exception:
            pass

    print(f"Simulating {num_task} tasks (saving to cache for future runs)...")
    torch.manual_seed(seed_entropy)
    tasks = []
    ces = []
    batch_size = 5000
    num_samples = 100_000
    X = torch.randn(num_samples, dim)
    X_rounded = torch.round(X * 10.0) / 10.0

    for i in tqdm(range(0, num_task, batch_size), desc="Simulating tasks (caching for future runs)"):
        current_batch = min(batch_size, num_task - i)
        W_batch = torch.randn(dim, current_batch) / math.sqrt(dim)
        logits = torch.matmul(X_rounded, W_batch)
        probs = torch.sigmoid(logits)
        h = F.binary_cross_entropy(probs, probs, reduction='none')
        batch_ces = h.mean(dim=0).tolist()
        for j in range(current_batch):
            tasks.append(W_batch[:, j:j+1])
            ces.append(batch_ces[j])

    assert len(tasks) == num_task and len(ces) == num_task, f"Expected {num_task} tasks, got {len(tasks)}"

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"tasks": tasks, "ces": ces}, cache_file)
        print(f"Tasks cached successfully to {cache_file}")
    except Exception:
        pass

    return tasks, ces


def get_tasks_by_entropy(dim: int,
                         seed_entropy: int,
                         entropies: list[float] = None,
                         num_entropy_levels: int = None,
                         num_task: int = 100_000,
                         plot: bool = False
) -> tuple[list[Float[torch.Tensor, "D 1"]], list[float]]:
    """
    Selects tasks from a simulated pool that match target entropy values or intervals.

    Args:
        dim: Feature dimension.
        seed_entropy: Random seed for task generation.
        entropies: Optional list of explicit target conditional entropies.
        num_entropy_levels: Number of equidistant entropy levels if entropies is not given.
        num_task: Pool size of simulated tasks to search through.
        plot: Whether to generate a diagnostic plot of the entropy spectrum.
    """
    tasks, ces = get_all_simulated_tasks_and_ces(dim, seed_entropy, num_task)
    min_ce, max_ce = min(ces), max(ces)

    if entropies is not None and len(entropies) > 0:
        target_entropies = [float(e) for e in entropies]
        check_existence = True
    else:
        check_existence = False
        n = num_entropy_levels if num_entropy_levels is not None else 10
        if n == 1:
            target_entropies = [(min_ce + max_ce) / 2.0]
        else:
            target_entropies = np.linspace(min_ce, max_ce, n).tolist()

    selected_tasks = []
    selected_entropies = []

    for target in target_entropies:
        closest_idx = min(range(len(ces)), key=lambda i: abs(ces[i] - target))
        closest_task = tasks[closest_idx]
        closest_ce = ces[closest_idx]
        val_rounded = round(float(closest_ce), 2)

        if check_existence and round(float(target), 2) != val_rounded:
            print(f"Warning: Entropy {target} does not exist, approximating to closest entropy: {val_rounded}")

        selected_tasks.append(closest_task)
        selected_entropies.append(val_rounded)

    assert len(selected_tasks) > 0, "No tasks selected"
    assert len(selected_tasks) == len(selected_entropies), "Task and entropy count mismatch"

    if plot:
        from src.analysis.single_run import plot_conditional_entropy
        plot_conditional_entropy(ces, selected_entropies)

    return selected_tasks, selected_entropies
