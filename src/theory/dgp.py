import numpy as np
import torch
from jaxtyping import Float


def DGP_1D(x: np.ndarray,
           weight: float,
           bias: float
) -> np.ndarray:
    """
    Evaluates 1D logistic sigmoid probabilities.

    Args:
        x: Input coordinates.
        weight: Slope/weight parameter.
        bias: Intercept/bias parameter.
    """
    return 1 / (1 + np.exp(-(bias + weight * x)))


def calculate_conditional_entropy(weights: np.ndarray,
                                  biases: np.ndarray,
                                  x_samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes conditional entropy over 1D weight and bias slices and grids.

    Args:
        weights: Array of candidate weight values.
        biases: Array of candidate bias values.
        x_samples: Monte Carlo samples from the input distribution.
    """
    entropies_w = []
    for w in weights:
        p = DGP_1D(x_samples, w, 0.0)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        h = -p * np.log(p) - (1 - p) * np.log(1 - p)
        entropies_w.append(np.mean(h))

    entropies_b = []
    for b in biases:
        p = DGP_1D(x_samples, 1.0, b)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        h = -p * np.log(p) - (1 - p) * np.log(1 - p)
        entropies_b.append(np.mean(h))

    W_grid, B_grid = np.meshgrid(weights, biases)
    H_grid = np.zeros_like(W_grid)
    for i in range(W_grid.shape[0]):
        for j in range(W_grid.shape[1]):
            p = DGP_1D(x_samples, W_grid[i, j], B_grid[i, j])
            p = np.clip(p, 1e-12, 1 - 1e-12)
            h = -p * np.log(p) - (1 - p) * np.log(1 - p)
            H_grid[i, j] = np.mean(h)

    return np.array(entropies_w), np.array(entropies_b), W_grid, B_grid, H_grid


def tensor_to_vec(tensor: Float[torch.Tensor, "1 D"]
) -> str:
    """
    Formats a coordinate vector into a comma-separated string for prompt text.

    Args:
        tensor: 1D point coordinate tensor.
    """
    vec_list = tensor.tolist()
    n_list = len(vec_list)
    vec = ""
    for i in range(n_list):
        vec += f"{vec_list[i]:.2f}"
        if i != n_list - 1:
            vec += ", "
    vec += ", "

    return vec


def get_context(dim: int,
                seed_context: int,
                num_train: int,
                num_contexts: int
) -> Float[torch.Tensor, "CO TR D"]:
    """
    Generates synthetic in-context training examples.

    Args:
        dim: Feature dimension.
        seed_context: Random seed for context coordinate sampling.
        num_train: Number of examples per context.
        num_contexts: Number of independent in-context sequences.
    """
    torch.manual_seed(seed_context)
    X_train = torch.randn(num_contexts, num_train, dim)
    X_train_rounded = torch.round(X_train * 10) / 10.0

    return X_train_rounded


def get_query(dim: int,
              seed_query: int,
              num_val: int
) -> Float[torch.Tensor, "VA D"]:
    """
    Generates validation query points for evaluation.

    Args:
        dim: Feature dimension.
        seed_query: Random seed for query coordinates.
        num_val: Number of validation query points.
    """
    torch.manual_seed(seed_query)
    X_val = torch.randn(num_val, dim)
    X_val_rounded = torch.round(X_val * 10) / 10.0

    return X_val_rounded


def get_single_point_text(W: Float[torch.Tensor, "1 D"],
                          seed: int,
                          X_point: Float[torch.Tensor, "1 D"]
) -> tuple[str, str]:
    """
    Constructs prompt text and binary label for an individual input point.

    Args:
        W: Task weight vector.
        seed: Random seed for Bernoulli label sampling.
        X_point: Feature coordinates of the point.
    """
    gen = torch.Generator().manual_seed(int(seed))
    
    logits = torch.matmul(X_point, W).squeeze(0)
    prob_of_1 = torch.sigmoid(logits)
    Y = torch.bernoulli(prob_of_1, generator=gen).long()

    inp = tensor_to_vec(X_point)
    output = Y.item()

    prompt = "Input: " + inp + "Output: "
    label = f"{output}"

    return prompt, label


def get_text_logistic_regression(W: Float[torch.Tensor, "1 D"],
                                 X_train: Float[torch.Tensor, "t D"],
                                 X_val_point: Float[torch.Tensor, "1 D"],
                                 seed_val: int,
                                 seed_train: int
) -> tuple[str, str]:
    """
    Constructs the full in-context prompt with demonstration examples and an unlabeled query.

    Args:
        W: Task weight vector.
        X_train: In-context training example coordinates.
        X_val_point: Validation query coordinate.
        seed_val: Random seed for query label sampling.
        seed_train: Base seed for in-context example labels.
    """
    num_train = X_train.size(0)
    text = ""
    for i in range(num_train):
        X_train_point = X_train[i]
        example_seed = seed_train * 1_000_000 + i
        example_prompt, example_label = get_single_point_text(W=W,
                                                              seed=example_seed,
                                                              X_point=X_train_point)
        if i != 0:
            text += ", "
        text += example_prompt + example_label

    query_seed = seed_val * 10_000 + 9999
    query_prompt, query_label = get_single_point_text(W=W,
                                                      seed=query_seed,
                                                      X_point=X_val_point)
    text += ", " + query_prompt 

    return text, query_label
