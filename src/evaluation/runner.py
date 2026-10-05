import gc
import time
from datetime import datetime, timedelta
from pathlib import Path

from jaxtyping import Float
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm.distributed.parallel_state import destroy_model_parallel

from src.theory.dgp import (
    get_context,
    get_query,
    get_text_logistic_regression,
)
from src.theory.entropy import get_tasks_by_entropy
from src.evaluation.llm import (
    get_batch_next_token_log_loss,
    load_llm_and_tokenizer,
)
from src.utils.checks import check_text_equal_decoded_tokens
from src.utils.config import save_config
from src.utils.logging import log_exp, log_mem

# -------------------------
# B: batch size
# C: chunk size
# T: maximum token length
# V: vocabulary size
# TR: train size
# VA: validation size
# CO: context size
# -------------------------


def save_results(cfg,
                 model_cfg,
                 losses_by_entropy,
                 start_time: float
) -> None:
    """
    Saves the experiment results dataframe, runtime, and configuration.

    Args:
        cfg (DictConfig): OmegaConf configuration object.
        model_cfg (Any): Model configuration object containing name and params.
        losses_by_entropy: Evaluated log loss matrix.
        start_time (float): Epoch timestamp when the experiment started.
    """
    start_datetime = datetime.fromtimestamp(start_time)
    timestamp_str = start_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    if not hasattr(cfg, "run") or not cfg.run:
        cfg.run = str(timestamp_str)

    RUN_PATH = Path("results") / "runs" / f"run_{cfg.run}"
    RUN_PATH.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(RUN_PATH)

    MODEL_PATH = RUN_PATH / f"{model_cfg.name} ({model_cfg.params})"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    df_losses = pd.DataFrame(losses_by_entropy)
    df_losses.to_parquet(MODEL_PATH / "losses.parquet")

    save_config(cfg, RUN_PATH)
    print(f"{model_cfg.name} ({model_cfg.params}) Data and config successfully saved.")


def process_one_model(model_cfg,
                      cfg,
                      X_train: Float[torch.Tensor, "CO TR D"],
                      X_val: Float[torch.Tensor, "VA D"],
                      start_time: float
) -> list:
    """
    Evaluates in-context learning losses for a single model across tasks.

    Args:
        model_cfg (Any): Model configuration object containing name, id, and params.
        cfg (DictConfig): OmegaConf configuration object.
        X_train (Tensor): In-context training examples tensor.
        X_val (Tensor): Validation queries tensor.
        start_time (float): Epoch timestamp when the experiment started (e.g., time.time()).

    Returns:
        list: losses_by_entropy.
    """
    model_start_time = time.time()
    
    log_mem("Before loading model")

    llm, tokenizer, sampling_params = load_llm_and_tokenizer(model_cfg.id,
                                                             cfg.gpu_ptge)
    log_mem("After loading model")

    target_entropies = getattr(cfg, 'entropies', None)
    num_entropy_levels = getattr(cfg, 'num_entropy_levels', None)

    requested_tasks, selected_entropies = get_tasks_by_entropy(
        dim=cfg.dim,
        seed_entropy=cfg.seed_entropy,
        entropies=target_entropies,
        num_entropy_levels=num_entropy_levels if num_entropy_levels is not None else 10,
        num_task=getattr(cfg, 'entropy_num_tasks', 100_000),
        plot=True
    )
    cfg.entropies = selected_entropies

    list_tasks = range(len(requested_tasks))
    if getattr(cfg, "old_geom", False):
        list_train = np.unique(np.geomspace(1, cfg.max_train, num=cfg.num_train, dtype=int))
    else:
        list_train = np.unique(np.round(np.geomspace(1, cfg.max_train, num=cfg.num_train)).astype(int))
    list_contexts = range(cfg.num_contexts)
    list_val = range(cfg.num_val)

    losses_by_entropy = np.empty((len(list_tasks), len(list_train))) # EN, TR
    losses_by_context_and_size = np.empty((len(list_contexts), len(list_train))) # CO, TR

    first_context_done = False
    context_start_time = None

    for w in tqdm(list_tasks, desc="   Entropies", leave=False):

        W = requested_tasks[w] # (D, 1)

        for c in tqdm(list_contexts, desc="   Contexts", leave=False):

            if not first_context_done:
                context_start_time = time.time()

            X_train_context = X_train[c, :, :] # (TR, D)

            for (t_i, t) in enumerate(tqdm(list_train, desc="   Training Sizes", leave=False)):

                X_train_points = X_train_context[:t, :] # (t, D)

                tokenized_dataset = []
                label_ids = []

                for v in list_val:

                    X_val_point = X_val[v] # (1, D)

                    text, label = get_text_logistic_regression(W,
                                                            X_train_points,
                                                            X_val_point,
                                                            cfg.seed_val * c * t * v,
                                                            cfg.seed_train)

                    token_ids = tokenizer.encode(text, add_special_tokens=True)
                    label_id = tokenizer.encode(label, add_special_tokens=False)[-1]

                    check_text_equal_decoded_tokens(text, tokenizer.decode(token_ids, skip_special_tokens=True))
                    
                    tokenized_dataset.append(token_ids)
                    label_ids.append(label_id)

                formatted_prompts = [{"prompt_token_ids": token_ids} for token_ids in tokenized_dataset]
                request_output = llm.generate(formatted_prompts,
                                            sampling_params=sampling_params,
                                            use_tqdm=False)

                batch_loss = get_batch_next_token_log_loss(request_output,
                                                           label_ids)

                losses_by_context_and_size[c,t_i] = batch_loss

            if not first_context_done:
                first_context_done = True
                context_elapsed = time.time() - context_start_time
                total_contexts = len(list_tasks) * len(list_contexts)
                remaining_contexts = max(0, total_contexts - 1)
                remaining_seconds = remaining_contexts * context_elapsed

                elapsed_model_seconds = time.time() - model_start_time
                expected_total_seconds = elapsed_model_seconds + remaining_seconds

                expected_runtime_str = str(timedelta(seconds=int(expected_total_seconds)))
                expected_end_datetime = datetime.now() + timedelta(seconds=int(remaining_seconds))
                expected_end_str = expected_end_datetime.strftime("%Y-%m-%d %H:%M:%S")

                print(f"{model_cfg.name} ({model_cfg.params}) Expected Runtime: {expected_runtime_str} | Expected End Date: {expected_end_str}")

        losses_by_size = np.mean(losses_by_context_and_size, axis=0).tolist() # (TR)

        assert len(losses_by_size) == len(list_train), f"Got {len(losses_by_size)} trained model losses, expected {len(list_train)}"

        losses_by_entropy[w] = losses_by_size

        gc.collect()
        torch.cuda.empty_cache()

        log_mem("After inference")

    destroy_model_parallel()
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    model_end_time = time.time()
    model_elapsed_seconds = model_end_time - model_start_time

    model_runtime_str = str(timedelta(seconds=int(model_elapsed_seconds)))
    model_cfg.run_time = model_runtime_str

    print(f"{model_cfg.name} ({model_cfg.params}) Runtime: {model_runtime_str}")

    save_results(cfg,
                 model_cfg,
                 losses_by_entropy,
                 start_time)

    return losses_by_entropy


def run_icl_training(cfg
) -> None:
    """
    Runs in-context learning evaluation on logistic regression tasks across models.

    Args:
        cfg (DictConfig): OmegaConf configuration object.
    """     
    start_time = time.time()
    timestamp_str = datetime.fromtimestamp(start_time).strftime("%Y_%m_%d_%H_%M_%S")
    if not hasattr(cfg, "run") or not cfg.run:
        cfg.run = str(timestamp_str)
    RUN_PATH = Path("results") / "runs" / f"run_{cfg.run}"
    RUN_PATH.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(RUN_PATH)

    kwargs = {
        "Experiment": getattr(cfg, "exp", "icl_training"),
        "Dimension": cfg.dim,
        "Maximum number of training samples": cfg.max_train,
        "Number of training sizes": cfg.num_train,
        "Number of contexts": cfg.num_contexts,
        "Number of validation samples": cfg.num_val
    }

    log_exp(**kwargs)    

    X_train = get_context(cfg.dim,
                          cfg.seed_train,
                          cfg.max_train,
                          cfg.num_contexts).cpu() # (CO, TR, D)
    
    X_val = get_query(cfg.dim,
                      cfg.seed_val,
                      cfg.num_val).cpu() # (VA, D)

    for _, (_, model_cfg) in enumerate(cfg.models.items()):
        
        print(f"\n{model_cfg.name} ({model_cfg.params}):")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        torch.cuda.set_per_process_memory_fraction(cfg.gpu_ptge, device=device_id)
        destroy_model_parallel()
        
        try:
            process_one_model(model_cfg,
                              cfg,
                              X_train,
                              X_val,
                              start_time)
            print(f"{model_cfg.name} ({model_cfg.params}) completed.")
        except (torch.OutOfMemoryError, RuntimeError) as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda" in err_str or "failed core proc" in err_str:
                print(f"Skipping {model_cfg.name} ({model_cfg.params}): Insufficient GPU memory.")
                model_cfg.skipped = True
                destroy_model_parallel()
                gc.collect()
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    end_time = time.time()
    elapsed_seconds = end_time - start_time

    runtime_str = str(timedelta(seconds=int(elapsed_seconds)))
    cfg.run_time = runtime_str

    if hasattr(cfg, "run_dir") and cfg.run_dir:
        final_run_path = Path(cfg.run_dir)
    elif hasattr(cfg, "run") and cfg.run:
        final_run_path = Path("results") / "runs" / f"run_{cfg.run}"
    else:
        final_run_path = Path("results") / "runs" / f"run_{datetime.fromtimestamp(start_time).strftime('%Y_%m_%d_%H_%M_%S')}"
    
    final_run_path.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(final_run_path)
    save_config(cfg, final_run_path)
    
    print("\nAll models processed successfully!")
    print(f"Total Experiment Runtime: {runtime_str}")


run_evaluation = run_icl_training
