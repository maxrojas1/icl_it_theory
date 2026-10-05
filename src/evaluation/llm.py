from vllm import LLM, SamplingParams


def load_llm_and_tokenizer(model_id: str,
                           gpu_ptge: float,
                           temperature: float = 0.0
) -> tuple:
    """
    Initializes the vLLM engine, tokenizer, and greedy sampling configuration.

    Args:
        model_id: HuggingFace model identifier or local checkpoint path.
        gpu_ptge: Fraction of GPU memory allocated to vLLM.
        temperature: Generation temperature (0.0 for deterministic evaluation).
    """
    llm = LLM(
        model=model_id,
        quantization=None,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_ptge,
        max_logprobs=20,
        max_model_len=32768
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=1,
        logprobs=20
    )

    return llm, tokenizer, sampling_params


def get_text_next_token_log_loss(next_token_results_dict: dict,
                                 label_id: int
) -> float:
    """
    Extracts the negative log-likelihood (nats) of the ground-truth label token.

    Args:
        next_token_results_dict: Dictionary of top-k token logprob structures from vLLM.
        label_id: Target ground-truth token ID.
    """
    if next_token_results_dict and label_id in next_token_results_dict:
        label_data = next_token_results_dict[label_id]
        logp = label_data.logprob
        return -logp 
    else:
        raise ValueError("The Label ID is not on the top-k Tokens, please increase k.")


def get_batch_next_token_log_loss(request_outputs: list,
                                  labels_id: list[int]
) -> float:
    """
    Computes mean log-loss across a batch of validation query completions.

    Args:
        request_outputs: List of vLLM RequestOutput objects.
        labels_id: Ground-truth token IDs for each corresponding query.
    """
    loss_per_val = []

    for text_request_output, label_id in zip(request_outputs, labels_id):
        text_completion_output = text_request_output.outputs[0]
        text_next_token_results_dict = text_completion_output.logprobs[0]
        text_log_loss = get_text_next_token_log_loss(text_next_token_results_dict,
                                                     label_id)
        loss_per_val.append(text_log_loss)

    batch_loss = sum(loss_per_val) / len(loss_per_val)
    return batch_loss
