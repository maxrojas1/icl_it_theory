import argparse
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_model(model_id: str,
                   quant_bits: int,
                   save_dir: str | Path | None = None,
                   dataset: str = "open_platypus"
) -> str:
    """
    Quantizes a causal language model to WNA16 using compressed-tensors.

    Args:
        model_id: Identifier or local path of the base model.
        quant_bits: Bit-width for weight quantization.
        save_dir: Target destination path for compressed model artifacts.
        dataset: Calibration dataset identifier.
    """
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError:
        raise ImportError(
            "llmcompressor is required for model quantization. "
            "Please install it using: pip install llmcompressor"
        )

    clean_name = model_id.split("/")[-1]
    if save_dir is None:
        out_path = Path("models") / f"{clean_name}-quantized.w{quant_bits}a16"
    else:
        out_path = Path(save_dir)

    out_path.mkdir(parents=True, exist_ok=True)

    orig_hf_offline = os.environ.pop("HF_HUB_OFFLINE", None)
    orig_trans_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)

    try:
        print(f"Loading {model_id} for quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        scheme = f"W{quant_bits}A16"
        recipe = GPTQModifier(
            targets="Linear",
            scheme=scheme,
            ignore=["lm_head"]
        )

        print(f"Compressing {model_id} to {scheme}...")
        oneshot(
            model=model,
            dataset=dataset,
            recipe=recipe,
            max_seq_length=2048,
            num_calibration_samples=512
        )

        print(f"Saving compressed model to {out_path}...")
        model.save_pretrained(out_path, save_compressed=True)
        tokenizer.save_pretrained(out_path)
    finally:
        if orig_hf_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = orig_hf_offline
        if orig_trans_offline is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = orig_trans_offline

    print(f"Quantization complete: {out_path}")
    return str(out_path)


def get_or_quantize_model(model_id: str,
                          quant_bits: int | None = None,
                          save_dir: str | Path | None = None
) -> str:
    """
    Resolves model checkpoint, triggering quantization if bits are specified and not yet cached.

    Args:
        model_id: Base model identifier or path.
        quant_bits: Target weight quantization bits, or None to use base model.
        save_dir: Optional custom storage path for quantized weights.
    """
    if quant_bits is None or "quantized" in model_id.lower():
        return model_id

    clean_name = model_id.split("/")[-1]
    if save_dir is None:
        out_path = Path("models") / f"{clean_name}-quantized.w{quant_bits}a16"
    else:
        out_path = Path(save_dir)

    if out_path.exists() and (out_path / "config.json").exists():
        print(f"Using cached quantized model: {out_path}")
        return str(out_path)

    return quantize_model(
        model_id=model_id,
        quant_bits=quant_bits,
        save_dir=out_path
    )


def main() -> None:
    """
    CLI entrypoint to compress and store models given a quantization bit level.
    """
    parser = argparse.ArgumentParser(description="Model quantization utility")
    parser.add_argument("--model", "--model_id", type=str, required=True, dest="model_id", help="Hugging Face model ID or path")
    parser.add_argument("--quant_bits", "--bits", type=int, default=None, dest="quant_bits", help="Quantization bits (e.g. 8, 4)")
    parser.add_argument("--save_dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    resolved_path = get_or_quantize_model(
        model_id=args.model_id,
        quant_bits=args.quant_bits,
        save_dir=args.save_dir
    )
    print(f"Selected model: {resolved_path}")


if __name__ == "__main__":
    main()
