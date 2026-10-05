def check_text_equal_decoded_tokens(text: str,
                                    decoded_tokens: str
) -> None:
    """
    Checks that the prompt text matches the decoded tokens.

    Args:
        text: Prompt text.
        decoded_tokens: Decoded tokenizer output.
    """
    if text != decoded_tokens:
        raise AssertionError(f"Decoded tokens do not match prompt text:\n  Prompt:  {text}\n  Decoded: {decoded_tokens}")
