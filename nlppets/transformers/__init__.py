try:
    import transformers
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "transformers module not installed. Please install it with nlppets[transformers]",
        name="transformers",
    ) from None
