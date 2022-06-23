try:
    import transformers
except ImportError:
    raise ImportError(
        "transformers module not installed. Please install it with nlppets[transformers]"
    ) from None
