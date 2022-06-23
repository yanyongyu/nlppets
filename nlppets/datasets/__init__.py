try:
    import datasets
except ImportError:
    raise ImportError(
        "datasets module not installed. Please install it with nlppets[datasets]"
    ) from None

from .builders import ChineseRawTextDatasetBuilder as ChineseRawTextDatasetBuilder
