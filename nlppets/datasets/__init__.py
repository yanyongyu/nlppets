try:
    import datasets
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "datasets module not installed. Please install it with nlppets[datasets]",
        name="datasets",
    ) from None

from .builders import RawTextDatasetBuilder as RawTextDatasetBuilder
