"""Local text ingestion, dataset preparation, and next-token datasets."""

from puzzel.data.pipeline import DatasetPreparationError, prepare_dataset
from puzzel.data.prepared import (
    DatasetVerificationError,
    PreparedSplitDataset,
    verify_prepared_dataset,
)
from puzzel.data.text_dataset import TextSequenceDataset, encode_text, encode_text_file

__all__ = [
    "DatasetPreparationError",
    "DatasetVerificationError",
    "PreparedSplitDataset",
    "TextSequenceDataset",
    "encode_text",
    "encode_text_file",
    "prepare_dataset",
    "verify_prepared_dataset",
]
