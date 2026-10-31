import pytest
import torch

from puzzel.data.text_dataset import TextSequenceDataset, encode_text_file
from puzzel.tokenizer import BOS_TOKEN_ID, EOS_TOKEN_ID, ByteTokenizer


def test_windows_are_shifted_by_one_token() -> None:
    dataset = TextSequenceDataset(torch.arange(8), sequence_length=4)

    assert len(dataset) == 4
    inputs, targets = dataset[0]
    assert inputs.tolist() == [0, 1, 2, 3]
    assert targets.tolist() == [1, 2, 3, 4]
    assert dataset[3][0].tolist() == [3, 4, 5, 6]


def test_dataset_rejects_short_or_non_1d_inputs() -> None:
    with pytest.raises(ValueError, match=r"at least sequence_length \+ 1"):
        TextSequenceDataset([1, 2, 3], sequence_length=3)
    with pytest.raises(ValueError, match="one-dimensional"):
        TextSequenceDataset(torch.ones(2, 3), sequence_length=2)


def test_file_encoder_adds_boundary_tokens(tmp_path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("hi", encoding="utf-8")

    tokens = encode_text_file(source, ByteTokenizer())

    assert tokens.tolist() == [BOS_TOKEN_ID, ord("h"), ord("i"), EOS_TOKEN_ID]
