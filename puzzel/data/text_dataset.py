"""Local UTF-8 text loading and next-token sliding-window dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from puzzel.tokenizer import TextTokenizer


def encode_text(
    text: str,
    tokenizer: TextTokenizer,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Tensor:
    """Tokenize text into a one-dimensional CPU tensor."""
    token_ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    return torch.tensor(token_ids, dtype=torch.long)


def encode_text_file(
    path: str | Path,
    tokenizer: TextTokenizer,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Tensor:
    """Read one UTF-8 text file and return its token IDs as a CPU tensor.

    This initial implementation intentionally favors clarity over streaming and
    reads the whole file into memory. Keep input files small on constrained
    machines; a sharded/memory-mapped reader is a future data-pipeline upgrade.
    """
    text = Path(path).read_text(encoding="utf-8")
    return encode_text(text, tokenizer, add_bos=add_bos, add_eos=add_eos)


class TextSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Fixed-length input/next-token pairs from one flat token sequence."""

    def __init__(self, token_ids: Sequence[int] | Tensor, sequence_length: int) -> None:
        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
            raise ValueError("sequence_length must be an integer")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        tokens = torch.as_tensor(token_ids, dtype=torch.long)
        if tokens.dim() != 1:
            raise ValueError("token_ids must be a one-dimensional sequence")
        if tokens.numel() < sequence_length + 1:
            raise ValueError(
                "the token sequence must contain at least sequence_length + 1 "
                "tokens to form one next-token example"
            )
        self.token_ids = tokens.contiguous()
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.token_ids.numel() - self.sequence_length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index
        end = start + self.sequence_length
        inputs = self.token_ids[start:end]
        targets = self.token_ids[start + 1 : end + 1]
        return inputs, targets
