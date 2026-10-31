import pytest

from puzzel.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    BPETokenizer,
    ByteTokenizer,
)


def test_utf8_byte_round_trip_with_optional_boundaries() -> None:
    tokenizer = ByteTokenizer()
    text = "Puzzel — नमस्ते 🌍\n"

    ids = tokenizer.encode(text, add_bos=True, add_eos=True)

    assert ids[0] == BOS_TOKEN_ID
    assert ids[-1] == EOS_TOKEN_ID
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(ids, skip_special_tokens=False) == f"<|bos|>{text}<|eos|>"


def test_literal_special_token_spelling_is_plain_text() -> None:
    tokenizer = ByteTokenizer()
    text = "literal <|bos|> marker"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_tokenizer_definition_round_trips(tmp_path) -> None:
    path = tmp_path / "byte-tokenizer.json"
    ByteTokenizer().save(path)

    restored = ByteTokenizer.load(path)

    assert restored.vocab_size == 260
    assert restored.decode(restored.encode("hello")) == "hello"


def test_bpe_trains_merges_and_round_trips_utf8(tmp_path) -> None:
    text = ("banana bandana — नमस्ते! " * 8)
    tokenizer = BPETokenizer.train(
        text,
        target_vocab_size=300,
        min_frequency=2,
    )

    encoded_without_specials = tokenizer.encode(text)
    encoded_with_specials = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert 260 < tokenizer.vocab_size <= 300
    assert len(encoded_without_specials) < len(text.encode("utf-8"))
    assert tokenizer.decode(encoded_with_specials) == text
    assert encoded_with_specials[0] == BOS_TOKEN_ID
    assert encoded_with_specials[-1] == EOS_TOKEN_ID

    path = tmp_path / "learned-bpe.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.to_dict() == tokenizer.to_dict()
    assert restored.decode(restored.encode(text)) == text


def test_bpe_training_is_deterministic_and_validates_target() -> None:
    text = "abracadabra " * 5
    first = BPETokenizer.train(text, target_vocab_size=280, min_frequency=2)
    second = BPETokenizer.train(text, target_vocab_size=280, min_frequency=2)

    assert first.to_dict() == second.to_dict()
    with pytest.raises(ValueError, match="target_vocab_size"):
        BPETokenizer.train(text, target_vocab_size=259)


def test_disk_backed_bpe_training_matches_in_memory_training(tmp_path) -> None:
    documents = ["local corpus one " * 4, "local corpus two — नमस्ते " * 3]
    expected = BPETokenizer.train(
        documents, target_vocab_size=285, min_frequency=2
    )
    actual = BPETokenizer.train_streaming(
        lambda: iter(documents),
        target_vocab_size=285,
        min_frequency=2,
        temp_dir=tmp_path,
    )

    assert actual.to_dict() == expected.to_dict()
    assert actual.decode(actual.encode(documents[1])) == documents[1]


def test_invalid_byte_token_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        ByteTokenizer().decode([260])
