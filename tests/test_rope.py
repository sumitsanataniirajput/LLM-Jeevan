"""Unit tests for Rotary Position Embeddings (RoPE)."""

import torch

from puzzel.model.rope import RotaryPositionEmbeddings, apply_rotary_pos_emb


def test_rope_output_shape_matches_input() -> None:
    rope = RotaryPositionEmbeddings(head_dim=32)
    query = torch.randn(2, 4, 10, 32)
    key = torch.randn(2, 4, 10, 32)
    q_rot, k_rot = rope(query, key)
    assert q_rot.shape == query.shape
    assert k_rot.shape == key.shape


def test_rope_preserves_vector_norm() -> None:
    rope = RotaryPositionEmbeddings(head_dim=32)
    query = torch.randn(2, 4, 10, 32)
    key = torch.randn(2, 4, 10, 32)
    q_rot, k_rot = rope(query, key)
    # RoPE is a rotation, so it preserves the L2 norm of each vector.
    query_norm = torch.norm(query, dim=-1)
    q_rot_norm = torch.norm(q_rot, dim=-1)
    key_norm = torch.norm(key, dim=-1)
    k_rot_norm = torch.norm(k_rot, dim=-1)
    torch.testing.assert_close(query_norm, q_rot_norm, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(key_norm, k_rot_norm, atol=1e-5, rtol=1e-5)


def test_rope_head_dim_must_be_even() -> None:
    try:
        RotaryPositionEmbeddings(head_dim=31)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_rope_different_positions_produce_different_rotations() -> None:
    rope = RotaryPositionEmbeddings(head_dim=32)
    query = torch.randn(1, 1, 5, 32)
    key = torch.randn(1, 1, 5, 32)
    q_rot, _ = rope(query, key)
    # Position 0 and position 4 should have different rotated values.
    assert not torch.equal(q_rot[0, 0, 0], q_rot[0, 0, 4])


def test_rope_apply_rotary_pos_emb_functional() -> None:
    query = torch.randn(2, 4, 10, 32)
    key = torch.randn(2, 4, 10, 32)
    q_rot, k_rot = apply_rotary_pos_emb(query, key, head_dim=32)
    assert q_rot.shape == query.shape
    assert k_rot.shape == key.shape


def test_rope_cache_extension() -> None:
    rope = RotaryPositionEmbeddings(head_dim=16, max_position_embeddings=8)
    # First call within cache.
    q1 = torch.randn(1, 1, 5, 16)
    k1 = torch.randn(1, 1, 5, 16)
    rope(q1, k1)
    assert rope._cache_max_position == 8
    # Second call exceeding cache should extend it.
    q2 = torch.randn(1, 1, 20, 16)
    k2 = torch.randn(1, 1, 20, 16)
    rope(q2, k2)
    assert rope._cache_max_position >= 20


def test_rope_deterministic_same_input() -> None:
    rope = RotaryPositionEmbeddings(head_dim=32)
    query = torch.randn(2, 4, 10, 32)
    key = torch.randn(2, 4, 10, 32)
    q1, k1 = rope(query.clone(), key.clone())
    q2, k2 = rope(query.clone(), key.clone())
    torch.testing.assert_close(q1, q2)
    torch.testing.assert_close(k1, k2)