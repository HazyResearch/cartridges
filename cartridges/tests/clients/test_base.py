"""Test script for base client flatten and reconstruct functionality"""

import pytest
import numpy as np
from cartridges.clients.base import TopLogprobs, FlatTopLogprobs


class TestTopLogprobsFlattenReconstruct:
    """Test suite for TopLogprobs flatten and reconstruct methods."""

    def test_flatten_basic_functionality(self):
        """Test basic flatten functionality with simple logprobs."""
        # Create simple test data: 3 tokens, 4 top logprobs each
        logprobs = np.array([
            [-0.1, -0.5, -1.0, -2.0],  # token 0: high prob first
            [-0.2, -0.3, -1.5, -3.0],  # token 1: high prob first two
            [-0.05, -0.8, -1.2, -2.5] # token 2: very high prob first
        ])
        token_ids = np.array([
            [10, 20, 30, 40],
            [15, 25, 35, 45], 
            [12, 22, 32, 42]
        ])
        
        top_logprobs = TopLogprobs(logprobs=logprobs, token_ids=token_ids)
        flat = top_logprobs.flatten(threshold=0.8)
        
        assert isinstance(flat, FlatTopLogprobs)
        assert flat.shape == (3, 4)
        assert len(flat.token_idx) == len(flat.token_id) == len(flat.logprobs)

    def test_reconstruct_basic_functionality(self):
        """Test basic reconstruct functionality."""
        # Create test data
        logprobs = np.array([
            [-0.1, -0.5, -1.0, -2.0],
            [-0.2, -0.3, -1.5, -3.0],
            [-0.05, -0.8, -1.2, -2.5]
        ])
        token_ids = np.array([
            [10, 20, 30, 40],
            [15, 25, 35, 45],
            [12, 22, 32, 42]
        ])
        
        top_logprobs = TopLogprobs(logprobs=logprobs, token_ids=token_ids)
        flat = top_logprobs.flatten(threshold=0.99)
        reconstructed = flat.reconstruct()
        
        assert isinstance(reconstructed, TopLogprobs)
        assert reconstructed.logprobs.shape == (3, 4)
        assert reconstructed.token_ids.shape == (3, 4)

    def test_flatten_reconstruct_roundtrip_identity(self):
        """Test that flatten->reconstruct preserves the original data structure."""
        # Create test data with varying probabilities
        logprobs = np.array([
            [-0.01, -0.69, -2.3, -4.6],   # ~0.99, ~0.5, ~0.1, ~0.01
            [-0.11, -0.22, -0.36, -1.6],  # ~0.90, ~0.80, ~0.70, ~0.20
            [-0.05, -1.2, -1.6, -2.3]     # ~0.95, ~0.30, ~0.20, ~0.10
        ])
        token_ids = np.array([
            [100, 200, 300, 400],
            [150, 250, 350, 450],
            [120, 220, 320, 420]
        ])
        
        original = TopLogprobs(logprobs=logprobs, token_ids=token_ids)
        
        # Test with threshold that should keep most data
        flat = original.flatten(threshold=0.99)
        reconstructed = flat.reconstruct()
        
        # Check that the structure is preserved
        assert reconstructed.logprobs.shape == original.logprobs.shape
        assert reconstructed.token_ids.shape == original.token_ids.shape
        
        # Check that kept entries match exactly
        T, K = original.logprobs.shape
        for row in range(T):
            # Find how many entries were kept for this row
            row_mask = flat.token_idx == row
            n_kept = row_mask.sum()
            
            if n_kept > 0:
                # Check that the first n_kept entries match
                np.testing.assert_array_equal(
                    reconstructed.logprobs[row, :n_kept],
                    original.logprobs[row, :n_kept]
                )
                np.testing.assert_array_equal(
                    reconstructed.token_ids[row, :n_kept],
                    original.token_ids[row, :n_kept]
                )
                
                # Check that remaining entries are filled with defaults
                if n_kept < K:
                    assert np.all(reconstructed.logprobs[row, n_kept:] == -np.inf)
                    assert np.all(reconstructed.token_ids[row, n_kept:] == -1)

    def test_flatten_threshold_behavior(self):
        """Test that different thresholds produce different amounts of data."""
        # Create logprobs with known cumulative probabilities
        # Row 0: probs ≈ [0.9, 0.09, 0.009, 0.001] -> cumsum ≈ [0.9, 0.99, 0.999, 1.0]
        logprobs = np.array([
            [np.log(0.9), np.log(0.09), np.log(0.009), np.log(0.001)],
            [np.log(0.8), np.log(0.15), np.log(0.04), np.log(0.01)],
        ])
        token_ids = np.array([
            [10, 20, 30, 40],
            [15, 25, 35, 45]
        ])
        
        top_logprobs = TopLogprobs(logprobs=logprobs, token_ids=token_ids)
        
        # Test different thresholds
        flat_50 = top_logprobs.flatten(threshold=0.5)
        flat_90 = top_logprobs.flatten(threshold=0.9)
        flat_99 = top_logprobs.flatten(threshold=0.99)
        
        # Higher thresholds should keep more data
        assert len(flat_50.logprobs) <= len(flat_90.logprobs) <= len(flat_99.logprobs)

    def test_flatten_keeps_all_entries_when_topk_mass_is_incomplete(self):
        """A truncated top-k row may not contain the requested probability mass."""
        logprobs = np.log(np.array([
            [0.30, 0.20, 0.10],
            [0.50, 0.25, 0.20],
        ]))
        token_ids = np.array([
            [10, 20, 30],
            [40, 50, 60],
        ])

        flat = TopLogprobs(logprobs=logprobs, token_ids=token_ids).flatten(
            threshold=0.99
        )

        np.testing.assert_array_equal(flat.token_id, token_ids.ravel())
        np.testing.assert_array_equal(flat.token_idx, [0, 0, 0, 1, 1, 1])

    def test_flatten_validation_errors(self):
        """Test that flatten raises appropriate validation errors."""
        logprobs = np.array([[-0.1, -0.5], [-0.2, -0.3]])
        token_ids = np.array([[10, 20], [15, 25]])
        top_logprobs = TopLogprobs(logprobs=logprobs, token_ids=token_ids)
        
        # Test invalid threshold values
        with pytest.raises(ValueError, match="threshold must be in"):
            top_logprobs.flatten(threshold=0.0)
        
        with pytest.raises(ValueError, match="threshold must be in"):
            top_logprobs.flatten(threshold=1.5)
        
        # Test mismatched shapes
        bad_logprobs = TopLogprobs(
            logprobs=np.array([[-0.1, -0.5]]),
            token_ids=np.array([[10, 20], [15, 25]])
        )
        with pytest.raises(ValueError, match="Shapes of logprobs and token_ids differ"):
            bad_logprobs.flatten()
        
        # Test wrong dimensions
        bad_dims = TopLogprobs(
            logprobs=np.array([-0.1, -0.5]),  # 1D instead of 2D
            token_ids=np.array([10, 20])
        )
        with pytest.raises(ValueError, match="must be 2-D arrays"):
            bad_dims.flatten()

    def test_empty_and_edge_cases(self):
        """Test edge cases like empty arrays and single elements."""
        # Test single token, single logprob
        single_logprobs = np.array([[-0.1]])
        single_token_ids = np.array([[42]])
        single_top = TopLogprobs(logprobs=single_logprobs, token_ids=single_token_ids)
        
        flat = single_top.flatten(threshold=0.5)
        reconstructed = flat.reconstruct()
        
        assert flat.shape == (1, 1)
        assert len(flat.logprobs) == 1
        assert reconstructed.logprobs.shape == (1, 1)
        np.testing.assert_array_equal(reconstructed.logprobs, single_logprobs)
        np.testing.assert_array_equal(reconstructed.token_ids, single_token_ids)

    def test_reconstruct_preserves_missing_entries(self):
        """Test that reconstruct properly handles missing entries with correct fill values."""
        # Create a case where we know some entries will be dropped
        logprobs = np.array([
            [-0.01, -3.0, -4.0, -5.0],  # Only first entry should be kept with high threshold
        ])
        token_ids = np.array([
            [100, 200, 300, 400]
        ])
        
        top_logprobs = TopLogprobs(logprobs=logprobs, token_ids=token_ids)
        flat = top_logprobs.flatten(threshold=0.5)  # Should only keep first entry
        reconstructed = flat.reconstruct()
        
        # First entry should be preserved
        assert reconstructed.logprobs[0, 0] == logprobs[0, 0]
        assert reconstructed.token_ids[0, 0] == token_ids[0, 0]
        
        # Remaining entries should be filled with defaults
        assert np.all(reconstructed.logprobs[0, 1:] == -np.inf)
        assert np.all(reconstructed.token_ids[0, 1:] == -1)

    def test_data_types_preserved(self):
        """Test that data types are preserved through flatten/reconstruct."""
        logprobs = np.array([[-0.1, -0.5]], dtype=np.float32)
        token_ids = np.array([[10, 20]], dtype=np.int32)
        
        top_logprobs = TopLogprobs(logprobs=logprobs, token_ids=token_ids)
        flat = top_logprobs.flatten()
        reconstructed = flat.reconstruct()
        
        assert flat.logprobs.dtype == np.float32
        assert flat.token_id.dtype == np.int32
        assert reconstructed.logprobs.dtype == np.float32
        assert reconstructed.token_ids.dtype == np.int32