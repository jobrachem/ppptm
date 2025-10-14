# import numpy as np
# from numpy.testing import assert_almost_equal

# from ppptm.initialize import spatially_smoothed_mean_and_var  # Adjust import as needed


# def test_spatially_smoothed_mean_and_var():
#     # Generate simple test data
#     np.random.seed(42)
#     y = np.array([[1, 2, 3], [4, 5, 6]])
#     loc = np.array([[0, 0], [1, 1], [2, 2]])
#     sub = np.array([[0.5, 0.5], [1.5, 1.5]])

#     # Test with a reasonable bandwidth
#     smoothed_means, smoothed_vars = spatially_smoothed_mean_and_var(
#         y, loc, sub, bandwidth=1.0
#     )

#     # Check output shape
#     assert smoothed_means.shape == (2,), "Mean output shape is incorrect"
#     assert smoothed_vars.shape == (2,), "Variance output shape is incorrect"

#     # Sanity check: results should be between min/max of input
#     assert np.all(smoothed_means >= np.min(y)), "Means should be within observed range"
#     assert np.all(smoothed_means <= np.max(y)), "Means should be within observed range"

#     # Test with very small bandwidth (almost no smoothing)
#     smoothed_means_small_bw, _ = spatially_smoothed_mean_and_var(
#         y, loc, loc, bandwidth=0.1
#     )
#     assert np.allclose(smoothed_means_small_bw, y.mean(axis=0), atol=0.1), (
#         "Small bandwidth should give nearly local values"
#     )

#     # Test with very large bandwidth (should approach global mean)
#     smoothed_means_large_bw, _ = spatially_smoothed_mean_and_var(
#         y, loc, sub, bandwidth=10
#     )
#     expected_global_mean = np.mean(y)
#     (
#         assert_almost_equal(smoothed_means_large_bw, expected_global_mean, decimal=2),
#         "Large bandwidth should approach global mean",
#     )
