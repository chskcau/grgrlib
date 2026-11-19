import numpy as np
import pytest

from grgrlib.stats import mvn_logpdf


def test_precisely_measured_direction():
    mean = np.zeros(3)
    cov = np.diag([1.0, 1.0, 1e-12])
    at_mean = mvn_logpdf(mean, mean, cov)
    shifted = mvn_logpdf(np.array([0.0, 0.0, 0.01]), mean, cov)

    expected = -0.5 * (3 * np.log(2 * np.pi) + np.log(1e-12))
    assert at_mean == pytest.approx(expected, abs=1e-12, rel=0)
    assert shifted - at_mean == pytest.approx(-50_000_000, abs=1e-7, rel=0)


def test_singular_covariance_support():
    mean = np.zeros(3)
    cov = np.diag([1.0, 1.0, 0.0])
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [1.0, -2.0, 0.0]])
    expected = [-np.log(2 * np.pi), -np.inf, -np.log(2 * np.pi) - 2.5]
    np.testing.assert_allclose(mvn_logpdf(points, mean, cov), expected, atol=1e-12)


def test_singular_covariance_with_rotated_support():
    mean = np.array([1.0, -1.0])
    cov = np.full((2, 2), 2.0)
    # The positive eigenvalue is 4. The first deviation (2, 2) has squared norm 8.
    expected = -0.5 * (np.log(2 * np.pi) + np.log(4.0) + 2.0)
    assert mvn_logpdf(mean + 2.0, mean, cov) == pytest.approx(expected)
    assert mvn_logpdf(np.array([3.0, -3.0]), mean, cov) == -np.inf


def test_zero_covariance():
    mean = np.array([2.0, 3.0])
    cov = np.zeros((2, 2))
    assert mvn_logpdf(mean, mean, cov) == 0.0
    assert mvn_logpdf(mean + 1.0, mean, cov) == -np.inf


def test_correlated_positive_definite_covariance():
    mean = np.array([1.0, -1.0])
    cov = np.array([[4.0, 2.0], [2.0, 2.0]])
    points = np.array([[1.0, -1.0], [3.0, -1.0], [3.0, 1.0]])
    # det(cov) = 4 and the squared Mahalanobis distances are 0, 2 and 2.
    expected = -np.log(2 * np.pi) - np.log(2.0) - np.array([0.0, 1.0, 1.0])
    actual = mvn_logpdf(points, mean, cov)
    assert actual.shape == (3,)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert np.isscalar(mvn_logpdf(points[0], mean, cov))


def test_indefinite_covariance():
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        mvn_logpdf(np.zeros(2), np.zeros(2), cov)


def test_proportional_rows_with_a_rounded_positive_cholesky_pivot():
    # cov = 3 * outer([18, 7], [18, 7]). Its only positive eigenvalue is 1119.
    cov = np.array([[972.0, 378.0], [378.0, 147.0]])
    mean = np.zeros(2)
    assert mvn_logpdf(mean, mean, cov) == pytest.approx(-0.5 * np.log(2 * np.pi * 1119))
    assert mvn_logpdf(np.array([-7.0, 18.0]), mean, cov) == -np.inf


def test_small_positive_pivot_from_correlation():
    delta = np.finfo(np.float64).eps
    cov = np.array([[1.0, 1.0], [1.0, 1.0 + delta]])
    mean = np.zeros(2)
    # det(cov) = delta. For (0, sqrt(delta)) the squared Mahalanobis distance is 1.
    point = np.array([0.0, np.sqrt(delta)])
    expected = -np.log(2 * np.pi) - 0.5 * (np.log(delta) + 1.0)
    assert mvn_logpdf(point, mean, cov) == pytest.approx(expected, abs=1e-12, rel=0)


def test_positive_definite_covariance_at_cholesky_roundoff():
    # The next float above 1/3 gives 3 * cov[1, 1] - 1 = 2**-53 exactly.
    cov = np.array([[3.0, 1.0], [1.0, np.nextafter(1.0 / 3.0, np.inf)]])
    mean = np.zeros(2)
    expected = -np.log(2 * np.pi) + 26.5 * np.log(2.0)
    assert mvn_logpdf(mean, mean, cov) == pytest.approx(expected, abs=1e-12, rel=0)
