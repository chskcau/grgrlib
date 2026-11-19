# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, types, vectorize
from numba.extending import intrinsic
from math import erfc, frexp, ldexp, pi


SQRT2 = np.sqrt(2.0)
SQRT2pi = np.sqrt(2.0 * pi)


@njit(nogil=True, cache=True)
def subt(A, B):
    res = A
    for i in range(len(A)):
        for j in range(len(A)):
            res[i][j] = A[i][j] - B[i][j]
    return res


@njit(nogil=True, cache=True)
def cholesky(A):
    """Performs a Cholesky decomposition of on symmetric, pos-def A. Returns lower-triangular L (full sized, zeroed above diag)"""
    n = A.shape[0]
    L = np.zeros_like(A)

    # Perform the Cholesky decomposition
    for row in range(n):
        for col in range(row + 1):
            tmp_sum = np.dot(L[row, :col], L[col, :col])
            if row == col:  # Diagonal elements
                L[row, col] = np.sqrt(max(A[row, row] - tmp_sum, 0))
            elif np.abs(L[col, col]) < 1e-5:
                L[row, col] = 0
            else:
                L[row, col] = (1.0 / L[col, col]) * (A[row, col] - tmp_sum)
    return L


@njit(nogil=True, cache=True)
def numba_rand_norm(loc=0, scale=1, size=1):
    """A numba interface to create the user experience of np.random.normal with the size argument."""

    out = np.empty(size)

    for i in np.ndindex(size):
        out[i] = np.random.normal(loc, scale)

    return out


@vectorize(nopython=True, cache=True)
def normal_cdf(x, mu, sig):
    return erfc((mu - x) / sig / SQRT2) / 2.0


@njit(nogil=True, cache=True)
def normal_pdf(x, mu, sig):
    return 1 / (sig * SQRT2pi) * np.exp(-0.5 * ((x - mu) / sig) ** 2)


@njit(nogil=True, cache=True)
def histogram_weights(a, bins, weights):

    sorting_index = np.argsort(a)
    sa = a[sorting_index]
    sw = weights[sorting_index]
    cw = np.hstack((np.zeros(1), sw.cumsum()))

    bin_index = np.hstack(
        (
            np.searchsorted(sa, bins[:-1], "left"),
            np.searchsorted(sa, bins[-1:], "right"),
        )
    )

    cum_n = cw[bin_index]

    n = np.diff(cum_n)

    return n, bins


@njit(cache=True, nogil=True)
def shredder_basic(M, tol, verbose):

    m, n = M.shape
    Q = np.eye(m)

    j = 0

    for i in range(min(m, n)):

        while j < n:

            a = np.ascontiguousarray(Q.T[i:]) @ np.ascontiguousarray(M[:, j])
            do = False
            for ia in a:
                if abs(ia) > tol:
                    do = True
                    break

            if do:
                # apply Householder transformation
                v, tau = householder_reflection(a)

                H = np.identity(m - i)
                H -= tau * np.outer(v, v)

                Q[:, i:] = np.ascontiguousarray(Q[:, i:]) @ H

                if verbose:
                    print("...shredding row", j, "and col", i)
                break

            else:
                j += 1

    return Q


@njit(cache=True, nogil=True)
def shredder_pivoting(M, tol, verbose):
    """The DS-decomposition with pivoting"""

    m, n = M.shape
    M = M.copy().astype(np.float64)
    Q = np.eye(m)
    P = np.arange(n)

    j = 0

    for i in range(min(m, n)):

        # pivoting loop
        k = i
        while True:
            if k == n:
                break
            if np.any(np.abs(M[i:, k]) > tol):
                if k != i:
                    # move columns
                    M[:, np.array([i, k])] = M[:, np.array([k, i])]
                    P[np.array([i, k])] = P[np.array([k, i])]
                break
            k += 1

        while j < n:
            if np.any(np.abs(M[i:, j]) > tol):

                # apply Householder transformation
                a = M[i:, j]
                v, tau = householder_reflection(a)

                H = np.identity(m - i)
                H -= tau * np.outer(v, v)
                M[i:] = H @ M[i:]
                Q[:, i:] = np.ascontiguousarray(Q[:, i:]) @ H

                if verbose:
                    print("...shredding row", j, "and col", i)
                break

            else:
                j += 1

    return Q, M, P


@njit(cache=True, nogil=True)
def shredder_non_pivoting(M, tol, verbose):
    """The DS-decomposition with anti-pivoting"""

    m, n = M.shape
    M = M.copy().astype(np.float64)
    Q = np.eye(m)
    P = np.arange(n)

    j = 0

    for i in range(min(m, n)):

        # anti-pivoting
        # for each i move those columns forward that do not need transformation

        for k in range(j, n):
            if np.all(np.abs(M[i:, k]) < tol):
                # move columns
                if k != j:
                    M[:, np.array([j, k])] = M[:, np.array([k, j])]
                    P[np.array([j, k])] = P[np.array([k, j])]

                if verbose:
                    print("...switching rows", j, "and", k)
                j += 1

        while j < n:
            if np.any(np.abs(M[i:, j]) > tol):

                # apply Householder transformation
                a = M[i:, j]
                v, tau = householder_reflection(a)

                H = np.identity(m - i)
                H -= tau * np.outer(v, v)
                M[i:] = H @ M[i:]
                Q[:, i:] = np.ascontiguousarray(Q[:, i:]) @ H

                if verbose:
                    print("...shredding row", j, "and col", i)
                break

            else:
                j += 1

    return Q, M, P


def shredder(M, pivoting=None, tol=1e-11, verbose=False):
    """The QS decomposition from "Efficient Solution of Models with Occasionally Binding Constraints" (Gregor Boehl)

    The QS decomposition uses Householder reflections to bring a system in the row-echolon form. This is a dispatcher for the sub-functions.
    """

    if pivoting is None:
        q = shredder_basic(M, tol, verbose)
        return q, q.T @ M
    elif pivoting:
        return shredder_pivoting(M, tol, verbose)
    elif not pivoting:
        return shredder_non_pivoting(M, tol, verbose)
    else:
        raise NotImplementedError


@njit(cache=True, nogil=True, fastmath=True)
def householder_reflection(x):
    alpha = x[0]
    s = np.linalg.norm(x[1:]) ** 2
    v = x.copy()

    if s == 0:
        tau = 0.0
    else:
        t = np.sqrt(alpha ** 2 + s)
        v[0] = alpha - t if alpha <= 0 else -s / (alpha + t)

        tau = 2 * v[0] ** 2 / (s + v[0] ** 2)
        v /= v[0]

    return v, tau


@njit(cache=True, nogil=True, fastmath=True)
def householder_reflection_right(x):
    alpha = x[-1]
    s = np.linalg.norm(x[:-1]) ** 2
    v = x.copy()

    if s == 0:
        tau = 0.0
    else:
        t = np.sqrt(alpha ** 2 + s)
        v[-1] = alpha - t if alpha <= 0 else -s / (alpha + t)

        tau = 2 * v[-1] ** 2 / (s + v[-1] ** 2)
        v /= v[-1]

    return v, tau


_cond = 1e-9


@njit(cache=True)
def psd_func(M):
    """Compute the symmetric eigendecomposition

    Note that eigh takes care of array conversion, chkfinite, and assertion that the matrix is square.
    """

    s, u = np.linalg.eigh(M)
    eps = _cond * np.max(np.abs(s))
    d = s[s > eps]

    s_pinv = [0 if abs(x) <= eps else 1 / x for x in s]
    s_pinv_np = np.array(s_pinv)
    U = np.multiply(u, np.sqrt(s_pinv_np))

    return U, np.sum(np.log(d)), len(d)


@intrinsic
def _fma_float64(typingctx, a, b, c):
    """Fused multiply-add with one rounding, including on Python before 3.13."""
    def codegen(context, builder, signature, args):
        return builder.fma(*args)

    return types.float64(types.float64, types.float64, types.float64), codegen


@njit(cache=True)
def _mvn_cholesky(cov):
    try:
        lower = np.linalg.cholesky(cov)
    except Exception:
        # Numba only supports catching Exception here.
        lower = np.empty_like(cov)
    else:
        # A rounded square root can leave a spurious positive pivot for exactly
        # dependent rows. Check each pivot against its own marginal scale.
        tol = len(cov) * np.finfo(np.float64).eps * np.diag(cov)
        if np.all(np.diag(lower) ** 2 > tol):
            return lower

    # Recompute the Schur complements without square roots. Subtract the two
    # cross-products before dividing so proportional rows cancel directly.
    # Scaling the pivot by a power of two avoids squaring the covariance scale.
    schur = cov.copy()
    for j in range(len(cov)):
        pivot = schur[j, j]
        if pivot <= 0:
            return np.empty((0, 0))
        mantissa, exponent = frexp(pivot)
        lower[j, j] = np.sqrt(pivot)
        for i in range(j + 1, len(cov)):
            scaled = ldexp(schur[i, j], -exponent)
            lower[i, j] = schur[i, j] / lower[j, j]
            for k in range(j + 1, i + 1):
                product = scaled * schur[k, j]
                error = _fma_float64(-scaled, schur[k, j], product)
                diff = _fma_float64(mantissa, schur[i, k], -product)
                schur[i, k] = (diff + error) / mantissa
    return lower


@njit(cache=True)
def _mvn_logpdf_rows(dev, cov):
    dim = cov.shape[0]
    if dim == 0 or cov.shape[1] != dim or dev.shape[1] != dim:
        raise ValueError("cov must be square and match the observation dimension")
    if not np.all(np.isfinite(cov)):
        raise ValueError("cov must contain only finite values")

    log_2pi = np.log(2 * np.pi)
    roundoff = dim * np.finfo(np.float64).eps
    lower = _mvn_cholesky(cov)
    if lower.size == 0:
        values, vectors = np.linalg.eigh(cov)
        # Bound eigenvalue roundoff on each direction's own scale.
        abs_vectors = np.abs(vectors)
        tol = roundoff * np.sum(abs_vectors * (np.abs(cov) @ abs_vectors), axis=0)
        if np.any(values < -tol):
            raise ValueError("cov must be positive semidefinite")
        positive = values > tol
        coordinates = dev @ vectors
        scaled = coordinates[:, positive] / np.sqrt(values[positive])
        log_norm = np.sum(positive) * log_2pi + np.sum(np.log(values[positive]))
        out = -0.5 * (log_norm + np.sum(scaled ** 2, axis=1))

        # Allow for cancellation in the projection onto the null space.
        null_tol = roundoff * (np.abs(dev) @ abs_vectors[:, ~positive])
        for i in range(len(out)):
            if np.any(np.abs(coordinates[i, ~positive]) > null_tol[i]):
                out[i] = -np.inf
        return out

    log_norm = dim * log_2pi + 2 * np.sum(np.log(np.diag(lower)))
    out = np.empty(dev.shape[0])
    whitened = np.empty(dim)
    for i in range(len(out)):
        for j in range(dim):
            residual = dev[i, j]
            for k in range(j):
                residual -= lower[j, k] * whitened[k]
            whitened[j] = residual / lower[j, j]
        out[i] = -0.5 * (log_norm + np.dot(whitened, whitened))
    return out


@njit(cache=True)
def mvn_logpdf(x, mean, cov):
    """Gaussian log-density for one observation or a batch of observation rows.

    For singular covariances the density is relative to the affine support
    through mean. Observations outside that support have log-density -inf.
    """
    dev = np.asarray(x - mean, dtype=np.float64)
    rows = np.atleast_2d(dev)
    out = _mvn_logpdf_rows(rows, np.asarray(cov, dtype=np.float64))
    # Empty-tuple indexing unboxes a scalar and preserves a batch of length one.
    return out.reshape(dev.shape[:-1])[()]
