"""Semantic--geometric posterior refinement used by CIPHER-TM.

The functions in this module implement only the numerical mechanisms described in
the paper: cosine-neighborhood propagation and posterior-to-prototype feedback.  They
are deliberately independent of data loading, model providers, experiment runners,
and artifact I/O.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def _matrix(name: str, values: np.ndarray) -> np.ndarray:
    """Convert an array-like value to a finite, two-dimensional float matrix."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _unit_rows(values: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized vectors, leaving zero rows at zero."""
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0.0)


def _posterior_matrix(name: str, values: np.ndarray) -> np.ndarray:
    """Validate a document-topic matrix on the probability simplex."""
    matrix = _matrix(name, values)
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one topic")
    if np.any(matrix < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(f"each {name} row must sum to one")
    return matrix


def _integer(name: str, value: int, *, minimum: int) -> int:
    """Validate a public integer parameter without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def row_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize nonnegative matrix rows to probability distributions.

    An all-zero row carries no topic evidence, so it is mapped to a uniform
    distribution instead of being left outside the probability simplex.  Inputs are
    converted to ``float64`` and never modified in place.

    Args:
        values: A two-dimensional nonnegative matrix with at least one column.

    Returns:
        A matrix of the same shape whose rows sum to one.

    Raises:
        ValueError: If the input is not finite, two-dimensional, nonnegative, or has
            no columns.
    """
    matrix = _matrix("values", values)
    if matrix.shape[1] == 0:
        raise ValueError("values must have at least one column")
    if np.any(matrix < 0.0):
        raise ValueError("values must be nonnegative")

    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.empty_like(matrix)
    nonzero = totals[:, 0] > 0.0
    normalized[nonzero] = matrix[nonzero] / totals[nonzero]
    normalized[~nonzero] = 1.0 / matrix.shape[1]
    return normalized


def cosine_knn_transition(embeddings: np.ndarray, *, neighbors: int) -> sp.csr_matrix:
    """Construct the paper's symmetrized cosine k-NN transition matrix.

    Each document is joined to its nearest non-self neighbors with nonnegative cosine
    similarity as the edge weight.  The directed graph is symmetrized by retaining the
    larger weight in either direction and then row-normalized.  Documents without a
    positive-similarity edge remain isolated (their transition row is zero).

    Args:
        embeddings: ``(documents, dimensions)`` document representation matrix.
        neighbors: Maximum number of outgoing nearest neighbors before symmetrization.

    Returns:
        A sparse ``(documents, documents)`` row-stochastic transition matrix.  Rows for
        isolated documents sum to zero; all other rows sum to one.

    Raises:
        ValueError: If embeddings are empty or invalid, or ``neighbors`` is not positive.
    """
    values = _matrix("embeddings", embeddings)
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must contain at least one document and one dimension")
    neighbor_count = _integer("neighbors", neighbors, minimum=1)
    document_count = values.shape[0]
    if document_count == 1:
        return sp.csr_matrix((1, 1), dtype=np.float64)

    query_count = min(neighbor_count + 1, document_count)
    model = NearestNeighbors(n_neighbors=query_count, metric="cosine", n_jobs=1).fit(values)
    distances, indices = model.kneighbors(values)

    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for row, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        accepted = 0
        for distance, column in zip(row_distances, row_indices, strict=True):
            column = int(column)
            if column == row:
                continue
            if accepted >= neighbor_count:
                break
            accepted += 1
            # Numerical cosine distances can stray just outside their theoretical range.
            weight = min(1.0, max(1.0 - float(distance), 0.0))
            if weight > 0.0:
                rows.append(row)
                columns.append(column)
                weights.append(weight)

    graph = sp.csr_matrix(
        (weights, (rows, columns)),
        shape=(document_count, document_count),
        dtype=np.float64,
    )
    graph = graph.maximum(graph.T).tocsr()
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    inverse_degrees = np.divide(
        1.0,
        degrees,
        out=np.zeros_like(degrees),
        where=degrees > 0.0,
    )
    return (sp.diags(inverse_degrees, format="csr") @ graph).tocsr()


def _normalized_transition(
    transition: sp.spmatrix | np.ndarray, document_count: int
) -> sp.csr_matrix:
    """Validate and defensively row-normalize a transition/affinity matrix."""
    if sp.issparse(transition):
        matrix = sp.csr_matrix(transition, dtype=np.float64)
    else:
        dense = _matrix("transition", transition)
        matrix = sp.csr_matrix(dense)
    if matrix.shape != (document_count, document_count):
        raise ValueError("transition must be square with one row and column per posterior row")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("transition must contain only finite values")
    if np.any(matrix.data < 0.0):
        raise ValueError("transition must be nonnegative")

    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0,
        row_sums,
        out=np.zeros_like(row_sums),
        where=row_sums > 0.0,
    )
    return (sp.diags(inverse, format="csr") @ matrix).tocsr()


def propagate_posteriors(
    anchors: np.ndarray,
    transition: sp.spmatrix | np.ndarray,
    *,
    alpha: float,
    iterations: int,
) -> np.ndarray:
    """Propagate anchored topic posteriors with a random walk with restart.

    This implements ``Q <- rownorm(alpha * P @ Q + (1-alpha) * anchors)``.
    Reintroducing the anchors on every iteration prevents local semantic corrections
    from being washed out by graph smoothing.  Passing zero iterations returns the
    normalized anchors.

    Args:
        anchors: ``(documents, topics)`` nonnegative anchored topic weights.
        transition: A nonnegative square transition or affinity matrix.  It is
            defensively row-normalized; isolated rows are permitted.
        alpha: Graph-propagation weight in the closed interval ``[0, 1]``.
        iterations: Number of propagation updates.

    Returns:
        Refined document-topic distributions on the probability simplex.

    Raises:
        ValueError: If shapes, values, or parameters are invalid.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and between 0 and 1")
    step_count = _integer("iterations", iterations, minimum=0)
    base = row_normalize(anchors)
    matrix = _normalized_transition(transition, base.shape[0])

    current = base.copy()
    for _ in range(step_count):
        update = alpha * (matrix @ current) + (1.0 - alpha) * base
        # An isolated graph row with alpha=1 has no outgoing probability mass.
        # Retaining its anchor is the continuous alpha->1 limit and avoids an
        # arbitrary uniform-topic assignment.
        isolated = update.sum(axis=1) == 0.0
        update[isolated] = base[isolated]
        current = row_normalize(update)
    return current


def posterior_weighted_prototypes(embeddings: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
    """Compute L2-normalized topic prototypes from soft document assignments.

    For topic ``k``, the unnormalized center is ``sum_i q[i, k] * z[i]``.  A topic
    receiving zero total vector mass is represented by the zero vector rather than an
    arbitrary direction.

    Args:
        embeddings: ``(documents, dimensions)`` document representation matrix. Rows are
            L2-normalized internally to recover the paper's spherical document vectors.
        posteriors: ``(documents, topics)`` topic probabilities on the simplex.

    Returns:
        ``(topics, dimensions)`` L2-normalized prototype matrix.

    Raises:
        ValueError: If the matrices are empty, invalid, or disagree on document count.
    """
    values = _unit_rows(_matrix("embeddings", embeddings))
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must contain at least one document and one dimension")
    raw_posteriors = _matrix("posteriors", posteriors)
    if raw_posteriors.shape[0] != values.shape[0]:
        raise ValueError("embeddings and posteriors must have the same number of rows")
    probabilities = _posterior_matrix("posteriors", raw_posteriors)

    centers = probabilities.T @ values
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    return np.divide(centers, norms, out=np.zeros_like(centers), where=norms > 0.0)


def prototype_feedback(
    embeddings: np.ndarray,
    posteriors: np.ndarray,
    *,
    temperature: float,
    rho: float,
    rounds: int,
) -> np.ndarray:
    """Feed posterior-weighted topic prototypes back into document mixtures.

    Each round rebuilds prototypes from the current posterior, obtains a temperature-
    scaled soft assignment from embedding--prototype similarities, and interpolates it
    with the current posterior using ``rho``.  This is the paper's semantic-to-geometric
    feedback operation, without any surrounding pipeline orchestration.

    Args:
        embeddings: ``(documents, dimensions)`` document representation matrix. Rows are
            L2-normalized internally to recover the paper's spherical document vectors.
        posteriors: ``(documents, topics)`` starting topic probabilities on the simplex.
        temperature: Positive softmax temperature.
        rho: Weight retained from the current posterior, in ``[0, 1]``.
        rounds: Number of prototype-feedback updates.

    Returns:
        Refined document-topic distributions.

    Raises:
        ValueError: If shapes, values, or parameters are invalid.
    """
    values = _unit_rows(_matrix("embeddings", embeddings))
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must contain at least one document and one dimension")
    raw_posteriors = _matrix("posteriors", posteriors)
    if raw_posteriors.shape[0] != values.shape[0]:
        raise ValueError("embeddings and posteriors must have the same number of rows")
    current = _posterior_matrix("posteriors", raw_posteriors).copy()

    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    rho = float(rho)
    if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be finite and between 0 and 1")
    round_count = _integer("rounds", rounds, minimum=0)

    for _ in range(round_count):
        centers = posterior_weighted_prototypes(values, current)
        logits = (values @ centers.T) / temperature
        if not np.all(np.isfinite(logits)):
            raise ValueError("embedding similarities must remain finite after scaling")
        logits -= logits.max(axis=1, keepdims=True)
        assignments = row_normalize(np.exp(logits))
        current = row_normalize(rho * current + (1.0 - rho) * assignments)
    return current
