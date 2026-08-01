from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from cipher_tm.refinement import (
    cosine_knn_transition,
    posterior_weighted_prototypes,
    propagate_posteriors,
    prototype_feedback,
    row_normalize,
)


def test_row_normalize_returns_simplex_and_uses_uniform_zero_row() -> None:
    values = np.array([[2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    normalized = row_normalize(values)

    np.testing.assert_allclose(normalized[0], [0.5, 0.25, 0.25])
    np.testing.assert_allclose(normalized[1], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(normalized.sum(axis=1), 1.0)
    np.testing.assert_array_equal(values, [[2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0, -0.1]]),
        np.array([[1.0, np.nan]]),
        np.empty((1, 0)),
    ],
)
def test_row_normalize_rejects_non_probability_matrices(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        row_normalize(values)


def test_cosine_knn_transition_is_symmetrized_and_row_stochastic() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [-1.0, 0.0],
        ]
    )

    transition = cosine_knn_transition(embeddings, neighbors=1)

    assert sp.isspmatrix_csr(transition)
    np.testing.assert_allclose(
        transition.toarray(),
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )


def test_cosine_knn_transition_excludes_self_for_duplicate_embeddings() -> None:
    transition = cosine_knn_transition(np.array([[1.0, 0.0], [1.0, 0.0]]), neighbors=1)

    np.testing.assert_allclose(transition.toarray(), [[0.0, 1.0], [1.0, 0.0]])


def test_propagate_posteriors_applies_restart_on_every_iteration() -> None:
    anchors = np.array([[1.0, 0.0], [0.0, 1.0]])
    transition = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]])

    once = propagate_posteriors(anchors, transition, alpha=0.5, iterations=1)
    twice = propagate_posteriors(anchors, transition, alpha=0.5, iterations=2)

    np.testing.assert_allclose(once, [[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(twice, [[0.75, 0.25], [0.25, 0.75]])


def test_propagate_posteriors_preserves_anchors_without_graph_updates() -> None:
    anchors = np.array([[4.0, 1.0], [0.0, 3.0]])

    result = propagate_posteriors(
        anchors,
        sp.csr_matrix((2, 2)),
        alpha=0.82,
        iterations=3,
    )

    np.testing.assert_allclose(result, [[0.8, 0.2], [0.0, 1.0]])


def test_propagate_posteriors_keeps_isolated_anchors_at_alpha_one() -> None:
    anchors = np.array([[0.8, 0.2], [0.1, 0.9]])

    result = propagate_posteriors(
        anchors,
        sp.csr_matrix((2, 2)),
        alpha=1.0,
        iterations=2,
    )

    np.testing.assert_allclose(result, anchors)


def test_posterior_weighted_prototypes_match_soft_topic_centers() -> None:
    embeddings = np.array([[3.0, 0.0], [0.0, 4.0], [-2.0, 0.0]])
    posteriors = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    centers = posterior_weighted_prototypes(embeddings, posteriors)

    root_five = np.sqrt(5.0)
    np.testing.assert_allclose(
        centers,
        [[2.0 / root_five, 1.0 / root_five], [-2.0 / root_five, 1.0 / root_five]],
    )
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.0)


def test_prototype_feedback_is_invariant_to_embedding_row_scale() -> None:
    posteriors = np.array([[0.9, 0.1], [0.1, 0.9]])

    unit = prototype_feedback(
        np.eye(2),
        posteriors,
        temperature=0.1,
        rho=0.5,
        rounds=1,
    )
    scaled = prototype_feedback(
        np.array([[5.0, 0.0], [0.0, 0.25]]),
        posteriors,
        temperature=0.1,
        rho=0.5,
        rounds=1,
    )

    np.testing.assert_allclose(scaled, unit)


def test_prototype_operations_require_simplex_posteriors() -> None:
    invalid = np.array([[2.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="sum to one"):
        posterior_weighted_prototypes(np.eye(2), invalid)
    with pytest.raises(ValueError, match="sum to one"):
        prototype_feedback(
            np.eye(2),
            invalid,
            temperature=0.1,
            rho=0.5,
            rounds=1,
        )


def test_posterior_weighted_prototypes_leave_empty_topic_at_zero() -> None:
    centers = posterior_weighted_prototypes(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
    )

    np.testing.assert_allclose(centers, [[1 / np.sqrt(2), 1 / np.sqrt(2)], [0.0, 0.0]])


def test_prototype_feedback_uses_temperature_and_posterior_interpolation() -> None:
    embeddings = np.eye(2)
    posteriors = np.array([[0.9, 0.1], [0.1, 0.9]])

    refined = prototype_feedback(
        embeddings,
        posteriors,
        temperature=0.1,
        rho=0.5,
        rounds=1,
    )

    assert refined[0, 0] > posteriors[0, 0]
    assert refined[1, 1] > posteriors[1, 1]
    np.testing.assert_allclose(refined[0], refined[1, ::-1])
    np.testing.assert_allclose(refined.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0, "rho": 0.5, "rounds": 1}, "temperature"),
        ({"temperature": 0.1, "rho": -0.1, "rounds": 1}, "rho"),
        ({"temperature": 0.1, "rho": 1.1, "rounds": 1}, "rho"),
        ({"temperature": 0.1, "rho": 0.5, "rounds": -1}, "rounds"),
    ],
)
def test_prototype_feedback_rejects_invalid_parameters(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prototype_feedback(np.eye(2), np.eye(2), **kwargs)


def test_refinement_functions_reject_shape_mismatches() -> None:
    with pytest.raises(ValueError, match="transition"):
        propagate_posteriors(np.eye(2), np.eye(3), alpha=0.5, iterations=1)
    with pytest.raises(ValueError, match="same number of rows"):
        posterior_weighted_prototypes(np.eye(2), np.ones((3, 2)))
    with pytest.raises(ValueError, match="same number of rows"):
        prototype_feedback(
            np.eye(2),
            np.ones((3, 2)),
            temperature=0.1,
            rho=0.5,
            rounds=1,
        )
