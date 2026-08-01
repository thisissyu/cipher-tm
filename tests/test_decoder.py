import numpy as np
import pytest
import scipy.sparse as sp

from cipher_tm.decoder import (
    candidate_npmi,
    decode_topic_words,
    weighted_topic_term_scores,
)


def test_scores_use_soft_document_topic_posteriors() -> None:
    counts = sp.csr_matrix([[2.0, 0.0], [0.0, 1.0]])
    posteriors = np.array([[0.75, 0.25], [0.25, 0.75]])

    salience, exclusivity, weighted = weighted_topic_term_scores(counts, posteriors)

    expected_weighted = np.array([[1.5, 0.25], [0.5, 0.75]])
    topic_totals = expected_weighted.sum(axis=1, keepdims=True)
    word_given_topic = expected_weighted / topic_totals
    global_probability = np.array([2.0 / 3.0, 1.0 / 3.0])
    expected_salience = np.log1p(expected_weighted) * np.log1p(
        word_given_topic / global_probability
    )

    np.testing.assert_allclose(weighted, expected_weighted)
    np.testing.assert_allclose(salience, expected_salience)
    np.testing.assert_allclose(exclusivity, [[0.75, 0.25], [0.25, 0.75]])


def test_candidate_npmi_uses_binary_document_cooccurrence() -> None:
    counts = sp.csr_matrix(
        [
            [8.0, 1.0, 0.0],
            [2.0, 9.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 3.0],
        ]
    )

    scores = candidate_npmi(counts, [0, 1, 2])

    np.testing.assert_allclose(np.diag(scores), 1.0)
    assert scores[0, 1] == pytest.approx(1.0)
    assert scores[0, 2] == pytest.approx(-1.0)
    assert scores[1, 2] == pytest.approx(-1.0)


@pytest.mark.parametrize("candidate", [[1.5], ["1"], [True]])
def test_candidate_npmi_rejects_lossy_index_coercion(candidate: list[object]) -> None:
    with pytest.raises(ValueError, match="integer"):
        candidate_npmi(np.eye(2), candidate)  # type: ignore[arg-type]


def test_candidate_npmi_rejects_unobserved_terms() -> None:
    with pytest.raises(ValueError, match="unobserved"):
        candidate_npmi(np.array([[1.0, 0.0], [1.0, 0.0]]), [1])


def test_candidate_npmi_reports_huge_indices_as_out_of_range() -> None:
    with pytest.raises(ValueError, match="outside the vocabulary"):
        candidate_npmi(np.eye(2), [2**100])


def test_coherence_can_promote_a_cooccurring_candidate() -> None:
    counts = sp.csr_matrix(
        [
            [10.0, 1.0, 0.0],
            [10.0, 1.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
        ]
    )
    posteriors = np.ones((4, 1))
    vocabulary = ["anchor", "cooccurring", "frequent-but-disjoint"]

    salience_only = decode_topic_words(
        counts,
        posteriors,
        vocabulary,
        top_n=2,
        candidate_pool=3,
        beta=0.0,
        gamma=0.0,
        global_unique=False,
    )
    with_coherence = decode_topic_words(
        counts,
        posteriors,
        vocabulary,
        top_n=2,
        candidate_pool=3,
        beta=0.0,
        gamma=0.45,
        global_unique=False,
    )

    assert salience_only.words == (("anchor", "frequent-but-disjoint"),)
    assert with_coherence.words == (("anchor", "cooccurring"),)


def test_global_uniqueness_uses_deterministic_round_robin_allocation() -> None:
    counts = sp.csr_matrix([[6.0, 5.0, 4.0, 3.0], [6.0, 5.0, 4.0, 3.0]])
    posteriors = np.array([[1.0, 0.0], [0.0, 1.0]])

    decoded = decode_topic_words(
        counts,
        posteriors,
        ["a", "b", "c", "d"],
        top_n=2,
        candidate_pool=1,
        beta=0.0,
        gamma=0.0,
        global_unique=True,
    )

    # Rank-major turns are topic 0 -> topic 1 -> topic 0 -> topic 1.
    assert decoded.indices == ((0, 2), (1, 3))
    assert decoded.words == (("a", "c"), ("b", "d"))


def test_global_uniqueness_rejects_an_infeasible_vocabulary() -> None:
    with pytest.raises(ValueError, match="global topic-word uniqueness is impossible"):
        decode_topic_words(
            np.ones((2, 3)),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            ["a", "b", "c"],
            top_n=2,
            global_unique=True,
        )


def test_decoder_rejects_duplicate_vocabulary_terms() -> None:
    with pytest.raises(ValueError, match="terms must be unique"):
        decode_topic_words(
            np.eye(2),
            np.eye(2),
            ["same", "same"],
            top_n=1,
            global_unique=True,
        )


def test_decoder_uses_only_observed_vocabulary_terms() -> None:
    counts = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    decoded = decode_topic_words(
        counts,
        np.eye(2),
        ["left", "right", "absent"],
        top_n=1,
        global_unique=True,
    )

    assert {index for topic in decoded.indices for index in topic} == {0, 1}
    expanded = decode_topic_words(
        counts,
        np.eye(2),
        ["left", "right", "absent"],
        top_n=2,
        global_unique=False,
    )
    assert all(set(topic) == {0, 1} for topic in expanded.indices)

    with pytest.raises(ValueError, match="observed_vocabulary_size"):
        decode_topic_words(
            counts,
            np.eye(2),
            ["left", "right", "absent"],
            top_n=3,
            global_unique=False,
        )


def test_decoder_requires_simplex_posteriors() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        weighted_topic_term_scores(np.eye(2), np.array([[20.0, 0.0], [0.0, 1.0]]))


def test_decoder_rejects_topics_without_posterior_mass() -> None:
    with pytest.raises(ValueError, match="every topic must have positive posterior mass"):
        decode_topic_words(
            np.ones((2, 3)),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            ["a", "b", "c"],
            top_n=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_n": 1.5},
        {"candidate_pool": 1.5},
        {"top_n": True},
        {"global_unique": 1},
        {"beta": True},
        {"gamma": False},
        {"exclusivity_scale": True},
    ],
)
def test_decoder_rejects_non_integral_or_non_boolean_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        decode_topic_words(
            np.eye(2),
            np.eye(2),
            ["left", "right"],
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("counts", "posteriors", "message"),
    [
        (np.ones((2, 3)), np.ones((3, 2)), "rows must match"),
        (np.ones((2, 3)), np.array([[1.0, -0.1], [0.0, 1.0]]), "non-negative"),
        (np.ones((2, 3)), np.array([[0.0, 0.0], [0.0, 1.0]]), "sum to one"),
    ],
)
def test_decoder_validates_posterior_shape_and_values(
    counts: np.ndarray,
    posteriors: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_topic_words(counts, posteriors, ["a", "b", "c"], top_n=1)
